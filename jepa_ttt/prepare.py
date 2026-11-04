"""Download artifacts, validate the dataset and fit the reward head.

Usage:
    python -m jepa_ttt.prepare --workdir /path/to/workdir
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tarfile
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the dataset's HDF5 compression filters)
import numpy as np
import torch
import torch.nn.functional as F
import zstandard
from huggingface_hub import hf_hub_download

from .adaptation import ROWS_PER_WINDOW, prediction_loss
from .envs import ShiftedTwoRoom, task_scores
from .lewm import load_pretrained
from .planning import ACTION_BLOCK, HISTORY, make_reward_head

MODEL_REPO = "quentinll/lewm-tworooms"
MODEL_REVISION = "77adaae0bc31deab21c93740d1f8bb947cd0bdec"
DATA_REPO = "quentinll/lewm-tworooms"
DATA_REVISION = "6903a2de048b13819d812da0b4dd661290bc01e4"
DATA_ARCHIVE = "tworoom.tar.zst"

# Appendix A1.1: AdamW, batch 512, patience 15, at most 100 epochs. The paper gives
# no episode count for the head; 3,000 matches its world-model training data.
REWARD_EPISODES = 3000
REWARD_BATCH = 512
REWARD_LR = 1e-3
REWARD_WEIGHT_DECAY = 1e-5
REWARD_PATIENCE = 15
REWARD_MAX_EPOCHS = 100
REWARD_VAL_FRACTION = 0.1

MAX_RENDER_MAE = 5.0  # mean absolute difference per channel, on the 0-255 scale
MIN_SPEED, MAX_SPEED = 4.5, 5.5  # the environment's default speed is 5.0
MAX_ALIGNED_MSE_RATIO = 0.5  # measured on the real data before any experiment: 0.04


def require_device(name: str, allow_cpu: bool) -> torch.device:
    """Refuse to run on the CPU unless explicitly allowed."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU is visible. Run this on a GPU machine; "
            "--allow-cpu exists only for debugging tiny settings."
        )
    if device.type == "cpu" and not allow_cpu:
        raise SystemExit("Refusing to run on the CPU without --allow-cpu.")
    return device


def extract_dataset(archive: Path, target_dir: Path) -> Path:
    """Stream-decompress the archive and return the extracted ``.h5`` path."""
    existing = sorted(target_dir.rglob("*.h5"))
    if existing:
        return existing[0]
    # Extract aside and rename, so an interrupted extraction never looks complete.
    partial = target_dir.with_name(target_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    with archive.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                tar.extractall(partial, filter="data")
    if not sorted(partial.rglob("*.h5")):
        raise FileNotFoundError(f"no .h5 file inside {archive}")
    shutil.rmtree(target_dir, ignore_errors=True)
    os.replace(partial, target_dir)
    return sorted(target_dir.rglob("*.h5"))[0]


def compute_action_stats(h5: h5py.File) -> dict:
    """Mirror upstream's normalizer: drop NaN rows, then per-dimension mean/std."""
    actions = h5["action"][:]
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"expected actions (N, 2), got {actions.shape}")
    actions = actions[~np.isnan(actions).any(axis=1)]
    return {
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0, ddof=1).astype(float).tolist(),
    }


def validate_render_match(h5: h5py.File, samples: int = 32) -> float:
    """Check dataset frames against this environment's renderer at the same state.

    A large difference means the installed environment no longer matches the
    data the checkpoint was trained on, so the run must not continue.
    """
    total = h5["pixels"].shape[0]
    rows = np.linspace(0, total - 1, samples).astype(int)
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=0, start_mode="fixed")
    errors = []
    for row in rows:
        env.agent_position = torch.as_tensor(h5["proprio"][row][:2], dtype=torch.float32)
        rendered = env.render().astype(np.float32)
        stored = h5["pixels"][row].astype(np.float32)
        if stored.shape != rendered.shape:
            raise ValueError(f"frame shape {stored.shape} != render {rendered.shape}")
        errors.append(float(np.abs(stored - rendered).mean()))
    return float(np.mean(errors))


def validate_dynamics(h5: h5py.File, max_rows: int = 200_000) -> dict:
    """Check that dataset rows follow p[t + 1] = p[t] + speed * a[t].

    The world model pairs each frame with the block that starts at it, so the
    action in row t must be the one executed from the position in row t. A
    median is used because wall and border collisions break the relation.
    """
    lengths, offsets = h5["ep_len"][:], h5["ep_offset"][:]
    rows = min(int(h5["action"].shape[0]), max_rows)
    positions = h5["proprio"][:rows, :2].astype(np.float64)
    actions = h5["action"][:rows].astype(np.float64)

    same_episode = np.zeros(rows, dtype=bool)
    for offset, length in zip(offsets, lengths):
        # Rows offset + 1 .. offset + length - 2 have a same-episode row on both sides.
        first, last = int(offset) + 1, min(int(offset + length) - 1, rows - 1)
        if first < last:
            same_episode[first:last] = True
    index = np.flatnonzero(same_episode)
    index = index[
        ~np.isnan(actions[index]).any(axis=1)
        & ~np.isnan(actions[index - 1]).any(axis=1)
        & ~np.isnan(actions[index + 1]).any(axis=1)
    ]
    if index.size == 0:
        raise ValueError("no usable rows to validate dynamics")

    moved = positions[index + 1] - positions[index]
    magnitude = np.linalg.norm(actions[index], axis=1)
    usable = magnitude > 0.1
    speed = float(np.median(np.linalg.norm(moved[usable], axis=1) / magnitude[usable]))
    aligned = float(np.median(np.linalg.norm(moved - speed * actions[index], axis=1)))
    previous = float(np.median(np.linalg.norm(moved - speed * actions[index - 1], axis=1)))
    following = float(np.median(np.linalg.norm(moved - speed * actions[index + 1], axis=1)))
    return {
        "speed": speed,
        "aligned_error": aligned,
        "previous_error": previous,
        "next_error": following,
    }


@torch.no_grad()
def validate_checkpoint_alignment(
    model, h5: h5py.File, stats: dict, windows: int = 128, seed: int = 0
) -> dict:
    """Measure the checkpoint's prediction error on real data with this pipeline.

    Dense windows are built the same way as at test time. If preprocessing or
    action normalization were wrong, real actions would predict no better than
    actions from another window. A one-row lag is caught by validate_dynamics.
    """
    device = next(model.parameters()).device
    lengths, offsets = h5["ep_len"][:], h5["ep_offset"][:]
    rng = np.random.default_rng(seed)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    eligible = np.flatnonzero(lengths >= ROWS_PER_WINDOW)
    if eligible.size == 0:
        raise ValueError(f"no episode has {ROWS_PER_WINDOW} rows")
    latents, blocks = [], []
    attempts = 0
    while len(latents) < windows:
        attempts += 1
        if attempts > 100 * windows:
            raise ValueError("too few windows without NaN actions")
        episode = rng.choice(eligible)
        start = int(offsets[episode]) + int(rng.integers(0, lengths[episode] - ROWS_PER_WINDOW + 1))
        actions = h5["action"][start : start + HISTORY * ACTION_BLOCK]
        if np.isnan(actions).any():
            continue
        keyframes = [start + j * ACTION_BLOCK for j in range(HISTORY + 1)]
        latents.append(model.encode_frames(np.stack([h5["pixels"][k] for k in keyframes])))
        normalized = ((actions - mean) / std).astype(np.float32)
        blocks.append(torch.as_tensor(normalized.reshape(HISTORY, -1), device=device))
    z, b = torch.stack(latents), torch.stack(blocks)
    return {
        "aligned_mse": prediction_loss(model, z, b).item(),
        "shuffled_mse": prediction_loss(model, z, b.roll(1, dims=0)).item(),
        "windows": windows,
    }


def encode_episodes(
    model, h5: h5py.File, episodes: np.ndarray, batch_size: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Encode every frame of the chosen episodes.

    Returns latents (N, D) on the CPU, positions (N, 2) and episode ids (N,).
    """
    lengths, offsets = h5["ep_len"][:], h5["ep_offset"][:]
    latents, positions, owners = [], [], []
    for number, episode in enumerate(np.sort(episodes)):
        start, end = int(offsets[episode]), int(offsets[episode] + lengths[episode])
        frames = h5["pixels"][start:end]
        positions.append(h5["proprio"][start:end, :2])
        owners.append(np.full(end - start, episode))
        for first in range(0, end - start, batch_size):
            latents.append(model.encode_frames(frames[first : first + batch_size]).cpu())
        if (number + 1) % 500 == 0:
            print(f"encoded {number + 1}/{len(episodes)} episodes", flush=True)
    return torch.cat(latents), np.concatenate(positions), np.concatenate(owners)


def fit_reward_head(
    latents: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    device: torch.device,
    seed: int = 0,
    max_epochs: int = REWARD_MAX_EPOCHS,
    patience: int = REWARD_PATIENCE,
) -> tuple[torch.nn.Module, dict]:
    """Fit the head by MSE with early stopping on the held-out episodes."""
    torch.manual_seed(seed)
    head = make_reward_head(latents.size(-1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=REWARD_LR, weight_decay=REWARD_WEIGHT_DECAY)
    train_x = latents[train_mask].to(device)
    train_y = targets[train_mask].to(device)
    val_x = latents[~train_mask].to(device)
    val_y = targets[~train_mask].to(device)

    best_mse = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(train_x.size(0), device=device)
        for first in range(0, order.size(0), REWARD_BATCH):
            index = order[first : first + REWARD_BATCH]
            loss = F.mse_loss(head(train_x[index]).squeeze(-1), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(head(val_x).squeeze(-1), val_y).item()
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(head.state_dict())
        elif epoch - best_epoch >= patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    head.requires_grad_(False)
    return head, {"val_mse": best_mse, "best_epoch": best_epoch, "epochs_run": epoch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--reward-episodes", type=int, default=REWARD_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)
    artifacts = args.workdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    weights = hf_hub_download(
        MODEL_REPO, "weights.pt", revision=MODEL_REVISION, local_dir=artifacts
    )
    config = hf_hub_download(
        MODEL_REPO, "config.json", revision=MODEL_REVISION, local_dir=artifacts
    )
    archive = hf_hub_download(
        DATA_REPO,
        DATA_ARCHIVE,
        revision=DATA_REVISION,
        repo_type="dataset",
        local_dir=artifacts,
    )
    h5_path = extract_dataset(Path(archive), artifacts / "dataset")
    print(f"dataset: {h5_path}", flush=True)

    model = load_pretrained(weights, config, device)
    with h5py.File(h5_path, "r") as h5:
        print("columns:", {k: h5[k].shape for k in h5.keys()}, flush=True)
        stats = compute_action_stats(h5)
        render_mae = validate_render_match(h5)
        print(f"render MAE against dataset frames: {render_mae:.3f}", flush=True)
        if render_mae > MAX_RENDER_MAE:
            raise SystemExit(
                f"Rendered frames differ from the dataset (MAE {render_mae:.2f} > "
                f"{MAX_RENDER_MAE}). The environment does not match the checkpoint's "
                "training data, so results would be invalid."
            )
        dynamics = validate_dynamics(h5)
        print(f"dataset dynamics: {dynamics}", flush=True)
        if not MIN_SPEED <= dynamics["speed"] <= MAX_SPEED:
            raise SystemExit(
                f"Dataset speed {dynamics['speed']:.2f} is outside "
                f"[{MIN_SPEED}, {MAX_SPEED}], but this environment moves at 5.0."
            )
        if dynamics["aligned_error"] >= min(dynamics["previous_error"], dynamics["next_error"]):
            raise SystemExit(
                "Dataset actions are not aligned as p[t + 1] = p[t] + speed * a[t], "
                "so dense windows would pair frames with the wrong actions."
            )
        alignment = validate_checkpoint_alignment(model, h5, stats)
        print(f"checkpoint alignment: {alignment}", flush=True)
        if alignment["aligned_mse"] > MAX_ALIGNED_MSE_RATIO * alignment["shuffled_mse"]:
            raise SystemExit(
                "The checkpoint predicts real data no better with its real actions than "
                f"with shuffled ones ({alignment}), so this pipeline does not match it."
            )

        rng = np.random.default_rng(args.seed)
        total_episodes = h5["ep_len"].shape[0]
        chosen = rng.choice(
            total_episodes, size=min(args.reward_episodes, total_episodes), replace=False
        )
        latents, positions, owners = encode_episodes(model, h5, chosen, args.encode_batch)

    targets = torch.as_tensor(task_scores(positions), dtype=torch.float32)
    shuffled = rng.permutation(chosen)
    val_count = max(1, int(round(len(shuffled) * REWARD_VAL_FRACTION)))
    train_mask = torch.as_tensor(~np.isin(owners, shuffled[:val_count]))
    head, metrics = fit_reward_head(latents, targets, train_mask, device, seed=args.seed)

    torch.save(head.cpu().state_dict(), artifacts / "reward_head.pt")
    report = {
        "action_stats": stats,
        "render_mae": render_mae,
        "dynamics": dynamics,
        "alignment": alignment,
        "reward_head": metrics,
        "frames": int(latents.size(0)),
        "episodes": int(len(chosen)),
        "score_range": [float(targets.min()), float(targets.max())],
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
    }
    (artifacts / "prepare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
