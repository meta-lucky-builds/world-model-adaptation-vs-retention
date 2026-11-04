"""Run one arm on one shift for one seed and append results as JSON lines.

Usage:
    python -m jepa_ttt.run --workdir W --arm persistent_replay --shift grid --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .adaptation import ALL_ARMS, FROZEN
from .envs import SHIFTS
from .experiment import (
    ActionStats,
    collect_fixed_trajectories,
    run_deployment,
)
from .lewm import load_pretrained
from .planning import PlannerConfig, make_reward_head
from .prepare import require_device


def load_artifacts(workdir: Path, device: torch.device):
    """Load the checkpoint, fitted reward head and action statistics."""
    artifacts = workdir / "artifacts"
    report_path = artifacts / "prepare.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} is missing; run jepa_ttt.prepare first.")
    report = json.loads(report_path.read_text())
    model = load_pretrained(artifacts / "weights.pt", artifacts / "config.json", device)
    head = make_reward_head()
    head.load_state_dict(
        torch.load(artifacts / "reward_head.pt", map_location="cpu", weights_only=True)
    )
    head.to(device).eval()
    head.requires_grad_(False)
    stats = ActionStats(
        mean=np.asarray(report["action_stats"]["mean"], dtype=np.float32),
        std=np.asarray(report["action_stats"]["std"], dtype=np.float32),
    )
    return model, head, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--shift", choices=SHIFTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-dense-until", type=int, default=0)
    parser.add_argument("--eval-dense-every", type=int, default=1)
    parser.add_argument("--reversal-episodes", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=PlannerConfig.num_samples)
    parser.add_argument("--iterations", type=int, default=PlannerConfig.iterations)
    parser.add_argument("--elites", type=int, default=PlannerConfig.elites)
    parser.add_argument("--start-mode", choices=("uniform_left", "fixed"), default="uniform_left")
    parser.add_argument("--tag", default="", help="separates smoke tests from real runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--threads", type=int, default=1, help="CPU threads for simulator rendering"
    )
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)
    torch.set_num_threads(args.threads)
    planner = PlannerConfig(
        num_samples=args.num_samples, iterations=args.iterations, elites=args.elites
    )
    name = f"{args.arm}__{args.shift}__seed{args.seed}"
    results = args.workdir / (f"results_{args.tag}" if args.tag else "results")
    results.mkdir(parents=True, exist_ok=True)
    out_path = results / f"{name}.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    model, head, stats = load_artifacts(args.workdir, device)

    # All arms on a shift share one trajectory set from the pretrained model.
    trajectory_path = results / (
        f"trajectories__{args.shift}__{args.start_mode}__n{args.trajectories}.pt"
    )
    if trajectory_path.exists():
        trajectories = torch.load(trajectory_path, weights_only=True)
    else:
        trajectories = collect_fixed_trajectories(
            model,
            head,
            stats,
            args.shift,
            planner,
            args.trajectories,
            args.eval_batch,
            args.start_mode,
        )
        # Write then rename so a crash never leaves a partial file behind.
        partial = trajectory_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(trajectories, partial)
        os.replace(partial, trajectory_path)

    with out_path.open("w") as handle:
        header = {
            "kind": "config",
            "args": {k: str(v) for k, v in vars(args).items()},
            "planner": dataclasses.asdict(planner),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        handle.write(json.dumps(header) + "\n")

        def log(row: dict) -> None:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"[{name}] {row['kind']} ep {row['episode']:>4} "
                f"final {row['score_final']:.3f} "
                f"peak {row['score_peak']:.3f} success {row['success_rate']:.2f} "
                f"mse {row.get('latent_mse')} applied mse "
                f"{row.get('latent_mse_applied')} updates {row['updates']}",
                flush=True,
            )

        started = time.perf_counter()
        run_deployment(
            model,
            head,
            stats,
            args.arm,
            args.shift,
            args.seed,
            planner,
            args.episodes,
            args.eval_every,
            args.eval_episodes,
            args.eval_batch,
            args.start_mode,
            trajectories,
            log,
            final_state_path=(None if args.arm == FROZEN else results / f"{name}__final.pt"),
            dense_until=args.eval_dense_until,
            dense_every=args.eval_dense_every,
            reversal_episodes=args.reversal_episodes,
        )
        handle.write(json.dumps({"kind": "done", "seconds": time.perf_counter() - started}) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
