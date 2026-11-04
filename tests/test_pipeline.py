"""End-to-end run of prepare, run and report with a synthetic dataset.

Needs the published checkpoint through JEPA_TTT_WEIGHTS and JEPA_TTT_CONFIG.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import zstandard

from jepa_ttt import prepare, report, run
from jepa_ttt.envs import ShiftedTwoRoom

pytestmark = pytest.mark.skipif(
    "JEPA_TTT_WEIGHTS" not in os.environ, reason="checkpoint not supplied"
)


def synthetic_archive(path: Path) -> None:
    """Build a small dataset that follows the real environment's dynamics."""
    rng = np.random.default_rng(0)
    env = ShiftedTwoRoom("none")
    frames, positions, actions = [], [], []
    lengths = []
    for episode in range(10):
        env.reset_task(seed=episode)
        steps = 24  # long enough to hold a dense window
        for step in range(steps):
            frames.append(env.render())
            positions.append(env.agent_position.numpy().copy())
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            if step == steps - 1:
                actions.append(np.full(2, np.nan, dtype=np.float32))
            else:
                actions.append(action)
                env.step(action)
        lengths.append(steps)
    h5_path = path.parent / "tworoom.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["pixels"] = np.stack(frames)
        h5["proprio"] = np.stack(positions)
        h5["action"] = np.stack(actions)
        h5["ep_len"] = np.array(lengths)
        h5["ep_offset"] = np.cumsum([0, *lengths[:-1]])
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(h5_path, arcname="tworoom.h5")
    path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    h5_path.unlink()


def fake_download(source: dict[str, Path]):
    def download(repo_id, filename, revision=None, repo_type=None, local_dir=None):
        target = Path(local_dir) / filename
        shutil.copy(source[filename], target)
        return str(target)

    return download


def run_module(module, monkeypatch, *argv) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, argv)])
    module.main()


def test_full_pipeline(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "source.tar.zst"
    synthetic_archive(archive)
    monkeypatch.setattr(
        prepare,
        "hf_hub_download",
        fake_download(
            {
                "weights.pt": Path(os.environ["JEPA_TTT_WEIGHTS"]),
                "config.json": Path(os.environ["JEPA_TTT_CONFIG"]),
                prepare.DATA_ARCHIVE: archive,
            }
        ),
    )
    # Random synthetic actions say nothing about the threshold set on real data.
    monkeypatch.setattr(prepare, "MAX_ALIGNED_MSE_RATIO", float("inf"))
    workdir = tmp_path / "work"
    run_module(prepare, monkeypatch, "--workdir", workdir, "--device", "cpu", "--allow-cpu")
    artifacts = workdir / "artifacts"
    report_data = json.loads((artifacts / "prepare.json").read_text())
    assert report_data["render_mae"] == pytest.approx(0.0, abs=1e-6)
    assert report_data["dynamics"]["speed"] == pytest.approx(5.0, abs=0.01)
    assert report_data["frames"] == 240
    assert report_data["alignment"]["aligned_mse"] > 0
    assert (artifacts / "reward_head.pt").exists()

    common = (
        "--workdir",
        workdir,
        "--device",
        "cpu",
        "--allow-cpu",
        "--seed",
        "0",
        "--episodes",
        "2",
        "--eval-every",
        "2",
        "--eval-episodes",
        "2",
        "--eval-batch",
        "2",
        "--trajectories",
        "2",
        "--num-samples",
        "4",
        "--iterations",
        "1",
        "--elites",
        "2",
        "--tag",
        "smoke",
    )
    jobs = [
        ("frozen", "none"),
        ("frozen", "grid"),
        ("persistent_replay", "grid"),
        ("persistent_local", "grid"),
        ("reset_local", "grid"),
    ]
    for arm, shift in jobs:
        run_module(run, monkeypatch, "--arm", arm, "--shift", shift, *common)

    results = workdir / "results_smoke"
    rows = [
        json.loads(line)
        for line in (results / "persistent_local__grid__seed0.jsonl").read_text().splitlines()
    ]
    assert rows[0]["kind"] == "config" and rows[-1]["kind"] == "done"
    evals = [row for row in rows if row["kind"] == "eval"]
    assert [row["episode"] for row in evals] == [0, 2]
    assert evals[-1]["updates"] == 8  # four per episode with batch size 8
    assert evals[-1]["latent_mse"] is not None
    state = torch.load(results / "persistent_local__grid__seed0__final.pt", weights_only=True)
    assert state["total_updates"] == 8

    with pytest.raises(SystemExit):
        run_module(run, monkeypatch, "--arm", "frozen", "--shift", "grid", *common)

    capsys.readouterr()
    run_module(report, monkeypatch, "--results", results, "--shift", "grid")
    output = capsys.readouterr().out
    assert "persistent_replay" in output
    assert "planner works without a shift" in output
    assert "Stage 1" in output
