"""Aggregate runs with the paper's metrics and apply the stage 1 gates.

Usage:
    python -m jepa_ttt.report --results W/results --shift grid
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Table A2 and Table A5 of the paper, for context only: the paper trained its
# own world model, so absolute values are not expected to match exactly.
PAPER = {
    ("none", "frozen"): (0.804, 0.804),
    ("grid", "frozen"): (0.211, 0.211),
    ("grid", "persistent_replay"): (0.799, 0.725),
    ("grid", "persistent_local"): (0.789, 0.719),
    ("grid", "reset_local"): (0.218, 0.213),
    ("wave", "frozen"): (0.557, 0.557),
    ("wave", "persistent_replay"): (0.799, 0.728),
    ("wave", "persistent_local"): (0.797, 0.769),
    ("wave", "reset_local"): (0.597, 0.561),
}

# Declared before any result exists.
MIN_NO_SHIFT_SCORE = 0.70  # the paper's frozen no-shift value is 0.804
MIN_SHIFT_DAMAGE = 0.15  # the shift must lower the frozen score at least this much
MIN_ADAPTATION_GAIN = 0.20  # persistent_replay best minus frozen best
MIN_PERSISTENCE_GAIN = 0.10  # persistent_local best minus reset_local best
MIN_EXTENSION_AUC_GAIN = 0.02  # persistent_replay_x4 AUC minus persistent_replay AUC


def read_run(path: Path) -> dict | None:
    """Return a run's evaluation rows, or None if it did not finish."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get("kind") != "done":
        print(f"skipping unfinished run {path.name}")
        return None
    evals = [r for r in rows if r.get("kind") == "eval"]
    return {
        "arm": evals[0]["arm"],
        "shift": evals[0]["shift"],
        "seed": evals[0]["seed"],
        "evals": evals,
        "reversal": [r for r in rows if r.get("kind") == "reversal"],
    }


def run_auc(episodes: np.ndarray, scores: np.ndarray) -> float:
    """Normalized trapezoidal area of one run's held-out score curve."""
    if episodes.size == 1:
        return float(scores[0])
    area = np.sum((scores[1:] + scores[:-1]) * np.diff(episodes) / 2.0)
    return float(area / (episodes[-1] - episodes[0]))


def summarize(runs: list[dict], metric: str) -> dict:
    """Best: max of the across-run mean after episode 0. AUC: mean of run AUCs."""
    episodes = np.array([e["episode"] for e in runs[0]["evals"]])
    curves = []
    for run in runs:
        current = np.array([e["episode"] for e in run["evals"]])
        if not np.array_equal(current, episodes):
            raise ValueError("runs of one arm have different evaluation episodes")
        curves.append([e[metric] for e in run["evals"]])
    curves = np.asarray(curves, dtype=np.float64)
    mean_curve = curves.mean(axis=0)
    later = episodes > 0
    best = float(mean_curve[later].max()) if later.any() else float(mean_curve[0])
    aucs = [run_auc(episodes, curve) for curve in curves]
    mse = [r["evals"][-1]["latent_mse"] for r in runs]
    return {
        "runs": len(runs),
        "best": best,
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "success_last": float(np.mean([r["evals"][-1]["success_rate"] for r in runs])),
        "latent_mse_last": None if None in mse else float(np.mean(mse)),
        "updates_last": int(runs[0]["evals"][-1]["updates"]),
    }


def paired_auc_gains(
    base_runs: list[dict], other_runs: list[dict], metric: str
) -> dict[int, float]:
    """AUC of ``other`` minus ``base`` for each seed both arms finished."""

    def by_seed(runs: list[dict]) -> dict[int, float]:
        result = {}
        for run in runs:
            episodes = np.array([e["episode"] for e in run["evals"]])
            scores = np.array([e[metric] for e in run["evals"]], dtype=np.float64)
            result[run["seed"]] = run_auc(episodes, scores)
        return result

    base, other = by_seed(base_runs), by_seed(other_runs)
    return {seed: other[seed] - base[seed] for seed in sorted(base.keys() & other.keys())}


def gate(name: str, passed: bool | None, detail: str) -> bool | None:
    label = {True: "PASS", False: "FAIL", None: "MISSING"}[passed]
    print(f"{label:8}{name}: {detail}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--shift", default="grid", choices=("grid", "wave"))
    parser.add_argument("--metric", default="score_final", choices=("score_final", "score_peak"))
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.results.glob("*.jsonl")):
        run = read_run(path)
        if run is not None:
            grouped[(run["shift"], run["arm"])].append(run)
    if not grouped:
        raise SystemExit(f"no finished runs in {args.results}")

    summaries = {key: summarize(runs, args.metric) for key, runs in grouped.items()}
    print(
        f"{'shift':6}{'arm':20}{'runs':>5}{'best':>8}{'auc':>8}{'std':>8}"
        f"{'succ':>7}{'mse':>10}{'upd':>6}   paper best/auc"
    )
    for (shift, arm), s in sorted(summaries.items()):
        paper = PAPER.get((shift, arm))
        reference = f"{paper[0]:.3f}/{paper[1]:.3f}" if paper else "-"
        mse = f"{s['latent_mse_last']:.4f}" if s["latent_mse_last"] is not None else "-"
        print(
            f"{shift:6}{arm:20}{s['runs']:>5}{s['best']:>8.3f}{s['auc']:>8.3f}"
            f"{s['auc_std']:>8.3f}{s['success_last']:>7.2f}{mse:>10}"
            f"{s['updates_last']:>6}   {reference}"
        )

    def best(shift: str, arm: str) -> float | None:
        entry = summaries.get((shift, arm))
        return entry["best"] if entry else None

    print()
    no_shift = best("none", "frozen")
    frozen = best(args.shift, "frozen")
    replay = best(args.shift, "persistent_replay")
    local = best(args.shift, "persistent_local")
    reset = best(args.shift, "reset_local")

    def compare(a, b):
        return None if a is None or b is None else a - b

    damage = compare(no_shift, frozen)
    gain = compare(replay, frozen)
    persistence = compare(local, reset)
    outcomes = [
        gate(
            "planner works without a shift",
            None if no_shift is None else no_shift >= MIN_NO_SHIFT_SCORE,
            f"{no_shift} >= {MIN_NO_SHIFT_SCORE}",
        ),
        gate(
            "shift damages the frozen model",
            None if damage is None else damage >= MIN_SHIFT_DAMAGE,
            f"{damage} >= {MIN_SHIFT_DAMAGE}",
        ),
        gate(
            "persistent replay beats frozen",
            None if gain is None else gain >= MIN_ADAPTATION_GAIN,
            f"{gain} >= {MIN_ADAPTATION_GAIN}",
        ),
        gate(
            "persistence beats episodic reset",
            None if persistence is None else persistence >= MIN_PERSISTENCE_GAIN,
            f"{persistence} >= {MIN_PERSISTENCE_GAIN}",
        ),
    ]
    if all(outcome is True for outcome in outcomes):
        print("\nStage 1 passed: moving to stage 2 is justified.")
    elif any(outcome is False for outcome in outcomes):
        print("\nStage 1 failed: investigate before stage 2.")
    else:
        print("\nStage 1 incomplete: some runs are missing.")

    # The extension is reported separately and never decides the stage 1 result.
    gains = paired_auc_gains(
        grouped.get((args.shift, "persistent_replay"), []),
        grouped.get((args.shift, "persistent_replay_x4"), []),
        args.metric,
    )
    if gains:
        mean_gain = float(np.mean(list(gains.values())))
        improved = mean_gain >= MIN_EXTENSION_AUC_GAIN and all(gain > 0 for gain in gains.values())
        print(
            f"\nExtension, 4x updates versus persistent_replay: AUC gain per seed "
            f"{ {seed: round(gain, 3) for seed, gain in gains.items()} }, mean {mean_gain:.3f}. "
            f"{'Improved' if improved else 'No clear improvement'} (needs mean >= "
            f"{MIN_EXTENSION_AUC_GAIN} and every seed positive)."
        )


if __name__ == "__main__":
    main()
