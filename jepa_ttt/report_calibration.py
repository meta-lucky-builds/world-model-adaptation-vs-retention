"""Judge action calibration against persistent fine-tuning with fixed rules.

Usage:
    python -m jepa_ttt.report_calibration --results W/results_calibration
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import MIN_NO_SHIFT_SCORE, paired_auc_gains, read_run, run_auc

CALIBRATION = "action_calibration"
# x16 does as many updates as the calibrator, and x4 was the best stage 1 arm.
BASELINES = ("persistent_replay_x4", "persistent_replay_x16")
TABLE_ARMS = (
    "frozen",
    "persistent_replay_x4",
    "persistent_replay_x8",
    "persistent_replay_x16",
    CALIBRATION,
    "calibration_before_now",
    "calibration_fast",
    "latent_correction",
    "rollout_loss_x4",
    "anchored_x4",
)
SHIFTS = ("grid", "wave")
# Each idea against the arm it changes one thing from. These are separate questions
# from whether calibration won, and with this many, one lucky pass is possible.
IDEA_BASELINES = {
    "calibration_fast": CALIBRATION,
    "latent_correction": "persistent_replay_x16",
    "rollout_loss_x4": "persistent_replay_x4",
    "anchored_x4": "persistent_replay_x4",
}
BEFORE_NOW = "calibration_before_now"
MAX_AUC_COST = 0.02  # how much AUC before-plus-now may lose to plain calibration
BEFORE_NOW_REVERSAL_EPISODE = 10
EXPECTED_SEEDS = {0, 1, 2}

# Declared before any result exists.
MIN_AUC_GAIN = 0.02
RECOVERY_FRACTION = 0.9  # of the frozen model's no-shift score
MAX_RECOVERY_RATIO = 0.5  # calibration must recover in at most half a baseline's episodes
MAX_APPLIED_MSE_RATIO = 0.5  # of the frozen model's error on commanded actions
REVERSAL_EPISODES = (0, 10, 50)  # reported only, because old windows dilute replay


def mean_curve(runs: list[dict], metric: str = "score_final"):
    """Return evaluation episodes and the across-run mean score at each."""
    episodes = np.array([e["episode"] for e in runs[0]["evals"]])
    curves = []
    for run in runs:
        if [e["episode"] for e in run["evals"]] != episodes.tolist():
            raise ValueError("runs of one arm have different evaluation episodes")
        curves.append([e[metric] for e in run["evals"]])
    return episodes, np.mean(curves, axis=0)


def episodes_to_reach(episodes: np.ndarray, curve: np.ndarray, target: float) -> int | None:
    """First evaluated episode at or above ``target`` that also holds at the next one."""
    above = curve >= target
    for index in np.flatnonzero(above):
        if index == len(curve) - 1 or above[index + 1]:
            return int(episodes[index])
    return None


def reversal_score(runs: list[dict], episode: int, metric: str = "score_final") -> float | None:
    """Mean score ``episode`` episodes after the world switched back."""
    scores = [row[metric] for run in runs for row in run["reversal"] if row["episode"] == episode]
    return float(np.mean(scores)) if scores else None


def recovers_faster(calibrated: int | None, baseline: int | None, horizon: int = 500) -> bool:
    """Calibration must recover in at most half the baseline's episodes.

    A baseline that never recovers counts as recovering at the end of the run.
    """
    if calibrated is None:
        return False
    return calibrated <= MAX_RECOVERY_RATIO * (horizon if baseline is None else baseline)


def premise_holds(frozen_run: dict) -> tuple[bool, float, float]:
    """Does telling the frozen model the applied actions explain the shift?"""
    row = frozen_run["evals"][0]
    commanded, applied = row["latent_mse"], row["latent_mse_applied"]
    return applied <= MAX_APPLIED_MSE_RATIO * commanded, commanded, applied


def print_table(grouped: dict, shift: str, target: float, metric: str) -> dict:
    """Print one shift's results and return each arm's recovery episode."""
    print(
        f"{'arm':24}{'runs':>5}{'best':>8}{'auc':>8}{'succ':>7}{'mse':>9}{'upd':>7}"
        f"{'ep to ' + format(target, '.2f'):>12}"
        + "".join(f"{'back ' + str(e):>9}" for e in REVERSAL_EPISODES)
    )

    def show(value):
        return "-" if value is None else f"{value:.3f}"

    recovery = {}
    for arm in TABLE_ARMS:
        runs = grouped.get((shift, arm))
        if not runs:
            continue
        episodes, curve = mean_curve(runs, metric)
        later = episodes > 0
        best = float(curve[later].max()) if later.any() else float(curve[0])
        auc = float(
            np.mean(
                [run_auc(episodes, np.array([e[metric] for e in run["evals"]])) for run in runs]
            )
        )
        recovery[arm] = episodes_to_reach(episodes, curve, target)
        mses = [r["evals"][-1]["latent_mse"] for r in runs]
        mse = None if None in mses else float(np.mean(mses))
        print(
            f"{arm:24}{len(runs):>5}{best:>8.3f}{auc:>8.3f}"
            f"{np.mean([r['evals'][-1]['success_rate'] for r in runs]):>7.2f}"
            f"{show(mse):>9}{runs[0]['evals'][-1]['updates']:>7}"
            f"{str(recovery[arm]) if recovery[arm] is not None else 'never':>12}"
            + "".join(f"{show(reversal_score(runs, e, metric)):>9}" for e in REVERSAL_EPISODES)
        )
    return recovery


def has_all_seeds(*arms: list[dict]) -> bool:
    """Each arm must have finished exactly the expected seeds."""
    return all({run["seed"] for run in runs} == EXPECTED_SEEDS for runs in arms)


def judge_ideas(grouped: dict, shift: str, no_shift: float, metric: str) -> dict:
    """Print each further idea's fixed test on one shift, and return the verdicts."""
    verdicts = {}
    for arm, base in IDEA_BASELINES.items():
        runs, base_runs = grouped.get((shift, arm)), grouped.get((shift, base))
        if not runs or not base_runs:
            print(f"MISSING  {arm} or {base}")
            verdicts[arm] = None
            continue
        if not has_all_seeds(runs, base_runs):
            print(f"MISSING  {arm} and {base} haven't both finished seeds {EXPECTED_SEEDS}")
            verdicts[arm] = None
            continue
        gains = paired_auc_gains(base_runs, runs, metric)
        mean_gain = float(np.mean(list(gains.values())))
        passed = mean_gain >= MIN_AUC_GAIN and all(g > 0 for g in gains.values())
        rounded = {seed: round(gain, 3) for seed, gain in gains.items()}
        print(
            f"{'PASS' if passed else 'FAIL'}  {arm} AUC gain over {base}: {rounded}, "
            f"mean {mean_gain:.3f}; needs mean >= {MIN_AUC_GAIN} and every seed positive"
        )
        verdicts[arm] = passed

    runs, plain = grouped.get((shift, BEFORE_NOW)), grouped.get((shift, CALIBRATION))
    if not runs or not plain or not has_all_seeds(runs, plain):
        print(
            f"MISSING  {BEFORE_NOW} and {CALIBRATION} haven't both finished seeds {EXPECTED_SEEDS}"
        )
        verdicts[BEFORE_NOW] = None
        return verdicts
    gains = paired_auc_gains(plain, runs, metric)
    mean_gain = float(np.mean(list(gains.values())))
    back = reversal_score(runs, BEFORE_NOW_REVERSAL_EPISODE, metric)
    back_plain = reversal_score(plain, BEFORE_NOW_REVERSAL_EPISODE, metric)
    if back is None or back_plain is None:
        print(f"MISSING  {BEFORE_NOW} or {CALIBRATION} switch-back scores")
        verdicts[BEFORE_NOW] = None
        return verdicts
    target = RECOVERY_FRACTION * no_shift
    recovers = back >= target and back > back_plain
    affordable = mean_gain >= -MAX_AUC_COST
    print(
        f"{'PASS' if recovers else 'FAIL'}  {BEFORE_NOW} {BEFORE_NOW_REVERSAL_EPISODE} episodes "
        f"after switching back: {back}, plain {back_plain}; needs >= {target:.2f} and above plain"
    )
    print(
        f"{'PASS' if affordable else 'FAIL'}  {BEFORE_NOW} AUC change from plain calibration: "
        f"{mean_gain:.3f}; may not lose more than {MAX_AUC_COST}"
    )
    verdicts[BEFORE_NOW] = recovers and affordable
    return verdicts


def judge_shift(grouped: dict, shift: str, no_shift: float, metric: str) -> bool | None:
    """Print one shift's results; return whether calibration won, or None if incomplete."""
    print(f"\n=== {shift} shift")
    target = RECOVERY_FRACTION * no_shift
    recovery = print_table(grouped, shift, target, metric)

    frozen = grouped.get((shift, "frozen"))
    calibrated = grouped.get((shift, CALIBRATION))
    baselines = {arm: grouped.get((shift, arm)) for arm in BASELINES}
    if not frozen or not calibrated or not all(baselines.values()):
        print("INCOMPLETE: a frozen, baseline or calibration run is missing")
        return None
    seeds = {run["seed"] for run in calibrated}
    if any({run["seed"] for run in runs} != seeds for runs in baselines.values()):
        print("INCOMPLETE: arms finished different seeds")
        return None

    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    holds, commanded, applied = premise_holds(frozen[0])
    print(
        f"{verdict(holds)}  Premise: frozen model error {commanded:.4f} on commanded actions, "
        f"{applied:.4f} on applied actions; needs at most {MAX_APPLIED_MSE_RATIO} of commanded"
    )
    won = holds
    episodes = np.array([e["episode"] for e in calibrated[0]["evals"]])
    for arm, runs in baselines.items():
        gains = paired_auc_gains(runs, calibrated, metric)
        mean_gain = float(np.mean(list(gains.values())))
        higher = mean_gain >= MIN_AUC_GAIN and all(g > 0 for g in gains.values())
        rounded = {seed: round(gain, 3) for seed, gain in gains.items()}
        print(
            f"{verdict(higher)}  AUC gain over {arm}: {rounded}, mean {mean_gain:.3f}; "
            f"needs mean >= {MIN_AUC_GAIN} and every seed positive"
        )
        faster = recovers_faster(recovery[CALIBRATION], recovery[arm], int(episodes[-1]))
        print(
            f"{verdict(faster)}  Sustained {target:.2f}: calibration at episode "
            f"{recovery[CALIBRATION]}, {arm} at {recovery[arm]}; needs at most half"
        )
        won = won and higher and faster
    return won


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metric", default="score_final", choices=("score_final", "score_peak"))
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.results.glob("*.jsonl")):
        run = read_run(path)
        if run is not None:
            grouped[(run["shift"], run["arm"])].append(run)
    no_shift_runs = grouped.get(("none", "frozen"))
    if not no_shift_runs:
        raise SystemExit(f"no finished frozen no-shift run in {args.results}")
    no_shift = no_shift_runs[0]["evals"][0][args.metric]
    print(f"Frozen no-shift score: {no_shift:.3f}")
    planner_works = no_shift >= MIN_NO_SHIFT_SCORE
    if not planner_works:
        print(f"FAIL  The no-shift score must be at least {MIN_NO_SHIFT_SCORE}")

    outcomes = [judge_shift(grouped, shift, no_shift, args.metric) for shift in SHIFTS]
    print("\n=== Further ideas, each judged on its own")
    idea_verdicts = defaultdict(list)
    for shift in SHIFTS:
        print(f"--- {shift} shift")
        for arm, verdict in judge_ideas(grouped, shift, no_shift, args.metric).items():
            idea_verdicts[arm].append(verdict)
    for arm, verdicts in idea_verdicts.items():
        if all(v is True for v in verdicts):
            summary = "passed on both shifts"
        elif any(v is False for v in verdicts):
            summary = "did not pass on both shifts"
        else:
            summary = "incomplete"
        print(f"{arm}: {summary}")
    print(
        "\nExcept for before-plus-now, the 'back' columns are reported only, because after "
        "the world silently returns nearly all replay data comes from the shifted world."
    )
    if planner_works and all(outcome is True for outcome in outcomes):
        print("Action calibration WON on both shifts: try it on Reacher next.")
    elif not planner_works or any(outcome is False for outcome in outcomes):
        print("Action calibration did not win on every shift: see the failures above.")
    else:
        print("Incomplete: some runs are missing.")


if __name__ == "__main__":
    main()
