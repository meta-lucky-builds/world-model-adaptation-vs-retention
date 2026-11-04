"""Run every arm of stage 1, in parallel on one GPU, and skip finished runs.

Usage:
    python -m jepa_ttt.sweep --workdir W --jobs 4

Restarting after an interruption repeats only the runs without a finished log.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adaptation import ARMS, FROZEN, STAGE1_ARMS
from .report import MIN_NO_SHIFT_SCORE, MIN_SHIFT_DAMAGE


def is_finished(path: Path) -> bool:
    """A run counts as finished only if its log ends with the ``done`` record."""
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    if not lines:
        return False
    try:
        return json.loads(lines[-1]).get("kind") == "done"
    except json.JSONDecodeError:
        return False


def plan_runs(
    shifts: list[str], seeds: list[int], arms: tuple[str, ...] = STAGE1_ARMS
) -> tuple[list[tuple], list[tuple]]:
    """Return the frozen runs, which go first, and the adaptive runs.

    A frozen run has one evaluation and no adaptation, so one seed is enough.
    Frozen runs also collect the shared trajectories, so they must finish before
    adaptive runs on the same shift start.
    """
    frozen = [(FROZEN, shift, seeds[0]) for shift in ("none", *shifts)]
    adaptive = [(arm, shift, seed) for shift in shifts for arm in arms for seed in seeds]
    return frozen, adaptive


def frozen_score(results: Path, shift: str, seed: int) -> float:
    path = results / f"{FROZEN}__{shift}__seed{seed}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return next(row["score_final"] for row in rows if row.get("kind") == "eval")


def check_baselines(results: Path, shifts: list[str], seed: int) -> list[str]:
    """Return the shifts worth running, or stop if the planner itself fails.

    Adaptation cannot be judged if the planner fails without a shift, and a shift
    is left out if it does not hurt the frozen model.
    """
    no_shift = frozen_score(results, "none", seed)
    print(f"frozen no-shift score: {no_shift:.3f}", flush=True)
    if no_shift < MIN_NO_SHIFT_SCORE:
        raise SystemExit(
            f"Frozen no-shift score {no_shift:.3f} is below {MIN_NO_SHIFT_SCORE}: the "
            "planner or reward head is not working, so adaptive runs were not started."
        )
    kept = []
    for shift in shifts:
        shifted = frozen_score(results, shift, seed)
        print(f"frozen {shift} score: {shifted:.3f}", flush=True)
        if no_shift - shifted >= MIN_SHIFT_DAMAGE:
            kept.append(shift)
        else:
            print(
                f"LEAVING OUT the {shift} shift: it only lowers the frozen score from "
                f"{no_shift:.3f} to {shifted:.3f}, too little to recover.",
                flush=True,
            )
    if not kept:
        raise SystemExit("No shift hurts the frozen model; adaptive runs were not started.")
    return kept


def run_one(job: tuple, args, results: Path, logs: Path, gpu: int = 0) -> tuple[tuple, int]:
    arm, shift, seed = job
    name = f"{arm}__{shift}__seed{seed}"
    if is_finished(results / f"{name}.jsonl"):
        print(f"skip {name}: already finished", flush=True)
        return job, 0
    command = [
        sys.executable,
        "-m",
        "jepa_ttt.run",
        "--workdir",
        str(args.workdir),
        "--arm",
        arm,
        "--shift",
        shift,
        "--seed",
        str(seed),
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(args.eval_every),
        "--eval-dense-until",
        str(args.eval_dense_until),
        "--eval-dense-every",
        str(args.eval_dense_every),
        "--reversal-episodes",
        str(0 if arm == FROZEN else args.reversal_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-batch",
        str(args.eval_batch),
        "--trajectories",
        str(args.trajectories),
        "--num-samples",
        str(args.num_samples),
        "--iterations",
        str(args.iterations),
        "--elites",
        str(args.elites),
        "--device",
        args.device,
        "--overwrite",
    ]
    if args.tag:
        command += ["--tag", args.tag]
    if args.allow_cpu:
        command.append("--allow-cpu")
    print(f"start {name} on GPU {gpu}", flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Lets many runs share a card without fragmenting its memory.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if args.gpus > 1:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = 1
    # One retry, because several runs can peak in memory at the same moment.
    for attempt in range(2):
        with (logs / f"{name}.attempt{attempt}.log").open("w") as log:
            code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            ).returncode
        if code == 0:
            break
    print(f"{'done' if code == 0 else f'FAILED ({code})'} {name}", flush=True)
    return job, code


def run_jobs(jobs: list[tuple], args, results: Path, logs: Path) -> list[tuple[tuple, int]]:
    """Run jobs concurrently, keeping the load even across GPUs.

    Slots are handed out per GPU rather than per job, so a run that finishes
    early, or is skipped as finished, frees capacity on the card it was using.
    """
    slots: queue.Queue[int] = queue.Queue()
    for slot in range(args.jobs):
        slots.put(slot % args.gpus)

    def run_in_slot(job: tuple) -> tuple[tuple, int]:
        gpu = slots.get()
        try:
            return run_one(job, args, results, logs, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        return list(pool.map(run_in_slot, jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, help="runs at once, across all GPUs")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs to spread runs across")
    parser.add_argument("--shifts", nargs="+", default=["grid"], choices=("grid", "wave"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-dense-until", type=int, default=0)
    parser.add_argument("--eval-dense-every", type=int, default=1)
    parser.add_argument("--reversal-episodes", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(STAGE1_ARMS), choices=tuple(ARMS))
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--skip-baseline-gate",
        action="store_true",
        help="for smoke tests, whose few episodes make the frozen scores meaningless",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    results = args.workdir / (f"results_{args.tag}" if args.tag else "results")
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.jobs < 1 or args.gpus < 1:
        raise SystemExit("--jobs and --gpus must be at least 1")

    frozen, adaptive = plan_runs(args.shifts, args.seeds, tuple(args.arms))
    # Frozen runs cover different shifts, so their trajectory files never collide.
    outcomes = run_jobs(frozen, args, results, logs)
    if any(code != 0 for _, code in outcomes):
        raise SystemExit(f"Frozen runs failed; see {logs}. Adaptive runs were not started.")
    if not args.skip_baseline_gate:
        kept = check_baselines(results, args.shifts, args.seeds[0])
        adaptive = [job for job in adaptive if job[1] in kept]
    outcomes = run_jobs(adaptive, args, results, logs)
    failed = [job for job, code in outcomes if code != 0]
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}; see {logs}")
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
