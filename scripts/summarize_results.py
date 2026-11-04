"""Build results/summary.csv and results/verdicts.csv from the saved reports.

Usage:
    python scripts/summarize_results.py          # write both files
    python scripts/summarize_results.py --check  # fail if either file is out of date

The five ideas' per-seed logs weren't kept, so their verdicts use the means in the reports. That
settles every verdict here: any arm below the mean-gain threshold fails regardless of its seeds.
Passes that also need "every seed positive" would be marked as unconfirmed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jepa_ttt.report import MIN_NO_SHIFT_SCORE, run_auc  # noqa: E402
from jepa_ttt.report_calibration import (  # noqa: E402
    BEFORE_NOW,
    CALIBRATION,
    IDEA_BASELINES,
    MAX_AUC_COST,
    MIN_AUC_GAIN,
    RECOVERY_FRACTION,
    SHIFTS,
)

RESULTS = ROOT / "results"
REPORTS = {
    "calibration": RESULTS / "calibration" / "report.txt",
    "ideas": RESULTS / "ideas" / "report.txt",
}
COLUMNS = (
    "source", "shift", "arm", "runs", "best", "auc", "success", "latent_mse", "updates",
    "episodes_to_target", "back_0", "back_10", "back_50",
)  # fmt: skip
ROW = re.compile(
    r"^(?P<arm>[a-z0-9_]+)\s+(?P<runs>\d+)\s+(?P<best>[\d.]+)\s+(?P<auc>[\d.]+)\s+"
    r"(?P<success>[\d.]+)\s+(?P<latent_mse>[\d.]+|-)\s+(?P<updates>\d+)\s+"
    r"(?P<episodes_to_target>\d+|never)\s+(?P<back_0>[\d.]+|-)\s+(?P<back_10>[\d.]+|-)\s+"
    r"(?P<back_50>[\d.]+|-)$"
)
ROUNDING = 0.001  # means in the reports are rounded to three decimals
PAPER_EVAL_EVERY = 50  # the paper evaluates at episodes 0, 50, ..., 500


def parse_report(path: Path, source: str) -> tuple[float, list[dict]]:
    """Return the no-shift score and one row per arm and shift."""
    text = path.read_text()
    no_shift = float(re.search(r"Frozen no-shift score: ([\d.]+)", text).group(1))
    rows, shift = [], None
    for line in text.splitlines():
        header = re.match(r"=== (\w+) shift", line)
        if header:
            shift = header.group(1)
        elif line.startswith("==="):
            shift = None
        elif shift and (match := ROW.match(line.strip())):
            rows.append({"source": source, "shift": shift, **match.groupdict()})
    return no_shift, rows


def load_rows() -> tuple[float, list[dict]]:
    rows, scores = [], set()
    for source, path in REPORTS.items():
        no_shift, parsed = parse_report(path, source)
        scores.add(no_shift)
        # Both runs share one set of frozen baselines, so keep only the first copy.
        rows += [r for r in parsed if r["arm"] != "frozen" or source == "calibration"]
    if len(scores) != 1:
        raise SystemExit(f"The reports disagree on the no-shift score: {scores}")
    return scores.pop(), rows


def build_verdicts(no_shift: float, rows: list[dict]) -> list[dict]:
    """Apply each idea's fixed rule to the report means."""
    by_key = {(r["shift"], r["arm"]): r for r in rows}
    target = RECOVERY_FRACTION * no_shift
    verdicts = []

    def auc(shift: str, arm: str) -> float:
        return float(by_key[(shift, arm)]["auc"])

    for shift in SHIFTS:
        for arm, base in IDEA_BASELINES.items():
            gain = auc(shift, arm) - auc(shift, base)
            # Only a clear shortfall can be settled from means.
            settled = gain + ROUNDING < MIN_AUC_GAIN
            verdict = "fail" if settled else "unconfirmed: needs per-seed logs"
            verdicts.append(
                {"shift": shift, "arm": arm, "baseline": base, "auc_change": f"{gain:+.3f}",
                 "switch_back_10": by_key[(shift, arm)]["back_10"], "verdict": verdict}
            )  # fmt: skip
        gain = auc(shift, BEFORE_NOW) - auc(shift, CALIBRATION)
        back = float(by_key[(shift, BEFORE_NOW)]["back_10"])
        plain = float(by_key[(shift, CALIBRATION)]["back_10"])
        passed = back >= target and back > plain and gain - ROUNDING >= -MAX_AUC_COST
        verdicts.append(
            {"shift": shift, "arm": BEFORE_NOW, "baseline": CALIBRATION,
             "auc_change": f"{gain:+.3f}", "switch_back_10": f"{back:.3f}",
             "verdict": "pass" if passed else "fail"}
        )  # fmt: skip
    return verdicts


def read_records(arm: str, shift: str, seed: int) -> list[dict]:
    path = RESULTS / "calibration" / "runs" / f"{arm}__{shift}__seed{seed}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def seed_auc_at_paper_episodes(records: list[dict]) -> float:
    """AUC using only the paper's evaluations, at episodes 0, 50, ..., 500."""
    evals = [r for r in records if r.get("kind") == "eval" and r["episode"] % PAPER_EVAL_EVERY == 0]
    episodes = np.array([e["episode"] for e in evals])
    return run_auc(episodes, np.array([e["score_final"] for e in evals]))


def auc_at_paper_episodes(arm: str, shift: str) -> float:
    return float(
        np.mean([seed_auc_at_paper_episodes(read_records(arm, shift, s)) for s in range(3)])
    )


def build_seed_ranges() -> list[dict]:
    """Per-seed spread for the arms whose raw logs were kept."""
    arms = (
        "persistent_replay_x4",
        "persistent_replay_x8",
        "persistent_replay_x16",
        "action_calibration",
    )
    rows = []
    for shift in SHIFTS:
        for arm in arms:
            aucs, paper_aucs, backs, starts = [], [], [], []
            for seed in range(3):
                records = read_records(arm, shift, seed)
                evals = [r for r in records if r.get("kind") == "eval"]
                episodes = np.array([e["episode"] for e in evals])
                aucs.append(run_auc(episodes, np.array([e["score_final"] for e in evals])))
                paper_aucs.append(seed_auc_at_paper_episodes(records))
                starts.append(evals[0]["score_final"])
                backs.append(
                    next(
                        r["score_final"]
                        for r in records
                        if r.get("kind") == "reversal" and r["episode"] == 0
                    )
                )
            rows.append(
                {
                    "shift": shift,
                    "arm": arm,
                    "auc_seed0": f"{aucs[0]:.3f}",
                    "auc_seed1": f"{aucs[1]:.3f}",
                    "auc_seed2": f"{aucs[2]:.3f}",
                    "back_0_seed0": f"{backs[0]:.3f}",
                    "back_0_seed1": f"{backs[1]:.3f}",
                    "back_0_seed2": f"{backs[2]:.3f}",
                    "start_score_seed0": f"{starts[0]:.4f}",
                    "start_score_seed1": f"{starts[1]:.4f}",
                    "start_score_seed2": f"{starts[2]:.4f}",
                    "auc_paper_episodes_seed0": f"{paper_aucs[0]:.3f}",
                    "auc_paper_episodes_seed1": f"{paper_aucs[1]:.3f}",
                    "auc_paper_episodes_seed2": f"{paper_aucs[2]:.3f}",
                }
            )
    return rows


def to_csv(rows: list[dict], columns) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    no_shift, rows = load_rows()
    if no_shift < MIN_NO_SHIFT_SCORE:
        raise SystemExit(f"The no-shift score {no_shift} is below {MIN_NO_SHIFT_SCORE}")
    verdicts = build_verdicts(no_shift, rows)
    seed_ranges = build_seed_ranges()
    outputs = {
        RESULTS / "per_seed.csv": to_csv(seed_ranges, list(seed_ranges[0])),
        RESULTS / "summary.csv": to_csv(rows, COLUMNS),
        RESULTS / "verdicts.csv": to_csv(verdicts, list(verdicts[0])),
    }
    for path, text in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f"{path} is out of date; run scripts/summarize_results.py")
        else:
            path.write_text(text, newline="\n")
    print(f"{len(rows)} rows and {len(verdicts)} verdicts {'checked' if args.check else 'written'}")


if __name__ == "__main__":
    main()
