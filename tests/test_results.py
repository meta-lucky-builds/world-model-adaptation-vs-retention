"""The saved results, the derived tables and the README must agree."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def test_summary_has_every_arm_on_both_shifts():
    summary = rows("summary.csv")
    arms = {r["arm"] for r in summary}
    assert arms == {
        "frozen", "persistent_replay_x4", "persistent_replay_x8", "persistent_replay_x16",
        "action_calibration", "calibration_before_now", "calibration_fast", "latent_correction",
        "rollout_loss_x4", "anchored_x4",
    }  # fmt: skip
    for arm in arms:
        assert {r["shift"] for r in summary if r["arm"] == arm} == {"grid", "wave"}


def test_summary_matches_the_saved_reports():
    from scripts.summarize_results import load_rows

    assert load_rows()[1] == [{**row} for row in rows("summary.csv")], (
        "run scripts/summarize_results.py"
    )


def test_verdicts_match_the_rules():
    verdicts = {(r["shift"], r["arm"]): r["verdict"] for r in rows("verdicts.csv")}
    for shift in ("grid", "wave"):
        assert verdicts[(shift, "calibration_before_now")] == "pass"
        for arm in ("calibration_fast", "latent_correction", "rollout_loss_x4", "anchored_x4"):
            assert verdicts[(shift, arm)] == "fail"


def test_calibration_report_records_the_failure():
    report = (RESULTS / "calibration" / "report.txt").read_text()
    assert "Action calibration did not win on every shift" in report
    assert report.count("PASS  Premise") == 2


def test_raw_logs_reproduce_the_reported_means():
    """Recompute AUC from the per-seed logs that were kept."""
    import json

    import numpy as np

    from jepa_ttt.report import run_auc

    expected = {
        (r["shift"], r["arm"]): float(r["auc"])
        for r in rows("summary.csv")
        if r["source"] == "calibration" and r["arm"] != "frozen"
    }
    assert len(expected) == 8
    for (shift, arm), auc in expected.items():
        aucs = []
        for seed in range(3):
            path = RESULTS / "calibration" / "runs" / f"{arm}__{shift}__seed{seed}.jsonl"
            evals = [json.loads(line) for line in path.read_text().splitlines()]
            evals = [e for e in evals if e.get("kind") == "eval"]
            episodes = np.array([e["episode"] for e in evals])
            aucs.append(run_auc(episodes, np.array([e["score_final"] for e in evals])))
        assert float(np.mean(aucs)) == pytest.approx(auc, abs=0.0006)


def test_readme_tables_use_the_saved_numbers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = re.search(r"<!-- results-table -->(.*?)<!-- /results-table -->", readme, re.S).group(1)
    checked = 0
    summary = {(r["shift"], r["arm"]): r for r in rows("summary.csv")}
    for line in table.splitlines():
        cells = [c.strip(" `*") for c in line.strip().strip("|").split("|")]
        if len(cells) != 9 or cells[0] not in {r["arm"] for r in summary.values()}:
            continue
        arm = cells[0]
        grid, wave = summary[("grid", arm)], summary[("wave", arm)]
        assert cells[1:] == [
            grid["auc"], grid["episodes_to_target"], grid["back_0"], grid["back_50"],
            wave["auc"], wave["episodes_to_target"], wave["back_0"], wave["back_50"],
        ], arm  # fmt: skip
        checked += 1
    assert checked == 10
