# Results and where they came from

| Path | What it is | Source | Per-seed logs? |
|---|---|---|---|
| `stage1/report_transcribed.txt` | Paper arms and x4, on 4x RTX 3090 | Copied from terminal output | No, the instance was destroyed |
| `calibration/report.txt` | x4, x8, x16 and `action_calibration` | Saved by `run_calibration.sh` | Yes, in `calibration/runs/` |
| `calibration/prepare.json` | Data checks and reward head stats for that run | Saved by `jepa_ttt.prepare` | - |
| `ideas/report.txt` | The five further ideas | Saved by `run_ideas.sh` | No, they weren't downloaded |
| `ideas/run.log` | That run's full terminal log | Captured from the run's output | - |
| `offline/` | Checks of the calibrator on a small random-policy dataset, and on saved weights | Output of `scripts/offline_checks/` | - |
| `summary.csv`, `verdicts.csv` | Tables derived from the two saved reports | `scripts/summarize_results.py` | - |
| `per_seed.csv` | AUC, switch-back and starting scores by seed, for the arms with logs | `scripts/summarize_results.py` | - |
| `figures/` | README figures | `scripts/make_figures.py` | - |

The `ideas` report says "incomplete" because its baselines are in the `calibration` run, on another
machine. `verdicts.csv` compares them using the means in both reports.

Model weights, the dataset and test trajectories are too large for this repository. The run scripts
regenerate them.

**Offline checks:** `retention.txt` ran on the saved weights from the calibration run, while the other
three files ran before the final calibrator design was fixed. `retention.py` needs those weights,
which aren't in this repository, but `run_calibration.sh` writes them again.
