#!/usr/bin/env bash
# Run only the five further ideas, next to a finished run_calibration.sh comparison.
# Usage: bash run_ideas.sh [workdir] [runs at once] [GPUs]
# No earlier arm is rerun: their saved results are the comparison, on the same reward head.
# Runs at once and GPUs default to "auto", which fits the machine.
# It is safe to run again after an interruption: finished runs are skipped.
set -euo pipefail

cd "$(dirname "$0")"
WORKDIR="$(realpath -m "${1:-$HOME/jepa_ttt_work}")"
JOBS="${2:-auto}"
GPUS="${3:-auto}"
source scripts/common.sh
# On a separate machine, set JEPA_TTT_STANDALONE=1 and restore the first run's shared files.
# The earlier arms' results are then merged in afterward to compare them.
STANDALONE="${JEPA_TTT_STANDALONE:-0}"

if pgrep -f "run_calibration.sh" >/dev/null; then
  echo "run_calibration.sh is still running: wait for it to finish, so this isn't sharing its GPUs."
  exit 1
fi
for needed in artifacts/reward_head.pt artifacts/prepare.json artifacts/weights.pt artifacts/config.json \
  results_calibration/frozen__none__seed0.jsonl results_calibration/frozen__grid__seed0.jsonl \
  results_calibration/frozen__wave__seed0.jsonl \
  results_calibration/trajectories__grid__uniform_left__n100.pt \
  results_calibration/trajectories__wave__uniform_left__n100.pt; do
  if [ ! -f "$WORKDIR/$needed" ]; then
    echo "Missing $WORKDIR/$needed: let run_calibration.sh finish or restore its saved files first."
    exit 1
  fi
done
[ "$STANDALONE" = 1 ] && echo "Standalone: not waiting for the earlier arms." || python - "$WORKDIR/results_calibration" <<'PY'
import json, sys
from pathlib import Path

results = Path(sys.argv[1])
names = [f"frozen__{shift}__seed0" for shift in ("none", "grid", "wave")]
for arm in ("persistent_replay_x4", "persistent_replay_x8", "persistent_replay_x16", "action_calibration"):
    names += [f"{arm}__{shift}__seed{seed}" for shift in ("grid", "wave") for seed in (0, 1, 2)]
unfinished = []
for name in names:
    path = results / f"{name}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    if not lines or json.loads(lines[-1]).get("kind") != "done":
        unfinished.append(name)
if unfinished:
    sys.exit(f"These earlier runs aren't finished, so nothing could be compared: {unfinished}")
print("All", len(names), "earlier runs are finished.")
PY
run_preflight

ARMS="calibration_before_now calibration_fast latent_correction rollout_loss_x4 anchored_x4"

echo "=== 1/4 Logic tests"
run_tests tests/test_ideas.py tests/test_calibration.py

echo "=== 2/4 Timing smoke test with the real planner settings"
with_gpu_sampling "$WORKDIR/gpu_ideas_smoke.csv" \
  python -m jepa_ttt.sweep --workdir "$WORKDIR" --tag ideas_smoke --jobs "$JOBS" --gpus "$GPUS" \
  --arms $ARMS --shifts grid wave --seeds 0 \
  --episodes 5 --eval-every 5 --trajectories 10 \
  --reversal-episodes 1 --skip-baseline-gate
print_timings "$WORKDIR/results_ideas_smoke"

echo "=== 3/4 The five ideas, with the same settings as run_calibration.sh"
python -m jepa_ttt.sweep --workdir "$WORKDIR" --tag calibration --jobs "$JOBS" --gpus "$GPUS" \
  --arms $ARMS --shifts grid wave \
  --eval-dense-until 200 --eval-dense-every 10 --reversal-episodes 50

echo "=== 4/4 Results, compared with the saved earlier arms"
python -m jepa_ttt.report_calibration --results "$WORKDIR/results_calibration" \
  | tee "$WORKDIR/report_ideas.txt"
