#!/usr/bin/env bash
# Compare action calibration with persistent fine-tuning on the Two-Room shifts.
# Usage: bash run_calibration.sh [workdir] [runs at once] [GPUs]
# Runs at once and GPUs default to "auto", which fits the machine.
# It is safe to run again after an interruption: finished runs are skipped.
set -euo pipefail

cd "$(dirname "$0")"
WORKDIR="$(realpath -m "${1:-$HOME/jepa_ttt_work}")"
JOBS="${2:-auto}"
GPUS="${3:-auto}"
source scripts/common.sh
run_preflight

ARMS="persistent_replay_x4 persistent_replay_x8 persistent_replay_x16 action_calibration"

echo "=== 1/5 Download artifacts, validate data, fit the reward head"
if [ -f "$WORKDIR/artifacts/prepare.json" ]; then
  echo "already prepared"
else
  python -m jepa_ttt.prepare --workdir "$WORKDIR"
fi

echo "=== 2/5 Logic tests"
run_tests

echo "=== 3/5 Timing smoke test with the real planner settings"
with_gpu_sampling "$WORKDIR/gpu_calibration_smoke.csv" \
  python -m jepa_ttt.sweep --workdir "$WORKDIR" --tag calibration_smoke --jobs "$JOBS" --gpus "$GPUS" \
  --arms $ARMS --shifts grid wave --seeds 0 \
  --episodes 5 --eval-every 5 --trajectories 10 \
  --reversal-episodes 1 --skip-baseline-gate
print_timings "$WORKDIR/results_calibration_smoke"

echo "=== 4/5 Full comparison"
python -m jepa_ttt.sweep --workdir "$WORKDIR" --tag calibration --jobs "$JOBS" --gpus "$GPUS" \
  --arms $ARMS --shifts grid wave \
  --eval-dense-until 200 --eval-dense-every 10 --reversal-episodes 50

echo "=== 5/5 Results"
python -m jepa_ttt.report_calibration --results "$WORKDIR/results_calibration" \
  | tee "$WORKDIR/report_calibration.txt"
