#!/usr/bin/env bash
# Reproduce the paper's arms and the 4x-update extension on one Two-Room shift at a time.
# Usage: bash run_stage1.sh [workdir] [runs at once] [GPUs]
# Runs at once and GPUs default to "auto", which fits the machine.
# It is safe to run again after an interruption: finished runs are skipped.
set -euo pipefail

cd "$(dirname "$0")"
WORKDIR="$(realpath -m "${1:-$HOME/jepa_ttt_work}")"
JOBS="${2:-auto}"
GPUS="${3:-auto}"
source scripts/common.sh
run_preflight

echo "=== 1/5 Download artifacts, validate data, fit the reward head"
if [ -f "$WORKDIR/artifacts/prepare.json" ]; then
  echo "already prepared"
else
  python -m jepa_ttt.prepare --workdir "$WORKDIR"
fi

echo "=== 2/5 Logic tests"
run_tests

echo "=== 3/5 Timing smoke test with the real planner settings"
with_gpu_sampling "$WORKDIR/gpu_smoke.csv" \
  python -m jepa_ttt.sweep --workdir "$WORKDIR" --tag smoke --jobs "$JOBS" --gpus "$GPUS" \
  --shifts grid wave --seeds 0 \
  --episodes 5 --eval-every 5 --eval-episodes 10 --eval-batch 10 --trajectories 10 \
  --skip-baseline-gate
print_timings "$WORKDIR/results_smoke"

echo "=== 4/5 Full comparison"
python -m jepa_ttt.sweep --workdir "$WORKDIR" --jobs "$JOBS" --gpus "$GPUS" --shifts grid wave

echo "=== 5/5 Results"
for shift in grid wave; do
  python -m jepa_ttt.report --results "$WORKDIR/results" --shift "$shift" \
    | tee "$WORKDIR/report_$shift.txt"
done
