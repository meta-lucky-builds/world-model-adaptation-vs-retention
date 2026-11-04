# Shared steps for the run scripts. Source it after setting WORKDIR, JOBS and GPUS.

# Small CPU jobs slow down badly with one thread per host core on a rented machine.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TEST_TIME_LIMIT=1500

run_preflight() {
  local resolved
  resolved="$(python -m jepa_ttt.preflight --workdir "$WORKDIR" --jobs "$JOBS" --gpus "$GPUS")"
  JOBS="${resolved% *}"
  GPUS="${resolved#* }"
  python -m pip install --quiet -r requirements.txt
}

# run_tests <file>...: stop on the first failure, or if tests take far too long.
run_tests() {
  local code=0
  JEPA_TTT_WEIGHTS="$WORKDIR/artifacts/weights.pt" \
  JEPA_TTT_CONFIG="$WORKDIR/artifacts/config.json" \
  timeout "$TEST_TIME_LIMIT" python -m pytest -q -x "$@" || code=$?
  if [ "$code" -eq 124 ]; then
    echo "The tests passed the ${TEST_TIME_LIMIT} s limit. No results are affected, so send this log."
  fi
  return "$code"
}

# with_gpu_sampling <csv file> <command>...: report peak memory and use during a command.
with_gpu_sampling() {
  local csv="$1" sampler code=0
  shift
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 5 > "$csv" 2>/dev/null &
  sampler=$!
  "$@" || code=$?
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  python - "$csv" <<'PY'
import sys
from pathlib import Path

rows = [
    [part.strip() for part in line.split(",")]
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.count(",") == 3
]
for gpu in sorted({row[0] for row in rows if row[0].isdigit()}):
    mine = [row for row in rows if row[0] == gpu and all(part.isdigit() for part in row)]
    if mine:
        print(
            f"GPU {gpu}: peak memory {max(int(r[2]) for r in mine)} of {mine[0][3]} MiB, "
            f"mean use {sum(int(r[1]) for r in mine) / len(mine):.0f}%"
        )
PY
  return "$code"
}

# print_timings <results folder>: how long each short run took.
print_timings() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    done = json.loads(path.read_text().splitlines()[-1])
    print(f"{path.stem}: {done.get('seconds', 0):.0f} s")
PY
}
