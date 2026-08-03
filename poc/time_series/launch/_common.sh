#!/usr/bin/env bash
# Shared setup for every time-series launcher.  Sourced, not executed.
#
# Environment knobs (all optional, all overridable on the command line):
#   PY            python interpreter          (default: the project conda env)
#   DEVICE        auto | cpu | cuda | cuda:0  (default: auto)
#   SEEDS         "0 1 2"                     (default: whatever the config says)
#   JOBS          parallel config slots       (default: 1)
#   THREADS       torch/OMP threads per job   (default: 8 for JOBS=1, else 4)
#   OUT           log root                    (default: logs/ts)
#   EXTRA         extra args passed to runner.py

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH:-.}"

# ── interpreter ────────────────────────────────────────────────────────────
if [ -z "${PY:-}" ]; then
  for cand in "$HOME/miniconda3/envs/expllm_env/bin/python" \
              "$HOME/anaconda3/envs/expllm_env/bin/python" \
              "$(command -v python3)"; do
    [ -x "$cand" ] && PY="$cand" && break
  done
fi
[ -n "${PY:-}" ] || { echo "no python interpreter found; set PY=..."; exit 1; }

DEVICE="${DEVICE:-auto}"
JOBS="${JOBS:-1}"
if [ "$JOBS" -gt 1 ]; then THREADS="${THREADS:-4}"; else THREADS="${THREADS:-8}"; fi
OUT="${OUT:-logs/ts}"
EXTRA="${EXTRA:-}"

# A probabilistic circuit is thousands of SMALL tensor ops, so BLAS/OMP
# oversubscription costs more than it buys — capping threads per process and
# running several configs in parallel is strictly better than one process
# fighting itself across all cores.
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE_DIR="$OUT/_console"
mkdir -p "$CONSOLE_DIR"

banner() {
  echo
  echo "=============================================================================="
  echo " $*"
  echo "=============================================================================="
}

hostinfo() {
  echo "host      : $(hostname)"
  echo "python    : $PY"
  echo "device    : $DEVICE   jobs: $JOBS   threads/job: $THREADS"
  echo "log root  : $OUT"
  echo "started   : $(date)"
  "$PY" - <<'PYEOF'
import torch
print(f"torch     : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]    : {p.name}  {p.total_memory / 2**30:.1f} GB")
PYEOF
}

# Queue of `runner.py` invocations, executed by `flush_queue` with JOBS slots.
QUEUE_FILE="$(mktemp -t ts_queue.XXXXXX)"
trap 'rm -f "$QUEUE_FILE"' EXIT

queue_config() {                     # queue_config <config.yaml> [extra args...]
  local cfg="$1"; shift
  local name
  name="$(basename "${cfg%.yaml}")"
  local seeds_arg=""
  [ -n "${SEEDS:-}" ] && seeds_arg="--seeds $SEEDS"
  # NUL-terminated, and read back with `xargs -0`.  Without -0, xargs applies
  # its OWN quote processing to the queue file and strips the double quotes
  # around the trailing echo, after which bash sees a bare `(` and every
  # queued config dies with "syntax error near unexpected token `('".
  printf '%s\0' "$PY -m poc.time_series.runner $cfg --device $DEVICE --log-root $OUT/$name $seeds_arg $EXTRA $* > $CONSOLE_DIR/${name}_${STAMP}.log 2>&1; echo \"[\$(date +%H:%M:%S)] done $name (exit \$?)\"" >> "$QUEUE_FILE"
}

flush_queue() {
  local n
  n="$(tr -cd '\0' < "$QUEUE_FILE" | wc -c | tr -d ' ')"
  [ "$n" -eq 0 ] && { echo "(nothing queued)"; return 0; }
  echo "running $n config(s), $JOBS at a time; per-config console logs in $CONSOLE_DIR/"
  # A failing config must not take the batch down: each line ends with `echo`,
  # so xargs always sees success and continues.  Failures are visible in the
  # per-config log and in each run's status.json.
  xargs -0 -P "$JOBS" -I{} bash -c '{}' < "$QUEUE_FILE"
  : > "$QUEUE_FILE"
}
