#!/bin/bash
# Run ONE experiment config end-to-end on a personal workstation: all its
# seeds, all its internal sweeps (adapt modes, vtree methods, baselines), then
# the cross-seed metric summary (logs/summary_<name>.json).
#
# Two modes:
#   sequential (default) — one process does the whole seed loop and aggregates
#                          itself. Lowest memory, simplest.
#   --parallel [N]       — fan the seeds across up to N background processes
#                          (default: ceil(cores/4), min 1), then aggregate once
#                          all seeds finish. Faster on a many-core box; uses
#                          ~N times the memory.
#
# Usage:
#   bash local_scripts/run_config_local.sh config/adbench_demo.yaml
#   bash local_scripts/run_config_local.sh config/vtree_ablation.yaml --parallel 3
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG=${1:?usage: run_config_local.sh <config.yaml> [--parallel [N]]}
shift || true

PARALLEL=0
JOBS=0
if [ "${1:-}" = "--parallel" ]; then
    PARALLEL=1
    shift
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then JOBS=$1; shift; fi
fi

NAME=$(basename "${CONFIG}" .yaml)
LOG="logs/local_${NAME}.txt"

if [ "${PARALLEL}" -eq 0 ]; then
    echo "[config] ${CONFIG} — sequential (log: ${LOG})"
    python3 -m src.experiment "${CONFIG}" 2>&1 | tee "${LOG}"
    echo "[config] ${CONFIG} done"
    exit 0
fi

# --- parallel-over-seeds path ---------------------------------------------
read -ra SEEDS <<< "$(oafa_seeds "${CONFIG}")"
if [ "${JOBS}" -le 0 ]; then
    NCPU="$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 4 )"
    JOBS=$(( (NCPU + 3) / 4 )); JOBS=$(( JOBS < 1 ? 1 : JOBS ))
fi
echo "[config] ${CONFIG} — parallel: ${#SEEDS[@]} seeds, up to ${JOBS} at a time (log: ${LOG})"

: > "${LOG}"
fail=0
pids=()   # FIFO of running seed PIDs; portable to bash 3.2 (no `wait -n`)
for SEED in "${SEEDS[@]}"; do
    (
        echo "[seed ${SEED}] start $(date +%T)"
        python3 -m src.experiment "${CONFIG}" --seed "${SEED}"
        echo "[seed ${SEED}] done $(date +%T)"
    ) >> "${LOG}" 2>&1 &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "${JOBS}" ]; then
        wait "${pids[0]}" || fail=1
        pids=("${pids[@]:1}")
    fi
done
for pid in ${pids[@]+"${pids[@]}"}; do
    wait "${pid}" || fail=1
done

echo "[config] ${CONFIG} — all seeds finished, aggregating"
python3 -m src.experiment "${CONFIG}" --aggregate-only 2>&1 | tee -a "${LOG}"

if [ "${fail}" -ne 0 ]; then
    echo "[config] ${CONFIG}: one or more seeds failed — see ${LOG}" >&2
    exit 1
fi
echo "[config] ${CONFIG} done"
