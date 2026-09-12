#!/bin/bash
# Remaining diagnosis studies on a CUDA workstation (roadmap steps 5, 7, 8).
#
#   bash local_scripts/run_diagnosis_ws.sh              # 3 studies in parallel
#   bash local_scripts/run_diagnosis_ws.sh --sequential # one at a time
#   DEVICE=cuda:1 bash local_scripts/run_diagnosis_ws.sh
#
# Order, and why:
#   0. preflight   CUDA visible, C-MAPSS present, the diagnosis tests pass on
#                  THIS machine, and a five-arm smoke run completes on the GPU.
#                  Any failure stops here: six FD001 runs were once lost to a
#                  device bug that only fired after twelve minutes of training.
#   1. studies     the FD001 reliability study (15 runs: the candidate as it
#                  stands, +restarts, +restarts and cosine decay) and, unless
#                  RUN_LEGACY=1 adds them back, nothing else.  Parallel by
#                  default -- their accuracy numbers do not depend on timing.
#   1b. gate       the pre-registered reliability gate.  The FD003 pilot runs
#                  ONLY if it passes: piloting the variance of a fit that is
#                  decided by its initialisation measures the initialisation,
#                  and then sizes the confirmation from it.
#   2. benchmark   run ALONE, so no other process contends for the machine.
#                  It is the latency source of truth; the per-query seconds
#                  inside the parallel studies are contaminated by concurrency
#                  and are not to be quoted as costs.
#   3. reports     one report per study root.
#   4. bundle      logs/ws_results_<date>.tgz, to copy back to the laptop.
#
# Every output goes under logs/ts/ws/, so nothing here can collide with, or be
# silently resumed from, a run made on another machine or device.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export PYTHONPATH=.

PY=python3
DEVICE="${DEVICE:-cuda}"
ROOT=logs/ts/ws
OUT=logs/ws_diagnosis
mkdir -p "${ROOT}" "${OUT}"
PARALLEL=1
[ "${1:-}" = "--sequential" ] && PARALLEL=0

stamp() { date +%T; }

# ── 0. preflight ────────────────────────────────────────────────────────────
DEVICE="${DEVICE}" ${PY} - <<'EOF' || exit 1
import os, sys, torch
dev = os.environ["DEVICE"]
if dev.startswith("cuda") and not torch.cuda.is_available():
    sys.exit("[preflight] torch sees no CUDA device; check the driver / env")
if dev.startswith("cuda"):
    i = torch.device(dev).index or 0
    print(f"[preflight] {torch.cuda.get_device_name(i)} · torch {torch.__version__} "
          f"· CUDA {torch.version.cuda}")
EOF
if [ ! -f data/cmapss/train_FD001.txt ]; then
    echo "[preflight] data/cmapss/ is missing (it is gitignored); copy it from the laptop" >&2
    exit 1
fi

echo "[preflight] diagnosis test suite $(stamp)"
${PY} -m pytest -q -x \
    tests/test_diagnosis_study.py tests/test_diagnosis_baselines.py \
    tests/test_diagnosis_detectors.py tests/test_diagnosis_donors.py \
    tests/test_diagnosis_missingness.py tests/test_confirm_diagnosis.py \
    tests/test_report_diagnosis.py tests/test_tier1_relational.py \
    tests/test_training_restarts.py tests/test_gate_reliability.py \
    > "${OUT}/preflight_tests.log" 2>&1 \
    || { echo "[preflight] tests FAILED — see ${OUT}/preflight_tests.log" >&2; exit 1; }
tail -1 "${OUT}/preflight_tests.log"

echo "[preflight] GPU smoke, five arms + restarts, decay and the new split $(stamp)"
${PY} -m poc.time_series.runner config/ts/diagnosis_smoke.yaml --device "${DEVICE}" \
    --stop-on-error --force --log-root "${ROOT}/diagnosis_smoke" \
    --set model.restarts=2 --set model.lr_schedule=cosine \
    --set 'eval.diagnosis_split_weights=[1,2,1]' \
    > "${OUT}/preflight_smoke.log" 2>&1 \
    || { echo "[preflight] smoke FAILED — see ${OUT}/preflight_smoke.log" >&2; exit 1; }
echo "[preflight] ok $(stamp)"

# ── 1. studies ──────────────────────────────────────────────────────────────
# The FD001 architecture/donor/candidate studies are RECORDED (logs/ts/ws at
# commit 8e56173); re-running them buys nothing until the fit is reliable.
# RUN_LEGACY=1 puts them back for a clean-machine reproduction.
STUDIES=(diagnosis_fd001_reliability)
[ "${RUN_LEGACY:-0}" = "1" ] && STUDIES+=(diagnosis_fd001 diagnosis_donors diagnosis_fd001_candidate)
study() {
    local name=$1
    echo "[study] start ${name} $(stamp)"
    ${PY} -m poc.time_series.runner "config/ts/${name}.yaml" --device "${DEVICE}" \
        --log-root "${ROOT}/${name}" > "${OUT}/${name}.log" 2>&1
    local rc=$?
    echo "[study] done  ${name} exit=${rc} $(stamp)"
    return ${rc}
}

fail=0
if [ "${PARALLEL}" -eq 1 ]; then
    # Split the CPU between the three processes rather than oversubscribing it:
    # the detectors and fitted comparators are CPU-side even with the circuit
    # on the GPU.
    NCPU="$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 6 )"
    export OMP_NUM_THREADS=$(( NCPU / ${#STUDIES[@]} > 1 ? NCPU / ${#STUDIES[@]} : 1 ))
    export MKL_NUM_THREADS=${OMP_NUM_THREADS}
    echo "[study] ${#STUDIES[@]} in parallel, ${OMP_NUM_THREADS} CPU threads each"
    pids=()
    for s in "${STUDIES[@]}"; do study "${s}" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
else
    for s in "${STUDIES[@]}"; do study "${s}" || fail=1; done
fi

# ── 1b. reliability gate, then the FD003 pilot if it passes ─────────────────
GATE_JSON="${OUT}/reliability_gate.json"
echo "[gate] reliability $(stamp)"
${PY} -m poc.time_series.gate_reliability "${ROOT}/diagnosis_fd001_reliability" \
    --json "${GATE_JSON}" | tee "${OUT}/reliability_gate.log"
gate_rc=${PIPESTATUS[0]}
if [ "${gate_rc}" -eq 0 ]; then
    echo "[study] start diagnosis_fd003_pilot $(stamp)"
    ${PY} -m poc.time_series.runner config/ts/diagnosis_fd003_pilot.yaml \
        --device "${DEVICE}" --log-root "${ROOT}/diagnosis_fd003_pilot" \
        > "${OUT}/diagnosis_fd003_pilot.log" 2>&1 \
        || { echo "[study] diagnosis_fd003_pilot FAILED" >&2; fail=1; }
    STUDIES+=(diagnosis_fd003_pilot)
    echo "[study] done  diagnosis_fd003_pilot $(stamp)"
else
    # Not a failure of the run: a failure of the gate, which is a RESULT.
    echo "[gate] FAILED — the FD003 pilot is skipped by protocol; see ${GATE_JSON}"
fi

# ── 2. benchmark, alone ─────────────────────────────────────────────────────
unset OMP_NUM_THREADS MKL_NUM_THREADS
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" >/dev/null
echo "[bench] start $(stamp) — CPU, circuit and comparators on one device"
${PY} -m poc.time_series.bench_relational --methods \
    --out "${ROOT}/bench_relational_methods.json" \
    --method-channels 8 12 16 --method-windows 4 8 12 \
    --method-batches 64 256 512 --method-masks 1 4 16 \
    > "${OUT}/bench.log" 2>&1 || fail=1
echo "[bench] done $(stamp)"

# ── 3. reports ──────────────────────────────────────────────────────────────
for s in "${STUDIES[@]}"; do
    echo "[report] ${s}"
    ${PY} -m poc.time_series.report_diagnosis "${ROOT}/${s}" --quiet --reps 1000 \
        --score conditional_max --score relational_max \
        > "${OUT}/report_${s}.log" 2>&1 || { echo "[report] ${s} failed" >&2; fail=1; }
done

# ── 4. bundle ───────────────────────────────────────────────────────────────
BUNDLE="logs/ws_results_$(date +%Y%m%d_%H%M).tgz"
tar czf "${BUNDLE}" "${ROOT}" "${OUT}"
echo "[bundle] ${BUNDLE} ($(du -h "${BUNDLE}" | cut -f1)) — copy this back to the laptop"

if [ "${fail}" -ne 0 ]; then
    echo "[ws] finished WITH FAILURES — check ${OUT}/*.log" >&2
    exit 1
fi
echo "[ws] all done $(stamp)"
