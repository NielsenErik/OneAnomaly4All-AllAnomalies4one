#!/bin/bash
# Queue the diagnosis programme on the workstation with task spooler (tsp).
#
#   bash local_scripts/queue_diagnosis_ws_tsp.sh              # the whole chain
#   bash local_scripts/queue_diagnosis_ws_tsp.sh --tests-only # just the suites
#   DEVICE=cuda:1 bash local_scripts/queue_diagnosis_ws_tsp.sh
#
# Why a queue rather than `run_diagnosis_ws.sh` directly: the jobs survive a
# closed SSH session, each stage keeps its own exit status, and the ONE stage
# that must not share the machine (the benchmark, which is the latency source
# of truth) is guaranteed to run alone because the queue holds a single slot.
#
# The chain, and what it depends on:
#
#   0 tests      the diagnosis suites incl. the new restart/gate ones
#   1 smoke      five arms on the GPU, with restarts, decay and the new split
#   2 reliability  config/ts/diagnosis_fd001_reliability.yaml   (15 runs)
#   3 gate       the PRE-REGISTERED reliability gate; exit 1 means FAIL
#   4 pilot      config/ts/diagnosis_fd003_pilot.yaml (5 runs) — queued with
#                -W on the gate, so a failed gate SKIPS it.  That is the
#                protocol, not a convenience: piloting the variance of a fit
#                that is decided by its initialisation measures the
#                initialisation and then sizes the confirmation from it.
#   5 bench      alone, one slot, nothing else running
#   6 reports    one per study root that exists
#   7 bundle     logs/ws_results_<date>.tgz
#
# Each stage runs only if the previous one finished WELL (-W), except the
# reports and the bundle, which run after whatever happened (-D) so a failed
# stage still leaves a readable record.
#
#   TS_SOCKET=/tmp/ts_diagnosis_ws tsp          # the queue
#   TS_SOCKET=/tmp/ts_diagnosis_ws tsp -c <id>  # one job's output
#   TS_SOCKET=/tmp/ts_diagnosis_ws tsp -k <id>  # kill a running job
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TS="${TS:-$(command -v tsp || command -v ts)}"
if [ -z "${TS}" ]; then
    echo "task spooler not found: apt install task-spooler (binary tsp or ts)" >&2
    exit 1
fi
export TS_SOCKET="${TS_SOCKET:-/tmp/ts_diagnosis_ws}"
PY="${PY:-python3}"
DEVICE="${DEVICE:-cuda}"
ROOT=logs/ts/ws
OUT=logs/ws_diagnosis
mkdir -p "${ROOT}" "${OUT}"

# One slot: the stages are sequential by dependency anyway, and the benchmark
# must not share the machine with a training run.
"${TS}" -S 1 >/dev/null 2>&1

# `-W <id>` (run only if that job succeeded) and `-D <id>` (run once it ends,
# either way) are what make this a chain.  Older builds have neither; there,
# fall back to `-d`, which is "after the previous job", and say so — the
# difference is that a failed stage no longer skips the stages behind it.
CHAIN=1
"${TS}" -h 2>&1 | grep -q -- "-W" || CHAIN=0
[ "${CHAIN}" -eq 0 ] && echo "[queue] this tsp has no -W/-D: chaining with -d, " \
    "so a FAILED stage will NOT skip the ones after it — watch the queue"

# after_ok <id> / after_any <id> -> the flags for a job that follows it
after_ok()  { [ "${CHAIN}" -eq 1 ] && echo "-W $1" || echo "-d"; }
after_any() { [ "${CHAIN}" -eq 1 ] && echo "-D $1" || echo "-d"; }

queue() {   # queue <label> <dep-flags> <command…>
    local label=$1; shift
    local dep=$1; shift
    # shellcheck disable=SC2086
    "${TS}" -L "${label}" ${dep} bash -c "$*"
}

export PYTHONPATH=.

TESTS="${PY} -m pytest -q -x \
    tests/test_diagnosis_study.py tests/test_diagnosis_baselines.py \
    tests/test_diagnosis_detectors.py tests/test_diagnosis_donors.py \
    tests/test_diagnosis_missingness.py tests/test_confirm_diagnosis.py \
    tests/test_report_diagnosis.py tests/test_tier1_relational.py \
    tests/test_training_restarts.py tests/test_gate_reliability.py \
    > ${OUT}/preflight_tests.log 2>&1"
id_tests=$(queue tests "" "${TESTS}")
echo "[queue] ${id_tests}  tests"

if [ "${1:-}" = "--tests-only" ]; then
    echo "[queue] tests only; watch with: TS_SOCKET=${TS_SOCKET} ${TS}"
    exit 0
fi

id_smoke=$(queue smoke "$(after_ok "${id_tests}")" \
    "${PY} -m poc.time_series.runner config/ts/diagnosis_smoke.yaml \
        --device ${DEVICE} --stop-on-error --force \
        --log-root ${ROOT}/diagnosis_smoke \
        --set model.restarts=2 --set model.lr_schedule=cosine \
        --set 'eval.diagnosis_split_weights=[1,2,1]' \
        > ${OUT}/preflight_smoke.log 2>&1")
echo "[queue] ${id_smoke}  smoke"

id_rel=$(queue reliability "$(after_ok "${id_smoke}")" \
    "${PY} -m poc.time_series.runner config/ts/diagnosis_fd001_reliability.yaml \
        --device ${DEVICE} --log-root ${ROOT}/diagnosis_fd001_reliability \
        > ${OUT}/diagnosis_fd001_reliability.log 2>&1")
echo "[queue] ${id_rel}  reliability study (15 runs)"

id_gate=$(queue gate "$(after_ok "${id_rel}")" \
    "${PY} -m poc.time_series.gate_reliability ${ROOT}/diagnosis_fd001_reliability \
        --json ${OUT}/reliability_gate.json | tee ${OUT}/reliability_gate.log; \
     exit \${PIPESTATUS[0]}")
echo "[queue] ${id_gate}  reliability gate (exit 1 = FAIL, skips the pilot)"

id_pilot=$(queue fd003_pilot "$(after_ok "${id_gate}")" \
    "${PY} -m poc.time_series.runner config/ts/diagnosis_fd003_pilot.yaml \
        --device ${DEVICE} --log-root ${ROOT}/diagnosis_fd003_pilot \
        > ${OUT}/diagnosis_fd003_pilot.log 2>&1")
echo "[queue] ${id_pilot}  FD003 pilot (runs only if the gate passes)"

id_bench=$(queue bench "$(after_any "${id_pilot}")" \
    "${PY} -m poc.time_series.bench_relational --methods \
        --out ${ROOT}/bench_relational_methods.json \
        --method-channels 8 12 16 --method-windows 4 8 12 \
        --method-batches 64 256 512 --method-masks 1 4 16 \
        > ${OUT}/bench.log 2>&1")
echo "[queue] ${id_bench}  benchmark (alone on the machine)"

id_report=$(queue reports "$(after_any "${id_bench}")" \
    "for s in diagnosis_fd001_reliability diagnosis_fd003_pilot; do \
        [ -d ${ROOT}/\$s ] || continue; \
        ${PY} -m poc.time_series.report_diagnosis ${ROOT}/\$s --quiet --reps 1000 \
            --score conditional_max --score relational_max \
            > ${OUT}/report_\$s.log 2>&1 || echo \"report \$s failed\" >&2; \
     done")
echo "[queue] ${id_report}  reports"

id_bundle=$(queue bundle "$(after_any "${id_report}")" \
    "tar czf logs/ws_results_\$(date +%Y%m%d_%H%M).tgz ${ROOT} ${OUT} && \
     ls -lh logs/ws_results_*.tgz | tail -1")
echo "[queue] ${id_bundle}  bundle"

echo
echo "queued on TS_SOCKET=${TS_SOCKET}"
"${TS}"
echo
echo "watch:   TS_SOCKET=${TS_SOCKET} ${TS}"
echo "output:  TS_SOCKET=${TS_SOCKET} ${TS} -c <id>     (or -t <id> to tail)"
echo "gate:    cat ${OUT}/reliability_gate.log"
