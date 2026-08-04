#!/bin/bash
# Run EVERY experimental setup on a personal workstation: every config under
# config/ (each one sweeps its own seeds, adapt modes, vtree methods and
# baselines internally) and produce every metric summary
# (logs/summary_<name>.json). The workstation analogue of submit_all.sh.
#
# Configs run one after another (each is itself heavy); pass --parallel to
# parallelize the SEEDS within each config — see run_config_local.sh.
#
# Usage:
#   bash local_scripts/run_all_local.sh                      # all config/*.yaml
#   bash local_scripts/run_all_local.sh config/adbench_demo.yaml config/sos_ablation.yaml
#   bash local_scripts/run_all_local.sh --parallel 3         # all, seeds parallel
#   bash local_scripts/run_all_local.sh --parallel 3 config/vtree_ablation.yaml
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "${HERE}/_common.sh"

# split args into the optional "--parallel [N]" prefix and the config list
PASS=()
CONFIGS=()
if [ "${1:-}" = "--parallel" ]; then
    PASS+=(--parallel)
    shift
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then PASS+=("$1"); shift; fi
fi
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(config/*.yaml)
fi

echo "######################################################################"
echo "# Running ${#CONFIGS[@]} config(s): ${CONFIGS[*]}"
[ ${#PASS[@]} -gt 0 ] && echo "# per-config seed mode: ${PASS[*]}"
echo "######################################################################"

status=0
for CONFIG in "${CONFIGS[@]}"; do
    echo
    echo "===== ${CONFIG} ====="
    # a failure in one config must not abort the rest of the matrix
    # ${PASS[@]+...} guard: expanding an empty array under `set -u` errors on
    # macOS' bash 3.2
    if ! bash "${HERE}/run_config_local.sh" "${CONFIG}" ${PASS[@]+"${PASS[@]}"}; then
        echo "!!! ${CONFIG} FAILED — continuing with the rest" >&2
        status=1
    fi
done

echo
echo "All configs complete (exit status ${status}). Summaries: logs/summary_*.json"
exit ${status}
