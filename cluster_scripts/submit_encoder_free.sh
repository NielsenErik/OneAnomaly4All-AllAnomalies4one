#!/bin/bash
# Submit every Direction-2 (encoder-free / raw-PC) config to SLURM: the ef_*.yaml
# matrix (raw tabular density, monotone+SOS, vtree sweep, multi-source routed
# mixture, and the ProbMoE-router F1 detection kill-shot). Thin wrapper over
# submit_all.sh. These are all CPU-bound (no image backbone).
#
# Run from the repository root on the login node:
#   bash cluster_scripts/submit_encoder_free.sh
set -euo pipefail
cd "$(dirname "$0")/.."

shopt -s nullglob
CONFIGS=(config/ef_*.yaml)
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No config/ef_*.yaml found" >&2
    exit 1
fi

echo "Encoder-free (Direction 2) configs: ${CONFIGS[*]}"
bash cluster_scripts/submit_all.sh "${CONFIGS[@]}"
