#!/bin/bash
# Submit every Direction-1 (encoder / latent-PC) config to SLURM: the enc_*.yaml
# matrix (tabular / text / multimodal × monotone+SOS × vtree sweep). Thin wrapper
# over submit_all.sh, which fans each config out seed-parallel and chains an
# aggregation job. GPU requests are added automatically for image configs.
#
# Run from the repository root on the login node:
#   bash cluster_scripts/submit_encoder.sh
set -euo pipefail
cd "$(dirname "$0")/.."

shopt -s nullglob
CONFIGS=(config/enc_*.yaml)
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No config/enc_*.yaml found" >&2
    exit 1
fi

echo "Encoder (Direction 1) configs: ${CONFIGS[*]}"
bash cluster_scripts/submit_all.sh "${CONFIGS[@]}"
