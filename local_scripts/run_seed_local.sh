#!/bin/bash
# Run ONE seed of ONE config locally — the workstation analogue of a single
# SLURM array task (run_config_array.sbatch). Writes logs/<seed>/...; the
# cross-seed summary is produced separately (run_config_local.sh aggregates,
# or call:  python -m src.experiment <config> --aggregate-only).
#
# Usage:
#   bash local_scripts/run_seed_local.sh config/adbench_demo.yaml 0
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG=${1:?usage: run_seed_local.sh <config.yaml> <seed>}
SEED=${2:?usage: run_seed_local.sh <config.yaml> <seed>}

echo "[seed] ${CONFIG} seed ${SEED} on $(hostname)"
python3 -m src.experiment "${CONFIG}" --seed "${SEED}"
echo "[seed] ${CONFIG} seed ${SEED} complete"
