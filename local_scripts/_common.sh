#!/bin/bash
# Shared setup for the local (personal-workstation) runners. Sourced, not run.
# No SLURM, no module system — just put us in the repo root, activate a Python
# environment if one is configured/available, and load .env for HF_TOKEN.
#
# Override the conda env name with:  OAFA_CONDA_ENV=myenv bash local_scripts/...
# If conda / the env is absent we fall back to whatever `python3` is on PATH
# (e.g. an already-activated venv), so the scripts work on any workstation.

# repo root = parent of this script's directory, regardless of CWD
OAFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${OAFA_ROOT}"
mkdir -p logs

OAFA_CONDA_ENV="${OAFA_CONDA_ENV:-expllm_env}"

if command -v conda >/dev/null 2>&1; then
    # `conda activate` needs the shell hook in non-interactive shells
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda activate "${OAFA_CONDA_ENV}" 2>/dev/null; then
        echo "[env] conda environment '${OAFA_CONDA_ENV}' active"
    else
        echo "[env] conda present but '${OAFA_CONDA_ENV}' not found — using current python3"
    fi
else
    echo "[env] no conda on PATH — using current python3 ($(command -v python3))"
fi

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "[env] sourced .env"
else
    echo "[env] no .env file (HF_TOKEN may be unset; only image/text loaders need it)"
fi

export TOKENIZERS_PARALLELISM=false
# leave one core for the OS by default; override with OMP_NUM_THREADS=N
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    NCPU="$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 4 )"
    export OMP_NUM_THREADS=$(( NCPU > 1 ? NCPU - 1 : 1 ))
fi

# read the seed list ("0 1 2") from a config file
oafa_seeds() {
    python3 - "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(" ".join(str(s) for s in cfg.get("seeds", [0])))
EOF
}
