#!/bin/bash
# Seed-parallel execution: one SLURM array task = one seed of one config.
# Each task writes logs/<seed>/...; aggregate afterwards with
# aggregate.sbatch (or: python -m src.experiment <config> --aggregate-only).
#
# Usage (array indices = the seeds to run, must match the config's `seeds`):
#   sbatch --array=0-4 cluster_scripts/run_config_array.sbatch config/adbench_demo.yaml
#
#SBATCH -p long
#SBATCH --job-name=oafa_seed
#SBATCH --output=logs/slurm_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=35G
#SBATCH --time=48:00:00
# CPU-bound; uncomment for configs with image datasets:
##SBATCH --gres=gpu:1

module load CUDA/12.5.0
source /home/erik.nielsen/.bashrc
conda activate expllm_env

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found. HF_TOKEN might not be loaded."
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG=${1:?usage: sbatch --array=0-N run_config_array.sbatch <config.yaml>}

echo "Avvio job su nodo: $(hostname) — seed ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"
mkdir -p logs

python3 -m src.experiment "${CONFIG}" --seed "${SLURM_ARRAY_TASK_ID}"
echo "Seed ${SLURM_ARRAY_TASK_ID} Job Complete"
