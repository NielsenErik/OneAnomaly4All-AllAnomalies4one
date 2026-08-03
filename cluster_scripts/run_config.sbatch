#!/bin/bash
# Run ONE experiment config (all its seeds sequentially) on a single node.
#
# Usage:
#   sbatch cluster_scripts/run_config.sbatch config/adbench_demo.yaml
#
#SBATCH -p long
#SBATCH --job-name=oafa_config
#SBATCH --output=logs/slurm_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=35G
#SBATCH --time=48:00:00
# The PC experiments are CPU-bound; request a GPU only for configs with
# image datasets (pretrained backbone featurization):
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

CONFIG=${1:?usage: sbatch run_config.sbatch <config.yaml>}

echo "Avvio job su nodo: $(hostname)"
echo "Config: ${CONFIG}"
mkdir -p logs

python3 -m src.experiment "${CONFIG}"
echo "Config Job Complete"
