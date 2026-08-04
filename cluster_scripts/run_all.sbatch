#!/bin/bash
# Run EVERY experimental setup (all configs, all seeds, all internal sweeps —
# adapt modes, vtree methods, baselines) sequentially inside ONE SLURM job and
# produce every metric summary.  Use this when you want the complete metric
# table from a single allocation instead of the seed-parallel array fan-out of
# submit_all.sh.
#
# Usage:
#   sbatch cluster_scripts/run_all.sbatch                       # every config/*.yaml
#   sbatch cluster_scripts/run_all.sbatch config/adbench_demo.yaml config/sos_ablation.yaml
#
# A GPU is requested by default because multimodal_demo featurizes images; it
# sits idle for the CPU-bound tabular/text configs but keeps one script able to
# run the whole matrix. Drop --gres below if you only run tabular/text configs.
#
#SBATCH -p long
#SBATCH --job-name=oafa_run_all
#SBATCH --output=logs/slurm_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=35G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

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

# configs to run: all of config/*.yaml unless an explicit subset is passed
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(config/*.yaml)
fi

echo "Avvio job su nodo: $(hostname)"
echo "Configs: ${CONFIGS[*]}"
mkdir -p logs

# one invocation per config: each does its full seed loop + cross-seed
# aggregation (logs/summary_<name>.json). A failure in one config must not
# abort the rest of the matrix.
status=0
for CONFIG in "${CONFIGS[@]}"; do
    echo "===== ${CONFIG} ====="
    if ! python3 -m src.experiment "${CONFIG}"; then
        echo "!!! ${CONFIG} FAILED — continuing with the rest"
        status=1
    fi
done

echo "All configs complete (exit status ${status})"
exit ${status}
