#!/bin/bash
# Cross-seed aggregation only (no training): reads logs/<seed>/results.jsonl
# written by the array tasks and writes logs/summary_<name>.json.
# Submitted automatically by submit_all.sh with --dependency=afterok:<array>.
#
# Usage:
#   sbatch cluster_scripts/aggregate.sbatch config/adbench_demo.yaml
#
#SBATCH -p long
#SBATCH --job-name=oafa_aggregate
#SBATCH --output=logs/slurm_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

source /home/erik.nielsen/.bashrc
conda activate expllm_env

CONFIG=${1:?usage: sbatch aggregate.sbatch <config.yaml>}

python3 -m src.experiment "${CONFIG}" --aggregate-only
echo "Aggregation Complete"
