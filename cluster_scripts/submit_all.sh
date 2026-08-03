#!/bin/bash
# Submit ALL experiments to SLURM: for every config under config/, one
# seed-parallel array job (indices taken from the config's `seeds` list)
# plus an aggregation job chained with --dependency=afterok.
#
# Run from the repository root on the login node:
#   bash cluster_scripts/submit_all.sh            # all configs
#   bash cluster_scripts/submit_all.sh config/adbench_demo.yaml   # subset
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(config/*.yaml)
fi

for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "${CONFIG}" .yaml)

    # read the seed list (e.g. "0,1,2") AND whether the config touches an image
    # dataset (mvtec / image:* spec) — those need a GPU for backbone features,
    # the pure tabular/text configs are CPU-bound. One pass, two answers.
    read -r SEEDS NEEDS_GPU < <(python3 - "$CONFIG" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
seeds = ",".join(str(s) for s in cfg.get("seeds", [0]))

def specs(key):
    for e in cfg.get(key, []) or []:
        yield (e if isinstance(e, str) else e.get("spec", "")).lower()

needs_gpu = any("mvtec" in s or s.startswith("image")
                for s in list(specs("train")) + list(specs("test")))
print(seeds, "gpu" if needs_gpu else "cpu")
EOF
)

    GRES_ARG=()
    if [ "${NEEDS_GPU}" = "gpu" ]; then
        GRES_ARG=(--gres=gpu:1)
        echo "  ${NAME}: image dataset detected -> requesting a GPU"
    fi

    ARRAY_ID=$(sbatch --parsable \
        --array="${SEEDS}" \
        --job-name="oafa_${NAME}" \
        ${GRES_ARG[@]+"${GRES_ARG[@]}"} \
        cluster_scripts/run_config_array.sbatch "${CONFIG}")
    echo "submitted ${CONFIG}: array job ${ARRAY_ID} (seeds ${SEEDS})"

    AGG_ID=$(sbatch --parsable \
        --dependency="afterok:${ARRAY_ID}" \
        --job-name="oafa_${NAME}_agg" \
        cluster_scripts/aggregate.sbatch "${CONFIG}")
    echo "submitted aggregation ${AGG_ID} (afterok:${ARRAY_ID})"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
