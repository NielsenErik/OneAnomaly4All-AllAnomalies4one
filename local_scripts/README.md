# Local runners (personal workstation)

Run the full experiment matrix without SLURM. Each config under `config/` is a
complete experimental setup that internally sweeps **seeds × vtree methods ×
adapt modes × baselines** and writes the metric summaries
(`logs/summary_<name>.json`, AUROC + held-out NLL). These scripts are the
workstation analogue of `cluster_scripts/`.

| local | cluster equivalent | does |
|-------|--------------------|------|
| `run_all_local.sh`    | `submit_all.sh` / `run_all.sbatch` | every config, every metric |
| `run_config_local.sh` | `run_config.sbatch`                | one config, all seeds + summary |
| `run_seed_local.sh`   | `run_config_array.sbatch` (one task) | one seed of one config |

## Usage

```bash
# everything (all config/*.yaml), sequential — simplest, lowest memory
bash local_scripts/run_all_local.sh

# everything, parallelizing the seeds within each config (3 at a time)
bash local_scripts/run_all_local.sh --parallel 3

# a subset of setups
bash local_scripts/run_all_local.sh config/adbench_demo.yaml config/sos_ablation.yaml

# one config (optionally seed-parallel)
bash local_scripts/run_config_local.sh config/vtree_ablation.yaml
bash local_scripts/run_config_local.sh config/vtree_ablation.yaml --parallel 3

# one seed only (then aggregate with: python -m src.experiment <cfg> --aggregate-only)
bash local_scripts/run_seed_local.sh config/adbench_demo.yaml 0
```

## Environment

`_common.sh` (sourced by all three) cd's to the repo root, activates a conda
env if available, and sources `.env` for `HF_TOKEN`.

- `OAFA_CONDA_ENV` — conda env name (default `expllm_env`). If conda or the env
  is absent, the current `python3` is used (e.g. an already-active venv), so the
  scripts run on any workstation.
- `OMP_NUM_THREADS` — thread cap (default: cores − 1).

`config/multimodal_demo.yaml` featurizes images and wants a GPU + `HF_TOKEN`;
the tabular/text configs are CPU-only.
