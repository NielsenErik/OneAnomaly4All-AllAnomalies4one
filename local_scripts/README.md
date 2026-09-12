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
| `run_diagnosis_ws.sh` | —                                  | the diagnosis programme, start to finish, in one process |
| `queue_diagnosis_ws_tsp.sh` | —                            | the same programme as a task-spooler (`tsp`) chain |

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

## The diagnosis programme (task spooler)

`run_diagnosis_ws.sh` runs preflight, the FD001 reliability study, the
pre-registered gate, the FD003 pilot (only if the gate passes), the benchmark
and the reports in one foreground process. `queue_diagnosis_ws_tsp.sh` queues
the same stages as separate `tsp` jobs instead, which is what you want when the
session can drop: the jobs outlive the SSH connection, each stage keeps its own
exit status, and a single queue slot guarantees the benchmark — the latency
source of truth — never shares the machine with a training run.

```bash
# needs: apt install task-spooler   (binary `tsp`, or `ts` on some distros)
bash local_scripts/queue_diagnosis_ws_tsp.sh              # the whole chain
bash local_scripts/queue_diagnosis_ws_tsp.sh --tests-only # just the suites
DEVICE=cuda:1 bash local_scripts/queue_diagnosis_ws_tsp.sh

export TS_SOCKET=/tmp/ts_diagnosis_ws   # the queue this script uses
tsp             # the queue and each job's exit status
tsp -c <id>     # one job's output (-t <id> tails a running one)
tsp -k <id>     # kill the running job;  tsp -r <id> removes a queued one
```

Stages are chained with `-W` (run only if the previous finished well), so a
failed stage stops the ones that depend on it; the reports and the bundle use
`-D` (run once the previous ends, either way) so a failure still leaves a
readable record. The FD003 pilot hanging off `-W` on the gate is the protocol,
not a convenience: a pilot run on an unreliable fit measures the unreliability
and then sizes the confirmation from it.
