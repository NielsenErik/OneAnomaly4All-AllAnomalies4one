# Launchers — running the time-series experiments on a workstation

Everything below assumes the repo root as the working directory. The scripts
resolve the interpreter, cap thread counts, create the log tree and tee console
output themselves; you do not need to `export PYTHONPATH` or activate anything.

## The three commands

```bash
# 1. wiring check — minutes.  Run after any change, before any batch.
bash poc/time_series/launch/run_smoke.sh

# 2. one experiment, watched live
bash poc/time_series/launch/run_config.sh config/ts/cmapss_ad.yaml

# 3. the whole thing, unattended
bash poc/time_series/launch/run_workstation.sh
```

## Environment knobs

| variable | default | meaning |
|---|---|---|
| `PY` | project conda env, else `python3` | interpreter |
| `DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `SEEDS` | whatever the config says | e.g. `SEEDS="0 1 2 3 4"` |
| `JOBS` | `1` | configs run in parallel |
| `THREADS` | `8` (`4` when `JOBS>1`) | torch/OMP threads **per job** |
| `OUT` | `logs/ts` | log root |
| `TIERS` | `1 2 3 4 5` | which tiers `run_workstation.sh` runs |
| `FORCE` | unset | redo runs that already completed |
| `DRY` | unset | print the plan, run nothing |
| `EXTRA` | empty | extra args forwarded to the runner |

### On a 64 GB / RTX 4080 box

```bash
JOBS=3 THREADS=4 DEVICE=cpu bash poc/time_series/launch/run_workstation.sh
```

is usually the fastest configuration, and that is not a typo. A probabilistic
circuit is a deep DAG of **many small** tensor ops rather than a few large
matmuls, so it is launch-latency bound: three CPU processes at four threads each
beat one process at twelve threads (BLAS oversubscription on tiny tensors) and
often beat a single GPU process as well. The GPU wins as `K`, the window and the
batch grow — `DEVICE=cuda` with `model.K=16` and `dataset.window=32` is where it
starts to pay. **Measure on your data before assuming**; the plumbing supports
both and the run's `env.json` records which was used.

Memory: a run holds the windowed dataset plus one circuit, typically well under
2 GB. Three parallel jobs on C-MAPSS peak around 6 GB. N-C-MAPSS is the
exception — the first parse of a release reads the whole HDF5 (1–5 GB) before
caching a compact `.npz`; run those **one at a time the first time**
(`JOBS=1 TIERS=3`), after which the cache makes them cheap.

## Tiers

| tier | configs | question |
|---|---|---|
| 0 | `smoke` | is anything broken? (always runs) |
| 1 | `cmapss_ad`, `cmapss_explain` | **the credibility gap and the contribution** — does parity-on-detection / exclusivity-on-explanation hold on real engines? |
| 2 | `cmapss_calibration`, `cmapss_rul` | does conformalising the exact predictive close the coverage gap? does the censoring negative reproduce off our own generator? |
| 3 | `ncmapss_ad`, `ncmapss_rul` | does it survive real flight conditions? |
| 4 | `synthetic_*`, `scaling` | the control arm every real number is read against |
| 5 | `cmapss_structure`, `capacity_sweep` | is it the structure or just the budget? |

Tiers are ordered by decision value, not by cost: an interrupted batch should
still have answered the questions that matter.

## Resuming, and what to do when something fails

Every run writes `status.json`. A run whose status is `ok` **and** whose config
hash matches is skipped on the next launch, so re-running after a crash, a
reboot or a Ctrl-C costs nothing:

```bash
bash poc/time_series/launch/run_workstation.sh          # picks up where it stopped
FORCE=1 bash poc/time_series/launch/run_workstation.sh  # redo everything
```

A failed variant does not stop the batch. To find and read failures:

```bash
grep -l '"status": "failed"' logs/ts/*/*/*/status.json
cat logs/ts/cmapss_rul/censor_frac-0.7_*/seed0/run.log
```

The most likely failures, and what they mean:

- `DegenerateModelError` — the model is provably carrying no information
  (constant predictive or constant density). **This is the guardrail working.**
  Do not tune around it: check `model.tau_where` (must be `deep`),
  `model.weight_jitter` (must be > 0) and the leaf jitter. Three wrong results
  in this project came from exactly this failure going unnoticed.
- `data missing` — the config skipped itself; see `data/README.md`.
- `no healthy training windows` — `dataset.window` exceeds the shortest
  trajectory, or `healthy_frac` is too strict for that fleet.

## Reading the output

```
logs/ts/<experiment>/
  summary.md                     tables, per stage, mean ± sd over seeds
  summary.csv                    the same in long format — load this in pandas
  summary.json
  index.jsonl                    one line per run: status, wall time, hash
  <variant>/seed<N>/
    run.log                      full console transcript
    config.json / env.json       resolved config; git commit, GPU, threads
    results.jsonl                the comparable rows
    metrics.json                 nested per-stage metrics
    history_*.csv                per-epoch training curves
    status.json                  ok | failed, wall time, peak RSS / GPU
    artifacts/                   scores.npz, attributions.npz, figures
```

Re-aggregate at any time without rerunning anything:

```bash
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts --recursive
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts/cmapss_ad --stage explain
```

## Ad-hoc runs

The runner takes the same arguments as the launchers, plus overrides:

```bash
PYTHONPATH=. python -m poc.time_series.runner config/ts/cmapss_rul.yaml --dry-run
PYTHONPATH=. python -m poc.time_series.runner config/ts/cmapss_ad.yaml \
    --only subset-FD002 --seeds 0 1 2 3 4 --set model.K=12 eval.plots=false
```

`--set` is repeatable and typed (`model.K=12` is an int, `eval.plots=false` is a
bool). `--only` filters variants by substring. `--dry-run` prints the expansion
and exits.
