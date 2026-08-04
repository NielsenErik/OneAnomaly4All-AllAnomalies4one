# PoC — Time Series: Anomaly Detection + Remaining Useful Life

Proof of concept for the proposal in [`brainstorm_time_series.md`](../../brainstorm_time_series.md),
built on the region-graph DAG rebuild of the circuit layer.

Two things are being demonstrated, and they are separable:

1. **The engineering claim (settled).** `DensityPC` builds a *tree*, so a leaf
   position is instantiated `K^depth` times and windowed multivariate input is
   not representable. `RegionGraphPC` builds each region once and shares it
   across the `K` parents that mix it — `O(d·K²)` instead of `O(d·K^depth)` —
   with all four circuit properties intact.
2. **The research claim (under test).** One joint circuit over `(window, τ)`
   answers detection *and* prognosis as exact queries on a single density, and
   trains on right-censored units with an exact likelihood. Whether that beats
   the trivial adversaries is what these scripts measure. **It does not
   currently win on detection** — see [Results](#results).

---

## Quick start — the config-driven pipeline (use this)

Since 2026-08-03 there is one entry point for training, evaluation, logging and
aggregation, driven by YAML configs, and it runs on **real C-MAPSS and
N-C-MAPSS** as well as the simulator:

```bash
bash poc/time_series/launch/run_smoke.sh                            # ~3 min, checks everything
bash poc/time_series/launch/run_config.sh config/ts/cmapss_ad.yaml  # one experiment
bash poc/time_series/launch/run_workstation.sh                      # the full batch
```

- configs: [`config/ts/`](../../config/ts) — one file per experiment, with the
  question it answers written at the top
- launchers, tiers, env knobs, resume, failure triage:
  [`launch/README.md`](launch/README.md)
- data acquisition and what is real vs injected: [`data/README.md`](../../data/README.md)
- stages: `ad`, `explain`, `rul`, `calibration`, `scaling` — all five run from
  the same config and write the same structured rows

```
runner.py      expand config -> variants x seeds, resume, isolate failures
pipeline.py    the five stages
datasets.py    synthetic | cmapss:FD00x | ncmapss:DS0x, one interface
data_real.py   the NASA loaders
conformal.py   split conformal on the circuit's own predictive
ts_logging.py  per-run artifacts (config, env, git, curves, status)
aggregate.py   many runs -> summary.csv / summary.md
```

The scripts documented in the rest of this file (`run_ad.py`, `run_rul.py`,
`run_explain.py`, `bench_scaling.py`) still work unchanged and remain the
quickest way to poke at one thing:

```bash
export PYTHONPATH=.
python -m poc.time_series.bench_scaling      # 1. tree vs DAG scaling      (~1 min)
python -m poc.time_series.run_ad             # 2. anomaly detection        (~10 min)
python -m poc.time_series.run_rul            # 3. RUL / survival           (~25 min)
```

If `python` is not the project interpreter:

```bash
PYTHONPATH=. ~/miniconda3/envs/expllm_env/bin/python -m poc.time_series.bench_scaling
```

The simulator needs no downloads and is seeded and reproducible; the two real
datasets are opt-in (`python -m poc.time_series.check_data` says what is
present and how to get the rest).

---

## Files

| File                 | What it is                                                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data.py`          | C-MAPSS-shaped synthetic fleet simulator + windowing + task builders. Optional loader for the real C-MAPSS text files.                                                                     |
| `circuits.py`      | `WindowPC` (density over a window) and `SurvivalPC` (joint over window × τ with exact censored likelihood), plus vtree selection.                                                    |
| `baselines.py`     | Simple tier (z-score, moving average, diagonal Gaussian, PCA, 1-NN, Mahalanobis) and advanced tier (IForest, GMM, conv AE, Deep SVDD); RUL: ridge, MLP, conformalised quantile regression. |
| `metrics.py`       | AUROC / AP (no point adjustment), CRPS, interval score, PICP/MPIW, calibration error, NASA score.                                                                                          |
| `bench_scaling.py` | Experiment 1 — the layout comparison.                                                                                                                                                     |
| `bench_device.py`  | CPU vs GPU vs `torch.compile`, for both evaluators, with the correctness gate. Run it once per machine before committing to a batch.                                                       |
| `run_ad.py`        | Experiment 2 — detection, dead-sensor query, typed decomposition, vtree ablation.                                                                                                         |
| `run_rul.py`       | Experiment 3 — censoring ablation, calibration vs CQR, survival under partial evidence.                                                                                                   |

---

## Which evaluator, which device

Two evaluators, identical semantics, selected with `evaluator: layered | recursive`
in any config or `--evaluator` on the runner:

* **layered** (default) — `CompiledCircuit`, §6a of `src/probabilistic_circuits.py`.
  Compiles the DAG once into a topological layer schedule, so the Python loop is
  O(depth) instead of O(#nodes): 1057 nodes → 16 layers on the standard 8×14/K=6
  window. Gated against the recursion on a real batch at every fit.
* **recursive** — the per-node reference. Works on every circuit (including SOS,
  which the compiled path refuses rather than guessing at). Keep it for debugging
  and for the A/B; do not produce results with it.

The device answer follows from the evaluator, and this is the part that is easy to
get backwards:

| evaluator | GPU vs CPU                                      |
| --------- | ----------------------------------------------- |
| recursive | GPU **loses** at every batch size (0.19–0.54×)  |
| layered   | GPU **wins** from batch ~128, ~2.2× at 2048+    |

So "a circuit is GPU-hostile" was a statement about the recursion, not about
circuits. Measure your own machine:

```bash
PYTHONPATH=. python -m poc.time_series.bench_device            # both evaluators, batch sweep
PYTHONPATH=. python -m poc.time_series.bench_device --compile  # + torch.compile
BENCH_DEVICE=1 bash poc/time_series/launch/run_workstation.sh  # before an overnight batch
```

Every `run.log` records the device, the evaluator and windows/s, so a finished run
says which regime it was in instead of leaving it to be assumed.

---

## Experiment 1 — tree vs DAG (`bench_scaling.py`)

```bash
python -m poc.time_series.bench_scaling                       # default K=4
python -m poc.time_series.bench_scaling --K 6                 # more components
python -m poc.time_series.bench_scaling --dims 8 16 32 64 128 --K 4
python -m poc.time_series.bench_scaling --K 4 --batch 128
```

Builds both layouts over the *same* vtree, counts distinct nodes and
parameters, and times construction and one forward pass. The tree is skipped
above 300k predicted leaf modules (building it takes longer than the rest of
the PoC).

**Flags:** `--K` mixture components · `--dims` feature dimensions to sweep ·
`--batch` batch size for the timed forward pass.

---

## Experiment 2 — anomaly detection (`run_ad.py`)

```bash
# headline table: circuit vs all baselines, 3 seeds
python -m poc.time_series.run_ad

# fast smoke run (simple baselines only, 1 seed)
python -m poc.time_series.run_ad --seeds 0 --epochs 15 --fast

# the two queries no baseline can express
python -m poc.time_series.run_ad --missing --typed

# structure ablation: same budget, only the vtree changes
python -m poc.time_series.run_ad --vtree-ablation

# harder / easier task
python -m poc.time_series.run_ad --strength 0.4 --inject-rate 0.15
python -m poc.time_series.run_ad --window 16 --channels 25 --units 100

# capacity sweep
python -m poc.time_series.run_ad --K 12 --leaf-components 3 --epochs 60

# CURVATURE vtrees (Ollivier-Ricci / Forman bottleneck cuts)
python -m poc.time_series.run_ad --vtree orc    --fast --out logs/poc_ts_ad_orc.json
python -m poc.time_series.run_ad --vtree forman --fast --out logs/poc_ts_ad_forman.json

# SQUARED (SOS) circuit — subtractive mixtures, still exactly normalised.
# Keep K small: the partition function pairs every node with every node in its
# region, so SOS is O(K^4) per region where the monotone circuit is O(K^2).
python -m poc.time_series.run_ad --sos --K 2 --epochs 30 --fast --out logs/poc_ts_ad_sos.json
python -m poc.time_series.run_ad --sos --K 3 --vtree orc --seeds 0 --fast

# REGION GRAPHS (n-ary; curvature picks the arity from the data)
python -m poc.time_series.run_ad --vtree orc_rg     --fast
python -m poc.time_series.run_ad --vtree forman_rg  --fast
python -m poc.time_series.run_ad --vtree spectral_rg --fast

# MULTI-PARTITION region graph: a sum node mixes several decompositions of the
# same scope.  Exact density/marginals/box queries survive; structured
# decomposability does NOT, so --sos is refused with an explanation.
python -m poc.time_series.run_ad --vtree orc_rg_multi --fast

# CHAIN (HMM-shaped) structure — the order-sensitive one; currently the best
# configuration in this PoC.  --delta adds first differences (unit Jacobian).
python -m poc.time_series.run_ad --vtree chain --typed --missing
python -m poc.time_series.run_ad --vtree chain --delta --fast
python -m poc.time_series.run_ad --vtree chain_grouped --fast

# structure ablation across vtrees AND region graphs
python -m poc.time_series.run_ad --vtree-ablation \
    --ablation-methods random time chain orc_rg forman_rg spectral chow_liu
```

### Structure choices (`--vtree`)

| value | kind | what it encodes |
|---|---|---|
| `time`, `channel`, `channel_groups` | binary vtree | hand-built temporal / per-sensor locality |
| `chow_liu`, `spectral`, `orc`, `forman` | binary vtree | learned, binary cuts |
| `orc_rg`, `forman_rg`, `spectral_rg` | region graph | learned, **n-ary** — curvature picks the arity |
| `orc_rg_multi`, `forman_rg_multi` | region graph | **several partitions per region** (drops structured decomposability) |
| `chain`, `chain_grouped`, `chain_full` | region graph | **HMM-shaped**, order-sensitive |
| `random` | binary vtree | control |

**Key flags**

| Flag                                         | Default                 | Meaning                                                                                                |
| -------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `--seeds`                                  | `0 1 2`               | seeds to average over                                                                                  |
| `--window` / `--stride`                  | `8` / `2`           | window length, sliding stride                                                                          |
| `--channels` / `--units` / `--regimes` | `14` / `60` / `3` | fleet size and shape                                                                                   |
| `--strength`                               | `1.0`                 | anomaly magnitude (lower = harder)                                                                     |
| `--inject-rate`                            | `0.12`                | fraction of healthy test windows that get an injected anomaly                                          |
| `--vtree`                                  | `time`                | `time`, `channel`, `channel_groups`, `chow_liu`, `spectral`, `orc`, `forman`, `random` |
| `--K` / `--leaf-components`              | `6` / `1`           | mixture units per region; components per leaf                                                          |
| `--epochs` / `--lr`                      | `40` / `0.05`       | training                                                                                               |
| `--missing`                                | off                     | dead-sensor query (exact marginalisation vs imputation)                                                |
| `--typed`                                  | off                     | exact marginal-vs-structural decomposition                                                             |
| `--vtree-ablation`                         | off                     | run the structure comparison instead of the main table                                                 |
| `--fast`                                   | off                     | skip IForest / GMM / conv-AE / Deep SVDD                                                               |
| `--out`                                    | `logs/poc_ts_ad.json` | results file                                                                                           |

### What the task contains

Anomalies are reported **per kind**, because the aggregate hides everything
that matters:

- `spike`, `offset`, `drift` — **marginal** anomalies: some channel visits
  values it should not. Per-channel scoring can see these.
- `decouple` — a channel's values are permuted along time. Its marginal over
  the window is *exactly unchanged*; only temporal structure breaks.
- `desync` — a channel is swapped with the same channel from another normal
  window. Marginals stay in-distribution; only cross-channel coupling breaks.
- `organic` — genuine late-stage degradation, nothing injected.

`decouple` and `desync` are the **structural** anomalies. A per-channel
detector cannot see them even in principle, so they are the only rows where a
joint density has anything to prove. They exist because
[arXiv:2606.02670](https://arxiv.org/pdf/2606.02670) (2026) shows anomalies in
SMAP/MSL/SMD/SWaT are largely univariate — a benchmark without structural
anomalies cannot test a joint model at all.

The normal data has **nonlinear** cross-channel coupling (tanh and squared
responses within sensor groups) and multiple operating regimes. An earlier
linear-Gaussian version of the generator was discarded: Mahalanobis scored
AUROC 1.000 on it, because for linear-Gaussian data a full-covariance Gaussian
*is* the optimal detector and the benchmark measured nothing.

---

## Experiment 3 — RUL and survival (`run_rul.py`)

```bash
# full run: censoring ablation + baselines + partial-evidence query
python -m poc.time_series.run_rul

# add the survival-function table
python -m poc.time_series.run_rul --survival-demo

# quick single seed
python -m poc.time_series.run_rul --seeds 0 --epochs 20 --no-partial

# heavier censoring — where the exact censored likelihood should matter most
python -m poc.time_series.run_rul --censor-frac 0.6

# τ coupled deep in the circuit instead of at the root
python -m poc.time_series.run_rul --tau-where deep --K 12

# finer RUL resolution
python -m poc.time_series.run_rul --bins 40 --cap 130
```

**Key flags**

| Flag                            | Default          | Meaning                                                                              |
| ------------------------------- | ---------------- | ------------------------------------------------------------------------------------ |
| `--censor-frac`               | `0.35`         | fraction of fleet units truncated before failure                                     |
| `--bins` / `--cap`          | `20` / `130` | ordinal RUL bins; RUL capped at this many cycles                                     |
| `--tau-where`                 | `root`         | `root` (K×K coupling at the top) or `deep` (τ paired low, coupling propagates) |
| `--K` / `--leaf-components` | `10` / `1`   | circuit capacity                                                                     |
| `--alpha`                     | `0.10`         | 1 − nominal coverage (0.10 → 90% intervals)                                        |
| `--n-dead`                    | `3`            | sensors killed in the partial-evidence query                                         |
| `--survival-demo`             | off              | print S(t\|x) bucketed by true remaining life                                        |
| `--no-partial`                | off              | skip the partial-evidence query                                                      |

### The three comparisons

**A. Censoring.** The *same* circuit with the *same* budget, trained twice:
dropping censored units (what a regressor must do), versus adding the exact
`log P(τ ≥ c)` term for them. If the second does not help, censoring handling
buys nothing and that part of the claim dies.

**B. Calibration vs CQR.** Conformalised quantile regression is the
prognostics field's current answer to calibrated RUL intervals
([arXiv:2212.14612](https://arxiv.org/abs/2212.14612); CQR-LSTM, *Sensors*
2026). It has a finite-sample coverage guarantee, so **matching** it is the
bar, not beating it. Reported on CRPS, interval score, PICP and MPIW.

**C. Query reach.** Survival with dead sensors. The circuit marginalises them
out of the joint exactly; every baseline must impute. CQR cannot appear in this
table at all — it needs a complete feature vector to emit an interval. This is
the axis where exactness has no cheap substitute, and per the brainstorm
document it is where the T1 claim should be anchored.

---

## Results

### Experiment 1 — settled

`K=4`, balanced vtree, measured on this machine:

| d   | tree leaves      | DAG leaves | tree params | DAG params | tree build         | DAG build        |
| --- | ---------------- | ---------- | ----------- | ---------- | ------------------ | ---------------- |
| 16  | 4,096            | 64         | 10,532      | 1,040      | 0.15 s             | 0.00 s           |
| 32  | 32,768           | 128        | 84,260      | 2,192      | 2.13 s             | 0.01 s           |
| 64  | 262,144          | 256        | 674,084     | 4,496      | **183.21 s** | **0.02 s** |
| 112 | ~1.8e6 (skipped) | 448        | —          | 7,952      | —                 | 0.03 s           |
| 256 | ~1.7e7 (skipped) | 1,024      | —          | 18,320     | —                 | 0.07 s           |

Realistic windows, DAG only, all four properties validated:

| window | d   | DAG leaves | DAG params | tree would need |
| ------ | --- | ---------- | ---------- | --------------- |
| 8×14  | 112 | 448        | 7,952      | ~1.8e6 leaves   |
| 16×25 | 400 | 1,600      | 28,688     | ~1.0e8 leaves   |
| 32×25 | 800 | 3,200      | 57,488     | ~8.4e8 leaves   |

The rebuild does what it was supposed to do. All **183 pre-existing tests still
pass** — this is a layout change, not a modelling change.

### Experiment 2 — detection: a draw on the score, a win on the queries

Seed 0, `--strength 0.6 --fast --typed --missing`, K=6, post-fix
(`logs/poc_ts_ad_fixed.json`):

| detector                       | AUROC           | AP              | [decouple]      | [desync]        | [drift] | [offset] | [spike] |
| ------------------------------ | --------------- | --------------- | --------------- | --------------- | ------- | -------- | ------- |
| Mahalanobis                    | **0.917** | **0.895** | 0.500           | 0.845           | 1.000   | 0.997    | 0.973   |
| 1-NN distance                  | 0.908           | 0.867           | 0.487           | 0.816           | 0.990   | 0.986    | 0.956   |
| **RegionGraphPC** (time) | 0.903           | 0.858           | 0.516           | 0.838           | 0.999   | 0.996    | 0.855   |
| ↳ PC, structural-only score   | 0.882           | 0.832           | **0.585** | **0.890** | 0.916   | 0.956    | 0.695   |
| z-score (per channel)          | 0.805           | 0.707           | 0.477           | 0.547           | 0.833   | 0.699    | 0.842   |
| diagonal Gaussian              | 0.767           | 0.659           | 0.495           | 0.569           | 0.802   | 0.548    | 0.714   |
| moving-average residual        | 0.753           | 0.499           | 0.459           | 0.638           | 0.711   | 0.574    | 0.993   |
| PCA reconstruction             | 0.745           | 0.593           | 0.427           | 0.612           | 0.708   | 0.639    | 0.544   |

Read honestly:

- **On the plain score the circuit does not win.** 0.903 vs Mahalanobis 0.917
  and 1-NN 0.908 — a statistical tie at one seed, and no reason to prefer a
  circuit if the density score is all you want. Fixing the symmetry bug moved it
  from 0.879 to 0.903, so it is now *competitive*, not *better*.
- **On the two structural anomaly types the circuit's structural score is best
  in the table** — `decouple` 0.585 vs 0.427–0.500 for everything else,
  `desync` 0.890 vs Mahalanobis 0.845. These are the anomalies with an
  unchanged per-channel marginal, so this is the one place a joint density has
  something a distance can't reach.
- `decouple` remains weak in absolute terms (0.585). Time-permuting a channel
  is still nearly invisible to a window density over per-`(t,c)` leaves. This
  is the most informative negative in the PoC.

#### Circuit variants: curvature vtrees and the squared (SOS) circuit

Same protocol as the table above (3 seeds, `--fast`, K=6, default strength),
only the circuit changes:

| circuit variant                            | AUROC                   | AP                      | [decouple]      | [desync] |
| ------------------------------------------ | ----------------------- | ----------------------- | --------------- | -------- |
| RegionGraphPC, **time** vtree, K=6   | **0.9202 ± 0.0028** | **0.8798 ± 0.0079** | 0.5445          | 0.8461   |
| RegionGraphPC, **orc** vtree, K=6    | 0.9101 ± 0.0077         | 0.8697 ± 0.0135         | 0.5150          | 0.8003   |
| **SquaredPC / SOS**, time vtree, K=2 | 0.9020 ± 0.0167         | 0.8571 ± 0.0404         | 0.5121          | 0.7461   |
| RegionGraphPC, **forman** vtree, K=6 | 0.9001 ± 0.0028         | 0.8511 ± 0.0168         | 0.4862          | 0.7753   |
| *(reference)* Mahalanobis                  | 0.9316 ± 0.0079         | 0.9119 ± 0.0056         | 0.5507          | 0.8703   |

**Curvature vtrees lose to the hand-built temporal vtree**, consistently across
all three seeds (ORC −1.0 AUROC, Forman −2.0). This is a clean negative for the
curvature-vtree line *in this domain*: where the true dependency structure is
known a priori (time adjacency), a learner that must rediscover it from a noisy
MI estimate does strictly worse than being told.

Worth taking seriously rather than tuning away, because it is exactly the
outcome the 2026-06-11 evaluation flagged as the risk for that line ("geometry
is decoration"). Time series was chosen as the venue for that ablation
*because* it has ground-truth structure — and the ground truth is winning. What
survives: curvature may still pay where structure is genuinely unknown (raw
tabular), and ORC consistently beats Forman, which at least orders the two
curvature notions.

**The SOS circuit does not pay for itself here either** (0.9020 at K=2, with
much higher seed variance, ±0.0167 vs ±0.0028). Subtractive mixtures are more
expressive per component, but the squared construction pairs every node with
every node in its region, so the partition function is O(K⁴) per region against
O(K²) for the monotone circuit — at equal wall-clock the monotone circuit
affords K=6 where SOS affords K=2, and the extra components win. That is a
real, reportable trade-off rather than a tuning artefact, though a matched-FLOP
rather than matched-K comparison would state it more rigorously.

Caveat on all four rows: one synthetic generator, `--fast` baselines, three
seeds. This ranks the variants; it does not settle the research question.

#### The typed decomposition (the actual result)

Mean worst-channel surprise, `--typed`:

| window kind | n    | marginal        | conditional | structural      |
| ----------- | ---- | --------------- | ----------- | --------------- |
| normal      | 1180 | 11.27           | 7.93        | **0.04**  |
| decouple    | 34   | 10.79           | 10.18       | 2.66            |
| **desync**  | 26   | **11.41** | 29.80       | **25.28** |
| spike       | 34   | 30.03           | 27.76       | 1.68            |
| offset      | 41   | 66.99           | 74.38       | 14.71           |
| drift       | 28   | 100.82          | 97.96       | 8.13            |
| organic     | 104  | 188.34          | 193.07      | 30.08           |

The `desync` row is the point. Its **marginal surprise is 11.41 against a
normal baseline of 11.27** — statistically indistinguishable, exactly as
designed, since the injected channel's values came from a real normal window.
Its **structural term is 25.28 against a normal baseline of 0.04**. The
decomposition separates "this channel visited a strange value" from "this
channel is strange *given the others*", exactly, from two marginals of one
circuit. `spike` shows the mirror image: marginal 30.03, structural 1.68 —
correctly typed as univariate, i.e. something a per-channel z-score already
handles.

This is the T3 claim from the brainstorm document working as advertised, and it
is a measuring instrument no baseline in the table can provide.

#### Dead sensors: exact marginalisation vs imputation

Three sensors killed at test time, same trained model:

| scoring                          | AUROC           | AP    |
| -------------------------------- | --------------- | ----- |
| PC, exact marginalisation        | **0.895** | 0.843 |
| PC, mean-imputed                 | 0.860           | 0.792 |
| z-score, mean-imputed            | 0.781           | 0.669 |
| diagonal Gaussian, mean-imputed  | 0.740           | 0.635 |
| moving-average, mean-imputed     | 0.721           | 0.442 |

Integrating the dead sensors out beats imputing them by **+3.6 AUROC on the
identical circuit** — the gap is attributable to the query, not the model. Note
the honest caveat: the PC with dead sensors (0.895) still sits just below
Mahalanobis with *all* sensors (0.917), so this argues for the query's value,
not for the model's superiority.

### Experiment 3 — the censoring claim is NOT supported by this run

Two degeneracy bugs were found first, both fixed in
`src/probabilistic_circuits.py`. The initial run produced an **identical
survival curve for every unit** (`S(20)=0.911` whether a unit had 5 cycles left
or 130) and byte-identical CRPS for the full, marginalised and imputed
variants — `p(τ|x)` did not depend on `x` at all. Causes:

1. **Leaf jitter did not cover all leaf types.** `_fit_leaves_with_jitter` only
   perturbed scalar-`mu` leaves, so `CategoricalLeaf` (the τ variable) and
   `GaussianMixtureLeaf` siblings started identical and stayed identical. This
   also explains why `--leaf-components 3` hurt in Experiment 2.
2. **Sum-node weights were initialised uniformly.** In the tree layout each
   component owns a private sub-circuit, so leaf jitter suffices. In the DAG the
   `K` units of a region are sum nodes over the *same shared product list*, so
   uniform weights make them identical functions — and because each unit is
   paired symmetrically with every sibling unit above it, they receive
   identical gradients and the symmetry never breaks. Every region silently
   collapses to one effective component. `RegionGraphPC` now takes
   `weight_jitter` (default 0.5) and perturbs every sum node at construction.

Bug 2 is a genuine property of the DAG layout that the tree layout masks —
worth stating in any writeup of the rebuild.

#### After both fixes: the survival function works

Seed 0, 25 epochs, K=10, `--survival-demo`. Mean predicted `S(t|x)` grouped by
**true** remaining life:

| true RUL | n   | S(20)           | S(40)           | S(60)           |
| -------- | --- | --------------- | --------------- | --------------- |
| 0–20     | 77  | 0.632           | 0.363           | **0.170** |
| 20–50    | 120 | 0.724           | 0.522           | 0.372           |
| 50–90    | 163 | 0.877           | 0.784           | 0.703           |
| 90–131   | 464 | **0.982** | **0.966** | **0.932** |

Monotone in both directions and correctly ordered: units about to fail get low
survival, healthy units get high survival. This is one exact query on the same
density that produced the point RUL and the anomaly score, and it is the
qualitative behaviour the T1 claim predicts.

#### But the quantitative claim fails

Seed 0, 25 epochs (`logs/poc_ts_rul.json`):

| model                                | RMSE            | MAE             | CRPS            | interval score  | PICP (nominal 0.90) | MPIW  |
| ------------------------------------ | --------------- | --------------- | --------------- | --------------- | ------------------- | ----- |
| ridge regression                     | **23.72** | **18.57** | —              | —              | —                  | —    |
| MLP regressor                        | 25.70           | 19.76           | —              | —              | —                  | —    |
| quantile reg. + conformal (CQR)      | 26.77           | 20.62           | —              | **119.85** | **0.796**     | 68.71 |
| SurvivalPC (drop censored)           | 27.92           | 22.46           | **13.89** | 121.98          | 0.512               | 69.44 |
| SurvivalPC (**exact censored lik.**) | 28.62           | 22.12           | 14.57           | 125.88          | 0.489               | 64.70 |

Three negatives, all of which matter:

1. **The exact censored likelihood did not help — it slightly hurt** (CRPS
   13.89 → 14.57, RMSE 27.92 → 28.62). This is the *central* claim of the
   proposal's T1 and this run does not support it. Either 35% censoring at this
   sample size is not enough for the extra units to pay for themselves, or the
   censored term is being down-weighted by the joint window likelihood, or the
   implementation is subtly wrong. It needs to be chased before T1 is written up
   anywhere.
2. **Calibration is bad.** PICP 0.49–0.51 against a nominal 0.90, versus CQR's
   0.796. Conformal's finite-sample guarantee is doing real work here and the
   circuit's exactness is not substituting for it. Exact ≠ calibrated: the
   density is exactly normalised with respect to a model that is simply wrong.
3. **Plain ridge regression wins on point accuracy** (RMSE 23.72 vs 27.92).

The circuit does win CRPS outright — but no baseline here emits a full
predictive PMF, so that column has no competitor and proves little on its own.

**Verdict: Experiment 3 currently falsifies the headline version of T1.** What
survives is the qualitative demonstration (one density, four exact queries,
correctly ordered survival curves) and the machinery. The claim that exact
censoring handling *improves prognosis* is unsupported and should not be
repeated until this is resolved. Suggested next probes, in order:
`--censor-frac 0.6` (more censoring), `--tau-where deep --K 12` (richer
coupling), longer training, and per-regime rather than global normalisation.

---

## Honest limitations

- **Synthetic data.** The generator is shaped like C-MAPSS but is not C-MAPSS.
  Every number here is a sanity check on machinery, not evidence about real
  machines. `data.py::load_cmapss_fleet` is a drop-in replacement once the NASA
  `CMaps` files are placed in `data/cmapss/` — that is the first thing to do
  before believing any of this.
- **Window-level evaluation only.** No event-wise or range-based metrics, no
  VUS-PR. Point adjustment is not used anywhere, deliberately.
- **i.i.d. windows.** The circuit models the window distribution, not the
  process; overlapping windows are not independent draws.
- **Determinism is absent** with mixture leaves, so exact MPE imputation of a
  dead sensor is not available — only marginalisation.
- **SOS mode has no box query yet.** `SquaredPC` supports exact marginals but
  `pair_log_interval` is unimplemented, so survival queries are monotone-circuit
  only for now. The Gaussian case is closed-form (the product of two Gaussians
  is a scaled Gaussian, so its box integral is a Φ difference) — it is just not
  written.
- **`--tau-where root`** caps the window↔τ coupling at a `K×K` discrete latent.
  `--tau-where deep` is the more expressive option and is barely explored.

## Experiment 4 — explanation quality (`explain.py` + `run_explain.py`)

```bash
python -m poc.time_series.run_explain                       # the three claims
python -m poc.time_series.run_explain --examples --plots    # case studies + figures
python -m poc.time_series.run_explain --shapley 8           # exact-conditional Shapley
python -m poc.time_series.run_explain --no-deletion --seeds 0
```

Detection is a three-way tie (conv-AE 0.9446 ± 0.0131, chain PC 0.9368 ± 0.0092,
Mahalanobis 0.9364 ± 0.0097), so this is the experiment that decides whether the
circuit earns its place. "Explainable" is split into three separable claims:

| claim | what is measured | why it is separable |
|---|---|---|
| **1. Correctness** | channel localisation AUROC / prec@k vs the generator's ground-truth corrupted channels | a method can be complete and faithful while pointing at the wrong sensor |
| **2. Completeness** | do the attributions sum to the score, residual in nats | for the circuit this is a *theorem* (chain rule on exact marginals); for SHAP it is an estimation target |
| **3. Faithfulness** | deletion curves — neutralise the most-blamed channel first, re-score | model-agnostic, so nobody wins by being self-consistent |

The adversaries are deliberately strong. **Gaussian conditional** computes the
*same* conditional attribution exactly from the precision matrix — so the claim
can never be "only PCs do conditionals", it has to be "PCs do conditionals for
a model class that actually fits the data". **AE sampling-SHAP** is what
KernelSHAP really estimates: a marginal-sampling value function, because the
conditional is unavailable for the model it explains.

Results, 3 seeds (`--seeds 0 1 2 --epochs 40 --shapley 6`):

| attribution method | localisation AUROC | prec@k | deletion AUC ↓ |
| --- | --- | --- | --- |
| **PC conditional (exact)** | **0.9021 ± 0.0165** | **0.7510** | 0.3811 |
| PC Shapley (exact conditionals) | 0.8986 ± 0.0245 | 0.7471 | 0.4048 |
| AE reconstruction (per channel) | 0.8570 ± 0.0324 | 0.6710 | 0.5441 |
| PC marginal (exact) | 0.8350 ± 0.0247 | 0.5798 | **0.3624** |
| Gaussian conditional (exact) | 0.7750 ± 0.0251 | 0.6546 | 0.5742 |
| PC structural (exact) | 0.7747 ± 0.0434 | 0.6197 | 0.4898 |
| z-score (per channel) | 0.7451 ± 0.0167 | 0.5088 | 0.6002 |
| AE sampling-SHAP (32/ch) | 0.4975 ± 0.0289 | 0.1377 | 0.7948 |

Completeness: **max residual 1.8e-5 nats, mean 1.1e-6** — float32 round-off, not
approximation error.

#### Localisation AUROC by anomaly kind — where the effect actually lives

| method | spike | offset | drift | decouple | desync |
| --- | --- | --- | --- | --- | --- |
| PC conditional (exact) | 0.970 | **0.988** | **0.957** | 0.623 | **0.851** |
| PC Shapley | **0.985** | 0.981 | 0.951 | 0.614 | 0.834 |
| AE reconstruction | **1.000** | 0.885 | 0.887 | 0.651 | 0.782 |
| Gaussian conditional | 0.990 | 0.855 | 0.761 | **0.757** | 0.763 |
| PC marginal (exact) | 0.995 | 0.912 | 0.949 | 0.539 | 0.545 |
| PC structural (exact) | 0.782 | 0.825 | 0.751 | 0.698 | **0.851** |
| z-score | 0.999 | 0.794 | 0.805 | 0.453 | 0.530 |
| AE sampling-SHAP | 0.631 | 0.504 | 0.460 | 0.443 | 0.451 |

This table is the real result, and it is more interesting than the pooled one:

- **`spike` is a solved problem.** z-score gets 0.999, AE gets 1.000. Anything
  that claims credit for univariate point anomalies is claiming credit for
  nothing.
- **`desync` is where the joint model is irreplaceable.** PC conditional and PC
  structural both reach 0.851 while **PC marginal collapses to 0.545** — near
  chance on the *same circuit*. A 0.31 gap between the marginal and conditional
  view of one model is as clean a demonstration as this PoC produces: the
  information is in the conditional, and only a tractable model has it.
- **`decouple` goes to the Gaussian conditional (0.757), not the PC.** An honest
  negative: for a time-permutation anomaly a second-order model captures the
  relevant structure better than our mixture does. Worth chasing, not hiding.
- **Sampling-SHAP is at chance everywhere** (0.443–0.631). At 32 samples/channel
  the Monte-Carlo error swamps the signal — and it already costs more wall-clock
  than all four exact PC views combined.

**Caveat on the deletion column:** it is scored with the circuit's own
`pc.score`, so PC attributions have a home-field advantage there and the ranking
should not be read as method-independent. The localisation columns have no such
problem — ground truth comes from the generator, not from any model.

### The practical check (`--examples --plots`)

Numbers do not tell you whether an engineer could act on the output, so one
window per anomaly kind is printed and plotted (`logs/figs/case_*.png`, three
panels: raw window, exact (t,c) surprise grid, marginal-vs-structural split,
with ground-truth channels marked). Representative output:

```
[desync]   score 76.2 nats            [injected: desync, channels [8, 10]]
   ch10  broken relationship  marginal   -5.77  structural  48.38  worst at t=7  <- CORRECT
[spike]    score 132.8 nats           [injected: spike, channels [12]]
   ch12  broken relationship  marginal   17.53  structural  65.86  worst at t=3  <- CORRECT
[offset]   score 295.8 nats           [injected: offset, channels [4, 11, 13]]
   ch4/11/13  broken relationship — all three CORRECT
```

The `desync` line is the whole argument in one row: channel 10's **marginal
surprise is −5.77**, i.e. its values are *more* likely than average taken alone —
it would pass every per-channel alarm in existence — while its structural term is
48.4. The sensor is individually plausible and jointly impossible. The `spike`
figure shows the mirror case: a single bright cell at (t=3, ch12), localised
exactly in both time and channel.

Examples are chosen at the **median** anomaly score of each kind, not the most
extreme, so they are not cherry-picked.

## Bottom line

| claim | verdict |
|---|---|
| DAG rebuild removes the `K^depth` blowup, properties intact | **holds** |
| Chain (HMM-shaped) region graph is the best structure | **holds** — 0.9368 vs 0.9108 for a balanced temporal vtree |
| Circuit **matches** the best detectors | **holds** — three-way tie with conv-AE and Mahalanobis |
| Circuit **beats** them on detection | **no** — and should not be claimed |
| Exact attribution beats *approximate* attribution | **holds** — 0.899 vs 0.523 for sampling-SHAP |
| Attribution is exactly complete | **holds** — residual 3e-5 nats (round-off) |
| Exact typed (marginal vs structural) decomposition is informative | **holds — strongest result** |
| Exact marginalisation beats imputation under dead sensors | **holds** (+3.6 AUROC, same model) |
| Curvature vtrees beat hand-built temporal structure | **no** — consistently behind |
| SOS beats the monotone circuit at equal wall-clock | **no** |
| Survival function is qualitatively correct | **holds** |
| Exact censored likelihood improves prognosis | **no** — the central T1 claim is unsupported |

The defensible thesis is therefore **parity on detection, exclusivity on
explanation**: the circuit ties the best detectors on AUROC, and is the only one
of them that can say *why* correctly, completely, and in a form an operator can
act on. Concretely, on a `desync` window it reports that a sensor is
individually more likely than average yet jointly impossible — a statement no
per-channel monitor can make and no reconstruction error can express.

That is a narrower claim than the brainstorm document's T1 headline (exact
censored survival), which this PoC does not support. It is also a stronger one
than "we win by 0.002 AUROC", because it does not depend on a margin that seed
noise can erase.

## What would make this convincing

In priority order:

1. **Chase the censoring negative.** `--censor-frac 0.6`, `--tau-where deep
   --K 12`, longer training, per-regime normalisation. If it stays negative on
   real C-MAPSS, T1 should be retired rather than rewritten.
2. **Fix or explain `decouple` (~0.5 for everything).** A time-permuted channel
   should be exactly what a time-structured joint density catches. Candidates:
   temporal-difference leaves, explicit lag features, deeper coupling.
3. **Fix the calibration gap** (PICP 0.49 vs nominal 0.90). Exact normalisation
   is not calibration; this may need a post-hoc conformal layer *on top of* the
   circuit, which would be an honest hybrid rather than a defeat.
4. **Swap in real C-MAPSS** (`data.py::load_cmapss_fleet`), then ESA-ADB.
   Nothing above is evidence about real machines yet.
5. Matched-FLOP rather than matched-K comparison for SOS.
