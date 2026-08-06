# EXPLAIN — how the pipeline actually works

A walkthrough of the time-series PoC in plain language, with diagrams and real
numbers taken from actual runs.

This is the "what is going on here" document. Two companions:
`CLAUDE.md` is the *contract* (what must always be true), `hands_off.md` is the
*state of play* (what is measured, what is broken, what to do next). This file
is neither — it explains the machine.

---

## Contents

1. [The question in one page](#1-the-question-in-one-page)
2. [The one idea: a circuit is a calculator](#2-the-one-idea-a-circuit-is-a-calculator)
3. [Stage 1 — from engines to windows](#3-stage-1--from-engines-to-windows)
4. [Stage 2 — choosing the shape](#4-stage-2--choosing-the-shape)
5. [Stage 3 — the circuit itself](#5-stage-3--the-circuit-itself)
6. [Stage 4 — the questions you can ask](#6-stage-4--the-questions-you-can-ask)
7. [The two models](#7-the-two-models)
8. [The experiment pipeline](#8-the-experiment-pipeline)
9. [How to run it](#9-how-to-run-it)
10. [Where the traps are](#10-where-the-traps-are)

---

## 1. The question in one page

You have a fleet of jet engines. Each one has ~15 sensors, read once per flight
cycle, until it fails. Two questions:

```
   DETECTION            "is this engine behaving abnormally right now,
                         and WHICH sensor is responsible?"

   PROGNOSIS            "how many cycles does this engine have left,
                         and how sure are we?"
```

Most models answer one of these with a number. This project answers both with a
**probability distribution over everything at once** — every sensor, every
timestep — and then *asks it questions*.

The bet is that one honest distribution beats several separate predictors,
because you can ask it things no predictor can answer:

```
  "how surprising is this window?"                  -> detection
  "how surprising is sensor 5 GIVEN the others?"    -> localisation
  "what if sensors 3, 7, 11 are dead?"              -> integrate them out, exactly
  "how long left, as a full distribution?"          -> prognosis
```

All four are the **same object**, queried differently. No retraining between
them.

The catch: for that to work, the distribution has to be **exactly computable**.
Not sampled, not approximated. That is what a probabilistic circuit buys you,
and what the four structural properties in `CLAUDE.md` are protecting.

---

## 2. The one idea: a circuit is a calculator

A probabilistic circuit is a little computation graph made of three kinds of
node. That is genuinely all it is.

```
        LEAF                 PRODUCT                    SUM
   one variable          combine variables         combine options
   ("sensor 3 is         ("sensor 3 AND            ("mode A or mode B",
    normal-ish            sensor 4 together")       weighted)
    around 0.2")

    ┌───────┐              ┌───────┐                 ┌───────┐
    │  N()  │              │   ×   │                 │   +   │
    └───────┘              └───┬───┘                 └───┬───┘
        x₃                 ┌───┴───┐               ┌─────┴─────┐
                        ┌──┴──┐ ┌──┴──┐         w₁ │       w₂  │
                        │ x₃  │ │ x₄  │         ┌──┴──┐   ┌────┴┐
                        └─────┘ └─────┘         │ ... │   │ ... │
                                                └─────┘   └─────┘
```

Evaluate it bottom-up and the number at the root is `p(x)` — the density of the
whole window. That is the entire forward pass.

### Why the rules matter

Two rules make this calculator able to answer *more* than just `p(x)`:

```
 SMOOTHNESS       every branch of a SUM covers the same variables
                  (you can't average "apples" with "apples and oranges")

 DECOMPOSABILITY  the branches of a PRODUCT cover DIFFERENT variables
                  (no variable appears twice in one multiplication)
```

Obey both and something remarkable follows: **to integrate a variable out, you
just replace its leaf with a 1.** No integral is ever computed at run time.

```
   want p(x₀, x₁) with x₂ marginalised out?

         +                              +
         │                              │
         ×             becomes          ×
      ┌──┼──┐                        ┌──┼──┐
     x₀  x₁  x₂                     x₀  x₁  1.0     <- leaf returns 1
                                                       instead of its density
```

That one trick is why "what if these sensors are dead?" costs a forward pass
instead of a new model. Every exotic query later in this document is a variation
on it.

Two more properties are needed for some queries and are also maintained:
**determinism** (for MPE / most-likely-explanation) and **structured
decomposability** (for the squared/SOS mode, and for anything comparing two
circuits). See `CLAUDE.md`.

---

## 3. Stage 1 — from engines to windows

```
  raw fleet                    windows                      task
  ─────────                    ───────                      ────

  engine 1  ▁▂▃▄▅▆▇█ fail      ┌────────┐
  engine 2  ▁▂▃▄▅ (censored)   │ 20 rows│ ─┐
  engine 3  ▁▂▃▄▅▆▇█ fail      │ 15 cols│  │  X_train  (N, 300)
    ...                        └────────┘  ├─ X_test   (M, 300)
  100 engines                   sliding    │  y_test   0/1
                                  ↓ ↓ ↓    │  affected  which channels
                              ┌────────┐  ─┘
                              │ 20 rows│
                              └────────┘
```

### The flattening — the one indexing rule to remember

A window is 20 timesteps × 15 sensors. It gets flattened into one vector of 300
numbers, **row-major**:

```
                     feature index = t · C + c
                                     ↑       ↑
                                  timestep  channel

   t=0:  [ c0  c1  c2 ... c14 ]   -> features   0.. 14
   t=1:  [ c0  c1  c2 ... c14 ]   -> features  15.. 29
   t=2:  [ c0  c1  c2 ... c14 ]   -> features  30.. 44
    ...
   t=19: [ c0  c1  c2 ... c14 ]   -> features 285..299
```

Everything downstream assumes this. The chain structure assumes it, the
per-channel queries assume it, the layout controls exist to test it.

### The two sources, one interface

```
  simulate_fleet()          load_cmapss_fleet()      load_ncmapss_fleet()
  synthetic AR(1)           real NASA C-MAPSS        real N-C-MAPSS
        │                          │                        │
        └──────────────┬───────────┴────────────────────────┘
                       ▼
                    Fleet            series, rul, regime, health, censored
                       │
              ┌────────┴────────┐
              ▼                 ▼
         make_ad_task     make_rul_task
```

Same code path for real and synthetic — deliberately. "Does it hold on real
data?" has to be a config change, or the two answers are not comparable.

### Labels, and where they come from

For **detection**, real turbofan data has no anomaly labels, so anomalies are
*injected* into held-out healthy windows. Five kinds, in two families:

```
  MARGINAL — a channel visits values it shouldn't.
             A per-channel detector can see these.

    spike     one timestep, 1-2 channels, ±4σ      ▁▁▁█▁▁▁
    offset    step change from time t onward        ▁▁▁▄▄▄▄
    drift     slow ramp across a group of channels  ▁▂▃▄▅▆▇

  STRUCTURAL — every channel's own distribution is left EXACTLY intact,
               only the RELATIONSHIPS break.  A per-channel detector
               cannot see these, even in principle.

    decouple  shuffle one channel along time        ▁▂▃▄  ->  ▃▁▄▂
              (same values, same histogram, order destroyed)

    desync    replace a channel with the same channel from
              ANOTHER healthy engine
              (a perfectly normal signal, in the wrong company)
```

The structural pair is the whole justification for modelling the joint
distribution. `tests/test_ad_diagnostics.py` checks the claim is literally
true — that `decouple` leaves the sorted values of each channel unchanged.

Windows in the ambiguous middle of life are **discarded** rather than labelled,
because inventing a label there is the benchmark flaw that Wu & Keogh fault this
whole literature for.

---

## 4. Stage 2 — choosing the shape

Before there is a circuit there is a **shape**: which variables get multiplied
together, and in what order. Two equivalent vocabularies:

```
   VTREE                          REGION GRAPH
   binary, one split per node     n-ary, can have several splits

        {0,1,2,3}                      {0,1,2,3}
         /     \                        /      \
     {0,1}     {2,3}                {0,1}      {2,3}
      / \       / \                  / \        / \
     0   1     2   3                0   1      2   3
```

The region graph is the general one, and it is what the code uses internally
(`region_graph_from_vtree` converts).

### The chain (the one that keeps winning)

For a window, the natural shape is a chain — one link per timestep:

```
  window = 3 timesteps, 2 channels    features: t·2 + c

   {0,1,2,3,4,5}                     ← the whole window
        │
        ×──────────────┐
        │              │
    {0,1}         {2,3,4,5}          ← timestep 0  |  the rest
     │                 │
     ×                 ×───────┐
   ┌─┴─┐               │       │
  {0} {1}           {2,3}   {4,5}    ← timestep 1  |  timestep 2
                     │        │
                     ×        ×
                   ┌─┴─┐    ┌─┴─┐
                  {2} {3}  {4} {5}
```

Read it as an HMM: each timestep is emitted from a hidden state, and the states
are chained. The regions above are printed straight out of
`chain_region_graph(3, 2)`.

> **What we now know (`hands_off.md` §B.3):** the chain's advantage is *not*
> that it is chain-shaped. Permuting the timestep ORDER costs ~0.07 nats —
> nothing. Permuting ALL features, which also breaks the "channels of one
> timestep stay adjacent" grouping, costs ~5 nats. So the win is the
> **blocking**. The vtrees `chain_perm_blocks` and `chain_perm_features` exist
> to keep measuring this: identical circuit, identical parameter count (912 in
> the test case), only the variable→position map differs.

### The menu

| shape | idea |
|---|---|
| `chain`, `chain_grouped`, `chain_full` | HMM-shaped, hand-built |
| `time`, `channel`, `channel_groups` | balanced splits by time or by sensor |
| `chow_liu` | learned from mutual information |
| `spectral`, `spectral_rg` | learned by spectral clustering of the MI graph |
| `orc_rg`, `forman_rg` | learned by **graph curvature** (the research bet) |
| `random` | the control — is structure doing anything at all? |
| `chain_perm_blocks`, `chain_perm_features` | the layout controls (§B.3) |

Every one of them yields a valid vtree or region graph, so **all four circuit
properties hold no matter which you pick**. The structural constraint is free;
the choice is purely about fit.

---

## 5. Stage 3 — the circuit itself

Take the shape, and put `K` computational units at every region:

```
  region {2,3,4,5}          K = 4 units, each a SUM node
    ┌─────┬─────┬─────┬─────┐
    │ +₁  │ +₂  │ +₃  │ +₄  │
    └──┬──┴──┬──┴──┬──┴──┬──┘
       └─────┴──┬──┴─────┘
                │  every unit sums over the SAME shared list of products
       ┌────────┴────────┐
       │  ×  ×  ×  ×     │   products pairing units of {2,3} with units of {4,5}
       └─────────────────┘
```

Think of `K` as "how many different behaviours can this part of the window be
in". More units = more expressive, more parameters, more chance of overfitting.

### Why it is a DAG and not a tree

Because those product nodes are **shared**. If every unit built its own copy of
its children, the circuit would be `K^depth` nodes — exponential. Sharing makes
it linear:

```
   TREE (naive)                 DAG (what we do)

      +                              +
    ┌─┴─┐                          ┌─┴─┐
    ×   ×                          ×   ×
   ┌┴┐ ┌┴┐                          \ /
   × × × ×                           ×          ← one node, two parents
  ...........                        │
  K^depth nodes                   K·depth nodes
```

This is what `RegionGraphPC` builds. `bench_scaling.py` measures the difference.

> **Trap.** Because the graph is shared, `nn.Module.to(device)` walks every
> shared node once per path and is **exponential in depth** — a 0.2 s fit became
> 400 s, silently, with correct results. Use `move_circuit_()`. Never `.to()`.

### The leaves

| leaf | for | notes |
|---|---|---|
| `GaussianLeaf` | a sensor value | median/MAD init, **width floor** |
| `GaussianMixtureLeaf` | a sensor with several modes | quantile init, same floor |
| `CategoricalLeaf` | the RUL bin τ | has a closed-form interval — required for survival queries |
| `InputNode` | heavy-tailed default | Gaussian/Laplace/Student-t mixture |

The **width floor** deserves a note, because it caused two of the project's
silent bugs. A leaf fitted to a near-constant sensor wants width → 0, which
makes its density → ∞ and NaNs the whole circuit on the first gradient step.
So every leaf's width is floored at `max(1% of the feature's own spread, 1e-3)`,
at initialisation *and* during training. On real FD001 a whole sensor channel is
median-constant, so this path is exercised on every run, not in the corner.

### Two evaluators, same numbers

```
  RECURSIVE                        LAYERED (compiled, default)
  one Python frame per NODE        one per LAYER
  10³–10⁵ frames per step          ~16 frames per step
  the reference                    4–36× faster; GPU wins above batch ~128
```

The compiled one is gated against the recursive one on a real batch at fit time,
so a fast-but-wrong evaluator cannot slip through.

> **Trap.** The compiled evaluator holds its **own copy** of the parameters.
> `write_back()` syncs compiled → DAG; nothing syncs DAG → compiled. So editing
> the DAG after `fit` is silently ignored. Call `pc.pc.use_recursive()` first.

---

## 6. Stage 4 — the questions you can ask

This is the payoff. **One trained circuit, many questions.**

```
                        ┌──────────────────────┐
                        │   trained circuit    │
                        │      p(x)            │
                        └──────────┬───────────┘
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   log_prob       log_marginal  log_box    conditional    chain rule
   "how likely"   "ignore some" "in a range" "given rest"  "who is to blame"
        │              │           │           │              │
   detection      dead sensors  survival   localisation   attribution
```

### Query 1 — density (detection)

`score(x) = −log p(x)`. Higher = weirder. That is the whole detector.

### Query 2 — exact marginal (dead sensors)

Three sensors fail on the aircraft. A reconstruction model has to *impute* them
— invent values, then be surprised by its own inventions. The circuit simply
takes them out of the question:

```
  score_with_missing(X, dead_channels=[3, 7, 11])
       └─> replaces those leaves with 1.0 and evaluates
           = −log p(everything else), exactly
```

Measured on real data, the exact-marginal route beats mean-imputation by a wide
margin (CRPS 8.64 vs 16.00 in one recorded run). This is a query a
reconstruction-based detector cannot express at all.

### Query 3 — the typed decomposition (localisation)

For each channel `c`, two views, both exact:

```
   marginal_c    = −log p(x_c)             "is this sensor odd on its own?"
   conditional_c = −log p(x_c | x_−c)      "is it odd GIVEN the others?"
   structural_c  = conditional_c − marginal_c
```

Real output from a trained model (window 8, 6 channels), one window of each kind:

```
--- normal   (truth: nothing)          −log p(x) = 44.8
                ch0   ch1   ch2   ch3   ch4   ch5
  marginal      9.2  14.4   9.6  12.5   8.8   8.6
  conditional   3.2  16.9   3.7   6.5   2.0   4.9
  structural   -6.1   2.5  -5.9  -6.0  -6.8  -3.7

--- spike    (truth: channel 5)        −log p(x) = 88.3
                ch0   ch1   ch2   ch3   ch4   ch5
  marginal      9.1   8.1   9.5  17.7   9.5  57.9   <- ch5 screams
  conditional   2.4   9.1   5.4  11.2   2.3  51.0   <- and still screams
  structural   -6.8   1.0  -4.1  -6.5  -7.2  -6.9
```

Read the spike row: channel 5's marginal is 57.9 against a typical 9 — the
univariate view finds it immediately, and `structural` is unremarkable because
nothing about the *relationships* broke. That is the correct answer, and it is
why the project does **not** claim the structural view wins everywhere. On
plain spikes a per-channel z-score reaches AUROC 0.996 and deserves to.

The structural view earns its keep on `desync`, where a perfectly normal signal
appears in the wrong company. There it is a *statistical* effect across many
windows rather than something you see in one:

```
  desync detection, by view (AUROC vs normal windows)
     marginal     0.507      <- chance, BY CONSTRUCTION
     conditional  0.636
     structural   0.640
```

Note the honesty of `marginal ≈ 0.5`: the injection was designed to leave the
marginal untouched, and it does.

### Query 4 — attribution with a guarantee

The chain rule of probability says, for **any** ordering of the variables:

```
  log p(x) = Σᵢ log p(x_πᵢ | x_π₁ … x_πᵢ₋₁)
```

Every term is a difference of two exact marginals of this circuit. So the
per-feature contributions **sum to the score exactly**. Completeness is a
theorem here, not something you estimate and hope for.

Real numbers from a 6-feature circuit:

```
  chain-rule contributions
    [-0.708, -1.521, -0.703, -1.835, -1.000, -1.009]
                                              sum = -6.7766
    log p(x)                                      = -6.7766     ✔ exact
```

In the full runs the residual is ~1.5e-5 nats — float32 rounding, nothing more.

Compare with sampling-SHAP on an autoencoder, the standard alternative: it
scored **0.4975** localisation AUROC (chance) at 32 samples/channel, and
**0.5152** at 128. Quadrupling the budget bought nothing. The circuit's exact
conditional route scores **0.902**. That gap is the project's actual
contribution.

### Query 5 — box queries (survival)

For a *range* rather than a point:

```
   P(τ ≥ 40 cycles | x)
```

With τ discretised into ordinal bins and given a `CategoricalLeaf`, this is a
suffix sum over the bins — exact, one pass, no CDF approximation:

```
   bins:   0   1   2   3   4   5   6   7   8   9
           ░   ░   ░   ░  ███ ███ ███ ███ ███ ███
                          └──────── P(τ ≥ 4) ────┘
```

That is what makes right-censored training possible (§7), and it is checked
against the pmf suffix sum to 6e-6 nats in `tests/test_rul_diagnostics.py`.

---

## 7. The two models

Both are thin wrappers over the same circuit machinery.

### `WindowPC` — the density over a window

```
  X (N, window·C)
      │
      ├─ build region graph from `vtree_method`
      ├─ K units per region, jittered weights
      ├─ fit leaves (median/MAD, floored)
      ├─ compile + gate against the recursion
      ├─ train:  minimise  −mean log p(x)
      └─ guardrail:  assert_informative()
             ├─ is the score constant in x?          -> refuse
             └─ does it IGNORE whole channels?       -> refuse
```

### `SurvivalPC` — the joint density over (window, τ)

One extra variable, τ = the remaining-life bin, attached low in the structure:

```
   window features 0 .. 299        τ = feature 300
        └────────────┬──────────────────┘
                     joint p(x, τ)
```

The training objective is where the idea lives:

```
   engine we watched FAIL at bin k     ℓ = log p(x, τ = k)
   engine still ALIVE when we stopped  ℓ = log P(x, τ ≥ c)   ← a box query
                                                (not a guess, not a hazard head)
```

Both terms come from the same circuit and the same normalisation, so censored
and uncensored engines sit on one likelihood scale.

At test time:

```
   predict(X) ─┬─> pmf         full distribution over bins
               ├─> mean, mode  point estimates
               ├─> q05, q95    interval, as bin CENTRES
               └─> q05_edge, q95_edge   the same bins, as EDGES
```

> **Read the edges for coverage.** The centre/edge distinction is not cosmetic:
> scoring bin centres against a target measured in cycles throws away half a bin
> at each end, which is what produced the project's "PICP 0.38–0.52" scare.
> On real C-MAPSS the same predictions score **0.35–0.41 on centres and
> 0.93–0.98 on edges** (`hands_off.md` §B.2).

---

## 8. The experiment pipeline

One YAML → many runs → one table.

```
   config/ts/cmapss_ad.yaml
            │
            │  grid: / variants:  ×  seeds:
            ▼
   ┌──────────────────┐
   │     runner       │  resumable · isolated · self-describing
   └────────┬─────────┘
            │  one run = one variant × one seed
            ▼
   ┌──────────────────┐
   │   pipeline       │  stages: ad | explain | rul | calibration | scaling
   └────────┬─────────┘
            ▼
   logs/ts/<experiment>/<variant>/seed<N>/
        ├── config.json     the RESOLVED config (no guessing later)
        ├── env.json        git commit + dirty flag, host, GPU, thread counts
        ├── run.log         the full console transcript
        ├── history_*.csv   per-epoch training curves
        ├── metrics.json    scalars
        ├── results.jsonl   one comparable row per method
        ├── status.json     config hash -> skip if already done
        └── artifacts/      scores.npz, attributions.npz, rul_pred_*.npz
            │
            ▼
   aggregate.py  ->  summary.md / summary.csv   (mean ± sd over seeds)
```

### The five stages

| stage | what it measures |
|---|---|
| `ad` | detection vs the full baseline suite, + the dead-sensor query, + the typed split |
| `explain` | localisation vs ground truth, completeness (a theorem), faithfulness (deletion curves) |
| `rul` | censoring ablation, accuracy vs ridge/MLP/CQR, survival under partial evidence |
| `calibration` | split conformal on the circuit's own predictive, **split by engine** |
| `scaling` | tree vs DAG layout |

### Four properties worth relying on

```
  RESUMABLE       a finished run whose config hash matches is skipped.
                  A crash at hour 9 of 12 costs nothing.

  ISOLATED        one variant blowing up is a recorded FAILURE, and the batch
                  continues.  A degenerate model is a failed run, never a row
                  of numbers.

  SELF-DESCRIBING every run stores its resolved config, git commit + dirty
                  flag, host, GPU, threads, curves, peak RSS and transcript.
                  (This is how we caught that the 2026-08-04 results ran at a
                  pre-bugfix commit — see §B.9 of the hand-off.)

  HONEST BY       splits are always by ENGINE, never by window: overlapping
  CONSTRUCTION    windows of one engine are near-duplicates, and calibrating
                  on them reports a coverage that evaporates on a new engine.
```

### Tiers

```
  tier 0   smoke — wiring only, ~3 min.  Its numbers mean nothing.
  tier 1   real C-MAPSS: detection + explanation      ← the headline
  tier 2   real C-MAPSS: calibration + RUL
  tier 3   real N-C-MAPSS
  tier 4   synthetic controls + layout scaling
  tier 5   ablations: structure (real data), capacity
```

Tiers are ordered so that an interrupted night still ran the tiers that decide.

---

## 9. How to run it

```bash
export PYTHONPATH=.

# 0. the fast checks — 17 seconds, run these BEFORE a batch, not after
pytest tests/test_ad_diagnostics.py tests/test_rul_diagnostics.py \
       tests/test_experiment_hygiene.py -q

# 1. wiring check, every stage, a few epochs — ~3 min
bash poc/time_series/launch/run_smoke.sh

# 2. one experiment
bash poc/time_series/launch/run_config.sh config/ts/cmapss_ad.yaml

# 3. the real batch
TIERS="1 2" JOBS=3 bash poc/time_series/launch/run_workstation.sh

# 4. read it
python -m poc.time_series.aggregate logs/ts --recursive
```

Overrides without editing configs:

```bash
python -m poc.time_series.runner config/ts/cmapss_rul.yaml \
       --seeds 0 1 2 3 4 --set model.K=10 --only vtree-chain --device cpu
```

**Device note.** A circuit is thousands of *small* operations, so it is
launch-latency bound. `DEVICE=cpu JOBS=3` often beats one CUDA process; the GPU
pays off as `K`, the window and the batch grow — and only with the layered
evaluator. `env.json` records which one actually ran.

---

## 10. Where the traps are

The full list lives in `hands_off.md` §6. The ones that have actually cost a
wrong answer:

```
  ✗ .to(device) on a circuit          exponential in depth.  use move_circuit_()
  ✗ editing the DAG after fit         the compiled copy shadows it.
                                      call pc.pc.use_recursive() first
  ✗ weight_jitter = 0                 every region collapses to one component
                                      (now refused at construction)
  ✗ tau_where = "root"                predictive collapses to a constant
                                      at most K (now caught by a guardrail)
  ✗ q05/q95 for coverage              they are bin CENTRES.  use q05_edge/q95_edge
  ✗ a flag that switches two things   the bug SHAPE behind five of six failures
```

That last one is the real lesson and it is worth stating plainly:

> **Every wrong answer this project has produced came from a comparison where
> two things changed at once, and nobody checked the thing that was supposed to
> stay fixed.** A flag that gated two settings. A "control" arm that used a
> different kind of leaf. A metric measured in the wrong units. A re-run on a
> different commit *and* a different machine.
>
> The training loss never revealed any of them. Only a query did — after the
> conclusion had been drawn.

That is why `tests/test_ad_diagnostics.py`, `tests/test_rul_diagnostics.py` and
`tests/test_experiment_hygiene.py` exist. They do not test whether the code
runs. They test whether **the comparison you are about to make means anything**.

Seventeen seconds. Run them first.
