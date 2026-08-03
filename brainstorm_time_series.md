# Brainstorm — Time Series: Anomaly Detection + Remaining Useful Life on Industrial & Spacecraft Data

**Date:** 2026-07-30
**Status:** research / scoping document. **No code written.** This is the "what is needed and what is the novelty" pass requested before implementation.
**Basis:** `src/probabilistic_circuits.py` (1803 LOC, read in full), `src/directions.py`, `src/datasets.py`, `src/experiment.py`, plus a literature sweep (sources at the end).

---

## 0. Executive summary — the one-paragraph version

Extending the existing exact-PC stack to time series is **mostly a data-layer + structure-layer job**, with one hard blocker (`DensityPC` builds a *tree*, not a DAG, so parameters blow up as `K^depth` — infeasible for windowed multivariate input). The **anomaly-detection half is a re-skin**: windows are just vectors, and the current density/marginal machinery applies unchanged. The **RUL half is where the actual novelty is**, and it is not incremental: a joint circuit over `(window, time-to-failure)` answers *both* `−log p(x)` (detection) and `P(τ > t | x)` (survival / prognosis) as **exact queries on one density**, and — the part nobody appears to have published — **right-censored units can be trained on with an exact likelihood**, because the censored contribution `log P(τ > c | x)` is precisely an axis-aligned box integral, which smooth + decomposable circuits compute in closed form. Searches for `probabilistic circuits ∩ RUL`, `SPN ∩ hazard/survival/time-to-event`, and `PC ∩ prognostics` returned **nothing**. That is the strongest unclaimed nugget found in this sweep, and unlike the multimodal headline (parked/KILL per the 2026-06-25 evaluation) it does not collide with the PC-FUSION line.

**Recommendation:** treat time series as **the flagship evidence domain for Paper A ("AD as a tractable query")** rather than a new project, with the exact-censored-survival query (T1) as the headline contribution and the time×channel vtree ablation (T4) absorbing the Paper C / curvature kill-shot — time series is the *first* domain in the portfolio where the vtree has a **ground-truth structure** (time adjacency) to be checked against, which the tabular experiments structurally cannot provide.

---

## 1. Inventory: what the existing code already gives us, for free

Read against `src/probabilistic_circuits.py`. Nothing in this column needs to be rebuilt.

| Component | File location | Transfers to TS as-is? |
|---|---|---|
| Vtree types, save/load, LCA depth | `probabilistic_circuits.py:51–152` | ✅ yes — a vtree over `w·C` window variables is still just a vtree |
| Vtree learners: `chow_liu`, `spectral`, `orc`, `forman`, `random` (`learned_vtree`) | `:294–724` | ✅ yes, and **more meaningful here** than on tabular (see T4) |
| Consensus vtree / co-grouping (`cogroup_matrix`, `consensus_vtree`) | `:190–248` | ✅ yes — becomes cross-machine / cross-mission structure transfer |
| Leaves: Gaussian, GaussianMixture, Categorical, heavy-tailed `InputNode` | `:731–954` | ✅ yes; heavy-tailed `InputNode` is *well* suited to sensor telemetry spikes |
| `DensityPC` (monotone, structured-decomposable) | `:1038–1128` | ⚠️ semantics yes, **scale no** (§3.1 blocker) |
| Exact `log_prob`, `log_marginal`, `log_partition`, `mpe` | `:1135–1221` | ✅ yes — `log_marginal` is the missing-sensor story (T2) |
| `SquaredPC` (SOS, subtractive, exactly normalised) | `:1552–1690` | ✅ yes; box-integral extension is closed-form for Gaussian leaves (§3.3) |
| 4 property validators | `:1417–1487` | ✅ yes, unchanged |
| Cardinality routing / selection marginals (ProbMoE) | `:1253–1355` | ✅ yes — and gets **ground-truth labels** for the first time (T7) |
| Two directions + shared `AnomalyDetector` interface, NLL + contrastive training | `directions.py` | ✅ yes |
| `add_modality` (structure transfer, no retraining of existing experts) | `directions.py:571–618` | ✅ yes — becomes "add a new machine / new mission" |
| Config + cluster harness (`config/`, `cluster_scripts/`, `logging_utils`) | — | ✅ yes, reuse verbatim |

**Conclusion:** roughly 70% of the machinery exists. The missing 30% is concentrated in three places: (i) scalability of the circuit builder, (ii) interval/CDF queries on leaves, (iii) a time-series data layer.

---

## 2. Problem formalisation — both tasks as circuit queries

This is the framing that makes the whole thing one model instead of two.

Let a window be `x = (x_{t−w+1..t}) ∈ ℝ^{w×C}` for `C` channels, flattened to `d = w·C` variables. Let `τ ∈ ℝ₊` (or a discretised bin index) be the time-to-failure at the window's right edge. Train **one** circuit over the joint scope `V = {window vars} ∪ {τ}` (plus optionally regime/unit covariates).

| Task | Query | Exact under which properties? |
|---|---|---|
| **Detection** (unsupervised AD) | `s(x) = −log p(x) = −log ∫ p(x, τ) dτ` | smoothness + decomposability |
| **Detection with dead/missing sensors** | `−log p(x_O)` for observed subset `O` | smoothness + decomposability (already implemented: `log_marginal`) |
| **Channel/time attribution** | `log p(x_c | x_{−c}) = log p(x) − log p(x_{−c})` | same — two marginals, one subtraction |
| **Typed anomaly** (marginal vs. structural) | `log p(x_c)` vs. `log p(x_c | x_{−c})`; interaction info `log p(A,B) − log p(A) − log p(B)` | same |
| **RUL point/mean** | `E[τ | x] = ∫ τ p(τ | x) dτ` | needs a moment query — exact for Gaussian/GM leaves, trivial for categorical bins |
| **RUL predictive density** | `p(τ | x) = p(x, τ) / p(x)` | ratio of two exact marginals ⇒ exact |
| **Survival / failure-before-horizon** | `S(t | x) = P(τ > t | x) = [∫_t^∞ p(x,τ)dτ] / p(x)` | **box (interval) query** — exact if leaves expose a CDF |
| **Censored-unit training** | `ℓ = log P(τ > c | x)` for a unit that survived past `c` | same box query ⇒ **exact censored likelihood, no approximation** |
| **Regime / unit localisation** | selection marginals `m_j(x)`, cardinality posterior `P(k|x)` | already implemented; ProbMoE-exact |
| **Imputation of a failed sensor** | MPE over missing channels given evidence | needs **determinism** — caveat, see §6 |

The economically important row is the last-but-few: **censored training**. Every real prognostics dataset is dominated by units that did not fail during observation (and in C-MAPSS the *entire test split* is right-censored by construction — trajectories are truncated before failure and a ground-truth RUL is supplied). Conventional deep RUL regressors handle this by *discarding* censored units, by heuristic pseudo-labels, or by a Cox/Weibull head with its own likelihood. A circuit handles it by evaluating one more exact query on the same object it already trained.

---

## 3. Gap analysis — what has to be built

### 3.1 BLOCKER: `DensityPC` is a tree, not a DAG — parameters explode

`DensityPC._build` (`:1065–1074`) creates, for each of the `K = n_sum_components` product components at a vtree-internal node, a **freshly recursed left and right subtree**. `notes.md:75` states this explicitly as a design decision ("`n_sum_components^depth` leaf instances per feature — more parameters but simpler gradient flow").

Arithmetic for a *modest* time-series window: 14 sensors × 8 downsampled steps = `d = 112`, balanced depth ≈ 7, `K = 3` ⇒ `3^7 = 2187` copies × 112 ≈ **245k leaf modules**. A realistic window (25 channels × 100 steps = 2500 dims, depth 12) is `3^12 ≈ 5.3×10^5` copies per feature — not merely slow, structurally impossible.

**Required fix (does not touch any of the four properties):** rebuild the circuit as a **region graph / layered DAG**, the standard RAT-SPN / EiNet layout:

- each vtree node owns `K` *units* (sum nodes) over its scope, not `K` independent subtrees;
- a product layer pairs left-units × right-units (`K²` products, or `K` under matched-index pairing);
- each parent sum mixes those products.

Smoothness holds (all units at a node share the node's scope), decomposability holds (left/right scopes disjoint by the vtree), **structured** decomposability holds (every product splits exactly at a vtree node). Cost becomes `O(d · K²)` instead of `O(d · K^depth)`. The existing `validate_*` helpers work unchanged because they are type-driven and already memoise by `id(node)` (`:1358–1414`) — they were written for DAGs.

This is the single largest engineering item and it **benefits the tabular/image work too**. It should be done first, independently of the time-series decision.

### 3.2 Interval / CDF queries on leaves (enables everything RUL)

Add to the `LeafNode` contract (`:731–769`):

```
log_cdf(v)                      -> log F(v)
log_interval(lo, hi)            -> log ∫_lo^hi f
pair_log_interval(other, lo,hi) -> (log|∫_lo^hi f·g|, sign)     # SOS mode
```

- `GaussianLeaf`: `Φ` via `torch.special.ndtr` / `torch.distributions.Normal.cdf`. Trivial.
- `GaussianMixtureLeaf`: mixture of `Φ`s. Trivial.
- `CategoricalLeaf`: partial sum over logits. Trivial — and this is the **cheapest correct route for `τ`**: discretise RUL into ordinal bins and the survival function is literally a suffix sum of a categorical leaf, exact by construction, no CDF subtleties, and it sidesteps the "RUL is non-negative and heavy-tailed" modelling headache.
- `InputNode` (Gaussian/Laplace/Student-t mixture, `:910–953`): all three have closed-form CDFs (`Φ`, Laplace CDF, Student-t via regularised incomplete beta / `torch.distributions.StudentT.cdf`).
- **SOS mode**: worth noting that this *also* stays exact. The product of two Gaussians is a scaled Gaussian — `N(v;μ₁,σ₁²)N(v;μ₂,σ₂²) = N(μ₁;μ₂,σ₁²+σ₂²)·N(v;μ*,σ*²)` with `σ*² = (σ₁^{-2}+σ₂^{-2})^{-1}` — so `pair_log_interval` is just the existing `pair_log_integral` (`:800–804`) times a `Φ` difference. Squared circuits therefore support exact survival queries too. Non-obvious, worth a sentence in a paper.

Then a module-level `eval_log_box(root, lo, hi, evidence)` mirroring `eval_log_marginal` (`:1135–1157`): at a leaf, return `log_interval` for boxed vars, `log_prob` for observed vars, `0` for fully marginalised vars; product sums, sum log-sum-exps. ~30 lines.

### 3.2b Region graphs, and the chain/HMM structure (added 2026-08-02 after the PoC)

The PoC changed the picture here, so this section supersedes the "just pick a
vtree" framing below.

**A vtree forces two things that are assumptions, not requirements:** every
scope splits in exactly *two*, and in exactly *one* way. A **region graph**
drops both — a region is a scope, a partition is a tuple of disjoint child
regions, and a region may carry several alternative partitions. A vtree is the
binary single-partition special case. This matters concretely for curvature:
deleting negatively curved edges naturally disconnects a graph into k ≥ 2
components, and forcing that back into a binary vtree throws the k-way
structure away. Implemented as `RegionNode` / `curvature_region_graph`, with
the arity chosen by the data (keep cutting while the next edge still has
negative curvature — i.e. while it is still a bottleneck).

Property bookkeeping, which is the whole trade: smoothness and decomposability
hold for **any** region graph, so exact density, marginals and box/survival
queries are unaffected. **Structured decomposability holds iff every region has
one partition** — so multi-partition region graphs are more expressive but give
up SOS and circuit multiplication. `SquaredPC` enforces this and refuses
multi-partition input with an explanation.

**The bigger finding: window structure is order-blind, and that is fixable.**
A balanced vtree over a flattened window makes the circuit a finite mixture of
factorised distributions, so permuting the timesteps barely changes the
density. The PoC's `decouple` anomaly (permute one channel along time) sat at
chance. Two fixes, which compose:

1. **First differences** (`delta_window_transform`): model (x₁, Δx₂, …, Δx_w).
   The map is unit lower-triangular, so |det J| = 1 — the density stays exactly
   normalised with *no* Jacobian correction and log p stays comparable.
2. **A chain region graph** (`chain_region_graph`): a right-linear caterpillar
   over time, R_t = {t..w} split as ({t}, R_{t+1}). This makes the circuit
   **literally an HMM** — the K units of R_t are the K hidden states at time t,
   the sum weights between regions are the transition matrix, the leaves are
   state-conditional Gaussian-mixture emissions, and circuit evaluation *is*
   the forward algorithm. Every timestep sits at a different depth, so the
   density is not permutation-invariant.

This is the tractable analogue of an LSTM: same recurrent shape, but a discrete
K-valued latent instead of a continuous state vector — exactly the substitution
that buys exact inference. Expressiveness scales with K; note that stacking
independent chains (a factorial HMM) is **not** tractable, so that is not the
way to grow it. Related work to cite: Dynamic SPNs (Melibari et al., PGM 2016)
are the template-unrolled version of this, and the "why not just use a DSPN?"
question now has a concrete answer — we need a joint over (window, τ) and
structured decomposability, which template unrolling does not give.

**Empirically the chain is the best structure found so far** (AUROC 0.9315 vs
0.9108 for a balanced temporal vtree, and `decouple` from ~0.50 to 0.65), which
promotes "structure design" from an ablation (T4) to a first-class part of the
contribution.

### 3.3 Temporal vtree constructors

New builders alongside `chow_liu_vtree` / `spectral_vtree` / `curvature_vtree`:

- `dyadic_time_vtree(w, C)` — split time recursively, channels innermost (locality in time);
- `channel_major_vtree(w, C)` — split channels first, then time (locality in channel);
- `interleaved / hierarchical` — channel groups × dyadic time (physically motivated: turbine sensors grouped by sub-component — fan/LPC/HPC/LPT/HPT in N-CMAPSS);
- pass-through to the existing learned methods on the flattened `w·C` MI matrix.

This set *is* the T4 experiment (§5).

### 3.4 Data layer — `src/datasets_ts.py`

Mirror `datasets.py`'s `AnomalyDataset` with a `TimeSeriesDataset` / `RunToFailureDataset`:

```
TimeSeriesDataset:  X (T×C), y (T,) point labels, unit/segment ids, split protocol
RunToFailureDataset: per-unit trajectories, τ per timestep, censoring flag δ, regime id
```

Needs: per-unit normalisation (fit on the unit's early/healthy portion, **not** globally — leakage), sliding-window extraction with an explicit `stride`, and a **no-overlap guarantee between train and test windows** (overlapping windows across a split boundary is a classic silent leak in this literature).

### 3.5 Evaluation layer

**Detection.** Do *not* use point-adjusted F1. The field has moved: TSB-AD (NeurIPS 2024 D&B, "The Elephant in the Room") removes point adjustment entirely and identifies **VUS-PR** as the most reliable measure; use VUS-PR / VUS-ROC + AUC-PR + event-wise (affiliation / range-based) metrics. ESA-ADB ships its own evaluation pipeline and novel metrics — use theirs on their data, or reviewers will (correctly) object.

**RUL.** RMSE and the NASA asymmetric score are the community's table stakes, but they only score a point estimate and therefore **throw away the entire contribution**. The probabilistic metrics are the point: CRPS on `p(τ|x)`, PICP/MPIW, reliability diagrams, and — for the censored setting — **integrated Brier score and censoring-aware concordance (C-index)**, which are exactly the metrics that make the survival framing legible to reviewers.

### 3.6 Baselines (must include the trivial ones — see §5.0)

- **Detection:** from the TSB-AD suite — Sub-PCA, windowed IForest, KMeansAD, MatrixProfile/STOMP, KNN-distance; deep: USAD, TranAD, OmniAnomaly, Anomaly Transformer; **trivial:** per-channel z-score, moving-average residual, 1-NN Euclidean distance to nearest training window. TSB-AD's headline finding is that the simple methods are extremely competitive — plan for that, don't be surprised by it.
- **RUL:** CNN/LSTM regressors, MC-dropout, deep quantile regression, **conformalised quantile regression** (the sharpest adversary for calibrated intervals), plus classical Weibull AFT / Cox for the censored comparison.

---

## 4. Datasets — concrete shortlist with the caveats that matter

### 4.1 Anomaly detection

| Dataset | Domain | Why | Caveat |
|---|---|---|---|
| **ESA-ADB** (ESA / Airbus DS / KP Labs, arXiv:2406.17826) | **spacecraft** telemetry, 3 ESA missions, years of data, hundreds of channels, expert-annotated | The credible spacecraft benchmark. Real operators, real annotations, published evaluation protocol. This is what SMAP/MSL *should* have been. | Large; needs preprocessing effort. A July 2026 deployability study (arXiv:2607.07335) already benchmarks on it — read before designing splits. |
| SMAP / MSL (NASA) | spacecraft | Universally reported ⇒ comparability | **Known flawed** (Wu & Keogh, TKDE 2021: trivial anomalies, absurd anomaly density, run-length artefacts). Report only as legacy comparison, never as evidence. |
| **TSB-AD** (NeurIPS 2024 D&B) | 1070 curated series from 40 datasets, 40 algorithms | Reliability, curated labels, VUS-PR, no point adjustment | Mostly univariate/short; use as the *methodological* anchor, not the industrial story. |
| **SMD**, **SWaT / WADI**, **PSM**, **SKAB**, **MetroPT** | server / water treatment / server / industrial rig / metro compressor | The industrial-machine half | Overused; SWaT/WADI need a licence request; see the univariate caveat below. |
| **CATS** (Controlled Anomalies Time Series) | simulated multi-system | Ground-truth *multivariate* anomalies by construction | Synthetic. |

> ⚠️ **The critique to pre-empt.** *Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate* (Pinet et al., arXiv:2606.02670, 2026) shows via distance-correlation analysis that anomalies in SMAP, MSL, SMD and SWaT are largely detectable per-channel — so a *joint* density model has little room to demonstrate its value on them. Any claim that "modelling cross-channel dependence helps" on those datasets is dead on arrival.
>
> **Turn this into an asset, don't fight it.** The exact typed decomposition in T3 (`log p(x_c)` vs. `log p(x_c | x_{−c})`) is precisely an *instrument for measuring* whether an anomaly is univariate or genuinely multivariate — an exact, information-theoretic version of what that paper does with a correlation heuristic. Reproducing their finding *with our own model as the measuring device*, and then showing the model's advantage concentrates on the subset it identifies as truly multivariate, is a much stronger paper than pretending the critique doesn't exist.

### 4.2 Remaining useful life

| Dataset | Why | Caveat |
|---|---|---|
| **C-MAPSS** FD001–FD004 (NASA) | Universal RUL benchmark. FD002/FD004 have **6 operating conditions** — the ground-truth regime label that makes T7 testable. Test split is **right-censored by construction** ⇒ perfect fit for the exact censored likelihood. | Saturated; also the standard piecewise-linear RUL cap (`R_c = 125`) is an arbitrary label hack — **our survival query does not need it**, which is itself an argument to make. |
| **N-CMAPSS** DS01–DS08 (PHM 2021 Data Challenge) | Realistic flight profiles, 47 sensors, 1 Hz, ~63M timestamps, unseen test units, fault localised to fan/LPC/HPC/LPT/HPT | Big; needs downsampling/flight-cycle aggregation. `CruiseBench` (arXiv:2607.19380, 2026) is a newer real-flight-aligned split — check it for a fairer protocol. |
| **FEMTO / PRONOSTIA** and **XJTU-SY** bearings | Different physics (vibration, run-to-failure), tests cross-domain structure transfer | High sample rate ⇒ spectral features, not raw windows |
| NASA battery / MetroPT | Cheap secondary evidence | Small |

**Spacecraft RUL** specifically: there is no clean public run-to-failure spacecraft dataset. Honest options: (a) frame the spacecraft contribution as **detection + survival-until-next-anomaly** on ESA-ADB (the survival query applies to *any* time-to-event, not only hardware death), or (b) use NASA battery degradation as the space-adjacent RUL proxy and say so plainly. Do **not** claim spacecraft RUL on C-MAPSS.

---

## 5. Novelty analysis

### 5.0 The pre-registered kill-shot (apply to every idea below)

The 2026-06-25 novelty sweep established the recurring meta-finding for this project: *every exact-query reframing lives or dies on trivial-baseline redundancy.* Time series does not exempt anything. For each idea, the redundancy adversary is named explicitly, and each must be run **before** the idea is written up, on a frozen protocol.

---

### **T1 — RUL as an exact tractable query, with exact censored likelihood** ⭐ PURSUE

**Claim.** One structured-decomposable circuit over `(window, τ)` yields, exactly and from a single trained object: the anomaly score `−log p(x)`, the full RUL predictive density `p(τ|x)`, the survival function `P(τ > t | x)`, and a **training likelihood that is exact for right-censored units** (`log P(τ > c | x)` is an axis-aligned box integral). Nothing is approximated, nothing is post-hoc calibrated.

**Why it's novel.** Literature sweep found **zero** hits for PCs/SPNs applied to RUL, prognostics, hazard functions, survival, or time-to-event. The prognostics field's current answer to calibrated uncertainty is *post-hoc conformal prediction* (arXiv:2212.14612; CQR-LSTM, Sensors 2026; bearing CP at PHME 2026) — i.e. a wrapper that repairs a miscalibrated point model. We would be offering a model whose predictive distribution is exact by construction and whose censoring handling is a *query*, not a loss-function design choice.

**The nugget** (the sentence a reviewer repeats): *"The same circuit that scores the anomaly answers 'what is the probability this unit survives another 50 cycles' — exactly, and it learned from the units that never failed."*

**Kill-shot / adversary.** Deep quantile regression + conformalised quantile regression, on CRPS + PICP/MPIW + integrated Brier score, on C-MAPSS FD001–FD004 and one N-CMAPSS subset. If conformal-CQR matches the circuit on calibration *and* beats it on RMSE, the exactness buys nothing observable and T1 collapses to a curiosity. **Pre-register:** exactness must buy either (i) better calibration in the *low-data / heavily-censored* regime where conformal's finite-sample guarantee is weakest, or (ii) a query conformal cannot answer at all (joint survival over multiple horizons; survival under missing sensors; `P(τ>t | partial evidence)`). Option (ii) is the safer bet — pick a query with no conformal analogue.

**Risk.** Modelling `τ` jointly with a high-dimensional window can let the window variables dominate the likelihood and starve the `τ` leaf. Mitigation: ordinal-categorical `τ` leaf + a term-weighted objective, or condition rather than joint-model (but conditioning costs the "one joint object" story — a real trade-off, decide empirically).

---

### **T3 — Exact typed anomaly decomposition as an instrument for the "mostly univariate" critique** ⭐ PURSUE (as Paper A section)

**Claim.** Decompose the window score exactly into per-channel marginal surprise `−log p(x_c)` and per-channel *conditional* surprise `−log p(x_c | x_{−c})`; their gap is exactly the channel's participation in cross-channel dependence. Aggregate over a benchmark to produce an exact, model-based univariate-vs-multivariate taxonomy of its anomalies.

**Why now.** arXiv:2606.02670 (2026) just made this question the field's live methodological issue and answered it with distance correlation. We can answer it exactly. This converts the biggest threat to the whole time-series direction into the contribution.

**Adversary.** Per-channel z-score and the paper's own distance-correlation statistic. Must show the exact decomposition (a) reproduces their qualitative finding, and (b) makes a *prediction they can't* — e.g. which specific channels the dependence lives in, validated against ESA-ADB's expert annotations or N-CMAPSS's known fault-component labels.

**Note:** this is the time-series instantiation of N3 "Typed-AD" from the 2026-06-25 sweep, which was DEMOTED to a Paper A section because its exactness moat was already published for the tabular case. In time series it has a *live external hook* it lacked before. Same verdict (section, not paper) but much higher value.

---

### **T4 — Time×channel vtree: the first domain with ground-truth structure** ⭐ PURSUE (absorbs the Paper C / curvature kill-shot)

**Claim.** Structured decomposability is a *free* constraint (established in the 2026-06-11 evaluation: all four properties hold for any vtree), so the only question is structure *quality*. Time series is the first domain in this project where the correct answer is partly known a priori: adjacent timesteps are dependent, sensors group by physical sub-component (N-CMAPSS labels faults by fan/LPC/HPC/LPT/HPT). So a structure learner can be **scored against ground truth**, not only against held-out NLL.

**Experiment** (this *is* the ablation `config/vtree_ablation.yaml` was built for, transplanted to a domain where it means something): `{random, chow_liu, spectral(NCut), orc, forman, dyadic-time, channel-major, physical-grouping}` × matched parameter budget × {held-out NLL, VUS-PR, RUL CRPS, **agreement with known temporal/physical grouping**}.

**Why this is the right home for it.** The 2026-06-11 evaluation gated the curvature idea (ORC vtrees) on a one-week ORC-vs-spectral-NCut ablation and flagged that on tabular data the finding could be "geometry is decoration." On time series there is an *interpretable* outcome either way: if ORC recovers time-adjacency and physical sensor groups without being told about them, that is a positive result independent of the NLL delta; if it doesn't, that is a clean, publishable negative. Tabular data offers no such fallback.

**Adversary.** `spectral_vtree` (already implemented as the honest adversary, `:640–695`) and the hand-built `dyadic_time_vtree`. If a trivially hand-specified temporal vtree matches or beats every learner, the learning story dies but the *structured-decomposability-is-free* story survives — still a section.

---

### **T7 — Routing over operating regimes, with ground-truth regime labels** — REFINE → strong section

**Claim.** The ProbMoE cardinality/selection-marginal machinery (already implemented, `:1253–1355`, `ProbRoutedRawPC`) routes over per-regime or per-unit sub-circuits. C-MAPSS FD002/FD004 have **six labelled operating conditions**; N-CMAPSS has flight phases.

**Why this matters for the portfolio.** The 2026-06-24 F1 evaluation flagged that the routing-localisation claim had **no ground truth** on ADBench — `argmax_j m_j(x)` could not be scored, only compared to `argmax_j log p_j(x)`. Operating conditions supply exactly the missing labels. This is a free upgrade to an already-implemented idea, and it costs one dataset loader.

**Adversary.** Unchanged and unforgiving: `argmax_j log p_j(x)` (no router) and plain mixture responsibilities (`mixture_responsibilities`, `directions.py:800`). If those recover the regime label as well as the cardinality marginals do, the routing machinery is redundant — which the F1 evaluation already called a *structural* redundancy risk. The ground-truth labels make the verdict crisp instead of arguable, which is worth having even if the verdict is negative.

---

### **T2 — Exact marginalisation under sensor dropout / dead channels** — REFINE (section, not paper)

**Claim.** Spacecraft telemetry has transmission gaps, saturated channels and decommissioned sensors; industrial rigs have failed sensors. A PC scores `−log p(x_O)` exactly with no imputation. Sweep the channel-dropout rate at test time; competitors must impute first.

**Why it's only a section.** (a) The PC-FUSION line (Natarajan/Kersting/Blasch, AISTATS'25 arXiv:2403.03281 + FUSION'26) already owns "exact + calibrated + missing-modality" for PCs — this is that argument in a new domain. (b) The honest adversary — mean/last-value imputation feeding a deep detector — is often *fine*, because sensor dropout at test time is not adversarial. Still worth running: it is cheap, it is the most operationally persuasive plot for a PHM audience, and it composes with T1 (`P(τ>t | partial evidence)` is a query conformal genuinely cannot answer — see T1's option (ii)).

---

### **T5 — Forecast-based vs. density-based AD, controlled within one model** — nice-to-have analysis

The prediction-error paradigm (`‖x_t − x̂_t‖`) and the density paradigm (`−log p(x)`) are never compared cleanly because they use different models. One circuit answers both: joint window density, *and* the exact one-step conditional `log p(x_t | x_{t−w..t−1})`. A controlled head-to-head is a genuinely useful contribution to the TS-AD methodology conversation — but it is an analysis section, not a paper.

---

### **T6 — Likelihood as a health index bridging detection and prognosis** — connective tissue, low standalone novelty

`−log p(x_t)` as a degradation health index, with first-passage over a threshold as the changepoint, is well-trodden in PHM. Its value here is *structural*: it is the sentence that makes T1 a unified framework rather than two models in a trenchcoat — "the detection score is the prognostic covariate, and both are queries on the same density." Keep as framing, do not claim as contribution.

---

### **T8 — Cross-machine / cross-mission structure transfer** — REFINE, Paper C material

`add_modality` (`directions.py:571`) already trains a new expert on an inherited vtree with zero structure search and zero disturbance to existing experts. Transfers to test: FD001→FD004 (same machine, new regimes), turbofan→bearing (different physics, aligned dimension via featuriser), mission A→mission B in ESA-ADB. This is Paper C's "structure transfer" thesis with a far more credible physical story than aligned tabular features — the alignment dependency flagged in `dev.md §3.2` and `notes.md:46` is *less* artificial when the shared axis is "time" than when it is "arbitrary latent dimension."

---

### Verdict table

| Id | Idea | Verdict | Home |
|---|---|---|---|
| **T1** | Exact RUL survival query + exact censored likelihood | **PURSUE** | **Paper A headline section, or standalone for a PHM/ML-applied venue** |
| **T3** | Exact typed (marginal vs. structural) decomposition | **PURSUE** | Paper A section — answers a live 2026 critique |
| **T4** | Time×channel vtree, ground-truth-checkable | **PURSUE** | Paper C / curvature kill-shot, relocated |
| **T7** | Regime routing with ground-truth regime labels | REFINE | upgrades the parked F1 evaluation |
| **T2** | Exact marginalisation under sensor dropout | REFINE | section; composes with T1 |
| **T5** | Forecast-vs-density within one model | keep | analysis section |
| **T8** | Cross-machine structure transfer | REFINE | Paper C |
| **T6** | Likelihood-as-health-index | framing only | connective tissue |

---

## 6. Scoop landscape and honest caveats

**Nearest neighbours in the literature** (none of them do AD or RUL):

- **CircuITS** — *Probabilistic Circuits for Irregular Multivariate Time Series Forecasting* (Klötergens, Yalavarthi, Schmidt-Thieme, arXiv:2604.27814, May 2026). A channel-recursive sum-product architecture for IMTS **forecasting**, claiming to be the first with structural marginalisation-consistency guarantees for IMTS. Evaluated on USHCN/PhysioNet/MIMIC. **This is the closest active group.** They are one hop from anomaly detection and two from prognostics. **Scoop window: months, not years.** Read the paper properly before committing.
- **Whittle Networks / WSPNs** (Yu, Ventola, Kersting, ICML 2021) and **RECOWNs** (arXiv:2106.04148) — PCs over the *spectral* representation of time series; **Predictive Whittle Networks** extends to forecasting. Own the "PC + time series likelihood" territory in the frequency domain. Their spectral construction is a plausible alternative leaf/feature layer for us and a mandatory related-work citation.
- **Dynamic SPNs** (Melibari, Poupart et al., PGM 2016) and **Recurrent SPNs** — template-unrolled circuits for sequences. The obvious "why not just use a DSPN?" reviewer question; have an answer ready (structured decomposability + a joint over `τ`, which template unrolling does not give you).
- **Conditional SPNs** (Shao et al., IJAR 2021) — gate-function conditioning; relevant if we go conditional rather than joint for `τ`.

**Unclaimed, as far as this sweep can tell:** PCs ∩ RUL / survival / hazard / censoring / prognostics. Multiple query formulations returned nothing. Treat as *probably* open, not *certainly* open — the prognostics literature is large, fragmented across PHM Society / RESS / Mechanical Systems and Signal Processing, and poorly indexed by ML-flavoured search terms. **Before committing, run one targeted pass over PHM Society proceedings, RESS, and MSSP for "tractable probabilistic model" / "sum-product" / "arithmetic circuit" prognostics.**

**Honest caveats to state up front in any writeup:**

1. **Determinism is not available** with Gaussian-mixture leaves (`validate_determinism`, `:1436` — full-support mixtures fail the check *by design*, and the docstring says so). So exact MPE-based imputation of a dead sensor is **not** on the table with the current leaf set; `mpe()` (`:1179`) is the max-product approximation there. Either claim MPE only for `n_sum_components=1`, or don't claim it. Do not let this leak into a paper as an unqualified "we support exact MPE."
2. **i.i.d. windows.** A window-level density treats overlapping windows as independent draws — false, and it inflates apparent training data. Use stride ≥ window length for evaluation splits, and be explicit that the circuit models the *window* distribution, not the process.
3. **Non-stationarity.** Normal behaviour drifts (spacecraft ageing, seasonal thermal cycles). A static density flags drift as anomaly. Either normalise per-unit per-regime, or make drift-handling an explicit limitation.
4. **The exactness must earn its keep.** Across every prior evaluation in this project the same failure mode recurs: an exactly-computed quantity that a trivial baseline approximates just as well for the *decision at hand*. Time series does not change this. T1's option-(ii) queries (survival under partial evidence, joint multi-horizon survival) are the ones with no cheap analogue — anchor the claim there.

---

## 7. Staged implementation plan (for when implementation starts)

**Stage T-0 — unblock (do regardless of the time-series decision).**
Region-graph / DAG rebuild of `DensityPC` (§3.1). Keep the current tree builder behind a flag for exact reproduction of existing results. Verify all four validators still pass and that `test_vtree.py` / `test_inference.py` are green. *This is the prerequisite for everything below.*

**Stage T-1 — interval queries.**
`log_cdf` / `log_interval` / `pair_log_interval` on all leaf types, `eval_log_box`, plus a `test_interval_queries.py` asserting `log_interval(-inf, +inf) == log_partition()` and agreement with numerical quadrature on a 1-D circuit. Small, self-contained, high leverage.

**Stage T-2 — data layer.**
`src/datasets_ts.py`: C-MAPSS first (small, fast, censored, has regime labels — it exercises T1, T4 and T7 simultaneously), then TSB-AD, then ESA-ADB, then N-CMAPSS.

**Stage T-3 — directions.**
`TimeSeriesPCDetector` (Direction 1: window featuriser — patching / spectral / statistics — into the shared latent PC, reusing `Featurizer`) and `WindowRoutedPC` (Direction 2: raw window, time×channel vtree, routed by unit/regime, subclassing `RoutedRawPC`). Same `AnomalyDetector` interface, so `experiment.py` and the cluster harness need no changes.

**Stage T-4 — the joint `(window, τ)` circuit and censored objective.** T1 proper.

**Stage T-5 — the batched kill-shot harness.**
Per the 2026-06-25 meta-finding, run T1 / T3 / T7 / T2's redundancy tests as **one** pre-registered harness with frozen hyperparameters, exactly as the existing `config/` + `cluster_scripts` infrastructure was built to do. Do not tune-then-report.

**Order of evidence, if time is short:** T-0 → T-1 → C-MAPSS loader → T1 kill-shot. That single path either establishes or kills the strongest idea in this document within roughly two weeks of compute, and it is the shortest route to knowing whether time series deserves the investment.

---

## 8. Sources

- [Probabilistic Circuits for Irregular Multivariate Time Series Forecasting (CircuITS), arXiv:2604.27814](https://arxiv.org/html/2604.27814) — nearest active competitor
- [Whittle Networks: A Deep Likelihood Model for Time Series (ICML 2021)](https://proceedings.mlr.press/v139/yu21c.html) · [code](https://github.com/ml-research/WhittleNetworks)
- [RECOWNs: Probabilistic Circuits for Trustworthy Time Series Forecasting, arXiv:2106.04148](https://arxiv.org/pdf/2106.04148)
- [Dynamic Sum-Product Networks for Tractable Inference on Sequence Data (PMLR v52)](https://proceedings.mlr.press/v52/melibari16.html)
- [Conditional Sum-Product Networks: Modular Probabilistic Circuits via Gate Functions (IJAR 2021)](https://www.sciencedirect.com/science/article/pii/S0888613X21001766)
- [Strudel: Learning Structured-Decomposable Probabilistic Circuits (PMLR v138)](http://proceedings.mlr.press/v138/dang20a/dang20a.pdf)
- [The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD, NeurIPS 2024 D&B)](https://openreview.net/forum?id=R6kJtWsTGy) · [site](https://thedatumorg.github.io/TSB-AD/) · [code](https://github.com/TheDatumOrg/TSB-AD)
- [Current Time Series Anomaly Detection Benchmarks are Flawed... (Wu & Keogh, arXiv:2009.13807 / TKDE 2021)](https://arxiv.org/abs/2009.13807)
- [Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate (arXiv:2606.02670)](https://arxiv.org/pdf/2606.02670) — the critique to pre-empt / weaponise
- [European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry (ESA-ADB, arXiv:2406.17826)](https://arxiv.org/abs/2406.17826) · [code](https://github.com/kplabs-pl/ESA-ADB) · [dataset](https://github.com/esa/anomaly-dataset)
- [Toward Deployable Satellite Anomaly Detection: Benchmark Study on Large-Scale ESA-ADB Telemetry (arXiv:2607.07335)](https://arxiv.org/abs/2607.07335)
- [PHM Society 2021 Data Challenge (N-CMAPSS)](https://data.phmsociety.org/wp-content/uploads/sites/9/2021/08/2021_Data_Challenge.pdf)
- [CruiseBench: A Real-Flight-Aligned N-CMAPSS Benchmark for Engine RUL Prediction (arXiv:2607.19380)](https://arxiv.org/html/2607.19380)
- [Conformal Prediction Intervals for Remaining Useful Lifetime Estimation (arXiv:2212.14612 / IJPHM)](https://arxiv.org/abs/2212.14612) — the calibration adversary
- [Turbofan RUL with Reliable Prediction Intervals via LSTM Quantile Regression and Conformal Calibration (Sensors, 2026)](https://www.mdpi.com/1424-8220/26/7/2249)
- [Uncertainty-Aware Bearing RUL Prediction Based on Conformal Prediction (PHME 2026)](https://papers.phmsociety.org/index.php/phme/article/view/4902)
- [Robust UQ for online RUL with randomly missing and partially faulty sensor data (RESS 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0951832025003783)
- [Deep probabilistic graphical modeling for robust MTS anomaly detection with missing data (RESS 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0951832023003241)
- [Probabilistic Circuits for Uncertainty Quantification (ICLR 2026 Blogposts)](https://iclr-blogposts.github.io/2026/blog/2026/probabilistic-circuits-for-uncertainty-quantification/)
