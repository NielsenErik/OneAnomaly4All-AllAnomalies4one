# Brainstorm: e-value conformal prediction on top of the exact PC

Handoff note. Written to be read cold in Claude Code with `components.py` open.
Status: **design settled, nothing measured yet.** Every number below that is not
already in `components.py`'s docstrings is a hypothesis, not a result.

---

## 0. What exists and what is being added

**Exists** (`components.py`): a structured-decomposable / region-graph PC over a
flattened `(window x channel)` sensor window jointly with a discretised
time-to-failure `tau`. Exact `log_prob`, exact `log_marginal`, exact
axis-aligned `log_box` (the survival / right-censoring query), exact
`log_partition`. `CompiledCircuit` gives a layer-parallel fast path gated
against the recursive evaluator.

**Being added**: a distribution-free wrapper that turns the circuit's exact
densities into *calibrated* statements —

- an **anomaly alarm** with a controlled false-alarm rate, and
- a **RUL prediction set** with marginal coverage,

using **e-values** rather than p-values, so the miscoverage level can be chosen
*after* seeing the data (post-hoc validity) and so the two statistics can be
merged and accumulated.

Reference implementation lives in `pc_conformal.py` (companion file).

**Papers.**
- Laxhammar & Falkman 2015 (`s10472-013-9381-7.pdf`) — ICAD, the inductive
  split, well-calibrated alarm rate, the requirement that a sequential score be
  monotone in time. This is the *anomaly* half.
- Gauthier, Bach & Jordan, AISTATS 2026 (`2510_04318v2.pdf`) — soft-rank
  e-variable, post-hoc validity (Prop. 2.2), adaptive coverage policy trained
  leave-one-out on the calibration set (Alg. 1), lambda-selection by
  bracket-and-bisect (Alg. 2). This is the *prediction set* half.

---

## 1. The one structural decision: three splits, not two

```
D_fit    the PC is fitted here, and ONLY here
D_cal    calibration scores; the circuit must never have seen these rows
D_test   deployment
```

The coverage policy `alpha_tilde_theta` **may** be trained on `D_cal` — that is
exactly what Prop. 2.2 licenses, and it is the whole reason for using e-values
instead of p-values. The **circuit may not**. Fitting or fine-tuning the PC on
`D_cal` destroys exchangeability of the calibration scores and no amount of
e-value machinery repairs it.

If we later want conformal-training of the circuit itself (differentiating set
size back into leaf/sum parameters — attractive, since the whole score path is
differentiable through `log_prob`), that needs a **fourth** split, or a nested
scheme. Do not do it by accident.

---

## 2. Two tasks, two different score constructions

They share `D_cal` but not the score. Conflating them is the most likely way
this goes wrong.

### 2A. Anomaly alarm (unsupervised, no label)

Score = NLL of the window with `tau` marginalised out:

```python
s(x) = -pc.log_marginal(z, marginalized=[tau_idx])
```

**The scale trap.** The soft-rank e-variable
`E = S_test / [(1/(n+1))(sum S_i + S_test)]` needs `S >= 0` and negatively
oriented. Two facts:

1. `E` is invariant to `S -> cS` but **not** to `S -> S + b`. So you cannot
   shift NLL positive; a large shift drives every `E -> 1` and kills all power.
2. The shift-free choice is `S(x) = exp(-log p(x))`, giving the clean form
   `E = (n+1) / (1 + sum_i p(x_test)/p(x_i))` and the ceiling `E <= n+1`.

But `log p` spans hundreds of nats here (see the `sigma_floor` docstrings —
unfloored leaves reached `log f ~ -2.3e11`). So `sum_i S_i ~ max_i S_i` and the
detector degenerates into "compare against the single worst calibration window."

**Decision: route Task A through a rank-based p-value, then calibrate p -> e**
with `f_kappa(p) = kappa * p^(kappa-1)`, `kappa in (0,1)`. Ranks are invariant to
*any* monotone transform of the score, so the NLL scale problem vanishes
entirely, and `E[f(P)] <= 1` keeps post-hoc validity. This is ICAD's own
machinery with an e-value cap on top.

`diagnose()` in `pc_conformal.py` reports a `saturation` field precisely to
catch the failure mode if we ever switch back to soft-rank here.

### 2B. RUL prediction set (supervised, label tau)

`tau` is already discretised (`bench_rul_capacity.make_task`, bins=25,
window=30) with a `CategoricalLeaf`. **That puts us literally in the paper's
classification case**, with `|Y| = n_bins`. Good — the regression case in their
Appendix C is degenerate (MAE score -> set size independent of `X_test` ->
constant-width sets that cannot adapt). We sidestep it.

Score:

```
log p(tau=k | x) = log p(x, tau=k) - log_marginal(x, marginalized=[tau])
S(x,k)           = exp( log p(tau=k*|x) - log p(tau=k|x) )   >= 1
```

Why this score and not a residual:

- `S_min = 1 > 0` **by construction**, which is assumption (i) of their
  Theorem 2.6. Most scores don't satisfy it; ours does for free.
- Thresholding `S` is thresholding the *density*, so `C(x)` is a genuine
  superlevel set of `p(tau|x)` — narrow where the circuit is confident, wide
  where it isn't. This is the locally-adaptive behaviour their own regression
  testbed cannot produce, and it is the strongest argument that the PC is the
  right base model for this wrapper.
- The score vector over bins **is** the test summary statistic `t(X_test)` the
  coverage policy consumes. One conditional pass feeds both the set and the
  policy.

Set:

```
C(x) = { k : S(x,k) < sum_i S_i / ((n+1) * alpha - 1) }
```

Note `alpha > 1/(n+1)` is required or the set is everything. Same resolution
floor Laxhammar flags in section 3.6 for ICAD. **The size of `D_cal` caps how
small an alpha we can even express** — this is a data-collection constraint, not
a tuning knob.

---

## 3. Why e-values and not just ICAD p-values

Three things p-values cannot do:

1. **Post-hoc alpha.** Choose the threshold after seeing the scores. This is
   what makes the learned coverage policy legal at all.
2. **Merging.** The weighted arithmetic mean of e-values is a valid e-value
   under *arbitrary* dependence (Vovk & Wang 2021). So
   `E_unit = 0.5 * E_anomaly + 0.5 * E_RUL` is a single valid unit-health
   statistic with **no** independence assumption between the density score and
   the RUL score — which is good, because they are obviously dependent (same
   circuit).
3. **Accumulation over time.** If each `E_t` is an e-value *conditional on the
   past*, `M_T = prod_t E_t` is a nonnegative martingale and Ville's inequality
   gives an **anytime-valid** alarm at `M_T >= 1/alpha`, no multiple-testing
   correction. ICAD gives nothing comparable.

   *Caveat, do not skip:* the conditional-validity requirement is real. Naively
   multiplying `E_t` computed against a fixed calibration set is **not** valid.
   The rigorous route is Vovk's conformal test martingales (bet on the p-value
   sequence). Treat product-accumulation as a TODO with a correctness gate, not
   as something to ship.

---

## 4. Where this actually breaks — the honest adversary list

Ordered by how likely they are to invalidate a result.

### 4.1 Exchangeability across windows (highest risk)

Sliding windows from one engine are strongly dependent. Treating `n_windows` of
them as `n` exchangeable calibration points inflates the effective sample size
badly and the coverage guarantee silently becomes decorative.

**Two defensible options:**

- **Per-unit scores.** One score per engine: `max` over that unit's windows.
  This has a second benefit — it makes the preliminary score *monotone
  non-decreasing in time*, which is exactly requirement 3 of Laxhammar
  section 4.1, and it is what makes a sequential alarm well-calibrated.
  `EvalueICAD.calibrate(..., unit_ids=...)` implements this.
- **Per-window scores**, accepted as a heuristic, with the caveat stated in
  every table.

FD001 has ~100 train units. Per-unit calibration on a 50/50 split gives
`n ~ 50`, hence `alpha >= 1/51 ~ 0.02`. That is the real operating floor. Say so
in the paper rather than reporting `alpha = 0.01` that the data cannot support.

### 4.2 Degradation IS non-exchangeability

Within one unit's life the distribution shifts by design — that shift is the
signal. We cannot simultaneously assume exchangeability over the life and detect
drift. Split the claims:

- **marginal coverage** -> population of units, cross-unit.
- **within-unit monitoring** -> martingale / e-process, not marginal coverage.

### 4.3 Censoring (largest statistical risk after 4.1)

Units still alive at time `c` have no observable `tau`, so no calibration score.
Dropping them biases `D_cal` toward early failures unless censoring is
independent of the covariates — which for run-to-failure benchmarks it roughly
is, but for real fleet data it is not.

The circuit gives `log P(tau > c | x)` **exactly** via `log_box` (see
`censored_log_survival`), and `CategoricalLeaf.log_interval` is inclusive, so
the suffix is `(c+1, inf)`. That gives a valid *bound* on the score for censored
units, which supports a **lower prediction bound on tau** but not a two-sided
set. Reference: conformalized survival analysis (Candes, Lei, Ren, Bates).

**Validate this before doing any modelling work on top of it.** It is the
easiest place to publish something wrong.

### 4.4 Circuit-side gotchas already documented in components.py

These are known and fixed but will re-bite if a config drifts:

- `weight_jitter > 0` is **required** for `RegionGraphPC` — with uniform initial
  weights the K units of a region are identical functions with identical
  gradients and the circuit silently becomes a product of marginals. The
  failure is invisible in the training loss and shows up as **`p(tau|x)`
  independent of `x`** — i.e. a flat, identical conditional for every unit,
  which would make every prediction set identical and every adaptivity claim
  vacuous. `CoveragePolicy` would still train happily. **Add an explicit assert
  that the conditional varies across the batch before trusting any set.**
- `sigma_floor` binds on whole median-constant channels (30/450 features on
  FD001, all of channel 3; 30/480 on FD003, channel 7; 60/630 on FD002/FD004).
  Any FD001 number is *not* automatically floor-free.
- `move_circuit_` not `.to()` — `.to()` is `_apply`, unmemoised over a DAG,
  and reintroduces the `K^depth` blowup through the back door.
- The 2026-08-04 real-data logs predate the `GaussianLeaf` fix; anything derived
  from them is re-measure-or-discard.

### 4.5 The order-blindness question is still open

Per the `chain_region_graph` and `permute_region_graph` notes: on synthetic
AR(1) windows, the timestep-block permutation cost only +0.07 nats while the
full permutation cost +5.17 — i.e. the chain's advantage was the *blocking*, not
the chain. If the conditional `p(tau|x)` is largely order-blind, the RUL sets
will be driven by per-step marginals, and the adaptivity story weakens. Worth
re-running the permutation control **on the conditional**, not just on the joint
NLL. `delta_window_transform` is unit-determinant, so it can be composed in with
no Jacobian correction and no loss of comparability.

---

## 5. Concrete next steps, in order

1. **Gate: exchangeability sanity.** Compute e-values on held-out *healthy*
   data and check `mean_e <= 1`. If it fails, exchangeability or the score sign
   is broken and everything downstream is void. `diagnose()` does this.
2. **Gate: conditional non-degeneracy.** Assert `p(tau|x)` actually varies
   across a batch (see 4.4). One line, catches the worst silent failure.
3. **Decide the unit of exchangeability** (4.1) and fix it for all experiments.
   Record the induced `alpha` floor.
4. **Task A end to end.** `EvalueICAD` on FD001. Reproduce Laxhammar's
   diagnostic: histogram of empirical alarm rate over ~100 random permutations
   of the unit ordering, at a nominal 1% (their Figs. 8-10). Expect the peak at
   or below nominal. If it sits above, go back to step 1.
5. **Task B fixed-alpha baseline first.** Coverage and mean set size at a grid
   of fixed alpha, using `e_prediction_set` with a constant. This is the
   `e-fixed` baseline of Table 1 and it must work before any policy is trained.
   Also record the `p-fixed` baseline (plain ICAD ranks) — it will have the
   smallest sets and no post-hoc validity; that trade is the point.
6. **Then** train `CoveragePolicy` (Alg. 1) and check it beats `e-fixed` at
   matched mean `alpha`. Their CIFAR numbers were 1.30 vs 1.42 — a ~8% size
   gain. If we see nothing, the conditional is probably too flat (4.5), not the
   policy's fault.
7. **Lambda selection** via bracket-and-bisect (Alg. 2) to hit a target mean set
   size in *bins*, which translates to a target RUL interval width in cycles —
   the number a maintenance planner actually wants.
8. **Censoring** (4.3), last, and with its own validation.

---

## 6. Things worth deciding but not yet decided

- **`kappa` for the p-to-e calibrator.** 0.5 is the default; the max attainable
  e-value is `0.5*sqrt(n+1)`. Smaller `kappa` is more sensitive to very small
  p-values (rare, severe anomalies), larger is flatter. Ablate.
- **`s_max` clip on the HPD scores.** Needed for Theorem 2.6's `S_max`, and to
  stop one numerically-zero bin from dominating `sum_i S_i`. Currently `1e6`,
  chosen by nothing. Check the empirical distribution of `S` and set it
  deliberately.
- **Whether the anomaly score should marginalise `tau` or box it.** Marginalising
  asks "is this window unusual for any RUL?". Boxing `tau` to the *predicted*
  range asks "is this window unusual *given* where we think it is in life?" —
  arguably the better detector, and only the PC can compute it.
- **Merging weights** for `E_unit`. Equal is a placeholder. Any fixed weights
  are valid; the choice is a power question, not a validity question.
- **Do we even want the martingale?** It's the most novel piece and the most
  likely to be wrong. Possibly a follow-up rather than part of this work.

---

## 7. Files

- `components.py` — in this repo, `src/probabilistic_circuits.py`. Unchanged;
  the wrapper needs nothing from it that isn't already public (`log_prob`,
  `log_marginal`, `log_box`).
- `pc_conformal.py` — in this repo, `poc/time_series/pc_conformal.py`. The
  conformal layer: scores, e-values, `EvalueICAD`, `EvaluePredictionSet`,
  `CoveragePolicy` + LOO training, `ConformalSurvivalBound`, diagnostics.

The wrapper is deliberately read-only with respect to the circuit. If it ever
needs to write into the circuit, that is a signal that a split has been
violated.

---

## 8. IMPLEMENTED 2026-08-11 — what was built, and what changed on contact

`poc/time_series/pc_conformal.py` + `tests/test_pc_conformal.py` (38 tests,
~1 s, no circuit fitting — the guarantees are checked against fakes so that
validity does not depend on the model being any good). Everything in §2-§5 is
in except the martingale (§3.3), which stays a TODO for the reason the document
already gives.

**Five things the design did not survive contact with, in order of importance:**

1. **§4.3 applies to Task B as well, and that was a live bug.** For a censored
   window `tau` is the censoring BOUND, not the label. Calibrating the
   prediction set at `tau_train` therefore scores ~19% of rows against a wrong
   value. `EvaluePredictionSet.calibrate` now takes `delta_cal` and drops them
   (a two-sided set genuinely cannot use them). Measured cost of the bug:
   p-fixed coverage 0.770 ± 0.058 mislabelled vs 0.777 ± 0.042 dropped over 7
   seeds — a correctness fix with a small effect. One seed read 0.667 vs 0.738
   and looked decisive; that was noise at n_cal ~ 40.

2. **The `p-fixed` baseline is not optional, it is the instrument.** §5 step 5
   lists it as a baseline to record; it is what *found* bug 1. The e-set's
   conservatism masked the mislabelling completely (coverage 1.000 either way).
   `report()` now computes it by default and returns `e_width_premium`.

3. **Per-unit `max` reduction answers a different question than assumed.** It
   guarantees every window of a new unit is covered simultaneously — strong,
   and near-vacuous here (11.6 of 12 bins). `reduce="random"` (one window per
   unit) gives ordinary window-level coverage while keeping the unit as the
   exchangeable object, and is now the default. Both are recorded in the
   diagnostics so a table cannot mix them silently.

4. **`kappa < 0.5` has infinite variance.** E[f(P)] = 1 exactly for all kappa in
   (0,1), but E[f(P)^2] = kappa^2 * int p^(2k-2) dp diverges at kappa <= 1/2.
   So the §6 kappa ablation is not a free sensitivity knob: below 0.5 the
   `mean_e <= 1` gate gets noisy and any measured "power" is dominated by rare
   huge values. Validity is untouched.

5. **The §2A scale trap reappears in §2B and needed the same treatment.** The
   set threshold divides `sum_i S_i`, so one confidently-wrong calibration unit
   can set it alone. `log_s_max` is therefore load-bearing rather than
   cosmetic, and `sum_concentration = max_i S_i / sum_i S_i` is reported.

**Two measured results that bear on whether this is worth pursuing:**

- **The gate fires on real models.** `assert_conditional_varies` refused 1 of 8
  seeds at ordinary settings (TV 4.4e-4 — `p(tau|x)` essentially constant).
  The degeneracy §4.4 warns about is not hypothetical, and a run that produced
  a full valid-looking table on that seed was one gate away.
- **The e-set is close to vacuous at this calibration size.** At n_cal ~ 40
  units, 12 bins, alpha=0.20: e-set coverage 1.000 with mean size 10.7/12,
  against p-fixed size ~4. Post-hoc validity is costing nearly all the
  informativeness. This is the headline number for "can e-values be exploited
  for a high-performing solution here" and it is currently NO for Task B at
  fleet scale — the resolution floor of §4.1 is the binding constraint, exactly
  as that section predicted. `ConformalSurvivalBound` (one-sided, keeps
  censored units) is the piece that is not limited this way and remains the
  best reason to keep this line.

---

## 9. The martingale (§3.3) — BUILT 2026-08-11, and it needs two controls

`ConformalTestMartingale`, `online_conformal_pvalues`, `martingale_lead_times`,
`naive_product_alarm_rate`, `assert_non_overlapping`. 48 tests.

**Construction.** Not the product of `EvalueICAD`'s e-values — those come from a
frozen calibration set, so they are marginally but not *conditionally* valid.
Instead Vovk's conformal test martingale: smoothed **online** conformal
p-values (i.i.d. uniform under exchangeability), a betting function with
integral 1, and `M_T = prod_t f(p_t)`. Ville then gives
`P(sup_T M_T >= 1/alpha) <= alpha` — anytime-valid, no horizon, no correction.
Default bet is the simple mixture over `eps`, so there is no free parameter; a
convex combination of martingales is a martingale, so grid coarseness cannot
affect validity.

**Finding 1 — the naive product is a trap, not a blow-up.** At the default
`kappa=0.5` it respects the bound *by accident*: the bet's drift is
`log k + 1 - k = -0.19` per step, so it goes bankrupt before the shared-
calibration bias matters. Weaken the drift and it breaks (exchangeable null,
T=2000, nominal 5%):

| kappa | drift/step | alarm rate |
|---|---|---|
| 0.50 | −0.193 | 2.3% |
| 0.80 | −0.023 | **13.7%** |
| 0.95 | −0.001 | **14.3%** |

Since §6 lists `kappa` as a parameter to ablate, following this document with
the naive product walks from an accidentally-safe setting into a 3x-invalid
one. Power is *not* the differentiator — on a strong changepoint both detect
100%.

**Finding 2 — OVERLAPPING WINDOWS INVALIDATE THE MONITOR.** The largest trap in
this half, and invisible without a control. At stride < window consecutive
scores are autocorrelated, so exchangeability is false before any degradation.
Running the monitor on the **healthy region only** (RUL > 100, nothing to find):

| stride (window=6) | full-life fired | median lead | **healthy-only fired (nominal 5%)** |
|---|---|---|---|
| 6 (disjoint) | 33% | 13 cyc | **2%** |
| 2 | 75% | 30 cyc | **17%** |
| 1 | 90% | 130 cyc | **57%** |

The full-life column looks like a spectacular win from denser windowing. It is
entirely the artefact — at stride 1 the "lead time" equals the RUL cap because
it fires on the first window of every unit. `assert_non_overlapping` now
refuses this; there is no correction, the null itself is false.

**Finding 3 — at the only valid setting the monitor is weak but honest.**
window=6, stride=6, alpha=0.05: fires on ~33% of units with median lead time
~13 cycles out of a 130-cycle horizon. The score is not the bottleneck — the
within-unit corr(anomaly score, RUL) is **−0.62** with 93% of units below −0.3,
so degradation is clearly present in the signal. What limits it is that the
score only crosses the healthy fleet's 95th percentile at RUL < 20 (47% of
windows there, 8% at RUL 20–50). Warm-starting with the healthy fleet makes it
a *level* test (fires 2%, lead 2.5 cyc); dropping the warm start makes it a
within-unit exchangeability test (33%, 13 cyc). Neither is operationally
useful yet.

**Verdict.** The machinery is correct and the guarantees hold. As an early-
warning device on this data it does not yet earn its place, and the reason is
diagnosed rather than mysterious: the density score separates too late. Two
things worth trying before abandoning it — a score built on the *trend* rather
than the level, and real C-MAPSS where degradation is not synthetic. Do not
report a lead-time number without the healthy-region control next to it.

---

**Stale premises this document was written against** (it predates the 08-06
re-measurement, which is in `logs/ts/` but not in `hands_off.md`):
§4.5's order-blindness question is CLOSED and the sign reversed — on real FD001
at identical parameter count, `chain` 0.7528, `chain_perm_blocks` 0.7512,
`chain_perm_features` 0.7923. Destroying time order costs nothing; destroying
the blocking as well *gains* 0.040 AUROC. And `leaf_components=1` (the default)
is flat across a 62x parameter range, so §5's gate 2 should be expected to fire
there rather than pass.