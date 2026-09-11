# Implementation plan — 8 September 2026

**Historical plan:** the Tier 1 gate failed. The [September 10 implementation roadmap](IMPLEMENTATION_PLAN_2026-09-10.md) is the current plan. Its protocol supersedes the constant-cost and uncached-baseline assumptions below; this file preserves the original preregistration and development record.

**Companion to `POC_REVIEW_2026-09-08.md`.** The review establishes that the broad
"universal anomaly detector" thesis is unsupported and that one question survives:
*can a compact tractable joint model retain useful relational fault localization when
the available sensor set changes, at lower query cost than equally capable alternatives?*

This document is what must be built to answer it. It is scoped to one focused cycle
(~13 working days) with an explicit kill gate, not a feature expansion.

---

## 0. The claim being implemented

> **X** — a channel-aligned structured-decomposable circuit, a two-pass (inside/outside)
> diagnosis algorithm over it, and a mask-calibrated relational statistic.
>
> **Y** — the full relational diagnosis map (every channel, under an arbitrary
> missing-sensor mask) at a cost that does not grow with the number of channels or
> the number of masks, with localization that survives sensors dropping out.
>
> **Z** — because marginalizing a channel is subtree deletion in the vtree, so masks
> reuse shared work, whereas Gaussian/GMM block conditionals require a fresh
> factorization per mask and Lüdtke-style subset search requires a fresh circuit
> pass per candidate subset.

Tier 1 *is* that contribution. Tiers 0, 2 and 3 exist to make it measurable and to
survive the obvious reviewer objections. If Tier 1 collapses, the honest output is a
negative result plus a benchmark, not a weaker version of the same paper.

The diagnostic quantity throughout is

    R_S(x) = log p(x_S) + log p(x_-S) - log p(x)

i.e. negative pointwise mutual information between a channel block and the rest.
It is *not* a new identity, it is not non-negative, and it is not an additive
allocation of the total anomaly score. The contribution is the algorithm, the
missing-sensor operating regime, and the validated advantage over strong tractable
baselines — never the identity itself.

---

## Tier 0 — evaluation and provenance repairs — **DONE 2026-09-08**

Nothing measured before these lands is reviewable.

**Status: all seven items implemented; 27 new tests in
`tests/test_tier0_evaluation_repairs.py`; full suite 433 passed / 0 failed.**
What changed beyond the table below, and what it costs:

* `ADTask` now carries `X_val`, `unit_train`, `unit_val`, `unit_test`
  (`data.py`), populated by all three builders.  `RULTask` already had unit
  ids; the AD path did not, which is why AD intervals could only ever be
  window-level.  Engine-level bootstraps (Tier 3.2) are now possible.
* **Every AD number recorded before today is on different data.** The
  validation carve removes ~20% of the training engines and the standardiser is
  refit without them, and `windowize` adds one window per unit.  Re-run before
  comparing anything to the old tables.
* On the existing logs, provenance gating drops **70 of 1297 rows** — all from
  the seven failed `cmapss_rul` runs.  `cmapss_structure_x_leaves` is
  unaffected (72/72), so the one clean structure result stands.  Those seven
  runs predate stage marking, so their completed full-input stages cannot be
  recovered without a re-run.
* Two existing tests were rewritten rather than re-thresholded: the RUL
  degeneracy guardrail (it pinned one lucky K; now pins the mechanism plus
  "at least one collapsed K is caught") and the PIT decomposition (it compared
  an absolute mean error against a *relative* variance error; now both in PIT
  sd units, where the finding is 19× rather than 2.7×).  Both docstrings carry
  the re-measured numbers.
* Found while testing, worth its own line: **this circuit is exactly factorised
  across channels for its first ~20 epochs** (dependence |Σ marginals − NLL| =
  0.0000 nats at epoch 20, 1.02 at 40, 2.75 at 60) while the training NLL falls
  smoothly throughout.  Under such a model every relational quantity — the
  diagnosis paper's whole subject — is identically zero, and nothing in the
  loss says so.  The crossover is in gradient steps, so this does not
  automatically apply to the 50-epoch C-MAPSS runs; it does mean Tier 1 must
  measure it there before interpreting any structural score.
  (`test_a_factorised_circuit_is_visible_in_the_structural_term`)
* Unrelated pre-existing annoyance: `tests/test_vtree.py::test_density_pc_128_features`
  takes 783 s of the suite's 840 s (tree-layout K^depth blowup at d=128).
  Deselect it for fast iteration: the rest of the suite is 64 s.

| # | Item | Where | What to do |
|---|---|---|---|
| 0.1 | Real held-out validation | `data.py:350` (`ADTask`), `datasets.py:192` (`make_ad_task_split`) | Add `X_val` carved from a third engine-disjoint split. `Standardizer` **and** structure learning fit on train engines only. Repoint `_held_out_nll` (`pipeline.py:140`) at `X_val`, rename the reported column, and add validation-loss checkpoint selection inside `WindowPC.fit` (`circuits.py:362`). |
| 0.2 | Run provenance | `ts_logging.py::read_results`, `aggregate.py` | Immutable attempt IDs and per-stage success flags. Aggregation must refuse rows from stages that did not complete, instead of reading `results.jsonl` blind. Partial results stay visible but are labelled, and a completed full-input stage is not erased because a later missing-data arm failed. |
| 0.3 | Tie-grouped average precision | `metrics.py:56` | Group thresholds by unique score value. The current implementation is ordering-dependent on tied/quantized scores (four tied scores with two positives give 1.0 or .4167 instead of .5). |
| 0.4 | Final endpoint included | `windowize`, `make_rul_task_split` (`datasets.py:489`) | The stride grid can miss the true last cycle; force inclusion of the final index so "last window" is the official benchmark endpoint. |
| 0.5 | Explicit nominal levels | `_eval_survival` in `pipeline.py` | Derive endpoints from the requested alpha rather than always reading `q05/q95` while scoring with a different alpha's penalty. |
| 0.6 | Honest naming | `explain.py:100` | Rename `sampling_shap` -> `replacement_sensitivity`. It replaces one channel at a time and averages **absolute** score changes: it is neither a coalition sampler nor a Shapley estimator. Retract the "exact attribution beats approximate SHAP" claim in all text. |
| 0.7 | Align the explanation claims | `explain.py:143` (`localization_report`), `explain.py:169` (`completeness_error`) | Completeness is proved for `chain_rule_attribution` while localization is usually reported for leave-one-out conditional surprise. Either localize with the attribution whose completeness is proved, or report the two as separate, separately-labelled claims. Do not combine them into one "correct and complete explanation" sentence. |

**Exit criterion for Tier 0:** every headline number can be traced to a completed
stage, a named validation split, and a stated nominal level.

---

## Tier 1 — the method (~5 days; contains the kill gate)

**Status: implemented 2026-09-08; 18 tests in `tests/test_tier1_relational.py`;
full suite 451 passed / 0 failed.**

### THE GATE HAS RUN, AND IT FAILS — 2026-09-08, real C-MAPSS FD001, 3 seeds

`logs/ts/tier1_kill_gate/`, 12/12 runs `ok`, 15/15 rows used, all four arms at
**174,240 parameters** (a binary vtree over d variables has d−1 internal
regions whatever its shape, so "matched" here is exact, not approximate).

| structure | val NLL | val AUROC | dependence (nats) | channel-blocked |
|---|---|---|---|---|
| `chow_liu` | 58.3 ±8.2 | **0.8268** ±0.027 | **34.2** | no |
| `random` | 70.1 ±12.7 | 0.7994 ±0.009 | 23.9 | no |
| `channel` | **40.5** ±8.3 | 0.7937 ±0.020 | 6.0 | yes |
| `channel_blocked` | 45.3 ±12.5 | 0.7934 ±0.027 | 5.6 | yes |

Against the pre-registered budget (≥ −0.02 AUROC, ≤ +5% NLL, vs the best
NON-blocked arm): **ΔAUROC −0.0334 → FAIL; ΔNLL −22.4% → inside.**

Read the two axes together, because they point in opposite directions and the
one-line verdict hides it:

* **Blocking BUYS density.** Both blocked arms beat both unblocked ones on
  held-out NLL by 22-42%.  The plan's §1.2 sentence — "if the blocked structure
  cannot hold density" — is not what happened; it holds density comfortably.
* **Blocking SELLS detection**, −0.033 AUROC against Chow-Liu, a third larger
  than the budget.  Consistent in every seed (−0.033, −0.039, and the mean),
  though the three-seed sd is ±0.027 and a three-seed sd is not a significance
  test (§3.2).  The rule was pre-registered on the MEAN and the mean fails it.
* **The mechanism is in the dependence column, and it is worse news than the
  AUROC gap.** The blocked circuits carry **5.6-6.0 nats** of cross-channel
  dependence against Chow-Liu's **34.2** — six times less.  R_S is *computed
  from* that dependence.  So the boundary that makes the relational map cheap
  is also what starves the quantity the map reports: the two-pass algorithm is
  correct and fast on a model that has comparatively little to say.  That is a
  structural objection to the design, not a tuning problem, and it is the thing
  to answer before any repair.
* **The channel ORDER is noise.** `channel_blocked` (Chow-Liu channel order)
  and `channel` (identity order) are 0.7934 vs 0.7937 AUROC and cross over
  between seeds.  The 1.1 hybrid's ordering claim is not supported; the
  blocking, not the order, is doing everything.

**Consequence, per §1.2:** the numbers from 1.3-1.6 are correct, measured and
reusable as ENGINEERING (see the cost curve below), but they are not findings
about a model worth deploying, and Tier 2/3 must not be built on this structure
as it stands.  Any repair — a blocked structure with more cross-channel
capacity at the top, Chow-Liu *within* each channel block, a different
candidate — is a NEW pre-registration on fresh seeds, not a re-read of this
gate.

*(Provenance note: the per-seed `VERDICT` rows in this batch were computed by
the stage before a mid-run fix to the comparator set — they compare against all
other arms including the blocked ones.  `run_tier1_gate.py` recomputes the
verdict from the arm rows and is authoritative; it is what the table above
reports.)*

Where it lives:

| Item | Code | Entry point |
|---|---|---|
| 1.1 | `channel_blocked_vtree`, `chow_liu_channel_order`, `block_boundary_units` (`src/probabilistic_circuits.py` §6c) | `WindowPC(vtree_method="channel_blocked")` |
| 1.2 | `stage_structure_gate`, `gate_verdict`, `channel_dependence` (`pipeline.py`); `match_K`, `structure_param_count` (`circuits.py`) | `config/ts/tier1_kill_gate.yaml`, `run_tier1_gate.py` |
| 1.3 | `RelationalCircuit.relational_map`, `inside_log_values`, `outside_log_values` (§6c) | `WindowPC.relational_map` |
| 1.4 | `RelationalCircuit.masked_log_prob`, `_boundary_values` | `WindowPC.score_with_masks`, `bench_relational.py` |
| 1.5 | `subset_search` / `naive_subset_search` (`relational.py`) | `stage_relational` |
| 1.6 | `MaskCalibrator` (`relational.py`) | `stage_relational`, `config/ts/tier1_relational.yaml` |

Four things worth carrying forward beyond the table below.

* **The algorithm.** Decomposability makes the root value degree-1 in the value
  of any one region, so `p(x) = Σ_u β_u·α_u` over the units of the channel's
  region with no constant term.  Contiguity makes that region the only place
  `x_c` enters.  Together: `p(x_-c) = Σ_u β_u·Z_u` falls out of one downward
  pass for every channel at once, and `p(x_c) = Σ_u β̄_u·α_u` from a
  data-independent `β̄` cached once.  Masks are then a per-window substitution
  at the boundary, which is why an arbitrary mask per window still costs one
  pass.  The two-pass map is checked against the 3·C oracle on every run, and
  that check is the correctness backbone of the paper.
* **Measured, on real FD001** (W=20, C=15, K=8, 23,921 nodes, 256 windows):
  the oracle spends 45 compiled queries / **22.3 s**; the two-pass map spends
  one leaf pass (0.62 s) plus **0.09 s** for two passes over the 1,121-node
  upper circuit, and agrees to 2.4e-4 nats.  ~30× end to end, and the gap grows
  with C.  Wall clock is reported next to node visits because the oracle runs
  on the compiled layer-parallel evaluator while the two-pass map runs on the
  per-node Python one: the visit counts compare the algorithms, the seconds
  compare these implementations of them.  The cost CURVE is
  `poc/time_series/bench_relational.py`; on a small synthetic circuit (W=8,
  K=4) it reads

  | | C=4 | C=8 | masks=1 | masks=4 | masks=16 |
  |---|---|---|---|---|---|
  | node visits, oracle ÷ two-pass | ×10.7 | ×21.2 | — | — | — |
  | wall clock, oracle ÷ two-pass | ×1.01 | ×0.91 | — | — | — |
  | mask reuse: node visits | — | — | 2,329 | 2,329 | 2,329 |
  | mask reuse: wall vs per-mask query | — | — | ×0.04 | ×0.44 | ×2.37 |

  i.e. the work ratio grows linearly in C exactly as claimed and the mask cost
  is flat, while WALL CLOCK on a 745-node circuit is a wash and only turns into
  the 30× above once the circuit is big enough for the leaf layer to dominate.
  Both halves go in the paper; reporting only the FD001 number would be the
  same kind of favourable-slice claim this project has already had to withdraw
  once.
* **The factorisation crossover is now a GATE, not a footnote.** Tier 0 found
  this circuit exactly factorised across channels for its first ~20 epochs.
  Re-measured here on the synthetic task: dependence 0.000 nats at 15 epochs,
  0.007 at 30, 2.03 at 60; on the small test task 1.5e-05 at 45 epochs and 1.33
  at 120.  Under such a model every relational quantity is identically zero and
  every structure is the same product of marginals, so `gate_verdict` returns
  **VOID** rather than PASS/FAIL below `eval.gate_min_dependence_nats`, every
  gate row carries `dependence_nats`, and the headline configs were moved from
  50 to 60 epochs.  A PASS obtained below the crossover would have been a
  statement about nothing.
* **The relational stage runs end to end on real FD001** (wiring check at 8
  epochs, `tier1_relational` on CPU): two-pass vs oracle 3.2e-05 nats; masks
  from full down to 3 observed sensors; subset search 16 passes against the
  per-candidate model's 857 for the same 413 candidates (**53×**, same values);
  mask-conditional calibration worst |FPR−α| **0.013 vs 0.025** mask-blind at
  α=0.05.  Every one of those numbers is from a circuit whose dependence was
  0.0007 nats — i.e. FACTORISED — so they are wiring, not findings, and the
  stage said so in its own log.  The check also exposed a real defect: the
  C-MAPSS loader's channel grouping puts 12 of 15 channels in group 0, so a
  "bank failure" mask was a blackout leaving three sensors; `bank_masks` now
  truncates any group above half the channels.
* **A seventh silent degeneracy, caught in this cycle.** The first version of
  the boundary finder walked the sub-circuit below each candidate unit
  remembering *hits* rather than *visits*, so it enumerated PATHS of a DAG:
  correct, and non-terminating at W=20 (>20 min, no output).  Fixed with a
  visited set; it is the same K^depth trap `move_circuit_` documents, reached
  through a different door.  There is no test that would have caught a
  20-minute hang, which is itself worth remembering.

### 1.1 Channel-blocked vtree with contiguity assertion
In the region-graph builder (`circuits.py`), add a structure in which each channel's
`W` timestep variables form exactly one vtree subtree. Add a validator that fails
loudly when channel scopes are not contiguous — this is a hard precondition for 1.3
and 1.4, and silent violation would produce plausible wrong numbers. Order the
channels within the top structure by Chow–Liu on channel-level mutual information;
that hybrid is the candidate for keeping density while gaining cheap queries.

### 1.2 Feasibility spike — **the kill gate** (1 day, do this first)
The 2026-08-06 re-measurement found that channel blocking *hurt* real-data detection
(+0.040 AUROC and -27 nats when the blocking was removed), and that Chow–Liu with one
leaf component tops the FD001 sweep at .8539. So measure the frontier before building
on it: blocked-vtree density and detection AUROC vs Chow–Liu and random, at **matched
parameter count**, with the Tier 0 validation split and checkpoint selection.

Pre-register the tolerable detection loss *before* running it. If the blocked
structure cannot hold density inside that budget, the cheap-query story has nothing
to sit on — stop, and write up the negative result plus the benchmark.

### 1.3 Two-pass relational map
New `WindowPC.relational_map(X)`: one upward pass plus one outside/downward pass over
the compiled evaluator, returning `log p(x_c)`, `log p(x_-c)` and `log p(x)` for **all**
channels simultaneously. The existing `typed_scores` (`circuits.py:521`) costs
`3 * C` circuit evaluations; keep it as the reference oracle and add a test asserting
the two-pass result matches it to float32 tolerance. That test is the correctness
backbone of the paper.

### 1.4 Arbitrary missing-sensor masks
Generalize `score_with_missing` (`circuits.py:506`) from a fixed dead-channel list to
per-window boolean channel masks, batched, evaluated by subtree deletion rather than
one re-run per mask. Instrument and report measured cost as a function of `C` and of
the number of distinct masks.

### 1.5 Subset search under partial observation
Beam/greedy search over channel subsets using `R_S(x)` as the value function, reusing
inside values across candidates so the search cost is bounded rather than one circuit
pass per candidate. This is the concrete separation from Lüdtke, Bartelt and
Stuckenschmidt (2022), *Outlier Explanation via Sum-Product Networks*
(https://arxiv.org/abs/2207.08414), who re-query per subset.

### 1.6 Mask-conditional calibration
Estimate the per-mask null distribution of `R_c` from healthy **validation** windows so
that a single alarm threshold keeps its false-alarm rate as sensors drop out. Cheap
here (same circuit, reused passes), expensive for the baselines (a refactorization per
mask). That asymmetry is a headline number, and it must be reported as measured cost,
not as asymptotic argument.

---

## Tier 2 — baselines that make the result credible (~2 days)

The reviewer objection is: *"a Gaussian mixture computes these marginals and
conditionals too."* The comparison must be method-vs-method on a matched query.

- **Block-Gaussian conditional.** `GaussianConditional.attribute` (`explain.py:86`)
  currently sums scalar coordinate quadratic terms and omits channel-dependent
  normalization constants, so it does not compute the same joint channel-block query
  as the PC. The current .8807 (Gaussian) vs .8625 (PC conditional) FD001 result is
  therefore uninterpretable in **either** direction. Implement the true block
  conditional with normalizers and a Schur complement per mask, instrumented for cost.
- **Full-covariance GMM conditional**, K in {3, 10}, same block query, same instrumentation.
- **Lüdtke et al. (2022) subset search reimplemented on your own circuit**, so the
  contrast isolates the algorithm rather than the model class.
- **Genuine permutation SHAP** for the AE and for the PC on one explicitly declared
  game (state the explained score, the coalition value function, the background
  semantics, and the sample budget), with matched budgets.
- **Cost harness**: wall clock, circuit-node visits, and matrix-op counts per query
  type and per mask. Accuracy-per-cost is half of the claim; K is not a proxy for cost.

---

## Tier 3 — protocol and evidence (~3 days, plus the second source)

### 3.1 Relational corruption generator
Faults that are **marginally plausible but relationally broken**: gain mismatch between
coupled channels, channel desynchronization, phase swap, copula shuffle within the
healthy marginal. If a corruption is detectable channel-wise, the relational statistic
has nothing to prove. Cross this with mask patterns (random-k dead, correlated
sensor-bank failure, drift-then-dropout) and a severity ladder.

### 3.2 Metrics
- Macro **per-window** channel ranking, not pooled — pooling mixes between-window
  severity with within-window ranking.
- Precision and recall at a fixed inspection budget.
- Paired differences with **engine-level** bootstrap confidence intervals; overlapping
  windows are not independent resampling units. Three-seed SDs are not a significance test.
- Deletion/repair scored with retained clean pre-corruption windows where available,
  and mean-replacement fidelity labelled as the model-specific test it is.

### 3.3 Freeze and evaluate once
Select structure, learning rate and thresholds on validation; evaluate the test split
once. Pre-register, before looking: the required localization gain, the tolerable
detection loss, and the query-latency/memory budget.

### 3.4 Second, genuinely independent source
PHM08 and N-C-MAPSS are the same simulator family, so they do not answer the
generality objection; C-MAPSS itself is NASA **simulator-generated** data
(https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) with repository-supplied
fault labels on top. Target measured multi-sensor data with real relational faults —
SWaT/WADI or SKAB are the natural fits. The IMS and CALCE loaders already in
`data_prognostics.py` are cheaper to reach but their faults are not relational, so they
do not test the claim.

---

## Explicitly out of scope this cycle

E-values, routing / mixture-of-PCs, curvature vtrees, SOS, multimodal transfer, and RUL
beyond one controlled follow-up. Each adds a distinct scientific claim and another set
of assumptions, and none of them repairs an uninformative conditional model. Revisit
only after the central diagnostic effect survives.

---

## Schedule and decision points

| Days | Work | Gate |
|---|---|---|
| ~~1–3~~ | ~~Tier 0~~ — **done 2026-09-08** | Met: validation split is engine-disjoint, aggregation refuses incomplete stages, 433/433 tests pass |
| 4 | Tier 1.2 feasibility spike | **Kill gate**: blocked structure holds density inside the pre-registered budget, or stop |
| 5–8 | Tier 1.1, 1.3–1.6 | Two-pass map matches the `typed_scores` oracle to float32 tolerance |
| 9–10 | Tier 2 | Matched block-Gaussian/GMM and reimplemented SPN-explanation baselines run on the same queries |
| 11–13 | Tier 3 | Relational corruption x mask grid, engine-level intervals, one frozen test evaluation |
| + | Second source | Effect reproduces off the C-MAPSS simulator |

**Continue** if relational localization and missing-sensor robustness improve
consistently over the matched tractable baselines at a practical measured cost, with a
distinct contribution beyond the 2022 SPN explanation method. A modest detection
disadvantage is acceptable if the diagnostic benefit is large and measured.

**Stop the broad method claim** if block-Gaussian/GMM match the diagnosis and the
missing-data behaviour at equal or lower cost, or if the advantage survives only on one
injection generator or one seed. That outcome is still publishable as a benchmark and a
negative result; it is not the universal-detector thesis.
