# Hand-off — time-series PoC

**Latest handoff — 2026-09-10:** the September 9 diagnosis implementation has been audited: **470 tests passed, one legacy slow test deselected, and all five fresh smoke runs completed**. Read [completion status](DIAGNOSIS_IMPLEMENTATION_STATUS_2026-09-09.md), then the [current implementation plan](IMPLEMENTATION_PLAN_2026-09-10.md). Full scientific studies and fitted diagnosis baselines remain pending. The historical Tier 1 FAIL is unchanged. Sections below retain the September 8 and earlier record; their interpretation of dependence magnitude is superseded by the September 9 review and diagnosis protocol.

_Last updated: 2026-09-08 (§E). §D (2026-08-11, late) follows it and is still
the reference for the leaf-components crossover; §C from earlier that day is
partly superseded — read §D.6 before citing it. The 2026-08-05 diagnostic-suite
pass is in §B, the σ-floor episode from earlier that day in §A, the 2026-08-03
hand-off in §0, two more archived after it. Nothing was deleted._

Read `CLAUDE.md` (hard constraints), then
[`poc/time_series/launch/README.md`](poc/time_series/launch/README.md) (how to
run anything) and [`data/README.md`](data/README.md) (what is real and what is
injected). This file is the state of play and the next actions.

**New here, or back after a while?** [`EXPLAIN.md`](EXPLAIN.md) walks through
how the whole pipeline works — data → structure → circuit → queries → the seven
stages — in plain language, with diagrams and worked numbers. Three documents,
three jobs: `CLAUDE.md` is the contract, `EXPLAIN.md` is the machine, this file
is the state of play.

The thesis was re-scoped on 2026-09-08 after the review in
[`POC_REVIEW_2026-09-08.md`](POC_REVIEW_2026-09-08.md); the plan of record is
[`IMPLEMENTATION_PLAN_2026-09-08.md`](IMPLEMENTATION_PLAN_2026-09-08.md), and
§E below is the state of play. **Every AD number in §D and earlier predates the
Tier 0 validation split and is on different data** — those sections remain the
reference for how the earlier findings were reached and what the degeneracies
cost, not for the numbers themselves.

**If you read one thing: §E.1. The Tier 1 kill gate has RUN and FAILED, and it
failed in the direction nobody had written down — the channel-blocked structure
holds density comfortably and loses DETECTION, because it carries six times
less cross-channel dependence than Chow-Liu. That dependence is the quantity
the whole diagnosis method reports. §E.2 says what the result does not mean.**

---

## E. LATEST (2026-09-08) — Tier 1 built, and its kill gate failed

### E.0 What happened since §D

§D.5's four steps were overtaken. `POC_REVIEW_2026-09-08.md` re-scoped the
thesis from "a universal anomaly detector" to one question — *can a compact
tractable joint model retain useful relational fault localisation when the
available sensor set changes, at lower query cost than equally capable
alternatives?* — and `IMPLEMENTATION_PLAN_2026-09-08.md` is what must be built
to answer it. Two tiers have landed today:

* **Tier 0** (evaluation and provenance repairs) — engine-disjoint validation
  split, checkpoint selection, run provenance, tie-grouped AP, final-endpoint
  windows, explicit nominal levels, honest naming of the AE baseline. 27 tests.
  **Every AD number recorded before it is on different data.**
* **Tier 1** (this section) — the channel-blocked structure, the two-pass
  relational map, arbitrary missing-sensor masks, subset search,
  mask-conditional calibration, and the kill gate that decides whether any of
  it is worth building on. 18 tests. Full suite **451 passed / 0 failed**.

### E.1 THE GATE, ANSWERED — FAIL, and not for the expected reason

`logs/ts/tier1_kill_gate/`, real C-MAPSS FD001, K=8, `leaf_components: 3`,
60 epochs, 3 seeds, 12/12 runs `ok`, 15/15 rows used. All four arms at
**174,240 parameters** — a binary vtree over d variables has d−1 internal
regions whatever its shape, so "matched capacity" here is exact rather than
approximate. Scored on the Tier 0 validation split (engine-disjoint, half for
NLL/checkpointing, half contaminated for a labelled detection score); the test
split was not touched.

| structure | val NLL | val AUROC | dependence (nats) | blocked |
|---|---|---|---|---|
| `chow_liu` | 58.3 ±8.2 | **0.8268 ±0.027** | **34.2** | no |
| `random` | 70.1 ±12.7 | 0.7994 ±0.009 | 23.9 | no |
| `channel` | **40.5 ±8.3** | 0.7937 ±0.020 | 6.0 | yes |
| `channel_blocked` | 45.3 ±12.5 | 0.7934 ±0.027 | 5.6 | yes |

Pre-registered budget (in the config, before the run): the candidate may lose
≤ 0.02 detection AUROC and ≤ 5% relative held-out NLL against the best
NON-blocked arm. Measured: **ΔAUROC −0.0334 → FAIL. ΔNLL −22.4% → inside.**

Three readings, and the second is the one that matters.

**1. Blocking BUYS density and SELLS detection.** Both blocked arms beat both
unblocked ones on held-out NLL by 22–42%, and lose ~0.033 AUROC. The plan's own
§1.2 sentence — "if the blocked structure cannot hold density…" — anticipated
the wrong failure: it holds density comfortably. The loss is detection alone,
consistent in every seed (−0.033, −0.039, −0.029), though the three-seed sd is
±0.027 and **a three-seed sd is not a significance test** (plan §3.2). The rule
was pre-registered on the mean; the mean fails it.

**2. The mechanism is in the dependence column, and it is worse news than the
AUROC gap.** `channel_dependence` measures |Σ_c log p(x_c) − log p(x)|, the
nats of cross-channel dependence the fitted circuit actually carries. The
blocked circuits carry **5.6–6.0**; Chow-Liu carries **34.2**; even `random`
carries 23.9. The relational statistic R_S is *computed from* that dependence.
So the boundary that makes the relational map cheap is the same boundary that
starves the quantity the map reports — a correct, fast algorithm running on a
model with comparatively little to say. That is a structural objection to the
design, not a tuning problem, and it is the thing to answer before any repair.

**3. The channel ORDER is noise.** `channel_blocked` (channels ordered by
Chow-Liu on channel-level MI) and `channel` (identity order) come out at 0.7934
vs 0.7937 and cross over between seeds — seed 0 favours identity by +0.009,
seed 1 favours Chow-Liu by +0.006. Plan §1.1's hybrid ("order the channels by
Chow-Liu; that hybrid is the candidate for keeping density while gaining cheap
queries") is **not supported**. The blocking does everything; the order does
nothing.

### E.2 What the gate does NOT say

* **It is not §D.1's contrast, and does not revive §C.1's Inversion 2.** §D.1
  compared `chain` against `chain_perm_features` — both chain-family, 32k
  params — and found destroying the blocking worth +0.008 (~1σ), i.e. the null.
  This gate compares a fully channel-blocked 174k structure against a *learned*
  174k one. Different question, different structures, and different data (Tier
  0 split). Both statements stand: within the chain family blocking does
  nothing; against Chow-Liu, blocking costs 0.033.
* **It does not say the method is wrong.** Everything in §E.4 is measured and
  correct. It says the structure this method requires is not, as built, a
  structure worth deploying on FD001.
* **It is one dataset, one anomaly generator, three seeds.** The anomalies are
  injected by our own injector (`data/README.md`), which is exactly the credibility
  gap the plan's Tier 3.4 exists to close.
* **It does not license picking a different candidate and re-reading.**
  `channel` beat `channel_blocked` in seed 0 and lost in seed 1; choosing the
  winner after the fact is how this project has fooled itself before. Any
  repair is a NEW pre-registration on fresh seeds.

### E.3 What was built, and where it lives

The algorithm, in one paragraph. Decomposability makes the circuit's value a
degree-1 polynomial in the value of any one region, so `p(x) = Σ_u β_u·α_u`
over the units of a channel's region, with no constant term. Contiguity — each
channel's W timestep variables forming exactly ONE region — makes that region
the only place `x_c` enters the circuit. Together: `p(x_-c) = Σ_u β_u·Z_u`
falls out of a single downward pass **for every channel at once**, and
`p(x_c) = Σ_u β̄_u·α_u` from a `β̄` computed once and cached because it does not
depend on the data. A missing sensor is then a substitution at that boundary
rather than a re-run, so a batch in which every window has a *different* sensor
failure still costs one pass.

| plan item | code | entry point |
|---|---|---|
| 1.1 structure + contiguity assertion | `src/probabilistic_circuits.py` §6c: `channel_blocked_vtree`, `chow_liu_channel_order`, `block_boundary_units` | `WindowPC(vtree_method="channel_blocked")` |
| 1.2 the kill gate | `pipeline.py`: `stage_structure_gate`, `gate_verdict`, `channel_dependence`; `circuits.py`: `match_K`, `structure_param_count` | `config/ts/tier1_kill_gate.yaml` → `run_tier1_gate.py` |
| 1.3 two-pass map | §6c: `inside_log_values`, `outside_log_values`, `RelationalCircuit.relational_map` | `WindowPC.relational_map` |
| 1.4 arbitrary masks | §6c: `masked_log_prob`, `_boundary_values` | `WindowPC.score_with_masks`, `bench_relational.py` |
| 1.5 subset search | `relational.py`: `subset_search`, `naive_subset_search` | `stage_relational` |
| 1.6 mask calibration | `relational.py`: `MaskCalibrator`, `mask_library` | `config/ts/tier1_relational.yaml` |

Two new pipeline stages (`structure_gate`, `relational`) go through the same
runner, provenance and aggregation as the other five, so nothing here needs a
special code path. `TIERS=6 bash poc/time_series/launch/run_workstation.sh`
runs the gate and runs the relational config **only on PASS**.

### E.4 What stands regardless of the verdict

These are engineering facts about the algorithm, not claims about the model:

* **Correctness.** The two-pass map agrees with the 3·C `typed_scores` oracle
  to **2.4e-04 nats** on real FD001 (|log p| ~ 60–200 nats) and **3.2e-05** in
  the 8-epoch wiring run. That check runs inside `stage_relational` on every
  run and raises if it moves; it is the correctness backbone of the method.
* **Cost, on real FD001** (W=20, C=15, K=8, 23,921 nodes, 256 windows): the
  oracle spends 45 compiled queries / **22.3 s**; the two-pass map spends one
  leaf pass (0.62 s) plus **0.09 s** over the 1,121-node upper circuit. ~30×.
* **Cost curve** (`bench_relational.py`, small synthetic circuit): work ratio
  ×10.7 at C=4 and ×21.2 at C=8 — linear in C, as claimed — while WALL CLOCK is
  a wash (×1.01, ×0.91) because the oracle runs on the compiled layer-parallel
  evaluator and the two-pass map on the per-node Python one. Mask cost is flat
  (2,329 visits at 1, 4 and 16 masks) and crosses the per-mask query between 4
  and 16 masks (×0.04 → ×2.37). **Report both halves.** Quoting only the FD001
  30× would be the same favourable-slice claim this project has withdrawn once.
* **Subset search**: 16 passes against the per-candidate cost model's 857 for
  the same 413 candidates, with the candidate VALUES agreeing to **3.1e-05
  nats** — the concrete separation from Lüdtke et al. (2022), measured on the
  same circuit so the contrast is the algorithm and not the model class.
  (Which subset each window ends up choosing agrees only 38–69% of the time,
  and that is expected rather than alarming: on a factorised wiring-run circuit
  most candidates tie to round-off, so the argmax is noise while the values it
  is taken over are exact. Re-measure agreement on a circuit that carries
  dependence before quoting it either way.)
* **Mask-conditional calibration**: worst |FPR − α| **0.013** mask-conditional
  vs **0.025** mask-blind at α=0.05 over 7 masks. Wiring-run numbers, on a
  factorised circuit — the mechanism works, the effect size means nothing yet.

### E.5 Traps found this cycle

* **A SEVENTH silent degeneracy, and the most expensive kind: a hang.** The
  first version of `block_boundary_units` walked the sub-circuit below each
  candidate unit remembering *hits* rather than *visits*, so it enumerated the
  PATHS of a DAG. Correct, and non-terminating at W=20 — over 20 minutes with
  no output, on a function that takes 0.37 s once fixed. Same K^depth trap
  `move_circuit_` documents, reached through a different door. **No test in
  this repo would have caught it**, because a hang is not a wrong answer;
  the small-W tests all passed.
* **The factorisation crossover is now a GATE, not a footnote.** Tier 0 found
  this circuit exactly factorised across channels for its first ~20 epochs.
  Re-measured: synthetic, dependence 0.000 nats at 15 epochs, 0.007 at 30,
  2.03 at 60; the small test task, 1.5e-05 at 45 and 1.33 at 120. Under such a
  model every relational quantity is identically zero AND every structure is
  the same product of marginals, so `gate_verdict` returns **VOID** rather than
  PASS/FAIL below `eval.gate_min_dependence_nats`, every gate row carries
  `dependence_nats`, and the Tier 1 configs run 60 epochs, not 50. A PASS
  obtained below the crossover would have been a statement about nothing.
* **A pre-registration mismatch, caught mid-run.** The config said "against the
  best structure that is NOT channel-blocked"; the first implementation of
  `gate_verdict` compared against every other arm, which would have failed the
  gate whenever the blocked-but-differently-ordered `channel` arm won — a
  statement about the channel ORDER, not about blocking. Fixed while seed 0 was
  still training; the stricter number is still reported as `delta_auroc_vs_any`.
  **Consequence for this batch:** the per-seed `VERDICT` rows in
  `results.jsonl` were written by the stage before the fix and use the old
  comparator. `run_tier1_gate.py` recomputes from the arm rows and is
  authoritative; the rows are now labelled "this seed only".
* **A mask library that asked an impossible question.** The C-MAPSS loader's
  channel grouping puts 12 of the 15 surviving channels in group 0, so the
  "sensor bank failure" mask left three sensors alive — a blackout, not a
  fault, scoring the model on something no method could answer and drowning the
  informative masks in the same table. `bank_masks` now truncates any group
  above half the channels.
* **Tolerances scale with the nat range, so do not tighten them to the toy
  value.** The oracle check agrees to 2.4e-04 nats on the trained FD001 circuit
  (|log p| ~45–70 nats accumulated over 23,921 nodes), 3.2e-05 in the 8-epoch
  wiring run, and ~1e-05–1e-06 on the toy circuits in the test file. All of it
  is float32 round-off — the identity is exact — but the ABSOLUTE size moves
  with the magnitude of log p and the depth of the circuit.
  `eval.oracle_tolerance` is 1e-3 for that reason.

### E.6 What to run, in order

```bash
export PYTHONPATH=.

# 0. the fast checks — ~30 s.  A failure here means the batch would produce
#    numbers that look reasonable and are wrong.
pytest tests/test_ad_diagnostics.py tests/test_rul_diagnostics.py \
       tests/test_experiment_hygiene.py tests/test_tier1_relational.py -q

# 1. re-read the gate (it is already run; this is free)
python -m poc.time_series.run_tier1_gate logs/ts/tier1_kill_gate
#    exit 0 PASS · 2 FAIL · 3 VOID (every arm factorised) · 4 NO_COMPARATOR

# 2. THE DECISION, and it is not a code change (§E.7)
```

There is no "step 3 config to launch". The gate said stop, and the next action
is a choice about the research direction, not another batch. Running
`config/ts/tier1_relational.yaml` on this structure would produce correct
numbers about a model the gate has already condemned — the same mistake §D.0
records, in a new place.

### E.7 The decision this leaves, stated honestly

Three options, in the order I would defend them:

1. **Write up the negative result plus the benchmark**, which is what the plan
   pre-committed to. The publishable content: the two-pass algorithm with its
   exactness proof and measured cost curve, the missing-sensor query, and the
   finding that *the structural constraint the cheap queries require costs 6×
   the cross-channel dependence the queries are about*. That last sentence is a
   real contribution to anyone else who tries this, and nobody has written it.
2. **One targeted repair, pre-registered afresh**: the blocked structures may
   be starved of cross-channel capacity ABOVE the boundary (they spend K² on
   within-channel timestep structure, which is exactly why their NLL is so
   good). A blocked vtree with more units at the top levels, or Chow-Liu
   *within* each channel block, tests that directly. It needs a new
   pre-registration, fresh seeds, and the same dependence column as the
   read-out — and it must be one experiment, not a search.
3. **Change the operating regime.** The gate scores detection of injected
   anomalies on a full sensor set. The thesis is about *diagnosis under missing
   sensors*, and the blocked structure's advantage there is COST, not
   capability: an unblocked circuit answers the same masked query exactly via
   `log_marginal` (`score_with_missing` works on any structure), it just cannot
   reuse the boundary across masks. The per-mask refactorisation cost belongs
   to the Gaussian/GMM baselines of Tier 2, and has not been measured yet.
   Making localisation-under-mask the primary axis is defensible — but it is a
   different pre-registration, it has to be written down before it is run
   rather than chosen because the first axis failed, and on this structure it
   would be measured on a model carrying 5.6 nats of the dependence it reports.

**Do not draft Paper A's method section on this structure.** What survives is
listed in §E.4 and it is engineering, not a result about engines.

### E.8 Corrections to this file's own record

- **§D.1's "at a healthy leaf setting neither timestep order nor channel
  blocking does anything"** — stands, and is now bounded. It is true *within the
  chain family at 32k params*. Against a learned Chow-Liu vtree at matched
  174k params on the Tier 0 split, full channel blocking costs −0.033 AUROC and
  gains 22% NLL (§E.1). The two are different contrasts, not a contradiction.
- **§D.5's steps 1–3** — superseded by the re-scope, not by a measurement. The
  lc=3 undertraining confound of §D.2 is still open and still worth clearing if
  anyone returns to the detection headline; Tier 1 sidestepped it by fixing
  lc=3 and 60 epochs everywhere and by matching parameters exactly.
- **Everything in §0–§D reporting an AD number** — measured before the Tier 0
  validation split existed. Not wrong, but not comparable to anything measured
  after 2026-09-08.

---

## D. 2026-08-11 (late) — the blocker answered, and why RUL collapses

### D.0 What happened since §C

§C.6 listed four steps. Steps 0, 1 and 3 ran. **Step 2 — patching the headline
configs to the winning cell — was skipped**, so step 3 re-ran all four headline
configs in the same degenerate cell §C.1 had already condemned. That batch cost
~4 h and produced nothing usable. If you read only one operational lesson from
this section, it is that step 2 was not optional bookkeeping; it was the entire
point of step 1.

Two smaller notes on the mechanics. The first launch of the blocker
(`logs/ts/_console/cmapss_structure_x_leaves_20260811_154901.log`) died instantly
with `FileNotFoundError` on its own config: the file was committed at 15:47:34
(`539001e`) and launched at 15:49:01, before the workstation had pulled. The
relaunch at 15:50:47 ran clean. And the conformal layer of §C.3 got its first
end-to-end exercise in the 15:48 smoke run — see §D.4.

### D.1 THE BLOCKER, ANSWERED — 36/36 clean

`logs/ts/cmapss_structure_x_leaves/`, real C-MAPSS FD001, K=8, 3 seeds, all 36
runs `ok`. RegionGraphPC density score. `train_nll` in the summaries is
MISNAMED — `run_ad.py:151` assigns `held_nll` to it, so it is held-out.

| vtree | lc=1 | lc=3 | Δ | NLL 1→3 | params 1→3 |
|---|---|---|---|---|---|
| `chain_perm_features` | 0.7923±0.0029 | **0.8347±0.0048** | **+0.042** | 162.3→97.1 | 15,360→32,160 |
| `chain_perm_blocks` | 0.7512±0.0056 | 0.8278±0.0077 | **+0.077** | 190.8→99.4 | 15,360→32,160 |
| `chow_liu` | **0.8539±0.0048** | 0.8266±0.0038 | −0.027 | 66.3→89.9 | 157,440→174,240 |
| `chain` | 0.7528±0.0081 | 0.8265±0.0068 | **+0.074** | 189.4→97.3 | 15,360→32,160 |
| `time` | 0.8325±0.0060 | 0.8126±0.0044 | −0.020 | 82.9→115.0 | 157,440→174,240 |
| `random` | 0.8388±0.0035 | 0.8089±0.0082 | −0.030 | 67.3→126.8 | 157,440→174,240 |

It is a crossover, and it answers all three pre-registered questions.

**1. The degeneracy is chain-specific, and `leaf_components: 3` fixes it.** All
three chain arms gain +0.04 to +0.08 and shed 65–92 nats. `chow_liu` does not
move in that direction at all. Reading (a) of the config fires: the structure
ablation must be re-read at lc=3 before any structure claim.

**2. §C.1's Inversion 2 is DEAD — it was itself an artefact of the dead cell.**
The "+0.040 AUROC from destroying channel blocking" was measured at lc=1. At
lc=3 the `chain_perm_features` − `chain` gap collapses to **+0.008**, about one
sd (0.0048 / 0.0068). All three chain arms land at 0.8265 / 0.8278 / 0.8347 —
indistinguishable. The correct statement is now the null one: **at a healthy
leaf setting neither timestep order nor channel blocking does anything.** The
2026-08-03 synthetic claim and its 2026-08-06 reversal were both reading the
degeneracy, in opposite directions.

**3. The `random` control is the uncomfortable one.** The pre-registered kill
criterion was "if random is not clearly worst in the chosen cell, structure is
doing nothing and the structure story comes out."

- At **lc=1** random is **2nd of 6** — it beats `time`, `chain` and both perm
  controls. Only `chow_liu` beats it, by +0.015 (~3σ: real, small). The whole
  lc=1 column separates cleanly by parameter count: all three 157k arms above
  all three 15k arms. Capacity is doing the work, not structure.
- At **lc=3** random IS worst (0.8089), so the criterion passes — but the
  winner is `chain_perm_features`, a deliberately scrambled control, and the
  total spread is only 0.026.

**The cell with the best absolute number (chow_liu @ lc=1, 0.8539) is the cell
where our own control says the structure claim is empty.** That tension is the
finding, and it should be stated as such rather than resolved by picking the
flattering cell.

The one clean, cell-independent result: at lc=3 the 32k chain-family arms BEAT
the 174k learned/large arms. **Matching Chow-Liu at 1/5 the parameters** is a
defensible efficiency claim that does not depend on the structure story.

### D.2 Which cell — and the confound that must be cleared first

Do NOT adopt lc=3 blindly. The three large arms get **worse held-out NLL at
lc=3 while gaining parameters** (random 67→127, time 83→115, chow_liu 66→90).
Capacity up, fit down, at fixed `epochs: 50, lr: 0.05` is the signature of
**undertraining, not a capacity effect**. The entire lc=3 large-arm column may
be an optimisation artefact.

So there is currently no cell where every arm is healthy: lc=1 cripples the
chain family, lc=3 may be starving the large arms of epochs.

Cheap resolution, and it must run before the headline configs are patched:
re-run only `chow_liu`/`time`/`random` at lc=3 with substantially more epochs.
If their NLL drops below the lc=1 values, the lc=3 regression was undertraining
and the structure ranking has to be re-read again. If it does not, lc=3 is a
real capacity ceiling for those arms and the chain family at lc=3 is the cell.

### D.3 WHY RUL FAILS — one root cause, not five problems

The headline: **RUL is not separately broken. It is the same degeneracy as the
AD cell, measured by the one query that has nowhere to hide.**

The mechanism, in order:

1. **τ is a separate leaf variable**, appended to the joint (`tau_idx = self.d`,
   `circuits.py:682`). It is not a function of the sensors. So the ONLY route by
   which an engine's window can change its predicted life is through learned
   **dependencies** between the τ leaf and the x leaves.
2. `tau_where: deep` is already set in `cmapss_rul.yaml:54`, so the known-bad
   `root` coupling of §3 is NOT the cause this time. The coupling is fine; what
   is missing is anything to couple.
3. **The circuit in this cell has learned almost no dependencies.** The
   "structural" score is defined at `circuits.py:528` as
   `−log p(x_c | x_−c) + log p(x_c)` — pointwise mutual information, i.e.
   exactly the learned dependency mass. From the same crossed sweep:

   | vtree | structural-only AUROC, lc=1 | lc=3 |
   |---|---|---|
   | `chain` | **0.5820±0.0030** | 0.7791±0.0027 |
   | `chain_perm_blocks` | 0.5706±0.0068 | 0.7832±0.0079 |
   | `chain_perm_features` | 0.6788±0.0035 | 0.7805±0.0149 |

   **0.58 is chance.** In that cell the density is close to a product of
   independent marginals.
4. No dependencies ⟹ τ ⊥ x under the model ⟹ `p(τ|x) = p(τ)`, the same curve
   for every engine ⟹ `E[τ|x]` constant. Measured by the guardrail: sd **0.759
   cycles** against a target spread of 34.5.

**The guardrail message and the 0.5820 structural score are the same fact
measured two ways.**

**Why AD survived and RUL did not.** The anomaly score is `−log p(x)`, carried
mostly by the leaf MARGINALS. A product of independent Gaussians is still a
working detector — it is literally the `diagonal Gaussian` baseline. So AD
degrades *gracefully*, down to roughly its own baseline (0.7528). RUL has no
marginal fallback: `p(τ)` alone says nothing about a specific engine. Strip the
dependencies and AD limps; RUL goes to a constant. That is why RUL is the most
sensitive detector of this defect in the whole pipeline, and why it fails on
some seeds and not others — whether the latent posterior fully collapses
depends on init jitter.

This also explains the accuracy gap with no extra hypothesis. Target spread
~34.5; the circuit gets RMSE ~25; ridge gets ~18. A model near "predict the
global mean" scores close to the target's own spread. The circuit is much
nearer that than to a model actually reading the sensors.

**CAVEAT, and the check that settles it.** The 0.5820 comes from the crossed
sweep's AD stage at `K=8`, NOT from the RUL runs, which use `rul_K: 12` and the
τ-augmented structure. It is the closest matched cell, not a direct measurement
of the failing model. The decisive test is one config line: run `cmapss_rul` at
`leaf_components: 3`. If this account is right the degeneracy failures should
largely vanish and RMSE should fall toward the baselines. **If they persist at
lc=3, RUL has an independent defect and the τ coupling itself is the suspect.**

### D.4 What tonight's headline batch exposed anyway

All of the below was produced in the dead cell and is void as measurement, but
three of the four are bugs that survive the cell and must be fixed regardless.

**(a) RUL: 5 ok, 0 skipped, 7 FAILED in 67.7 min.** Every failure is the §B.5
guardrail firing with `DegenerateModelError`, sd of `E[τ|x]` at 0.759–1.32
cycles against target spreads of 33.2–38.2. This is a **7th degeneracy**, and
the first one caught by a guardrail instead of shipped as a result. The
guardrail worked exactly as designed.

**(b) The RUL summary table is CONTAMINATED — fix before reading anything.**
The aggregator reads run directories; a failed re-run leaves the PREVIOUS
`metrics.json` in place. Concretely:
`logs/ts/cmapss_rul/censor_frac-0.7/seed1/metrics.json` is dated **4 Aug 15:59**
next to a `status.json` reading `failed` at **2026-08-11T17:52:19**. That is why
rows still report "3 seeds" when 7 of 12 runs died. The table is a blend of
tonight's 5 survivors and stale 4-August numbers, which per §A are themselves
pre-σ-floor and void. **A failed run must not contribute to a summary.**

**(c) The conformal path IGNORES `alpha`.** From
`logs/ts/cmapss_calibration/summary.md`, FD001:

```
SurvivalPC + split conformal (cqr) · a=0.10   PICP 0.9702±0.0043   MPIW 73.6706±1.6896
SurvivalPC + split conformal (cqr) · a=0.20   PICP 0.9702±0.0043   MPIW 73.6706±1.6896
```

Identical intervals for two different requested coverages (FD002 likewise:
0.9850 / 96.2659 both). `interval_score` differs (80.54 vs 77.10) only because
the SCORE formula uses alpha. The CQR baseline responds correctly
(0.8545 → 0.7465), so this is specific to the PC conformal path: alpha reaches
the scoring but not the interval construction. **The §C.3 conformal result is
real but currently pinned at one coverage level regardless of what is asked
for.** Every conformal number on record needs re-reading after this is fixed.

**(d) The censoring feature still loses to deleting data.** At `[last]` with
70% censoring — where exact censored likelihood should win biggest — it is
**35.08±1.65 RMSE vs 27.42±0.67** for simply dropping censored rows. It does win
at `[all]` (24.98 vs 27.25 at 0.5), so the effect flips by endpoint.

A mechanism for (d), offered as HYPOTHESIS, not measurement: if `p(τ|x)` has no
x-dependence, the censored-likelihood term — which integrates `P(τ > t | x)`
over the tail — cannot sharpen per-engine prediction, so its gradient lands on
the only component with capacity, the shared τ MARGINAL, dragging it toward
longer lives. Dropping censored rows instead trains on clean fully-observed
targets. The prediction is that the damage scales with how much the tail term
dominates, which matches: worst at `[last]` + 0.7. **This is consistent with
T1's death (§2) but does not re-open it** — it says the censoring machinery is
doing correct work on a model with no capacity to use it, so T1 has not yet had
a fair test at a healthy cell.

**(e) Not a bug, but do not tune for it.** RUL is ~250–345 s per run against
ridge's ~0.4 ms. `log_pmf` (`circuits.py:818`) runs one full circuit evaluation
PER BIN — reading an exact conditional off a discretised joint costs `n_bins` ×
a density evaluation, every batch. That is the price of the exact-conditional
formulation, not something tuning will recover.

### D.5 What to run, in order (SUPERSEDES §C.6)

```bash
# 0. THE THREE FIXES FIRST — all three survive the cell choice
#    (i)   failed runs must not contribute to summaries (§D.4b)
#    (ii)  plumb `alpha` into the PC conformal interval, not just the score (§D.4c)
#    (iii) nothing else — do NOT touch the guardrails

# 1. clear the lc=3 confound (§D.2): chow_liu/time/random only, more epochs
#    if their NLL drops below the lc=1 values, re-read D.1 before proceeding

# 2. the RUL root-cause test (§D.3) — ONE config line, decides the diagnosis
#    set leaf_components: 3 in config/ts/cmapss_rul.yaml, then:
bash poc/time_series/launch/run_config.sh config/ts/cmapss_rul.yaml
#    expect: the 7 DegenerateModelError failures largely vanish, RMSE falls
#    toward ~18. If they PERSIST, the tau coupling is the suspect, not the cell.

# 3. ONLY THEN patch leaf_components in the four headline configs and re-run.
#    This is the step that was skipped tonight (§D.0). Skipping it again wastes
#    another 4 h.
JOBS=3 THREADS=4 DEVICE=cpu TIERS="1 2 5" bash poc/time_series/launch/run_workstation.sh
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts
```

**Still do NOT draft the Paper A section.** What survives untouched: exact
attribution 0.902 vs 0.498 sampling-SHAP, completeness at 1.5e-5 nats,
box-query exactness at 6e-6, T1's death, and now D.1's parameter-efficiency
result. Everything else waits on step 3.

### D.6 Corrections to this file's own record

- **§C.1 Inversion 2 ("channel BLOCKING is the defect")** — DEAD (§D.1). It was
  measured inside the degenerate cell; at lc=3 the effect is +0.008, ~1σ. Both
  it and the §B.3 synthetic claim it reversed are artefacts. The surviving
  statement is the null: order and blocking both do nothing at a healthy cell.
- **§C.1 Inversion 1 ("the chain wins" is dead)** — stands, but weaken it. At
  lc=3 `chain` (32k params) ties `chow_liu` (174k). The chain is not a better
  structure; it is a cheaper one that is no worse.
- **§C.1 Inversion 3 (curvature vtrees dead)** — untouched by tonight's run;
  curvature arms were deliberately excluded from the crossed sweep.
- **§C.2 "what to read off it"** — all three questions now answered in §D.1.
  The section is superseded, not wrong.
- **§C.3 conformal results** — every PICP/MPIW figure needs re-reading after the
  alpha fix (§D.4c). The *relative* result (conformal beats raw predictive, and
  beats the CQR baseline on width) is unlikely to change sign; the levels will.
- **§C.6** — superseded by §D.5. Its step 2 was the one that mattered and it was
  the one skipped.
- **§B.5 "two blind guardrails"** — the RUL degeneracy guardrail is no longer
  blind: it caught 7 real failures tonight (§D.4a) that would otherwise have
  been reported as numbers.

---

## C. 2026-08-11 (earlier) — the void batch, and the conformal layer

### C.0 Why this section exists

Two things happened that this file did not record.

1. **The §B.10 batch ran on 2026-08-06.** Its results are in `logs/ts/` and
   nowhere else. They inverted three claims and exposed a sixth degenerate
   cell. Because the doc was never updated, work continued for five days
   against premises the data had already killed — including a design document
   (`poc/EVALUES_Brainstorm.md`) written on 2026-08-11 whose §4.5 treats a
   question the 08-06 logs had closed.
2. **A conformal / e-value layer was built on 2026-08-11** on top of the
   circuit, with a sequential-monitoring half. It is standalone and not yet a
   pipeline stage.

The process lesson is the same one §3 keeps making, one level up: **a result
that lives only in `logs/` has not been recorded.** Aggregate output is not a
hand-off. If a batch settles something, it goes in this file the same day or it
will be re-derived, or worse, contradicted by work built on the old premise.

### C.1 The 2026-08-06 re-measurement — three inversions, one dead cell

Real C-MAPSS FD001, 3 seeds, one commit. Sources: `logs/ts/cmapss_structure/`
and `logs/ts/capacity_sweep/`.

**Inversion 1 — "the chain wins" is DEAD.**

| structure | AUROC | NLL | params |
|---|---|---|---|
| `chow_liu` | **0.8539 ± 0.0048** | 66.3 | 157,400 |
| `chain_grouped` | 0.8453 ± 0.0041 | 89.2 | 138,200 |
| `random` | 0.8388 ± 0.0035 | 67.3 | 157,400 |
| `chain_perm_features` | 0.7923 ± 0.0029 | 162.3 | 15,360 |
| `chain` | 0.7528 ± 0.0081 | 189.4 | 15,360 |
| `chain_perm_blocks` | 0.7512 ± 0.0056 | 190.8 | 15,360 |

The chain is second-worst. A **random** vtree beats it by +0.086. `chow_liu`
tops every FD001 baseline on record.

Read the two comparisons separately, because only one of them is
capacity-matched. `chain` vs the perm controls IS matched (15,360 params each);
`chain` vs `chow_liu`/`random` is NOT (10x fewer parameters), so that row pair
says "the chain is small", not only "the chain is bad".

**Inversion 2 — the blocking is the DEFECT, not the advantage.** Within the
matched triple: destroying the timestep ORDER costs 0.0016 AUROC (nothing).
Destroying the channel BLOCKING as well **gains +0.040 AUROC and −27 nats**.
The synthetic measurement said the opposite (+0.07 vs +5.17 nats, §B.3). The
sign reversed on real data. So the chain models neither time order nor useful
grouping — `chain_perm_features` is simply a better circuit at the same size.

**Inversion 3 — curvature vtrees are dead**, dominated by Chow-Liu on both
AUROC and NLL. That line is closed for this domain.

**The sixth degenerate cell, and it is the expensive one.** From
`logs/ts/capacity_sweep/`: `vtree: chain` with `leaf_components: 1` is FLAT at
AUROC 0.7502–0.7528 across K = 2..16, i.e. across 1,428 → 88,704 parameters
(**62x**), NLL flat at ~189–199. At `leaf_components: 3` the same structure
scales normally (0.8128 → 0.8370).

`leaf_components` defaults to 1 (`poc/time_series/config.py:144`), and
`cmapss_ad.yaml` sets `vtree: chain` + `leaf_components: 1` explicitly while
`cmapss_explain`, `cmapss_rul` and `cmapss_calibration` set `chain` and inherit
the default. **All four headline configs ran inside the degenerate cell.** Every
ad / explain / rul / calibration number on record is therefore void — not
wrong-ish, but produced by a model that provably cannot use its own capacity.

This also means `cmapss_structure.yaml` is currently a misleading experiment:
it varies the vtree at `leaf_components: 1`, so its chain arms sit in the dead
cell while its `chow_liu` arm does not. It compares a crippled chain against a
healthy Chow-Liu and reports the difference as "structure".

### C.2 The blocker, and the config that clears it

Neither existing sweep can pick a replacement cell — `capacity_sweep` varies
K × leaf at fixed `vtree: chain`; `cmapss_structure` varies vtree at fixed
`leaf_components: 1`. The crossed sweep did not exist. It does now:

```
config/ts/cmapss_structure_x_leaves.yaml
  6 structures x leaf_components {1,3} x 3 seeds = 36 runs  (dry-run verified)
  chain | chain_perm_blocks | chain_perm_features | chow_liu | time | random
```

Runs on this config shape are ~14 s each (`logs/ts/cmapss_structure/*/status.json`),
so this is ~10 minutes at `JOBS=3`. **Nothing else should be re-measured until
it has answered**, because every other config inherits the cell it picks.

What to read off it: does `chain` jump from lc=1→3 while `chow_liu` barely
moves (a chain-specific interaction)? Is `random` clearly worst in the winning
cell — if not, structure is doing nothing and the structure story comes out of
the paper? And does `chain_perm_features > chain` survive outside the dead
cell, given that it was measured inside it?

### C.3 The e-value conformal layer (new, 2026-08-11)

`poc/time_series/pc_conformal.py` (1471 lines) + `tests/test_pc_conformal.py`
(48 tests, ~2 s). Full repo suite: **344 pass**. Design:
`poc/EVALUES_Brainstorm.md` §§8–9 (written back with the corrections below).

Read-only with respect to the circuit; needs nothing from
`src/probabilistic_circuits.py` that is not already public (`log_prob`,
`log_marginal`, `log_box`). Contents: `EvalueICAD` (rank-based p→e anomaly
alarm), `EvaluePredictionSet` (HPD-score RUL sets), `CoveragePolicy` + LOO
training, `EvalueMerge`, `ConformalSurvivalBound`, `ConformalTestMartingale`,
and the gates.

**Five things the design did not survive contact with:**

1. **Censored rows were being scored at their censoring bound.** For a censored
   window `tau` is a lower bound, not the label; calibrating a two-sided set at
   `tau_train` puts wrong values in the quantile. Fixed via `delta_cal`.
   Measured cost, 7 seeds: p-fixed coverage 0.770 ± 0.058 mislabelled vs
   0.777 ± 0.042 dropped. A correctness fix with a SMALL effect — one seed read
   0.667 vs 0.738 and looked decisive; that was noise at n_cal ≈ 40.
2. **The `p-fixed` baseline is the instrument, not a baseline.** It is what
   found (1); the e-set's conservatism masked the mislabelling completely
   (coverage 1.000 either way). `report()` now computes it by default.
3. **Per-unit `max` reduction answers a different question than assumed** —
   it guarantees every window of a new unit is covered, which is near-vacuous
   here (11.6 of 12 bins). `reduce="random"` is now the default.
4. **kappa < 0.5 has infinite variance.** E[f(P)] = 1 exactly for all
   kappa in (0,1) but E[f²] diverges below ½, so `mean_e` stops concentrating.
   The §6 kappa ablation is not a free knob.
5. **The Task-A scale trap reappears in Task B** via `sum_i S_i` in the set
   threshold; `log_s_max` is load-bearing and `sum_concentration` is reported.

**Two measured results that bear on whether the line is worth pursuing:**

- **The degeneracy gate fires on real models.** `assert_conditional_varies`
  refused 1 of 8 seeds at ordinary settings (TV 4.4e-4 — `p(tau|x)` essentially
  constant). A full valid-looking table on that seed was one check away.
- **The e-set is near-vacuous at fleet scale.** n_cal ≈ 40 units, 12 bins,
  alpha = 0.20: e-set coverage 1.000 at mean size **10.7/12**, against p-fixed
  size ~4. Post-hoc validity costs nearly all the informativeness, and the
  resolution floor 1/(n+1) over UNITS is the binding constraint. For Task B at
  fleet scale the honest answer is currently **no**.
  `ConformalSurvivalBound` — one-sided, and the only piece that can KEEP
  censored units, because `log_box` turns "alive at c" into a valid bound on
  the unobservable score — is not limited this way and remains the best reason
  to keep the line.

### C.4 The martingale, and the trap under it

Vovk's conformal test martingale: smoothed ONLINE conformal p-values (i.i.d.
uniform under exchangeability), betting function with integral 1,
`M_T = prod_t f(p_t)`, Ville gives `P(sup_T M_T >= 1/alpha) <= alpha`. Default
bet is the mixture over eps, so there is no free parameter.

**The naive product is a trap, not a blow-up.** Multiplying `EvalueICAD`'s
fixed-calibration e-values is not a martingale. At the default kappa=0.5 it
respects the bound BY ACCIDENT — the bet's drift is `log k + 1 - k = -0.19` per
step, so it goes bankrupt before the shared-calibration bias matters:

| kappa | drift/step | alarm rate (exchangeable null, nominal 5%) |
|---|---|---|
| 0.50 | −0.193 | 2.3% |
| 0.80 | −0.023 | **13.7%** |
| 0.95 | −0.001 | **14.3%** |

Since kappa is exactly what the design doc says to ablate, that is a trap.
Power does not separate them — both detect a strong changepoint 100%.

**OVERLAPPING WINDOWS INVALIDATE THE MONITOR.** The largest trap in this half
and invisible without a control. At stride < window consecutive scores are
autocorrelated, so exchangeability is false before any degradation. Running the
monitor on the HEALTHY region only (RUL > 100, nothing to find):

| stride (window=6) | full-life fired | median lead | **healthy-only fired (nominal 5%)** |
|---|---|---|---|
| 6 (disjoint) | 33% | 13 cyc | **2%** |
| 2 | 75% | 30 cyc | **17%** |
| 1 | 90% | 130 cyc | **57%** |

The full-life column reads as a spectacular win from denser windowing and is
entirely artefact — at stride 1 the "lead time" equals the RUL cap because it
fires on the first window of every unit. `assert_non_overlapping` refuses this
now; no correction repairs it, the null itself is false.

**At the only valid setting the monitor is weak but honest**: ~33% of units,
median lead ~13 cycles of a 130-cycle horizon. The score is NOT the bottleneck
— within-unit corr(anomaly score, RUL) is **−0.62**, 93% of units below −0.3.
What limits it is that the score only crosses the healthy fleet's 95th
percentile at RUL < 20. **Never quote a lead time without the healthy-region
control beside it**: both knobs that improve it (stride, dropping the warm
start) also raise the healthy firing rate.

### C.5 Data status

`data/cmapss` FD001–FD004 present on BOTH the laptop and `jawa17-desktop`, so
tiers 1, 2, 4, 5 are fully runnable. N-C-MAPSS was being added on 2026-08-11
(DS01–DS03 are what the configs need: `ncmapss_ad` grids `[DS01, DS02, DS03]`,
`ncmapss_rul` uses DS02). Everything else — ESA-ADB, SMAP/MSL, PHM08, battery,
bearings — still missing, so **this batch buys real-engine evidence, not
cross-domain evidence.**

Two operational notes. `skip_if_missing_data: true` means a tier with absent
data SKIPS silently; use `TIERS="1 2 5"` rather than including 3 so the absence
is explicit. And the workstation ran out of disk during the N-C-MAPSS unzip —
`~/.cache/huggingface` was 447 GB, pip cache 39 GB, Trash 37 GB. Leave real
headroom before a batch: the `.npz` parse cache and `logs/ts` both grow during
one, and an ENOSPC mid-batch wastes the run even though `status.json` resumes.

### C.6 What to run, in order

```bash
# 0. preflight
PYTHONPATH=. python -m poc.time_series.check_data
PYTHONPATH=. python -m pytest tests/test_ad_diagnostics.py \
    tests/test_rul_diagnostics.py tests/test_experiment_hygiene.py \
    tests/test_pc_conformal.py -q
bash poc/time_series/launch/run_smoke.sh

# 1. THE BLOCKER — 36 runs, ~10 min at JOBS=3
bash poc/time_series/launch/run_config.sh config/ts/cmapss_structure_x_leaves.yaml
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts/cmapss_structure_x_leaves

# 2. patch leaf_components (and possibly vtree) in the four headline configs
#    to the winning cell — the config hash changes, so step 3 re-runs them

# 3. the batch
JOBS=3 THREADS=4 DEVICE=cpu TIERS="1 2 5" bash poc/time_series/launch/run_workstation.sh

# 4. read it
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts
```

**Expect the guardrails to REJECT runs that previously produced numbers.** That
is the point; a rejected run is a result and the message names the channels or
the sd that failed. FD002/FD004 are the ones to watch — 60/630 features with
MAD ≡ 0 under six operating conditions.

**Still do NOT draft the Paper A section.** What survives untouched from §2:
exact attribution 0.902 vs 0.498 sampling-SHAP, completeness at 1.5e-5 nats,
box-query exactness at 6e-6, and T1's death. Everything else waits on step 3.

### C.7 Corrections to this file's own record

- **§2 "Structure ablation — the chain wins on AUROC *and* likelihood"** —
  DEAD, and inverted (§C.1). Do not cite it.
- **§B.3 "the chain's advantage is BLOCKING, not temporal order"** — the first
  half survives (order is worth ~nothing), the second half reverses: on real
  data the blocking is a DEFECT worth −0.040 AUROC.
- **§2 AD / explain / RUL / calibration tables** — all void, produced in the
  degenerate cell (§C.1). Not "suspended pending a re-run" as §B.2 put it for
  two of them: void, all four.
- **§B.7 step 0b** ("is the calibration stage worth its 4 missing runs") — still
  open, and now sharper: §C.3 says conformal was buying the half-bin, and the
  e-value layer does not change that for two-sided sets.

---

## B. 2026-08-05 (evening) — three diagnostic suites, and what they found

### B.1 What was built and why

Everything wrong in this project so far has been wrong in the same way: a
number that looked reasonable, produced by a model whose training loss looked
normal. Six times. The individual bugs were each fixed afterwards; the SHAPE
was only ever written down (§3). So this pass built the checks that shape
implies, and ran them:

| file | what it isolates | cost |
|---|---|---|
| `tests/test_ad_diagnostics.py` (15) | is the METRIC valid, is the GENERATOR's premise true, is the MODEL reading the data, is the STRUCTURE result about structure | 4 s |
| `tests/test_rul_diagnostics.py` (15) | objective validity, H1/H3/H5 measured, box-query exactness, what the miscalibration is made of | 11 s |
| `tests/test_experiment_hygiene.py` (8 + 4 xfail) | the four recurring bug SHAPES, one test each | 2 s |

```bash
PYTHONPATH=. pytest tests/test_ad_diagnostics.py tests/test_rul_diagnostics.py \
                    tests/test_experiment_hygiene.py -q -s      # 38 passed, 4 xfailed, 17 s
```

The four `xfail(strict=True)` are the already-known open items (§A.5 ×2, §A.7,
and `weight_jitter=0`). They flip to XPASS the day each is fixed — that is the
signal, not a green tick.

**These are pre-batch checks, not post-batch ones.** 17 seconds against runs
that take a night.

### B.2 FINDING 1 — "exact ≠ calibrated" is largely a UNIT MISMATCH

`SurvivalPC.predict` returns `q05`/`q95` as bin **centres**. `picp` scores them
against `rul_test` in **cycles**. Every reported PICP for the circuit comes
from that pair (`run_rul.py:82`, `pipeline.py:416`), and the conformal layer
adds a scalar to those same centres (`conformal.py:158`).

Measured at the recorded settings (bins=25, cap=130, K=12, τ deep, chain, 393
synthetic test windows):

| quantity | value | what it says |
|---|---|---|
| PICP, centres (as reported) | **0.616** | severe under-coverage at nominal 0.90 |
| PIT **variance** | **0.0841** vs 1/12 = 0.0833 | the dispersion is calibrated to 3 d.p. — the predictive is **not** overconfident |
| median distance of a miss outside the interval | **2.60 cycles** | = **exactly half a bin** (5.2/2) |
| PICP, same bins read as **edges** | **0.929** | MPIW 68.5 → 73.7, i.e. one bin width |
| true BIN inside the selected bins | **0.929** | the pmf covers its own target correctly |
| PIT mean | **0.408** vs 0.5 | a pure LOCATION shift; the shape term is 10× smaller |

So the density is not miscalibrated in width. What remains is a location
shift — and §B.4 shows that shift IS the censoring bias, so the write-up is
counting one defect twice. It also explains why post-hoc conformal "worked"
with no exactness guarantee: it was fitting back the half-bin the extraction
dropped.

**Status: CONFIRMED on real C-MAPSS, 2026-08-05 21:42** —
`logs/ts/cmapss_rul_endpoints/`, FD001, 4 censoring levels × 3 seeds, all 12
runs ok. Mean over seeds, `all` protocol:

| censor | arm | PICP centres | PICP **edges** | MPIW | PIT var (1/12 = .0833) |
|---|---|---|---|---|---|
| 0.2 | exact censored | 0.404 | **0.975** | 70.5 → 75.5 | 0.0772 |
| 0.2 | drop censored | 0.410 | **0.978** | 73.3 → 78.3 | 0.0742 |
| 0.35 | exact censored | 0.401 | **0.972** | 69.1 → 74.1 | 0.0789 |
| 0.35 | drop censored | 0.410 | **0.979** | 73.4 → 78.4 | 0.0737 |
| 0.5 | exact censored | 0.393 | **0.966** | 66.5 → 71.5 | 0.0810 |
| 0.5 | drop censored | 0.409 | **0.977** | 72.8 → 77.8 | 0.0734 |
| 0.7 | exact censored | 0.353 | **0.930** | 55.9 → 60.9 | 0.0891 |
| 0.7 | drop censored | 0.406 | **0.975** | 71.9 → 76.9 | 0.0752 |

Every row: 0.35–0.41 → 0.93–0.98 for **5 cycles** — one bin. Seed sd on PICP
is ~0.002. **This comparison has no confound available to it**: centres and
edges are two readings of the SAME pmf from the SAME fitted model, so nothing
about machine, commit or seed can explain the gap.

And the PIT variance is **below** 1/12 in seven of eight rows: the predictive
is if anything too DIFFUSE. "Exact but overconfident" is the wrong description
of every model measured.

**Verdict: the recorded PICP 0.38–0.52 is the endpoint convention.** Report
`picp_edge`. "Exact ≠ calibrated" comes out of the paper as a headline and
returns as a two-line note on reading intervals off a discrete predictive.
The residual defect is a LOCATION shift — see §B.9, where it turns out to be
the censoring bias, i.e. one defect the write-up was counting twice.

### B.3 FINDING 2 — the chain's advantage is BLOCKING, not temporal order

Capacity held exactly fixed (same circuit, same parameter count, only the
variable→position map changes), held-out NLL in nats:

| | base | timestep ORDER permuted | ALL features permuted |
|---|---|---|---|
| real (AR(1)) windows | 43.76 | **+0.07** | **+5.17** |
| temporal structure destroyed | 43.63 | −0.08 | **+6.50** |

Scrambling the timestep order costs nothing. Scrambling which channels sit
together costs 5 nats — and still costs 6.5 nats on data with no temporal
structure at all, so it is a property of the layout, not of the data.

**The chain wins because it keeps each timestep's channels contiguous.** "It
is HMM-shaped" is not the explanation, and the ablation table in §2 is ordering
region graphs by variable GROUPING. That also changes what the curvature/SOS/
multi-partition negatives mean — they may be losing on blocking granularity,
not on structure quality. **Status: SUSPENDED pending the same check on real
C-MAPSS** (§B.7 step 1).

Related validity precondition, now executable: `decouple` is vacuous unless
permuting the timesteps destroys enough lag-1 structure. window=4 destroys 21%
and every view scores 0.51–0.55 (chance); window=8 destroys 56%. Nobody
re-checks this pair when `window` or `phi_ar` moves.

### B.4 FINDING 3 — H1/H3/H5, measured at last

The five hypotheses of 2026-08-02 were written with the signature that would
confirm each, and then the expensive end-to-end run was done instead. Cheap
versions, run:

- **H5 (binning/cap damage) — FALSE, cleared.** The learned τ marginal is
  0.018 total-variation from the empirical histogram (top bin 0.587 vs 0.577).
  Stop looking here.
- **H1 (τ drowned by the window) — TRUE but modest.** τ takes 2.0% of the leaf
  gradient mass against a 3.2% dimensional share (1.6× under), ~8× under per
  parameter. Real, but not the order-of-magnitude effect it was assumed to be;
  the ratio grows with `window·C`, so re-measure at 450 features before
  building a re-weighting fix.
- **H3 (root coupling) — TRUE and worse than recorded.** sd of E[τ|x] in
  cycles: root 0.009 / 0.024 / **4.27** / 0.001 at K = 4/6/8/12; deep 19.3 /
  19.0 / 19.8 / 20.3. Root is CONSTANT at three of four K and merely feeble at
  the fourth. The collapse is **not monotone in K**, so any single-K ablation
  of `tau_where` is a coin flip.
- The trivial-maximiser mechanism behind the dead T1 gate now **reproduces in
  3 seconds**: bias −1.07 cycles at 15% censoring, +4.24 at 75%. T1 stays dead,
  for a reason that now fits in one sentence and one test.

### B.5 FINDING 4 — two blind guardrails and a new trap

- **`assert_informative` cannot see partial collapse.** A circuit that ignores
  whole channels has a perfectly variable score. Sabotage two of six channels
  and it passes; the per-channel sensitivity sweep catches it (blinded
  channels ~1 nat vs ~50–200 for the rest).
- **`predict`'s degeneracy threshold is 1e-3·cap = 0.1 cycles** against a
  target whose own sd is 31.5 cycles — **315× too loose**. A predictive with 8%
  of the target's spread is accepted silently. It catches total collapse only;
  it is not a quality check, and nothing downstream is either.
- **NEW TRAP — the compiled evaluator shadows the DAG.** After `fit`,
  `log_prob` routes through `CompiledCircuit`, which holds its own parameter
  tensors. `write_back()` syncs compiled→DAG; there is **no DAG→compiled
  sync**. So any post-fit edit to the DAG — a calibration pass, pruning, leaf
  surgery, a diagnostic — is silently a no-op with correct-looking results.
  Call `pc.pc.use_recursive()` first. Same family as `.to()` being exponential:
  a convenience that quietly does the wrong thing.

### B.6 Corrections to this file's own record

Checked against the tree, not assumed:

1. **§1 and §4 say real data is "PLUMBED AND TESTED, NOT YET RUN".** It has
   been run. `logs/ts/` holds **2092 result rows dated 2026-08-04**:
   `cmapss_ad` 12/12 runs, `cmapss_explain` 9/9, `cmapss_rul` 12/12,
   `cmapss_structure` 27/27, `cmapss_calibration` **10/14** (incomplete).
   `data/cmapss/` is populated; `data/ncmapss/` is not.
2. **Those results are nowhere in this file.** On real FD001 the PC actually
   *leads* detection — 0.8378 vs Mahalanobis 0.8246, conv-AE 0.8115 — on the
   axis §1 says not to claim. Also on real data every method sits at ~0.51 on
   `decouple`: one third of the anomaly taxonomy carries no signal there.
3. **§A.9 says `logs/rul_leaves_relative.json` is "new and valid".** It is not
   in the tree; only `rul_leaves_legacy.json` exists. The §A.2 relative table
   has no artifact behind it.
4. **§A.9 lists three files as uncommitted.** They are committed (`1da0529`).

A stale status board is the same bug shape as §3 — a control that does not
match its treatment. Reconcile before planning off this file.

### B.7 What to do, in order

**Step 0 — settle the calibration finding. DONE 2026-08-05 21:42.** Code in
(§B.8), run complete, read in §B.2 and §B.9. The finding was the endpoint
convention. Two follow-ups it created, both above step 1 in priority:
**(0a)** re-run tonight's commit on `jawa17-desktop` to de-confound the
censoring reversal (§B.9); **(0b)** decide whether the calibration stage is
still worth its 4 missing runs, given that conformal may be buying only the
half-bin.

```bash
# what is running (real C-MAPSS FD001, 4 censoring levels x 3 seeds, ~1 h CPU)
PYTHONPATH=. python -m poc.time_series.runner config/ts/cmapss_rul.yaml \
    --device cpu --log-root logs/ts/cmapss_rul_endpoints
# a NEW log root on purpose: the 2026-08-04 rows stay untouched for comparison
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts/cmapss_rul_endpoints
```

**How to read it.** Three columns now sit next to each other in the `rul`
table, and no one of them is decisive alone:

| picp | picp_edge | pit_var | verdict |
|---|---|---|---|
| low | ≈ 0.90 | ≈ 1/12 | the model was fine; the interval was read wrong. **Delete "exact ≠ calibrated" as a headline**, keep it as a two-line note on discrete predictive intervals, and narrow the conformal stage's purpose to "removes the censoring-induced location shift" |
| low | still low | ≫ 1/12 | the predictive really is overconfident. The finding is real, it is a good result, and it now has a mechanism to state |
| low | ≈ 0.90 | ≈ 1/12, `pit_mean` far from 0.5 | both: the width is right, the location is off — and §B.4 says that shift is the censoring bias, so it is ONE defect, not two |

**Early evidence, from the smoke run of the new code (synthetic, meaningless
magnitudes — the AGREEMENT is the point):**

```
raw exact predictive:  PICP 0.534 centres / 0.944 edges,  MPIW 108.3 / 119.2
conformal[cqr] a=0.10: PICP 0.944                          MPIW 119.2
```

Conformal reproduces the edge interval to three digits, in both coverage and
width. At this scale it is buying **exactly the half-bin and nothing else**.
If that holds on real data, the calibration stage is not measuring
calibration.

**Steps 1–3 — DONE 2026-08-05, evening.** §B.9 collapsed them into each other:
once the 2026-08-04 real-data logs turned out to be pre-σ-floor, everything has
to be re-measured at one commit anyway, so the code fixes had to land BEFORE
that batch rather than after it. All of them are in (§B.8):

| was | now |
|---|---|
| `np.linspace(0.1,0.9,1)` → 10th percentile | median at n=1 |
| `InputNode.fit` unfloored, and the DEFAULT factory | same relative floor as `GaussianLeaf`, `sigma_floor` buffer, `sigma` property |
| `leaf_components` confounds class/init/count | `mixture_at_1=True` opt-in builds `GaussianMixtureLeaf(n=1)`; the default is unchanged on purpose |
| `weight_jitter=0` accepted silently | refused in the constructor; `allow_zero_jitter=True` to build it deliberately |
| `predict` refuses below `1e-3·cap` = 0.1 cycles | refuses below 5% of the training target's own sd (`SurvivalPC.target_sd`) |
| `assert_informative` sees total collapse only | also refuses PARTIAL collapse, via `WindowPC.channel_sensitivity` against the median channel |
| structure ablation has no capacity-fixed control | `chain_perm_blocks` / `chain_perm_features` vtrees — same circuit, **identical parameter count (912 verified)**, only the variable→position map broken; both in `config/ts/cmapss_structure.yaml` |

Still open from step 3: `n_floor` (`bench_rul_leaves.py:125`) → use
`leaves_at_their_own_floor` from `tests/test_experiment_hygiene.py`; and the
compiled/DAG sync, which is currently a documented call to `use_recursive()`
rather than a guarded invariant.

**Step 4 — process.** Add the three suites to `run_workstation.sh` as tier 0.
17 seconds; every one of the six degeneracies would have been caught by a check
of this shape. And before any A/B: name what the flag switches, and test that
the "off" branch reproduces a recorded number.

### B.10 THE NEXT ACTION — one batch, on the workstation, at this commit

> **THIS RAN on 2026-08-06. Read §C.1 for what it found — it is not pending.**
> It answered question 2 (structure) and question 3 (the void numbers) and
> inverted three claims in the process. Question 1 (the censoring de-confound)
> is still open, because the batch also revealed that all four headline configs
> ran in a degenerate cell, so the censoring arm has to be re-measured outside
> it. The command block below is still the right shape; the configs it runs
> need the §C.2 fix first.

Everything above converges on a single run. It is not "re-run RUL": it is
**re-establish every real-data number at one commit**, because §B.9 voided the
2026-08-04 set, and it de-confounds the censoring reversal and answers the
structure question as a side effect.

```bash
# on jawa17-desktop, at THIS commit, after `git pull`
bash poc/time_series/launch/run_smoke.sh                  # ~3 min, proves the wiring
PYTHONPATH=. python -m pytest tests/test_ad_diagnostics.py \
    tests/test_rul_diagnostics.py tests/test_experiment_hygiene.py -q   # 17 s
TIERS="1 2 5" JOBS=3 bash poc/time_series/launch/run_workstation.sh
```

Three questions it settles, none of which can be answered any other way:

1. **the de-confound** — the censoring ablation at this commit on the machine
   that produced the 08-04 numbers. Only the σ-floor then differs. Until this
   runs, no sentence about the censored term's sign on real data is supportable
   and **T1 stays dead**.
2. **the structure question** (§B.3) — `chain` vs `chain_perm_blocks` vs
   `chain_perm_features` at identical parameter count. If blocks ≈ chain and
   features is much worse, the ablation table is about variable grouping and
   the "HMM-shaped" reading comes out of the paper.
3. **the void numbers** — ad, explain, structure and rul all re-measured under
   the σ-floor, the `InputNode` floor and both guardrails, so the whole set is
   mutually comparable for the first time.

Expect the guardrails to REJECT runs that previously produced numbers. That is
the point; a rejected run is a result, and the message names the channels or
the sd that failed.

**Do NOT draft the Paper A section (§4 action 4) until step 0 and step 1 land.**
Three claims are currently in motion: exact≠calibrated (suspended), "the chain
wins on AUROC and likelihood" (suspended), "mixture leaves add capacity"
(blocked by §A.4). What survives untouched: exact attribution 0.902 vs 0.857,
completeness at 1.5e-5 nats, box-query exactness at 6e-6, and T1's death.

### B.9 What else the re-run showed — one clean result, one CONFOUND

**Clean, because it is within one run on one tree: the censored term shifts the
predictive UP, monotonically with censoring.** PIT mean, `all` protocol
(>0.5 = under-predicting remaining life, <0.5 = over-predicting):

| censor | 0.2 | 0.35 | 0.5 | 0.7 |
|---|---|---|---|---|
| **exact censored** | 0.556 | 0.523 | **0.494** | **0.430** |
| drop censored | 0.595 | 0.586 | 0.588 | 0.576 |

The drop-censored arm under-predicts by a constant amount at every level —
the textbook bias of training only on units you saw fail. The exact censored
term removes that bias monotonically, passes through perfect calibration
around 50% censoring, and **overshoots into over-prediction at 70%**, where it
also over-sharpens (MPIW 55.9 vs 71.9, PIT var 0.0891 — the only row above
1/12).

That is the trivial maximiser of `log P(τ ≥ c | x)`, visible as a dose-response
curve rather than as one failed CRPS comparison. **It is a better statement of
why T1 died than the one on record**, and it costs nothing to make: the term
does exactly what the theory says, in both directions, and 70% censoring is
where the correction runs out of uncensored anchors.

**CONFOUNDED — do not act on this.** Tonight's run and the 2026-08-04 run use
the same config and the same data, and they disagree about the censoring
ablation:

| censor 0.7, `all` | CRPS exact | CRPS drop | who wins |
|---|---|---|---|
| 2026-08-04 (`logs/ts/cmapss_rul`) | 9.07 | 8.44 | drop — consistent with the dead gate |
| tonight (`…_endpoints`) | 11.34 | 11.82 | **exact** — the reverse |

Absolute levels moved too (RMSE ~19–21 → ~25–27). **Two things changed at
once**, which is the exact failure this whole session is about:

1. **commit** — 08-04 ran at `8816d97`, which is the PRE-σ-floor-fix code named
   in §A.1. Tonight ran at `1da0529` + the step-0 edits. The relative leaf
   floor forces wider leaves on every MAD-zero feature, which is a real model
   change, not a numerical one.
2. **machine** — 08-04: `jawa17-desktop`, RTX 4080, 8 threads. Tonight: this
   Mac, CPU, 4 threads.

So the reversal is not evidence of anything yet. **T1 stays dead.** To settle
it, run tonight's commit on `jawa17-desktop` — that holds the machine fixed
against 08-04 and leaves the σ-floor as the only difference. Until then no
sentence about the censored term's sign on real data is supportable.

Note this cuts both ways: **the 2026-08-04 real-data numbers were produced by
the pre-fix leaf code** and are not comparable to anything measured after
2026-08-05 either. That applies to `cmapss_ad`, `cmapss_explain` and
`cmapss_structure` as much as to RUL (§B.6).

### B.8 State of the tree

**Uncommitted, and this is now a large diff — commit before the workstation
batch (§B.10), which needs to run at a known commit.**

Tests: **331 passed, 0 failed, 0 skipped** (`pytest tests/ -q`, 23.8 min on this
Mac — `test_inference` and `test_vtree` dominate). Smoke run clean end to end on
all four stages. **All four strict xfails are now cleared** — they were the
signal that the open items were open, and they flipped as each was fixed.

| file | change |
|---|---|
| `tests/test_ad_diagnostics.py` | **new**, 15 tests |
| `tests/test_rul_diagnostics.py` | **new**, 16 tests (incl. the step-0 pin) |
| `tests/test_experiment_hygiene.py` | **new**, 12 tests (was 8 + 4 xfail) |
| `poc/time_series/circuits.py` | `bin_edges()`; `q05_edge`/`q95_edge` in `predict` beside the unchanged `q05`/`q95`; `window_leaf()` + `mixture_at_1` on both models; `channel_sensitivity()`; `assert_informative` refuses partial collapse; `predict` threshold relative to `target_sd`; the two layout-control vtrees |
| `src/probabilistic_circuits.py` | `InputNode` floored like `GaussianLeaf` (buffer + `sigma` property); `linspace` → median at n=1; `weight_jitter=0` refused (`allow_zero_jitter` escape); `permute_region_graph` + `timestep_block_permutation` |
| `poc/time_series/metrics.py` | `pit_values` / `pit_report`; the PICP caveat in the module docstring |
| `poc/time_series/pipeline.py` | `_eval_survival` returns `(metrics, pred)` and adds `picp_edge`/`mpiw_edge`/`interval_score_edge`/`pit_*`; `rul_pred_*.npz` artifact per fit per protocol; the same edge columns in `_partial_evidence`; a three-arm log line in the calibration stage |
| `poc/time_series/run_rul.py` | the same columns, so the old driver stays comparable |
| `poc/time_series/aggregate.py` | the new columns in `PREFERRED_COLUMNS` so they reach `summary.md` |
| `config/ts/cmapss_structure.yaml` | the two capacity-fixed layout controls |
| `tests/test_region_graph.py` | the `weight_jitter=0` collapse test now goes through the escape hatch, plus a new test that the default refuses it |

**The npz artifact is the part that outlives this question.** Every RUL fit now
persists `pmf`, both interval pairs, `rul_true`, `tau_true` and `bin_edges`, so
the next "what would this have been under a different convention?" is a
two-minute re-analysis rather than a re-run. The reason step 0 needed a re-run
at all is that the stage previously saved scalars only.

**The npz artifact is the part that outlives this question.** Every RUL fit now
persists `pmf`, both interval pairs, `rul_true`, `tau_true` and `bin_edges`, so
the next "what would this have been under a different convention?" is a
two-minute re-analysis rather than a re-run. The reason step 0 needed a re-run
at all is that the stage previously saved scalars only.

**Running now:** `logs/ts/cmapss_rul_endpoints/` (started 19:27, ~1 h, CPU).
The old `logs/ts/cmapss_rul/` rows from 2026-08-04 are untouched.

---

## A. Earlier on 2026-08-05 — the leaf σ-floor episode, and the one experiment now open

### A.1 What was wrong

`bench_rul_leaves --floor legacy` was **not** the pre-change behaviour. The
`use_relative_floor` flag gated the relative bound at **both** initialisation
and training, but the pre-change code (commit `8816d97`) only ever had it at
*init*:

| | init | runtime |
|---|---|---|
| pre-change (`8816d97`) | `σ = max(MAD·1.4826, 0.01·std, 1e-3)` | `+1e-5` epsilon |
| broken `legacy` branch | `σ = max(MAD·1.4826, 1e-5)` | `+1e-5` epsilon |
| **fixed `legacy`** | `σ = max(MAD·1.4826, 0.01·std, 1e-3)` | `+1e-5` epsilon |
| `relative` | same as fixed legacy | `max(0.01·std, 1e-3)` |

So `legacy` was a **third regime — no floor anywhere** — and the first
`logs/rul_leaves_legacy.json` was not comparable to the capacity table it was
meant to be read against.

**FIXED** in `src/probabilistic_circuits.py`: the init rule now applies in both
modes and the flag switches the **runtime** floor only.
`tests/test_leaf_sigma_floor.py` (11 tests) pins this so the two floors cannot
be conflated again — it fails against the pre-fix code.

**Scope of the contamination, measured, not assumed:** only the **1-component**
row was affected. The mixture init edit is a provable no-op on all four C-MAPSS
subsets (worst-case std 1.54e-2 on FD002 ⇒ `spread = std/n ≥ 1.5e-3` clears
both floors even at n=10). Rows 2–10 of the old table were valid pre-change
behaviour all along; the table was unusable because its *baseline* was wrong.

**Validated:** fixed `legacy`, FD001, 1 comp, 3 seeds, 60 ep → RMSE-last
**25.46** (CUDA) / **25.53** (CPU) vs the capacity table's **25.45**. The two
benches agree; the leaf question is well-posed.

### A.2 The two tables (FD001, K=12, bins=25, tau_where=deep, 60 ep, 3 seeds)

Ridge reference **15.96**.

**`--floor legacy`** (collapse permitted — the contaminated control):

| comps | RMSE last | per seed | CRPS | NLL test | σ min | @floor |
|---|---|---|---|---|---|---|
| 1 | 25.46 | 25.2 25.5 25.7 | 12.91 | 428.8 | 1.05e-03 | 0 |
| 2 | 23.19 | 23.5 20.6 25.5 | 11.42 | 52.6 | 1.00e-05 | 85 |
| 3 | 22.48 | 22.5 22.9 22.1 | 11.00 | 39.1 | 1.00e-05 | 105 |
| 5 | 21.81 | 21.0 22.4 22.0 | 10.71 | 82.4 | 1.00e-05 | 76 |
| 10 | 25.29 | 25.3 25.0 25.5 | 12.69 | 252.8 | 1.00e-05 | 45 |

**`--floor relative`** (the floor holding — the valid arm):

| comps | RMSE last | per seed | CRPS | NLL test | σ min | @floor |
|---|---|---|---|---|---|---|
| 1 | 25.65 | 25.7 25.6 25.6 | 12.70 | 363.6 | 2.15e-03 | 0 |
| **2** | **20.22** | 19.1 20.5 21.1 | 10.38 | 178.7 | 2.11e-03 | 0 |
| 3 | 20.61 | 19.4 21.4 21.0 | 10.33 | 162.3 | 2.11e-03 | 0 |
| 5 | 21.96 | 22.1 21.2 22.6 | 10.80 | **148.2** | 2.11e-03 | 0 |
| 10 | 24.21 | 25.4 21.7 25.5 | 12.17 | 286.3 | 2.11e-03 | 0 |

### A.3 What these say

1. **The floor makes the model better, not worse.** relative − legacy =
   +0.19, **−2.97**, **−1.87**, +0.15, −1.08. Collapse was *hurting* RMSE at
   2–3 components. The prior intuition ("collapse flatters RMSE", from the
   1-comp 24.24-vs-25.46 pair) does **not** generalise to the mixtures.
2. **The legacy density gain was an artefact, confirmed.** Legacy NLL 39.1 at
   3 comps vs relative 162.3 — 4× worse once spiking is forbidden. Never quote
   the legacy NLL column as model quality.
3. **Density capacity is real; RUL accuracy is not.** In the *valid* arm NLL
   falls monotonically 363.6 → 178.7 → 162.3 → **148.2** (best at 5 comps)
   while RMSE-last is best at **2** and degrades after. Same lesson as
   `forman_rg` in §2: *density fit ≠ the downstream task.* More components
   genuinely buy density and genuinely cost prognosis.
4. **Seed spread grows with components** (max−min: 0.1, 2.0, 2.0, 1.4, 3.8) —
   the 10-comp row is unstable (25.4 / 21.7 / 25.5).
5. **The circuit still loses to ridge on real FD001**, badly: best config 20.22
   vs ridge 15.96. The synthetic-data result "PC beats ridge on RMSE" (§2) does
   **not** reproduce on real C-MAPSS. This is a credibility finding, not a
   tuning problem — do not quote the synthetic RUL-vs-ridge comparison again
   without this caveat next to it.

### A.4 THE CONFOUND — why "1 vs 2 components" is not yet a clean contrast

`WindowPC._leaf_factory` (`poc/time_series/circuits.py:314`) and
`mixed_leaf_factory` (same file, :213) both do:

```python
return GaussianMixtureLeaf(i, n_components=c) if c > 1 else GaussianLeaf(i)
```

So `c=1` changes **three things at once** vs `c=2`:

| | c = 1 | c ≥ 2 |
|---|---|---|
| class | `GaussianLeaf` | `GaussianMixtureLeaf` |
| σ init | `max(MAD·1.4826, 0.01·std, 1e-3)` | `std / n` |
| mixture logits | none | learnable |

The σ inits are *different rules*, not the same rule at different n — MAD-based
robust scale vs `std/n`. On FD001 the mixture init never touches its floor
(min std 0.211 ⇒ `spread ≥ 0.021`), so the mixture always starts n× sharper.

**Therefore the 25.65 → 20.22 jump cannot currently be attributed to "having 2
components".** It is confounded with "being a `GaussianMixtureLeaf` initialised
at `std/n`".

### A.5 The decisive experiment (NOT run — the enabling edits were declined)

Add a **1-component `GaussianMixtureLeaf`** arm. It is mathematically a single
Gaussian, so it isolates the class/init from the component count:

- if `GMLeaf(n=1)` ≈ **20.2** → the win is the **initialisation rule**, and the
  fix is to seed `GaussianLeaf` at `std` rather than MAD (cheap, and it means
  mixture leaves are not needed at all);
- if `GMLeaf(n=1)` ≈ **25.6** → the win is **real capacity** from the second
  component, and mixture leaves earn their place.

~70 s per row on the workstation GPU. **Three edits are required first and none
of them is in the tree** (the first was declined mid-session, so I stopped
before the other two):

1. **`GaussianMixtureLeaf.fit` — latent bug, must fix before running n=1.**
   `np.linspace(0.1, 0.9, 1)` is `[0.1]`, **not** `[0.5]`, so a 1-component
   mixture centres its only Gaussian on the **10th percentile**. Harmless for
   n ≥ 2; rigs the n=1 comparison against the mixture. Use the median when
   `n_components == 1`.
2. **`_leaf_factory` / `mixed_leaf_factory`** — an opt-in (e.g. `--mixture-at-1`)
   so `c=1` can build `GaussianMixtureLeaf(n_components=1)`. Keep it opt-in;
   changing the default would move every recorded 1-comp number.
3. **`bench_rul_leaves.run_one`'s `n_floor`** — see A.6.

### A.6 The collapse diagnostic is blind in the arm that matters

`n_floor = int((sig < 1e-3).sum())` counts σ below an **absolute** 1e-3. In
`relative` mode every leaf's floor is `max(0.01·std, 1e-3) ≥ 1e-3`, so
**`@floor` is structurally 0 there and can never detect collapse.** The `0`s in
the relative table are not evidence of no collapse.

The real evidence points the other way: σ min = **2.11e-03** = exactly
`0.01 × 0.2112` = the relative floor of FD001's tightest feature. Leaves *are*
pressed against the floor; it is holding them, not making them unnecessary.

Fix: count leaves at **their own** floor (`σ ≤ 1.01 · leaf.sigma_floor`), which
is mode-independent. `leaf_sigmas()` must then also return the `sigma_floor`
buffers.

### A.7 Still-open bug, deliberately not fixed

`InputNode.fit` (`src/probabilistic_circuits.py`, the heavy-tailed
Gaussian/Laplace/Student-t leaf) still does `mad = median(|v−μ|) + 1e-6` with
**no relative floor** — the exact `σ ≈ 1.5e-6` → NaN path the `GaussianLeaf`
docstring describes as fixed. It is the **default `leaf_factory`** for the
generic builders (`RegionGraphPC` / `DensityPC` when no factory is passed), so
anything outside the RUL path is exposed. Left alone because changing a default
mid-campaign shifts recorded numbers. **Fix before any non-RUL run.**

### A.8 Corrected fact worth keeping

The `GaussianLeaf.fit` docstring used to claim the floor "binds on exactly the
20 broken features of FD002/FD004 and on ZERO features of FD001/FD003". That
was measured at window=20; under the RUL bench's task (window=30, bins=25) it
is false. Measured:

| subset | features | MAD == 0 | floor raises σ |
|---|---|---|---|
| FD001 | 450 | 30 (all of channel 3) | 30 |
| FD003 | 480 | 30 (all of channel 7) | 30 |
| FD002 | 630 | 60 | 90 (channels 15, 17, 18) |
| FD004 | 630 | 60 | 90 (channels 15, 17, 18) |

One whole median-constant sensor channel per subset. Note MAD = 0 with
std = 0.21: these are plateau channels with real excursions, not dead ones.
Now corrected in-code.

### A.9 State of the tree

**Uncommitted**, nothing pushed:

| file | change |
|---|---|
| `src/probabilistic_circuits.py` | init/runtime floor split in `GaussianLeaf.fit` + `GaussianMixtureLeaf.fit`; docstring corrections (A.8) |
| `poc/time_series/bench_rul_leaves.py` | `--floor` help text only |
| `tests/test_leaf_sigma_floor.py` | **new**, 11 tests |

Tests: 96 passed (`test_inference`, `test_compiled_circuit`, `test_vtree`,
`test_leaf_sigma_floor`), exit 0. Full suite not re-run since.

`logs/rul_leaves_legacy.json` has been regenerated with the fix and is valid.
`logs/rul_leaves_relative.json` is new and valid.

### A.10 Run these next

```bash
export PYTHONPATH=.
# after the three edits in A.5:
python -m poc.time_series.bench_rul_leaves --device cuda --floor relative \
    --components 1 2 --mixture-at-1 --out logs/rul_leaves_mix1.json
```

Read it as A.5 says. Until that has run, **do not** write "mixture leaves add
capacity" anywhere — the current evidence supports only "something about the
mixture leaf helps at 2 components, and it is not more components beyond 2".

---

## 0. What changed on 2026-08-03 (evening), and what to do with it

The three top-priority actions from the morning hand-off are now **built and
tested**; none of them has been *run at scale* yet, which is the next session's
job.

| morning action | state |
|---|---|
| 1. degeneracy guardrail (§8 below) | **DONE** — `DegenerateModelError` raised by `SurvivalPC.predict` and `WindowPC.assert_informative`; `tau_where` now defaults to `deep` everywhere; tests pin both |
| 2. conformalise the circuit's own predictive | **BUILT** — `poc/time_series/conformal.py`, split by ENGINE, two modes (CQR-style and PIT recalibration), wired as the `calibration` stage |
| 3. real data | **BUILT** — real C-MAPSS (all four subsets, official test units + RUL file) and N-C-MAPSS (HDF5, per-cycle or raw) behind the same task interface; anomaly/RUL protocol unchanged |

Plus the infrastructure that was missing: one config-driven runner for all five
stages, per-run logging (config, git commit, GPU, curves, status, artifacts),
resume-on-restart, cross-run aggregation, and workstation launchers.

**Run this first, next session:**

```bash
bash poc/time_series/launch/run_smoke.sh                     # ~3 min, proves the wiring
# put the NASA files in data/cmapss/ and data/ncmapss/ (data/README.md), then:
TIERS="1 2" JOBS=3 bash poc/time_series/launch/run_workstation.sh
```

Tier 1 is detection + explanation on real C-MAPSS — the credibility gap and the
contribution. Tier 2 is calibration + RUL. Everything is resumable, so an
interrupted night costs nothing.

**One bug worth knowing about**, found while building this: calling
`nn.Module.to(device)` on a region-graph circuit is *exponential in depth* —
`to()` recurses over `children()` with no memoisation, so every shared
sub-circuit is visited once per path. That is the same `K^depth` blowup the DAG
rebuild removed, reintroduced through a PyTorch convenience method (a 6×8
window went from 0.2 s to 400 s per fit, silently and with correct results).
Use `move_circuit_(circuit, device)` from `src/probabilistic_circuits.py`;
never `.to()`. A test pins it.

---

## 1. Status board

| workstream | state |
|---|---|
| DAG / region-graph rebuild of the circuit layer | **DONE**, 216 tests pass |
| AD detection | **DONE** — parity with the best baselines, not superiority |
| Explainability (the contribution) | **DONE** — holds under every robustness check run so far |
| T1 "RUL as an exact censored survival query" | **DEAD** — failed the pre-registered gate on a valid model |
| RUL as a *model* (not as a novelty claim) | **works** — beats ridge/MLP/CQR on RMSE, but badly uncalibrated |
| Degeneracy guardrail | **DONE** (2026-08-03 evening) |
| Conformal layer on the exact predictive | **BUILT, not yet run at scale** |
| Experiment pipeline / configs / launchers / logging | **DONE** (2026-08-03 evening), 29 new tests |
| Real data (C-MAPSS) | **RUN** — 2092 rows dated 2026-08-04 in `logs/ts/` (ad 12/12, explain 9/9, rul 12/12, structure 27/27, calibration 10/14). N-C-MAPSS still not run: `data/ncmapss/` is empty. **The results are not yet folded into this file** — see §B.6 |
| Leaf σ floor (init vs runtime) | **FIXED + TESTED** 2026-08-05 (§A); the `legacy` A/B is now valid |
| "do mixture leaves help?" | **OPEN and CONFOUNDED** — §A.4/A.5; needs the 1-component-mixture arm before any claim |
| Real-data RUL accuracy | **NEGATIVE** — circuit 20.22 vs ridge 15.96 on real FD001 (§A.3); the synthetic "PC beats ridge" does not reproduce |
| "exact ≠ calibrated" | **RESOLVED — it was the endpoint convention** (§B.2). Real C-MAPSS, 12/12 runs: PICP 0.35–0.41 on bin centres, **0.93–0.98 on bin edges**, for one bin of extra width; PIT variance below 1/12 in 7 of 8 rows. Report `picp_edge`; demote the finding to a note |
| Comparability of the 2026-08-04 real-data logs | **BROKEN** — they ran at `8816d97`, the PRE-σ-floor-fix commit (§B.9). Not comparable to anything measured after 2026-08-05 |
| "the chain wins on structure" | **SUSPENDED** — §B.3, the advantage is timestep BLOCKING (+5.17 nats) not temporal order (+0.07); re-check on real data |
| Diagnostic suites (AD / RUL / hygiene) | **DONE** 2026-08-05 evening — 38 tests + 4 strict xfail, 17 s, §B.1 |

**One-line thesis that the evidence supports:** *parity on detection,
exclusivity on explanation.* The circuit ties the best detectors and is the
only one that can say why — correctly, completely, and in a form an operator
can act on.

**Do not claim:** that the circuit detects better (it does not), or that exact
censoring handling improves prognosis (it does not).

---

## 2. What the overnight batch settled

### AD detection — three-way tie (3 seeds, current generator)

| detector | AUROC | AP |
|---|---|---|
| conv autoencoder | 0.9446 ± 0.0131 | 0.9198 |
| **RegionGraphPC (chain)** | 0.9368 ± 0.0092 | 0.9033 |
| Mahalanobis | 0.9364 ± 0.0097 | 0.9049 |
| 1-NN distance | 0.9311 ± 0.0108 | 0.8886 |
| z-score | 0.8514 ± 0.0125 | 0.7274 |

### Structure ablation — the chain wins on AUROC *and* likelihood

| vtree / region graph | AUROC | train NLL |
|---|---|---|
| **chain (HMM-shaped)** | **0.9368** | **37.62** |
| time (balanced temporal) | 0.9176 | 63.84 |
| orc_rg (n-ary curvature) | 0.8933 | 39.42 |
| spectral | 0.8895 | 44.21 |
| forman_rg | 0.8834 | **32.84** |
| random | 0.8825 | 72.64 |
| chow_liu | 0.8752 | 43.98 |
| channel | 0.8704 | 44.97 |
| SOS / squared (K=2) | 0.9067 | — |
| orc_rg_multi (multi-partition) | 0.9118 | — |

Note `forman_rg`: **best likelihood, near-worst AUROC**. Density fit and
detection are different objectives — worth a sentence in the paper, and a
warning against selecting structure on NLL.

Curvature region graphs lose to the hand-built chain. SOS and multi-partition
both lose too. All three are reportable negatives; none needs re-running.

### Explanation quality vs ground truth (3 seeds) — the contribution

| attribution method | localisation AUROC | prec@k |
|---|---|---|
| **PC conditional (exact)** | **0.9021 ± 0.0165** | 0.7510 |
| PC Shapley (exact conditionals) | 0.8997 ± 0.0246 | **0.7511** |
| AE reconstruction | 0.8570 ± 0.0324 | 0.6710 |
| PC marginal (exact) | 0.8350 ± 0.0247 | 0.5798 |
| Gaussian conditional (exact) | 0.7750 ± 0.0251 | 0.6546 |
| PC structural (exact) | 0.7747 ± 0.0434 | 0.6197 |
| z-score | 0.7451 ± 0.0167 | 0.5088 |
| AE sampling-SHAP (32/ch) | 0.4975 ± 0.0289 | 0.1377 |

- Completeness residual **1.78e-5 nats** (float32 round-off) — completeness is
  a theorem here, not an estimation target.
- **The robustness check that mattered:** raising sampling-SHAP from 32 to 128
  samples/channel moved it 0.4975 → **0.5152**. Still chance. Quadrupling the
  budget buys nothing, which kills the "you under-resourced the baseline"
  objection — the strongest available attack on this result.
- Per-kind: on `desync` PC conditional/structural reach 0.851 while PC
  *marginal* collapses to 0.545 — a 0.31 gap between two views of the SAME
  circuit. On `spike`, z-score gets 0.999: never claim credit for univariate
  anomalies. On `decouple` the Gaussian conditional wins (0.757) — an honest
  negative.

### RUL — the gate, run twice

The first gate was **void**: every run used `--tau-where root`, which is
degenerate (see §3). Re-run with `--tau-where deep --K 12`, 3 seeds:

| censoring | CRPS drop | CRPS censored | Δ | RMSE cens | PICP |
|---|---|---|---|---|---|
| 20% | 11.016 | 10.937 | +0.079 | 23.116 | 0.519 |
| 35% | 12.245 | 12.160 | +0.085 | 24.713 | 0.497 |
| 50% | 10.508 | 10.981 | −0.472 | 22.701 | 0.490 |
| 70% | 11.953 | 14.819 | **−2.866** | 28.185 | 0.384 |

**GATE FAILED, validly this time.** The trend is the reverse of the hypothesis:
the censored term should be most valuable at 70% censoring and does the most
damage there, over-predicting remaining life (RMSE 28.19 vs 24.62).

*Mechanism:* `log P(τ ≥ c | x)` has a trivial maximiser — push all mass above
every censoring time. Only uncensored units anchor against it, and at 70% there
are too few. A real property of the objective with a free categorical τ, not a
coding bug. Per the pre-registration, **T1 is retired to a limitations
paragraph.**

*But the model improved anyway* (from the τ-placement fix, not from censoring):

| | PC (τ deep) | ridge | CQR |
|---|---|---|---|
| RMSE @20% | **23.12** | 24.35 | 27.72 |
| RMSE @50% | **22.61** | 24.00 | 32.76 |
| CRPS | **10.5–12.2** | n/a | n/a |
| PICP (nominal 0.90) | 0.38–0.52 | — | **0.82–0.91** |

### The most interesting scientific finding

**Exact ≠ calibrated.** The density is exactly normalised and its 90% intervals
cover 38–52%. Post-hoc conformal, with no exactness guarantee whatsoever,
reaches 82–91%. This directly complicates the project's "exact therefore
trustworthy" framing and deserves its own paragraph.

---

## 3. The bug class that has now cost five wrong answers — FIX THIS FIRST

Three silent degeneracies have each produced a confident, wrong, *published-to-
me* result. All three were invisible in the training loss and surfaced only in
a query:

1. **Leaf jitter didn't cover all leaf types** — `CategoricalLeaf` /
   `GaussianMixtureLeaf` siblings started identical and stayed identical.
2. **Sum-node weights initialised uniformly** — in the DAG the K units of a
   region are sums over the *same* shared product list, so they are identical
   functions receiving identical gradients. Fixed via `weight_jitter`.
3. **τ attached at the root** — `predict()` returned a *constant* 102.4 cycles
   for all 851 test windows (sd 0.0). The entire first RUL gate was run on this.

**DONE, 2026-08-03 evening.** `SurvivalPC.predict` raises `DegenerateModelError`
when `E[τ|x]` has sd below `1e-3·cap`, and `WindowPC.assert_informative` does
the same for a constant density (it is called on every model the pipeline
fits). `tau_where` now defaults to `deep` in `SurvivalPC`, `run_rul.py` and the
config schema. Two tests pin the behaviour, including the escape hatch
(`predict(..., check_degenerate=False)`), which you have to ask for explicitly.

A fourth member of this bug family turned up the same day and is worth adding
to the list: **`nn.Module.to()` on a region-graph circuit is exponential in
depth** (no memoisation over `children()`), silently turning a 0.2 s fit into
400 s with correct results. Use `move_circuit_`.

**Fifth (2026-08-05), and the first one that corrupted an *experiment* rather
than a model: the leaf σ INIT floor and RUNTIME floor were gated by one flag**,
so the `--floor legacy` control arm was a regime that had never existed (§A.1).
The run looked completely normal — it produced a full table with plausible
numbers. Fixed and pinned by `tests/test_leaf_sigma_floor.py`.

The generalisable lesson, and the reason this one is worth a paragraph: **an
A/B flag must switch exactly one thing, and a test should assert that the
"off" branch reproduces a recorded number.** Both control arms here (`legacy`,
and `c=1` in the leaf sweep — §A.4) turned out to differ from their treatment
in more than one respect. Check the control, not just the treatment.

A sixth, still open: the collapse *diagnostic* itself is blind in the arm that
matters (§A.6) — `@floor` counts an absolute 1e-3 that a relative floor makes
unreachable, so it reports 0 by construction.

**A seventh, found 2026-08-05 evening and the same convenience-does-the-wrong-
thing shape as `.to()`: the compiled evaluator SHADOWS the DAG.** After `fit`,
`RegionGraphPC.log_prob` routes through `CompiledCircuit`, which holds its own
parameter tensors. `write_back()` syncs compiled→DAG; nothing syncs
DAG→compiled. So any post-fit edit to the DAG is silently a no-op and the
scores keep looking correct. Call `pc.pc.use_recursive()` first.
`tests/test_experiment_hygiene.py::test_dag_edits_do_not_reach_the_compiled_copy`
pins it.

**And the two guardrails do not cover what their names suggest** (§B.5):
`assert_informative` passes a circuit that ignores whole channels, and
`predict`'s degeneracy threshold is 1e-3·cap = 0.1 cycles against a target sd
of 31.5. Both are total-collapse detectors, not quality checks.

The tests that encode all seven shapes now exist and run in 17 s — see §B.1.
Run them before a batch, not after.

---

## 4. Next actions, in priority order

> **SUPERSEDED by §B.7.** This list is from earlier on 2026-08-05 and is kept
> because its reasoning still holds; what changed is the ordering (the
> calibration re-run now comes first) and action 3, which has already happened
> (§B.6). Read §B.7 first, then this for context.

0. **The 1-component-mixture arm** (§A.5) — three small edits, then one ~5 min
   run. It is first only because it is cheap and because an open confound is
   currently blocking any statement about leaf capacity. Do not let it displace
   action 3.
1. ~~**Degeneracy guardrail**~~ — **DONE** (§3).
2. ~~**Conformalise the circuit's own predictive**~~ — **BUILT**
   (`poc/time_series/conformal.py`, `calibration` stage,
   `config/ts/cmapss_calibration.yaml`). Split by ENGINE, not by window:
   overlapping windows of one engine are near-duplicates and calibrating on
   them would report a coverage that evaporates on a new unit. Two modes:
   CQR-style additive (finite-sample guarantee) and PIT recalibration (sharper,
   no guarantee) — the gap between them says whether the miscalibration is a
   location or a shape error. **Still to do: run it at scale and read the
   result.** *Do not* present it as evidence for T1; T1 is dead independently.
3. **Real data — RUN IT.** The plumbing is done and tested (real C-MAPSS all
   four subsets with the official test units and RUL file; N-C-MAPSS with
   per-cycle aggregation; censoring simulated on real trajectories; the anomaly
   protocol byte-identical to the synthetic one). What has *not* happened is
   the run: the NASA files are not in the repo. Put them in `data/cmapss/` and
   `data/ncmapss/` (`python -m poc.time_series.check_data`), then run tiers 1–3.
   Until that has happened, every number in §2 still comes from a generator we
   wrote, and the referee's objection stands.
   - Note on scope: turbofan data has no anomaly labels, so injected anomalies
     remain ours even on real data. That is a deliberate trade — injection is
     the only source of the per-channel ground truth the *localisation* claim is
     scored against. ESA-ADB (real annotations) is still the right next source
     after this, and is not yet plumbed.
4. **Write-up** as a Paper A section ("AD as a tractable query"): parity on
   detection, exclusivity on explanation, plus the exact≠calibrated result.

**Explicitly do NOT:**
- re-run the structure / curvature / SOS sweep **on synthetic data** — it has
  answered. (`config/ts/cmapss_structure.yaml` re-asks it on *real* data, where
  it is genuinely open because the hand-built chain was arguably told the
  answer by our own generator. It is tier 5: informative, not decisive.)
- extend RUL beyond action 2 — the gate settled it;
- chase the 0.008 AUROC gap to conv-AE — seed noise, and the wrong claim anyway.

---

## 5. Reproduce

```bash
# the pipeline (preferred — resumable, logged, aggregated)
bash poc/time_series/launch/run_smoke.sh                            # ~3 min
bash poc/time_series/launch/run_config.sh config/ts/cmapss_ad.yaml  # one experiment
TIERS="1 2" JOBS=3 bash poc/time_series/launch/run_workstation.sh   # real-data core
PYTHONPATH=. python -m poc.time_series.aggregate logs/ts --recursive

# the old single-purpose drivers still work unchanged
export PYTHONPATH=.
PY=~/miniconda3/envs/expllm_env/bin/python
bash poc/time_series/run_all_overnight.sh          # ≈5.5 h, gate-first ordering
$PY -m poc.time_series.summarize_overnight logs/overnight
$PY -m poc.time_series.run_rul --seeds 0 1 2 --vtree chain --tau-where deep \
    --K 12 --censor-frac 0.7 --epochs 60 --no-partial
$PY -m poc.time_series.run_explain --seeds 0 1 2 --shapley 8 --plots --examples
```

Results from the August batch: `logs/overnight/` (+ `logs/overnight2/` for the
corrected gate), figures in `logs/overnight/figs/`. Pipeline runs land in
`logs/ts/<experiment>/` with `summary.md` / `summary.csv` per experiment.

---

## 6. File map + gotchas

| what | where |
|---|---|
| region graphs, chain/HMM structure, delta transform, box queries | `src/probabilistic_circuits.py` |
| `RegionGraphPC` (DAG), `SquaredPC(region_graph=)`, `move_circuit_` | same |
| `WindowPC` / `SurvivalPC`, `attach_variable`, degeneracy guardrails | `poc/time_series/circuits.py` |
| attribution, metrics, plots, worked examples | `poc/time_series/explain.py` |
| synthetic fleet, windowing, ground-truth channels | `poc/time_series/data.py` |
| **real C-MAPSS / N-C-MAPSS loaders, censoring, health proxy** | `poc/time_series/data_real.py` |
| **dataset registry + task builders (all three sources)** | `poc/time_series/datasets.py` |
| **the five experiment stages** | `poc/time_series/pipeline.py` |
| **config schema, grid/variant expansion, overrides** | `poc/time_series/config.py` |
| **runner (resume, isolation, plan)** | `poc/time_series/runner.py` |
| **per-run logging, env/git capture, resume keys** | `poc/time_series/ts_logging.py` |
| **cross-run aggregation → csv/md** | `poc/time_series/aggregate.py` |
| **split conformal on the exact predictive** | `poc/time_series/conformal.py` |
| **configs, one per question** | `config/ts/*.yaml` |
| **launchers, tiers, env knobs** | `poc/time_series/launch/` |
| data acquisition, real-vs-injected table | `data/README.md` |
| old single-purpose drivers | `run_ad.py`, `run_rul.py`, `run_explain.py`, `bench_scaling.py` |
| old batch + one-page summary | `run_all_overnight.sh`, `summarize_overnight.py` |
| **AD diagnostics (metric / generator / model / structure)** | `tests/test_ad_diagnostics.py` |
| **RUL diagnostics (objective / H1-H5 / exactness / calibration)** | `tests/test_rul_diagnostics.py` |
| **the four recurring bug shapes, one test each** | `tests/test_experiment_hygiene.py` |

- τ must get a `CategoricalLeaf` (closed-form interval). `InputNode` has no
  closed-form CDF and raises if boxed.
- **`InputNode.fit` has no σ floor** and is the *default* `leaf_factory` — see
  §A.7. Fix before any non-RUL run.
- **Leaf σ has two floors, init and runtime.** `use_relative_floor` switches the
  runtime one only; the init rule is the same in both modes (§A.1). Do not
  re-couple them.
- **`c=1` in the leaf sweep is a different leaf *class*, not one component**
  (§A.4) — `GaussianLeaf` (MAD init) vs `GaussianMixtureLeaf` (`std/n` init).
- `np.linspace(0.1, 0.9, 1) == [0.1]`, so a 1-component `GaussianMixtureLeaf`
  centres on the 10th percentile, not the median (§A.5).
- `@floor` in `bench_rul_leaves` is **structurally 0** under `--floor relative`
  (§A.6). Use σ min against the leaf's own floor instead.
- `--tau-where root` is **degenerate**; `deep` is now the default everywhere.
  `root` is kept only as an ablation, and `predict` will now refuse it loudly
  when it collapses.
- **Never call `.to(device)` on a circuit** — use `move_circuit_`. See §0.
- **Never edit the DAG after `fit` without `pc.pc.use_recursive()` first** —
  the compiled evaluator holds its own parameters and your edit is a silent
  no-op (§B.5).
- **`q05`/`q95` from `SurvivalPC.predict` are bin CENTRES**, and `picp` scores
  them against a target in cycles. Use `q05_edge`/`q95_edge` for coverage of a
  continuous target (§B.2).
- `weight_jitter=0` silently collapses every region to one component.
- Multi-partition region graphs give up structured decomposability, so
  `SquaredPC` refuses them (by design, with an explanation).
- The deletion/faithfulness column is scored with the PC's own scorer, so PC
  attributions have home-field advantage there. Localisation columns do not —
  ground truth comes from the generator.
- The generator was fixed twice (Mahalanobis scored 1.000 on v1; `decouple` was
  undetectable on v2). Results establish *mechanism*, not *magnitude*.

---

## 7. The experiment pipeline (added 2026-08-03 evening)

One config → `variants × seeds` runs → one summary. Five stages, all driven
from the same YAML and all writing the same structured rows:

| stage | what it measures |
|---|---|
| `ad` | detection vs the full baseline suite + the dead-sensor query + the exact typed (marginal/conditional/structural) split |
| `explain` | correctness (localisation vs ground truth), completeness (a theorem), faithfulness (deletion curves) |
| `rul` | censoring ablation, point/distributional accuracy vs ridge/MLP/CQR, survival under partial evidence |
| `calibration` | split conformal on the circuit's own predictive, engine-level split |
| `scaling` | tree vs DAG layout |

Properties worth relying on:

- **Resumable.** Each run writes `status.json` with a config hash; a completed
  run with a matching hash is skipped. Re-launching after a crash costs nothing.
- **Isolated.** A failing variant is recorded and the batch continues. A
  degenerate model is a *failed run*, not a row of numbers.
- **Self-describing.** Every run stores its resolved config, git commit + dirty
  flag, host, GPU, thread counts, per-epoch curves, peak RSS/GPU, the full
  console transcript, and raw score/attribution arrays for later re-analysis.
- **Honest by construction.** Splits are always by unit; the anomaly protocol is
  identical across synthetic and real; the seed count is printed next to every
  mean ± sd.

Config knobs live in `poc/time_series/config.py::DEFAULTS` (the full schema with
comments). `grid:` gives a cartesian ablation, `variants:` gives named
non-cartesian ones, and they compose.

**Device note.** `DEVICE=cpu` with `JOBS=3` is usually faster than one CUDA
process on this workload: a circuit is thousands of *small* ops, so it is
launch-latency bound, and BLAS oversubscription on tiny tensors is a real cost.
The GPU pays off as `K`, the window and the batch grow. Both paths are
supported and `env.json` records which ran.

---
---

# ARCHIVE — hand-off of 2026-08-02 ("making RUL work")

_Superseded by the section above: the overnight batch answered its open
questions. Kept for the hypothesis list (H1–H5), which is still the best
record of what was suspected and why._

## (original title) Hand-off — making RUL work

_Last updated: 2026-08-02.  Previous hand-off (ProbMoE routing) is archived at
the bottom of this file — nothing was deleted._

Read `CLAUDE.md` first (hard constraints), then
`poc/time_series/README.md` (what the PoC measures and how to run it).
This note covers **only** the unfinished RUL work.

---

## 1. TL;DR

The AD half of the PoC works and the story is settled: **parity on detection,
exclusivity on explanation.**  The RUL half does not work yet.  Three
independent negatives:

| symptom | measured |
|---|---|
| exact censored likelihood makes things **worse** | CRPS 13.89 → 14.57 (old gen); 23.16 → 24.45 (new gen, chain) |
| calibration is bad | PICP 0.49–0.58 vs nominal 0.90; conformal-CQR got 0.80–0.84 |
| plain ridge wins point accuracy | RMSE 25.0 vs 43.5 |

**The single most important thing to know: that negative result is NOT yet a
fair test.**  It predates three changes that were made for the AD side and were
never applied to `SurvivalPC`.  Two of them I wired up today (untested at
scale); the third invalidates the old numbers outright.

---

## 2. Why the negative is not yet a fair test

1. **The generator changed after the RUL runs.**  The original simulator smoothed
   white noise with a 3-tap kernel, which left almost NO within-window temporal
   structure (measured: permuting a channel's timesteps changed the mean
   |lag-1 diff| by 5%).  It is now driven by AR(1) processes (`phi_ar=0.85`,
   `simulate_fleet`).  Every RUL number quoted in the README's Experiment 3 was
   measured on the OLD generator and must be re-measured.
2. **`SurvivalPC` never got the chain/HMM structure.**  `attach_variable` only
   handled vtrees, so a `RegionNode` (chain) crashed it.  **Fixed today** — it
   now accepts region graphs, with `where="root"` (τ couples to the K hidden
   states at the head of the chain) and `where="deep"` (τ couples to the LAST
   timestep, the one nearest failure).  Untested beyond a 3-epoch smoke run.
3. **`SurvivalPC` never got the delta transform.**  **Fixed today**: `delta=True`
   first-differences the WINDOW only, never τ.  Unit determinant, so the joint
   stays exactly normalised.

So: before concluding anything, re-run the censoring ablation on the current
generator with `--vtree chain`.  That is step 0 below.

---

## 3. What is wired and ready (done today, needs validation)

| capability | how |
|---|---|
| chain/HMM structure for RUL | `--vtree chain` (also `chain_grouped`, `chain_full`) |
| τ coupling position | `--tau-where root` (default) or `deep` |
| first differences | `--delta` |
| n-ary curvature region graphs | `--vtree orc_rg` / `forman_rg` / `spectral_rg` |
| heavier censoring | `--censor-frac 0.6` |
| finer RUL resolution | `--bins 40` |

Smoke-tested: all four of {chain, time} × {root, deep} × {delta on/off} build,
train and predict.

---

## 4. The failure, characterised precisely

What DOES work: the survival function is qualitatively correct and monotone.
Grouped by TRUE remaining life (post symmetry-fix):

| true RUL | S(20) | S(40) | S(60) |
|---|---|---|---|
| 0–20 | 0.632 | 0.363 | **0.170** |
| 90–131 | **0.982** | 0.966 | 0.932 |

So the model has learned real signal and the query machinery is correct.  What
fails is (a) the censored term does not add value, (b) the predictive is
over-confident (PICP ≈ 0.5 at nominal 0.9) while MPIW is wide (~110 cycles),
which is the signature of a **biased, not merely sharp** predictive.

---

## 5. Hypotheses, ranked, with the signature that confirms each

**H1 — the censored term is drowned by the window likelihood.**  The loss is
`log p(x, τ)` where x is 112-dim and τ is 1-dim.  The gradient is dominated by
fitting the window; the censoring information enters only through the τ leaf and
the K×K coupling.  *Signature:* censored and uncensored arms converge to nearly
the same τ marginal.  *Test:* up-weight the τ term, or train the coupling with
the window sub-circuit frozen after a warm start.  **Try this first — it is the
most likely single cause.**

**H2 — 35% censoring at this n is not enough to pay for itself.**  Censored units
add rows but each carries less information (an inequality, not a value).
*Signature:* the gap closes or reverses as `--censor-frac` rises.  *Test:*
sweep `--censor-frac 0.2 0.35 0.5 0.7`.  This is also the **sanity check**: at
0.7 censoring, dropping censored units throws away most of the fleet, so if the
exact term does not win there, something is wrong with it.

**H3 — τ at the root is too weak a coupling.**  `where="root"` gives a K×K
discrete coupling; p(τ|x) can only be a convex combination of K profiles.  With
K=10 that is a coarse regression.  *Signature:* predictions cluster on ≤K
distinct values.  *Test:* `--tau-where deep`, and raise K.  **Check for
clustering first — it is a two-line diagnostic and would confirm H3 immediately.**

**H4 — global vs per-regime normalisation.**  `make_rul_task` uses
`Standardizer(per_regime=True)` already, but the regime is assigned per-timestep
and a window can straddle a regime change.  *Signature:* error concentrated on
windows that straddle.  *Test:* drop straddling windows and re-score.

**H5 — the RUL cap / binning is doing damage.**  `cap=130`, `bins=20` → 6.5
cycles per bin.  Most windows sit at the cap, so the τ marginal is dominated by
one bin.  *Signature:* the τ leaf's learned distribution is near-degenerate at
the top bin.  *Test:* `--bins 40`, and try a lower cap.

---

## 6. Run these, in this order

```bash
export PYTHONPATH=.
PY=~/miniconda3/envs/expllm_env/bin/python

# STEP 0 — the fair test that has never been run (do this first)
$PY -m poc.time_series.run_rul --seeds 0 1 2 --vtree chain --epochs 60 \
    --survival-demo --out logs/rul_chain_fair.json

# STEP 1 — the sanity check that must pass (H2).  At 70% censoring the
# drop-censored arm is starved, so the exact term MUST win here.
$PY -m poc.time_series.run_rul --seeds 0 1 2 --vtree chain --censor-frac 0.7 \
    --epochs 60 --no-partial --out logs/rul_heavy_censor.json

# STEP 2 — coupling capacity (H3)
$PY -m poc.time_series.run_rul --seeds 0 --vtree chain --tau-where deep --K 12 \
    --epochs 60 --no-partial
$PY -m poc.time_series.run_rul --seeds 0 --vtree chain --bins 40 --K 12 \
    --epochs 60 --no-partial      # also probes H5

# STEP 3 — the AD-side tricks
$PY -m poc.time_series.run_rul --seeds 0 --vtree chain --delta --epochs 60 --no-partial

# The two-line H3 diagnostic (run in a REPL before STEP 2):
#   pred = pc.predict(task.X_test); print(len(np.unique(pred["mean"].round(1))))
#   -> if that is <= K, the coupling is the bottleneck, not the training.
```

---

## 7. Pre-registered kill condition (agreed 2026-08-02)

> If, after STEP 0 and STEP 1, the exact censored likelihood does **not** beat
> the drop-censored arm on CRPS in the 70%-censoring regime, then T1 ("RUL as an
> exact censored survival query") is **not a contribution**.  It becomes a
> limitations paragraph, and the time-series work ships on the AD/explainability
> result alone.

Write the outcome into
`memory/research-evaluations/2026-07-30-time-series-ad-rul.md` either way.
Do not extend the RUL experiments past one week without passing this gate — the
AD half is the paper, and RUL is currently the weaker claim by a wide margin.

---

## 8. Where things live

| what | where |
|---|---|
| `SurvivalPC` (joint over window × τ, censored loss) | `poc/time_series/circuits.py` |
| `attach_variable` (τ placement; region-graph aware) | same file |
| censored loss (the box query) | `SurvivalPC.fit`, the `~obs` branch |
| exact survival / pmf / predict | `SurvivalPC.log_survival` / `log_pmf` / `predict` |
| RUL task + censoring + binning | `poc/time_series/data.py::make_rul_task` |
| metrics (CRPS, PICP, MPIW, NASA, calibration) | `poc/time_series/metrics.py` |
| driver + the three comparisons | `poc/time_series/run_rul.py` |
| chain region graph, delta transform | `src/probabilistic_circuits.py` |

**Gotchas that cost time today:**
- τ MUST get a leaf with a closed-form interval (`CategoricalLeaf`).  The
  heavy-tailed `InputNode` has no closed-form CDF and raises if boxed.
- `RegionGraphPC(weight_jitter=0)` silently collapses every region to one
  effective component — it looks fine in the loss and shows up only as
  `p(τ|x)` independent of `x`.  Leave the default.
- Baselines see only the failure-observed subset (they cannot use censored
  units).  That is the point of the comparison, not a bug.
- `logs/poc_ts_rul.json` is overwritten per run; pass `--out`.

---

## 9. Test status

216 tests pass (`pytest tests/ -q`, ~20 min).  New:
`tests/test_region_graph.py` (33 tests: DAG layout, box/interval queries,
region graphs, chain, delta, SOS-on-region-graph).

---
---

# ARCHIVE — previous hand-off (2026-06-24, ProbMoE routing)

_Kept verbatim; superseded as the "current" hand-off but not obsolete._

# Hand-off — ProbMoE routing for the encoder-free direction

_Last updated: 2026-06-24_

This note hands off the state of the **ProbMoE × mixture-of-PCs** work so the
next session (or person) can continue without re-deriving context. Read
`CLAUDE.md` first for the project's hard constraints; this file covers only
what changed and what's next.

---

## 1. What this is

We merged the routing idea from **ProbMoE: Differentiable Probabilistic Routing
for Mixture-of-Experts** (Zhao, Shao, Van den Broeck, Zeng — UCLA StarAI, ICML
2026) into **Direction 2** (the encoder-free mixture of per-modality
sub-circuits). Direction 2 was already a routed mixture of PCs with a
hand-waved router; ProbMoE supplies a principled, differentiable,
cardinality-constrained router with **exact** inference over the routing latent.

The research framing this implements — **F1: "cardinality / routing posterior
as a tractable anomaly query"** — and its full evaluation live in:

- `~/.claude/projects/<this-project>/memory/research-evaluations/2026-06-24-probmoe-cardinality-routing-query.md`
  (verdict: **REFINE, gated on a 1-week kill-shot**)
- Related arc: `2026-06-10-top-tier-positioning.md` (Paper A "AD as a tractable
  query" — F1's natural home) and `2026-06-11-ollivier-ricci-vtree.md`.

The thesis in one line: _don't ask how likely x is, ask how much of the model
x needs_ — `P(k|x)`, `H[P(k|x)]`, and the selection marginals `m_j(x)`, all
exact, jointly calibrated with the density under one probability measure.

---

## 2. What was built (this is done and tested)

Two files were extended; nothing was discarded (per CLAUDE.md).

### `src/probabilistic_circuits.py` — new section **6b** (SIMPLE / ProbMoE primitives)
Exact tractable inference over WHICH and HOW MANY experts are selected:
- `cardinality_log_normalizers(log_p, log_1mp) -> (B, N+1)` — exact `log Z_k`
  for every k, by the O(N·k) log-space DP. Differentiable. A finite log-zero
  floor (`-1e30`) replaces `-inf` so exact-k marginal gradients stay defined
  (a true `-inf` gives 0/0 = NaN in `logaddexp` backward).
- `cardinality_log_posterior(logits, k_min, k_max) -> log P(|S|=k | x)` — the
  Dynamic-k cardinality posterior (ProbMoE Eq. 9).
- `cardinality_moments(logits, ...) -> {map, expected, entropy}` of `P(k|x)`.
- `selection_marginals(logits, k_min, k_max) -> (B, N)` — exact
  `m_j = ∂log Z / ∂log p_j` (ProbMoE Eq. 6) via autograd; works inside
  `torch.no_grad()` (uses a local `enable_grad` scope).

All four are verified against brute-force enumeration in the tests
(normalizers ≈1e-8, marginals ≈1e-7, exact-k row-sums = k).

### `src/directions.py` — new class `ProbRoutedRawPC(RoutedRawPC)`
- Inherits the encoder-free sub-circuits, their exact-NLL / contrastive
  training, `add_modality`, and the **unchanged exact mixture density**.
- Adds an **encoder-free, input-conditional router** built only from the
  experts' own exact log-densities:
  `r_i(x) = β·(log p_i(x) − mean_j log p_j(x)) + b_i`, with trainable β≥0
  (`router_log_temp`, softplus) and per-modality bias (`router_bias`).
- Query menu (all exact): `cardinality_log_posterior`, `cardinality_moments`,
  `routing_entropy`, `selection_marginals`, `localize`, `routing_shift`,
  `routing_score(signal=…)`.
- Baselines for the kill-shot, deliberately included so the routing machinery
  has something to beat: `mixture_responsibilities` (GMM responsibility),
  `expert_argmax` (no-router localization), `expert_log_probs` (raw material).

### Tests — `tests/test_prob_routing.py` (12 cases, all passing)
Brute-force exactness, density invariant, localization, trainability, Exact-k,
`add_modality` growth, routing scores. The directions + inference + routing
suites together: **60 passed**.

---

## 3. THE critical invariant (do not break this)

The density `log_prob` / `score` is the **exact, normalized mixture** with
**data-independent** SumNode weights — unchanged from `RoutedRawPC`. The router
is an **auxiliary tractable query, NOT part of the density**. An
input-conditional mixing weight would make the mixture weights depend on x and
**break normalization** (see `probabilistic_circuits.py` module docstring,
lines 26–29). `test_density_invariant_preserved` pins this: per-expert
`log_partition ≈ 0` and `ProbRoutedRawPC.log_prob == RoutedRawPC.log_prob`.

If you ever route the density through the router, you have left the project's
core contract. The routing posterior rides *alongside* the density, never in
place of it.

---

## 4. Restrictions / assumptions baked in

- The router evaluates every expert on the same input, so it needs the routed
  sub-circuits to **share their feature dimension** (the aligned encoder-free /
  shared-vtree regime, as in `build_consensus_routed_pc`). The density's
  mixed-dimension fallback is untouched.
- `selection_marginals` differentiates w.r.t. `log p` only (logits detached) —
  it is a routing-analysis query, **not** a path for training gradients into
  the experts. Router params (β, bias) ARE trainable through the entropy/other
  queries if you want to fit them.
- `_m_bar` (training-time mean routing, the `routing_shift` reference) is a
  plain dict recomputed at `fit`; it is **not** in `state_dict`. Re-fit or
  re-cache after loading if you rely on `routing_shift`.

---

## 5. Next step — the F1 kill-shot (this decides PURSUE vs FOLD)

This is the single fail-fast experiment. ~1 week on existing code. **Pre-register
it; freeze `[k_min, k_max]` on a held-out modality — sweeping it to maximize
AUROC is p-hacking.** Score detection and localization SEPARATELY.

On 5–8 ADBench-style datasets, per point compute: exact `log p(x)`; the routing
signal (`routing_entropy`, `routing_shift`); and the cheap baselines
(`expert_log_probs` → per-expert max; `mixture_responsibilities` → entropy).

1. **Detection** — does routing add AUROC *after* density? Use a 2-feature
   logistic head {density, routing} vs density-only with nested CV, or a DeLong
   test on the combined-score ROC. NOT linear partial correlation (under-credits
   nonlinear/threshold interaction).
2. **Localization** — does exact `m_j` (`localize`) beat `expert_argmax` at
   identifying the corrupted modality on synthetic multimodal corruptions?

**Decision rule:** detection null AND localization matched by argmax → fold into
Paper A as a one-paragraph negative result, stop. Either decisively positive →
PURSUE, write the intro around *that* result, and **ship fast** (see §6).

Note from the smoke test: on a trivially-separable synthetic the density already
hits AUROC 1.0 and the routing signals are weak/redundant — exactly the
structural redundancy this kill-shot exists to measure. Real ADBench pairs are
the test.

Not yet built: a `scripts`/config entry to run the two tests on ADBench. The
data pipeline is in `src/datasets.py`; `config/adbench_demo.yaml` is a starting
point.

---

## 6. Strategic context (read before investing weeks)

- **Highest scoop risk in the portfolio.** StarAI authored ProbMoE + SIMPLE +
  the canonical PC framework and are the named threat on Papers A & C.
  "MoE-of-PCs with exact-k" is one conversation from their roadmap. Window =
  **months** (shorter than the 12–18 mo curvature thread).
- **Do not compete on the object** (MoE-of-PCs) — they win. Compete on the
  **AD-query semantics** (`P(k|x)`, localization, routing-vs-density
  calibration), which is outside their LLM-routing agenda.
- **Killed in ideation:** "fixed-k mixture-of-PCs is a more expressive density
  model" — false; it marginalizes to a plain sum node. Never claim density
  expressiveness from fixed-k.
- **Redundancy is structural, not just empirical** — `P(k|x)` is a function of
  the same per-expert likelihoods that produce `p(x)`. The 2025–26 MoE-OOD
  literature (e.g. arXiv:2509.23830) already found routing entropy adds only
  marginal OOD signal. Your wedge is *exactness + joint calibration*, and
  *localization*, not a horse-race-winning scalar.

---

## 7. How to run

```bash
# environment (conda env with torch)
~/miniconda3/envs/expllm_env/bin/python -m pytest tests/test_prob_routing.py -q

# quick smoke
~/miniconda3/envs/expllm_env/bin/python - <<'PY'
import torch
from src.directions import ProbRoutedRawPC
data = {f"m{i}": torch.randn(150, 6) + i*3.0 for i in range(3)}
det = ProbRoutedRawPC({m: 6 for m in data}, n_sum_components=3, seed=0).fit(data, epochs=15, lr=0.1)
det.validate()
x = data["m1"][:5]
print("E[k|x]:", det.cardinality_moments(x)["expected"])
print("localize:", det.localize(x))
print("score (=-log p):", det.score(x))
PY
```

---

## 8. Open follow-ups (parked, not blocking)

- The **k-DPP reframe** (likelihood-weighted k-DPP over PC experts) gives
  cardinality a generative-parsimony justification F1 currently lacks — worth a
  separate session IF the kill-shot lands positive.
- **F2** (sparse routing as a training-time scaling enabler) is only worth it if
  the forward-FLOP saving for PC experts is measured and real (>~2×); for PC
  experts the forward pass shares work with the marginal, so the saving may be
  partly illusory.
- **F3** ("k-selectivity" as a new PC structural property interpolating
  determinism k=1 and full mixture k=N) — verify it isn't implied by existing
  structured-decomposability work before betting.
</content>
