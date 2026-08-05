# Hand-off — time-series PoC

_Last updated: 2026-08-05 (evening), after the diagnostic-suite pass (§B).
The σ-floor episode from earlier the same day follows unchanged in §A; the
2026-08-03 hand-off from §0; two more are archived after it. Nothing was
deleted._

Read `CLAUDE.md` (hard constraints), then
[`poc/time_series/launch/README.md`](poc/time_series/launch/README.md) (how to
run anything) and [`data/README.md`](data/README.md) (what is real and what is
injected). This file is the state of play and the next actions.

**If you read one thing: §B.2. Two of the results in §2 are now suspended
pending one re-run, and §B.7 is that re-run.**

---

## B. LATEST (2026-08-05, evening) — three diagnostic suites, and what they found

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

Tests: 87 green across the three new suites + `test_ts_pipeline` +
`test_leaf_sigma_floor`; 173 green across the fast half of `tests/`; the full
suite passes (exit 0) but takes >10 min, dominated by `test_inference` and
`test_vtree`. Smoke run clean end to end on all four stages. **All four strict
xfails are now cleared** — they were the signal that the open items were open,
and they flipped as each was fixed.

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
