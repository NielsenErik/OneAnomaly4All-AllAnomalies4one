# Hand-off — making RUL work (time-series PoC)

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
