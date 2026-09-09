# Circuit diagnosis development and assumption test plan

## Objective and invariant

The research target is an accuracy–cost advantage for **statistical sensor-inconsistency diagnosis under partial observation**. The method remains a normalized probabilistic circuit over raw sensor windows. Joint densities, channel marginals, complement marginals, and masked conditional objectives are computed exactly under that fitted circuit, subject to floating-point arithmetic. No imputation, Monte Carlo marginalization, noninvertible encoder, or input-dependent mixture weights have been introduced.

Smoothness, decomposability, and the chosen vtree's structured decomposability remain enforced. Determinism is not asserted for full-support mixtures, and this work makes no exact MPE claim for them. Exact inference is a computational property; density accuracy, localization quality, and calibration are separate hypotheses.

The September 8 gate and its stored verdict are historical evidence. This is a new exploratory protocol, `diagnosis_v1`. Its configurations do not redefine that gate or manufacture a new PASS from its results.

## Implemented development

| Component | Implementation | What it enables |
|---|---|---|
| Independent checkpoint plumbing | `pipeline.py::_fit_window_pc(..., X_checkpoint=...)` and corrected gate call | Checkpoint selection uses the intended validation subset. |
| Three independent validation groups | `diagnosis.py::validation_partitions` | Checkpoint, calibration, and evaluation engines cannot overlap. Fewer than three engines is refused. |
| Independent calibration objects | `one_per_unit` and updated relational stage | One score-independent random window per calibration engine; no claim that overlapping windows are exchangeable. |
| Conservative finite-sample threshold | `conservative_threshold`, repaired `MaskCalibrator` | Insufficient calibration data gives an infinite threshold, rather than an unjustified finite alarm threshold. |
| Familywise alarm | `MaskCalibrator.familywise_alarms`; max scores in diagnosis stage | Calibrates the event that any observed channel alarms separately from per-channel alarms. |
| Generic exact diagnosis | `WindowPC.diagnosis_map` | Same joint, marginal, conditional, and relational definitions on blocked and unblocked circuits. |
| Capacity at interfaces | `WindowPC(boundary_components=..., upper_components=...)` | Changes boundary and upper widths independently of within-channel width. Uses the existing circuit components and validators. |
| Circuit-native mixture control | `WindowPC(channel_mixture=True)` | A root mixture of products of temporal channel circuits; the same latent index is paired across all channels. |
| Exact conditional training | `WindowPC.fit(conditional_weight=...)` | Joint NLL plus masked channel conditional NLL from the same joint model. |
| Explicit checkpoint criterion | `select_metric=nll/objective` | Separates training-loss changes from checkpoint-selection changes. |
| Convergence instrumentation | `patience`, `min_epochs`, actual updates and histories | Records training budget, selected epoch, and early stopping; defaults preserve legacy training. |
| Paired score study | New `diagnosis` pipeline stage | Compares joint NLL, product-of-channel-marginals NLL, marginal maximum, conditional maximum, and relational maximum. |
| Known-law positive/negative controls | `diagnosis_controls.py` | Gaussian nominal law, independent whole-channel replacement, exact oracle marginals, independence and lost-evidence controls. |
| Ambiguity-aware localization | `localization_metrics` | No forced winner among tied channels; reports absent targets and end-to-end unique top-1 success. |
| Numerical singleton identity | `diagnosis_map` / `relational_map` | One observed channel has conditional = marginal and exactly zero relational score. |
| Honest work accounting | `QueryCost`, revised `bench_relational.py` | Node-times-row work proxy, boundary elements/bytes, output size, and median warmed timings, alongside traversal counts. |
| Paired uncertainty analysis | `compare_diagnosis.py` | Engine-cluster bootstrap after checking exact input/label/mask alignment and completed-run status. |

The shallow mixture is itself a circuit, not a replacement of the circuit framework. Each component uses tractable channel subcircuits. Above those boundaries, matched-index products and a single root sum encode

\[
p(x_O)=\sum_k\pi_k\prod_{c\in O}p_{kc}(x_c).
\]

This provides a serious test of whether additional hierarchical mixing is needed. Increasing `boundary_K` or `upper_K` changes the parameter count; the new study records the actual count and does not label these arms parameter-matched.

The conditional objective is

\[
L=L_{\mathrm{joint}}+\lambda C\,\mathbb E_{c,O}[-\log p_\theta(X_c\mid X_{O\setminus c})].
\]

One target channel and observation mask are sampled per training minibatch. The query itself is exact; sampling determines the stochastic training objective, not an approximation to marginal inference. Validation uses a fixed query library with each channel as target. `train_mask_drop_prob` controls independently dropping other channels. Joint NLL and training objective are logged separately; the selection-loss history is labeled according to its chosen criterion.

## What the new evaluation means

All new real-background studies evaluate validation engines. Dataset construction may load the official test split, but the diagnosis stage neither scores it nor uses it for model selection. Each retained clean evaluation window is paired with a corrupted copy. Identical seeds and configurations of the data/masks produce the same pairs across architecture variants. Artifacts retain input windows, original indices, engine IDs, affected-channel indicators, labels, masks, and all scalar scores.

Calibration uses the untouched calibration-engine group, with one random window per engine. The evaluation includes more windows per engine, so aggregate FPR is a descriptive window-weighted estimate, not an independent-binomial engine estimate. Engine resampling is used for comparative AUROC uncertainty. A confirmatory false-alarm study must use the prespecified sampling object consistently, as described below.

For known-law controls, each window is an independently generated unit; the protocol explicitly records this difference. The corrupted channel is channel 0, replaced using a fresh draw from the true nominal distribution. Its entire trajectory marginal is preserved in distribution. This is a known block-independence alternative, not a physical fault simulator. The Gaussian oracle is a statistical reference; its simple implementation is not a competitive Gaussian latency baseline.

Real-background replacements currently use the nominal training donor pool **without regime/health matching**. Artifacts declare that limitation. Results from this arm cannot establish a purely relational anomaly claim. Context-matched donors and external fault evidence are required before confirmation.

`loc_ap` is averaged over corrupted windows with both affected and unaffected observed channels. Additional columns count no observed target, all observed channels affected, and score ties. Unique top-1 accuracy is conditional on a unique ranking; `end_to_end_unique_top1` counts missed detection, ambiguity, and hidden targets as failures over all corrupted windows. Visible targets need not be identifiable: those concepts remain distinct.

For the joint detector, localization uses conditional surprise and is explicitly labeled as a different score. The relational maximum uses relational channel scores for both its alarm and its localization. No completeness claim is attached to either.

## Assumption matrix

Every hypothesis below has a falsifier. Implemented mathematical regression tests check identities and failure cases; they do not establish empirical superiority.

| ID | Assumption | Test and controls | Falsifier / response | Readiness |
|---|---|---|---|---|
| E1 | New capacity settings preserve the circuit contract | Structural validators, log partition, compiled/reference queries, all masks on small non-factorized models | Any mismatch blocks all scientific use | Implemented regression tests |
| E2 | Shared-latent mixture has the intended distribution | Compare circuit result with explicit component log-sum-exp of channel densities | Cross-index mixing or incorrect weighting | Implemented regression test |
| E3 | Fast queries match generic marginals | Every observation mask, heterogeneous masks per batch, multiple widths/layouts | Error beyond floating-point tolerance, nonfinite observed score | Implemented tests and runtime study checks |
| E4 | Conditional training retains exactness | Train with masked conditional loss; compare final compiled/generic marginals and partition | Normalization or gradient-path/query disagreement | Implemented regression tests |
| E5 | Work is being measured fairly | Compare repeated masks inside one chunk and several chunks; record output and boundary sizes | “Flat” work caused by counting only traversals | Implemented tests and benchmark |
| P1 | Checkpoint/calibration/evaluation are separate | Assert engine-set disjointness and captured checkpoint argument | Any engine reused across roles | Implemented regression tests/protocol artifact |
| P2 | Calibration can support the chosen level | Infinite-threshold boundary test; log calibration count | Fewer than 19 independent objects at alpha=.05 cannot yield a finite generic threshold | Implemented; power may be zero by design |
| P3 | Missing channels are not used as evidence | Hidden-target oracle test, NaN channel scores, whole-block marginalization | Imputed/hidden sensor influences observed query | Implemented tests |
| P4 | Comparisons are paired and independent at the claimed unit | Artifact equality checks; cluster bootstrap by engine | Different corruptions, masks, observations, or resampling of windows as engines | Implemented comparison CLI |
| S1 | The current failure is partly score mismatch | Same fitted circuit, all five score views; per-type detection and localization | Relational score offers no useful improvement on genuine relational controls | Runnable control and FD001 studies |
| S2 | Larger absolute dependence means better modeling | Signed/absolute discrepancies versus oracle error and diagnostic accuracy | Large discrepancy with poor conditional accuracy/power | Diagnostics implemented; inference “larger is better” rejected |
| S3 | Blocking's limitation is interface capacity | Compare blocked, interface, within-capacity, uniform-wide, shallow-mixture, unblocked; parameters and latency reported | Added interface capacity does not move diagnostic frontier | Runnable control study; confirm under matched budgets later |
| S4 | Apparent architecture failure is underoptimization | Longer maximum budgets, minimum epochs, patience, LR sweep; compare same architecture/objective | Stable converged result remains weak | Adam budget controls implemented; EM/Anemone deferred |
| S5 | Query-oriented training improves useful relationships | Conditional objective with NLL versus objective checkpoint selection; repeat on unseen masks/mechanisms | Gains restricted to selection metric, mask library, or training injector | Two selection variants runnable; mechanism generalization pending |
| S6 | A deep upper circuit is necessary | Circuit-native shallow mixture, matched quality/cost frontier | Mixture matches/betters deep model at lower cost | Runnable mixture arm |
| S7 | The anomaly is genuinely cross-channel | Known-law independent trajectory replacement; oracle; independence negative control; target-only temporal detector | Per-channel trajectory distribution already separates labels | Known-law controls runnable; donor audit pending for real data |
| S8 | Remaining sensors identify the faulty source | Two-sensor symmetry, singleton zero, hidden target, progressively remove witnesses | Ties or identical normal/abnormal observed laws | Mathematical controls implemented; graph-specific witnesses pending |
| S9 | Marginalization preserves useful power | Same mask and fault across all methods, stratified by available witnesses | Oracle also has no power: revise operating regime; model-only failure: improve learning | Basic random/group/singleton masks runnable |
| S10 | False alarms remain controlled under masks | Independent calibration/test engines; max-channel score; known versus unseen fixed masks; repeated trials | FPR inflation or frequent infinite thresholds | Known-mask machinery implemented; unseen-mask guarantee pending |
| S11 | The method handles fault-dependent dropout | Generate mask conditional on hidden value/fault; contrast with independent masks | Nominal calibration fails under informative missingness | Planned stress test; no such guarantee claimed |
| S12 | PC advantage survives strong tractable alternatives | Cached shrinkage/full/low-rank Gaussian and full-covariance GMM, plus mixture circuit | Competitor dominates accuracy–cost frontier | Mixture implemented; optimized fitted Gaussian/GMM pending |
| S13 | Speedup is algorithmic and operational | Warm/cold caches, batch/channel/mask sweeps, same precision, synchronized repetitions, memory | Advantage vanishes under batching/caching or preprocessing cost | Circuit benchmark implemented; cross-family comparison pending |
| S14 | The conclusion generalizes | Frozen candidate, independent dataset and fault mechanisms, engine-level intervals | Only one corpus/injector benefits | Planned confirmation |
| S15 | Restructuring/distillation can preserve the useful queries | First show an expressive teacher works; measure separators, query distortion and ranking margins | Conversion explodes or student loses relational quality | Conditional follow-up, not implemented |

## Execution sequence

Run from the repository root with the environment containing PyTorch and the project's dependencies. On this workspace:

```bash
export PYTHONPATH=.
DIAG_PY=/Users/eriknielsen/miniconda3/envs/expllm_env/bin/python
```

### 1. Correctness and integration

```bash
"$DIAG_PY" -m pytest tests/test_diagnosis_study.py tests/test_tier1_relational.py tests/test_tier0_evaluation_repairs.py -q
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_smoke.yaml
```

The smoke configuration has five variants, one seed, and three epochs. Its purpose is integration. It must not be cited as evidence that one method performs better. Near-flat scores are explicitly flagged; a threshold computed from little calibration data may legitimately be infinite.

### 2. Known-law positive and negative controls

```bash
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --dry-run
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --set eval.control_cross=0 --log-root logs/ts/diagnosis_controls_independent
```

The positive-control configuration expands to 10 variants × 3 seeds = 30 runs, with a maximum of 150 epochs. The independent negative control doubles that budget if all variants are used; start with `--only blocked` and `--only mixture` to check the mechanism before expanding it. Do not change seed lists in response to unfavorable outcomes.

First inspect the known-law oracle. It should discriminate joint versus replacement when correlation supplies information; target-hidden observations are identical by construction, and singleton relational scores are exactly zero. Individual target-channel marginal scores have no asymptotic discrimination under exact marginal preservation. A sum or maximum of several channel scores can still respond to a dependence change through its distribution, so it is not the same negative control as a single-channel detector.

Then compare the trained circuits with the oracle and with each other. Required plots: subtype AUROC and localization against actual parameter count; localization against warmed latency; joint and selection loss against optimizer updates; signed dependence discrepancy against performance. A stronger interface with more parameters is a capacity result until a matched-budget frontier establishes otherwise.

### 3. Training-budget ablation

```bash
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --only blocked --set model.epochs=300 model.lr=0.01 model.patience=50 --log-root logs/ts/diagnosis_controls_longer
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --only conditional_nll_selection --log-root logs/ts/diagnosis_controls_conditional_nll
```

`--only` is a substring filter; `blocked` also matches `unblocked`, intentionally providing the main comparison. Check `--dry-run` whenever selecting subsets. Evaluate training on gradient-update and wall-clock axes. Early stopping is a validation decision, not proof of convergence or global optimality. An EM/Anemone comparison requires a separate implementation and gradient/likelihood audit before it can be treated as a completed assumption test.

### 4. Real-background exploration

```bash
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_fd001.yaml --dry-run
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_fd001.yaml
```

This expands to 6 variants × 3 seeds = 18 runs, up to 180 epochs each. It is potentially expensive. Start with a single exploratory seed and the main structure comparison to estimate actual runtime, then execute the frozen full batch. With about 30 validation engines divided three ways, there are only about 10 independent calibration windows. Alpha=.10 may yield a finite threshold; alpha=.05 generally cannot. More windows from those engines do not fix that limitation.

Keep these results labeled semi-synthetic and exploratory. Before claiming a relational real-data gain, add context-matched donors using documented exogenous operating conditions or an independently fixed health proxy. Do not match using the held-out anomaly label or model score. Inspect donor/recipient population balance and train a target-channel-only temporal discriminator as a shortcut audit.

### 5. Paired comparisons and cost curves

```bash
"$DIAG_PY" -m poc.time_series.compare_diagnosis logs/ts/diagnosis_controls/interface/seed11/artifacts/diagnosis_mask0.npz logs/ts/diagnosis_controls/blocked/seed11/artifacts/diagnosis_mask0.npz --score relational_max --out logs/ts/diagnosis_controls/interface_vs_blocked_seed11.json
"$DIAG_PY" -m poc.time_series.bench_relational --channels 4 8 16 --masks 1 4 16 --epochs 3 --out logs/ts/diagnosis_cost_curve.json
```

Compare matching seeds/masks only. The CLI refuses unpaired data and incomplete stages. Its interval concerns a paired AUROC difference across independent units for one fitted-model pair. It does not incorporate all training uncertainty or correct for selecting a winner among many variants. Repeat seed-level comparisons and keep the two sources of variation separate.

The cost benchmark reports three warmed repetitions. Its node-evaluation count is a node-times-row proxy, not FLOPs: sum fan-in and kernel behavior still matter. Peak boundary bytes describes the boundary tensors, not the process's full peak memory. Measure end-to-end process/device memory and cold initialization separately before making a deployment claim.

### 6. Confirmation after exploration

Freeze one candidate and the strongest comparator, preprocessing, masks, scores, calibration sampling unit, and metrics. Select an independent source with a credible fault interpretation; another C-MAPSS subset is replication within a simulator family, not a wholly independent domain. Use fresh held-out units and mechanisms, not merely new random initializations on data repeatedly inspected during development.

Two proposed confirmation designs are available; choose one before running it:

1. **Efficiency claim:** paired localization AP is noninferior within 0.02, using a one-sided confidence bound, and median end-to-end diagnosis latency improves by at least 2× under a prespecified mask workload.
2. **Quality claim:** localization AP improves by at least 0.03 in point estimate with a confidence interval excluding zero, at matched latency or an agreed resource ceiling.

These are proposed practical margins, not the old gate and not literature-derived universal constants. Use pilot variance and the intended application to freeze the final sample size and margins before the confirmation data are examined. Require calibrated alarm behavior as an additional condition. For a nominal .05 alarm, a possible tolerance is an independent-unit upper confidence bound no greater than .07, but available fleet sizes may be inadequate; report that limitation rather than replacing engines with overlapping windows.

For a random-window operational claim, draw evaluation windows from new engines using the same rule as calibration. For a trajectory-level alarm, calibrate and evaluate a trajectory maximum. A benchmark AUROC over all windows may accompany either, but does not substitute for its operational false-alarm endpoint.

## Stop, refine, or expand

**Stop and fix correctness** if exact-query checks, normalization, split isolation, or artifact alignment fail. Statistical claims cannot proceed around an invalid query or protocol.

**Refine the model** if the oracle succeeds but the trained circuit fails. Use score choice, optimization, and interface-capacity ablations to identify the cause. If the shallow mixture matches the deep circuit, use the simpler circuit or change the claim; depth is not a contribution by itself.

**Refine the operating regime** if the oracle also fails when informative witnesses disappear. Add abstention or report ambiguous sets. Do not promise recovery of information absent from observations.

**Expand only after useful relationships are demonstrated.** Exact restructuring or query-aware distillation is the next ambitious circuit development, conditioned on a strong teacher and acceptable separator complexity. SOS, routing, RUL extensions, causal claims, and universal anomaly detection remain deferred. They are not required to test this claim and would obscure its failure modes.

## Verification record

The implementation was checked with the repository test suite excluding the known slow legacy `test_density_pc_128_features`: **469 passed, 1 deselected** at that checkpoint. This includes the first 18 new diagnosis tests. Subsequent targeted checks cover the additional positive-control and singleton-identity changes. The five-variant smoke study completed; final test/run counts are recorded in `DIAGNOSIS_IMPLEMENTATION_STATUS_2026-09-09.md`.

The larger 30-run known-law study and 18-run FD001 study are configured and dry-run validated, not claimed as completed experiments. Optimized fitted Gaussian/GMM baselines, EM/Anemone, context-matched donors, informative missingness, novel-mask statistical guarantees, and restructuring/distillation remain explicitly planned rather than silently treated as implemented.
