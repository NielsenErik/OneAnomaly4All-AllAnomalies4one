# Diagnosis implementation plan — 10 September 2026

## Scope and current decision

Complete the experimental framework for statistical sensor-inconsistency diagnosis under partial observation. Preserve normalized raw-input circuits and exact marginal queries. This is the current implementation roadmap; the September 8 plan and failed gate remain historical records. The September 9 literature review supplies the motivation, and `DIAGNOSIS_TEST_PLAN_2026-09-09.md` supplies the assumption matrix and scientific protocol.

The existing diagnosis implementation is runnable. Its smoke study establishes integration, not useful learned relationships or a scientific advantage. Do not expand to SOS, multimodal routing, RUL, causal attribution, or restructuring before the oracle/model/baseline comparisons below justify expansion.

## Completed foundation

- Tier 0 split, provenance, metric and checkpoint repairs.
- Channel boundaries, two-pass relational maps, heterogeneous masks, subset search, conservative mask calibration.
- Independent within-channel/boundary/upper widths; shallow shared-latent mixture; conditional training and explicit checkpoint criterion.
- Engine-disjoint checkpoint/calibration/evaluation groups, paired corruptions, ambiguity-aware localization and cluster AUROC bootstrap.
- Gaussian known-law oracle, independent trajectory replacement, independence and hidden-evidence controls.
- Five-arm smoke configuration, 30-run controls configuration and 18-run exploratory FD001 configuration.

See `DIAGNOSIS_IMPLEMENTATION_STATUS_2026-09-09.md` for the completion audit. The work below is pending unless explicitly identified as an existing capability. Proposed new filenames are implementation targets, not existing modules or runnable commands.

## Delivery order and acceptance criteria

### 1. Close the correctness and integration checkpoint

Existing files: `tests/test_diagnosis_study.py`, `tests/test_tier1_relational.py`, `tests/test_tier0_evaluation_repairs.py`, `poc/time_series/diagnosis.py`, `compare_diagnosis.py`.

Run the suite and five-arm smoke into a fresh output directory. Verify completed status for every variant, four mask artifacts per variant, paired comparison compatibility and singleton identities. Record the exact command, environment, exclusions and outcome in the status report. Keep old logs intact. Any normalization, exact-query, split-isolation or artifact-pairing failure blocks subsequent experiments.

### 2. Establish whether there is useful evidence to learn

Existing files: `diagnosis_controls.py`, `diagnosis.py`, `config/ts/diagnosis_controls.yaml`.

Execute the frozen positive-control grid, then the independent negative control. First inspect the oracle; then compare blocked, unblocked, interface, shallow-mixture and objective variants. Store all failed/void runs as well as successful ones. Report signed discrepancy, absolute discrepancy, joint NLL, conditional query error against the oracle, detection, ambiguity and localization separately. Add oracle conditional-error columns in `diagnosis.py` if they are needed for the report; absolute dependence alone is not an accuracy measure.

Acceptance: the correlated oracle detects replacement; the independent oracle has zero relational scores within numerical tolerance; hidden-target observations agree exactly; singleton relational scores are zero. Evaluate a target-channel-only detector as the marginal-preservation control. Learned models must show useful relational behavior before any expensive real-data expansion. A failed learned model with a successful oracle triggers step 3; a failed oracle triggers investigation of the observation regime.

### 3. Separate capacity from optimization

Existing files: `circuits.py`, `config.py`, `config/ts/diagnosis_controls.yaml`; new config: `config/ts/diagnosis_optimization.yaml`.

Use the existing width and objective variants, then freeze a small learning-rate/training-budget grid. Compare actual optimizer updates, selected epoch, wall time and parameter counts. Select hyperparameters only on checkpoint units. Compare conditional training with both NLL and objective selection. Keep training seeds distinct from independent evaluation units.

Acceptance: a table and loss curves distinguish an undertrained arm from a stable weak arm, and show the quality–cost frontier using actual resources. Larger-width results are capacity ablations until budget matching is measured. Add EM/Anemone only as a separate follow-up if optimization remains the identified bottleneck; its implementation must pass normalization, likelihood-update and exact-query regression checks before scientific use.

### 4. Implement equally capable fitted baselines

New file: `poc/time_series/diagnosis_baselines.py`; integrate in `diagnosis.py` and `config.py`; new tests: `tests/test_diagnosis_baselines.py`.

Provide fitted shrinkage/full-covariance Gaussian, diagonal-plus-low-rank Gaussian and full-covariance GMM. Keep the existing known-law Gaussian oracle as a reference with known parameters. Share a `fit(X_train)` / `diagnosis_map(X, observed)` contract whose output matches the existing circuit map (`log_px`, marginal/conditional quantities and `R`). Preserve the current flattened timestep/channel indexing. Support shared and per-row masks by grouping unique masks and restoring row order.

Fit means/covariances/mixture parameters on training data only. Select covariance regularization, rank and component count on checkpoint data. Cache observed-block factorizations and channel/complement terms by mask; use Cholesky solves and log determinants. Record stabilization and fitting failures. GMM marginals must sum component marginal densities with normalized fitted weights; do not average component log densities or retain fixed posterior weights after changing evidence. Invalidate caches after fitting or device/dtype changes.

Acceptance: small-dimensional direct Gaussian and explicit mixture calculations agree with every mask query; normalization and singleton identities hold; heterogeneous masks match independent calls; cached and uncached outputs agree. Nearly singular training data produce a documented regularized model or an explicit failure. All methods emit the same score/artifact schema and use identical splits, corruptions and masks.

### 5. Remove real-background shortcuts

Existing files: `data.py`, `datasets.py`, `diagnosis.py`; proposed new file: `diagnosis_donors.py`; new tests and config for donor matching.

Carry operating-condition metadata and donor/recipient engine IDs through window construction. Define matching using exogenous regime variables or a frozen, independently justified health proxy. Use training donors only, a documented distance/caliper and explicit no-match handling. Save donor IDs, matching variables, distances, unmatched counts and population-balance summaries. Fit a target-channel-only temporal classifier on separate development units to audit marginal shortcuts; never use anomaly labels or model scores for matching.

Acceptance: tests prevent held-out donors and label-based matching, reproduce assignments from seeds, and cover unavailable matches. Compare matched and unmatched corruption results. Keep the real-background arm labeled semi-synthetic; approximate matching does not establish exact marginal preservation or physical fault validity.

### 6. Test availability, identifiability and calibration

Existing files: `diagnosis_controls.py`, `relational.py`, `diagnosis.py`; proposed config: `config/ts/diagnosis_missingness.yaml`.

Add a known dependency graph with designated witnesses and progressively remove them. Add independent random masks, fixed unseen masks and fault/value-dependent dropout as separate regimes. Report visible target count, available witnesses, ambiguity, abstention, conditional localization and end-to-end success. Generate informative masks explicitly from the specified hidden value/fault mechanism and record that mechanism.

For an operational random-window claim, draw one score-independent window per calibration and evaluation engine by the same rule. For trajectory alarms, calibrate and evaluate trajectory maxima instead. Keep descriptive all-window AUROC separate. Freeze alpha and the mask workload before evaluation; insufficient independent calibration objects must retain an infinite threshold. Calibrate max-channel alarms separately from per-channel scores. Do not apply a known-mask guarantee to a new or informative mask mechanism.

Acceptance: missing evidence cannot influence queries; controls expose unidentifiable cases; repeated independent-unit trials report FPR uncertainty, power and infinite-threshold frequency. Unseen/informative-mask outcomes are stress-test results unless a separate justified procedure establishes a guarantee.

### 7. Produce fair comparisons and a reproducible report

Existing files: `bench_relational.py`, `compare_diagnosis.py`, `aggregate.py`; proposed new file: `report_diagnosis.py`.

Extend paired comparisons to localization AP and end-to-end success, resampling engines and retaining all paired windows within each sampled engine. Preserve exact input/label/mask alignment checks and reject incomplete stages. Report training-seed variability separately. Document ties, excluded/undefined endpoints and multiplicity from exploratory selection.

Benchmark circuit and fitted baselines at the same precision, device and data with channel/window/batch/mask sweeps. Record cold initialization, warm query latency, cache construction, cache size, output size, actual parameters, process/device peak memory and synchronized repeated timings. Keep node-times-row work labeled as a proxy, not FLOPs. Include preprocessing in end-to-end latency and distinguish throughput from single-window latency.

Acceptance: one report command consumes completed artifacts and generates metrics tables, intervals, quality–cost curves, convergence plots and failure counts without manually choosing favorable rows. Identical paired scores yield zero deltas; mismatched pairs and incomplete runs are refused. No speed claim may use only the favorable workload.

### 8. Freeze and execute confirmation

Dependencies: steps 2–7 demonstrate a useful candidate and a strong comparator. Write a new versioned confirmation config and protocol before examining confirmation data.

Freeze candidate, comparator, preprocessing, resource budget, scores, localization/tie rules, alpha, sampling unit, fault mechanisms, mask workload, sample size and intervals. Use pilot variance for sample size; fresh initialization seeds on repeatedly inspected data are not fresh confirmation units. Choose a credible independent dataset; another C-MAPSS subset is within-family replication.

Choose one primary endpoint in advance: localization AP noninferiority with a proposed 0.02 margin plus at least 2× end-to-end latency improvement, or a proposed 0.03 localization AP improvement with an interval excluding zero under a fixed resource ceiling. These margins require application/pilot justification before freezing. Require an operational false-alarm endpoint with independent-unit uncertainty as an additional condition.

Acceptance: release the frozen protocol, provenance, complete run inventory, primary interval and negative/ambiguous cases. If the comparator dominates, report the negative result. If evidence vanishes with witnesses, narrow the operating claim. Only a successful expressive teacher and a demonstrated cost bottleneck justify a separate restructuring/distillation plan.

## Commands available now

Run from the repository root; replace the interpreter with the environment's Python on another machine.

```bash
export PYTHONPATH=.
DIAG_PY=/Users/eriknielsen/miniconda3/envs/expllm_env/bin/python
"$DIAG_PY" -m pytest -q -k 'not test_density_pc_128_features'
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_smoke.yaml --log-root /tmp/diagnosis_completion_20260910
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --dry-run
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_controls.yaml --set eval.control_cross=0 --log-root logs/ts/diagnosis_controls_independent
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_fd001.yaml --dry-run
"$DIAG_PY" -m poc.time_series.runner config/ts/diagnosis_fd001.yaml
```

The full positive/negative controls require 30 runs each; FD001 requires 18. Dry-run validates expansion, not data availability or runtime correctness. Estimate compute from a fixed pilot before scheduling the full studies. `--only` is a substring filter: `blocked` also selects `unblocked`; inspect dry-run output. Use fresh log roots for changed protocols, preserve all seeds and failures, and do not relabel exploratory output as confirmation.
