# POC research and implementation review — 8 September 2026

**Decision: pursue a bounded validation of conditional diagnosis under missing sensors. The current evidence does not justify the broad universal-anomaly-detector thesis, unique explainability, or superior censored RUL prediction.**

The implementation is a useful research platform. The scientific contribution is still unproven. Some negative conclusions in the latest handoff are also overstated: the correct response is neither to trust the old positive story nor to accept every subsequent diagnosis of failure.

Scope: inspected the current source, configurations, handoff, dataset documentation, individual run statuses/results/metrics/training histories, and relevant primary literature. Current checkout: `0a2162f`; existing user edits to `hands_off.md` and `poc/EVALUES_Brainstorm.md` were preserved. Historical runs include dirty working trees, so a recorded commit is not a complete reconstruction of the executed code. No expensive benchmark retraining was performed. This is a targeted literature assessment, not an exhaustive priority search.

Verification: **150 tests passed in 22.09 seconds**, across `test_pc_conformal`, `test_experiment_hygiene`, `test_ad_diagnostics`, `test_rul_diagnostics`, `test_compiled_circuit`, and `test_ts_pipeline`. Additional small probes checked alpha sensitivity, finite-sample quantiles, endpoint selection, and tied average precision. Passing these tests supports the tested mechanisms; it does not establish the research claims.

## 1. What exists, and what that proves

There are two projects in the repository: the original multimodal/transfer framework in `src/`, and the more extensively evaluated windowed time-series POC in `poc/time_series/`. Their evidence should not be interchanged.

The time-series platform implements density scoring, whole-channel marginalization, conditional channel scores, chain-rule attribution, a discretized joint model of sensors and remaining life, censored likelihood, and calibration wrappers. It has structure/capacity controls, numerical diagnostics, compiled-versus-recursive checks, unit-based train/test separation, and reproducible configuration machinery. These are substantial reusable assets.

However, three different meanings of “exact” must stay separate:

1. **Exact inference under the fitted model:** computing its density, marginal, or discrete survival probability without Monte Carlo integration.
2. **Accurate modeling of the data:** whether that fitted distribution captures relevant dependencies and generalizes.
3. **Valid operational uncertainty:** whether coverage or false-alarm guarantees apply to the actual sampling/deployment protocol.

The first does not imply the other two. A correctly normalized wrong model can be confidently wrong. A latent PC is exact over its representation; a noninvertible encoder does not make it an exact density over the original inputs. Similarly, exact inference over RUL bins is not an exact continuous failure-time model.

## 2. Findings that materially change the current interpretation

### A. The purported held-out NLL is training NLL

`pipeline.py::stage_ad` calls `_held_out_nll(pc, task.X_train)`. That helper scores the first 1,024 training windows. The older `run_ad.py::vtree_ablation` likewise uses `task.X_train[:512]`.

Thus the summary column `train_nll` is substantially correctly named; the handoff's assertion that it is mislabeled held-out likelihood is wrong. The reported values cannot establish generalization or diagnose overfitting versus undertraining. A separate set of healthy validation engines is needed, with structure learning and all preprocessing fitted without those engines.

### B. The AE “sampling-SHAP” baseline is not SHAP

`explain.py::sampling_shap` replaces one channel at a time with a background draw and averages **absolute** changes in AE reconstruction score. It neither samples coalitions nor averages signed marginal contributions across permutations. More samples converge to this replacement-sensitivity statistic, not to a Shapley value.

Consequences: withdraw “exact attribution beats approximate SHAP,” the interpretation that Monte Carlo error explains the gap, and associated SHAP efficiency claims. Keep the measured statistic under an accurate name if useful. A valid comparison must specify the explained score, coalition value function, background semantics, and computational budget.

The PC's `shapley_channels` does average signed contributions along permutations, but for the game **v(S) = −log p(x_S)**. This is not automatically conditional-expectation SHAP of the full anomaly score. Its ordering average is sampled. Exact coalition queries do not make the finite-sample Shapley estimate exact.

### C. Completeness is being established for a different explanation

The localization winner is often the leave-one-channel-out conditional surprise. The completeness test checks `chain_rule_attribution`, a different construction. In general, the sum of leave-one-out conditional surprises is not the joint NLL. One cannot combine localization of one method and completeness of another into a single “correct and complete explanation” claim.

Chain-rule completeness is a valid telescoping identity. It holds for any distribution whose relevant marginals can be evaluated, including a Gaussian. It is not evidence of causal correctness or a new theorem about circuits. Permutation-based contribution estimates can also telescope exactly despite approximation in the individual attributions.

### D. The seven RUL failures concern imputation, not full-input collapse

All **7 failed statuses out of 12** in `logs/ts/cmapss_rul` point to `_partial_evidence`, specifically `imp = pc.predict(X_imp)`, after mean-imputing missing channels. Full-input evaluation had already produced results.

For example, `censor_frac-0.7/seed1/results.jsonl` records the censored model's prediction SD as **13.92 cycles [all]** and **25.80 [last]**. The **0.759-cycle** SD in its failure message belongs to the imputed input. The handoff's headline diagnosis that these failures show the full RUL model is constant is therefore unsupported.

This is potentially interesting evidence of sensitivity to imputation. But the exception occurs before the exact-marginal comparison completes, so it does not yet establish that marginalization solves it. Capture degenerate imputation as an explicit comparison outcome and complete the marginal arm. Retain failure diagnostics rather than silently accepting collapsed predictions.

Separately, an AD structural-score AUROC of 0.582 is neither literally chance nor proof that the model factorizes. AUROC measures discrimination for a particular corruption distribution, not the amount of learned dependence. It cannot establish sensor–RUL independence in a different fitted model. Compare conditional PMFs directly, against shuffled inputs and an unconditional target model.

### E. Failed-run aggregation is unsafe, but its mechanism differs from the handoff

`ts_logging.py::read_results` reads **results.jsonl**, not metrics.json, without checking status. All seven failed RUL directories contain ten result rows and a metrics file. The logger truncates results at the start of a rerun, writes stage results incrementally, and writes metrics only on successful exit.

Therefore failed directories can contain **new partial results alongside stale metrics**. The existing aggregate can include partial results from failed runs. It is not accurate to infer that the table necessarily reads stale metrics merely because those files exist.

Use immutable attempt IDs and stage-specific success states. For headline tables, require verified completed stages; identify partial results explicitly. Do not erase useful completed full-input stages solely because a later missing-data comparison failed, and do not silently report the run as complete.

The repeated explanation method rows were checked: localization and deletion are intentionally stored in separate rows with different metrics. These are not, by themselves, duplicate measurements.

### F. Alpha is not simply ignored by the conformal calibrator

`ConformalPredictive.calibrate` passes `self.alpha` to `conformal_quantile`. A controlled 100-point example gave correction **35.96** at alpha=.10 and **30.91** at alpha=.20, with different widths.

The recorded FD001 seed-0 run has `diag_q_hat = 2.5` at both levels. The identical outputs are consistent with a quantile plateau arising from binning/capping; they are not evidence of missing alpha plumbing. Its raw edge interval and conformal interval both cover **0.9660**, with width **72.416**. The raw center interval covers **0.3952**. In this example, conformal adds exactly half a bin on each side and reproduces the raw edge result.

There is a separate actual reporting defect: `_eval_survival(pc, task, alpha)` always reads `q05/q95`, while passing the requested alpha to the interval-score formula. The raw alpha=.20 row therefore evaluates the same 90% endpoints with a different penalty. Make nominal interval levels explicit. Fixed base endpoints are permissible in additive split conformal because the correction depends on alpha; they are not necessarily efficient.

### G. Coverage guarantees require more than a split by engine

The pipeline calibrator splits engines but pools **all observed calibration windows** into the ordinary split-conformal quantile. In the above example, `diag_n_cal = 1985` counts windows, not independent engines. Splitting prevents direct train/test engine leakage; it does not make overlapping calibration windows exchangeable with a new test window. Preprocessing is also fitted during task construction before the fit/calibration split.

The separate `pc_conformal.py` has better unit reductions, but that does not retroactively validate the older pipeline. Specify whether coverage targets a random window from a random engine, the official endpoint, or all windows of a new trajectory, then use matching calibration objects and a justified procedure.

`conformal_quantile` also clips an out-of-range order statistic to the largest observed score. For n=1 and alpha=.05 it returns a finite score; generic distribution-free 95% coverage needs the infinite-threshold/full-set convention. Empty calibration returning zero is likewise not a valid generic guarantee.

### H. Two smaller metric/protocol defects matter for publication

* **“Last window” can miss the actual final cycle.** `windowize` uses a stride grid and `make_rul_task_split` selects its last element. For T=24, window=20, stride=3, the returned endpoints are 19 and 22, while the true last index is 23. The task remains internally labeled consistently, but it is not precisely the official last-cycle benchmark. Explicitly include the final endpoint.
* **Average precision mishandles tied scores.** The implementation processes individual tied examples rather than grouping thresholds. Four identical scores with two positives give AP=1.0 or .4167 depending on ordering; the standard threshold-grouped AP is .5. This is especially relevant for degenerate/quantized scores. Existing AUROC is separately tie-corrected.

## 3. What the experimental numbers support

The cleanest structure comparison is `logs/ts/cmapss_structure_x_leaves`: 36/36 runs have successful statuses. Reported means ± sample SD over three seeds:

| FD001 configuration | Detection AUROC | Parameters |
|---|---:|---:|
| Chow–Liu, one leaf component | .8539 ± .0048 | 157,440 |
| Random, one leaf component | .8388 ± .0035 | 157,440 |
| Chain, one leaf component | .7528 ± .0081 | 15,360 |
| Chain, three leaf components | .8265 ± .0068 | 32,160 |
| Permuted-feature chain, three leaf components | .8347 ± .0048 | 32,160 |
| Chow–Liu, three leaf components | .8266 ± .0038 | 174,240 |

This establishes a large **structure × leaf-setting interaction**. It does not isolate why. `window_leaf` changes leaf class and initialization when the component count changes; the crossed sweep does not enable the available `mixture_at_1` control. Capacity, initialization, and optimization remain entangled.

The chain is economical in the three-component comparison. It does not match the best observed Chow–Liu configuration. “Same AUROC with one-fifth the parameters” is a conditional result at that leaf setting, not a demonstrated optimal efficiency frontier. Fixed epochs do not mean matched parameters, convergence, or wall time.

The large-arm regression with extra leaf components could reflect learning-rate instability, initialization, insufficient optimization, or generalization effects. More epochs alone are not a diagnosis. One Chow–Liu three-component seed has late training losses oscillating roughly between 67 and 81. Validate lower learning rates and checkpoint selection before committing to long reruns.

The current headline AD configuration is weak: recorded FD001 chain AUROC is **.7532**, versus **.8438** for 1-NN; FD004 chain is **.7992**, versus **.8817** for GMM. These are outcomes of the configurations actually run, not upper bounds on circuits. The old universal “parity on detection” summary cannot represent them.

On the explanation benchmark, FD001 Gaussian conditional localization is **.8807**, PC conditional **.8625**, and PC marginal **.8630**. On desynchronization, Gaussian conditional is **.7217** versus PC conditional **.5491**. The strongest advertised relational diagnosis is not established there. FD002 and FD004 show other patterns; conditional scores still do not dominate their own marginal scores overall.

The Gaussian baseline itself should be improved: `GaussianConditional.attribute` sums scalar coordinate conditional quadratic terms. It does not compute the same joint channel-block conditional density used by the PC and omits channel-dependent normalization constants. Add a matched block-Gaussian and a GMM conditional baseline before attributing differences to model expressiveness.

Current RUL still loses on useful prediction even after correcting the collapse diagnosis. In the failed-run example above, full-input last-window RMSE is **35.46** with censoring, **26.65** dropping censored rows, **17.89** ridge, and **17.10** MLP. These are identifiable partial-run outcomes, not a clean aggregate. Censoring helps all-window RMSE in that same run, demonstrating endpoint dependence rather than a universal benefit or failure.

## 4. Novelty: what is already known and what might be yours

### Not defensible as a new contribution by itself

* **PC-based anomaly detection or encoder + PC:** already studied, including autoencoder representations followed by RAT-SPNs in [Dietrichstein et al., 2022](https://arxiv.org/abs/2210.06188).
* **One trained circuit used for detection and explanations through tractable marginals:** directly overlaps [Lüdtke, Bartelt and Stuckenschmidt, 2022, Outlier Explanation via Sum-Product Networks](https://arxiv.org/abs/2207.08414). Their method searches anomalous feature subsets using marginal likelihoods. Your channelwise relational analysis differs, but the broad inference-based explanation story is prior art.
* **Exact conditional probabilities or missing-variable marginals:** Gaussian models, Gaussian mixtures, and other tractable models also provide these. The research question is which model gives a useful accuracy/query-cost tradeoff.
* **Conformal survival or adaptive e-value coverage:** these have existing methods and theory; see [Conformalized Survival Analysis](https://arxiv.org/abs/2103.09763), [E-Values Expand the Scope of Conformal Prediction](https://arxiv.org/abs/2503.13050), and [Adaptive Coverage Policies in Conformal Prediction](https://arxiv.org/abs/2510.04318). Combining their wrapper with a PC is not enough without a measurable PC-specific advantage.
* **Causal or root-cause explanation:** conditional surprise is associational. A faulty channel can make an innocent correlated channel look conditionally surprising. Injection recovery tests do not establish causality.

### The most interesting surviving question

**Can a compact tractable joint model retain useful relational fault localization when the available sensor set changes, at lower query cost than equally capable alternatives?**

The key diagnostic quantity is

\[
R_c(x)=\log p(x_c)+\log p(x_{-c})-\log p(x)
=-\log\frac{p(x_c,x_{-c})}{p(x_c)p(x_{-c})}.
\]

This is negative pointwise mutual information between the channel block and the rest. It distinguishes marginal plausibility from joint incompatibility under the model. It is invariant to invertible transformations performed separately within each channel, because Jacobian terms cancel. That makes it more suitable for comparing relationships than raw density magnitudes.

But it is not a new information identity, is not necessarily nonnegative, is not an additive allocation of total anomaly score, and does not identify the causal origin of a fault. A one-channel corruption can produce a large relational score in a dependent model. Negative marginal NLL does not mean “more likely than average”; densities can exceed one and depend on units.

The possible contribution is therefore the **validated diagnostic method and operating regime**: realistic relational corruptions, changing missingness, justified alarm calibration, and an advantage over exact Gaussian/GMM and existing SPN explanation methods. No current result proves this complete combination. It is nevertheless concrete enough to test with the existing platform.

### Other branches

* **Multimodal structure transfer:** an open hypothesis, not demonstrated by the time-series results. Equal feature counts do not establish aligned channel semantics. The latent pipeline's “zero_shot” featurizer is fitted on the new dataset's normal training split (`experiment.py::_featurizer`), so this is frozen-PC adaptation with target data, not target-data-free zero-shot transfer. Require target-only, random-structure, unrelated-source, and feature-permutation controls.
* **Curvature vtrees and SOS:** retain as available methods, but park further investment for this domain until a matched-budget positive result appears. Existing negatives do not establish universal impossibility.
* **Routing:** the auxiliary routing query is computed from expert likelihoods. It does not demonstrate extra information beyond those likelihoods. Any claimed benefit must beat simple responsibility/entropy summaries and equally flexible heads on the expert-score vector.

## 5. Why the e-value extension is not yet a rescue

Three implementation/theory distinctions are particularly important:

1. `ConformalSurvivalBound.calibrate` uses the deterministic inequality `L_raw(x) − tau <= L_raw(x) − censor_bound`. It **does not call the exact box-survival query**. Any fixed raw lower-bound predictor can use that conservative score bound. The docstring's claim that the calibration step uniquely requires PC box inference is incorrect. Its validity needs an appropriate exchangeability/sampling argument; the upper-bound inequality itself does not require censoring independent of covariates.
2. Arithmetic e-value merging needs component e-values valid under a **common specified null** and prespecified valid weights. Sharing a PC is neither necessary nor sufficient. A predictive e-value at the true future RUL is not immediately an observable current health alarm, because future RUL is unknown. Define the operational null and available information first.
3. Removing window overlap does **not** imply exchangeability: disjoint windows can remain autocorrelated and drift with operating conditions. `assert_nonoverlapping` catches one hazard, not all assumptions required by the conformal martingale. The literature has dedicated extensions for dependent processes, e.g. [Nettasinghe et al., 2023](https://proceedings.mlr.press/v202/nettasinghe23a.html).

For adaptive alpha, report the guarantee actually proved. [Proposition 2.2](https://arxiv.org/html/2510.04318v2) controls the expectation of miscoverage divided by the selected alpha. That is not generally identical to marginal coverage ≥ 1 − E[alpha]. The `nominal = 1 - mean(alpha)` diagnostic alone is not a validity test; record the corresponding normalized miscoverage statistic and uncertainty.

There is also a power issue: for conservative rank p-values p>=1/(n+1), the power calibrator gives e<=kappa*(n+1)^(1-kappa). At n=50 and kappa=.5, the maximum is about 3.57, below the 20 needed for a fixed 5% alarm. The generic p-value resolution floor understates this e-alarm limitation. Count calibration engines and assess attainable alarm thresholds before promising operational false-alarm control.

## 6. Data and evaluation limits

The documentation calls C-MAPSS backgrounds and degradation “real.” They are **NASA simulator-generated benchmark data**, not measured fleet failures; NASA explicitly describes the dataset as [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). This is a valuable independent simulator, but does not demonstrate field reliability. The anomaly corruptions and channel labels are additionally supplied by this repository. Only C-MAPSS and ADBench directories were present locally; loader availability is not evaluation evidence.

Deletion is scored through the PC and replaces channels with training means. It is a model-specific response test with potentially implausible replacements, not independent proof of fault removal. Use retained clean pre-corruption windows for controlled repair where possible, plus model-specific fidelity measures labeled as such.

Pooled channel-localization AUROC also mixes between-window severity and within-window channel ranking. Add macro per-window ranking, precision/recall at a fixed inspection budget, and intervals obtained by resampling engines rather than overlapping windows. Three seed SDs are not an equivalence or significance test. A scrambled control winning narrowly does not prove all structure is useless; a random control being second does not erase a positive matched-parameter Chow–Liu comparison.

AD training uses healthy data, whereas RUL training uses degradation trajectories. The existence of `SurvivalPC.anomaly_score` does not show that one fitted model performs both tasks well: a model trained to assign density to late-life degradation need not treat it as anomalous. Evaluate the proposed shared model explicitly, and define whether “anomaly” means degradation or sensor corruption.

## 7. Bounded continuation plan and stop rules

**Recommended budget: one focused validation cycle of roughly two working weeks, not another broad feature expansion.** These are proposed decision criteria, not thresholds selected from new results.

| Priority | Work | Decision it enables |
|---|---|---|
| 1 | Repair result provenance, actual held-out validation, endpoint selection, tied AP, and explanation labels; define calibration unit and nominal levels | Whether any headline is reviewable |
| 2 | Compare GaussianLeaf(1), GMMLeaf(1), GMMLeaf(3), with chain/Chow–Liu/random; controlled initialization, a small learning-rate sweep, validation checkpoint selection | Whether the large gain comes from capacity, initialization, or optimization |
| 3 | Evaluate matched block-Gaussian/GMM and an existing SPN explanation baseline; genuine permutation SHAP for an explicitly shared game where appropriate | Whether the explanation gain survives meaningful adversaries |
| 4 | Cross relational anomaly kind/severity with sensor masks; preserve clean windows; finish exact-marginal versus imputation comparisons even when an arm degenerates | Whether tractability provides a useful diagnostic advantage |
| 5 | Repeat the chosen protocol on a second independent source, preferably measured industrial/sensor data with defensible fault annotations | Whether the effect extends beyond this simulator and injection scheme |

Freeze choices using validation data before evaluating the final test. Preselect a practical localization improvement, an acceptable detection loss, and a query-latency/memory budget. Estimate paired differences with engine-level uncertainty. Report performance versus actual resource use rather than equating K across structures.

**Continue the diagnosis paper** if relational localization and missing-sensor robustness improve consistently over strong tractable alternatives at a practical cost, with a distinct contribution beyond the 2022 SPN explanation method. A modest detection disadvantage can be acceptable if the diagnostic benefit is large and measured.

**Stop the broad method claim** if Gaussian/GMM achieves the same diagnosis and missing-data performance at equal or lower cost, or if advantages survive only on one injection generator or selected seed. The result may still be a useful software/benchmark/negative-results contribution; it would not support the original universal detector thesis.

**Keep RUL secondary.** Give it one controlled follow-up after resolving initialization and missing-input failure behavior. Compare a target-only predictor, ridge/MLP, and a censor-aware predictive-distribution baseline under the same official endpoint protocol. Require benefit from censored information on a proper distributional score, alongside nondegenerate predictions and meaningful interval width. Do not keep extending this branch solely because its query mathematics is elegant.

**Defer e-values, routing, cross-modal transfer, curvature, and SOS expansion** until the central diagnostic effect survives. Each adds a distinct scientific claim and another set of assumptions; none repairs an uninformative conditional model.

The project is worth a tightly scoped continuation because the machinery can answer a useful question cheaply enough to test now. It is not yet a demonstrated novel method ready for a strong paper. The next milestone should be one credible diagnostic advantage, established with clean provenance and strong baselines.
