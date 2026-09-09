# Strengthening probabilistic circuits for relational anomaly diagnosis

## Executive assessment

The most promising research question is **whether a tractable joint model can identify observable violations of sensor relationships under changing sensor availability, with a better diagnostic accuracy–latency trade-off than equally capable alternatives**. The current implementation provides useful machinery for investigating this question. It does not yet establish the advantage.

The September 8 feasibility gate remains a failure under its original rule. However, the interpretation that channel blocking inherently makes the model unsuitable for diagnosis is stronger than the evidence supports. The gate evaluates joint negative log-likelihood as an anomaly score on a mixture of injected anomaly types. It does not evaluate the proposed relational statistic as its primary endpoint. The reported dependence diagnostic measures departure from the model's own product of marginals on validation observations; it does not measure how accurately the model captures real dependence.

The literature changes the priority order. Classical differential inference already provides the foundation for two-pass queries. Existing work also covers PC-based outlier subset selection, copula-based localization with missing values, and calibration conditional on missingness patterns. Optimization failures in PCs are sufficiently well documented that architectural impossibility should not be inferred from a short training run. Recent work introduces both improved parameter learning and distribution-preserving restructuring, providing concrete alternatives to abandoning the circuit design.[^1][^2][^3][^4][^5][^6][^7]

**Recommendation:** first correct the scientific measurement and remaining protocol mismatches; then compare the existing structures using relational scores, converged training, and a deliberately strong mixture baseline. Only then invest in a structure change. The preferred redesign is a channel-aligned circuit with independently controlled capacity within channels and across channels, trained and selected for the conditional queries it will actually answer. A more ambitious second route is to learn an expressive teacher and restructure or distill it into a query-efficient student.

This report covers literature available through September 9, 2026. Published results, repository observations, mathematical deductions, and proposed experiments are distinguished below. Proposed improvements have not been run. The literature coverage is targeted and broad enough to inform the design; it is not an exhaustive proof of novelty.

## 1. What the latest experiment establishes

The authoritative artifact is `logs/ts/tier1_kill_gate/gate_verdict.json`, alongside its summary and per-seed training histories. Three completed seed batches evaluate four structures with 174,240 parameters each. The base data are C-MAPSS FD001; anomalies used for this gate are artificially injected into validation windows. These are experiments on simulated engine measurements with additional synthetic corruption, not observed sensor-failure diagnosis in a deployed system. NASA describes the underlying benchmark as simulated degradation trajectories.[^8]

| Structure | AUROC, mean ± SD | Validation NLL, mean | Reported dependence diagnostic |
|---|---:|---:|---:|
| Chow–Liu | 0.8268 ± 0.0272 | 58.34 | 34.22 |
| Random | 0.7994 ± 0.0088 | 70.06 | 23.93 |
| Channel | 0.7937 ± 0.0199 | 40.52 | 6.03 |
| Channel-blocked | 0.7934 ± 0.0268 | 45.29 | 5.60 |

The blocked candidate loses 0.0334 AUROC against Chow–Liu, exceeding the original 0.02 budget. Preserve that verdict. Its lower NLL is a genuine advantage on the reported validation observations, but a percentage improvement in continuous NLL should not be interpreted as a scale-independent measure of density quality.

The per-type results contain more diagnostic information than the aggregate. Chow–Liu versus channel-blocked obtains 0.7126 versus 0.6126 AUROC on donor-channel replacement (`desync`), but only 0.5286 versus 0.5020 on time permutation (`decouple`). Both are strong on drift and offset. These results indicate an unresolved difficulty with particular forms of temporal and cross-channel corruption. They do not establish localization performance because the gate evaluates detection, not identification of affected channels.

### 1.1 The dependence statistic needs a different interpretation

`pipeline.py::channel_dependence` computes

\[
D_{\mathrm{abs}}=\frac1n\sum_i\left|\sum_c\log p_\theta(x_{ic})-\log p_\theta(x_i)\right|.
\]

This is an empirical absolute departure from factorization. Total correlation instead has the form

\[
\mathrm{TC}(p)=\mathbb E_{X\sim p}\left[\log p(X)-\sum_c\log p(X_c)\right].
\]

The current statistic differs both through the absolute value and through evaluation on data rather than samples from the model. A badly misspecified model can produce a large discrepancy. Therefore “six times the diagnostic” does not mean “six times the useful dependence.” Near-zero values on a finite sample are a useful collapse warning; they are not a global proof of exact factorization.

Record the signed statistic, its distribution, and joint-versus-independent detection performance. On synthetic data with a known generating distribution, additionally measure conditional log-loss and recovery of known dependence. Do not train by maximizing the absolute discrepancy: that could reward confidently wrong relationships.

### 1.2 The experiment leaves optimization unresolved

The Chow–Liu validation histories reach their best checkpoint at the final epoch in seeds 0 and 1. Channel-blocked reaches its best near the end in seeds 0 and 1. This does not prove undertraining, but it prevents a claim that all architectures have been compared at convergence.

Latent variable distillation was motivated by PC performance plateaus that increasing parameter count alone did not solve. More recently, Anemone addresses shortcomings of minibatch PC optimization through parameter-specific regularization of distribution changes. Neither result guarantees improvement here, but both make optimization a serious competing explanation.[^5][^6]

### 1.3 Three remaining implementation/protocol mismatches

**Checkpoint split.** `_val_halves` creates engine-disjoint halves intended for checkpoint/NLL evaluation and injected detection evaluation. However, `_fit_window_pc` still passes the full `task.X_val` to `pc.fit`. Thus checkpoint selection sees clean observations from both halves. This is not direct use of injected labels or the test set, but it contradicts the claimed separation. Pass the intended checkpoint half explicitly in future runs.

**Calibration quantiles.** `MaskCalibrator._quantile` clips an unavailable order statistic to the largest calibration score. For a conformal-style upper threshold with \(k=\lceil(n+1)(1-\alpha)\rceil>n\), the conservative convention is an infinite threshold. Moreover, overlapping windows cannot automatically be treated as independent calibration units. Correct quantile indexing does not solve the sampling problem.

**Complexity accounting.** `masked_log_prob` repeats boundary tensors over masks and evaluates an effective batch of size \(N M\). Its node-visit counter increments per traversal rather than per evaluated scalar/vector element. Consequently, flat node visits within a chunk do not imply flat arithmetic work as masks increase. Report tensor sizes, operations, memory, and synchronized latency as well as traversals.

These issues should be repaired in a new experiment version. They do not authorize retroactively changing the old pass/fail rule.

## 2. The closest literature and the remaining novelty boundary

| Research line | Established result | Implication for this project |
|---|---|---|
| Differential inference: Darwiche, UAI 2000 [^1] | Circuit evaluation and derivatives answer families of probabilistic queries. | A forward/backward pass alone is not a new inference principle. Specify the block interface and workload improvement. |
| PC framework: Choi, Vergari, Van den Broeck, 2020 [^9] | Structural properties determine tractable query classes. | Exact evaluation is relative to the fitted model and query conditions. |
| SPN outlier explanation: Lüdtke et al., 2023 [^2] | Marginal queries support outlier subset search without fitting a detector for every subset. | Compare objectives and search procedures, not just a slow oracle. |
| Marginal subset optimization: Khosravi et al., TPM 2022 [^3] | Outlier feature identification using probabilistic marginals and constrained optimization. | Removing features to restore normality is prior art; its guarantees do not automatically apply to the proposed statistic. |
| Copula localization: Horváth et al. [^4] | Dependence-based anomaly scoring, localization, and missing-value handling. | “Dependence + localization + missingness” is not sufficient novelty. |
| Missing-data AD: Dietterich and Zemicheal, 2018 [^10] | Evaluates imputation, marginalization, and other missing-data strategies, including mixtures. | Marginalization must be evaluated against strong alternatives, not only mean imputation. |
| Likelihood-ratio OOD: Ren et al., 2019 [^11] | Background statistics can confound likelihood; a reference distribution can improve detection. | Define the alternative corresponding to the relational score. |
| Typicality: Nalisnick et al., 2019 [^12] | High density is not synonymous with in-distribution behavior. | Better NLL and worse detection can coexist without a coding error. |
| Optimization: Liu et al., ICLR 2023; AISTATS 2026 [^5][^6] | Learning can fail to exploit available PC capacity. | Compare training mechanisms before declaring a structural limit. |
| Tensor parameterization: Mari et al., TPM 2023 [^13] | Connects circuit parameterizations and low-rank tensor decompositions. | Interface width and capacity placement matter, not just total parameters. |
| Restructuring: Zhang et al., AISTATS 2025 [^7] | Transforms structured PCs to other vtrees, with complexity depending on compatibility. | Train and query structures need not always coincide; size blow-up must be measured. |
| Efficient PC systems: EiNets and PyJuice [^14][^15] | Vectorization and optimized execution substantially change practical cost. | Benchmark against compiled, batched implementations. |
| Missingness calibration: Zaffran et al. [^16][^17] | Pattern-conditional validity requires a specified statistical setting. | Arbitrary computational masks do not imply arbitrary missingness guarantees. |
| Hierarchical inference: Lee et al. [^18] | Repeated/grouped observations need corresponding exchangeability arguments. | Engines and overlapping windows must be distinguished. |
| Benchmark audit: Pinet et al., MiLeTS 2026 [^19] | Their tests find no purely cross-channel anomaly segments in eight studied benchmarks. | Validate whether a benchmark requires the capability being claimed. |
| Hierarchical PC OOD: Bhumika K et al., August 2026 preprint [^20] | Uses internal likelihood representations for batch shift detection and node localization. | “Use internal PC information beyond root likelihood” is already occupied. |

The new HLV/HLD preprint is particularly relevant to positioning. Its main experiment is a batch-level distribution test, which differs from per-window sensor diagnosis. It also reports false-positive inflation on real held-out data as model discrepancy becomes detectable. It is a useful comparator for a hierarchical-score extension, not evidence that model-derived thresholds are automatically calibrated in the real world.[^20]

The defensible gap is narrower: **jointly learning a model and executing a family of block diagnosis queries under realistic observation patterns, with an empirically demonstrated advantage over optimized tractable competitors**. Establishing that gap requires experiments; this review does not establish priority for every possible implementation.

## 3. Give the score a precise statistical meaning

Let \(O\) denote the observed channels and \(c\in O\). The implemented statistic is

\[
R_{c,O}(x)=\log p(x_c)+\log p(x_{O\setminus c})-\log p(x_O).
\]

Define a particular alternative distribution

\[
q_{c,O}(x_O)=p(x_c)p(x_{O\setminus c}).
\]

Then \(R_{c,O}=\log(q_{c,O}/p_O)\). This is a log-likelihood ratio for a **block independence alternative**: the channel retains its marginal behavior while its relationship with the remaining observed channels is removed. The identity is elementary and should not be presented as a new statistic. Its value is that it states exactly which abnormality the method targets. Likelihood-ratio OOD research provides the broader motivation for choosing such a reference.[^11]

### 3.1 Detection, localization, and causal blame are different

The score measures evidence that a relationship has broken. It does not, by itself, identify which physical sensor caused the break. In a two-channel system,

\[
R_{1,\{1,2\}}=R_{2,\{1,2\}}.
\]

Likewise, a subset and its complement have the same block-independence score. No optimizer can remove this symmetry without additional assumptions or evidence. Observational feature explanations and interventions are distinct concepts in the explanation literature.[^21]

With several redundant sensors, a single corrupted channel may be identifiable because it disagrees with many mutually consistent channels. With only two sensors left, equal scores may be the correct result. Therefore evaluate ambiguous sets, top-k coverage, and abstention when the data cannot support unique localization. Label the task “statistical inconsistency localization” unless a fault mechanism or causal model supports stronger language.

One possible extension is an explicit model over fault hypotheses:

\[
P(F=c\mid x_O)\propto\pi_c q_{c,O}(x_O).
\]

Include a normal hypothesis and define each alternative's corruption mechanism. This yields a posterior only under that specified mixture of alternatives; normalizing arbitrary scores is not enough. A different corruption distribution \(g_c\) leads to \(g_c(x_c)p(x_{O\setminus c})\), rather than the independence alternative using \(p(x_c)\). Priors can express asymmetric fault rates, but they cannot create evidence absent from the observations.

### 3.2 Missingness can remove the distinguishing information

If a fault changes only the dependence between two channels, removing one can make the normal and abnormal observed distributions identical. Their distinction is then unidentifiable from the available observation. More generally, applying a fixed observation projection cannot increase KL divergence between normal and abnormal distributions; this is a direct information-theoretic consequence of marginalization.

Consequently, the target should be **retaining useful diagnosis when identifying evidence remains available**, with explicit degradation and abstention otherwise. “Localization survives arbitrary sensor loss” is scientifically untenable without qualification. Computational support for every mask is a separate claim.

### 3.3 The current injected anomalies test different mechanisms

`decouple` permutes timesteps within a channel. This preserves the multiset of scalar values, but generally changes the distribution of an entire channel trajectory. A competent univariate temporal model can detect it. It is not a clean test of cross-channel-only anomalies.

`desync` replaces an entire channel trajectory with a donor trajectory. It approximates a marginal-preserving alternative only if donor and recipient come from the same appropriate nominal population. Differences in regime, degradation stage, or engine population can introduce easier marginal signals. Use independent donors matched on externally defined context, and check whether a univariate temporal classifier can distinguish originals from replacements.

Pinet et al.'s benchmark audit supports the need for these checks, but its conclusion is restricted to its diagnostics and studied datasets. Their synthetic experiment also shows a strong simple linear baseline. A method should not be declared relational merely because its inputs are multivariate.[^19]

## 4. The strongest practical redesign

### 4.1 Start with a measurement-first comparison

Before new architecture work, evaluate four scores on each existing trained structure: joint NLL, independent-channel NLL, conditional surprise, and the relational ratio. Use the generic marginal oracle for unblocked structures. This answers whether the current gap comes mainly from the model, the score, or both.

Do this on an explicit joint-versus-independent synthetic task and a donor-replacement task with context controls. Keep aggregate detection as a secondary result, and measure localization separately. Use the old seeds for exploration only; a subsequent confirmation needs fresh held-out units or an untouched dataset, not merely new initializations on the same repeatedly inspected data.

### 4.2 Separate within-channel and cross-channel capacity

The current common \(K\) and equal total parameter counts do not guarantee equal capacity at the boundaries that carry sensor relationships. Introduce separate budgets for temporal subcircuits, channel boundary units, and upper mixing regions. Record interface widths, effective component occupancy, and parameter allocation by level.

The tensor interpretation of PCs motivates this design, but it is a proposed adaptation, not a result already demonstrated for this project.[^13] A useful elementary bound explains the mechanism: if a cut admits

\[
p(A,B)=\sum_{z=1}^{r}\pi_z p_z(A)p_z(B),
\]

then \(I_p(A;B)\le H(Z)\le\log r\). This holds under the fitted model with the stated nonnegative mixture representation. It does not bound the repository's empirical absolute total-dependence diagnostic, and the relevant \(r\) must be established from the actual circuit boundary rather than assumed equal to every configuration's \(K\).

Compare uniform capacity against more capacity at channel interfaces under both a total-parameter budget and an inference-latency budget. If increasing interface capacity improves held-out relational discrimination while preserving query efficiency, there is a mechanistic result worth developing. If all capacity placements behave similarly, the proposed bottleneck explanation is weakened.

### 4.3 Improve optimization without changing the inference contract

Run a bounded optimizer comparison: the current method, an EM implementation appropriate to the leaves, and—if integration is feasible—Anemone. Include independent component initialization or a modest cluster-based warm start. Report updates, wall-clock budget, and convergence curves rather than only epochs.[^6][^14]

Anemone's published gains are not on this sensor task, and its main parameter-learning argument does not automatically solve Gaussian-mixture leaf training. Integration must be validated against normalized likelihood and exact queries. Avoid presenting the adoption of a published optimizer as the new contribution.

### 4.4 Train for the queries being evaluated

A reasonable candidate is joint likelihood plus a masked conditional objective:

\[
\mathcal L(\theta)=\mathbb E[-\log p_\theta(X)]
+\lambda\mathbb E_{O,c\in O}[-\log p_\theta(X_c\mid X_{O\setminus c})].
\]

All terms come from one normalized joint model. The first term anchors global fit; the second emphasizes conditional predictions under the intended observation patterns. Conditional likelihood learning of SPNs is established, so novelty would need to reside in the design, analysis, and demonstrated workload advantage rather than the mere use of this objective.[^22]

The mask distribution is part of the method and must be specified. Normalize or weight terms appropriately when channel dimensions differ. Keep independent engine splits for model selection and calibration. Do not optimize the relational score on normal observations simply to make it large in magnitude.

A secondary option is a contrastive objective distinguishing nominal joint samples from controlled independent-block replacements. Noise-contrastive estimation supplies relevant precedent, but a loss using generated negatives is not automatically classical NCE with all its consistency assumptions.[^23] If synthetic fault examples are used for training or selection, describe the method as self-supervised or corruption-supervised as appropriate. Hold out entire corruption mechanisms to test whether the method learned relationships rather than the injector.

### 4.5 Channelwise invertible temporal transforms

An optional lower-cost improvement is an invertible transformation within each channel, such as a temporal differencing transform retaining the initial value, whitening, or a learned blockwise bijection. For \(z_c=T_c(x_c)\), the Jacobian terms cancel in \(R_{c,O}\) when the same transformed joint distribution defines all marginals. Whole-channel missingness remains straightforward because transformation does not mix observed and missing channels.

This is a derived invariance property, not a new flow principle. Sum-Product-Transform Networks provide relevant prior art.[^24] Avoid cross-channel transforms unless the missing-data queries remain tractable. A noninvertible encoder does not preserve the claim of exact raw-input density, and within-channel partial missingness needs separate treatment.

## 5. A more ambitious route: learn first, compile for diagnosis

Zhang et al. explicitly study restructuring a PC to a target vtree. This suggests learning an expressive unblocked teacher and converting it to a channel-aligned circuit when affordable. The conversion can incur exponential growth in general; the paper does not justify a universal cheap conversion.[^7]

First measure the relevant separator sizes on the existing Chow–Liu structure. Attempt exact conversion only on small cases where memory and runtime can be bounded. A distribution-preserving conversion would isolate the computational question cleanly: detection remains unchanged, and only query cost and representation size change.

If exact conversion is too large, distill a query-aligned student. Latent variable distillation is established PC methodology.[^5] The distinctive objective here would emphasize errors in the joint, channel, and complement log-probabilities across the intended masks, rather than global NLL alone.

For a teacher \(p\) and student \(s\), if the absolute log-probability error is at most \(\epsilon\) for each of the three terms of a particular relational query, then

\[
|R^p_{c,O}(x)-R^s_{c,O}(x)|\le3\epsilon.
\]

A teacher ranking margin greater than \(6\epsilon\) is therefore preserved between two channels satisfying these bounds. This is an elementary error-propagation lemma, not enough by itself for a theory contribution. Its practical value is specifying what distillation must preserve. Average empirical errors do not imply uniform bounds, and a student preserving an incorrect teacher is still incorrect.

This route has a potentially stronger scientific structure: an explicit trade-off among interface complexity, query distortion, and diagnostic stability. It also has greater implementation risk. Pursue it after the simpler measurement and capacity experiments demonstrate that there is useful relational information to preserve.

## 6. Baselines that can actually invalidate the claim

### 6.1 A mixture of channel models is indispensable

Consider

\[
p(x_O)=\sum_{k=1}^{K}\pi_k\prod_{c\in O}p_{kc}(x_c).
\]

Each component's channel model can represent a temporal trajectory. Missing channels are omitted. After computing \(\ell_{kc}=\log p_{kc}(x_c)\), form \(a_k=\log\pi_k+\sum_{c\in O}\ell_{kc}\). All leave-one-channel-out marginals are obtained by log-sum-exp of \(a_k-\ell_{kc}\), using stable exclusion sums if zero densities occur. Channel marginals use the same components with the prior mixture weights.

Thus all relational scores cost \(O(KC)\) after the channel likelihoods, for one observation mask. This is an algebraic baseline derived here; it is a shallow tractable model within the PC family. It directly challenges the assumption that fast all-channel queries are a unique benefit of the deep blocked design.

Use block-diagonal Gaussian mixtures or mixtures of tractable temporal channel models, with sufficient components selected on validation. Failure to beat this baseline in diagnosis per unit cost means the deep structure is unnecessary for the stated task. It would still leave a useful inference implementation, but a much weaker method claim.

### 6.2 A Gaussian does not need a fresh factorization for every channel

For a fixed observed set \(O\), compute the observed covariance \(\Sigma_{OO}\) and precision \(\Lambda^O=(\Sigma_{OO})^{-1}\). Each observed channel block has conditional covariance \((\Lambda^O_{cc})^{-1}\) and conditional mean

\[
\mu_c-(\Lambda^O_{cc})^{-1}\Lambda^O_{c,-c}(x_{-c}-\mu_{-c}).
\]

These are standard Gaussian conditioning identities.[^25] All channel conditionals share the observed precision calculation. For recurring masks, cache it. For related masks, investigate factorization updates. For a new mask, use the inverse of the observed covariance, not a principal block of the full precision, which would generally be the wrong marginal model.

A full-covariance GMM repeats the required setup per component and can also cache by mask. A block-diagonal mixture avoids cross-channel matrix factorization entirely. Report cold-mask and warm-mask timing, not a blanket assertion that every Gaussian query requires a new expensive solve.

### 6.3 The complete minimum comparison

| Baseline | Main purpose |
|---|---|
| Independent temporal channel models | Detect whether the corruption has a univariate shortcut. |
| Shrinkage Gaussian and low-rank Gaussian | Strong simple dependence models with exact marginal queries. |
| Block-diagonal and full-covariance GMMs | Test multimodality and cheap latent mixture inference. |
| Unblocked PC with batched marginals | Separate architecture quality from query execution. |
| Existing SPN subset explanations | Compare against actual prior methods, including their scoring/selection rules. |
| Copula-based model | Test whether flexible marginals plus dependence modeling suffice. |
| Linear temporal reconstruction baseline | Check whether the claimed relationship is easy to learn without a circuit. |

A reference oracle that reruns the same circuit establishes correctness and an implementation speedup. It is not a reproduction of Lüdtke et al.'s entire method. Their paper uses its own subset scoring and search procedures.[^2] Likewise, their results or Khosravi et al.'s results do not establish submodularity or an approximation guarantee for the signed \(R_S\) objective.[^3]

## 7. State computational claims accurately

Let \(L\) be the cost of evaluating all channel subcircuits for one window, \(U\) the cost of the upper circuit, and \(B\) the total boundary width. A full relational map under one mask has approximate work

\[
O(L+U+B+C).
\]

The advantage is avoiding roughly \(C\) repeated full evaluations. It is not independence from the number of channels: both circuit size and the output grow with \(C\). The block-boundary calculation is an application of differential circuit inference.[^1]

For \(M\) masks on the same window, cached lower values give approximately

\[
O(L+M(U+B+C))
\]

for full maps, with output size \(MC\). A different mask on each of \(N\) rows can share one batched program without grouping by mask; that statement does not eliminate work per row. Arbitrary subsets also require additional upper evaluations, even when lower computations are cached.

The reported FD001 speedup is valuable preliminary engineering evidence. It must be accompanied by small-circuit results, preprocessing and cache construction cost, memory, CPU/GPU synchronization, and the same precision/batching policy for competitors. EiNets and PyJuice make clear why implementation quality can dominate a wall-clock comparison.[^14][^15]

## 8. Calibration and missingness are part of the task definition

Pattern-conditioned calibration is established prior work. Zaffran et al. distinguish average coverage from coverage conditional on missingness patterns, and their later treatment provides impossibility results and useful assumptions for informative guarantees.[^16][^17] These are prediction-set results; adapting their ideas to an anomaly alarm requires stating the null distribution and score explicitly.

For this project, begin with masks fixed externally or sampled independently of the current observation, and independent held-out nominal engines. Freeze the score before calibration. Select a sampling target: one predetermined/random window per new engine, an endpoint, or a maximum over an entire trajectory. Use an engine-level calibration score appropriate to that target; hierarchical conformal work provides relevant alternatives when retaining multiple observations.[^18]

Calibrating each channel at level \(\alpha\) does not control the probability that any channel alarms at \(\alpha\). Calibrate the maximum over observed channels for a familywise window alarm if that is the operational endpoint. Repeated online alarms need an additional temporal definition. Report marginal FPR, per-mask FPR, and familywise FPR separately.

If dropout depends on the unseen sensor value or the fault itself, the correct observed-data model can involve \(P(M\mid X)\). Merely computing \(p(x_O)\) ignores informative missingness. Include this as a stress test with empirical results; do not promise distribution-free validity under arbitrary fault-driven dropout.

The current named-mask calibration dictionary also does not provide a threshold for an unseen mask. Computational ability to score that mask is insufficient. Options include a prespecified finite mask library, fresh on-demand calibration under independent masking, or a justified mask-generalization method. These alternatives have different statistical and computational costs.

## 9. Experimental roadmap and decision rules

The following sequence is a proposal for a new study, not a replacement interpretation of the original failed gate. Freeze endpoint definitions and budgets before the confirmatory runs.

| Stage | Work | Decision supported |
|---|---|---|
| A: measurement repair | Fix checkpoint-half plumbing, quantile boundary behavior, and arithmetic-aware cost accounting. Rename dependence diagnostic. | Can the next results be interpreted as stated? |
| B: score diagnosis | Existing structures × joint, independent, conditional, relational scores; use matched original/replacement pairs. | Is the failure mainly score choice or fitted relationships? |
| C: optimization and interface capacity | Bounded training comparison; uniform versus upper/interface capacity; mixture baseline. | Does a deeper blocked model provide useful extra capacity? |
| D: observation stress | Prespecified random, grouped, informative-witness, and unseen masks. Include ambiguous/impossible cases. | When does useful diagnosis remain identifiable? |
| E: confirmation | Freeze design; independent engines and another data source; optimized strong baselines. | Is there a reproducible method advantage? |

Use three complementary data families. First, synthetic multichannel processes with known dependence and an oracle likelihood ratio establish whether the proposed statistic targets the right mechanism. Second, semi-synthetic corruption of nominal C-MAPSS or another sensor source tests robustness on realistic backgrounds. Third, independently labeled fault data test external relevance; document whether labels identify the initiating fault, affected sensors, or only abnormal time segments. More subsets of C-MAPSS are useful replication, but not an independent application domain.

The primary diagnosis endpoint should be localization quality among observed and identifiable targets, such as average precision over affected channels or top-k set coverage. Also report end-to-end diagnosis success, counting missed detection as failure, to avoid evaluation only on easy detected cases. Report detection power at a calibrated FPR, subtype AUROC, and the false-positive cost of abstention or ambiguous sets.

Compute paired differences using the same corruptions and masks for all models. Resample engines or independent trajectories, not overlapping windows; keep training-seed variation separately visible. Three seed standard deviations are not confidence intervals for a population effect. Repeatedly trying seeds on the same held-out engines does not create a fresh test population.

For continuous NLL, replace the relative-percentage gate with paired NLL differences per observed scalar on identical observations and preprocessing, or held-out log-likelihood ratios against a fixed reference. Under a change of measurement units, each density's NLL acquires the same Jacobian offset, so their difference cancels while a percentage generally changes. This is an elementary mathematical correction, not a literature-dependent empirical claim.

Choose the substantive acceptance margins before confirmation. A useful design is: diagnostic noninferiority within a prespecified practical tolerance, a minimum end-to-end speedup under the stated mask workload, and acceptable calibrated false alarms. Alternatively require materially better diagnosis at matched latency. The tolerances should follow the intended application and pilot variability; a universal numerical threshold would be arbitrary here.

Stop the architectural expansion if a shallow mixture or Gaussian matches the diagnostic frontier at lower cost, if apparent gains disappear after marginal-preservation controls, or if the gains occur only for training-time corruption mechanisms. If relational scores work but blocking cannot preserve them cheaply, consider the teacher/restructuring route. If even an oracle has no information under a mask, change the claim rather than tuning the learner.

## 10. Scientific claims to write, test, or retire

| Claim | Status and appropriate wording |
|---|---|
| Exact raw-input density and selected marginals | Retain only for the validated normalized raw-input circuit and transformations preserving the relevant query tractability. |
| Two-pass all-channel computation | Retain as an implemented specialization of differential inference; specify assumptions and actual complexity. |
| Novel relational statistic | Retire. It is a likelihood ratio against an independence alternative. |
| Root-cause identification | Replace with inconsistency localization unless an identifiable fault model supports causal language. |
| Constant cost in channels and masks | Retire. State cached lower evaluation and per-mask upper work. |
| Sixfold better dependence in Chow–Liu | Retire. Report the actual empirical diagnostic without an accuracy interpretation. |
| Blocking cannot model useful dependence | Unresolved. Existing results reject one training/configuration/score combination under its gate. |
| Robustness to arbitrary missing sensors | Split into computational support, empirical performance, and statistical guarantees under stated assumptions. |
| Superior explanation to SHAP | Unsupported by the current replacement-sensitivity comparison. |
| Universal anomaly detection and superior RUL | Remain outside the supported central claim. |

A candidate paper claim, conditional on successful experiments, is:

> We learn channel-aligned tractable models for block-independence diagnosis under changing observation sets. By allocating capacity to sensor interfaces and optimizing the required conditional queries, the method achieves a measured diagnosis–latency advantage over optimized Gaussian, mixture, and circuit baselines, while exposing cases where missing information prevents unique localization.

A stronger algorithmic version would show that query-aware restructuring or distillation preserves diagnosis with bounded distortion and favorable resource costs. A successful optimizer swap, a faster Python implementation, or one extra AUROC table would not establish that larger contribution. Sum-of-squares circuits remain a legitimate expressive extension, but their known expressiveness results do not predict success on this task or preserve the current block algorithm without additional work; defer that branch until the simpler bottleneck is identified.[^26]

## Sources

The numbered notes provide the complete source inventory. Most sources are original papers or official proceedings; workshop and preprint status is identified where material. Repository evidence is listed separately because it is not an external research result.

[^1]: Adnan Darwiche. [A Differential Approach to Inference in Bayesian Networks](https://arxiv.org/abs/1301.3847). UAI 2000; arXiv archival upload 2013. Foundation for evaluation/derivative-based query reuse.
[^2]: Stefan Lüdtke, Christian Bartelt, Heiner Stuckenschmidt. [Outlier Explanation via Sum-Product Networks](https://arxiv.org/pdf/2207.08414). AAAI 2023; preprint 2022. Full text, particularly subset objectives and search algorithms.
[^3]: Pasha Khosravi, Antonio Vergari, Guy Van den Broeck. [Why Is This an Outlier? Explaining Outliers by Submodular Optimization of Marginal Distributions](https://khosravipasha.github.io/papers/tpm22_why_is_this_an_outlier_explain.pdf). TPM workshop 2022. Author-hosted paper; indexed full-text sections were accessible, while direct PDF retrieval was unreliable. Used for the problem formulation and prior-art boundary, not to import its guarantees.
[^4]: Gábor Horváth, Edith Kovács, Roland Molontay, Szabolcs Nováczki. [Copula-based anomaly scoring and localization for large-scale, high-dimensional continuous data](https://arxiv.org/abs/1912.02166). 2019 preprint; record identifies acceptance at ACM TIST. Dependence, localization, and missing-data precedent.
[^5]: Anji Liu, Honghua Zhang, Guy Van den Broeck. [Scaling Up Probabilistic Circuits by Latent Variable Distillation](https://arxiv.org/abs/2210.04398). ICLR 2023; preprint 2022. Learning bottlenecks and latent supervision.
[^6]: Anji Liu, Zilei Shao, Guy Van den Broeck. [Rethinking Probabilistic Circuit Parameter Learning](https://proceedings.mlr.press/v300/liu26c.html). AISTATS 2026, pp. 1693–1701. [Full text](https://arxiv.org/pdf/2505.19982). Anemone and minibatch learning analysis.
[^7]: Honghua Zhang, Benjie Wang, Marcelo Arenas, Guy Van den Broeck. [Restructuring Tractable Probabilistic Circuits](https://proceedings.mlr.press/v258/zhang25f.html). AISTATS 2025, pp. 2566–2574. [Full text](https://arxiv.org/pdf/2411.12256). Target-vtree restructuring and size limitations.
[^8]: NASA. [CMAPSS Jet Engine Simulated Data](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data). Official government dataset record, accessed September 9, 2026. Dataset provenance and fault-mode descriptions.
[^9]: YooJung Choi, Antonio Vergari, Guy Van den Broeck. [Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models](https://yoojungchoi.github.io/files/ProbCirc20.pdf). 2020. Structural properties and query tractability.
[^10]: Thomas G. Dietterich, Tadesse Zemicheal. [Anomaly Detection in the Presence of Missing Values](https://arxiv.org/abs/1809.01605). 2018. Missing-value strategies and mixture-model baseline relevance.
[^11]: Jie Ren et al. [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/pdf/1906.02845). NeurIPS 2019. Background likelihood confounding and reference distributions.
[^12]: Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Balaji Lakshminarayanan. [Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality](https://arxiv.org/abs/1906.02994). 2019 preprint. Density versus typicality.
[^13]: Antonio Mari, Gennaro Vessio, Antonio Vergari. [Unifying and Understanding Overparameterized Circuit Representations via Low-Rank Tensor Decompositions](https://april-tools.github.io/publications/mari2023lorapc). TPM workshop 2023. Author/lab abstract and bibliographic record; used for the tensor connection, not an unverified theorem about this implementation.
[^14]: Robert Peharz et al. [Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits](https://proceedings.mlr.press/v119/peharz20a.html). ICML 2020, pp. 7563–7574. Vectorized execution and EM.
[^15]: Anji Liu, Kareem Ahmed, Guy Van den Broeck. [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). ICML 2024. Optimized PC execution and learning.
[^16]: Margaux Zaffran, Aymeric Dieuleveut, Julie Josse, Yaniv Romano. [Conformal Prediction with Missing Values](https://proceedings.mlr.press/v202/zaffran23a.html). ICML 2023, pp. 40578–40604. Pattern-specific coverage motivation.
[^17]: Margaux Zaffran, Julie Josse, Yaniv Romano, Aymeric Dieuleveut. [Predictive Uncertainty Quantification with Missing Covariates](https://arxiv.org/pdf/2405.15641). 2024 preprint. Missingness assumptions and hardness results; theory concerns prediction sets.
[^18]: Yonghoon Lee, Rina Foygel Barber, Rebecca Willett. [Distribution-free inference with hierarchical data](https://arxiv.org/abs/2306.06342). 2023 preprint, revised 2025. Grouped observations and hierarchical exchangeability.
[^19]: Marc Pinet, Julien Cumin, Samuel Berlemont, Dominique Vaufreydaz. [Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate](https://arxiv.org/html/2606.02670v4). MiLeTS workshop at KDD 2026. Benchmark diagnostic study and controlled relational anomalies.
[^20]: Bhumika K, Vidhya S, Narayanan C Krishnan. [A Probabilistic Circuit-Induced Pseudo-Metric for Out-of-Distribution Detection](https://arxiv.org/html/2608.09117v2). August 2026 preprint. HLV/HLD, batch-level evaluation, and model-misspecification limitations.
[^21]: Dominik Janzing, Lenon Minorics, Patrick Blöbaum. [Feature relevance quantification in explainable AI: A causal problem](https://proceedings.mlr.press/v108/janzing20a.html). AISTATS 2020, pp. 2907–2916. Observational/interventional explanation distinction.
[^22]: Agastya Kalra, Abdullah Rashwan, Wilson Hsu, Pascal Poupart, Prashant Doshi, George Trimponias. [Online Structure Learning for Feed-Forward and Recurrent Sum-Product Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/66121d1f782d29b62a286909165517bc-Paper.pdf). NeurIPS 2018, §2.1. Describes established generative/conditional learning and vanishing-update issues.
[^23]: Michael Gutmann, Aapo Hyvärinen. [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a.html). AISTATS 2010, pp. 297–304. Contrastive density-learning precedent.
[^24]: Tomáš Pevný, Václav Smídl, Martin Trapp, Ondřej Poláček, Tomáš Oberhuber. [Sum-Product-Transform Networks: Exploiting Symmetries using Invertible Transformations](https://proceedings.mlr.press/v138/pevny20a.html). PGM 2020 proceedings. Transformations within tractable probabilistic models.
[^25]: David Bindel, Cornell CS 6241. [Numerics for Data Science: Gaussian distributions and conditioning](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-04.html). March 4, 2025 lecture notes. Covariance/precision conditioning identities; the report gives the channel-block specialization.
[^26]: Lorenzo Loconte, Stefan Mengel, Antonio Vergari. [Sum of Squares Circuits](https://arxiv.org/abs/2408.11778). AAAI 2025; revised preprint May 2025. Expressiveness comparisons among monotone, squared, and sum-of-squares circuits.

### Repository evidence

Inspected sources: `IMPLEMENTATION_PLAN_2026-09-08.md`; `POC_REVIEW_2026-09-08.md`; `config/ts/tier1_kill_gate.yaml`; `logs/ts/tier1_kill_gate/gate_verdict.json`; `logs/ts/tier1_kill_gate/summary.md`; the three seeds' validation histories; `poc/time_series/pipeline.py` (`_fit_window_pc`, `_val_halves`, `channel_dependence`, `stage_structure_gate`); `poc/time_series/data.py` (`_inject`); `poc/time_series/relational.py` (`MaskCalibrator`); and `src/probabilistic_circuits.py` (`RelationalCircuit`). These are local working-tree observations, not claims that the experiments have been independently reproduced. The earlier recorded 451-test result was not rerun for this literature review.
