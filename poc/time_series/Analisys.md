 Yes—the project is worth pursuing, but with a substantially narrower research claim. You have a useful implementation; you do not yet have convincing evidence for a universal anomaly detector, uniquely superior explanations, or better censored RUL prediction.

  I wrote the detailed assessment in POC_REVIEW_2026-09-08.md, including evidence, literature comparisons, implementation findings, and continuation/stop criteria.

  The most promising question is: can a compact probabilistic circuit identify broken sensor relationships reliably when some sensors become unavailable?

  Distinguishing “this sensor is unusual” from “this sensor is individually plausible but inconsistent with the others” is useful. Exact marginalization makes changing sensor availability a natural extension. However, the mathematical identities are established, and
  Gaussian/GMM models also support these queries. Your contribution must be a demonstrated advantage in diagnostic quality, robustness, or computational cost.

  Several findings materially change the current project narrative:

   Finding                                                                                                               Implication
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   The “held-out NLL” is calculated on training windows.                                                                 Existing likelihood comparisons do not establish generalization.
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   The AE function called “sampling-SHAP” averages absolute changes after replacing individual channels.                 It is not SHAP; the claimed exact-versus-approximate SHAP advantage is unsupported.
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Completeness is tested for chain-rule attribution, while localization often uses leave-one-out conditional scores.    You cannot combine those results into one “correct and complete” explanation claim.
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   All seven recorded RUL failures occur on mean-imputed inputs.                                                         They do not prove that the full-input RUL model collapsed, as the handoff suggests.
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Aggregation includes partial results from failed runs, while stale metrics files can remain.                          Results need verified provenance and stage-level completion before publication.
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   The conformal calibrator does use alpha; a controlled check changes the intervals.                                    Identical recorded intervals do not establish missing alpha plumbing. Raw interval evaluation has a separate fixed-90% defect.

  The results themselves remain mixed. In the successful FD001 structure sweep, Chow–Liu achieves AUROC 0.8539, random 0.8388, and the original chain 0.7528. Three-component leaves improve the chain to 0.8265, but that change also changes initialization and leaf
  implementation. It does not isolate a capacity benefit.

  The current explanation benchmark also contradicts “exclusivity on explanation”: on FD001, Gaussian conditional localization achieves 0.8807, versus 0.8625 for PC conditional attribution. Moreover, circuit-based outlier explanation through tractable marginals already
  has direct prior art: Lüdtke et al., 2022 (https://arxiv.org/abs/2207.08414).

  Two broader claims need restraint:

  - E-values are an extension, not evidence of PC-specific novelty. Some advertised benefits do not require a circuit. Nonoverlapping windows also do not automatically satisfy the exchangeability assumptions behind the monitoring guarantee.
  - C-MAPSS is simulator-generated benchmark data, not measured fleet failures. Your injected fault labels add another synthetic layer. This is useful experimental evidence, but not field validation. NASA dataset description
    (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

  My recommendation is one focused validation cycle of roughly two working weeks:

  1. Repair evaluation/provenance issues and establish genuinely held-out validation engines.
  2. Separate initialization effects from leaf capacity.
  3. Compare relational diagnosis against matched Gaussian/GMM conditionals and existing SPN explanation methods.
  4. Test diagnosis across realistic corruptions and missing-sensor patterns, then confirm the effect on another independent source.

  Continue if that diagnostic advantage survives. Stop expanding the broad method claim if simpler tractable models match it at lower cost. Keep RUL secondary and defer further routing, curvature, SOS, and e-value expansion until this central result exists.

  I ran 150 targeted tests, all passing. No experiment code was changed; the report is the only added file.

─ Worked for 8m 23s ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› If this project is fixed and hold, can it be pushed for ICML or AJCAI?


• I’ll assess the potential against ICML and IJCAI’s criteria, separating a corrected POC from the additional contribution a competitive submission would need.

• Searching the web

• Searched the web for site.icml.cc Conferences 2026 reviewer instructions originality significance soundness ...

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Yes—there is a plausible path to ICML or IJCAI, assuming you mean IJCAI by “AJCAI.” But fixing the POC and reproducing its gains would not, by itself, make it competitive. The remaining requirement is a clear, significant contribution beyond existing PC-based anomaly
  explanation.

  Both venues explicitly assess originality and significance alongside correctness. A technically sound system can still be incremental. ICML criteria (https://icml.cc/Conferences/2026/ReviewerInstructions), IJCAI criteria
  (https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-main-track/).

  My assessment is:

   What the corrected project demonstrates                                                                          Venue outlook
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PCs work for detection, explanations, and calibrated RUL                                                         Useful integration; insufficiently distinctive for either main track
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────────────
   A new diagnostic method consistently improves relational fault localization under missing sensors                Plausible IJCAI or ICML submission
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────────────
   A general learning/inference advance explains why and when that advantage occurs, with convincing experiments    Stronger ICML positioning

  The strongest paper would revolve around one claim:

  > Tractable joint models enable reliable relational anomaly diagnosis under changing sensor availability, with a demonstrably better accuracy–computation tradeoff than existing alternatives.

  That is a candidate contribution, not an established finding. To make it convincing, you need four things.

  1. A method beyond the existing identities.
     Marginal versus conditional surprise is useful, but the identity itself is established. The novelty could be a principled diagnostic procedure, learning objective, or efficient algorithm for finding anomalous sensor subsets under partial observation. It must clearly
     advance beyond existing SPN outlier explanation.

  2. A result that survives the strongest counterargument.
     The obvious reviewer objection is: “A Gaussian mixture can also compute these marginals and conditionals.” Your method needs to demonstrate where those alternatives fail or become expensive, and why the circuit avoids that limitation.

  3. Evidence across independent settings.
     Show relational anomalies, realistic sensor failures, multiple missingness patterns, and a second independent data source. Include strong tractable baselines and meaningful uncertainty estimates. Several C-MAPSS subsets alone provide limited evidence of broad
     generality.

  4. An explanation of the mechanism.
     Establish that the benefit comes from learned dependencies and tractable inference, rather than initialization, extra parameters, a favorable corruption generator, or a weak explanation baseline. A substantive theoretical result would help, but neither venue requires
     attaching a theorem to an otherwise strong empirical contribution.

  For venue fit, my judgment is that IJCAI is a natural target if the main advance is a new probabilistic reasoning/diagnosis method. ICML becomes especially compelling if you uncover a broadly applicable learning principle or a strong general empirical finding. This is
  about fit; IJCAI should not be treated as the fallback for insufficient novelty.

  I would not make RUL, e-values, routing, curvature, and multimodality equal contributions in the same submission. At present, they spread the evidence too thin. One substantial diagnostic result is more persuasive than five capabilities whose individual novelty is
  unclear.

  The decisive milestone is whether you can complete this sentence convincingly:

  > “Compared with the closest existing method, we introduce X, which makes Y possible or materially better, because Z.”

  If X remains “using a PC,” the top-tier case is weak. If it becomes a new method with a reproducible advantage that simpler tractable models cannot match, aiming for ICML or IJCAI is entirely reasonable.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› International Joint Conference on Artificial Intelligence


• Yes—IJCAI, the International Joint Conference on Artificial Intelligence, is the venue I meant.

  It is a credible target if the corrected project establishes a distinct methodological contribution, especially in probabilistic reasoning and diagnosis under partial observation.

  My recommendation would be to develop the paper around relational anomaly diagnosis with missing sensors, supported by strong Gaussian/GMM and existing PC explanation baselines. Fixing the implementation establishes correctness; demonstrating a new, useful capability
  establishes the IJCAI case.

