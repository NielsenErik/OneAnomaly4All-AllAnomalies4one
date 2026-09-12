# Diagnosis implementation completion record

Completed audit: **10 September 2026**, against commit `df9df38`. The filename preserves the reference already published in the September 9 test plan.

The diagnosis foundation is implemented and its integration is verified. The scientific study is not complete. The current file-level roadmap is [IMPLEMENTATION_PLAN_2026-09-10.md](IMPLEMENTATION_PLAN_2026-09-10.md); the assumption matrix and experimental rationale remain in [DIAGNOSIS_TEST_PLAN_2026-09-09.md](DIAGNOSIS_TEST_PLAN_2026-09-09.md).

## Verification performed in this audit

Environment: repository root, `/Users/eriknielsen/miniconda3/envs/expllm_env/bin/python` (Python 3.11); smoke configured for CPU.

| Check | Outcome |
|---|---|
| `python -m pytest -q -k 'not test_density_pc_128_features'` | **470 passed, 1 deselected**, 83.76 seconds; seven warnings |
| `python -m poc.time_series.runner config/ts/diagnosis_smoke.yaml --log-root /tmp/diagnosis_completion_20260910` | **5/5 runs completed**, blocked, upper_capacity, mixture, conditional and unblocked; four masks each |
| Controls config with `--dry-run` | **10 variants × 3 seeds = 30 runs** |
| FD001 config with `--dry-run` | **6 variants × 3 seeds = 18 runs** |
| `compare_diagnosis` on fresh mixture/blocked mask-0 artifacts, `--reps 100` | Completed paired comparison across 20 independent synthetic units |

The deselected legacy 128-feature density test remains unverified in this audit, matching the prior documented exclusion. Warnings concern TorchScript deprecation, physical-core discovery fallback, tensor-to-scalar conversion and correlation on constant data. No test failures occurred. The paired smoke comparison checks the artifact pipeline, not statistical superiority; its short training budget cannot support that conclusion.

Fresh smoke and comparison outputs are in `/tmp/diagnosis_completion_20260910`; these are temporary local artifacts. The previously committed five-arm smoke record remains in `logs/ts/diagnosis_smoke/`. This audit did not overwrite that record or rerun the historical Tier 1 gate.

## Delivered versus pending

The implemented inventory in the September 9 test plan is supported by the current suite and fresh smoke run: exact circuit diagnosis, interface capacity, shared-latent mixtures, conditional training, isolated checkpoint/calibration/evaluation roles, known-law controls, ambiguity-aware metrics, conservative thresholds, work accounting and paired AUROC comparison.

Still pending: the full trained controls and FD001 studies, optimized fitted Gaussian/GMM diagnosis baselines, matched donor construction and shortcut audit, graph-witness and informative-missingness experiments, unseen-mask statistical procedures, cross-family end-to-end benchmarks, localization uncertainty/report generation and independent confirmation. EM/Anemone and restructuring/distillation are conditional follow-ups. Existing generic anomaly baselines and the known-law Gaussian oracle do not complete the fitted masked-diagnosis baseline requirement.

The historical gate remains FAIL. Neither passing regression tests nor a completed smoke run reverses it. The next execution task is the frozen known-law study; the next substantial implementation task is the fitted baseline interface in step 4 of the current roadmap.

## Addendum, 12 September 2026

Roadmap steps 1-7 were executed on 10-11 September (workstation results in `logs/ts/ws/`, commit `8e56173`). Step 8 is NOT frozen: the candidate's fit is decided by its initialisation (38.9-nat checkpoint-NLL spread across five seeds of one architecture) and it failed the false-alarm condition (repeated-draw q95 0.33 at alpha 0.10 on ~10 calibration engines). The reliability work that has to clear both before a protocol can be frozen is described, with its pre-registered criteria, in [RELIABILITY_STEP_2026-09-12.md](RELIABILITY_STEP_2026-09-12.md).
