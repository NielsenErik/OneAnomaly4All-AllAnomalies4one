# Optimisation reliability — what was built on 12 September 2026, and what it is for

Written before the workstation run, so the criteria cannot be chosen after the
numbers. Roadmap context: [IMPLEMENTATION_PLAN_2026-09-10.md](IMPLEMENTATION_PLAN_2026-09-10.md)
steps 3 and 8. Results it reacts to: `logs/ts/ws/` at commit `8e56173`.

## The finding this answers

Five fresh seeds of ONE architecture (chow_liu, `leaf_components: 6`) on FD001,
from `logs/ts/ws/diagnosis_fd001_candidate/`:

| seed | best epoch | budget | checkpoint NLL | shape |
|---|---|---|---|---|
| 21 | 435 | 600 | 25.70 | stopped early |
| 22 | 80 | 600 | 59.06 | collapsed |
| 23 | 596 | 600 | 20.18 | at the ceiling |
| 24 | 599 | 600 | 27.23 | at the ceiling |
| 25 | 129 | 600 | 35.97 | collapsed |

A 38.9-nat spread across INITIALISATIONS, against a 0.06-nat spread between
architectures at a fixed learning rate on the synthetic grid. The advantage
over the fitted full-covariance Gaussian tracked that spread (Spearman −0.86
AUROC, −0.88 localisation AP, n = 8): well-fit seeds won by +0.017..+0.026
AUROC, collapsed seeds lost. Any architecture claim read off these runs is
currently a claim about initialisation luck, so no confirmation protocol can
be frozen on them.

Separately, the same study failed the step-8 false-alarm condition: observed
FPR 0.110 at α 0.10, with a repeated-draw q95 of **0.33**, on ~10 calibration
engines. One window per engine cannot resolve a rate finer than 1/n.

## What was implemented

1. **Restarts with checkpoint selection** (`poc/time_series/circuits.py`).
   `WindowPC.fit(restarts=n)` trains n independent initialisations of ONE
   structure — the vtree keeps the run seed, so a restart cannot quietly
   become a structure search — and keeps the one with the best checkpoint
   loss, the same label-free quantity early stopping already uses. Restarts
   are refused without checkpoint data. A restart more than
   `restart_abandon_margin` (relatively) behind the best completed one is
   abandoned after `min_epochs`, which is what makes four affordable; only
   losing attempts are ever cut, so the selected fit is the fit it would have
   been without the guard. Every attempt is logged and lands in the model row
   (`restarts_run`, `selected_restart`, `restart_checkpoint_losses`,
   `restarts_abandoned`), and `optimizer_steps` now counts all of them.
2. **Learning-rate schedules** (`lr_schedule: cosine | plateau`,
   `lr_min_factor`), so a run ends at an optimum rather than at the budget.
3. **Calibration split weights** (`eval.diagnosis_split_weights`). The
   checkpoint/calibration/evaluation split was a hard three-way `array_split`;
   it is now weighted, every partition still engine-disjoint and non-empty,
   and the protocol records `calibration_engines` / `evaluation_engines`.
   `(1, 2, 1)` takes FD001 from ~10 calibration engines to ~15.
4. **Both score views in the report** (`report_diagnosis`, `compare_diagnosis`).
   The default was `relational_max`, the weakest view measured (~0.71 AUROC
   against ~0.82 for `conditional_max` on the same artifacts), so an unflagged
   comparison silently argued against the circuit. The report now runs
   `conditional_max` and `relational_max` and tags every paired row with its
   view; `--score` is repeatable.
5. **The pre-registered gate** (`poc/time_series/gate_reliability.py`), which
   fails the recorded candidate study by construction (`tests/test_gate_reliability.py`).
6. **Configs**: `config/ts/diagnosis_fd001_reliability.yaml` (15 runs:
   candidate as-is, +restarts, +restarts and cosine) and
   `config/ts/diagnosis_fd003_pilot.yaml` (5 runs, step-8 variance sizing).
   `local_scripts/run_diagnosis_ws.sh` runs the reliability study, then the
   gate, and runs the FD003 pilot ONLY if the gate passes.

Defaults are unchanged (`restarts: 1`, `lr_schedule: none`, equal split), so
every earlier fit is reproduced exactly; restart 0 uses the run seed and the
same random streams as the single-restart path.

## Pre-registered acceptance criteria

- **Reliability** passes when the checkpoint-NLL spread over the five seeds is
  below 5 nats and no seed selects a checkpoint inside `min_epochs + patience`
  (collapse) or ends by exhausting its budget (ceiling). `gate_reliability`
  computes exactly this and prints every number it used.
- **False alarms** pass when the repeated-draw q95 falls below ~0.15 at
  α 0.10. A change in the point FPR alone is not a pass.
- **Anything else read off FD001 stays exploratory.** These are the engines the
  candidate was selected on. FD003 is within-family replication and sizes the
  confirmation; it does not confirm it.

## What is not claimed

Restarts and decay make the fit reliable, not better: if the reliable fit is
level with or worse than the fitted Gaussian, that is the result, and the
comparator wins. The cost also rises by up to `restarts×` — the benchmark
already reports the circuit as 6.7× slower than the Gaussian (median, 12/12
CPU workloads), and restarts widen that, which the quality–cost table must
carry rather than the accuracy table hiding it.
