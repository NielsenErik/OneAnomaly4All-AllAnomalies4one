"""
Tier 0 — the evaluation and provenance repairs, pinned one defect at a time.

Every test here corresponds to a finding in POC_REVIEW_2026-09-08.md §2 and to
a row of IMPLEMENTATION_PLAN_2026-09-08.md Tier 0.  They are cheap (seconds,
tiny circuits) because their job is to make a class of silent error loud, not
to measure anything:

  0.1  the reported held-out likelihood is scored on units the model never saw,
       and the epoch that model ends on is a decision made against it
  0.2  a run that did not finish cannot contribute rows to a table
  0.3  average precision does not depend on how ties happen to be sorted
  0.4  the window ending at the final timestep exists
  0.5  an interval reported at a nominal level is evaluated at that level
  0.6  the AE baseline is not called SHAP, because it is not SHAP
  0.7  completeness is claimed for the attribution it holds for, and the
       non-additive attributions carry the size of their gap

The failure these guard against is the one that has cost this project the most:
a number that is well-formed, plausible, and answering a different question
from the one its column name asks.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.circuits import SurvivalPC, WindowPC
from poc.time_series.data import (
    ADTask,
    make_ad_task,
    make_rul_task,
    split_fit_val_units,
    windowize,
)
from poc.time_series.datasets import make_ad_task_split
from poc.time_series.explain import (
    ADDITIVE_ATTRIBUTIONS,
    additivity_gap,
    completeness_error,
    pc_attributions,
)
from poc.time_series.metrics import average_precision, auroc
from poc.time_series.ts_logging import (
    RunLogger,
    provenance_report,
    read_results,
    row_is_trustworthy,
)

SMALL = dict(n_units=16, n_channels=6, n_regimes=2)


@pytest.fixture(scope="module")
def task():
    return make_ad_task(window=4, stride=2, seed=0, **SMALL)


# ═══════════════════════════════════════════════════════════════════════════
# 0.1  Held-out validation — engines, not windows
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_units_are_disjoint_from_fitting_units(task):
    """
    The defect: `stage_ad` reported `train_nll=_held_out_nll(pc, task.X_train)`.
    A held-out-sounding helper reading the training set means no recorded run
    can separate overfitting from undertraining, and every likelihood
    comparison across structures is uninterpretable.

    Engine-level disjointness is the part that matters.  Windows of one
    trajectory overlap and share a wear state, so a window-level split would
    report the training likelihood with extra steps.
    """
    assert task.X_val is not None and len(task.X_val) > 0
    fit_u = set(task.unit_train.tolist())
    val_u = set(task.unit_val.tolist())
    test_u = set(task.unit_test.tolist())
    assert fit_u and val_u and test_u
    assert not (fit_u & val_u), "validation shares an engine with fitting"
    assert not (fit_u & test_u) and not (val_u & test_u)
    assert len(task.unit_train) == len(task.X_train)
    assert len(task.unit_val) == len(task.X_val)
    assert len(task.unit_test) == len(task.X_test)


def test_official_split_builder_also_carries_units_and_validation():
    """The same guarantee on the path real benchmarks take (an official test
    fleet), not only on the single-fleet synthetic path."""
    from poc.time_series.data import simulate_fleet
    tr = simulate_fleet(seed=0, **SMALL)
    te = simulate_fleet(seed=1, **SMALL)
    t = make_ad_task_split(tr, te, window=4, stride=2, seed=0)
    assert t.X_val is not None and len(t.X_val)
    assert not (set(t.unit_train.tolist()) & set(t.unit_val.tolist()))
    assert len(t.unit_test) == len(t.X_test)
    assert t.meta["val_units"] >= 1


def test_the_standardiser_never_sees_the_validation_units():
    """
    Preprocessing is fitted data too.  If the standardiser sees the validation
    engines, the held-out likelihood is contaminated before the model starts —
    the review's §1 point that a latent/preprocessed pipeline is only as
    held-out as its least held-out stage.

    Checked by construction: the fleet is built so one unit is wildly
    off-scale, and forcing it into the validation set must not move the
    training windows' scale.
    """
    from poc.time_series.data import simulate_fleet
    fleet = simulate_fleet(seed=0, **SMALL)
    fleet.series[-1] = fleet.series[-1] * 50.0 + 500.0     # an extreme engine
    a = make_ad_task(window=4, stride=2, seed=0, fleet=fleet, val_units=0.2)
    b = make_ad_task(window=4, stride=2, seed=0, fleet=fleet, val_units=0.0)
    # The standardiser is fit on fewer units in `a`, so the two differ; what
    # must hold is that `a`'s training windows are standardised — i.e. finite
    # and of order 1 — rather than carrying another engine's 500-unit offset.
    assert torch.isfinite(a.X_train).all()
    assert float(a.X_train.abs().max()) < 100.0
    assert float(b.X_train.abs().max()) < 100.0


def test_split_fit_val_units_never_starves_the_fitting_set():
    rng = np.random.default_rng(0)
    for n in range(1, 8):
        fit, val = split_fit_val_units(range(n), 0.2, rng)
        assert len(fit) >= 1
        assert len(fit) + len(val) == n
        assert not (set(fit.tolist()) & set(val.tolist()))
        if n >= 2:
            assert len(val) >= 1, "a 2-unit fleet must still yield a val unit"
    fit, val = split_fit_val_units(range(10), 0.0, rng)
    assert len(val) == 0 and len(fit) == 10


def test_checkpoint_selection_returns_the_best_validation_epoch(task):
    """
    The model that gets scored must be the one the validation split chose, not
    whichever epoch the budget stopped on.  Pinned end-to-end: after `fit`, the
    circuit's val NLL must equal the best value recorded during training — which
    also proves the rollback reached the compiled evaluator AND the DAG, since
    `score` reads the DAG.
    """
    pc = WindowPC(task.window, task.n_channels, vtree_method="chow_liu",
                  n_sum_components=4, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=6, lr=0.05, X_val=task.X_val)
    assert len(pc.val_history) == 6
    assert 0 <= pc.best_epoch < 6
    assert pc.best_val_nll == pytest.approx(min(pc.val_history))
    assert float(pc.score(task.X_val).mean()) == pytest.approx(
        pc.best_val_nll, abs=1e-3)


def test_a_diverged_epoch_cannot_become_the_checkpoint(task):
    """
    NaN loses every comparison, including `<`.  A first-epoch NaN must not be
    able to lock out the finite epochs that follow it — the shape of bug that
    silently pins a model at its worst state.
    """
    pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                  n_sum_components=3, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=3, lr=0.05, X_val=task.X_val)
    assert np.isfinite(pc.best_val_nll)
    assert pc.best_epoch >= 0


def test_fitting_without_a_validation_split_selects_nothing(task):
    """No split, no selection, and no silent substitution of a training
    number: `val_history` stays empty and `best_epoch` stays -1."""
    pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                  n_sum_components=3, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=2, lr=0.05)
    assert pc.val_history == [] and pc.best_epoch == -1


# ═══════════════════════════════════════════════════════════════════════════
# 0.2  Provenance — an unfinished run is not a result
# ═══════════════════════════════════════════════════════════════════════════

def test_rows_from_a_failed_run_are_refused_but_completed_stages_survive():
    """
    The aggregation defect.  `results.jsonl` is truncated at the start of an
    attempt and appended to as stages finish, so a failed directory holds NEW
    partial rows; `metrics.json` is written only on a clean exit, so it can
    hold a STALE summary at the same time.  Reading rows blind mixes partial
    results into tables; reading metrics.json instead reports an older run.

    The rule: a row counts if its run finished, or if its own stage was marked
    complete before a later stage crashed.
    """
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "run1")
        with RunLogger(d, config={"name": "t"}, seed=0, swallow=True) as log:
            # exactly what `pipeline.run_stages` does around each stage
            log.result({"stage": "ad", "method": "PC", "auroc": 0.9})
            log.stage_ok("ad")
            log.result({"stage": "rul", "method": "PC", "crps": 1.0})
            log.stage_failed("rul", "RuntimeError: crashed inside rul")
            raise RuntimeError("crashed inside rul")

        st = json.load(open(os.path.join(d, "status.json")))
        assert st["status"] == "failed"
        assert st["stages"]["ad"] == "ok" and st["stages"]["rul"] == "failed"

        kept = read_results(root)
        assert [r["stage"] for r in kept] == ["ad"], (
            "the completed `ad` stage must survive a crash in `rul`, and the "
            "partial `rul` row must not reach a table")

        every = read_results(root, require_ok=False)
        assert len(every) == 2 and not every[1]["trustworthy"]

        prov = provenance_report(root)
        assert prov["rows_total"] == 2 and prov["rows_used"] == 1
        assert prov["rows_dropped"] == 1 and prov["incomplete_runs"] == ["run1"]


def test_run_stages_marks_each_stage_as_it_finishes(tmp_path):
    """The wiring, not just the logger's API: a crash in a later stage must
    leave the earlier stages' verdicts on disk."""
    from poc.time_series import pipeline

    def ok(cfg, seed, log):
        log.result({"stage": "ad", "method": "fake", "auroc": 0.7})
        return {}

    def boom(cfg, seed, log):
        log.result({"stage": "rul", "method": "fake", "crps": 9.9})
        raise RuntimeError("stage exploded")

    cfg = {"name": "t", "stages": ["ad", "rul"], "device": "cpu",
           "dataset": {"name": "synthetic"}}
    d = str(tmp_path / "run")
    saved = dict(pipeline.STAGE_FNS)
    pipeline.STAGE_FNS.update({"ad": ok, "rul": boom})
    try:
        with RunLogger(d, config=cfg, seed=0, swallow=True) as log:
            pipeline.run_stages(cfg, 0, log)
    finally:
        pipeline.STAGE_FNS.clear(); pipeline.STAGE_FNS.update(saved)

    st = json.load(open(os.path.join(d, "status.json")))
    assert st["status"] == "failed"
    assert st["stages"] == {"ad": "ok", "rul": "failed"}
    kept = read_results(str(tmp_path))
    assert [r["stage"] for r in kept] == ["ad"]


def test_a_directory_with_no_status_contributes_nothing():
    """A results.jsonl with no status.json beside it is output of unknown
    provenance — the case that made "seven attempts of a three-seed config"
    aggregate as 21 seeds."""
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "orphan")
        os.makedirs(d)
        with open(os.path.join(d, "results.jsonl"), "w") as f:
            f.write(json.dumps({"stage": "ad", "method": "m", "auroc": 1.0}) + "\n")
        assert read_results(root) == []
        assert not row_is_trustworthy({"stage": "ad"}, None)


def test_a_run_still_in_flight_contributes_only_its_finished_stages():
    """
    The same rule for a run that is still going (or was killed by a scheduler
    and left "running" on disk): a stage marked complete counts, an unmarked
    one does not.  Pinned because the intuitive reading of "status != ok ->
    drop everything" would silently discard hours of finished work on every
    partially-complete batch.
    """
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "inflight")
        os.makedirs(os.path.join(d, "artifacts"))
        with open(os.path.join(d, "results.jsonl"), "w") as f:
            f.write(json.dumps({"stage": "ad", "method": "m", "auroc": 0.6}) + "\n")
            f.write(json.dumps({"stage": "rul", "method": "m", "crps": 2.0}) + "\n")
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"status": "running", "attempt": "x",
                       "stages": {"ad": "ok"}}, f)
        assert [r["stage"] for r in read_results(root)] == ["ad"]


def test_every_row_carries_the_attempt_that_produced_it():
    """A run directory is reused by every re-run, so 'which execution produced
    this row' is otherwise unanswerable after the fact."""
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "run")
        attempts = []
        for _ in range(2):
            with RunLogger(d, config={"name": "t"}, seed=0) as log:
                log.result({"stage": "ad", "method": "m", "auroc": 0.5})
                log.stage_ok("ad")
                attempts.append(log.attempt)
        assert attempts[0] != attempts[1]
        rows = read_results(root)
        # results.jsonl is truncated per attempt: exactly one row, the newest
        assert len(rows) == 1 and rows[0]["attempt"] == attempts[1]
        cfg = json.load(open(os.path.join(d, "config.json")))
        assert cfg["attempt"] == attempts[1]


# ═══════════════════════════════════════════════════════════════════════════
# 0.3  Average precision — ties are a threshold, not an order
# ═══════════════════════════════════════════════════════════════════════════

def test_average_precision_groups_tied_scores():
    """
    Four identical scores carrying two positives: example-wise AP gives 1.0 or
    0.4167 depending on which order the sort happens to produce; the
    threshold-wise answer is 0.5 and is a property of the scores.  Quantised or
    degenerate detectors produce ties in bulk, so this is not a corner case.
    """
    assert average_precision([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0]) == pytest.approx(0.5)
    assert average_precision([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.5)
    assert average_precision([3.0, 2.0, 1.0], [1, 0, 1]) == pytest.approx(5 / 6)


def test_average_precision_is_permutation_invariant_under_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 3, 40).astype(float)          # heavily tied on purpose
    y = (rng.random(40) < 0.35).astype(int)
    base = average_precision(s, y)
    for _ in range(8):
        p = rng.permutation(len(s))
        assert average_precision(s[p], y[p]) == pytest.approx(base)
    # and a constant detector scores the positive rate, not 1.0
    assert average_precision(np.ones(10), [1] * 3 + [0] * 7) == pytest.approx(0.3)


def test_average_precision_matches_sklearn_where_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.integers(0, 4, 60).astype(float)
        y = (rng.random(60) < 0.3).astype(int)
        assert average_precision(s, y) == pytest.approx(
            sk.average_precision_score(y, s), abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# 0.4  The last cycle exists
# ═══════════════════════════════════════════════════════════════════════════

def test_windowize_includes_the_window_ending_at_the_last_timestep():
    """
    T=24, window=20, stride=3: the stride grid ends at 22 while the trajectory
    ends at 23.  `make_rul_task_split(test_windows="last")` takes the last row
    of that grid and calls it the official one-prediction-per-engine protocol,
    so it was scoring a window that is not the benchmark's, at the point of the
    trajectory where RUL is smallest.
    """
    x = np.arange(24 * 2, dtype=np.float32).reshape(24, 2)
    W, right = windowize(x, 20, 3)
    assert right[-1] == 23
    assert len(W) == len(right)
    assert np.array_equal(W[-1].reshape(20, 2), x[4:24])
    legacy = windowize(x, 20, 3, include_last=False)[1]
    assert legacy[-1] == 22, "the pre-fix grid must stay reproducible"


def test_windowize_does_not_duplicate_an_exact_final_window():
    x = np.arange(21 * 2, dtype=np.float32).reshape(21, 2)
    right = windowize(x, 20, 1)[1]
    assert right.tolist() == [19, 20]
    x2 = np.arange(20 * 2, dtype=np.float32).reshape(20, 2)
    assert windowize(x2, 20, 3)[1].tolist() == [19]
    assert len(windowize(x2[:5], 20, 3)[1]) == 0


def test_the_last_rul_window_is_the_last_cycle():
    """The protocol-level consequence: one prediction per engine, at the
    engine's final observation."""
    from poc.time_series.data import simulate_fleet
    from poc.time_series.datasets import make_rul_task_split
    tr = simulate_fleet(seed=0, **SMALL)
    te = simulate_fleet(seed=1, **SMALL)
    t = make_rul_task_split(tr, te, window=6, stride=4, n_bins=10, cap=100.0,
                            test_windows="last", seed=0)
    # one row per test unit, each at that unit's smallest RUL
    assert len(t.X_test) == len(te)
    for u in range(len(te)):
        sel = (t.unit_test == u)
        assert int(sel.sum()) == 1
        assert float(t.rul_test[sel]) == pytest.approx(
            float(min(te.rul[u][-1], t.cap)), abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 0.5  An interval is evaluated at the level it is reported at
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def survival():
    t = make_rul_task(seed=0, censor_frac=0.3, window=6, stride=4, n_units=20,
                      n_channels=5, n_regimes=2, n_bins=10, cap=100.0)
    pc = SurvivalPC(t.window, t.n_channels, t.n_bins, t.cap,
                    vtree_method="chain", n_sum_components=6, tau_where="deep",
                    seed=0, device="cpu")
    pc.fit(t.X_train, t.tau_train, t.delta_train, epochs=12)
    return t, pc


def test_predict_endpoints_follow_the_requested_alpha(survival):
    """
    The defect: `_eval_survival` always read `q05`/`q95` while passing the
    requested alpha into the interval-score penalty, so an alpha=0.20 row
    scored 90% endpoints under an 80% penalty and reported it as an 80%
    interval.
    """
    t, pc = survival
    p10 = pc.predict(t.X_test, alpha=0.10)
    p20 = pc.predict(t.X_test, alpha=0.20)
    w10 = (p10["q_hi"] - p10["q_lo"]).mean()
    w20 = (p20["q_hi"] - p20["q_lo"]).mean()
    assert w20 <= w10, "an 80% interval cannot be wider than a 90% one"
    assert not torch.equal(p10["q_lo"], p20["q_lo"]) or \
           not torch.equal(p10["q_hi"], p20["q_hi"]), (
        "the endpoints did not move with alpha at all — the level is being "
        "ignored, which is the defect this pins")


def test_the_fixed_90_columns_keep_their_meaning(survival):
    """`q05`/`q95` are never re-levelled, so every number already recorded
    against those names stays comparable; alpha selects `q_lo`/`q_hi`."""
    t, pc = survival
    p = pc.predict(t.X_test, alpha=0.30)
    p90 = pc.predict(t.X_test, alpha=0.10)
    assert torch.equal(p["q05"], p90["q05"]) and torch.equal(p["q95"], p90["q95"])
    assert torch.equal(p90["q_lo"], p90["q05"]) and torch.equal(p90["q_hi"], p90["q95"])
    assert bool((p["q_lo_edge"] <= p["q_lo"]).all())
    assert bool((p["q_hi_edge"] >= p["q_hi"]).all())


def test_eval_survival_records_the_level_it_scored(survival):
    from poc.time_series.pipeline import _eval_survival
    t, pc = survival
    m10, _ = _eval_survival(pc, t, 0.10)
    m20, _ = _eval_survival(pc, t, 0.20)
    assert m10["nominal_level"] == 0.90 and m20["nominal_level"] == 0.80
    assert m20["mpiw"] <= m10["mpiw"] + 1e-6
    assert m20["picp"] <= m10["picp"] + 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 0.6 / 0.7  Names and claims
# ═══════════════════════════════════════════════════════════════════════════

def test_the_ae_baseline_is_not_called_shap():
    """
    It replaces one channel at a time and averages ABSOLUTE score changes: no
    coalition of size > 1 is formed, no permutation is averaged, and signed
    contributions cannot cancel.  More samples converge to that statistic, not
    to a Shapley value, so the name has to go with the mathematics.
    """
    import poc.time_series.explain as ex
    assert hasattr(ex, "replacement_sensitivity")
    assert not hasattr(ex, "sampling_shap")
    src = open(os.path.join(os.path.dirname(__file__), "..", "poc",
                            "time_series", "pipeline.py")).read()
    assert "sampling_shap" not in src
    assert "AE replacement sensitivity" in src


def test_completeness_belongs_to_the_chain_rule_attribution(task):
    """
    CLAIM 2 is about `chain_rule_attribution` and nothing else.  The
    leave-one-out conditional surprises do not telescope: their sum exceeds the
    joint NLL by the window's dependence.  Both are legitimate; only one is a
    decomposition, and the review's §2.C defect was reporting the localisation
    of one beside the completeness of the other.
    """
    pc = WindowPC(task.window, task.n_channels, vtree_method="chow_liu",
                  n_sum_components=4, seed=0, device="cpu")
    # Enough epochs to have LEARNED dependence: see the test below — a circuit
    # at initialisation is a product of independent channels, and under that
    # model the conditional attribution IS additive.  The claim being
    # separated here only exists once the model has cross-channel structure.
    pc.fit(task.X_train, epochs=30, lr=0.05, X_val=task.X_val)
    X = task.X_test[:16]

    comp = completeness_error(pc, X)
    assert comp["attribution"] in ADDITIVE_ATTRIBUTIONS
    assert comp["max_residual_nats"] < 1e-2          # float32 round-off only

    attrs = pc_attributions(pc, X, shapley_orders=0, chain_rule=True)
    additive = attrs["PC chain-rule (exact, additive)"]
    cond = attrs["PC conditional (exact)"]

    gap_add = additivity_gap(pc, X, additive)
    gap_cond = additivity_gap(pc, X, cond)
    assert abs(gap_add["mean_abs_gap_nats"]) < 1e-2, (
        "the additive attribution must sum to the NLL")
    assert gap_cond["mean_abs_gap_nats"] > 10 * max(
        gap_add["mean_abs_gap_nats"], 1e-6), (
        "the conditional attribution is being treated as additive; its gap is "
        "the dependence in the window and must be reported, not implied away")


def test_a_factorised_circuit_is_visible_in_the_structural_term(task):
    """
    A guardrail found while writing the test above, and worth keeping.

    At initialisation (leaves fit closed-form, sum weights untrained) this
    circuit is EXACTLY a product of independent channels: the structural term
    −log p(x_c|x_-c) + log p(x_c) is 1e-6, the sum of marginals equals the
    joint NLL to float32, and the conditional attribution is additive.  That is
    the signature of a model carrying no cross-channel information — the same
    silent degeneracy class as the five already on record, and invisible in the
    training loss.

    Any relational claim (structural scores, conditional localisation, the
    diagnosis paper's whole premise) is vacuous under such a model, so the
    difference between "trained" and "factorised" must be measurable in one
    line.

    Measured on this fixture (chow_liu, K=4, 380 training windows), the
    dependence |Σ_c marginal − NLL| in nats by epoch:

        1: 0.0000   5: 0.0000   10: 0.0000   20: 0.0002
        30: 0.013   40: 1.020   60: 2.755

    i.e. the circuit is EXACTLY factorised for its first ~20 epochs and only
    then begins to couple channels, while the training NLL falls smoothly
    throughout (32.9 -> 15.2) and says nothing about it.  Three seeds at 40
    epochs: chow_liu 1.02/1.14/1.58, chain 4.66/5.10/5.15.  The crossover is in
    gradient STEPS, not epochs, so this does not by itself say anything about
    the recorded 50-epoch C-MAPSS runs — it says the quantity has to be
    measured there rather than assumed.
    """
    X = task.X_test[:16]

    def dependence(epochs: int) -> float:
        pc = WindowPC(task.window, task.n_channels, vtree_method="chow_liu",
                      n_sum_components=4, seed=0, device="cpu")
        pc.fit(task.X_train, epochs=epochs, lr=0.05)
        td = pc.typed_scores(X)
        with torch.no_grad():
            nll = -pc.pc.log_prob(pc._prep(X)).cpu()
        return float((td["marginal"].sum(1) - nll).abs().mean())

    at_init, trained = dependence(1), dependence(40)
    print(f"\n[factorisation] |sum marginals - NLL|: init {at_init:.2e} nats, "
          f"trained {trained:.3f} nats")
    assert at_init < 1e-3, (
        "the untrained circuit is no longer a clean product — this test's "
        "reference point moved, re-derive it before trusting the comparison")
    assert trained > 0.1, (
        "after training the circuit still factorises across channels: every "
        "relational quantity in this project is then identically zero and the "
        "conditional scores carry no information the marginals lack")


def test_additive_attributions_are_declared_not_inferred(task):
    """
    The pipeline labels rows from this tuple, so the tuple is a claim and has
    to be checked rather than trusted: every member must actually sum to the
    NLL, and no non-member may.

    Sampled Shapley qualifies for a reason worth stating: each permutation's
    chain-rule terms telescope to the total exactly, so an average over ANY set
    of permutations does too.  Only the ordering is sampled, never the
    conditionals — which is precisely the property the discarded "approximate
    SHAP" comparison was gesturing at.
    """
    assert "PC chain-rule (exact, additive)" in ADDITIVE_ATTRIBUTIONS
    assert "PC conditional (exact)" not in ADDITIVE_ATTRIBUTIONS
    assert "PC marginal (exact)" not in ADDITIVE_ATTRIBUTIONS
    assert "PC structural (exact)" not in ADDITIVE_ATTRIBUTIONS

    pc = WindowPC(task.window, task.n_channels, vtree_method="chow_liu",
                  n_sum_components=4, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=40, lr=0.05)     # enough to have dependence
    X = task.X_test[:12]
    attrs = pc_attributions(pc, X, shapley_orders=3, chain_rule=True)
    gaps = {k: additivity_gap(pc, X, v)["mean_abs_gap_nats"]
            for k, v in attrs.items()}
    print(f"\n[additivity] " + ", ".join(f"{k}: {v:.4f}" for k, v in gaps.items()))
    for name, gap in gaps.items():
        if name in ADDITIVE_ATTRIBUTIONS:
            assert gap < 1e-2, f"{name} is declared additive but misses by {gap}"
        else:
            assert gap > 0.1, (
                f"{name} is not declared additive yet sums to the NLL — either "
                "the declaration is wrong or the model has factorised")


# ═══════════════════════════════════════════════════════════════════════════
# End to end — the stages that the repairs above changed
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_cfg(stages):
    from poc.time_series.config import DEFAULTS
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg.update({"name": "t", "variant": "default", "seeds": [0], "device": "cpu",
                "stages": stages, "variant_axes": {}})
    cfg["dataset"].update({"units": 12, "channels": 5, "regimes": 2,
                           "window": 4, "stride": 3})
    cfg["model"].update({"K": 3, "epochs": 2})
    cfg["eval"].update({"fast_baselines": True, "plots": False,
                        "examples": False, "save_scores": False,
                        "deletion": False, "shapley_orders": 0,
                        "shap_samples": 2, "n_complete": 8,
                        "max_explain_windows": 40})
    return cfg


def test_ad_stage_reports_validation_separately_from_training(tmp_path):
    """The consumer side of 0.1: the row carries a val NLL scored on held-out
    engines, the count of those engines, and the epoch that was selected."""
    from poc.time_series.pipeline import run_stages

    cfg = _smoke_cfg(["ad"])
    rdir = str(tmp_path / "run")
    with RunLogger(rdir, config=cfg, seed=0) as log:
        run_stages(cfg, 0, log)
    rows = [r for r in read_results(rdir) if "RegionGraphPC" in str(r["method"])]
    assert rows, "the circuit row is missing"
    r = rows[0]
    assert "val_nll" in r and "train_nll" in r
    assert np.isfinite(r["val_nll"]) and r["val_units"] >= 1
    assert r["val_nll"] != r["train_nll"], (
        "the validation likelihood equals the training likelihood — the split "
        "is not reaching the model, which is the defect this fixes")
    assert 0 <= r["best_epoch"] < cfg["model"]["epochs"]
    assert os.path.exists(os.path.join(rdir, "history_pc_val_nll.csv"))


def test_explain_stage_separates_the_two_explanation_claims(tmp_path):
    """
    The consumer side of 0.6/0.7, end to end: every localisation row says
    whether its statistic is additive and how far it is from summing to the
    score, the completeness row names the attribution it belongs to, and the AE
    baseline is not called SHAP anywhere in the output.
    """
    from poc.time_series.pipeline import run_stages

    cfg = _smoke_cfg(["explain"])
    rdir = str(tmp_path / "run")
    with RunLogger(rdir, config=cfg, seed=0) as log:
        run_stages(cfg, 0, log)
    rows = read_results(rdir)
    by_method = {r["method"]: r for r in rows}

    assert not any("SHAP" in m for m in by_method), (
        f"a SHAP-named row survives: {sorted(by_method)}")
    assert any("replacement sensitivity" in m for m in by_method)

    loc = [r for r in rows if "loc_auroc" in r]
    assert loc, "no localisation rows were written"
    for r in loc:
        assert "additive" in r and "mean_abs_gap_nats" in r

    chain = by_method.get("PC chain-rule (exact, additive)")
    assert chain is not None and chain["additive"] is True
    assert chain["mean_abs_gap_nats"] < 1e-2
    cond = by_method["PC conditional (exact)"]
    assert cond["additive"] is False

    comp = by_method["PC chain-rule completeness"]
    assert comp["attribution"] in ADDITIVE_ATTRIBUTIONS
    assert comp["max_residual_nats"] < 1e-2
