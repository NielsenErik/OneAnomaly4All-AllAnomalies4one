"""Availability, identifiability and calibration contracts (roadmap step 6).

The theme: a missing sensor must not be able to influence a query, an
unidentifiable case must be exposed rather than answered, and a false-alarm
rate must come with its uncertainty and with how often the honest threshold was
infinite.
"""
import numpy as np
import pytest
import torch

from poc.time_series.diagnosis import (conservative_threshold, localization_metrics,
    operational_trials, trajectory_reduce)
from poc.time_series.diagnosis_controls import (gaussian_oracle_map, witness_availability,
    witness_covariance, witness_masks, witness_task)
from poc.time_series.relational import (fixed_unseen_mask, independent_random_masks,
    informative_masks, missingness_workload)


WINDOW, CHANNELS, WITNESSES, TARGET = 3, 5, (1, 2), 0


# ── availability: the evidence either was there or was not ──────────────────

def test_the_witness_graph_routes_every_path_through_the_witnesses():
    cov = witness_covariance(WINDOW, CHANNELS, WITNESSES, TARGET)
    target = [t * CHANNELS + TARGET for t in range(WINDOW)]
    others = [t * CHANNELS + c for t in range(WINDOW)
              for c in range(CHANNELS) if c != TARGET and c not in WITNESSES]
    assert float(cov[target][:, others].abs().max()) == 0.0
    assert float(torch.linalg.eigvalsh(cov).min()) > 0


def test_removing_every_witness_makes_the_correct_score_exactly_zero():
    """Not small — zero. A method reporting a signal here is reporting noise,
    and a method that abstains is behaving correctly."""
    t, cov = witness_task(window=WINDOW, channels=CHANNELS, witnesses=WITNESSES,
                          n_train=64, n_val=200, seed=3)
    masks = witness_masks(CHANNELS, TARGET, WITNESSES)
    none = gaussian_oracle_map(t.X_val, cov, CHANNELS, masks["witnesses_none"])
    assert float(none["R"][:, TARGET].abs().max()) < 1e-10
    every = gaussian_oracle_map(t.X_val, cov, CHANNELS, masks["witnesses_all"])
    assert float(every["R"][:, TARGET].abs().mean()) > 0.1


def test_witness_removal_is_monotone_in_the_evidence_it_leaves():
    t, cov = witness_task(window=WINDOW, channels=CHANNELS, witnesses=WITNESSES,
                          n_train=64, n_val=300, seed=7)
    masks = witness_masks(CHANNELS, TARGET, WITNESSES)
    scores = [float(gaussian_oracle_map(t.X_val, cov, CHANNELS,
                                        masks[name])["R"][:, TARGET].abs().mean())
              for name in ("witnesses_all", "witnesses_drop1", "witnesses_none")]
    assert scores[0] > scores[1] > scores[2]


def test_availability_reporting_names_the_unidentifiable_cases():
    masks = witness_masks(CHANNELS, TARGET, WITNESSES)
    facts = {n: witness_availability(m, TARGET, WITNESSES) for n, m in masks.items()}
    assert facts["witnesses_all"]["identifiable"] and facts["witnesses_all"]["witnesses_available"] == 2
    assert facts["witnesses_drop1"]["witnesses_available"] == 1
    assert not facts["witnesses_none"]["identifiable"]
    assert not facts["target_hidden"]["target_visible"]


# ── identifiability: hidden evidence cannot move a query ────────────────────

def test_a_hidden_target_leaves_every_remaining_query_bit_identical():
    t, cov = witness_task(window=WINDOW, channels=CHANNELS, witnesses=WITNESSES,
                          n_train=32, n_val=120, seed=11)
    altered = t.X_val.clone()
    altered[:, TARGET::CHANNELS] += 25.0                # a blatant change, hidden
    mask = witness_masks(CHANNELS, TARGET, WITNESSES)["target_hidden"]
    before = gaussian_oracle_map(t.X_val, cov, CHANNELS, mask)
    after = gaussian_oracle_map(altered, cov, CHANNELS, mask)
    assert torch.equal(before["log_px"], after["log_px"])
    assert torch.allclose(before["R"], after["R"], equal_nan=True)


# ── the missingness regimes ─────────────────────────────────────────────────

def test_independent_random_masks_do_not_depend_on_the_data():
    masks, meta = independent_random_masks(64, CHANNELS, k=2, seed=1)
    assert masks.shape == (64, CHANNELS)
    assert (masks.sum(1) == CHANNELS - 2).all()
    assert not meta["depends_on_values"] and not meta["depends_on_fault"]
    again, _ = independent_random_masks(64, CHANNELS, k=2, seed=1)
    assert torch.equal(masks, again)
    with pytest.raises(ValueError):
        independent_random_masks(4, CHANNELS, k=CHANNELS)


def test_fixed_unseen_pattern_is_shared_and_declares_it_is_not_exchangeable():
    masks, meta = fixed_unseen_mask(10, CHANNELS, [3])
    assert (masks == masks[0]).all()
    assert not bool(masks[0, 3])
    assert meta["exchangeable_with_calibration"].startswith("no")


def test_informative_dropout_actually_reads_the_hidden_state():
    generator = torch.Generator().manual_seed(2)
    X = torch.randn(200, WINDOW * CHANNELS, generator=generator)
    X[:, 4::CHANNELS] *= 8.0                            # channel 4 is loud
    value_masks, meta = informative_masks(X, WINDOW, CHANNELS, mechanism="value",
                                          strength=4.0, seed=0)
    assert meta["depends_on_values"]
    # the loud channel is the one that disappears
    assert float((~value_masks[:, 4]).float().mean()) > 0.5
    assert float((~value_masks[:, 0]).float().mean()) < 0.05
    affected = [[1] for _ in range(len(X))]
    fault_masks, fmeta = informative_masks(X, WINDOW, CHANNELS, affected,
                                           mechanism="fault", fault_dropout=1.0, seed=0)
    assert fmeta["depends_on_fault"]
    assert not fault_masks[:, 1].any()
    assert fault_masks.any(dim=1).all()                 # never an empty observation


def test_every_regime_is_generated_from_a_named_mechanism():
    X = torch.randn(32, WINDOW * CHANNELS)
    work = missingness_workload(
        ["independent_random", "fixed_unseen", "informative_value", "informative_fault"],
        X, WINDOW, CHANNELS, affected=[[2]] * 32, unseen=[4], seed=0)
    assert set(work) == {"independent_random", "fixed_unseen", "informative_value",
                         "informative_fault"}
    for name, (masks, meta) in work.items():
        assert masks.shape == (32, CHANNELS)
        assert masks.any(dim=1).all()
        assert meta["mechanism"] and "exchangeable_with_calibration" in meta
    with pytest.raises(KeyError, match="unknown missingness regime"):
        missingness_workload(["telepathy"], X, WINDOW, CHANNELS)


# ── localisation under per-row masks ────────────────────────────────────────

def test_localization_scores_each_row_against_its_own_observation():
    """A shared-pattern score would credit a method for a channel that was not
    there in that particular window."""
    attr = np.array([[9.0, 1.0, 0.0], [9.0, 1.0, 0.0]])
    affected = [[0], [0]]
    observed = np.array([[True, True, True], [False, True, True]])
    alarms = np.ones(2, dtype=bool)
    per_row = localization_metrics(attr, affected, observed, alarms)
    assert per_row["loc_n_no_observed_target"] == 1     # row 1 cannot be localised
    assert per_row["loc_n_unique"] == 1
    shared = localization_metrics(attr, affected, observed[0], alarms)
    assert shared["loc_n_no_observed_target"] == 0
    assert shared["loc_n_unique"] == 2


def test_abstention_is_counted_separately_from_a_wrong_answer():
    attr = np.array([[9.0, 1.0, 0.0]] * 4)
    metrics = localization_metrics(attr, [[0]] * 4, [True, True, True],
                                   np.array([True, False, False, True]))
    assert metrics["loc_n_abstained_no_alarm"] == 2
    assert metrics["loc_unique_top1_accuracy"] == 1.0   # always the right channel
    assert metrics["end_to_end_unique_top1"] == 0.5     # but only half alarmed


# ── operational calibration ─────────────────────────────────────────────────

def test_repeated_trials_report_spread_power_and_infinite_thresholds():
    rng = np.random.default_rng(0)
    units = np.repeat(np.arange(40), 5)
    calibration = rng.normal(size=len(units))
    evaluation = np.r_[rng.normal(size=200), rng.normal(3.0, size=200)]
    labels = np.r_[np.zeros(200), np.ones(200)].astype(int)
    out = operational_trials(calibration, units, evaluation, labels, alpha=0.1,
                             trials=50, seed=0)
    assert out["trial_calibration_objects"] == 40
    assert out["trial_fpr_q05"] <= out["trial_fpr_mean"] <= out["trial_fpr_q95"]
    assert out["trial_fpr_q95"] > out["trial_fpr_q05"]  # one draw is not the answer
    assert out["trial_power_mean"] > 0.5
    assert out["trial_infinite_threshold_frac"] == 0.0
    # too few independent objects: the honest threshold is infinite, every time
    few = operational_trials(calibration[:5], units[:5] // 1, evaluation, labels,
                             alpha=0.05, trials=10, seed=0)
    assert few["trial_infinite_threshold_frac"] == 1.0
    assert few["trial_fpr_mean"] == 0.0 and few["trial_power_mean"] == 0.0


def test_a_window_threshold_is_not_a_trajectory_threshold():
    """Reducing to per-engine maxima has to happen BEFORE calibration, and the
    two arms of the paired design are separate trajectories."""
    scores = np.array([1.0, 5.0, 2.0, 9.0])
    units = np.array([0, 0, 1, 1])
    arms = np.array([0, 1, 0, 1])
    reduced, out_units, out_arms = trajectory_reduce(scores, units, arms)
    assert list(reduced) == [1.0, 5.0, 2.0, 9.0]        # arms never merge
    assert list(out_arms) == [0, 1, 0, 1]
    merged, merged_units = trajectory_reduce(scores, units)
    assert list(merged) == [5.0, 9.0] and list(merged_units) == [0, 1]


def test_the_conservative_threshold_stays_infinite_rather_than_guessing():
    assert np.isinf(conservative_threshold([1.0], 0.05))
    assert conservative_threshold(np.arange(19), 0.05) == 18
