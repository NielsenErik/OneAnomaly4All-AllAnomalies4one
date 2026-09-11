"""Contracts for matched donor construction and the shortcut audit (step 5).

Two of these tests exist to stop a specific way of cheating rather than a
specific way of crashing: donors must never come from the engines the study
evaluates on, and neither matching nor the audit may touch anomaly labels or
model scores.
"""
import numpy as np
import pytest
import torch

from poc.time_series.data import contaminate_windows, make_ad_task
from poc.time_series.diagnosis_donors import (DonorCovariates, DonorMatcher,
    covariates_from_task, shortcut_audit, target_channel_features)


def task(seed=0, **kwargs):
    return make_ad_task(window=4, stride=3, seed=seed, n_units=14,
                        n_channels=5, n_regimes=3, **kwargs)


def test_window_construction_carries_regime_and_health_for_every_split():
    t = task()
    for split in ("train", "val", "test"):
        cov = covariates_from_task(t, split)
        assert len(cov) == len(getattr(t, f"X_{split}"))
        assert cov.names == ["regime", "health"]
        assert cov.unit is not None


def test_matching_improves_balance_and_stays_inside_the_caliper():
    t = task()
    donors, recipients = covariates_from_task(t, "train"), covariates_from_task(t, "val")
    result = DonorMatcher(caliper=0.25, seed=1).fit(donors).match(recipients)
    matched = result.donor_index >= 0
    assert matched.any()
    assert (result.distance[matched] <= 0.25 + 1e-12).all()
    for name in ("regime", "health"):
        assert abs(result.balance[f"smd_after_{name}"]) <= \
               abs(result.balance[f"smd_before_{name}"]) + 1e-9
    # exact stratum matching means the donor's regime equals the recipient's
    donor_regime = np.asarray(donors.regime)[result.donor_index[matched]]
    assert np.array_equal(donor_regime, np.asarray(recipients.regime)[matched])


def test_donors_come_only_from_the_training_pool():
    """A donor drawn from a held-out engine leaks the evaluation split into the
    corruption itself, and no amount of downstream care recovers from that."""
    t = task()
    donors = covariates_from_task(t, "train")
    result = DonorMatcher(seed=0).fit(donors).match(covariates_from_task(t, "val"))
    used = result.donor_index[result.donor_index >= 0]
    assert used.max() < len(t.X_train)
    train_units = set(np.asarray(t.unit_train).tolist())
    assert set(np.unique(result.donor_unit[result.donor_index >= 0]).tolist()) <= train_units
    val_units = set(np.asarray(t.unit_val).tolist())
    assert not (set(np.unique(result.donor_unit[result.donor_index >= 0]).tolist()) & val_units)


def test_matching_never_sees_labels_or_scores():
    """The covariate container has no slot for either, so a future change that
    tries to match on them cannot do it quietly."""
    assert set(DonorCovariates().__dataclass_fields__) == {"regime", "health", "unit"}
    with pytest.raises(TypeError):
        DonorCovariates(labels=np.zeros(3))                # type: ignore[call-arg]


def test_assignments_reproduce_from_the_seed_and_differ_across_seeds():
    t = task()
    donors, recipients = covariates_from_task(t, "train"), covariates_from_task(t, "val")
    a = DonorMatcher(seed=3).fit(donors).match(recipients).donor_index
    b = DonorMatcher(seed=3).fit(donors).match(recipients).donor_index
    c = DonorMatcher(seed=4).fit(donors).match(recipients).donor_index
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unavailable_matches_are_reported_and_left_uncorrupted():
    """A caliper tight enough to admit nothing must produce an unmatched arm,
    not a silent fallback to a random donor."""
    t = task()
    donors = covariates_from_task(t, "train")
    recipients = covariates_from_task(t, "val")
    result = DonorMatcher(caliper=1e-9, seed=0).fit(donors).match(recipients)
    assert result.n_unmatched > 0
    assert np.isinf(result.distance[result.donor_index < 0]).all()
    X, y, kinds, affected, used = contaminate_windows(
        t.X_val, 4, 5, inject_rate=1.0, donors=t.X_train, seed=1,
        donor_index=result.donor_index, return_donors=True)
    unmatched = result.donor_index < 0
    assert (y.numpy()[unmatched] == 0).all()
    assert torch.equal(X[torch.from_numpy(unmatched)], t.X_val[torch.from_numpy(unmatched)])
    assert (used[unmatched] == -1).all()


def test_matcher_refuses_a_source_with_no_matching_variables():
    empty = DonorCovariates(unit=np.arange(5))
    with pytest.raises(ValueError, match="matching on nothing"):
        DonorMatcher().fit(empty)
    with pytest.raises(ValueError, match="donor pool is empty"):
        DonorMatcher().fit(DonorCovariates(regime=np.zeros(0)))


def test_a_fixed_donor_assignment_makes_the_corruption_replayable():
    t = task()
    index = np.arange(len(t.X_val)) % len(t.X_train)
    first = contaminate_windows(t.X_val, 4, 5, inject_rate=1.0, donors=t.X_train,
                                seed=9, donor_index=index)[0]
    second = contaminate_windows(t.X_val, 4, 5, inject_rate=1.0, donors=t.X_train,
                                 seed=9, donor_index=index)[0]
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="one donor per recipient"):
        contaminate_windows(t.X_val, 4, 5, donors=t.X_train, donor_index=index[:3])
    with pytest.raises(IndexError):
        contaminate_windows(t.X_val, 4, 5, inject_rate=1.0, donors=t.X_train,
                            donor_index=np.full(len(t.X_val), 10 ** 6))


def test_shortcut_audit_detects_a_planted_marginal_shortcut_and_ignores_a_relational_one():
    """The audit is only useful if it is sensitive to exactly one thing."""
    generator = torch.Generator().manual_seed(4)
    window, channels, n = 4, 3, 300
    clean = torch.randn(n, window * channels, generator=generator)
    marginal = clean.clone()
    marginal[:, 0::channels] += 4.0                     # channel 0 shifted: a shortcut
    relational = clean.clone()
    relational[:, 1::channels] = torch.randn(n, window, generator=generator)  # channel 1 broken
    labels = np.r_[np.zeros(n), np.ones(n)]
    loud = shortcut_audit(torch.cat((clean, marginal)), labels,
                          torch.cat((clean, marginal)), labels, window, channels, 0)
    quiet = shortcut_audit(torch.cat((clean, relational)), labels,
                           torch.cat((clean, relational)), labels, window, channels, 0)
    assert loud["shortcut_auroc"] > 0.95
    assert abs(quiet["shortcut_auroc"] - 0.5) < 0.1


def test_shortcut_audit_refuses_a_single_class_split():
    X = torch.randn(20, 12)
    with pytest.raises(ValueError, match="both classes"):
        shortcut_audit(X, np.zeros(20), X, np.r_[np.zeros(10), np.ones(10)], 4, 3)


def test_target_channel_features_read_one_channel_only():
    X = torch.zeros(6, 4 * 3)
    X[:, 1::3] = 5.0                                    # only channel 1 is nonzero
    assert not target_channel_features(X, 4, 3, 0).any()
    assert target_channel_features(X, 4, 3, 1).any()
