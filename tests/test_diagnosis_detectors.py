"""Contracts for the trivial detection comparators.

These exist to be beaten, so the only thing that matters is that they are not
handicapped: the detector must see the observed channels as a detector fitted
on those channels would see them, and its cache must never serve a fit made
for a different sensor configuration.
"""
import numpy as np
import pytest
import torch

from poc.time_series.diagnosis_detectors import (MaskedDetector, available,
                                                 build_detectors, _columns)


def _data(n=200, window=4, channels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, window * channels, generator=g)


def test_columns_follow_the_flattened_timestep_channel_layout():
    cols = _columns(torch.tensor([True, False, True]), window=3, channels=3)
    # t * C + c for c in {0, 2}, t in {0, 1, 2}
    assert cols.tolist() == [0, 2, 3, 5, 6, 8]


def test_columns_refuse_an_empty_or_wrongly_shaped_pattern():
    with pytest.raises(ValueError, match="at least one observed"):
        _columns(torch.tensor([False, False]), window=2, channels=2)
    with pytest.raises(ValueError, match="one shared channel pattern"):
        _columns(torch.tensor([True, True, True]), window=2, channels=2)


@pytest.mark.parametrize("key", ["knn", "mahalanobis", "pca", "diag_gaussian"])
def test_refit_sees_only_the_observed_columns(key):
    """A refitted detector must be blind to the channels it does not observe.

    Corrupting a hidden channel cannot move a score that never read it; if it
    does, the comparator is being fed evidence the circuit is denied, and every
    comparison against it is wrong in the circuit's favour.
    """
    window, channels = 4, 3
    train, test = _data(300, window, channels, 0), _data(50, window, channels, 1)
    det = MaskedDetector(key, window, channels, seed=0, policy="refit").fit(train)
    mask = torch.tensor([True, True, False])            # channel 2 hidden
    base = det.score(test, mask)
    poisoned = test.clone()
    poisoned[:, [t * channels + 2 for t in range(window)]] += 50.0
    assert np.allclose(base, det.score(poisoned, mask))


def test_impute_uses_the_training_mean_not_the_query_value():
    window, channels = 3, 3
    train, test = _data(300, window, channels, 2), _data(40, window, channels, 3)
    det = MaskedDetector("mahalanobis", window, channels, seed=0,
                         policy="impute").fit(train)
    mask = torch.tensor([True, False, True])
    base = det.score(test, mask)
    poisoned = test.clone()
    poisoned[:, [t * channels + 1 for t in range(window)]] -= 30.0
    assert np.allclose(base, det.score(poisoned, mask))


def test_the_full_mask_reduces_to_the_ordinary_detector():
    window, channels = 4, 3
    train, test = _data(300, window, channels, 4), _data(60, window, channels, 5)
    full = torch.ones(channels, dtype=torch.bool)
    a = MaskedDetector("knn", window, channels, seed=0, policy="refit").fit(train)
    b = MaskedDetector("knn", window, channels, seed=0, policy="impute").fit(train)
    assert np.allclose(a.score(test, full), b.score(test, full))


def test_the_cache_is_per_pattern_and_refits_are_counted():
    window, channels = 3, 3
    train, test = _data(200, window, channels, 6), _data(30, window, channels, 7)
    det = MaskedDetector("pca", window, channels, seed=0, policy="refit").fit(train)
    one = torch.tensor([True, True, False])
    two = torch.tensor([True, False, True])
    det.score(test, one); det.score(test, one)          # cached: still one fit
    assert det.n_fits == 1
    det.score(test, two)
    assert det.n_fits == 2
    # Different patterns must not share a fit.
    assert not np.allclose(det.score(test, one), det.score(test, two))


def test_refitting_invalidates_the_cache():
    window, channels = 3, 3
    train, test = _data(200, window, channels, 8), _data(30, window, channels, 9)
    det = MaskedDetector("mahalanobis", window, channels, seed=0).fit(train)
    mask = torch.tensor([True, True, False])
    first = det.score(test, mask)
    det.fit(_data(200, window, channels, 99) * 3.0 + 1.0)
    assert not np.allclose(first, det.score(test, mask))


def test_a_window_shaped_detector_is_built_for_the_restricted_geometry():
    """z-score and the moving-average residual reshape to (window, channels).

    Handing them the original channel count and a narrower matrix reshapes the
    window silently and scores a different quantity, so the restricted count is
    what they must be constructed with.
    """
    window, channels = 4, 4
    train, test = _data(200, window, channels, 10), _data(30, window, channels, 11)
    det = MaskedDetector("zscore", window, channels, seed=0).fit(train)
    scores = det.score(test, torch.tensor([True, True, False, False]))
    assert scores.shape == (30,) and np.isfinite(scores).all()


def test_scoring_before_fitting_is_an_error_and_names_are_distinct():
    det = MaskedDetector("knn", 3, 3, seed=0)
    with pytest.raises(RuntimeError, match="fit before scoring"):
        det.score(_data(10, 3, 3), torch.ones(3, dtype=torch.bool))
    built = build_detectors(["knn", "pca"], 3, 3, policies=("refit", "impute"))
    assert len({d.name for d in built}) == 4


def test_an_unknown_detector_is_refused_loudly():
    with pytest.raises(KeyError, match="unknown detector"):
        MaskedDetector("not_a_detector", 3, 3)
    assert "mahalanobis" in available() and "knn" in available()
