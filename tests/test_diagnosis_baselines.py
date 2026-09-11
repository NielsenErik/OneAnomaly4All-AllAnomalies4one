"""Contracts for the fitted masked-diagnosis baselines (roadmap step 4).

These are the comparators the circuit is measured against, so a silent bug here
does not make a baseline look bad — it makes the circuit look good. Every test
compares a fast path against a deliberately naive dense recomputation, or
against an identity that has to hold whatever the fit turned out to be.
"""
import itertools

import numpy as np
import pytest
import torch

from poc.time_series.diagnosis_baselines import (BASELINES, GaussianDiagnoser,
    GMMDiagnoser, LowRankGaussianDiagnoser, MaskedDiagnoser, SingularModelError,
    _covariance, build_baselines, dense_reference_map)
from poc.time_series.diagnosis_controls import make_control_task


WINDOW, CHANNELS = 3, 3


def data(n_train=400, n_val=120, seed=5):
    task, cov = make_control_task(window=WINDOW, channels=CHANNELS,
                                  n_train=n_train, n_val=n_val, seed=seed)
    return task, cov


def fitted(kind, **kwargs):
    task, _ = data()
    model = BASELINES[kind](WINDOW, CHANNELS, **kwargs)
    return model.fit(task.X_train, task.X_val[:60]), task


@pytest.mark.parametrize("kind", ["gaussian", "lowrank", "gmm"])
def test_every_mask_matches_a_dense_recomputation_of_the_same_fit(kind):
    """The fast paths (subset caching, Woodbury, mixture marginals) must agree
    with a from-scratch dense Gaussian built from the SAME fitted parameters."""
    model, task = fitted(kind, **({"component_grid": (1,), "reg_grid": (1e-3,),
                                   "iterations": 30} if kind == "gmm" else {}))
    X = task.X_val[:16]
    if kind == "gaussian":
        mean, cov = model.mean_, model.cov_
    elif kind == "lowrank":
        mean = model.mean_
        cov = torch.diag(model.psi_) + model.W_ @ model.W_.T
    else:                                   # a one-component mixture IS a Gaussian
        mean, cov = model.parameters_.means[0], model.parameters_.covariances[0]
    for pattern in itertools.product([False, True], repeat=CHANNELS):
        if not any(pattern):
            continue
        fast = model.diagnosis_map(X, list(pattern))
        slow = dense_reference_map(mean, cov, WINDOW, CHANNELS, X, list(pattern))
        for key in ("log_px", "marginal", "conditional", "R"):
            assert torch.allclose(fast[key], slow[key], atol=1e-6, equal_nan=True), \
                (kind, pattern, key)


def test_mixture_marginal_sums_component_densities_under_fitted_weights():
    """Not an average of component log densities, and not a posterior reused
    after the evidence changed — both are plausible and both are wrong."""
    model, task = fitted("gmm", component_grid=(3,), reg_grid=(1e-3,), iterations=25)
    X = task.X_val[:12].double()
    params = model.parameters_
    for pattern in [(True, True, True), (True, False, True), (False, False, True)]:
        feats = [f for f in range(WINDOW * CHANNELS) if pattern[f % CHANNELS]]
        explicit = torch.logsumexp(torch.stack([
            torch.distributions.MultivariateNormal(
                params.means[j][feats],
                covariance_matrix=params.covariances[j][feats][:, feats]
            ).log_prob(X[:, feats]) for j in range(3)], dim=1) + params.log_weights, dim=1)
        got = model.diagnosis_map(X, list(pattern))["log_px"]
        assert torch.allclose(got, explicit, atol=1e-9)
        # the wrong-but-plausible alternatives must NOT coincide with it
        averaged = torch.stack([
            torch.distributions.MultivariateNormal(
                params.means[j][feats],
                covariance_matrix=params.covariances[j][feats][:, feats]
            ).log_prob(X[:, feats]) for j in range(3)], dim=1).mean(1)
        assert not torch.allclose(got, averaged, atol=1e-3)


@pytest.mark.parametrize("kind", ["gaussian", "lowrank", "gmm"])
def test_singleton_and_normalization_identities(kind):
    """One observed channel has no relation: R is exactly zero and the
    conditional collapses onto the marginal. An unobserved channel is nan."""
    model, task = fitted(kind, **({"component_grid": (2,), "reg_grid": (1e-3,),
                                   "iterations": 20} if kind == "gmm" else {}))
    X = task.X_val[:20]
    mask = [True] + [False] * (CHANNELS - 1)
    r = model.diagnosis_map(X, mask)
    assert torch.allclose(r["R"][:, 0], torch.zeros(len(X), dtype=r["R"].dtype), atol=1e-9)
    assert torch.equal(r["conditional"][:, 0], r["marginal"][:, 0])
    assert torch.isnan(r["R"][:, 1:]).all()
    # log p of the observed part alone is the channel's own marginal density
    assert torch.allclose(r["log_px"], -r["marginal"][:, 0], atol=1e-9)
    # and the full map's densities integrate to a proper density: a Gaussian
    # family is normalised by construction, so the check that bites is that the
    # joint never exceeds a marginal it contains.
    full = model.diagnosis_map(X)
    assert bool((full["log_px"] <= -full["marginal"].min(dim=1).values + 1e-6).all())


@pytest.mark.parametrize("kind", ["gaussian", "lowrank", "gmm"])
def test_heterogeneous_masks_match_independent_calls_and_preserve_row_order(kind):
    """Grouping unique masks is an optimisation; it must not permute rows."""
    model, task = fitted(kind, **({"component_grid": (2,), "reg_grid": (1e-3,),
                                   "iterations": 20} if kind == "gmm" else {}))
    X = task.X_val[:9]
    masks = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1],
                          [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype=torch.bool)
    grouped = model.diagnosis_map(X, masks)
    for i in range(len(X)):
        one = model.diagnosis_map(X[i:i + 1], masks[i])
        for key in ("log_px", "marginal", "conditional", "R"):
            assert torch.allclose(grouped[key][i:i + 1], one[key], atol=1e-9,
                                  equal_nan=True), (kind, i, key)
    assert torch.equal(grouped["mask"], masks)


@pytest.mark.parametrize("kind", ["gaussian", "lowrank", "gmm"])
def test_cached_and_uncached_queries_agree_and_the_cache_is_invalidated_on_refit(kind):
    model, task = fitted(kind, **({"component_grid": (2,), "reg_grid": (1e-3,),
                                   "iterations": 20} if kind == "gmm" else {}))
    X = task.X_val[:12]
    warm = model.diagnosis_map(X)
    assert model.cache_bytes > 0
    model._cache = {}                                   # cold
    cold = model.diagnosis_map(X)
    for key in ("log_px", "marginal", "conditional", "R"):
        assert torch.allclose(warm[key], cold[key], atol=1e-12, equal_nan=True)
    # A refit must not be served from the previous fit's factorisations.
    stale = model.factorization(tuple(range(WINDOW * CHANNELS)))
    model.fit(task.X_train[:100], task.X_val[:40])
    assert model.factorization(tuple(range(WINDOW * CHANNELS))) is not stale


def test_cost_ledger_counts_queries_and_masks_not_asymptotics():
    model, task = fitted("gaussian")
    X = task.X_val[:8]
    r = model.diagnosis_map(X, [True, True, True])
    # one joint plus, per observed channel, its own marginal and the complement
    assert r["cost"].passes == 1 + 2 * CHANNELS
    assert r["cost"].n_masks == 1
    assert r["cost"].output_elements == len(X) * CHANNELS
    masks = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool).repeat(4, 1)
    assert model.diagnosis_map(X, masks)["cost"].n_masks == 2


def test_selection_uses_checkpoint_data_and_records_the_grid():
    task, _ = data()
    model = GaussianDiagnoser(WINDOW, CHANNELS, shrinkage_grid=(0.0, 0.05, 0.5))
    model.fit(task.X_train, task.X_val[:60])
    assert model.selection_["split"] == "checkpoint"
    assert set(model.selection_["scores"]) == {"0.0", "0.05", "0.5"}
    assert model.lam_ in (0.0, 0.05, 0.5)
    # in-sample selection is allowed but must SAY so
    assert GaussianDiagnoser(WINDOW, CHANNELS).fit(task.X_train).selection_["split"] == "train"


def test_near_singular_training_data_is_regularised_or_refused_loudly():
    """A dead sensor gives a feature with exactly zero variance, so the sample
    covariance has a zero pivot. Either the shrinkage grid rescues it and the
    choice is recorded, or it raises — never a silent fallback to a different
    model, which is how a degenerate comparator ends up in a results table
    looking like a fair one."""
    task, _ = data()
    X = task.X_train.clone()
    X[:, 1] = 0.0                                       # a dead sensor
    model = GaussianDiagnoser(WINDOW, CHANNELS, shrinkage_grid=(0.0, 0.01, 0.2))
    model.fit(X, X[:80])
    # Whichever rescue applied — shrinkage or jitter — it is on the record.
    assert model.lam_ > 0 or model.stabilization_
    assert model.report()["stabilization_events"] == model.stabilization_
    strict = GaussianDiagnoser(WINDOW, CHANNELS, shrinkage_grid=(0.0,), jitter=0.0)
    with pytest.raises(SingularModelError):
        strict.fit(X, X[:80])


def test_every_method_emits_the_circuit_schema():
    """A comparison table is only a table if every method fills the same keys."""
    from poc.time_series.circuits import WindowPC
    task, _ = data()
    pc = WindowPC(WINDOW, CHANNELS, vtree_method="channel_blocked",
                  n_sum_components=2, device="cpu", seed=0)
    pc.fit(task.X_train, epochs=2, batch_size=64, X_val=task.X_val[:40])
    expected = set(pc.diagnosis_map(task.X_val[:5], [True, False, True]))
    for model in build_baselines(["gaussian", "lowrank", "gmm"], WINDOW, CHANNELS):
        model.fit(task.X_train, task.X_val[:60])
        got = model.diagnosis_map(task.X_val[:5], [True, False, True])
        assert set(got) == expected
        assert got["mask"].shape == (5, CHANNELS)
        assert got["R"].shape == got["marginal"].shape == (5, CHANNELS)
        assert set(model.report()) >= {"method", "fitted", "selection",
                                       "stabilization_events"}


def test_input_validation_refuses_the_shapes_that_would_be_wrong_quietly():
    model, task = fitted("gaussian")
    with pytest.raises(ValueError, match="mask must have shape"):
        model.diagnosis_map(task.X_val[:4], torch.ones(3, CHANNELS, dtype=torch.bool))
    with pytest.raises(ValueError):
        model.diagnosis_map(torch.randn(4, WINDOW * CHANNELS + 1))
    with pytest.raises(RuntimeError, match="fit before querying"):
        GaussianDiagnoser(WINDOW, CHANNELS).diagnosis_map(task.X_val[:4])
    with pytest.raises(FloatingPointError):
        GaussianDiagnoser(WINDOW, CHANNELS).fit(
            torch.full((20, WINDOW * CHANNELS), float("nan")))
    with pytest.raises(KeyError, match="unknown diagnosis baseline"):
        build_baselines(["nope"], WINDOW, CHANNELS)


def test_low_rank_query_never_materialises_the_dense_covariance():
    """The Woodbury path is the reason the low-rank arm is a distinct cost
    model; a factorisation that stored a k×k inverse would not be one."""
    model, task = fitted("lowrank", rank_grid=(2,))
    fac = model.factorization(tuple(range(WINDOW * CHANNELS)))
    assert fac.chol.shape == (2, 2)
    assert fac.psi_inv.shape == (WINDOW * CHANNELS,)
    assert fac.wt_psi_inv.shape == (2, WINDOW * CHANNELS)
