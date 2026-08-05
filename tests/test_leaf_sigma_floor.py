"""
Tests for the leaf width floors — the INIT floor and the RUNTIME floor.

These are two different things and conflating them has already cost one
wasted benchmark run.  `use_relative_floor` switches the RUNTIME floor only:

    init     sigma = max(MAD, 0.01*std, 1e-3)     in BOTH modes
    runtime  sigma >= 0.01*std (relative) | 1e-5 (legacy)

If the flag is ever wired to the initialisation as well, `--floor legacy`
stops being the pre-change behaviour and becomes a third regime with no floor
anywhere, and every A/B against a recorded number is meaningless.  That is
what these tests exist to catch.

The degenerate columns are not synthetic curiosities: MAD is exactly 0 on a
whole sensor channel of every C-MAPSS subset (30/450 features on FD001), so
the floored path is the one the RUL bench actually runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probabilistic_circuits import (
    GaussianLeaf,
    GaussianMixtureLeaf,
    relative_sigma_floor,
)

# name -> column.  Chosen so the floor binds on some and not others.
COLUMNS = {
    "ordinary": np.random.default_rng(0).normal(3.0, 2.0, 512),
    "mad_zero": np.r_[np.full(400, 1.0),
                      np.random.default_rng(1).normal(1.0, 0.5, 112)],
    "constant": np.full(512, 7.0),
    "tiny_spread": np.random.default_rng(2).normal(0.0, 1e-4, 512),
}


@pytest.fixture(autouse=True)
def _restore_flags():
    """The floors are CLASS attributes, so a test that flips one would leak
    into every later test in the session."""
    a, b = GaussianLeaf.use_relative_floor, GaussianMixtureLeaf.use_relative_floor
    yield
    GaussianLeaf.use_relative_floor = a
    GaussianMixtureLeaf.use_relative_floor = b


def _set_mode(relative: bool) -> None:
    GaussianLeaf.use_relative_floor = relative
    GaussianMixtureLeaf.use_relative_floor = relative


def _single_init(vals: np.ndarray) -> float:
    """The pre-change initialisation rule, verbatim from commit 8816d97."""
    mu = float(np.median(vals))
    mad = float(np.median(np.abs(vals - mu))) * 1.4826
    return max(mad, 0.01 * float(np.std(vals)), 1e-3)


def _mixture_init(vals: np.ndarray, n: int) -> float:
    """Ditto for the mixture leaf, which had NO relative bound at init."""
    return max(float(np.std(vals) + 1e-6) / n, 1e-3)


@pytest.mark.parametrize("name", list(COLUMNS))
def test_legacy_init_reproduces_pre_change_width(name):
    """--floor legacy must reproduce the recorded runs, so its INIT width has
    to equal the old rule.  (The old code realised that width as
    softplus(inv_softplus(s)) + 1e-5, i.e. s + 1e-5; the epsilon was a guard,
    not intent, so the target is s itself.)"""
    _set_mode(False)
    vals = COLUMNS[name]
    X = vals.reshape(-1, 1)

    leaf = GaussianLeaf(0)
    leaf.fit(X)
    assert leaf.sigma.item() == pytest.approx(_single_init(vals), rel=1e-5)

    for n in (2, 3, 5):
        mix = GaussianMixtureLeaf(0, n_components=n)
        mix.fit(X)
        assert mix.sigmas.min().item() == pytest.approx(_mixture_init(vals, n),
                                                        rel=1e-5)


@pytest.mark.parametrize("name", list(COLUMNS))
def test_runtime_floor_is_the_only_thing_the_flag_switches(name):
    """The buffer differs by mode; the initial width does not, except where
    the relative bound binds and sigma = floor + softplus() needs headroom."""
    vals = COLUMNS[name]
    X = vals.reshape(-1, 1)
    rel = relative_sigma_floor(vals)

    _set_mode(False)
    legacy = GaussianLeaf(0)
    legacy.fit(X)
    _set_mode(True)
    relative = GaussianLeaf(0)
    relative.fit(X)

    assert legacy.sigma_floor.item() == pytest.approx(1e-5)
    assert relative.sigma_floor.item() == pytest.approx(rel)

    binds = _single_init(vals) < 1.25 * rel
    if binds:
        # the documented asymmetry: 1.25x the floor, never below it
        assert relative.sigma.item() >= rel
        assert relative.sigma.item() == pytest.approx(1.25 * rel, rel=1e-4)
    else:
        assert relative.sigma.item() == pytest.approx(legacy.sigma.item(),
                                                      rel=1e-5)


@pytest.mark.parametrize("relative", [True, False])
def test_floor_holds_under_gradient_pressure(relative):
    """A floor that only holds at init is the bug the runtime floor fixed:
    fit a leaf to a near-constant column and push it with an explicit
    collapse objective (minimise sigma) for enough steps to reach the bound."""
    _set_mode(relative)
    vals = COLUMNS["constant"]
    leaf = GaussianLeaf(0)
    leaf.fit(vals.reshape(-1, 1))
    bound = relative_sigma_floor(vals) if relative else 1e-5

    opt = torch.optim.Adam(leaf.parameters(), lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        leaf.sigma.backward()          # drive sigma DOWN, hard
        opt.step()

    mode = "relative" if relative else "legacy"
    assert torch.isfinite(leaf.sigma)
    assert leaf.sigma.item() >= bound * (1 - 1e-6), (
        f"sigma {leaf.sigma.item():.3e} fell below its {mode} "
        f"floor {bound:.3e}")


def test_degenerate_column_density_is_finite_in_both_modes():
    """The original failure mode: MAD == 0 gives sigma ~ 1e-6, an ordinary
    observation lands at |z| ~ 7e5, and log-density is ~ -2.3e11 which NaNs
    the circuit on the first step.  Neither mode may do that any more, since
    both now floor the INITIALISATION."""
    for relative in (True, False):
        _set_mode(relative)
        vals = COLUMNS["mad_zero"]
        leaf = GaussianLeaf(0)
        leaf.fit(vals.reshape(-1, 1))
        lp = leaf.log_density(torch.tensor(vals, dtype=torch.float32))
        assert torch.isfinite(lp).all()
        assert lp.min().item() > -1e6, f"log-density {lp.min().item():.3e}"
