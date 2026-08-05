"""
AD diagnostics — WHERE the detection/explanation result comes from.

The AD half of the PoC reports parity on detection and exclusivity on
explanation.  Neither number tells you *why*, and every wrong answer this
project has produced was a number that looked reasonable.  The tests here are
not regression tests on a value; each one isolates one link in the chain

    generator  ->  injected anomaly  ->  density  ->  attribution  ->  metric

and fails loudly when that link is not carrying what the write-up says it is.
Four questions, in the order a referee would ask them:

  A. Is the METRIC valid?          (an oracle must score 1.0, noise must score 0.5)
  B. Is the GENERATOR's premise true?  (are "structural" anomalies really
     marginal-preserving, i.e. is the marginal/conditional gap what it claims?)
  C. Is the MODEL using the data?  (a density can depend on x while ignoring
     most channels — `assert_informative` cannot see that, and it is exactly
     the partial version of the three degeneracies in hand-off §3)
  D. Is the STRUCTURE result about structure?  (the chain was hand-built for a
     generator we wrote; the control is the same circuit on permuted features)

Everything runs on tiny circuits in a few seconds — these are meant to be run
before a batch, not after it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.baselines import ChannelZScore
from poc.time_series.circuits import DegenerateModelError, WindowPC
from poc.time_series.data import _inject, make_ad_task
from poc.time_series.explain import (
    completeness_error,
    deletion_curve,
    localization_report,
)
from poc.time_series.metrics import auroc

INJECTED = ("spike", "offset", "drift", "decouple", "desync")
MARGINAL_KINDS = ("spike", "offset", "drift")
STRUCTURAL_KINDS = ("decouple", "desync")


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — one small task and one small circuit, shared by every test
# ═══════════════════════════════════════════════════════════════════════════

WINDOW = 8          # see test_window_is_long_enough_for_decouple_to_mean_anything


@pytest.fixture(scope="module")
def task():
    # dead_channels=0: every channel carries signal, so "this channel does not
    # move the score" is unambiguous evidence of a model that ignores it
    # rather than of a channel with nothing to say.
    return make_ad_task(window=WINDOW, stride=3, seed=0, n_units=16,
                        n_channels=6, n_regimes=2, dead_channels=0,
                        inject_rate=0.5)


@pytest.fixture(scope="module")
def pc(task):
    m = WindowPC(task.window, task.n_channels, vtree_method="chain",
                 n_sum_components=4, seed=0, device="cpu")
    m.fit(task.X_train, epochs=25)
    return m


def _oracle_attribution(task) -> np.ndarray:
    a = np.zeros((len(task.X_test), task.n_channels))
    for i, chans in enumerate(task.affected_test):
        a[i, list(chans)] = 1.0
    return a


# ═══════════════════════════════════════════════════════════════════════════
# A.  Is the localisation METRIC valid?
#
# Every explanation claim in the write-up is a number out of
# `localization_report`.  Nothing has ever checked that this function scores a
# perfect attributor at 1.0 and a random one at 0.5 — and if it does not, the
# 0.902-vs-0.857 margin is measuring the metric, not the method.
# ═══════════════════════════════════════════════════════════════════════════

def test_localisation_metric_scores_an_oracle_at_one(task):
    rep = localization_report(_oracle_attribution(task), task.affected_test,
                              task.kind_test, keep=INJECTED)
    assert rep["n"] > 20, "too few localisable windows to say anything"
    assert rep["auroc"] > 0.999, f"oracle attribution scored {rep['auroc']:.3f}"
    assert rep["prec_at_k"] > 0.999


def test_localisation_metric_scores_noise_at_chance(task):
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(len(task.X_test), task.n_channels))
    rep = localization_report(noise, task.affected_test, task.kind_test,
                              keep=INJECTED)
    assert 0.40 < rep["auroc"] < 0.60, (
        f"a random attributor scores {rep['auroc']:.3f} — the metric has a "
        "prior of its own, so every reported localisation AUROC is inflated "
        "or deflated by it")


def test_localisation_metric_is_not_fooled_by_a_constant(task):
    """A method that says 'all channels equally' must not beat chance.  Ties
    are the failure mode a rank metric hides."""
    flat = np.ones((len(task.X_test), task.n_channels))
    rep = localization_report(flat, task.affected_test, task.kind_test,
                              keep=INJECTED)
    assert abs(rep["auroc"] - 0.5) < 1e-6


def test_deletion_curve_home_field_advantage_is_bounded(task, pc):
    """
    The faithfulness column is scored with the PC's own scorer, which the
    hand-off flags as home-field advantage but never quantifies.  Quantify it:
    a RANDOM attribution scored by the PC must not look good.  If it does, the
    deletion AUC is measuring the scorer's smoothness, not the explanation.
    """
    rng = np.random.default_rng(0)
    idx = (task.y_test.numpy() == 1)
    X = task.X_test[idx][:64]
    score_fn = lambda Z: pc.score(Z)

    _, auc_random = deletion_curve(
        score_fn, X, rng.normal(size=(len(X), task.n_channels)),
        task.window, task.n_channels, reference=task.X_train)
    attr = pc.typed_scores(X)["conditional"].numpy()
    _, auc_pc = deletion_curve(score_fn, X, attr, task.window,
                               task.n_channels, reference=task.X_train)

    assert auc_pc < auc_random, (
        f"PC attribution ({auc_pc:.3f}) does not beat a random one "
        f"({auc_random:.3f}) under the PC's own scorer — the faithfulness "
        "column carries no signal at all")
    # and the gap is the number that belongs next to the claim
    print(f"\n[deletion] PC {auc_pc:.3f} vs random {auc_random:.3f} "
          f"(home-field margin {auc_random - auc_pc:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# B.  Is the GENERATOR's premise true?
#
# The whole marginal-vs-conditional story rests on `decouple`/`desync` leaving
# per-channel marginals intact.  That is asserted in a docstring and has never
# been tested.  If it is false, the "structural anomaly" class is a mislabel
# and the 0.31 marginal-vs-conditional gap has a mundane explanation.
# ═══════════════════════════════════════════════════════════════════════════

def test_decouple_preserves_each_channel_marginal_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5))
    y, chans = _inject(x, "decouple", rng, strength=1.0)
    assert chans, "decouple must report the channels it touched"
    for c in range(x.shape[1]):
        assert np.allclose(np.sort(x[:, c]), np.sort(y[:, c])), (
            f"channel {c} marginal changed — `decouple` is not a purely "
            "structural anomaly, so a per-channel detector can see it and the "
            "structural claim is void")


def test_desync_channel_comes_from_the_normal_pool():
    """`desync` swaps in a donor channel: the marginal must be a NORMAL one
    (byte-identical to the donor), never a perturbed one."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5))
    donor = rng.normal(size=(12, 5))
    y, chans = _inject(x, "desync", rng, strength=1.0, donor=donor.reshape(-1))
    for c in chans:
        assert np.allclose(y[:, c], donor[:, c])
    untouched = [c for c in range(5) if c not in chans]
    for c in untouched:
        assert np.allclose(y[:, c], x[:, c])


def test_marginal_detector_is_blind_to_structural_anomalies(task):
    """
    NECESSARY CONDITION for the joint-density claim.  A per-channel z-score
    must be near chance on decouple/desync and strong on spike.  If it is not
    near chance, the "only a joint model can see these" framing is wrong; if
    it is not strong on spike, the task is not measuring what it claims.
    """
    zs = ChannelZScore(task.window, task.n_channels).fit(task.X_train)
    s = zs.score(task.X_test)
    kinds = np.array(task.kind_test)
    y = task.y_test.numpy()

    def auroc_for(group):
        keep = (y == 0) | np.isin(kinds, group)
        return auroc(s[keep], (np.isin(kinds, group)[keep]).astype(int))

    a_struct = auroc_for(list(STRUCTURAL_KINDS))
    a_spike = auroc_for(["spike"])
    print(f"\n[z-score] structural {a_struct:.3f}  spike {a_spike:.3f}")
    assert a_struct < 0.70, (
        f"a per-channel detector reaches {a_struct:.3f} on the structural "
        "anomalies — they are not structural")
    assert a_spike > 0.85, (
        f"z-score only reaches {a_spike:.3f} on spikes; the injection is too "
        "weak for the benchmark to discriminate methods")


def _temporal_signal_ratio(X: torch.Tensor, window: int, n_channels: int,
                           seed: int = 0) -> float:
    """
    How much of the window's mean |lag-1 difference| is destroyed by permuting
    its timesteps.  0 means a window is exchangeable in time, i.e. `decouple`
    produces a statistically identical window and cannot be detected by ANY
    method.  The generator docstring says this in prose and reports a
    one-off hand measurement; this is that measurement, executable.
    """
    A = np.asarray(X).reshape(-1, window, n_channels)
    rng = np.random.default_rng(seed)
    P = np.stack([a[rng.permutation(window)] for a in A])
    d_real = float(np.abs(np.diff(A, axis=1)).mean())
    d_perm = float(np.abs(np.diff(P, axis=1)).mean())
    return (d_perm - d_real) / max(d_real, 1e-12)


def test_window_is_long_enough_for_decouple_to_mean_anything(task):
    """
    VALIDITY PRECONDITION, not a model test.  `decouple` permutes a channel in
    time, so it is an anomaly only insofar as the window HAS temporal
    structure — which depends on `phi_ar` relative to `window`, a pair nobody
    re-checks when either is changed.  Measured here: window 4 leaves only
    ~21% signal (every view scores 0.51–0.55, i.e. chance), window 8 leaves
    ~56%.  Below the threshold the decouple column is vacuous and must not be
    reported at all.
    """
    r = _temporal_signal_ratio(task.X_train, task.window, task.n_channels)
    print(f"\n[temporal signal] window={task.window} destroys {r:.1%} of the "
          "lag-1 structure")
    assert r > 0.40, (
        f"permuting timesteps changes the window by only {r:.1%}: at "
        f"window={task.window} a `decouple` anomaly is nearly a normal "
        "window, so any decouple result — positive or negative — is noise")


def test_structural_view_beats_the_marginal_view_on_desync(task, pc):
    """
    The mechanism behind the explanation claim, as a per-kind contrast rather
    than an aggregate.  `desync` swaps in a normal channel from another unit:
    the marginal is in-distribution BY CONSTRUCTION, so a marginal view must
    be near chance while the conditional/structural views see it.  This is the
    reported 0.545-vs-0.851 gap, reduced to its smallest reproducible form.

    Note what is NOT asserted: that the structural view wins on marginal
    anomalies.  It does not, and should not — on a spike the marginal view is
    the right one (measured: marginal 0.881, structural 0.603).  A test that
    demanded the structural view win everywhere would be testing a claim the
    project does not make.
    """
    kinds = np.array(task.kind_test)
    keep = (kinds == "normal") | (kinds == "desync")
    y = (kinds[keep] == "desync").astype(int)
    assert y.sum() > 8, "too few desync windows"

    td = pc.typed_scores(task.X_test[keep])
    a = {k: auroc(v.sum(1).numpy(), y) for k, v in td.items()}
    print(f"\n[desync by view] marginal {a['marginal']:.3f}  "
          f"conditional {a['conditional']:.3f}  structural {a['structural']:.3f}")
    assert max(a["conditional"], a["structural"]) > a["marginal"] + 0.05, (
        "the joint views do not beat the marginal view on the one anomaly "
        "kind that is marginal-preserving by construction — the "
        "marginal/conditional distinction is not doing the work the "
        "explanation result attributes to it")


# ═══════════════════════════════════════════════════════════════════════════
# C.  Is the MODEL using the data?
#
# `assert_informative` only rejects a density that is constant in x.  A model
# that reads 2 of 6 channels passes it, produces a perfectly ordinary NLL
# curve, and cannot localise anything — the partial version of degeneracies
# #1 and #2.  Nothing in the pipeline looks for it.
# ═══════════════════════════════════════════════════════════════════════════

def _channel_sensitivity(pc, X: torch.Tensor, shift: float = 3.0) -> np.ndarray:
    """Mean |Δ −log p| when one whole channel is shifted by `shift` sd."""
    base = float(pc.score(X).mean())
    out = np.zeros(pc.n_channels)
    for c in range(pc.n_channels):
        Xp = X.clone()
        cols = [t * pc.n_channels + c for t in range(pc.window)]
        Xp[:, cols] += shift
        out[c] = abs(float(pc.score(Xp).mean()) - base)
    return out


def _blind_channels(sens: np.ndarray, frac: float = 0.10) -> list:
    """Channels whose influence on the density is under `frac` of the median
    channel's.  Relative, because the absolute nat scale moves with the window
    length, the number of channels and the fit quality — a fixed threshold
    would be exactly the kind of literal-vs-state mistake that made the
    `@floor` diagnostic blind (hand-off §A.6)."""
    med = float(np.median(sens))
    return [c for c, v in enumerate(sens) if v < frac * med]


def test_density_reads_every_channel(task, pc):
    sens = _channel_sensitivity(pc, task.X_train[:128])
    print("\n[channel sensitivity] " +
          "  ".join(f"c{c}:{v:.1f}" for c, v in enumerate(sens)))
    dead = _blind_channels(sens)
    assert not dead, (
        f"channels {dead} move the density by <10% of the median channel "
        "under a 3σ shift: the circuit is not reading them.  The score still "
        "varies with x, so `assert_informative` passes and every per-channel "
        "claim about these channels is empty.")


def test_channel_sensitivity_catches_a_partially_degenerate_model(task):
    """
    The diagnostic must actually fire.  Build the failure on purpose — a
    circuit whose leaves on the last two channels are frozen wide — and
    confirm both halves of the guardrail:

      (a) `assert_informative` now REFUSES it (fixed 2026-08-05), and
      (b) the constant-score check ALONE still would not have — which is why
          the extension was needed and what the coverage gap looked like.
    """
    m = WindowPC(task.window, task.n_channels, vtree_method="chain",
                 n_sum_components=4, seed=0, device="cpu")
    m.fit(task.X_train, epochs=8)

    from src.probabilistic_circuits import GaussianLeaf
    blind = {task.n_channels - 1, task.n_channels - 2}
    with torch.no_grad():
        for mod in m.pc.modules():
            if isinstance(mod, GaussianLeaf) and \
                    (mod.feature_idx % task.n_channels) in blind:
                mod.log_sigma.fill_(6.0)      # ~400 sd: the leaf says nothing

    # the compiled evaluator shadows the DAG: without this the sabotage above
    # is silently ignored.  See test_dag_edits_do_not_reach_the_compiled_copy
    # in tests/test_experiment_hygiene.py — the same trap catches real edits.
    m.pc.use_recursive()

    # (b) the old check — score sd only — sees nothing wrong
    sd = m.assert_informative(task.X_train, check_channels=False)
    assert sd > 1e-3, "the sabotaged model still has a perfectly varying score"

    # (a) the full check refuses it, and names the right channels
    with pytest.raises(DegenerateModelError) as exc:
        m.assert_informative(task.X_train)
    for c in blind:
        assert str(c) in str(exc.value)

    sens = _channel_sensitivity(m, task.X_train[:128])
    found = set(_blind_channels(sens))
    print("\n[sabotage] sensitivity " +
          "  ".join(f"c{c}:{v:.2f}" for c, v in enumerate(sens)))
    assert found == blind, (
        f"the sensitivity diagnostic flagged {sorted(found)} but the channels "
        f"deliberately blinded were {sorted(blind)} — the diagnostic is not "
        "reliable enough to trust the test above")


def test_completeness_is_a_theorem_not_an_estimate(task, pc):
    err = completeness_error(pc, task.X_test[:64])
    assert err["max_residual_nats"] < 1e-3, err
    print(f"\n[completeness] max residual {err['max_residual_nats']:.2e} nats")


def test_typed_decomposition_is_internally_consistent(task, pc):
    td = pc.typed_scores(task.X_test[:64])
    assert torch.allclose(td["structural"],
                          td["conditional"] - td["marginal"], atol=1e-5)
    # a conditional that is uniformly equal to its marginal means the circuit
    # learned no cross-channel dependence at all — a silently factorised model
    spread = float((td["conditional"] - td["marginal"]).abs().mean())
    assert spread > 1e-2, (
        f"conditional and marginal differ by {spread:.2e} nats on average: "
        "the circuit has collapsed to a product of per-channel marginals, "
        "which no aggregate metric would reveal")


# ═══════════════════════════════════════════════════════════════════════════
# D.  Is the STRUCTURE result about structure?
#
# "chain (HMM-shaped) wins on AUROC *and* likelihood" was measured on a
# generator we wrote, whose latent driver is an AR(1) chain.  The write-up
# already concedes the chain "was arguably told the answer by our own
# generator".  Two controls settle it, both holding CAPACITY EXACTLY FIXED —
# same circuit, same parameter count, only the variable->position map changes:
#
#   1. permute the TIMESTEP BLOCKS   -> isolates temporal ORDER
#   2. permute ALL FEATURES          -> also destroys same-timestep BLOCKING
#
# and each is run again on data whose temporal structure has been destroyed,
# where a real temporal effect must disappear.
# ═══════════════════════════════════════════════════════════════════════════

def _held_out_nll(X_tr: torch.Tensor, X_te: torch.Tensor, window: int,
                  n_channels: int, seed: int = 0) -> float:
    m = WindowPC(window, n_channels, vtree_method="chain", n_sum_components=4,
                 seed=seed, device="cpu")
    m.fit(X_tr, epochs=20)
    return float(m.score(X_te).mean())


def _mean_nll(X_tr, X_te, task, seeds=(0, 1)) -> float:
    return float(np.mean([_held_out_nll(X_tr, X_te, task.window,
                                        task.n_channels, seed=s)
                          for s in seeds]))


def _permute_timestep_blocks(X: torch.Tensor, window: int, n_channels: int,
                             seed: int = 1) -> torch.Tensor:
    """Reorder the TIMESTEPS in the feature layout, keeping each timestep's
    channels contiguous.  The chain now models the series in scrambled order;
    everything else about the circuit is unchanged."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(window, generator=g)
    return X.reshape(-1, window, n_channels)[:, p, :].reshape(len(X), -1)


def _permute_all_features(X: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """Destroy the layout entirely: timestep order AND same-timestep blocking."""
    g = torch.Generator().manual_seed(seed)
    return X[:, torch.randperm(X.shape[1], generator=g)]


def _destroy_time_structure(X: torch.Tensor, window: int, n_channels: int,
                            seed: int = 0) -> torch.Tensor:
    """Permute the timesteps of each window independently: the cross-channel
    structure WITHIN each timestep survives, the temporal order does not."""
    g = torch.Generator().manual_seed(seed)
    A = X.reshape(-1, window, n_channels).clone()
    for i in range(len(A)):
        A[i] = A[i][torch.randperm(window, generator=g)]
    return A.reshape(len(A), -1)


@pytest.fixture(scope="module")
def split(task):
    n = len(task.X_train)
    return task.X_train[: int(0.75 * n)], task.X_train[int(0.75 * n):]


@pytest.fixture(scope="module")
def layout_penalties(task, split):
    """(temporal-order penalty, blocking penalty) on real and on
    time-destructured data, in nats of held-out NLL, 2 seeds each."""
    tr, te = split
    W, C = task.window, task.n_channels
    out = {}
    for label, (a, b) in {
            "structured": (tr, te),
            "destructured": (_destroy_time_structure(tr, W, C, seed=2),
                             _destroy_time_structure(te, W, C, seed=3))}.items():
        base = _mean_nll(a, b, task)
        order = _mean_nll(_permute_timestep_blocks(a, W, C),
                          _permute_timestep_blocks(b, W, C), task)
        block = _mean_nll(_permute_all_features(a),
                          _permute_all_features(b), task)
        out[label] = {"base": base, "order": order - base, "block": block - base}
    print("\n[layout] penalty in nats vs the correct layout")
    for k, v in out.items():
        print(f"  {k:13s} base {v['base']:7.2f}  timestep-order {v['order']:+6.2f}"
              f"  channel-blocking {v['block']:+6.2f}")
    return out


def test_chain_advantage_is_blocking_not_temporal_order(layout_penalties):
    """
    THE control the structure ablation never had.  If the chain wins because
    it is HMM-shaped, scrambling the timestep order must cost something.

    Measured: it costs ~+0.07 nats (nothing), while scrambling which channels
    sit together costs ~+5 nats.  So the chain's advantage over the other
    region graphs is that it keeps each timestep's channels contiguous — a
    BLOCKING effect — and not that it models time as a chain.  That changes
    what the ablation table means: the row ordering is about variable
    grouping, and "the chain is HMM-shaped" is not the explanation.
    """
    p = layout_penalties["structured"]
    assert p["block"] > 5 * max(p["order"], 0.1), (
        f"timestep-order penalty {p['order']:+.2f} nats vs channel-blocking "
        f"penalty {p['block']:+.2f} nats — the two are now comparable, so the "
        "temporal-chain reading of the structure ablation has become "
        "defensible and this test's premise needs re-deriving")


def test_layout_penalty_survives_on_structureless_data(layout_penalties):
    """
    The same measurement on data with NO temporal structure left.  The
    blocking penalty is just as large there (~+6.5 nats), which is what makes
    it inadmissible as evidence about the data's temporal dependence: it is a
    property of the circuit's layout, present whether or not the signal is.
    """
    s, d = layout_penalties["structured"], layout_penalties["destructured"]
    assert d["block"] > 0.5 * s["block"], (
        "the blocking penalty collapsed on destructured data, which would "
        "make it a genuine signal test — good news, but it contradicts the "
        "recorded measurement and the conclusion above must be re-derived")
    assert abs(d["order"]) < 1.0, (
        f"timestep order matters by {d['order']:+.2f} nats on data whose "
        "temporal structure was destroyed — the control itself is broken")
