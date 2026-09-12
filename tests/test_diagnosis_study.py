"""Scientific contracts for the new circuit diagnosis study (no accuracy tuning)."""
import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poc.time_series.circuits import WindowPC
from poc.time_series.config import DEFAULTS
from poc.time_series.diagnosis import (validation_partitions, one_per_unit,
    partition_unit_counts, conservative_threshold, score_views,
    localization_metrics, paired_engine_bootstrap)
from poc.time_series.diagnosis_controls import (control_covariance,
    make_control_task, independent_replacement, gaussian_oracle_map)
from poc.time_series.metrics import auroc
from poc.time_series.relational import MaskCalibrator
from src.probabilistic_circuits import (QueryCost, GaussianLeaf, SumNode,
                                      BlockStructureError, eval_log_prob)


def model(channels=3, **kwargs):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(48, channels * 2, generator=generator)
    pc = WindowPC(2, channels, vtree_method="channel_blocked", n_sum_components=2,
                  device="cpu", seed=3, **kwargs)
    pc.build(x)
    # Deliberately differentiated components: identities must hold on a model
    # with dependence rather than agreeing vacuously at a product distribution.
    with torch.no_grad():
        for node in pc.pc.modules():
            if isinstance(node, GaussianLeaf):
                node.mu.copy_(torch.randn((), generator=generator) * 2)
            if isinstance(node, SumNode):
                node.weights.copy_(torch.randn(node.weights.shape, generator=generator) * 2)
    return pc, x


@pytest.mark.parametrize("mixture", [False, True])
def test_capacity_and_mixture_preserve_normalization_and_all_masks(mixture):
    pc, x = model(boundary_components=3, upper_components=3 if mixture else 4,
                  channel_mixture=mixture)
    pc.pc.validate()
    assert abs(float(pc.pc.log_partition())) < 1e-5
    assert [len(us) for us in pc.relational().units] == [3, 3, 3]
    full = pc.diagnosis_map(x[:8], backend="fast")
    assert float(full["R"].abs().max()) > 1e-3
    for pattern in itertools.product([False, True], repeat=3):
        fast = pc.diagnosis_map(x[:8], pattern, backend="fast")
        oracle = pc.diagnosis_map(x[:8], pattern, backend="oracle")
        for key in ("R", "conditional", "marginal", "log_px"):
            assert torch.allclose(fast[key], oracle[key], atol=1e-4, equal_nan=True)


def test_shared_latent_mixture_matches_independent_component_formula():
    pc, x = model(boundary_components=3, channel_mixture=True)
    rc = pc.relational()
    # Boundary traversal order is not the latent-component index order.
    regions = [pc.pc.region_graph]
    by_scope = {}
    while regions:
        region = regions.pop()
        by_scope[frozenset(region.scope)] = region
        for part in region.partitions:
            regions.extend(part)
    vals = [torch.stack([eval_log_prob(u, x[:8]) for u in
             pc.pc._regions[id(by_scope[frozenset(range(c, 6, 3))])]], dim=1)
            for c in range(3)]
    weights = pc.pc.root.log_weights()
    for mask in (torch.tensor([1, 1, 1], dtype=torch.bool),
                 torch.tensor([1, 0, 1], dtype=torch.bool)):
        components = sum(v for c, v in enumerate(vals) if mask[c])
        analytic = torch.logsumexp(components + weights, dim=1)
        assert torch.allclose(analytic, pc.diagnosis_map(x[:8], mask)["log_px"], atol=1e-4)


def test_generic_queries_support_unblocked_structures_and_per_row_masks():
    x = torch.randn(24, 6)
    pc = WindowPC(2, 3, vtree_method="time", n_sum_components=2, device="cpu")
    pc.build(x)
    masks = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=torch.bool)
    r = pc.diagnosis_map(x[:3], masks)
    for i, mask in enumerate(masks):
        expected = -pc.score_with_missing(x[i:i + 1], torch.nonzero(~mask).flatten().tolist())
        assert torch.allclose(r["log_px"][i:i+1], expected, atol=1e-5)
    assert torch.isnan(r["R"][~masks]).all()


def test_blocking_controls_refuse_split_channels():
    pc = WindowPC(2, 3, vtree_method="time", boundary_components=3)
    with pytest.raises(BlockStructureError):
        pc.build(torch.randn(12, 6))


def test_mask_replication_work_grows_even_inside_one_chunk():
    pc, x = model()
    rc = pc.relational()
    vals = rc.block_log_values(x[:5])
    costs = []
    for count in (1, 4):
        cost = QueryCost()
        rc.masked_log_prob(block_vals=vals, masks=torch.ones(count, 3, dtype=torch.bool), cost=cost)
        costs.append(cost)
    a, b = costs
    assert a.passes == b.passes
    assert b.node_evaluations == 4 * a.node_evaluations
    assert b.boundary_elements == 4 * a.boundary_elements
    assert b.output_elements == 4 * a.output_elements
    assert b.peak_boundary_bytes == 4 * a.peak_boundary_bytes


def test_mask_chunking_does_not_change_work_or_values():
    pc, x = model()
    rc = pc.relational()
    vals = rc.block_log_values(x[:5])
    masks = torch.tensor(list(itertools.product([False, True], repeat=3)))
    a, b = QueryCost(), QueryCost()
    xa = rc.masked_log_prob(block_vals=vals, masks=masks, chunk=1, cost=a)
    xb = rc.masked_log_prob(block_vals=vals, masks=masks, chunk=8, cost=b)
    assert torch.allclose(xa, xb, atol=1e-5)
    assert a.node_evaluations == b.node_evaluations
    assert b.peak_boundary_bytes > a.peak_boundary_bytes


def test_two_sensor_scores_are_ambiguous_and_single_sensor_has_no_relation():
    pc, x = model(channels=2)
    r = pc.diagnosis_map(x[:10])
    assert torch.allclose(r["R"][:, 0], r["R"][:, 1], atol=1e-5)
    one = pc.diagnosis_map(x[:10], [True, False])
    assert torch.equal(one["R"][:, 0], torch.zeros(10))
    loc = localization_metrics(r["R"].numpy(), [[0]] * 10, [True, True], np.ones(10, dtype=bool))
    assert loc["loc_n_ambiguous"] == 10
    assert loc["end_to_end_unique_top1"] == 0


def test_finite_calibration_does_not_fabricate_an_unavailable_quantile():
    assert np.isinf(conservative_threshold([1.], .05))
    cal = MaskCalibrator(None, alpha=.05)
    assert np.isinf(cal._quantile(np.array([[1., 2.]]))).all()
    assert conservative_threshold(np.arange(19), .05) == 18


def test_max_calibration_controls_a_different_event_than_channel_thresholds():
    pc, x = model()
    cal = MaskCalibrator(pc.relational(), alpha=.1).fit(x[:1], {"full": torch.ones(3, dtype=torch.bool)})
    assert not cal.familywise_alarms(np.full((5, 3), 1e20), "full").any()
    with pytest.raises(ValueError, match="does not match"):
        cal.report(x[1:3], {"full": torch.tensor([1, 1, 0])})


def test_validation_and_calibration_units_are_disjoint():
    units = torch.arange(12).repeat_interleave(3)
    task = SimpleNamespace(unit_val=units, X_val=torch.randn(36, 6))
    p = validation_partitions(task, 31)
    groups = [set(units[idx].tolist()) for idx in p.values()]
    assert all(not a & b for a, b in itertools.combinations(groups, 2))
    assert set.union(*groups) == set(range(12))
    idx = one_per_unit(p["calibration"], units)
    assert len(idx) == len(set(units[idx].tolist()))
    with pytest.raises(ValueError, match="three"):
        validation_partitions(SimpleNamespace(unit_val=torch.tensor([0, 1]), X_val=torch.randn(2, 6)))


def test_split_weights_move_engines_without_breaking_disjointness():
    """More calibration engines is the only lever on the operating point.

    One window per engine at alpha 0.10 cannot resolve a false-alarm rate
    finer than 1/n, and the equal three-way split left n ~ 10 on FD001 (q95
    0.33 against a nominal 0.10).  Re-weighting has to buy calibration
    engines, and must not buy them by sharing engines between partitions.
    """
    units = torch.arange(20).repeat_interleave(3)
    task = SimpleNamespace(unit_val=units, X_val=torch.randn(60, 6))
    equal = validation_partitions(task, 7)
    heavy = validation_partitions(task, 7, (1, 2, 1))
    def engines(p):
        return {k: set(units[idx].tolist()) for k, idx in p.items()}
    e, h = engines(equal), engines(heavy)
    assert len(h["calibration"]) > len(e["calibration"])
    for groups in (e, h):
        assert all(not a & b for a, b in itertools.combinations(groups.values(), 2))
        assert set.union(*groups.values()) == set(range(20))
        assert all(groups.values())
    # Same seed, same split: the weights are a protocol choice, not a draw.
    assert engines(validation_partitions(task, 7, (1, 2, 1))) == h


def test_partition_counts_keep_every_partition_nonempty_and_reject_bad_weights():
    assert partition_unit_counts(30, (1, 1, 1)) == [10, 10, 10]
    assert sum(partition_unit_counts(31, (1, 2, 1))) == 31
    # A weight small enough to round to nothing still gets an engine: an empty
    # calibration or evaluation set is not a smaller experiment, it is none.
    assert min(partition_unit_counts(5, (0.01, 10, 0.01))) >= 1
    for bad in [(1, 1), (1, 0, 1), (1, -1, 1), (1, float("nan"), 1)]:
        with pytest.raises(ValueError, match="weights"):
            partition_unit_counts(30, bad)
    with pytest.raises(ValueError, match="three"):
        partition_unit_counts(2, (1, 1, 1))


def test_fit_helper_uses_only_explicit_checkpoint_half(monkeypatch):
    from poc.time_series import pipeline
    task, _ = make_control_task(window=2, channels=3, n_train=16, n_val=12)
    checkpoint = task.X_val[:4]
    real_fit = WindowPC.fit
    seen = []
    def fit(self, *args, **kwargs):
        seen.append(kwargs["X_val"])
        return real_fit(self, *args, **kwargs)
    monkeypatch.setattr(WindowPC, "fit", fit)
    monkeypatch.setattr(WindowPC, "assert_informative", lambda *args, **kwargs: 1.)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["device"] = "cpu"
    cfg["model"].update(vtree="channel_blocked", K=2, epochs=1)
    log = SimpleNamespace(history=lambda *a: None, info=lambda *a: None)
    pipeline._fit_window_pc(cfg, task, 0, log, X_checkpoint=checkpoint)
    assert seen[0] is checkpoint


@pytest.mark.parametrize("mixture", [False, True])
def test_conditional_training_and_compilation_preserve_exact_queries(mixture):
    task, _ = make_control_task(window=2, channels=3, n_train=24, n_val=12)
    pc = WindowPC(2, 3, vtree_method="channel_blocked", n_sum_components=2,
                  boundary_components=3, channel_mixture=mixture, device="cpu")
    pc.fit(task.X_train, X_val=task.X_val, epochs=3, batch_size=12,
           conditional_weight=.2, select_metric="objective")
    assert pc.compiled is not None
    assert pc.optimizer_steps == 6
    assert len(pc.val_objective_history) == 3
    assert np.isfinite(pc.objective_history).all()
    assert not np.allclose(pc.objective_history, pc.history)
    assert pc.best_epoch == int(np.argmin(pc.val_objective_history))
    pc.pc.validate()
    assert abs(float(pc.pc.log_partition())) < 1e-5
    for mask in ([1, 1, 1], [1, 0, 1]):
        fast = pc.diagnosis_map(task.X_val, mask, backend="fast")
        slow = pc.diagnosis_map(task.X_val, mask, backend="oracle")
        assert torch.allclose(fast["R"], slow["R"], atol=1e-4, equal_nan=True)


def test_early_stopping_records_actual_updates():
    task, _ = make_control_task(window=2, channels=3, n_train=24, n_val=12)
    pc = WindowPC(2, 3, vtree_method="channel_blocked", n_sum_components=2, device="cpu")
    pc.fit(task.X_train, X_val=task.X_val, epochs=12, lr=0,
           patience=2, min_epochs=4, batch_size=12)
    assert pc.stopped_early
    assert len(pc.history) == 4
    assert pc.optimizer_steps == 8


def test_known_law_control_preserves_marginals_and_loses_hidden_fault_evidence():
    task, covariance = make_control_task(window=3, channels=3, n_train=32, n_val=120)
    bad = independent_replacement(task.X_val, covariance, 3, seed=90)
    assert torch.equal(task.X_val[:, 1::3], bad[:, 1::3])
    # Hide the replaced channel: the remaining evidence is exactly unchanged.
    a = gaussian_oracle_map(task.X_val, covariance, 3, [False, True, True])
    b = gaussian_oracle_map(bad, covariance, 3, [False, True, True])
    assert torch.equal(a["log_px"], b["log_px"])
    assert torch.allclose(a["R"], b["R"], equal_nan=True)
    # Observe only the replaced channel: the relation is identically zero.
    one = gaussian_oracle_map(bad, covariance, 3, [True, False, False])
    assert torch.equal(one["R"][:, 0], torch.zeros(len(bad), dtype=torch.double))


def test_oracle_independence_is_a_negative_control():
    covariance = control_covariance(2, 3, cross=0)
    result = gaussian_oracle_map(torch.randn(12, 6), covariance, 3)
    assert float(result["R"].abs().max()) < 1e-10


def test_known_oracle_has_power_when_relational_evidence_is_present():
    task, cov = make_control_task(window=4, channels=3, n_train=8, n_val=256,
                                  seed=912, cross=.95)
    bad = independent_replacement(task.X_val, cov, 3, seed=31)
    result = gaussian_oracle_map(torch.cat((task.X_val, bad)), cov, 3)
    assert auroc(result["R"][:, 0], np.r_[np.zeros(256), np.ones(256)]) > .8


def test_cluster_bootstrap_identical_scores_has_zero_paired_difference():
    scores = np.arange(12)
    labels = np.tile([0, 1], 6)
    units = np.repeat(np.arange(6), 2)
    out = paired_engine_bootstrap(scores, scores, labels, units, reps=20)
    assert out["auroc_delta"] == out["ci_low"] == out["ci_high"] == 0
