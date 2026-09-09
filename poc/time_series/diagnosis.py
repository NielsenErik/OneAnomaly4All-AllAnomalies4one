"""Circuit diagnosis study: exact queries, independent units, paired corruptions.

This stage is exploratory VALIDATION evaluation, never a reinterpretation of
the original structure gate. It does not touch the official test fleet.
"""
from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import torch

from .data import contaminate_windows
from .metrics import average_precision, auroc, detection_report
from .relational import mask_library


def validation_partitions(task, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Three engine-disjoint index sets. Never fall back to overlapping windows."""
    units = getattr(task, "unit_val", None)
    if units is None or len(units) != len(task.X_val):
        raise ValueError("diagnosis requires validation unit ids aligned with X_val")
    units = np.asarray(units)
    unique = np.unique(units)
    if len(unique) < 3:
        raise ValueError("at least three validation engines are required")
    groups = np.array_split(np.random.default_rng(seed).permutation(unique), 3)
    return {name: torch.from_numpy(np.flatnonzero(np.isin(units, group)))
            for name, group in zip(("checkpoint", "calibration", "evaluation"), groups)}


def one_per_unit(indices, units, seed=0) -> torch.Tensor:
    """Random window from each engine, selected independently of model scores."""
    indices, units = np.asarray(indices), np.asarray(units)
    rng = np.random.default_rng(seed)
    return torch.tensor([int(rng.choice(indices[units[indices] == unit]))
                         for unit in np.unique(units[indices])])


def conservative_threshold(scores, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty scores and alpha in (0, 1) required")
    k = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.sort(values)[k - 1]) if k <= len(values) else float("inf")


def score_views(result: Dict) -> Dict[str, np.ndarray]:
    """Higher means more anomalous; no data-fitted score fusion."""
    mask = result["mask"].cpu().numpy()
    if not mask.any(axis=1).all():
        raise ValueError("detection scores require at least one observed channel")
    m = result["marginal"].cpu().numpy()
    cd = result["conditional"].cpu().numpy()
    r = result["R"].cpu().numpy()
    if not all(np.isfinite(a[mask]).all() for a in (m, cd, r)):
        raise FloatingPointError("non-finite score on an observed channel")
    return {"joint": -result["log_px"].cpu().numpy(),
            "independent": np.where(mask, m, 0).sum(1),
            "marginal_max": np.where(mask, m, -np.inf).max(1),
            "conditional_max": np.where(mask, cd, -np.inf).max(1),
            "relational_max": np.where(mask, r, -np.inf).max(1)}


def dependence_diagnostics(result: Dict) -> Dict[str, float]:
    views = score_views(result)
    signed = views["independent"] - views["joint"]
    return {"empirical_log_dependence_mean": float(signed.mean()),
            "empirical_log_dependence_abs_mean": float(np.abs(signed).mean()),
            "empirical_log_dependence_sd": float(signed.std()),
            "empirical_log_dependence_q05": float(np.quantile(signed, .05)),
            "empirical_log_dependence_q95": float(np.quantile(signed, .95))}


def localization_metrics(attr, affected, observed, alarms, tie_tolerance=1e-4):
    """No forced winner among tied channels; report observable targets only.

    An observed target is not necessarily identifiable. Ambiguity is reported
    empirically; this function does not infer a causal fault graph from scores.
    End-to-end unique top-1 includes EVERY corrupted example as denominator.
    """
    aps, unique_hits, ambiguous = [], [], 0
    absent, all_targets, n_faults, successes = 0, 0, 0, 0
    obs = np.flatnonzero(np.asarray(observed))
    for i, truth in enumerate(affected):
        if not truth:
            continue
        n_faults += 1
        y = np.isin(obs, truth)
        if not y.any():
            absent += 1
            continue
        if y.all():
            all_targets += 1
        else:
            aps.append(average_precision(np.asarray(attr)[i, obs], y))
        values = np.asarray(attr)[i, obs]
        winners = np.flatnonzero(values >= values.max() - tie_tolerance)
        if len(winners) != 1 or len(obs) < 2:
            ambiguous += 1
            continue
        hit = bool(y[winners[0]])
        unique_hits.append(hit)
        successes += int(hit and alarms[i])
    return {"loc_ap": float(np.mean(aps)) if aps else float("nan"),
            "loc_ap_n": len(aps), "loc_n_no_observed_target": absent,
            "loc_n_all_observed_are_targets": all_targets,
            "loc_n_ambiguous": ambiguous, "loc_n_unique": len(unique_hits),
            "loc_unique_top1_accuracy": float(np.mean(unique_hits)) if unique_hits else float("nan"),
            "end_to_end_unique_top1": successes / n_faults if n_faults else float("nan")}


def paired_engine_bootstrap(a, b, labels, units, reps=1000, seed=0):
    """Paired AUROC difference a-b, cluster bootstrap on independent engines.

    Repeated training seeds are a separate axis, not additional engines.
    """
    a, b, labels, units = map(np.asarray, (a, b, labels, units))
    if not (a.shape == b.shape == labels.shape == units.shape) or a.ndim != 1:
        raise ValueError("paired scores, labels and units must be aligned vectors")
    if reps < 1 or len(np.unique(units)) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap requires finite scores, >=2 units and positive reps")
    unique = np.unique(units)
    groups = [np.flatnonzero(units == u) for u in unique]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        delta = auroc(a[idx], labels[idx]) - auroc(b[idx], labels[idx])
        if np.isfinite(delta):
            draws.append(delta)
    if not draws:
        raise ValueError("no bootstrap replicate contained both classes")
    return {"auroc_delta": auroc(a, labels) - auroc(b, labels),
            "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)),
            "bootstrap_reps_used": len(draws), "n_units": len(unique)}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def stage_diagnosis(cfg, seed, log):
    from .pipeline import prepare_task, _fit_window_pc, _row

    ev = cfg["eval"]
    covariance = None
    if ev.get("diagnosis_control", False):
        from .diagnosis_controls import make_control_task
        task, covariance = make_control_task(window=cfg["dataset"]["window"],
            channels=cfg["dataset"]["channels"], n_train=int(ev.get("control_train", 512)),
            n_val=int(ev.get("control_val", 300)), seed=seed,
            cross=float(ev.get("control_cross", .8)), temporal=float(ev.get("control_temporal", .6)))
    else:
        _, task = prepare_task(cfg, seed, log, "ad")
    split_seed = int(ev.get("diagnosis_split_seed", 701))
    parts = validation_partitions(task, split_seed)
    cal_idx = one_per_unit(parts["calibration"], task.unit_val, split_seed + 1)
    eval_idx = parts["evaluation"]
    cap = int(ev.get("diagnosis_max_windows", 512))
    if cap > 0 and len(eval_idx) > cap:
        # Uniform random cap, not the first engines. Preserve indices for replay.
        rng = np.random.default_rng(split_seed + 2)
        eval_idx = eval_idx[torch.tensor(np.sort(rng.choice(len(eval_idx), cap, replace=False)))]
    Xclean = task.X_val[eval_idx]
    Xcal = task.X_val[cal_idx]
    protocol = {"version": "diagnosis_v1", "evaluation_split": "exploratory_validation",
                "calibration_object": "one_random_window_per_engine",
                "mask_assumption": "externally_fixed_independent_of_current_observation",
                "donor_policy": "true_nominal_independent" if covariance is not None else "training_nominal_unmatched_context",
                "independent_unit": "window" if covariance is not None else "engine",
                "corruption_seed": seed + 12000,
                "split_seed": split_seed,
                "indices": {k: v.tolist() for k, v in parts.items()},
                "unit_ids": {k: np.unique(np.asarray(task.unit_val)[v]).tolist() for k, v in parts.items()},
                "calibration_indices_used": cal_idx.tolist(),
                "evaluation_indices_used": eval_idx.tolist()}
    log.artifact_json("diagnosis_protocol", protocol)
    pc = _fit_window_pc(cfg, task, seed, log, X_checkpoint=task.X_val[parts["checkpoint"]])
    pc.pc.validate()
    partition = float(pc.pc.log_partition().detach())
    if not np.isfinite(partition) or abs(partition) > 1e-4:
        raise AssertionError(f"circuit not normalized: log Z={partition}")
    log.result(_row(cfg, "diagnosis", "model", log_partition=partition,
        **pc.size(), optimizer_steps=pc.optimizer_steps, epochs_run=len(pc.history),
        best_epoch=pc.best_epoch, stopped_early=pc.stopped_early,
        selection_metric=pc.select_metric, checkpoint_nll=pc.best_val_nll,
        boundary_widths=[len(us) for us in pc.relational().units] if pc.is_channel_blocked else []))
    Xbad, _, kinds_bad, affected_bad = contaminate_windows(Xclean,
        task.window, task.n_channels, inject_rate=1.0,
        strength=float(cfg["dataset"]["strength"]), kinds=tuple(ev["kinds"]),
        donors=task.X_train, seed=seed + 12000)
    if covariance is not None:
        from .diagnosis_controls import independent_replacement
        Xbad = independent_replacement(Xclean, covariance, task.n_channels, seed + 12000)
        kinds_bad, affected_bad = ["independent_block"] * len(Xbad), [[0] for _ in Xbad]
    X = torch.cat((Xclean, Xbad))
    labels = np.r_[np.zeros(len(Xclean)), np.ones(len(Xbad))].astype(int)
    kinds = ["normal"] * len(Xclean) + list(kinds_bad)
    affected = [[] for _ in Xclean] + list(affected_bad)
    units = np.tile(np.asarray(task.unit_val)[eval_idx], 2)
    lib = mask_library(task.n_channels, task.channel_groups,
        ks=tuple(ev["mask_ks"]), n_per_k=int(ev["masks_per_k"]), seed=split_seed)
    if ev.get("diagnosis_single_sensor", True):
        lib["single_sensor_0"] = torch.arange(task.n_channels) == 0
    rows = []
    alpha = float(ev.get("diagnosis_alpha", .1))
    for mask_idx, (name, mask) in enumerate(lib.items()):
        log.info(f"  diagnosis: {name}, {int(mask.sum())} observed, {len(X)} paired windows")
        backend = "fast" if pc.is_channel_blocked else "oracle"
        if pc.is_channel_blocked:
            probe = X[:min(len(X), int(ev["oracle_check_windows"]))]
            fast = pc.diagnosis_map(probe, mask, backend="fast")
            oracle = pc.diagnosis_map(probe, mask, backend="oracle")
            errors = []
            for k in ("marginal", "conditional", "R", "log_px"):
                a, b = fast[k], oracle[k]
                valid = torch.ones_like(a, dtype=torch.bool) if k == "log_px" else fast["mask"]
                if not bool(torch.isfinite(a[valid]).all() and torch.isfinite(b[valid]).all()):
                    raise AssertionError("non-finite observed exact-query result")
                errors.append(float((a[valid] - b[valid]).abs().max()))
            err = max(errors)
            if err > float(ev["oracle_tolerance"]):
                raise AssertionError(f"masked exact-query check failed: {err}")
        else:
            err = 0.0
        # Warmup includes cached block constants; report setup separately.
        _sync(pc.device)
        t0 = time.perf_counter()
        pc.diagnosis_map(X[:min(4, len(X))], mask, backend=backend)
        _sync(pc.device)
        setup_s = time.perf_counter() - t0
        _sync(pc.device)
        t0 = time.perf_counter()
        result = pc.diagnosis_map(X, mask, backend=backend)
        _sync(pc.device)
        query_s = time.perf_counter() - t0
        calibration = pc.diagnosis_map(Xcal, mask, backend=backend)
        views, cv = score_views(result), score_views(calibration)
        evidence = {"labels": labels, "units": units, "kinds": np.asarray(kinds),
                    "observed": mask.numpy(), "pair_index": np.tile(eval_idx.numpy(), 2),
                    "input_windows": X.numpy(),
                    "affected": np.asarray([[c in a for c in range(task.n_channels)] for a in affected]),
                    **views}
        log.artifact_npz(f"diagnosis_mask{mask_idx}", **evidence)
        log.result(_row(cfg, "diagnosis", f"query {name}", mask=name,
            backend=backend, oracle_max_error=err, query_s=query_s,
            warmup_s=setup_s, n=len(X), n_observed=int(mask.sum()),
            **{f"cost_{k}": v for k, v in result["cost"].as_dict().items()},
            **dependence_diagnostics({k: v[:len(Xclean)] if isinstance(v, torch.Tensor) else v
                                      for k, v in result.items()})))
        for score, values in views.items():
            threshold = conservative_threshold(cv[score], alpha)
            alarms = values > threshold
            attr_key = {"joint": "conditional", "independent": "marginal",
                        "marginal_max": "marginal", "conditional_max": "conditional",
                        "relational_max": "R"}[score]
            loc = localization_metrics(result[attr_key].numpy(), affected, mask, alarms)
            row = _row(cfg, "diagnosis", f"{score} | {name}", score=score, mask=name,
                localization_score=attr_key, alpha=alpha, calibration_n=len(Xcal),
                threshold=threshold if np.isfinite(threshold) else None,
                threshold_infinite=not np.isfinite(threshold),
                score_range=float(np.ptp(values)),
                numerically_flat=bool(np.ptp(values) < float(ev["oracle_tolerance"])),
                fpr=float(alarms[labels == 0].mean()), power=float(alarms[labels == 1].mean()),
                **detection_report(values, labels, kinds), **loc)
            log.result(row)
            rows.append(row)
        if covariance is not None:
            from .diagnosis_controls import gaussian_oracle_map
            truth = gaussian_oracle_map(X, covariance, task.n_channels, mask)
            oracle_views = score_views(truth)
            for score, values in oracle_views.items():
                log.result(_row(cfg, "diagnosis", f"known-law {score} | {name}",
                    mask=name, score=score, reference="known_generating_distribution",
                    **detection_report(values, labels, kinds)))
    log.info("  Diagnosis completed on validation engines only; no confirmatory PASS/FAIL inferred.")
    return {"protocol": protocol, "rows": rows, "mask_artifacts": list(lib)}
