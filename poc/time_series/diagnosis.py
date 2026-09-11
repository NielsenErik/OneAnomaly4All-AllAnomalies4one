"""Circuit diagnosis study: exact queries, independent units, paired corruptions.

This stage is exploratory VALIDATION evaluation, never a reinterpretation of
the original structure gate. It does not touch the official test fleet.

What one run produces, and which roadmap step each piece answers:

  step 2  the circuit's masked diagnosis map beside the known-law oracle, with
          the oracle's own detection AND the circuit's error against it
  step 3  actual optimiser updates, selected epoch, wall time and parameter
          counts, so an undertrained arm is distinguishable from a weak one
  step 4  fitted Gaussian / low-rank / GMM comparators answering the SAME
          masked queries on the SAME splits, corruptions and masks
  step 5  matched donors, recorded donor identities, and a target-channel-only
          shortcut audit fitted on separate development units
  step 6  missingness regimes generated from written-down mechanisms, witness
          availability, abstention, and repeated independent-unit trials that
          report false-alarm uncertainty rather than one point estimate
  step 7  per-method artifacts carrying the attribution matrices and alarms, so
          localisation and end-to-end success can be compared PAIRED offline

Nothing here decides a PASS/FAIL. The historical structure gate stays failed
until a frozen confirmation protocol says otherwise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import contaminate_windows
from .metrics import average_precision, auroc, detection_report
from .relational import mask_library, missingness_workload


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


# Which per-channel attribution belongs with which window score. A score view
# and a localisation map that disagree about what they are measuring produce a
# table that cannot be read, so the pairing is stated once, here.
ATTRIBUTION_OF = {"joint": "conditional", "independent": "marginal",
                  "marginal_max": "marginal", "conditional_max": "conditional",
                  "relational_max": "R"}


def dependence_diagnostics(result: Dict) -> Dict[str, float]:
    views = score_views(result)
    signed = views["independent"] - views["joint"]
    return {"empirical_log_dependence_mean": float(signed.mean()),
            "empirical_log_dependence_abs_mean": float(np.abs(signed).mean()),
            "empirical_log_dependence_sd": float(signed.std()),
            "empirical_log_dependence_q05": float(np.quantile(signed, .05)),
            "empirical_log_dependence_q95": float(np.quantile(signed, .95))}


def oracle_agreement(result: Dict, truth: Dict) -> Dict[str, float]:
    """Signed and absolute error of a fitted map against the known-law oracle.

    Step 3 asks for an ACCURACY column, and absolute dependence is not one: a
    circuit can carry the right amount of dependence in the wrong places. These
    compare the quantities the study actually reports — the conditional term
    first, since that is what every relational score is built from — on the
    observed entries only, because an unobserved channel has no target value.
    """
    mask = result["mask"].cpu().numpy()
    out: Dict[str, float] = {}
    for key in ("conditional", "marginal", "R"):
        a = result[key].cpu().numpy()[mask]
        b = truth[key].cpu().numpy()[mask]
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            continue
        error = a[finite] - b[finite]
        out[f"oracle_{key}_bias"] = float(error.mean())
        out[f"oracle_{key}_mae"] = float(np.abs(error).mean())
        out[f"oracle_{key}_max_abs"] = float(np.abs(error).max())
        if b[finite].std() > 0:
            out[f"oracle_{key}_corr"] = float(np.corrcoef(a[finite], b[finite])[0, 1])
    error = (result["log_px"].cpu().numpy() - truth["log_px"].cpu().numpy())
    out["oracle_log_px_bias"] = float(error.mean())
    out["oracle_log_px_mae"] = float(np.abs(error).mean())
    return out


def localization_metrics(attr, affected, observed, alarms, tie_tolerance=1e-4):
    """No forced winner among tied channels; report observable targets only.

    An observed target is not necessarily identifiable. Ambiguity is reported
    empirically; this function does not infer a causal fault graph from scores.
    End-to-end unique top-1 includes EVERY corrupted example as denominator.

    `observed` is a (C,) pattern shared by every row, or an (N, C) per-row one
    — the missingness regimes of step 6 produce the second kind, and scoring
    them against a shared pattern would credit a method for channels that were
    not there.
    """
    attr = np.asarray(attr)
    observed = np.asarray(observed)
    if observed.ndim == 1:
        observed = np.broadcast_to(observed, attr.shape)
    aps, unique_hits, ambiguous = [], [], 0
    absent, all_targets, n_faults, successes, abstained = 0, 0, 0, 0, 0
    for i, truth in enumerate(affected):
        if not truth:
            continue
        n_faults += 1
        obs = np.flatnonzero(observed[i])
        y = np.isin(obs, truth)
        if not y.any():
            absent += 1
            continue
        if y.all():
            all_targets += 1
        else:
            aps.append(average_precision(attr[i, obs], y))
        values = attr[i, obs]
        winners = np.flatnonzero(values >= values.max() - tie_tolerance)
        if len(winners) != 1 or len(obs) < 2:
            ambiguous += 1
            continue
        hit = bool(y[winners[0]])
        unique_hits.append(hit)
        if not alarms[i]:
            abstained += 1
        successes += int(hit and alarms[i])
    return {"loc_ap": float(np.mean(aps)) if aps else float("nan"),
            "loc_ap_n": len(aps), "loc_n_no_observed_target": absent,
            "loc_n_all_observed_are_targets": all_targets,
            "loc_n_ambiguous": ambiguous, "loc_n_unique": len(unique_hits),
            "loc_n_abstained_no_alarm": abstained,
            "loc_unique_top1_accuracy": float(np.mean(unique_hits)) if unique_hits else float("nan"),
            "end_to_end_unique_top1": successes / n_faults if n_faults else float("nan")}


# ═══════════════════════════════════════════════════════════════════════════
# Paired comparison (step 7)
# ═══════════════════════════════════════════════════════════════════════════

def cluster_bootstrap(statistic: Callable[[np.ndarray], float], units,
                      reps: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Resample ENGINES with replacement, keeping every window of each draw.

    The unit of independence is the engine, not the window: overlapping windows
    from one engine are not independent draws and an interval computed over
    them is too narrow. Repeated training seeds are a separate axis and are
    never additional engines.
    """
    units = np.asarray(units)
    unique = np.unique(units)
    if reps < 1 or len(unique) < 2:
        raise ValueError("bootstrap requires >=2 units and positive reps")
    groups = [np.flatnonzero(units == u) for u in unique]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        value = statistic(idx)
        if np.isfinite(value):
            draws.append(float(value))
    if not draws:
        raise ValueError("no bootstrap replicate produced a defined statistic")
    return {"ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)),
            "bootstrap_reps_used": len(draws), "n_units": len(unique)}


def paired_engine_bootstrap(a, b, labels, units, reps=1000, seed=0):
    """Paired AUROC difference a-b, cluster bootstrap on independent engines."""
    a, b, labels, units = map(np.asarray, (a, b, labels, units))
    if not (a.shape == b.shape == labels.shape == units.shape) or a.ndim != 1:
        raise ValueError("paired scores, labels and units must be aligned vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap requires finite scores, >=2 units and positive reps")
    out = cluster_bootstrap(
        lambda idx: auroc(a[idx], labels[idx]) - auroc(b[idx], labels[idx]),
        units, reps, seed)
    return {"auroc_delta": auroc(a, labels) - auroc(b, labels), **out}


def paired_localization_bootstrap(a_attr, b_attr, affected, observed, a_alarms,
                                  b_alarms, units, metric="loc_ap",
                                  reps=1000, seed=0):
    """Paired localisation-AP or end-to-end-success difference, same resampling.

    `affected` is a boolean (N, C) matrix — the same encoding the run's npz
    artifacts store — so this can be computed offline from two completed runs
    without re-querying either model.
    """
    if metric not in ("loc_ap", "end_to_end_unique_top1"):
        raise ValueError("metric must be loc_ap or end_to_end_unique_top1")
    a_attr, b_attr = np.asarray(a_attr), np.asarray(b_attr)
    affected = np.asarray(affected, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    a_alarms, b_alarms = np.asarray(a_alarms, dtype=bool), np.asarray(b_alarms, dtype=bool)
    units = np.asarray(units)
    if not (a_attr.shape == b_attr.shape == affected.shape) or a_attr.ndim != 2:
        raise ValueError("attribution and truth matrices must be aligned (N, C)")

    def rows(idx):
        truth = [np.flatnonzero(affected[i]).tolist() for i in idx]
        obs = observed[idx] if observed.ndim == 2 else observed
        return truth, obs

    def delta(idx):
        truth, obs = rows(idx)
        first = localization_metrics(a_attr[idx], truth, obs, a_alarms[idx])[metric]
        second = localization_metrics(b_attr[idx], truth, obs, b_alarms[idx])[metric]
        return first - second

    everything = np.arange(len(units))
    truth, obs = rows(everything)
    point = (localization_metrics(a_attr, truth, obs, a_alarms)[metric]
             - localization_metrics(b_attr, truth, obs, b_alarms)[metric])
    return {f"{metric}_delta": point, "metric": metric,
            **cluster_bootstrap(delta, units, reps, seed)}


# ═══════════════════════════════════════════════════════════════════════════
# Operational alarm accounting (step 6)
# ═══════════════════════════════════════════════════════════════════════════

def trajectory_reduce(scores, units, arms=None):
    """Per-engine maximum: the statistic a trajectory-level alarm actually uses.

    Calibrating on windows and alarming on a trajectory maximum are different
    events, and using a window threshold for a trajectory alarm inflates the
    false-alarm rate by roughly the number of windows. Reduce first, calibrate
    on the reduced object.

    `arms` splits each engine into separate trajectories — in the paired design
    the clean and corrupted copies of one engine's windows share a unit id, and
    reducing across both would put a corrupted window's score into the healthy
    trajectory's maximum and leave the study with no negatives at all. Passing
    the labels gives one healthy and one corrupted trajectory per engine, which
    are the objects an operational false-alarm rate is actually about.
    """
    scores, units = np.asarray(scores, dtype=float), np.asarray(units)
    if arms is None:
        keys = [(u,) for u in units]
    else:
        arms = np.asarray(arms)
        keys = list(zip(units.tolist(), arms.tolist()))
    order, seen = [], set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            order.append(key)
    keys_arr = np.array([hash(k) for k in keys])
    reduced, out_units, out_arms = [], [], []
    for key in order:
        sel = keys_arr == hash(key)
        reduced.append(scores[sel].max())
        out_units.append(key[0])
        out_arms.append(key[1] if len(key) > 1 else 0)
    if arms is None:
        return np.asarray(reduced), np.asarray(out_units)
    return np.asarray(reduced), np.asarray(out_units), np.asarray(out_arms)


def operational_trials(calibration_scores, calibration_units, evaluation_scores,
                       labels, alpha: float, trials: int = 100, seed: int = 0
                       ) -> Dict[str, float]:
    """Repeat the one-window-per-engine calibration draw and report its spread.

    One draw gives one threshold and one false-alarm rate, and reporting that
    number alone hides the fact that the next draw would have given a different
    one. Runs whose draw leaves too few calibration objects keep an INFINITE
    threshold — never a finite fallback — and how often that happens is itself
    a reported number.
    """
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    calibration_units = np.asarray(calibration_units)
    evaluation_scores = np.asarray(evaluation_scores, dtype=float)
    labels = np.asarray(labels)
    if trials < 1:
        raise ValueError("at least one trial is required")
    unique = np.unique(calibration_units)
    fprs, powers, infinite = [], [], 0
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        picked = np.array([int(rng.choice(np.flatnonzero(calibration_units == u)))
                           for u in unique])
        threshold = conservative_threshold(calibration_scores[picked], alpha)
        if not np.isfinite(threshold):
            infinite += 1
            fprs.append(0.0)
            powers.append(0.0)
            continue
        alarms = evaluation_scores > threshold
        fprs.append(float(alarms[labels == 0].mean()) if (labels == 0).any() else np.nan)
        powers.append(float(alarms[labels == 1].mean()) if (labels == 1).any() else np.nan)
    fprs, powers = np.asarray(fprs, dtype=float), np.asarray(powers, dtype=float)
    return {"trials": int(trials), "trial_alpha": float(alpha),
            "trial_fpr_mean": float(np.nanmean(fprs)),
            "trial_fpr_q05": float(np.nanquantile(fprs, .05)),
            "trial_fpr_q95": float(np.nanquantile(fprs, .95)),
            "trial_fpr_exceeds_alpha_frac": float(np.nanmean(fprs > alpha)),
            "trial_power_mean": float(np.nanmean(powers)),
            "trial_power_q05": float(np.nanquantile(powers, .05)),
            "trial_infinite_threshold_frac": infinite / trials,
            "trial_calibration_objects": int(len(unique))}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# Method adapters and the mask workload
# ═══════════════════════════════════════════════════════════════════════════

class CircuitMethod:
    """The trained circuit behind the same call signature as a baseline."""

    def __init__(self, pc, backend: str = "auto"):
        self.pc, self.backend = pc, backend
        self.name = "circuit"

    def diagnosis_map(self, X, mask=None):
        return self.pc.diagnosis_map(X, mask, backend=self.backend)

    def size(self):
        return self.pc.size()

    @property
    def device(self):
        return self.pc.device


class BaselineMethod:
    """A fitted comparator; identical contract, its own cost ledger."""

    def __init__(self, model):
        self.model = model
        self.name = model.name

    def diagnosis_map(self, X, mask=None):
        return self.model.diagnosis_map(X, mask)

    def size(self):
        return self.model.size()

    @property
    def device(self):
        return self.model.device


class OracleMethod:
    """The known generating law, scored like any other method.

    Two different questions get confused when the oracle appears only as an
    error reference.  "Is the circuit close to the truth?" is answered by the
    agreement columns.  "Was there anything to detect at all?" is answered only
    by scoring the truth itself — an oracle that cannot separate the corrupted
    windows means the task is unlearnable, and a learned model that fails such
    a task has not been shown to be weak.  Both control acceptance criteria in
    step 2 (the correlated oracle detects; the independent oracle scores zero)
    are statements about THIS method's scores, so it emits the same rows and
    the same artifacts as the circuit.

    It takes one shared channel pattern: a per-row regime has no single
    reference conditional, and inventing one would compare against a law the
    control never specified.  Such masks are recorded as skipped, not faked.
    """

    name = "known_law_oracle"
    shared_masks_only = True

    def __init__(self, covariance, channels: int):
        self.covariance, self.channels = covariance, int(channels)

    def diagnosis_map(self, X, mask=None):
        from .diagnosis_controls import gaussian_oracle_map
        from src.probabilistic_circuits import QueryCost
        if mask is not None and torch.as_tensor(mask).ndim != 1:
            raise ValueError(f"{self.name}: one shared channel pattern only")
        out = gaussian_oracle_map(X, self.covariance, self.channels, mask)
        observed = int(out["mask"][0].sum())
        # The ledger is honest about what this implementation does: one dense
        # multivariate-normal solve per observed channel plus one joint.
        out["cost"] = QueryCost(passes=observed + 1,
                                node_visits=0, node_evaluations=0,
                                output_elements=int(out["R"].numel()),
                                n_masks=1)
        return out

    def size(self):
        d = self.covariance.shape[0]
        return {"parameters": 0, "known_parameters": d * (d + 1) // 2, "nodes": 0}

    @property
    def device(self):
        return self.covariance.device


@dataclass
class MaskEntry:
    """One item of the frozen mask workload, with its mechanism attached."""
    name: str
    evaluation: torch.Tensor          # (C,) shared or (N, C) per row
    calibration: torch.Tensor
    meta: Dict = field(default_factory=dict)

    @property
    def shared(self) -> bool:
        return self.evaluation.ndim == 1

    def observed_counts(self) -> Dict[str, float]:
        m = self.evaluation if self.evaluation.ndim == 2 else self.evaluation.unsqueeze(0)
        counts = m.sum(1).double()
        return {"n_observed": float(counts.mean()),
                "n_observed_min": int(counts.min()), "n_observed_max": int(counts.max())}


def build_mask_workload(ev, task, X_eval, X_cal, affected, split_seed: int
                        ) -> List[MaskEntry]:
    """The fixed library, then any requested missingness regimes.

    Both halves are frozen before the models are queried. The library masks are
    externally fixed and shared, which is the regime the conservative threshold
    covers; every regime added after it is generated from a named mechanism and
    is a stress test until a separate procedure says otherwise.
    """
    entries: List[MaskEntry] = []
    lib = mask_library(task.n_channels, task.channel_groups,
                       ks=tuple(ev["mask_ks"]), n_per_k=int(ev["masks_per_k"]),
                       seed=split_seed)
    if ev.get("diagnosis_single_sensor", True):
        lib["single_sensor_0"] = torch.arange(task.n_channels) == 0
    for name, mask in lib.items():
        entries.append(MaskEntry(name, mask, mask, {
            "mechanism": "externally_fixed_shared",
            "depends_on_values": False, "depends_on_fault": False,
            "exchangeable_with_calibration": "yes — calibration uses this same pattern"}))

    witnesses = ev.get("diagnosis_witnesses") or []
    if witnesses:
        from .diagnosis_controls import witness_masks, witness_availability
        target = int(ev.get("diagnosis_witness_target", 0))
        for name, mask in witness_masks(task.n_channels, target, witnesses).items():
            entries.append(MaskEntry(name, mask, mask, {
                "mechanism": "witness_removal", "depends_on_values": False,
                "depends_on_fault": False,
                "exchangeable_with_calibration": "yes — calibration uses this same pattern",
                **witness_availability(mask, target, witnesses)}))

    regimes = ev.get("diagnosis_missingness") or []
    if regimes:
        work = missingness_workload(regimes, X_eval, task.window, task.n_channels,
                                    affected=affected,
                                    k=int(ev.get("diagnosis_missing_k", 1)),
                                    unseen=ev.get("diagnosis_unseen_dead"),
                                    seed=split_seed + 5)
        # Calibration windows are healthy, so a fault-driven mechanism has
        # nothing to act on there; that asymmetry IS the regime, and it is
        # recorded rather than patched over with fabricated calibration faults.
        cal = missingness_workload(regimes, X_cal, task.window, task.n_channels,
                                   affected=[[] for _ in range(len(X_cal))],
                                   k=int(ev.get("diagnosis_missing_k", 1)),
                                   unseen=ev.get("diagnosis_unseen_dead"),
                                   seed=split_seed + 6)
        for name, (masks, meta) in work.items():
            entries.append(MaskEntry(name, masks, cal[name][0],
                                     {**meta, "calibration_mechanism":
                                      "same rule, healthy windows, no fault term"}))
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# The stage
# ═══════════════════════════════════════════════════════════════════════════

def stage_diagnosis(cfg, seed, log):
    from .pipeline import prepare_task, _fit_window_pc, _row

    ev = cfg["eval"]
    covariance, control_kind = None, None
    if ev.get("diagnosis_control", False):
        witnesses = ev.get("diagnosis_witnesses") or []
        control_args = dict(window=cfg["dataset"]["window"],
                            channels=cfg["dataset"]["channels"],
                            n_train=int(ev.get("control_train", 512)),
                            n_val=int(ev.get("control_val", 300)), seed=seed)
        if witnesses:
            from .diagnosis_controls import witness_task
            task, covariance = witness_task(
                witnesses=witnesses, target=int(ev.get("diagnosis_witness_target", 0)),
                temporal=float(ev.get("control_temporal", .6)), **control_args)
            control_kind = "known_witness_graph"
        else:
            from .diagnosis_controls import make_control_task
            task, covariance = make_control_task(
                cross=float(ev.get("control_cross", .8)),
                temporal=float(ev.get("control_temporal", .6)), **control_args)
            control_kind = "known_gaussian"
    else:
        _, task = prepare_task(cfg, seed, log, "ad")
    split_seed = int(ev.get("diagnosis_split_seed", 701))
    parts = validation_partitions(task, split_seed)
    cal_idx = one_per_unit(parts["calibration"], task.unit_val, split_seed + 1)
    eval_idx = parts["evaluation"]
    operational = bool(ev.get("diagnosis_operational", False))
    if operational:
        # One score-independent window per evaluation engine, drawn by the SAME
        # rule as calibration. The all-window table stays available below and
        # stays labelled descriptive.
        eval_idx = one_per_unit(parts["evaluation"], task.unit_val, split_seed + 3)
    cap = int(ev.get("diagnosis_max_windows", 512))
    if cap > 0 and len(eval_idx) > cap:
        # Uniform random cap, not the first engines. Preserve indices for replay.
        rng = np.random.default_rng(split_seed + 2)
        eval_idx = eval_idx[torch.tensor(np.sort(rng.choice(len(eval_idx), cap, replace=False)))]
    Xclean = task.X_val[eval_idx]
    Xcal = task.X_val[cal_idx]

    # ── corruption, with or without matched donors (step 5) ──────────────
    donor_policy = ("true_nominal_independent" if covariance is not None
                    else "training_nominal_unmatched_context")
    donor_report: Dict[str, object] = {}
    donor_index = None
    if covariance is None and ev.get("diagnosis_donor_matching", False):
        from .diagnosis_donors import DonorMatcher, covariates_from_task
        donors = covariates_from_task(task, "train")
        recipients = covariates_from_task(task, "val")
        recipients = type(recipients)(
            regime=None if recipients.regime is None else recipients.regime[eval_idx.numpy()],
            health=None if recipients.health is None else recipients.health[eval_idx.numpy()],
            unit=None if recipients.unit is None else recipients.unit[eval_idx.numpy()])
        matcher = DonorMatcher(caliper=float(ev.get("diagnosis_donor_caliper", .25)),
                               strategy=str(ev.get("diagnosis_donor_strategy", "caliper_random")),
                               seed=split_seed + 7).fit(donors)
        match = matcher.match(recipients)
        donor_index = match.donor_index
        donor_policy = "training_nominal_matched_context"
        donor_report = {"matching_variables": match.matching_variables,
                        "caliper": match.caliper, "n_unmatched": match.n_unmatched,
                        **match.report, **match.balance}
        log.artifact_npz("diagnosis_donor_assignment", **match.as_artifact())
        log.info(f"  donors: matched {match.report['matched_fraction']:.1%} of "
                 f"{match.report['n_recipients']} recipients on "
                 f"{match.matching_variables} within caliper {match.caliper}")

    protocol = {"version": "diagnosis_v2", "evaluation_split": "exploratory_validation",
                "calibration_object": "one_random_window_per_engine",
                "evaluation_object": "one_random_window_per_engine" if operational
                                     else "all_validation_windows_capped",
                "mask_assumption": "externally_fixed_independent_of_current_observation",
                "donor_policy": donor_policy, "donor_report": donor_report,
                "control": control_kind,
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
        fit_s=getattr(pc, "fit_seconds", float("nan")),
        final_train_nll=float(pc.history[-1]) if len(pc.history) else float("nan"),
        lr=float(cfg["model"]["lr"]), epochs_budget=int(cfg["model"]["epochs"]),
        batch_size=int(cfg["model"]["batch_size"]),
        boundary_widths=[len(us) for us in pc.relational().units] if pc.is_channel_blocked else []))

    Xbad, _, kinds_bad, affected_bad, donors_used = contaminate_windows(Xclean,
        task.window, task.n_channels, inject_rate=1.0,
        strength=float(cfg["dataset"]["strength"]), kinds=tuple(ev["kinds"]),
        donors=task.X_train, seed=seed + 12000, donor_index=donor_index,
        return_donors=True)
    if covariance is not None:
        from .diagnosis_controls import independent_replacement
        target = int(ev.get("diagnosis_witness_target", 0))
        Xbad = independent_replacement(Xclean, covariance, task.n_channels,
                                       seed + 12000, target=target)
        kinds_bad = ["independent_block"] * len(Xbad)
        affected_bad = [[target] for _ in Xbad]
        donors_used = np.full(len(Xbad), -1, dtype=np.int64)
    X = torch.cat((Xclean, Xbad))
    labels = np.r_[np.zeros(len(Xclean)), np.ones(len(Xbad))].astype(int)
    kinds = ["normal"] * len(Xclean) + list(kinds_bad)
    affected = [[] for _ in Xclean] + list(affected_bad)
    affected_matrix = np.asarray([[c in a for c in range(task.n_channels)] for a in affected])
    units = np.tile(np.asarray(task.unit_val)[eval_idx], 2)

    # ── marginal-shortcut audit on separate development units (step 5) ───
    if ev.get("diagnosis_shortcut_audit", True):
        try:
            dev_idx = parts["checkpoint"]
            Xdev_clean = task.X_val[dev_idx]
            dev_bad, _, _, _, _ = contaminate_windows(Xdev_clean, task.window,
                task.n_channels, inject_rate=1.0,
                strength=float(cfg["dataset"]["strength"]), kinds=tuple(ev["kinds"]),
                donors=task.X_train, seed=seed + 13000, return_donors=True)
            if covariance is not None:
                from .diagnosis_controls import independent_replacement
                dev_bad = independent_replacement(Xdev_clean, covariance,
                    task.n_channels, seed + 13000,
                    target=int(ev.get("diagnosis_witness_target", 0)))
            from .diagnosis_donors import shortcut_audit as _audit
            audit = _audit(torch.cat((Xdev_clean, dev_bad)),
                           np.r_[np.zeros(len(Xdev_clean)), np.ones(len(dev_bad))],
                           X, labels, task.window, task.n_channels,
                           channel=int(ev.get("diagnosis_witness_target", 0)))
            log.result(_row(cfg, "diagnosis", "shortcut audit",
                            donor_policy=donor_policy, **audit))
            log.info(f"  shortcut audit: target-channel-only AUROC "
                     f"{audit['shortcut_auroc']:.3f} (0.5 = no marginal shortcut)")
        except (ValueError, FloatingPointError) as exc:
            log.info(f"  shortcut audit unavailable: {exc}")

    # ── comparators (step 4) ─────────────────────────────────────────────
    methods: List = [CircuitMethod(pc, "fast" if pc.is_channel_blocked else "oracle")]
    if covariance is not None and ev.get("diagnosis_oracle_method", True):
        # Read this arm FIRST. It bounds what any method on this task can do.
        methods.append(OracleMethod(covariance, task.n_channels))
    names = list(ev.get("diagnosis_baselines") or [])
    if names:
        from .diagnosis_baselines import build_baselines
        for model in build_baselines(names, task.window, task.n_channels, seed=seed,
                                     settings=ev.get("diagnosis_baseline_settings")):
            t0 = time.perf_counter()
            model.fit(task.X_train, task.X_val[parts["checkpoint"]])
            log.info(f"  baseline {model.name}: fit {time.perf_counter() - t0:.2f}s · "
                     f"{model.size()['parameters']:,} params · "
                     f"selection {model.selection_.get('chosen')}")
            log.result(_row(cfg, "diagnosis", f"model {model.name}",
                            method_class="fitted_baseline",
                            fit_s=model.fit_seconds, **model.size(),
                            selection_split=model.selection_.get("split"),
                            selection_choice=str(model.selection_.get("chosen")),
                            selection_score=model.selection_.get("chosen_score"),
                            n_stabilizations=len(model.stabilization_)))
            log.artifact_json(f"baseline_{model.name}_report", model.report())
            methods.append(BaselineMethod(model))

    oracle_scored = any(isinstance(m, OracleMethod) for m in methods)

    # ── trivial detection comparators (fit once, refit per mask lazily) ──
    detectors: List = []
    det_names = list(ev.get("diagnosis_detectors") or [])
    if det_names:
        from .diagnosis_detectors import build_detectors
        policies = tuple(ev.get("diagnosis_detector_policies") or ("refit",))
        detectors = build_detectors(det_names, task.window, task.n_channels,
                                    seed=seed, policies=policies)
        for det in detectors:
            det.fit(task.X_train)
        log.info(f"  detection comparators: {[d.name for d in detectors]}")

    entries = build_mask_workload(ev, task, X, Xcal, affected, split_seed)
    rows = []
    alpha = float(ev.get("diagnosis_alpha", .1))
    trials = int(ev.get("diagnosis_operational_trials", 0))
    artifacts: List[str] = []
    for mask_idx, entry in enumerate(entries):
        log.info(f"  diagnosis: {entry.name} "
                 f"({entry.observed_counts()['n_observed']:.1f} observed on average), "
                 f"{len(X)} paired windows, {len(methods)} method(s)")
        for method in methods:
            is_circuit = isinstance(method, CircuitMethod)
            if getattr(method, "shared_masks_only", False) and not entry.shared:
                # Recorded, not silently absent: a missing oracle row on a
                # heterogeneous mask has to be distinguishable from an oracle
                # that ran and scored badly.
                log.result(_row(cfg, "diagnosis",
                                f"skipped {entry.name} | {method.name}",
                                mask=entry.name, method_name=method.name,
                                skipped="method takes one shared channel pattern; "
                                        "this mask is per-row"))
                continue
            err = 0.0
            if is_circuit and pc.is_channel_blocked:
                probe_n = min(len(X), int(ev["oracle_check_windows"]))
                probe = X[:probe_n]
                probe_mask = (entry.evaluation if entry.shared
                              else entry.evaluation[:probe_n])
                fast = pc.diagnosis_map(probe, probe_mask, backend="fast")
                slow = pc.diagnosis_map(probe, probe_mask, backend="oracle")
                errors = []
                for k in ("marginal", "conditional", "R", "log_px"):
                    a, b = fast[k], slow[k]
                    valid = torch.ones_like(a, dtype=torch.bool) if k == "log_px" else fast["mask"]
                    if not bool(torch.isfinite(a[valid]).all() and torch.isfinite(b[valid]).all()):
                        raise AssertionError("non-finite observed exact-query result")
                    errors.append(float((a[valid] - b[valid]).abs().max()))
                err = max(errors)
                if err > float(ev["oracle_tolerance"]):
                    raise AssertionError(f"masked exact-query check failed: {err}")
            # Cold setup (cached block constants / factorisations) is reported
            # separately from warm query latency; folding them together hides
            # which method pays a per-mask construction cost.
            warm_mask = entry.evaluation if entry.shared else entry.evaluation[:min(4, len(X))]
            _sync(method.device)
            t0 = time.perf_counter()
            method.diagnosis_map(X[:min(4, len(X))], warm_mask)
            _sync(method.device)
            setup_s = time.perf_counter() - t0
            _sync(method.device)
            t0 = time.perf_counter()
            result = method.diagnosis_map(X, entry.evaluation)
            _sync(method.device)
            query_s = time.perf_counter() - t0
            calibration = method.diagnosis_map(Xcal, entry.calibration)
            views, cv = score_views(result), score_views(calibration)
            observed = result["mask"].cpu().numpy()
            attribution = {k: result[k].cpu().numpy()
                           for k in ("marginal", "conditional", "R")}
            alarm_by_score, threshold_by_score = {}, {}
            for score, values in views.items():
                threshold = conservative_threshold(cv[score], alpha)
                threshold_by_score[score] = threshold
                alarm_by_score[score] = values > threshold
            tag = f"diagnosis_mask{mask_idx}" if is_circuit \
                else f"diagnosis_{method.name}_mask{mask_idx}"
            # The artifact carries the identity of the mask that produced it.
            # A frozen confirmation workload is a list of mask NAMES, and an
            # artifact that only knows its own index cannot be checked against
            # one without trusting a filename convention.
            evidence = {"mask_name": np.asarray(entry.name),
                        "mask_index": np.asarray(mask_idx),
                        "mask_mechanism": np.asarray(str(entry.meta.get("mechanism"))),
                        "labels": labels, "units": units, "kinds": np.asarray(kinds),
                        "observed": observed, "pair_index": np.tile(eval_idx.numpy(), 2),
                        "input_windows": X.numpy(), "affected": affected_matrix,
                        "donor_index": np.tile(donors_used, 2),
                        **views,
                        **{f"attr_{k}": v for k, v in attribution.items()},
                        **{f"alarm_{k}": v for k, v in alarm_by_score.items()}}
            log.artifact_npz(tag, **evidence)
            artifacts.append(tag)
            log.result(_row(cfg, "diagnosis", f"query {entry.name} | {method.name}",
                mask=entry.name, method_name=method.name,
                mask_mechanism=entry.meta.get("mechanism"),
                mask_shared=entry.shared,
                backend=method.backend if is_circuit else "fitted",
                # A latency is only comparable to another measured on the same
                # device. The fitted comparators run float64 and stay on the
                # CPU (MPS has no double); a GPU circuit timed against them is a
                # device comparison, and this column is what exposes it.
                device=str(method.device),
                oracle_max_error=err, query_s=query_s, warmup_s=setup_s, n=len(X),
                **entry.observed_counts(),
                **{f"cost_{k}": v for k, v in result["cost"].as_dict().items()},
                **dependence_diagnostics({k: v[:len(Xclean)] if isinstance(v, torch.Tensor) else v
                                          for k, v in result.items()})))
            if covariance is not None and entry.shared:
                # The known-law oracle takes ONE shared pattern; a per-row
                # regime has no single reference map, and pretending otherwise
                # would compare against the wrong conditional.
                from .diagnosis_controls import gaussian_oracle_map
                truth = gaussian_oracle_map(X, covariance, task.n_channels,
                                            entry.evaluation)
                log.result(_row(cfg, "diagnosis",
                    f"known-law error {entry.name} | {method.name}",
                    mask=entry.name, method_name=method.name,
                    reference="known_generating_distribution",
                    **oracle_agreement(result, truth)))
            for score, values in views.items():
                threshold = threshold_by_score[score]
                alarms = alarm_by_score[score]
                attr_key = ATTRIBUTION_OF[score]
                loc = localization_metrics(attribution[attr_key], affected,
                                           observed, alarms)
                row = _row(cfg, "diagnosis", f"{score} | {entry.name} | {method.name}",
                    score=score, mask=entry.name, method_name=method.name,
                    mask_mechanism=entry.meta.get("mechanism"),
                    localization_score=attr_key, alpha=alpha, calibration_n=len(Xcal),
                    threshold=threshold if np.isfinite(threshold) else None,
                    threshold_infinite=not np.isfinite(threshold),
                    score_range=float(np.ptp(values)),
                    numerically_flat=bool(np.ptp(values) < float(ev["oracle_tolerance"])),
                    fpr=float(alarms[labels == 0].mean()), power=float(alarms[labels == 1].mean()),
                    **{k: v for k, v in entry.meta.items()
                       if k in ("target_visible", "witnesses_available",
                                "identifiable", "depends_on_values", "depends_on_fault")},
                    **detection_report(values, labels, kinds), **loc)
                if trials:
                    row.update(operational_trials(
                        cv[score], np.asarray(task.unit_val)[cal_idx], values,
                        labels, alpha, trials, seed=split_seed + 11))
                if ev.get("diagnosis_trajectory", False):
                    reduced, _, reduced_labels = trajectory_reduce(values, units, labels)
                    cal_reduced, _ = trajectory_reduce(
                        cv[score], np.asarray(task.unit_val)[cal_idx])
                    traj_threshold = conservative_threshold(cal_reduced, alpha)
                    traj_alarm = reduced > traj_threshold
                    row.update({
                        "trajectory_n": int(len(reduced)),
                        "trajectory_auroc": auroc(reduced, reduced_labels),
                        "trajectory_threshold_infinite": not np.isfinite(traj_threshold),
                        "trajectory_fpr": float(traj_alarm[reduced_labels == 0].mean())
                                          if (reduced_labels == 0).any() else float("nan"),
                        "trajectory_power": float(traj_alarm[reduced_labels == 1].mean())
                                            if (reduced_labels == 1).any() else float("nan")})
                log.result(row)
                rows.append(row)
        # ── trivial detection comparators, same mask, same windows ───────
        #
        # Detection is the axis on which this method has to be at least
        # competitive before any diagnosis claim is interesting, and the
        # comparator that matters is not another circuit: it is a detector
        # refitted on whatever sensors are left. These emit detection rows
        # only — no localisation, because they produce one number per window
        # and an invented attribution is not a comparison.
        if detectors and entry.shared:
            for det in detectors:
                t0 = time.perf_counter()
                values = det.score(X, entry.evaluation)
                det_s = time.perf_counter() - t0
                cal_values = det.score(Xcal, entry.calibration)
                if not np.isfinite(values).all() or not np.isfinite(cal_values).all():
                    log.info(f"  detector {det.name} produced non-finite scores "
                             f"on {entry.name}; recorded as failed")
                    log.result(_row(cfg, "diagnosis",
                                    f"detector | {entry.name} | {det.name}",
                                    mask=entry.name, method_name=det.name,
                                    failed="non-finite detector score"))
                    continue
                threshold = conservative_threshold(cal_values, alpha)
                alarms = values > threshold
                det_row = _row(cfg, "diagnosis",
                    f"detector | {entry.name} | {det.name}",
                    score="detector", mask=entry.name, method_name=det.name,
                    method_class="detection_baseline",
                    detector_policy=det.policy, detector_display=det.display,
                    mask_mechanism=entry.meta.get("mechanism"),
                    alpha=alpha, calibration_n=len(Xcal),
                    threshold=threshold if np.isfinite(threshold) else None,
                    threshold_infinite=not np.isfinite(threshold),
                    score_range=float(np.ptp(values)),
                    numerically_flat=bool(np.ptp(values) < float(ev["oracle_tolerance"])),
                    fpr=float(alarms[labels == 0].mean()),
                    power=float(alarms[labels == 1].mean()),
                    query_s=det_s, fit_s=det.fit_seconds, n_fits=det.n_fits,
                    **entry.observed_counts(),
                    **detection_report(values, labels, kinds))
                if trials:
                    det_row.update(operational_trials(
                        cal_values, np.asarray(task.unit_val)[cal_idx], values,
                        labels, alpha, trials, seed=split_seed + 11))
                if ev.get("diagnosis_trajectory", False):
                    reduced, _, reduced_labels = trajectory_reduce(values, units, labels)
                    cal_reduced, _ = trajectory_reduce(
                        cal_values, np.asarray(task.unit_val)[cal_idx])
                    traj_threshold = conservative_threshold(cal_reduced, alpha)
                    traj_alarm = reduced > traj_threshold
                    det_row.update({
                        "trajectory_n": int(len(reduced)),
                        "trajectory_auroc": auroc(reduced, reduced_labels),
                        "trajectory_threshold_infinite": not np.isfinite(traj_threshold),
                        "trajectory_fpr": float(traj_alarm[reduced_labels == 0].mean())
                                          if (reduced_labels == 0).any() else float("nan"),
                        "trajectory_power": float(traj_alarm[reduced_labels == 1].mean())
                                            if (reduced_labels == 1).any() else float("nan")})
                log.result(det_row)
                rows.append(det_row)
                log.artifact_npz(f"diagnosis_{det.name}_detect_mask{mask_idx}",
                                 mask_name=np.asarray(entry.name), labels=labels,
                                 units=units, detector=values,
                                 alarm_detector=alarms)

        if covariance is not None and entry.shared and not oracle_scored:
            # Fallback only. When the oracle runs as a METHOD it already emits a
            # strictly richer row for the same numbers — calibrated threshold,
            # localisation, cost and a paired artifact — and emitting both under
            # one method name double-counts every oracle AUROC in the aggregate
            # while leaving half the rows without the columns beside it.
            from .diagnosis_controls import gaussian_oracle_map
            truth = gaussian_oracle_map(X, covariance, task.n_channels, entry.evaluation)
            for score, values in score_views(truth).items():
                log.result(_row(cfg, "diagnosis", f"known-law {score} | {entry.name}",
                    mask=entry.name, score=score, method_name="known_law_oracle",
                    reference="known_generating_distribution",
                    **{k: v for k, v in entry.meta.items()
                       if k in ("target_visible", "witnesses_available", "identifiable")},
                    **detection_report(values, labels, kinds)))
    log.info("  Diagnosis completed on validation engines only; no confirmatory PASS/FAIL inferred.")
    return {"protocol": protocol, "rows": rows, "mask_artifacts": [e.name for e in entries],
            "artifacts": artifacts, "methods": [m.name for m in methods]}
