"""
Experiment stages — the training/evaluation pipeline itself.

Five stages, all driven from one resolved config and all writing the same
structured rows, so anything they produce can be aggregated across datasets,
structures and seeds without special cases:

  ad            train a window density, score detection against the full
                baseline suite, run the two queries no baseline can express
                (dead sensors by exact marginalisation; the exact typed
                marginal/conditional/structural split)
  explain       exact attribution vs the strong adversaries (Gaussian
                conditional, AE reconstruction, AE replacement sensitivity),
                scored on correctness / completeness / faithfulness against
                per-channel ground truth.  Completeness is a property of the
                CHAIN-RULE attribution only; every row carries `additive` and
                `mean_abs_gap_nats` so the two claims stay separable.
  rul           the joint (window, τ) circuit: censoring ablation, point and
                distributional accuracy against ridge/MLP/CQR, survival under
                partial evidence
  calibration   split conformal ON the circuit's own predictive — the answer to
                "exact ≠ calibrated", with the unit-level split that makes the
                coverage claim mean something
  scaling       tree vs DAG layout, the engineering claim

Every stage is written so that a real dataset and the synthetic one go through
exactly the same code path.  That is the point: "does it hold on real data?"
must be a config change, never a reimplementation, or the two answers are not
comparable.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .baselines import (
    ChannelZScore,
    ConvAutoencoder,
    detection_baselines,
    rul_baselines,
)
from .circuits import DegenerateModelError, SurvivalPC, WindowPC, resolve_device
from .conformal import ConformalPredictive, split_units
from .data import contaminate_windows
from .datasets import (
    build_ad_task,
    build_rul_task,
    dataset_available,
    dataset_id,
    describe_task,
    load_fleets,
)
from .explain import (
    GaussianConditional,
    ae_channel_error,
    ADDITIVE_ATTRIBUTIONS,
    additivity_gap,
    completeness_error,
    deletion_curve,
    explain_window,
    format_explanation,
    localization_report,
    pc_attributions,
    plot_case_study,
    plot_deletion_curves,
    plot_localization,
    replacement_sensitivity,
    zscore_channel,
)
from .metrics import (
    calibration_error,
    crps_from_interval,
    crps_from_pmf,
    detection_report,
    mae,
    mpiw,
    nasa_score,
    picp,
    pit_report,
    rmse,
)
from .ts_logging import RunLogger
from src.probabilistic_circuits import BlockStructureError


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dcfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["dataset"]


def _mcfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["model"]


def _ecfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["eval"]


def _row(cfg: Dict[str, Any], stage: str, method: str, **metrics) -> Dict[str, Any]:
    """One comparable result line.  The variant axes are inlined so a table can
    be pivoted on them without reading any config file."""
    row = {"stage": stage, "experiment": cfg.get("name"),
           "variant": cfg.get("variant", "default"),
           "dataset": dataset_id(_dcfg(cfg)), "method": method}
    for k, v in (cfg.get("variant_axes") or {}).items():
        row[f"axis:{k}"] = v
    row.update(metrics)
    return row


def _fit_window_pc(cfg: Dict[str, Any], task, seed: int, log: RunLogger,
                   tag: str = "pc", X_checkpoint: Optional[torch.Tensor] = None) -> WindowPC:
    m = _mcfg(cfg)
    t0 = time.time()
    pc = WindowPC(task.window, task.n_channels, vtree_method=m["vtree"],
                  n_sum_components=int(m["K"]),
                  leaf_components=int(m["leaf_components"]),
                  channel_groups=task.channel_groups, use_sos=bool(m["sos"]),
                  delta=bool(m["delta"]), weight_jitter=float(m["weight_jitter"]),
                  seed=seed, device=cfg.get("device"),
                  evaluator=cfg.get("evaluator", "layered"),
                  boundary_components=m.get("boundary_K"),
                  upper_components=m.get("upper_K"),
                  channel_mixture=bool(m.get("channel_mixture", False)))
    X_val = X_checkpoint if X_checkpoint is not None else getattr(task, "X_val", None)
    pc.fit(task.X_train, epochs=int(m["epochs"]), lr=float(m["lr"]),
           batch_size=int(m["batch_size"]), log_every=max(int(m["epochs"]) // 8, 1),
           X_val=X_val, select_best=bool(m.get("select_best", True)),
           conditional_weight=float(m.get("conditional_weight", 0.0)),
           mask_drop_prob=float(m.get("train_mask_drop_prob", 0.25)),
           select_metric=m.get("select_metric", "nll"),
           patience=int(m.get("patience", 0)), min_epochs=int(m.get("min_epochs", 1)),
           restarts=int(m.get("restarts", 1)),
           lr_schedule=str(m.get("lr_schedule", "none")),
           lr_min_factor=float(m.get("lr_min_factor", 0.05)),
           restart_abandon_margin=float(m.get("restart_abandon_margin", 0.25)))
    fit_s = time.time() - t0
    log.history(f"{tag}_train_nll", pc.history)
    log.history(f"{tag}_train_objective", pc.objective_history)
    log.history(f"{tag}_selection_loss", pc.val_objective_history)
    if pc.val_history:
        log.history(f"{tag}_val_nll", pc.val_history)
    sd = pc.assert_informative(task.X_train)         # loud, not silent (§3)
    # windows/s makes the device AND evaluator choice auditable after the fact.
    # It matters: on the recursive evaluator a GPU run is legitimately slower
    # than CPU, on the layered one it is ~2× faster above batch 128, and the
    # only way to tell which regime a finished run was in is to log it.
    thr = len(task.X_train) * len(pc.history) / max(fit_s, 1e-9)
    ev = "layered" if pc.compiled is not None else "recursive"
    log.info(f"  {tag}: fit {fit_s:.1f}s · {pc.size()['parameters']:,} params · "
             f"score sd {sd:.3f} · device {pc.device} · {ev} · {thr:,.0f} win/s")
    if getattr(pc, "restart_traces", None) and len(pc.restart_traces) > 1:
        # Every attempt, not just the winner: a study that cannot see the
        # losing restarts cannot tell a reliable architecture from a lucky one.
        # These go to the log and to the model row, not to `log.history` —
        # a history file is indexed by EPOCH, and these are indexed by restart.
        for t in pc.restart_traces:
            log.info(f"  {tag}: restart {t['restart']} (init seed {t['init_seed']}) "
                     f"checkpoint {t['selection_loss']:.3f} at epoch {t['best_epoch']} "
                     f"of {t['epochs_run']} run"
                     + (" · abandoned" if t["abandoned"] else "")
                     + (" · stopped early" if t["stopped_early"] else ""))
        log.info(f"  {tag}: selected restart {pc.selected_restart} of "
                 f"{len(pc.restart_traces)}")
    if pc.val_history:
        log.info(f"  {tag}: validation on {len(X_val)} windows from "
                 f"{len(set(task.unit_val.tolist())) if X_checkpoint is None and getattr(task, 'unit_val', None) is not None else 'explicit subset of'} "
                 f"held-out units · best epoch {pc.best_epoch} of "
                 f"{int(m['epochs'])} · val nll {pc.best_val_nll:.3f}")
    else:
        log.info(f"  {tag}: NO validation split — the reported likelihood is "
                 "the training likelihood and selects nothing")
    pc.fit_seconds = fit_s                            # type: ignore[attr-defined]
    return pc


def _nll(pc: WindowPC, X: Optional[torch.Tensor], n: int = 1024) -> float:
    """
    Mean −log p over the first `n` windows of `X`.  Deliberately unnamed as to
    which split it is scoring: the CALLER says that, in the column name, and
    the caller is what got this wrong.  `stage_ad` used to report
    `train_nll=_held_out_nll(pc, task.X_train)` — a held-out-sounding helper
    reading the training set — so no recorded run could tell overfitting from
    undertraining.  Returns NaN when the split does not exist, which is what
    every downstream table should show rather than a training number wearing a
    validation label.
    """
    if X is None or not len(X):
        return float("nan")
    with torch.no_grad():
        return float(pc.score(X[:n]).mean())


def prepare_task(cfg: Dict[str, Any], seed: int, log: RunLogger, kind: str):
    """Load fleets and build one task, logging exactly what was built."""
    spec = _dcfg(cfg)
    t0 = time.time()
    pair = load_fleets(spec, seed=seed)
    task = (build_ad_task(pair, spec, seed=seed) if kind == "ad"
            else build_rul_task(pair, {**spec, "stride": spec.get("rul_stride", 3)},
                                seed=seed))
    desc = describe_task(task)
    log.info(f"  data: {pair}  ({time.time() - t0:.1f}s)")
    log.info(f"  task: {desc}")
    # Known defects of the source, printed beside the numbers they qualify.
    # SMAP/MSL's triviality and OPSSAT's segment-level labels are the kind of
    # thing that gets dropped between a run and a write-up; here they cannot be.
    for c in pair.meta.get("caveats", []):
        log.info(f"  caveat: {c}")
    log.metrics({f"task_{kind}": desc,
                 "dataset_caveats": list(pair.meta.get("caveats", []))})
    return pair, task


# ═══════════════════════════════════════════════════════════════════════════
# Stage: anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

def stage_ad(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    ev = _ecfg(cfg)
    pair, task = prepare_task(cfg, seed, log, "ad")
    y, kinds = task.y_test, task.kind_test
    out: Dict[str, Any] = {}

    pc = _fit_window_pc(cfg, task, seed, log)
    s_pc = pc.score(task.X_test)
    name = ("SquaredPC/SOS" if _mcfg(cfg)["sos"] else "RegionGraphPC")
    name += " +delta" if _mcfg(cfg)["delta"] else ""
    name += f" [{_mcfg(cfg)['vtree']}, K={_mcfg(cfg)['K']}]"
    rep = detection_report(s_pc, y, kinds)
    log.result(_row(cfg, "ad", name, **rep, fit_s=pc.fit_seconds,
                    params=pc.size()["parameters"],
                    train_nll=_nll(pc, task.X_train),
                    val_nll=_nll(pc, getattr(task, "X_val", None)),
                    val_units=len(set(task.unit_val.tolist()))
                    if getattr(task, "unit_val", None) is not None else 0,
                    best_epoch=pc.best_epoch))
    out["circuit"] = rep
    scores = {name: s_pc.numpy()}

    # ── baselines, simple tier first ─────────────────────────────────────
    if ev["baselines"]:
        for b in detection_baselines(task.window, task.n_channels, seed=seed,
                                     include_slow=not ev["fast_baselines"],
                                     device=cfg.get("device")):
            t0 = time.time()
            try:
                b.fit(task.X_train)
                sb = b.score(task.X_test)
                rb = detection_report(sb, y, kinds)
                scores[b.name] = np.asarray(sb, dtype=float).ravel()
                log.result(_row(cfg, "ad", b.name, **rb, fit_s=time.time() - t0))
            except Exception as exc:                  # a broken optional dep
                log.info(f"  baseline {b.name} failed: {exc}")
                log.result(_row(cfg, "ad", b.name, auroc=float("nan"),
                                error=str(exc)[:120]))

    # ── query 1: dead sensors — exact marginalisation vs imputation ───────
    if ev["missing"]:
        dead = list(range(0, task.n_channels,
                          max(task.n_channels // 4, 1)))[: int(ev["n_dead"])]
        Xte = task.X_test.reshape(-1, task.window, task.n_channels).clone()
        Xte[:, :, dead] = 0.0                          # data is standardised
        X_imp = Xte.reshape(len(task.X_test), -1)
        exact = pc.score_with_missing(task.X_test, dead)
        imputed = pc.score(X_imp)
        log.result(_row(cfg, "ad", f"PC · {len(dead)} dead (exact marginal)",
                        **detection_report(exact, y, kinds), dead=len(dead)))
        log.result(_row(cfg, "ad", f"PC · {len(dead)} dead (mean-imputed)",
                        **detection_report(imputed, y, kinds), dead=len(dead)))
        if ev["baselines"]:
            for b in detection_baselines(task.window, task.n_channels, seed=seed,
                                         include_slow=False,
                                         device=cfg.get("device"))[:3]:
                b.fit(task.X_train)
                log.result(_row(cfg, "ad", f"{b.name} · {len(dead)} dead (imputed)",
                                **detection_report(b.score(X_imp), y, kinds),
                                dead=len(dead)))
        out["missing"] = {"dead_channels": dead}

    # ── query 2: exact typed decomposition ───────────────────────────────
    if ev["typed"]:
        td = pc.typed_scores(task.X_test)
        y_np, k_np = y.numpy(), np.asarray(kinds)
        typed: Dict[str, Dict[str, float]] = {}
        for kind in ["normal"] + sorted(set(k_np[y_np == 1])):
            sel = k_np == kind
            if not sel.any():
                continue
            typed[kind] = {
                "marginal": float(td["marginal"][sel].max(1).values.mean()),
                "conditional": float(td["conditional"][sel].max(1).values.mean()),
                "structural": float(td["structural"][sel].max(1).values.mean()),
                "n": int(sel.sum()),
            }
        struct_score = td["structural"].max(1).values
        log.result(_row(cfg, "ad", "PC · structural-only score",
                        **detection_report(struct_score, y, kinds)))
        out["typed"] = typed
        scores["PC structural-only"] = struct_score.numpy()
        log.info("  typed decomposition (mean worst-channel surprise):")
        for k, v in typed.items():
            log.info(f"    {k:>9}  n={v['n']:>5}  marginal {v['marginal']:8.2f}  "
                     f"conditional {v['conditional']:8.2f}  "
                     f"structural {v['structural']:8.2f}")

    if ev["save_scores"]:
        log.artifact_npz("ad_scores", y=y.numpy(),
                         kinds=np.asarray(kinds, dtype=object).astype("U16"),
                         **{k.replace(" ", "_")[:40]: v for k, v in scores.items()})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Stage: explanation quality
# ═══════════════════════════════════════════════════════════════════════════

def stage_explain(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    ev = _ecfg(cfg)
    pair, task = prepare_task(cfg, seed, log, "ad")

    # Attribution costs O(C) circuit passes per view, so on real data (tens of
    # thousands of test windows) the full test set is neither affordable nor
    # necessary — a capped, ORDER-PRESERVING subsample keeps every anomaly kind
    # in proportion.
    cap_n = int(ev["max_explain_windows"] or 0)
    if cap_n and len(task.X_test) > cap_n:
        idx = np.linspace(0, len(task.X_test) - 1, cap_n).astype(int)
        task.X_test = task.X_test[idx]
        task.y_test = task.y_test[idx]
        task.kind_test = [task.kind_test[i] for i in idx]
        task.affected_test = [task.affected_test[i] for i in idx]
        log.info(f"  explain: capped test set to {cap_n} windows")

    pc = _fit_window_pc(cfg, task, seed, log)
    ae = ConvAutoencoder(task.window, task.n_channels, seed=seed,
                         device=cfg.get("device")).fit(task.X_train)
    gc = GaussianConditional(task.window, task.n_channels).fit(task.X_train)
    zs = ChannelZScore(task.window, task.n_channels).fit(task.X_train)

    attrs: Dict[str, np.ndarray] = {}
    t0 = time.time()
    attrs.update(pc_attributions(pc, task.X_test,
                                 shapley_orders=int(ev["shapley_orders"]),
                                 chain_rule=bool(ev.get("chain_rule_attr", True))))
    pc_exact_s = time.time() - t0
    attrs[gc.name] = gc.attribute(task.X_test)
    attrs["AE reconstruction (per channel)"] = ae_channel_error(ae, task.X_test)
    t0 = time.time()
    attrs[f"AE replacement sensitivity ({ev['shap_samples']}/ch)"] = \
        replacement_sensitivity(ae, task.X_test, task.X_train,
                                n_samples=int(ev["shap_samples"]), seed=seed)
    rs_s = time.time() - t0
    attrs["z-score (per channel)"] = zscore_channel(zs, task.X_test)
    log.info(f"  attribution cost: PC exact (all views) {pc_exact_s:.1f}s · "
             f"AE replacement sensitivity (one view) {rs_s:.1f}s "
             "— NOT a SHAP comparison, see explain.replacement_sensitivity")

    kinds = list(ev["kinds"])
    present = {k for k in task.kind_test}
    kinds = [k for k in kinds if k in present]
    if not kinds:
        log.info("  no injected anomaly kinds in this test set — nothing to localise")
        return {}

    # ── 1. correctness ───────────────────────────────────────────────────
    per_kind: Dict[str, Dict[str, float]] = {}
    n_comp = int(ev["n_complete"])
    for n, a in attrs.items():
        rep = localization_report(a, task.affected_test, task.kind_test, kinds)
        pk = {k: localization_report(a, task.affected_test, task.kind_test,
                                     [k])["auroc"] for k in kinds}
        per_kind[n] = pk
        # Every localisation row says whether that statistic is a
        # decomposition of the anomaly score or a diagnostic, and by how much
        # it misses additivity.  The completeness row below belongs to the
        # additive one alone; without this column the two claims read as one.
        additive = n in ADDITIVE_ATTRIBUTIONS
        gap = additivity_gap(pc, task.X_test[:n_comp], a[:n_comp])
        log.result(_row(cfg, "explain", n, loc_auroc=rep["auroc"],
                        prec_at_k=rep["prec_at_k"], n_windows=rep["n"],
                        additive=additive, **gap,
                        **{f"loc_auroc[{k}]": v for k, v in pk.items()}))

    # ── 2. completeness — of the CHAIN-RULE attribution, and of nothing else
    comp = completeness_error(pc, task.X_test[:n_comp])
    log.info(f"  completeness of {comp['attribution']}: max residual "
             f"{comp['max_residual_nats']:.2e} nats, mean "
             f"{comp['mean_residual_nats']:.2e}")
    log.info("  the other attributions are not decompositions of the score; "
             "their `mean_abs_gap_nats` column says how far each is from "
             "summing to it")
    log.result(_row(cfg, "explain", "PC chain-rule completeness", **comp))

    # ── 3. faithfulness ──────────────────────────────────────────────────
    curves: Dict[str, np.ndarray] = {}
    if ev["deletion"]:
        sel = np.array([k in kinds and bool(a) for k, a
                        in zip(task.kind_test, task.affected_test)])
        if sel.any():
            Xs = task.X_test[torch.from_numpy(sel)]
            for n, a in attrs.items():
                c, auc = deletion_curve(pc.score, Xs, a[sel], task.window,
                                        task.n_channels, reference=task.X_train)
                curves[n] = c
                log.result(_row(cfg, "explain", n, deletion_auc=auc))
            log.info("  deletion curves scored with the circuit's own scorer — "
                     "PC attributions have home-field advantage here, the "
                     "localisation columns do not")

    # ── figures + worked examples ────────────────────────────────────────
    if ev["plots"] or ev["examples"]:
        fig_dir = os.path.join(log.artifacts_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        if ev["plots"]:
            summary = {n: {"auroc": localization_report(
                a, task.affected_test, task.kind_test, kinds)["auroc"]}
                for n, a in attrs.items()}
            plot_localization(summary, os.path.join(fig_dir, "localization.png"))
            if curves:
                plot_deletion_curves(curves, os.path.join(fig_dir, "deletion.png"))
        if ev["examples"]:
            _worked_examples(pc, task, kinds, fig_dir, plots=bool(ev["plots"]),
                             log=log)

    if ev["save_scores"]:
        log.artifact_npz("attributions",
                         **{k.replace(" ", "_")[:40]: v for k, v in attrs.items()})
        if curves:
            log.artifact_json("deletion_curves",
                              {k: np.asarray(v).tolist() for k, v in curves.items()})
    return {"per_kind": per_kind, "completeness": comp,
            "cost_s": {"pc_exact": pc_exact_s,
                       "replacement_sensitivity": rs_s}}


def _worked_examples(pc, task, kinds: Sequence[str], fig_dir: str,
                     plots: bool, log: RunLogger) -> None:
    """One MEDIAN-scoring window per kind — extremes would flatter everything."""
    log.info("  worked examples (median-scoring window of each kind):")
    for kind in ["normal", *kinds]:
        idxs = [i for i, k in enumerate(task.kind_test) if k == kind]
        if not idxs:
            continue
        s = pc.score(task.X_test[idxs]).numpy()
        i = idxs[int(np.argsort(s)[len(s) // 2])]
        x = task.X_test[i]
        exp = explain_window(pc, x, top=3)
        truth = task.affected_test[i] or None
        log.info(f"   [{kind}]")
        for line in format_explanation(exp, truth=truth, kind=kind).splitlines():
            log.info(line)
        if plots:
            plot_case_study(exp, x, task.window, task.n_channels, truth, kind,
                            os.path.join(fig_dir, f"case_{kind}.png"))


# ═══════════════════════════════════════════════════════════════════════════
# Stage: RUL / survival
# ═══════════════════════════════════════════════════════════════════════════

def _fit_survival(cfg: Dict[str, Any], task, seed: int, log: RunLogger,
                  use_censored: bool, tag: str) -> SurvivalPC:
    m = _mcfg(cfg)
    t0 = time.time()
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method=m["vtree"],
                    n_sum_components=int(m["rul_K"] or m["K"]),
                    leaf_components=int(m["leaf_components"]),
                    tau_where=m["tau_where"], delta=bool(m["delta"]),
                    channel_groups=task.channel_groups,
                    weight_jitter=float(m["weight_jitter"]),
                    seed=seed, device=cfg.get("device"))
    pc.fit(task.X_train, task.tau_train, task.delta_train,
           epochs=int(m["rul_epochs"] or m["epochs"]), lr=float(m["lr"]),
           batch_size=int(m["batch_size"]), use_censored=use_censored,
           log_every=max(int(m["rul_epochs"] or m["epochs"]) // 8, 1))
    log.history(f"{tag}_nll", pc.history)
    pc.fit_seconds = time.time() - t0                 # type: ignore[attr-defined]
    log.info(f"  {tag}: fit {pc.fit_seconds:.1f}s · {pc.size()['parameters']:,} params")
    return pc


def _eval_survival(pc: SurvivalPC, task, alpha: float
                   ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Returns (metrics, prediction) — the prediction so the caller can persist
    it, because the one question this stage could not answer from its own logs
    was "what would the coverage have been under a different endpoint
    convention?" (hand-off §B.2).  Scalars only is a false economy.

    Three interval columns, deliberately:

      picp / mpiw            bin CENTRES — what every recorded number used
      picp_edge / mpiw_edge  bin EDGES — the interval the pmf actually claims
      pit_*                  the density's own calibration, with no
                             discrete-vs-continuous mismatch in it at all

    All interval columns are taken at the endpoints of the REQUESTED level and
    the level is recorded next to them in `nominal_level`.  They used to read
    `q05`/`q95` unconditionally while passing α into the interval-score
    penalty, so an α=0.20 row scored the 90% endpoints under an 80% penalty and
    called the result an 80% interval.  At α=0.10 — every recorded run — the
    endpoints are the same ones as before and the numbers are unchanged.

    Read them together.  A large picp_edge − picp gap with pit_var near 1/12
    means the model was fine and the interval was being read wrong; a low
    picp_edge with pit_var well above 1/12 means the predictive really is
    overconfident.  Reporting only the first column cannot distinguish these,
    which is how "exact != calibrated" got as far as it did.
    """
    pred = pc.predict(task.X_test, alpha=alpha)       # raises if degenerate (§3)
    true = task.rul_test
    bw = task.cap / task.n_bins
    m = {
        "rmse": rmse(pred["mean"], true), "mae": mae(pred["mean"], true),
        "nasa": nasa_score(pred["mean"], true),
        "crps": crps_from_pmf(pred["pmf"], task.tau_test, bw),
        "nominal_level": round(1.0 - alpha, 4),
        "picp": picp(pred["q_lo"], pred["q_hi"], true),
        "mpiw": mpiw(pred["q_lo"], pred["q_hi"]),
        "interval_score": crps_from_interval(pred["q_lo"], pred["q_hi"], true, alpha),
        "picp_edge": picp(pred["q_lo_edge"], pred["q_hi_edge"], true),
        "mpiw_edge": mpiw(pred["q_lo_edge"], pred["q_hi_edge"]),
        "interval_score_edge": crps_from_interval(
            pred["q_lo_edge"], pred["q_hi_edge"], true, alpha),
        "calib_err": calibration_error(pred["pmf"], task.tau_test),
        "pred_sd": float(pred["mean"].std()),
    }
    m.update(pit_report(pred["pmf"], task.tau_test))
    return m, pred


def _test_protocol_views(task, protocols: Sequence[str]):
    """
    (name, task_view) for each test-window protocol, from ONE task.

    "all" scores every window of every test unit; "last" scores only the final
    window per unit (the literature protocol).  Both are TEST-TIME selections —
    the training windows are bit-identical — so running them as separate config
    variants retrained the same circuit twice.  Verified equal to rebuilding
    the task with `rul_test_windows: last`.
    """
    import copy as _copy

    views = []
    for name in protocols:
        if name == "all":
            views.append(("all", task))
            continue
        if task.unit_test is None:
            raise ValueError("rul_test_windows='last' needs per-window unit ids")
        u = task.unit_test.numpy()
        idx = torch.as_tensor(
            [int(np.where(u == unit)[0][-1]) for unit in dict.fromkeys(u.tolist())],
            dtype=torch.long)
        v = _copy.copy(task)
        v.X_test = task.X_test[idx]
        v.tau_test = task.tau_test[idx]
        v.rul_test = task.rul_test[idx]
        v.regime_test = task.regime_test[idx]
        v.unit_test = task.unit_test[idx]
        views.append(("last", v))
    return views


def stage_rul(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    ev = _ecfg(cfg)
    alpha = float(ev["alpha"])
    pair, task = prepare_task(cfg, seed, log, "rul")
    out: Dict[str, Any] = {}
    kept: Optional[SurvivalPC] = None
    # One fit, evaluated under every test-window protocol asked for.
    protocols = list(ev.get("test_protocols") or [_dcfg(cfg).get("rul_test_windows", "all")])
    views = _test_protocol_views(task, protocols)

    # ── A. the censoring ablation: same model, same budget, one term differs
    arms = [(True, "SurvivalPC (exact censored lik.)")]
    if ev["censoring_ablation"]:
        arms.insert(0, (False, "SurvivalPC (drop censored)"))
    censored_frac = float(1.0 - task.delta_train.float().mean())
    for use_c, label in arms:
        if not use_c and censored_frac <= 0:
            continue                                   # nothing to drop
        try:
            tag = "surv_censored" if use_c else "surv_dropped"
            pc = _fit_survival(cfg, task, seed, log, use_c, tag=tag)
            for pname, view in views:
                ptag = f" [{pname}]" if len(views) > 1 else ""
                r, pred = _eval_survival(pc, view, alpha)
                log.result(_row(cfg, "rul", f"{label}{ptag}", **r,
                                fit_s=pc.fit_seconds,
                                params=pc.size()["parameters"],
                                test_protocol=pname,
                                censored_frac=censored_frac))
                log.info(f"  {label}{ptag}: PICP {r['picp']:.3f} centres / "
                         f"{r['picp_edge']:.3f} edges (nominal {1 - alpha:.2f}), "
                         f"MPIW {r['mpiw']:.1f} / {r['mpiw_edge']:.1f}; "
                         f"PIT mean {r['pit_mean']:.3f} var {r['pit_var']:.4f} "
                         f"(1/12 = {1/12:.4f})")
                # The whole predictive, not just its summaries.  Without this
                # the endpoint question of §B.2 could not be re-asked without a
                # full re-run — which is exactly what happened.
                log.artifact_npz(
                    f"rul_pred_{tag}_{pname}",
                    pmf=pred["pmf"].numpy(), mean=pred["mean"].numpy(),
                    q05=pred["q05"].numpy(), q95=pred["q95"].numpy(),
                    q05_edge=pred["q05_edge"].numpy(),
                    q95_edge=pred["q95_edge"].numpy(),
                    # the endpoints this run's metrics were actually scored at
                    alpha=float(alpha),
                    q_lo=pred["q_lo"].numpy(), q_hi=pred["q_hi"].numpy(),
                    q_lo_edge=pred["q_lo_edge"].numpy(),
                    q_hi_edge=pred["q_hi_edge"].numpy(),
                    rul_true=view.rul_test.numpy(),
                    tau_true=view.tau_test.numpy(),
                    bin_edges=np.linspace(0.0, view.cap, view.n_bins + 1))
            if use_c:
                kept = pc
        except DegenerateModelError as exc:
            # A degenerate model is a FAILED run, not a row of numbers.
            log.info(f"  !! {label}: {exc}")
            log.result(_row(cfg, "rul", label, error="degenerate",
                            censored_frac=censored_frac))
            raise

    # ── B. baselines.  They cannot use censored units, by construction. ───
    if ev["baselines"]:
        keep = task.delta_train == 1
        bw = task.cap / task.n_bins
        Xb = task.X_train[keep]
        yb = (task.tau_train[keep].float() + 0.5) * bw
        for b in rul_baselines(seed=seed, alpha=alpha, device=cfg.get("device")):
            t0 = time.time()
            b.fit(Xb, yb)
            fit_s = time.time() - t0
            for pname, view in views:
                ptag = f" [{pname}]" if len(views) > 1 else ""
                pred = b.predict(view.X_test)
                r = {"rmse": rmse(pred["mean"], view.rul_test),
                     "mae": mae(pred["mean"], view.rul_test),
                     "nasa": nasa_score(pred["mean"], view.rul_test)}
                if "lo" in pred:
                    r.update({"picp": picp(pred["lo"], pred["hi"], view.rul_test),
                              "mpiw": mpiw(pred["lo"], pred["hi"]),
                              "interval_score": crps_from_interval(
                                  pred["lo"], pred["hi"], view.rul_test, alpha)})
                log.result(_row(cfg, "rul", f"{b.name}{ptag}", **r, fit_s=fit_s,
                                test_protocol=pname,
                                train_rows=int(keep.sum())))

    # ── C. query reach: survival under partial evidence ───────────────────
    if kept is not None and ev["partial_evidence"]:
        out["partial_evidence"] = _partial_evidence(kept, task, int(ev["n_dead"]),
                                                    cfg, log)
    if kept is not None and ev["survival_demo"]:
        out["survival"] = _survival_table(kept, task, log)
    return out


def _partial_evidence(pc: SurvivalPC, task, n_dead: int, cfg, log) -> Dict[str, float]:
    """
    Dead sensors: the circuit integrates them OUT of the joint exactly; every
    regressor and CQR must impute.  CQR cannot appear in this comparison at all
    — it needs a complete feature vector to emit an interval.
    """
    dead = list(range(0, task.n_channels, max(task.n_channels // 4, 1)))[:n_dead]
    marg = [t * task.n_channels + c for t in range(task.window) for c in dead]
    bw = task.cap / task.n_bins

    full = pc.predict(task.X_test)
    X_imp = task.X_test.reshape(-1, task.window, task.n_channels).clone()
    X_imp[:, :, dead] = 0.0
    X_imp = X_imp.reshape(len(task.X_test), -1)
    imp = pc.predict(X_imp)

    rows = []
    with torch.no_grad():
        for k in range(task.n_bins):
            z = pc._augment(task.X_test,
                            torch.full((len(task.X_test),), float(k),
                                       device=pc.device))
            rows.append(pc.pc.log_marginal(z, marg).cpu())
    joint = torch.stack(rows, dim=1)
    p = (joint - torch.logsumexp(joint, dim=1, keepdim=True)).exp()
    centers = pc.bin_centers()
    edges = pc.bin_edges()
    cdf = p.cumsum(1)
    q_idx = lambda lv: (cdf < lv).sum(1).clamp(max=task.n_bins - 1)
    lo_i, hi_i = q_idx(0.05), q_idx(0.95)

    res = {
        "n_dead": len(dead),
        # This comparison is fixed at 90%: all three arms use the same 5/95
        # endpoints, so the level is a property of the table, not of a config
        # knob.  Stated rather than implied, since the columns are named picp.
        "nominal_level": 0.90,
        "crps_full": crps_from_pmf(full["pmf"], task.tau_test, bw),
        "crps_exact_marginal": crps_from_pmf(p, task.tau_test, bw),
        "crps_imputed": crps_from_pmf(imp["pmf"], task.tau_test, bw),
        "rmse_full": rmse(full["mean"], task.rul_test),
        "rmse_exact_marginal": rmse((p * centers).sum(1), task.rul_test),
        "rmse_imputed": rmse(imp["mean"], task.rul_test),
        # centres and edges, as everywhere else (§B.2) — this is the row that
        # recorded picp_exact_marginal 0.32 on real C-MAPSS
        "picp_exact_marginal": picp(centers[lo_i], centers[hi_i], task.rul_test),
        "picp_exact_marginal_edge": picp(edges[lo_i], edges[hi_i + 1],
                                         task.rul_test),
        "picp_imputed": picp(imp["q05"], imp["q95"], task.rul_test),
        "picp_imputed_edge": picp(imp["q05_edge"], imp["q95_edge"],
                                  task.rul_test),
        **{f"marginal_{k}": v for k, v in
           pit_report(p, task.tau_test).items()},
    }
    # own stage name: its metrics are full/marginal/imputed triplets, which
    # would sit as empty cells in the main RUL table
    log.result(_row(cfg, "rul_partial", f"PC · {len(dead)} dead sensors", **res))
    log.info(f"  partial evidence ({len(dead)} dead): CRPS exact-marginal "
             f"{res['crps_exact_marginal']:.3f} vs imputed {res['crps_imputed']:.3f} "
             f"(full {res['crps_full']:.3f})")
    return res


def _survival_table(pc: SurvivalPC, task, log,
                    horizons=(20, 40, 60)) -> Dict[str, Any]:
    """S(t|x) bucketed by TRUE remaining life — the qualitative check."""
    bw = task.cap / task.n_bins
    true = task.rul_test.numpy()
    surv = {h: pc.log_survival(task.X_test, h / bw - 1.0).exp().numpy()
            for h in horizons}
    table: Dict[str, Any] = {}
    log.info("  exact survival S(t|x), grouped by TRUE remaining life:")
    for lo, hi in [(0, 20), (20, 50), (50, 90), (90, int(task.cap) + 1)]:
        sel = (true >= lo) & (true < hi)
        if not sel.any():
            continue
        row = {f"S({h})": float(surv[h][sel].mean()) for h in horizons}
        row["n"] = int(sel.sum())
        table[f"{lo}-{hi}"] = row
        cells = "  ".join(f"S({h})={row[f'S({h})']:.3f}" for h in horizons)
        log.info(f"    RUL {lo:>3}-{hi:<3} n={row['n']:>5}  {cells}")
    return table


# ═══════════════════════════════════════════════════════════════════════════
# Stage: conformal calibration of the circuit's own predictive
# ═══════════════════════════════════════════════════════════════════════════

def stage_calibration(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    """
    Exact density + guaranteed coverage.

    The circuit is refit on a subset of the training UNITS; the held-out units
    are the conformal calibration set.  Splitting by unit rather than by window
    is not pedantry — overlapping windows of one engine are near-duplicates, and
    calibrating on them would report a coverage that does not transfer to a new
    engine, which is the only coverage anyone cares about.
    """
    ev = _ecfg(cfg)
    # ONE fit, MANY alphas.  alpha never enters training — it selects a
    # quantile of the conformal scores at the very end — so putting it in the
    # config grid retrained a bit-identical circuit once per alpha.  On real
    # C-MAPSS that was ~37 min of GPU per duplicate, half the calibration tier.
    alphas = [float(a) for a in (ev.get("alphas") or [ev["alpha"]])]
    pair, task = prepare_task(cfg, seed, log, "rul")

    if task.unit_train is None:
        raise ValueError("calibration stage needs per-window unit ids "
                         "(RULTask.unit_train); rebuild the task")
    fit_mask, cal_mask = split_units(task.unit_train, float(ev["cal_frac"]), seed)
    # censored windows carry no point label, so they cannot calibrate
    cal_mask = cal_mask & (task.delta_train.numpy() == 1)
    if cal_mask.sum() < 20:
        raise ValueError(f"only {int(cal_mask.sum())} calibration windows — "
                         "raise eval.cal_frac or use more units")
    log.info(f"  conformal split: {int(fit_mask.sum())} fit windows / "
             f"{int(cal_mask.sum())} calibration windows "
             f"({len(np.unique(task.unit_train.numpy()))} units total)")

    sub = _subset_rul_task(task, fit_mask)
    pc = _fit_survival(cfg, sub, seed, log, use_censored=True, tag="surv_conformal")

    bw = task.cap / task.n_bins
    sel = torch.from_numpy(cal_mask)
    X_cal = task.X_train[sel]
    # calibrate against TRUE cycles when the task carries them; the binned
    # target would make the coverage guarantee a statement about a rounded
    # quantity (bin width = cap / n_bins cycles), which is not what is claimed
    y_cal = (task.rul_train[sel] if task.rul_train is not None
             else (task.tau_train[sel].float() + 0.5) * bw)
    true = task.rul_test
    out: Dict[str, Any] = {}

    # alpha goes in the METHOD NAME, not just a column: the aggregator groups
    # by method, so two alphas sharing a name would be averaged into a number
    # that means nothing.
    for alpha in alphas:
        tag = f" · a={alpha:.2f}" if len(alphas) > 1 else ""
        raw, _ = _eval_survival(pc, task, alpha)
        log.result(_row(cfg, "calibration",
                        f"SurvivalPC (raw exact predictive){tag}", alpha=alpha, **raw))
        out[f"raw@{alpha}"] = raw
        # The comparison this stage exists for now has three arms, not two:
        # raw-on-centres, raw-on-EDGES, and conformal.  If conformal only
        # matches the edge arm it is buying nothing but the half-bin (§B.2).
        log.info(f"  raw exact predictive{tag}: PICP {raw['picp']:.3f} centres / "
                 f"{raw['picp_edge']:.3f} edges, MPIW {raw['mpiw']:.1f} / "
                 f"{raw['mpiw_edge']:.1f}")

        for mode in ev["conformal_modes"]:
            cp = ConformalPredictive(pc, alpha=alpha, mode=mode).calibrate(X_cal, y_cal)
            pred = cp.predict(task.X_test)
            r = {
                "rmse": rmse(pred["mean"], true), "mae": mae(pred["mean"], true),
                "crps": crps_from_pmf(torch.as_tensor(pred["pmf"]), task.tau_test, bw),
                "picp": picp(pred["lo"], pred["hi"], true),
                "mpiw": mpiw(pred["lo"], pred["hi"]),
                "interval_score": crps_from_interval(pred["lo"], pred["hi"], true, alpha),
                **{f"diag_{k}": v for k, v in cp.diagnostics.items()},
            }
            log.result(_row(cfg, "calibration",
                            f"SurvivalPC + split conformal ({mode}){tag}",
                            alpha=alpha, **r))
            log.info(f"  conformal[{mode}] a={alpha:.2f}: PICP {r['picp']:.3f} "
                     f"(nominal {1 - alpha:.2f}), MPIW {r['mpiw']:.1f} "
                     f"(raw PICP {raw['picp']:.3f}, MPIW {raw['mpiw']:.1f})")
            out[f"{mode}@{alpha}"] = r

        # the adversary, trained on exactly the same fit windows.  It DOES
        # depend on alpha (pinball loss + conformal width), so it is refit per
        # alpha — it is an MLP, which costs seconds, not the circuit.
        if ev["baselines"]:
            keep = sub.delta_train == 1
            yb = (sub.tau_train[keep].float() + 0.5) * bw
            for b in rul_baselines(seed=seed, alpha=alpha, device=cfg.get("device")):
                if "conformal" not in b.name.lower():
                    continue
                b.fit(sub.X_train[keep], yb)
                pred = b.predict(task.X_test)
                log.result(_row(cfg, "calibration", f"{b.name}{tag}", alpha=alpha,
                                rmse=rmse(pred["mean"], true),
                                picp=picp(pred["lo"], pred["hi"], true),
                                mpiw=mpiw(pred["lo"], pred["hi"]),
                                interval_score=crps_from_interval(
                                    pred["lo"], pred["hi"], true, alpha)))
    return out


def _subset_rul_task(task, mask: np.ndarray):
    """Shallow copy of a RULTask restricted to a subset of TRAINING windows."""
    import copy as _copy
    sel = torch.from_numpy(np.asarray(mask))
    sub = _copy.copy(task)
    sub.X_train = task.X_train[sel]
    sub.tau_train = task.tau_train[sel]
    sub.delta_train = task.delta_train[sel]
    sub.regime_train = task.regime_train[sel]
    if task.unit_train is not None:
        sub.unit_train = task.unit_train[sel]
    return sub


# ═══════════════════════════════════════════════════════════════════════════
# Stage: structure gate — Tier 1.2, the kill gate
# ═══════════════════════════════════════════════════════════════════════════
#
# The 2026-08-06 re-measurement found that channel BLOCKING hurt real-data
# detection (removing it gained +0.040 AUROC and 27 nats) and that Chow-Liu at
# one leaf component topped the FD001 sweep.  Tier 1's queries all need
# blocking.  So the frontier is measured BEFORE anything is built on it, at
# matched parameters, on the validation split — and the tolerable loss is
# written down in the config, before the run, where it can be read back.
#
# What this stage does NOT do: touch the test set.  A structure chosen on the
# test split makes every later number a selection artefact (plan §3.3).

def _val_halves(task) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the validation windows in two, ENGINE-disjoint where unit ids exist.

    Half A carries the held-out likelihood and the checkpoint choice; half B is
    contaminated to give the gate a labelled detection score.  Keeping them
    disjoint stops the arm that overfits the injector from also picking its own
    checkpoint on the same windows.
    """
    X = task.X_val
    units = getattr(task, "unit_val", None)
    if X is None or not len(X):
        raise ValueError(
            "the structure gate needs a validation split (dataset.val_units > 0): "
            "choosing a structure on the test set is what Tier 0 removed")
    if units is None:
        cut = len(X) // 2
        idx_a = torch.arange(0, cut)
        idx_b = torch.arange(cut, len(X))
    else:
        uids = sorted(set(units.tolist()))
        first = set(uids[: max(len(uids) // 2, 1)])
        sel = torch.tensor([int(u) in first for u in units.tolist()])
        idx_a, idx_b = torch.nonzero(sel).ravel(), torch.nonzero(~sel).ravel()
        if not len(idx_b):                     # only one validation engine
            cut = len(X) // 2
            idx_a, idx_b = torch.arange(0, cut), torch.arange(cut, len(X))
    return X[idx_a], X[idx_b], idx_b


def channel_dependence(pc: WindowPC, X: torch.Tensor, n: int = 512) -> float:
    """Legacy gate diagnostic: mean absolute log-ratio to product marginals.

    This empirical finite-sample diagnostic is not total correlation, a
    dependence-accuracy measure, or a proof of global factorization. Retained
    under its original name for compatibility with historical gate artifacts.
    New studies record signed and absolute discrepancies separately.
    """
    W, C = pc.window, pc.n_channels
    Xp = pc._prep(X[:n])
    with torch.no_grad():
        joint = pc.pc.log_prob(Xp)
        tot = torch.zeros_like(joint)
        for c in range(C):
            others = [t * C + k for t in range(W) for k in range(C) if k != c]
            tot = tot + pc.pc.log_marginal(Xp, others)
    return float((tot - joint).abs().mean())


def stage_structure_gate(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    from .circuits import match_K, structure_param_count
    ev, m = _ecfg(cfg), _mcfg(cfg)
    pair, task = prepare_task(cfg, seed, log, "ad")
    X_fit_val, X_lab_val, _ = _val_halves(task)
    Xc, y_val, kinds_val, affected_val = contaminate_windows(
        X_lab_val, task.window, task.n_channels,
        inject_rate=float(ev["gate_inject_rate"]),
        strength=float(_dcfg(cfg)["strength"]),
        kinds=tuple(ev["kinds"]), donors=task.X_train, seed=seed + 1000)
    log.info(f"  gate: {len(X_fit_val)} clean val windows for NLL/checkpoint, "
             f"{len(Xc)} labelled val windows "
             f"({int(y_val.sum())} anomalous) for detection")

    ref = str(ev["gate_reference"])
    k_grid = [int(k) for k in ev["gate_k_grid"]]
    target = structure_param_count(task.window, task.n_channels, ref, int(m["K"]),
                                   leaf_components=int(m["leaf_components"]),
                                   X=task.X_train[:512],
                                   channel_groups=task.channel_groups, seed=seed)
    log.info(f"  gate: matching every structure to {ref} K={m['K']} = "
             f"{target:,} parameters")

    rows: List[Dict[str, Any]] = []
    for method in ev["gate_structures"]:
        K, n_par = match_K(task.window, task.n_channels, method, target,
                           k_grid=k_grid, leaf_components=int(m["leaf_components"]),
                           X=task.X_train[:512],
                           channel_groups=task.channel_groups, seed=seed)
        sub = copy.deepcopy(cfg)
        sub["model"]["vtree"], sub["model"]["K"] = method, K
        t0 = time.time()
        pc = _fit_window_pc(sub, task, seed, log, tag=f"gate_{method}",
                            X_checkpoint=X_fit_val)
        fit_s = time.time() - t0
        val_nll = _nll(pc, X_fit_val, n=4096)
        s = pc.score(Xc)
        rep = detection_report(s, y_val, kinds_val)
        dep = channel_dependence(pc, X_fit_val)
        row = dict(vtree=method, K=K, params=n_par,
                   param_ratio=round(n_par / max(target, 1), 4),
                   val_nll=val_nll, fit_s=round(fit_s, 1),
                   dependence_nats=dep,
                   blocked=bool(pc.is_channel_blocked),
                   best_epoch=pc.best_epoch, **rep)
        rows.append(row)
        log.result(_row(cfg, "structure_gate", f"{method} [K={K}]", **row))
        log.info(f"  {method:<16} K={K:<3} params {n_par:>8,}  "
                 f"val_nll {val_nll:8.3f}  val_auroc {rep['auroc']:.4f}  "
                 f"dependence {dep:7.3f} nats  blocked={row['blocked']}")

    # Per-seed, and labelled as such in the table: the decision is taken on the
    # seed AVERAGE by `run_tier1_gate.py`, which recomputes it from the arm rows
    # above.  A single seed's verdict is a progress indicator, not the gate.
    verdict = gate_verdict(rows, ev)
    log.result(_row(cfg, "structure_gate",
                    "VERDICT (this seed only — run_tier1_gate is authoritative)",
                    **verdict))
    log.metrics({"structure_gate_verdict": verdict})
    log.info(f"  GATE ({verdict['rule']}): {verdict['verdict']} — "
             f"{verdict['reason']}")
    return {"rows": rows, "verdict": verdict}


def gate_verdict(rows: Sequence[Dict[str, Any]], ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the PRE-REGISTERED budget to one seed's gate rows.

    Both halves must hold for the blocked structure to survive: it may lose at
    most `gate_max_auroc_loss` detection AUROC against the best unblocked arm,
    and its held-out NLL may exceed the best arm's by at most
    `gate_max_nll_loss_frac` (relative, because the nat scale moves with window
    length and channel count).  A relative NLL budget is the only one that
    transfers across datasets; an absolute one would silently mean something
    different on every subset.

    `poc/time_series/run_tier1_gate.py` applies the same rule to the seed
    AVERAGE, which is the number the decision is actually taken on — one seed
    is not a decision.
    """
    cand = str(ev["gate_candidate"])
    got = {r["vtree"]: r for r in rows}
    if cand not in got:
        return {"verdict": "ERROR", "rule": "n/a",
                "reason": f"candidate {cand!r} not among {sorted(got)}"}
    # The comparison set is the arms that are NOT channel-blocked, which is what
    # the pre-registration says and what the question is: what does BLOCKING
    # cost?  A blocked arm that beats the candidate says the channel ORDER is
    # wrong, not that the boundary is unaffordable — a different (and cheaper)
    # problem, and failing the gate on it would stop the wrong thing.
    #
    # [2026-09-08, mid-run] This originally compared against every other arm.
    # The mismatch with the config's own wording was found while the first
    # seed was still running and is corrected here; `delta_auroc_vs_any` keeps
    # the stricter number visible so nothing is hidden by the fix.
    rest = [r for r in rows if r["vtree"] != cand]
    others = [r for r in rest if not r.get("blocked", False)]
    if not rest:
        return {"verdict": "ERROR", "rule": "n/a",
                "reason": "no comparison structures were run"}
    # Before comparing structures, check there is anything for a structure to
    # do.  A circuit that carries no cross-channel dependence is a product of
    # per-channel marginals: every vtree gives the same function, so the
    # comparison is empty and so is every relational quantity built on it.
    floor = float(ev.get("gate_min_dependence_nats", 1e-3))
    # (computed before the comparator check so a VOID sweep is reported as VOID
    # rather than as a missing comparator)
    deps = {r["vtree"]: float(r.get("dependence_nats", float("nan"))) for r in rows}
    finite = [v for v in deps.values() if np.isfinite(v)]
    if finite and max(finite) < floor:
        return {"verdict": "VOID",
                "rule": f"max dependence >= {floor} nats before any comparison",
                "candidate": cand, "dependence_nats": deps,
                "reason": (f"every arm is factorised across channels "
                           f"(max dependence {max(finite):.2e} < {floor} nats), so "
                           "the structure comparison — and every relational "
                           "quantity below it — is identically empty")}
    if not others:
        return {"verdict": "NO_COMPARATOR",
                "rule": "compare against the best NON-blocked structure",
                "candidate": cand, "dependence_nats": deps,
                "blocked": {r["vtree"]: bool(r.get("blocked", False)) for r in rows},
                "reason": ("every arm in the sweep turned out to be "
                           "channel-blocked, so the budget has nothing to "
                           "measure the blocking against. That is itself a "
                           "result — the learners chose the boundary — but it "
                           "is not a PASS; add an arm that provably splits a "
                           "channel (e.g. 'time') and re-run")}
    best_auroc = max(float(r["auroc"]) for r in others)
    best_nll = min(float(r["val_nll"]) for r in others)
    any_auroc = max(float(r["auroc"]) for r in rest)
    any_nll = min(float(r["val_nll"]) for r in rest)
    d_auroc = float(got[cand]["auroc"]) - best_auroc
    d_nll = (float(got[cand]["val_nll"]) - best_nll) / max(abs(best_nll), 1e-9)
    max_loss = float(ev["gate_max_auroc_loss"])
    max_nll = float(ev["gate_max_nll_loss_frac"])
    auroc_ok = d_auroc >= -max_loss
    nll_ok = d_nll <= max_nll
    ok = auroc_ok and nll_ok
    return {
        "verdict": "PASS" if ok else "FAIL",
        "rule": (f"delta_auroc >= -{max_loss} and delta_nll_frac <= {max_nll}, "
                 f"against the best NON-blocked arm "
                 f"({', '.join(r['vtree'] for r in others)})"),
        "dependence_nats": deps,
        "candidate": cand,
        "delta_auroc": round(d_auroc, 5),
        "delta_nll_frac": round(d_nll, 5),
        "best_other_auroc": round(best_auroc, 5),
        "best_other_nll": round(best_nll, 4),
        "comparators": [r["vtree"] for r in others],
        # the stricter comparison, reported but NOT the rule: against every
        # other arm including the blocked ones
        "delta_auroc_vs_any": round(float(got[cand]["auroc"]) - any_auroc, 5),
        "delta_nll_frac_vs_any": round(
            (float(got[cand]["val_nll"]) - any_nll) / max(abs(any_nll), 1e-9), 5),
        "candidate_auroc": round(float(got[cand]["auroc"]), 5),
        "candidate_nll": round(float(got[cand]["val_nll"]), 4),
        # WHICH half broke.  The two can point in opposite directions — a
        # structure that buys density and sells detection is a different
        # situation from one that is simply worse, and a bare PASS/FAIL would
        # hide the difference from the person deciding what to do next.
        "auroc_ok": bool(auroc_ok),
        "nll_ok": bool(nll_ok),
        "failed_on": ([] if ok else
                      ([] if auroc_ok else ["detection AUROC"])
                      + ([] if nll_ok else ["held-out NLL"])),
        "reason": (f"{cand} is {d_auroc:+.4f} AUROC and {d_nll:+.2%} NLL from "
                   f"the best unblocked structure" + _budget_clause(auroc_ok, nll_ok)),
    }


def _budget_clause(auroc_ok: bool, nll_ok: bool) -> str:
    """Which half of the pre-registered budget held, in words."""
    if auroc_ok and nll_ok:
        return "; inside both budgets"
    if auroc_ok:
        return "; inside the detection budget, outside the NLL one"
    if nll_ok:
        return "; inside the NLL budget, outside the detection one"
    return "; outside both budgets"


# ═══════════════════════════════════════════════════════════════════════════
# Stage: relational diagnosis — Tier 1.3-1.6
# ═══════════════════════════════════════════════════════════════════════════

def _mask_localization(attr: np.ndarray, affected: Sequence[Sequence[int]],
                       kinds: Sequence[str], keep: Sequence[str],
                       mask: torch.Tensor) -> Dict[str, float]:
    """
    Localisation restricted to the sensors that are actually present.

    Two things must happen or the number is a fiction: a channel that is not
    observed can never be ranked (its score is nan), and a window whose
    corrupted channels are ALL dead has no findable truth left and is dropped
    rather than counted as a miss.  `n_unfindable` reports how many that was —
    it is a property of the mask, not of the method, and hiding it would make
    aggressive masks look easy.
    """
    obs = [c for c in range(len(mask)) if bool(mask[c])]
    a2, k2, rows = [], [], []
    unfindable = 0
    for i, (a, k) in enumerate(zip(affected, kinds)):
        if not a or k not in keep:
            continue
        vis = [c for c in a if c in obs]
        if not vis:
            unfindable += 1
            continue
        rows.append(attr[i, obs])
        a2.append([obs.index(c) for c in vis])
        k2.append(k)
    if not rows:
        return {"auroc": float("nan"), "prec_at_k": float("nan"), "n": 0,
                "n_unfindable": unfindable}
    out = localization_report(np.stack(rows), a2, k2, keep)
    out["n_unfindable"] = unfindable
    return out


def stage_relational(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    from .relational import (MaskCalibrator, mask_library, naive_subset_search,
                             subset_search)
    from .diagnosis import validation_partitions, one_per_unit
    ev = _ecfg(cfg)
    pair, task = prepare_task(cfg, seed, log, "ad")
    cap = int(ev["max_explain_windows"] or 0)
    if cap and len(task.X_test) > cap:
        idx = np.linspace(0, len(task.X_test) - 1, cap).astype(int)
        task.X_test, task.y_test = task.X_test[idx], task.y_test[idx]
        task.kind_test = [task.kind_test[i] for i in idx]
        task.affected_test = [task.affected_test[i] for i in idx]
        log.info(f"  relational: capped test set to {cap} windows")

    partitions = validation_partitions(task, seed + 701)
    pc = _fit_window_pc(cfg, task, seed, log,
                        X_checkpoint=task.X_val[partitions["checkpoint"]])
    if not pc.is_channel_blocked:
        raise BlockStructureError(
            f"stage 'relational' needs a channel-blocked structure; "
            f"model.vtree={_mcfg(cfg)['vtree']!r} is not one. The two-pass map "
            "is exact only through a single channel boundary (Tier 1.1).")
    out: Dict[str, Any] = {}
    y, kinds = task.y_test, task.kind_test
    keep = list(ev["kinds"])

    # ── 1.3 correctness: the two-pass map against the 3·C oracle ─────────
    n_chk = min(int(ev["oracle_check_windows"]), len(task.X_test))
    Xchk = task.X_test[:n_chk]
    t0 = time.time()
    td = pc.typed_scores(Xchk)
    oracle_s = time.time() - t0
    t0 = time.time()
    rm = pc.relational_map(Xchk)
    two_s = time.time() - t0
    diffs = {k: float((td[k] - rm[k]).abs().max()) for k in
             ("marginal", "conditional", "structural")}
    worst = max(diffs.values())
    log.result(_row(cfg, "relational", "two-pass vs 3C oracle",
                    max_abs_diff_nats=worst, oracle_s=round(oracle_s, 3),
                    two_pass_s=round(two_s, 3), n=n_chk,
                    speedup=round(oracle_s / max(two_s, 1e-9), 2),
                    **{f"diff_{k}": v for k, v in diffs.items()},
                    **{f"cost_{k}": v for k, v in rm["cost"].as_dict().items()}))
    log.info(f"  two-pass vs oracle: max |Δ| {worst:.2e} nats over {n_chk} windows "
             f"({oracle_s:.2f}s oracle → {two_s:.2f}s two-pass)")
    if worst > float(ev["oracle_tolerance"]):
        raise AssertionError(
            f"the two-pass relational map disagrees with the 3·C oracle by "
            f"{worst:.3e} nats (> {ev['oracle_tolerance']}). Everything "
            "downstream of it is void until that is explained.")
    out["oracle_check"] = {"max_abs_diff_nats": worst, **diffs}

    # a factorised circuit makes every relational number identically zero and
    # says nothing about it in the loss (Tier 0 finding, 2026-09-08)
    dep = float(np.abs(rm["marginal"].sum(1).numpy()
                       + rm["log_px"].numpy()).mean())
    log.result(_row(cfg, "relational", "dependence (Σ marginals − joint)",
                    dependence_nats=dep))
    log.info(f"  dependence |Σ log p(x_c) − log p(x)| = {dep:.4f} nats "
             f"({'FACTORISED — relational scores are empty' if dep < 1e-3 else 'ok'})")
    out["dependence_nats"] = dep

    # ── 1.4 masks: detection and localisation as sensors drop out ────────
    lib = mask_library(task.n_channels, task.channel_groups,
                       ks=tuple(int(k) for k in ev["mask_ks"]),
                       n_per_k=int(ev["masks_per_k"]), seed=seed)
    out["masks"] = list(lib)
    for name, mask in lib.items():
        s_obs = pc.score_with_masks(task.X_test, mask)
        cost_det = pc.last_query_cost.as_dict()
        r = pc.relational_map(task.X_test, mask=mask)
        attr = np.nan_to_num(r["structural"].numpy(), nan=-np.inf)
        det = detection_report(s_obs, y, kinds)
        loc = _mask_localization(attr, task.affected_test, kinds, keep, mask)
        struct_det = detection_report(
            torch.from_numpy(np.nanmax(r["structural"].numpy(), axis=1)), y, kinds)
        log.result(_row(cfg, "relational", f"mask {name}", mask=name,
                        n_observed=int(mask.sum()), **det,
                        struct_auroc=struct_det["auroc"],
                        loc_auroc=loc["auroc"], loc_prec_at_k=loc["prec_at_k"],
                        loc_n=loc["n"], loc_unfindable=loc["n_unfindable"],
                        **{f"cost_{k}": v for k, v in r["cost"].as_dict().items()},
                        det_passes=cost_det["passes"],
                        det_node_visits=cost_det["node_visits"]))
        log.info(f"  mask {name:<18} obs={int(mask.sum()):>2}  "
                 f"auroc {det['auroc']:.4f}  struct {struct_det['auroc']:.4f}  "
                 f"loc prec@k {loc['prec_at_k']:.3f} (n={loc['n']}, "
                 f"{loc['n_unfindable']} unfindable)")

    # ── 1.5 subset search, fast vs the per-candidate cost model ──────────
    rc = pc.relational()
    sel = [i for i, (a, k) in enumerate(zip(task.affected_test, kinds))
           if a and k in keep][: int(ev["subset_windows"])]
    if sel:
        Xs = task.X_test[torch.tensor(sel)]
        truth = [set(task.affected_test[i]) for i in sel]
        for name in ["full"] + [n for n in lib if n != "full"][: int(ev["subset_masks"])]:
            mask = lib[name]
            fast = subset_search(rc, pc._prep(Xs), mask=mask,
                                 max_size=int(ev["subset_max_size"]),
                                 beam=int(ev["subset_beam"]))
            vis = [t & {c for c in range(task.n_channels) if bool(mask[c])}
                   for t in truth]
            keep_i = [i for i, t in enumerate(vis) if t]
            exact = float(np.mean([set(fast.subsets[i]) == vis[i] for i in keep_i])) \
                if keep_i else float("nan")
            jac = float(np.mean([
                len(set(fast.subsets[i]) & vis[i]) /
                max(len(set(fast.subsets[i]) | vis[i]), 1) for i in keep_i])) \
                if keep_i else float("nan")
            row = dict(mask=name, n=len(keep_i), exact_set_match=exact,
                       jaccard=jac, candidates=fast.candidates_scored,
                       **{f"cost_{k}": v for k, v in fast.cost.as_dict().items()})
            if bool(ev["subset_naive_baseline"]):
                slow = naive_subset_search(pc, Xs, mask=mask,
                                           max_size=int(ev["subset_max_size"]),
                                           beam=int(ev["subset_beam"]))
                row["naive_passes"] = slow.cost.passes
                row["naive_seconds"] = round(slow.cost.seconds, 4)
                row["value_max_abs_diff"] = float(
                    np.nanmax(np.abs(fast.values - slow.values)))
                row["subset_agreement"] = float(np.mean(
                    [set(a) == set(b) for a, b in zip(fast.subsets, slow.subsets)]))
            log.result(_row(cfg, "relational", f"subset search · {name}", **row))
            log.info(f"  subset search {name:<18} exact {exact:.3f} jaccard "
                     f"{jac:.3f}  candidates {fast.candidates_scored}  "
                     f"passes {fast.cost.passes}"
                     + (f" vs naive {row.get('naive_passes')}"
                        if "naive_passes" in row else ""))
        out["subset_windows"] = len(sel)

    # ── 1.6 mask-conditional calibration ─────────────────────────────────
    cal_idx = one_per_unit(partitions["calibration"], task.unit_val, seed + 702)
    eval_idx = one_per_unit(partitions["evaluation"], task.unit_val, seed + 703)
    X_a, X_b = task.X_val[cal_idx], task.X_val[eval_idx]
    log.artifact_json("relational_calibration_protocol", {
        "sampling_object": "one_random_window_per_engine",
        "checkpoint_indices": partitions["checkpoint"].tolist(),
        "calibration_indices": cal_idx.tolist(), "evaluation_indices": eval_idx.tolist(),
        "mask_assumption": "externally_fixed"})
    cal = MaskCalibrator(rc, alpha=float(ev["relational_alpha"]))
    cal.fit(pc._prep(X_a), lib)
    rows = cal.report(pc._prep(X_b), lib)
    for r in rows:
        log.result(_row(cfg, "relational", f"calibration · {r['mask']}", **r))
    drift_blind = float(np.nanmax([abs(r["fpr_blind"] - r["alpha"]) for r in rows]))
    drift_mask = float(np.nanmax([abs(r["fpr_masked"] - r["alpha"]) for r in rows]))
    log.result(_row(cfg, "relational", "calibration · worst drift",
                    worst_abs_drift_masked=drift_mask,
                    worst_abs_drift_blind=drift_blind,
                    alpha=float(ev["relational_alpha"]),
                    n_masks=len(rows), n_fit=len(X_a), n_eval=len(X_b),
                    **{f"cost_{k}": v for k, v in cal.cost.as_dict().items()}))
    log.info(f"  calibration: worst |FPR − α| = {drift_mask:.3f} mask-conditional "
             f"vs {drift_blind:.3f} mask-blind (α={ev['relational_alpha']}, "
             f"{len(rows)} masks)")
    out["calibration"] = {"rows": rows, "worst_drift_masked": drift_mask,
                          "worst_drift_blind": drift_blind}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Stage: layout scaling (tree vs DAG)
# ═══════════════════════════════════════════════════════════════════════════

def stage_scaling(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    from .bench_scaling import bench
    K = int(_mcfg(cfg)["K"])
    rows = []
    for d in _ecfg(cfg)["scaling_dims"]:
        r = bench(int(d), K, batch=64)
        rows.append(r)
        log.result(_row(cfg, "scaling", f"d={d}", **r))
        log.info(f"  d={d:>5}  DAG leaves {r['dag_leaves']:>8,}  "
                 f"DAG params {r['dag_params']:>9,}  "
                 f"tree leaves {r['tree_leaves'] if r['tree_leaves'] is not None else '~' + format(r['tree_leaves_predicted'], '.1e') + ' (skipped)'}")
    return {"rows": rows}


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════

from .diagnosis import stage_diagnosis


STAGE_FNS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "ad": stage_ad,
    "structure_gate": stage_structure_gate,
    "relational": stage_relational,
    "diagnosis": stage_diagnosis,
    "explain": stage_explain,
    "rul": stage_rul,
    "calibration": stage_calibration,
    "scaling": stage_scaling,
}


def run_stages(cfg: Dict[str, Any], seed: int, log: RunLogger) -> Dict[str, Any]:
    """Run every configured stage for one seed, inside one run directory."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = resolve_device(cfg.get("device"))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    log.info(f"stages {cfg['stages']} · device {dev} · "
             f"dataset {dataset_id(_dcfg(cfg))}")

    results: Dict[str, Any] = {}
    for stage in cfg["stages"]:
        t0 = time.time()
        log.info(f"--- stage: {stage} ---")
        try:
            results[stage] = STAGE_FNS[stage](cfg, seed, log)
        except Exception as exc:
            # Mark the stage that died, then re-raise: the run still fails (the
            # batch runner and `is_complete` must keep seeing that), but the
            # stages that finished before it stay usable, and the aggregate can
            # tell "this run crashed in `rul`" from "this run's `ad` numbers
            # are partial".
            log.stage_failed(stage, f"{type(exc).__name__}: {exc}")
            raise
        log.stage_ok(stage)
        results.setdefault("_timing", {})[stage] = round(time.time() - t0, 2)
        log.info(f"--- stage {stage} done in {time.time() - t0:.1f}s ---")
    log.metrics(results)
    return results
