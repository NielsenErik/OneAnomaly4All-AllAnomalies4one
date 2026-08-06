"""
Experiment stages — the training/evaluation pipeline itself.

Five stages, all driven from one resolved config and all writing the same
structured rows, so anything they produce can be aggregated across datasets,
structures and seeds without special cases:

  ad            train a window density, score detection against the full
                baseline suite, run the two queries no baseline can express
                (dead sensors by exact marginalisation; the exact typed
                marginal/conditional/structural split)
  explain       the project's actual contribution: exact attribution vs the
                strong adversaries (Gaussian conditional, AE reconstruction,
                sampling-SHAP), scored on correctness / completeness /
                faithfulness against per-channel ground truth
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
    completeness_error,
    deletion_curve,
    explain_window,
    format_explanation,
    localization_report,
    pc_attributions,
    plot_case_study,
    plot_deletion_curves,
    plot_localization,
    sampling_shap,
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
                   tag: str = "pc") -> WindowPC:
    m = _mcfg(cfg)
    t0 = time.time()
    pc = WindowPC(task.window, task.n_channels, vtree_method=m["vtree"],
                  n_sum_components=int(m["K"]),
                  leaf_components=int(m["leaf_components"]),
                  channel_groups=task.channel_groups, use_sos=bool(m["sos"]),
                  delta=bool(m["delta"]), weight_jitter=float(m["weight_jitter"]),
                  seed=seed, device=cfg.get("device"),
                  evaluator=cfg.get("evaluator", "layered"))
    pc.fit(task.X_train, epochs=int(m["epochs"]), lr=float(m["lr"]),
           batch_size=int(m["batch_size"]), log_every=max(int(m["epochs"]) // 8, 1))
    fit_s = time.time() - t0
    log.history(f"{tag}_train_nll", pc.history)
    sd = pc.assert_informative(task.X_train)         # loud, not silent (§3)
    # windows/s makes the device AND evaluator choice auditable after the fact.
    # It matters: on the recursive evaluator a GPU run is legitimately slower
    # than CPU, on the layered one it is ~2× faster above batch 128, and the
    # only way to tell which regime a finished run was in is to log it.
    thr = len(task.X_train) * int(m["epochs"]) / max(fit_s, 1e-9)
    ev = "layered" if pc.compiled is not None else "recursive"
    log.info(f"  {tag}: fit {fit_s:.1f}s · {pc.size()['parameters']:,} params · "
             f"score sd {sd:.3f} · device {pc.device} · {ev} · {thr:,.0f} win/s")
    pc.fit_seconds = fit_s                            # type: ignore[attr-defined]
    return pc


def _held_out_nll(pc: WindowPC, X: torch.Tensor, n: int = 1024) -> float:
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
                    train_nll=_held_out_nll(pc, task.X_train)))
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
                                 shapley_orders=int(ev["shapley_orders"])))
    pc_exact_s = time.time() - t0
    attrs[gc.name] = gc.attribute(task.X_test)
    attrs["AE reconstruction (per channel)"] = ae_channel_error(ae, task.X_test)
    t0 = time.time()
    attrs[f"AE sampling-SHAP ({ev['shap_samples']}/ch)"] = sampling_shap(
        ae, task.X_test, task.X_train, n_samples=int(ev["shap_samples"]), seed=seed)
    shap_s = time.time() - t0
    attrs["z-score (per channel)"] = zscore_channel(zs, task.X_test)
    log.info(f"  attribution cost: PC exact (all views) {pc_exact_s:.1f}s · "
             f"AE sampling-SHAP (one view) {shap_s:.1f}s")

    kinds = list(ev["kinds"])
    present = {k for k in task.kind_test}
    kinds = [k for k in kinds if k in present]
    if not kinds:
        log.info("  no injected anomaly kinds in this test set — nothing to localise")
        return {}

    # ── 1. correctness ───────────────────────────────────────────────────
    per_kind: Dict[str, Dict[str, float]] = {}
    for n, a in attrs.items():
        rep = localization_report(a, task.affected_test, task.kind_test, kinds)
        pk = {k: localization_report(a, task.affected_test, task.kind_test,
                                     [k])["auroc"] for k in kinds}
        per_kind[n] = pk
        log.result(_row(cfg, "explain", n, loc_auroc=rep["auroc"],
                        prec_at_k=rep["prec_at_k"], n_windows=rep["n"],
                        **{f"loc_auroc[{k}]": v for k, v in pk.items()}))

    # ── 2. completeness (a theorem for the circuit, a target for SHAP) ────
    comp = completeness_error(pc, task.X_test[: int(ev["n_complete"])])
    log.info(f"  completeness: max residual {comp['max_residual_nats']:.2e} nats, "
             f"mean {comp['mean_residual_nats']:.2e}")
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
            "cost_s": {"pc_exact": pc_exact_s, "sampling_shap": shap_s}}


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

    Read them together.  A large picp_edge − picp gap with pit_var near 1/12
    means the model was fine and the interval was being read wrong; a low
    picp_edge with pit_var well above 1/12 means the predictive really is
    overconfident.  Reporting only the first column cannot distinguish these,
    which is how "exact != calibrated" got as far as it did.
    """
    pred = pc.predict(task.X_test)                    # raises if degenerate (§3)
    true = task.rul_test
    bw = task.cap / task.n_bins
    m = {
        "rmse": rmse(pred["mean"], true), "mae": mae(pred["mean"], true),
        "nasa": nasa_score(pred["mean"], true),
        "crps": crps_from_pmf(pred["pmf"], task.tau_test, bw),
        "picp": picp(pred["q05"], pred["q95"], true),
        "mpiw": mpiw(pred["q05"], pred["q95"]),
        "interval_score": crps_from_interval(pred["q05"], pred["q95"], true, alpha),
        "picp_edge": picp(pred["q05_edge"], pred["q95_edge"], true),
        "mpiw_edge": mpiw(pred["q05_edge"], pred["q95_edge"]),
        "interval_score_edge": crps_from_interval(
            pred["q05_edge"], pred["q95_edge"], true, alpha),
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

STAGE_FNS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "ad": stage_ad,
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
        results[stage] = STAGE_FNS[stage](cfg, seed, log)
        results.setdefault("_timing", {})[stage] = round(time.time() - t0, 2)
        log.info(f"--- stage {stage} done in {time.time() - t0:.1f}s ---")
    log.metrics(results)
    return results
