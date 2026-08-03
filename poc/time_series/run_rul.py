"""
PoC part 3 — remaining useful life as an EXACT QUERY, with right censoring.

This is the part of the proposal that is not a re-skin of the tabular work.
One joint circuit over (window, τ) is trained once, with the exact censored
likelihood, and then answers four different questions with no retraining:

    p(τ | x)            full predictive distribution over remaining life
    E[τ | x]            point RUL
    S(t | x) = P(τ > t) survival / "will it last another t cycles?"
    −log p(x)           the anomaly score, τ marginalised out

Three comparisons are run, each of which can falsify the claim:

  A. CENSORING.  Same circuit, trained (i) on failures only — what a regressor
     is forced to do — and (ii) on failures plus the exact P(τ ≥ c) term for
     censored units.  If (ii) does not help, censoring handling buys nothing.
  B. CALIBRATION.  Against conformalised quantile regression, the field's
     current answer to calibrated RUL intervals.  CQR has a finite-sample
     coverage guarantee, so matching it is the bar; beating it on CRPS while
     matching coverage is the claim.
  C. QUERY REACH.  Survival under PARTIAL EVIDENCE (some sensors dead), which
     CQR cannot express at all: it needs a complete feature vector to emit an
     interval, so it must impute, while the circuit marginalises exactly.

Run:
    PYTHONPATH=. python -m poc.time_series.run_rul
    PYTHONPATH=. python -m poc.time_series.run_rul --seeds 0 1 2 --epochs 60
    PYTHONPATH=. python -m poc.time_series.run_rul --survival-demo
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .baselines import rul_baselines
from .circuits import SurvivalPC
from .data import make_rul_task
from .metrics import (
    calibration_error,
    crps_from_interval,
    crps_from_pmf,
    mae,
    mpiw,
    nasa_score,
    picp,
    rmse,
)


def _bin_width(task) -> float:
    return task.cap / task.n_bins


def fit_circuit(task, args, seed: int, use_censored: bool) -> SurvivalPC:
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method=args.vtree, n_sum_components=args.K,
                    leaf_components=args.leaf_components,
                    tau_where=args.tau_where, delta=args.delta,
                    channel_groups=task.channel_groups, seed=seed)
    pc.fit(task.X_train, task.tau_train, task.delta_train,
           epochs=args.epochs, lr=args.lr, use_censored=use_censored,
           verbose=args.verbose)
    return pc


def eval_circuit(pc: SurvivalPC, task, alpha: float = 0.10) -> Dict[str, float]:
    pred = pc.predict(task.X_test)
    true = task.rul_test
    return {
        "rmse": rmse(pred["mean"], true),
        "mae": mae(pred["mean"], true),
        "nasa": nasa_score(pred["mean"], true),
        "crps": crps_from_pmf(pred["pmf"], task.tau_test, _bin_width(task)),
        "picp": picp(pred["q05"], pred["q95"], true),
        "mpiw": mpiw(pred["q05"], pred["q95"]),
        "interval_score": crps_from_interval(pred["q05"], pred["q95"], true, alpha),
        "calib_err": calibration_error(pred["pmf"], task.tau_test),
    }


def eval_baseline(model, task, alpha: float = 0.10) -> Dict[str, float]:
    pred = model.predict(task.X_test)
    true = task.rul_test
    out = {"rmse": rmse(pred["mean"], true), "mae": mae(pred["mean"], true),
           "nasa": nasa_score(pred["mean"], true)}
    if "lo" in pred:
        out.update({
            "picp": picp(pred["lo"], pred["hi"], true),
            "mpiw": mpiw(pred["lo"], pred["hi"]),
            "interval_score": crps_from_interval(pred["lo"], pred["hi"], true, alpha),
        })
    return out


def run_once(args, seed: int) -> Dict[str, Dict[str, float]]:
    task = make_rul_task(window=args.window, stride=args.stride, seed=seed,
                         n_bins=args.bins, cap=args.cap, n_units=args.units,
                         n_channels=args.channels, n_regimes=args.regimes,
                         censor_frac=args.censor_frac)
    print(f"  {task}")
    print(f"    units: {task.meta['train_units']} train "
          f"({task.meta['censored_units']} censored), "
          f"{task.meta['test_units']} test")
    res: Dict[str, Dict[str, float]] = {}

    # A. censoring ablation — same model, same budget, one term differs
    for use_cens, label in [(False, "SurvivalPC (drop censored)"),
                            (True, "SurvivalPC (exact censored lik.)")]:
        t0 = time.time()
        pc = fit_circuit(task, args, seed, use_censored=use_cens)
        r = eval_circuit(pc, task, args.alpha)
        r["fit_s"] = time.time() - t0
        r["params"] = pc.size()["parameters"]
        res[label] = r
        if use_cens:
            res["_pc"] = pc            # keep for the partial-evidence query

    # B. baselines.  They cannot use censored units, so they see the
    #    failure-observed subset — the same data as the ablation's first arm.
    keep = task.delta_train == 1
    Xb, yb = task.X_train[keep], task.rul_test.new_tensor(
        (task.tau_train[keep].float() + 0.5) * _bin_width(task))
    for b in rul_baselines(seed=seed, alpha=args.alpha):
        t0 = time.time()
        b.fit(Xb, yb)
        r = eval_baseline(b, task, args.alpha)
        r["fit_s"] = time.time() - t0
        res[b.name] = r

    return res, task


def partial_evidence_demo(pc: SurvivalPC, task, n_dead: int = 3) -> Dict[str, float]:
    """
    C. Survival with dead sensors.  The circuit marginalises them out exactly;
    every baseline must impute.  Reported as the degradation in CRPS/coverage
    from the fully observed case.
    """
    dead = list(range(0, task.n_channels, max(task.n_channels // 4, 1)))[:n_dead]
    marg = [t * task.n_channels + c for t in range(task.window) for c in dead]

    full = pc.predict(task.X_test)
    X_imp = task.X_test.clone().reshape(-1, task.window, task.n_channels)
    X_imp[:, :, dead] = 0.0
    X_imp = X_imp.reshape(len(task.X_test), -1)
    imp = pc.predict(X_imp)

    # exact: marginalise the dead sensors OUT of the joint, then renormalise
    # over τ.  Nothing is filled in, so the predictive stays a valid density.
    rows = []
    with torch.no_grad():
        for k in range(task.n_bins):
            z = pc._augment(task.X_test, torch.full((len(task.X_test),), float(k)))
            rows.append(pc.pc.log_marginal(z, marg))
    joint = torch.stack(rows, dim=1)
    logp = joint - torch.logsumexp(joint, dim=1, keepdim=True)
    p = logp.exp()
    centers = pc.bin_centers()
    cdf = p.cumsum(1)
    q = lambda lv: centers[(cdf < lv).sum(1).clamp(max=task.n_bins - 1)]
    bw = _bin_width(task)
    return {
        "crps_full": crps_from_pmf(full["pmf"], task.tau_test, bw),
        "crps_exact_marginal": crps_from_pmf(p, task.tau_test, bw),
        "crps_imputed": crps_from_pmf(imp["pmf"], task.tau_test, bw),
        "rmse_full": rmse(full["mean"], task.rul_test),
        "rmse_exact_marginal": rmse((p * centers).sum(1), task.rul_test),
        "rmse_imputed": rmse(imp["mean"], task.rul_test),
        "picp_exact_marginal": picp(q(0.05), q(0.95), task.rul_test),
        "picp_imputed": picp(imp["q05"], imp["q95"], task.rul_test),
        "n_dead": len(dead),
    }


def survival_demo(pc: SurvivalPC, task, horizons=(20, 40, 60)) -> None:
    """S(t|x) at several horizons, bucketed by true remaining life."""
    print("\n=== exact survival function  S(t | x) = P(τ > t | x) ===")
    print("mean predicted survival probability, grouped by TRUE remaining life\n")
    bw = _bin_width(task)
    true = task.rul_test.numpy()
    buckets = [(0, 20), (20, 50), (50, 90), (90, int(task.cap) + 1)]
    header = f"{'true RUL':>12} {'n':>6} " + " ".join(
        f"{'S(' + str(h) + ')':>9}" for h in horizons)
    print(header)
    print("-" * len(header))
    surv = {h: pc.log_survival(task.X_test, h / bw - 1.0).exp().numpy()
            for h in horizons}
    for lo, hi in buckets:
        sel = (true >= lo) & (true < hi)
        if not sel.any():
            continue
        cells = " ".join(f"{surv[h][sel].mean():>9.3f}" for h in horizons)
        print(f"{f'{lo}–{hi}':>12} {int(sel.sum()):>6} {cells}")
    print("\nA calibrated model gives S(t) → 1 for units far from failure and "
          "S(t) → 0 for\nunits about to fail.  This is one query on the density "
          "that also produced the\npoint RUL and the anomaly score.\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--channels", type=int, default=14)
    ap.add_argument("--units", type=int, default=60)
    ap.add_argument("--regimes", type=int, default=3)
    ap.add_argument("--censor-frac", type=float, default=0.35)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--vtree", default="time")
    ap.add_argument("--tau-where", default="root", choices=["root", "deep"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--leaf-components", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.10, help="1−nominal coverage")
    ap.add_argument("--n-dead", type=int, default=3)
    ap.add_argument("--delta", action="store_true",
                    help="first-difference the window (unit Jacobian)")
    ap.add_argument("--per-regime", action="store_true", default=True)
    ap.add_argument("--survival-demo", action="store_true")
    ap.add_argument("--no-partial", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="logs/poc_ts_rul.json")
    args = ap.parse_args(argv)

    runs, partials, last_pc, last_task = [], [], None, None
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        res, task = run_once(args, seed)
        last_pc, last_task = res.pop("_pc"), task
        runs.append(res)
        if not args.no_partial:
            partials.append(partial_evidence_demo(last_pc, task, args.n_dead))

    cols = ["rmse", "mae", "nasa", "crps", "interval_score", "picp", "mpiw", "calib_err"]
    print(f"\n\n=== RUL, mean over {len(args.seeds)} seed(s) "
          f"(nominal coverage {1 - args.alpha:.0%}) ===\n")
    head = f"{'model':>34} " + " ".join(f"{c:>9}" for c in cols)
    print(head)
    print("-" * len(head))
    summary = {}
    for k in runs[0]:
        row = {}
        for c in cols:
            vals = [r[k][c] for r in runs if c in r[k]]
            row[c] = float(np.mean(vals)) if vals else None
        summary[k] = row
        cells = " ".join(f"{row[c]:>9.3f}" if row[c] is not None else f"{'—':>9}"
                         for c in cols)
        print(f"{k:>34} {cells}")

    print("\n  rmse/mae/nasa: point accuracy (lower better).  crps/interval_score: "
          "distributional\n  quality (lower better).  picp: should be close to "
          f"{1 - args.alpha:.2f}; mpiw: interval width\n  (lower = sharper, only "
          "meaningful at matched picp).  calib_err: 0 = perfect.\n")

    if partials:
        print("=== survival under PARTIAL EVIDENCE "
              f"({partials[0]['n_dead']} dead sensors) ===\n")
        keys = ["crps_full", "crps_exact_marginal", "crps_imputed",
                "rmse_full", "rmse_exact_marginal", "rmse_imputed",
                "picp_exact_marginal", "picp_imputed"]
        for k in keys:
            print(f"  {k:>24}: {np.mean([p[k] for p in partials]):8.3f}")
        print("\n  'exact_marginal' integrates the dead sensors out of the joint; "
              "'imputed' fills\n  them with the training mean, which is what any "
              "regressor or conformal method\n  must do.  CQR cannot appear in "
              "this table at all — it has no marginal query.\n")

    if args.survival_demo and last_pc is not None:
        survival_demo(last_pc, last_task)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "runs": runs,
                   "partial_evidence": partials}, f, indent=2, default=str)
    print(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
