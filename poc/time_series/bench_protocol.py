"""
Is the RUL task harder than the one the literature reports on?

    PYTHONPATH=. python -m poc.time_series.bench_protocol
    PYTHONPATH=. python -m poc.time_series.bench_protocol --subset FD003 --seeds 0 1 2

WHY THIS EXISTS.  On C-MAPSS FD001, published RMSE is ~11.0-11.5 cycles
(protocol: one prediction per test engine, RUL capped at 125).  In this repo a
plain RIDGE gets ~17.8 and the circuit ~20.2.  A 6-7 cycle gap on the BASELINE
cannot be a modelling result — a linear model is a linear model — so it has to
come from how the task is built.  Until that is settled, every model comparison
in the RUL stage is being made on a harder-than-standard problem, and "the
circuit loses to ridge" and "the circuit is far from SOTA" are not comparable
claims.

This sweeps the task-construction knobs ONE AT A TIME, holding the model fixed
(ridge, which has no hyperparameters worth blaming), so each row attributes a
number of cycles to a specific choice:

  window       20 here vs 30 in most published pipelines
  rul_stride   3 here (windows are subsampled) vs 1
  censor_frac  0.2-0.7 here (simulated in-service censoring) vs 0 published —
               censoring is this project's contribution, but it also DELETES
               the end of every censored trajectory, which is where the signal
               for a short RUL lives
  protocol     last window per unit (the published one) vs all windows

Read the output as: if "literature-style" lands near 15-16, the remaining gap
to 11 is model quality and the task is fine.  If it lands near 17-18, the task
is the confound and no amount of circuit work will close it.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Dict, List

import numpy as np

from .baselines import RidgeRUL
from .datasets import build_rul_task, load_fleets
from .metrics import rmse


def last_window_mask(task) -> np.ndarray:
    """Index of each test unit's final window — the published protocol."""
    u = task.unit_test.numpy()
    return np.array([int(np.where(u == c)[0][-1]) for c in dict.fromkeys(u.tolist())])


def run_one(subset: str, window: int, stride: int, censor: float,
            seed: int) -> Dict[str, float]:
    spec = {
        "source": "cmapss", "subset": subset, "window": window, "stride": stride,
        "cap": 125, "official_test": True, "per_regime": "auto",
        "censor_frac": censor, "bins": 25, "train_units": 0.6,
        "max_test_windows": None, "rul_test_windows": "all",
        "groups": 3, "regimes": 3, "dead_channels": 2,
        "healthy_frac": 0.35, "organic_frac": 0.85, "inject_rate": 0.12,
        "strength": 1.0,
    }
    task = build_rul_task(load_fleets(spec, seed=seed), spec, seed=seed)

    # ridge trains on point labels, so censored windows carry no usable target
    keep = task.delta_train == 1
    bw = task.cap / task.n_bins
    y = (task.rul_train[keep] if task.rul_train is not None
         else (task.tau_train[keep].float() + 0.5) * bw)
    model = RidgeRUL().fit(task.X_train[keep], y)

    idx = last_window_mask(task)
    pred_last = model.predict(task.X_test[idx])["mean"]
    pred_all = model.predict(task.X_test)["mean"]
    return {
        "rmse_last": rmse(pred_last, task.rul_test[idx]),
        "rmse_all": rmse(pred_all, task.rul_test),
        "n_train": int(keep.sum()),
        "n_units": len(idx),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sota", type=float, default=11.3,
                    help="published RMSE for this subset, for the gap column")
    args = ap.parse_args(argv)

    # (label, window, rul_stride, censor_frac) — one knob moves per row
    rows = [
        ("this repo's RUL config", 20, 3, 0.2),
        ("  + window 30", 30, 3, 0.2),
        ("  + stride 1 (no subsampling)", 20, 1, 0.2),
        ("  + no censoring", 20, 3, 0.0),
        ("literature-style (all three)", 30, 1, 0.0),
    ]

    print(f"\nC-MAPSS {args.subset} · ridge regression · protocol = last window "
          f"per engine\nseeds {args.seeds}   published RMSE ~{args.sota}\n")
    hdr = (f"{'task construction':<32}{'RMSE [last]':>16}{'RMSE [all]':>14}"
           f"{'train win':>11}{'gap to SOTA':>13}")
    print(hdr); print("-" * len(hdr))
    for label, w, s, c in rows:
        got = [run_one(args.subset, w, s, c, seed) for seed in args.seeds]
        last = [g["rmse_last"] for g in got]
        alls = [g["rmse_all"] for g in got]
        print(f"{label:<32}{np.mean(last):>10.2f}±{np.std(last):<5.2f}"
              f"{np.mean(alls):>9.2f}±{np.std(alls):<4.2f}"
              f"{got[0]['n_train']:>11,}{np.mean(last) - args.sota:>+13.2f}")
    print("\nIf the last row is near the published number, the task is fine and the\n"
          "gap is model quality.  If it stays 6+ cycles above, the task construction\n"
          "is the confound and model comparisons on it are not comparable to the\n"
          "literature (they may still be internally valid).\n")


if __name__ == "__main__":
    main()
