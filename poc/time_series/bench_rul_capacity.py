"""
Why is the circuit's E[tau|x] worse than ridge on the same features?

    PYTHONPATH=. python -m poc.time_series.bench_rul_capacity
    PYTHONPATH=. python -m poc.time_series.bench_rul_capacity --seeds 0 1 2
    PYTHONPATH=. python -m poc.time_series.bench_rul_capacity --subset FD003 --quick

THE FACT THIS EXISTS TO EXPLAIN.  On C-MAPSS FD001, protocol = last window per
engine, ridge gets RMSE ~16.0 and SurvivalPC ~20.2 on IDENTICAL features.  A
joint density whose conditional mean loses to least squares is underfitting the
conditional, and no amount of temporal-state machinery bolted on top will fix
that — it would be adding capacity where the deficit is not.

`bench_protocol.py` already ruled out the task: literature-style construction
costs only ~1.2 cycles, and ridge at ~16 sits in the published classical band
(RF 17.9, GB 15.7, LSTM 14.9-16.1).  So the gap is the model, and it is one of
four things.  This sweeps them ONE AT A TIME against a fixed task, with ridge
fitted on the same split as the reference line:

  epochs      is it simply UNDER-TRAINED?  If RMSE keeps falling with epochs,
              the answer is optimisation, not architecture, and everything
              below is premature.
  rul_K       coupling capacity between the window and tau.  The K window
              units are the K mixture components tau is coupled to, so this is
              literally "how many degradation modes can be told apart".
  bins        tau's discretisation.  25 bins over a 125-cycle cap is 5-cycle
              resolution; the quantisation floor on RMSE is bin_width/sqrt(12)
              ~ 1.44 cycles, far too small to explain a 4-cycle gap — but the
              BINNING also coarsens the training signal, which is a separate
              effect this measures.
  tau_where   'deep' couples tau low in the region graph and propagates
              upward; 'root' caps it at a KxK latent.  Recorded as a known
              degeneracy risk in this project.

READ IT AS: whichever knob moves RMSE toward the ridge line is the deficit.
If NONE of them does, the deficit is the OBJECTIVE — maximising joint
likelihood does not optimise conditional-mean accuracy, and the fix is a
hybrid generative/discriminative term, not more capacity.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .baselines import RidgeRUL
from .circuits import DegenerateModelError, SurvivalPC
from .datasets import build_rul_task, load_fleets
from .metrics import crps_from_pmf, rmse

# The task is held FIXED at the literature-comparable construction, so that a
# knob's effect is never confounded with a task change.  Censoring is off for
# the same reason: it is this project's contribution, not part of the question
# "can the circuit match a linear model at all".
TASK = dict(window=30, stride=1, censor_frac=0.0)


def make_task(subset: str, bins: int, seed: int):
    spec = {
        "source": "cmapss", "subset": subset, "cap": 125, "official_test": True,
        "per_regime": "auto", "bins": bins, "train_units": 0.6,
        "max_test_windows": None, "rul_test_windows": "all", "groups": 3,
        "regimes": 3, "dead_channels": 2, "healthy_frac": 0.35,
        "organic_frac": 0.85, "inject_rate": 0.12, "strength": 1.0, **TASK,
    }
    return build_rul_task(load_fleets(spec, seed=seed), spec, seed=seed)


def last_idx(task) -> np.ndarray:
    u = task.unit_test.numpy()
    return np.array([int(np.where(u == c)[0][-1]) for c in dict.fromkeys(u.tolist())])


def ridge_reference(task) -> Dict[str, float]:
    keep = task.delta_train == 1
    bw = task.cap / task.n_bins
    y = (task.rul_train[keep] if task.rul_train is not None
         else (task.tau_train[keep].float() + 0.5) * bw)
    m = RidgeRUL().fit(task.X_train[keep], y)
    i = last_idx(task)
    return {"rmse_last": rmse(m.predict(task.X_test[i])["mean"], task.rul_test[i]),
            "rmse_all": rmse(m.predict(task.X_test)["mean"], task.rul_test)}


def fit_circuit(task, seed: int, K: int, tau_where: str, epochs: int,
                device: Optional[str], leaf_components: int = 1
                ) -> Tuple[Dict[str, float], float]:
    t0 = time.time()
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method="chain", n_sum_components=K,
                    leaf_components=leaf_components, tau_where=tau_where,
                    channel_groups=task.channel_groups, seed=seed, device=device)
    pc.fit(task.X_train, task.tau_train, task.delta_train, epochs=epochs,
           lr=0.05, batch_size=256, use_censored=True)
    pred = pc.predict(task.X_test)
    i = last_idx(task)
    bw = task.cap / task.n_bins
    out = {
        "rmse_last": rmse(np.asarray(pred["mean"])[i], task.rul_test[i]),
        "rmse_all": rmse(pred["mean"], task.rul_test),
        "crps": crps_from_pmf(torch.as_tensor(pred["pmf"]), task.tau_test, bw),
        "nll": float(pc.history[-1]),
        "params": pc.size()["parameters"],
    }
    return out, time.time() - t0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    ap.add_argument("--epochs", type=int, default=60, help="baseline budget")
    ap.add_argument("--quick", action="store_true",
                    help="one seed, short budget — wiring check, not a result")
    ap.add_argument("--max-train", type=int, default=None,
                    help="subsample training windows (a probe knob: it changes "
                         "the answer, so never report a number measured with it)")
    args = ap.parse_args(argv)
    seeds = [args.seeds[0]] if args.quick else args.seeds
    base_ep = 15 if args.quick else args.epochs

    # baseline point, then one knob moved per row
    BASE = dict(K=12, bins=25, tau_where="deep", epochs=base_ep)
    rows: List[Tuple[str, Dict[str, Any]]] = [("baseline (current config)", {})]
    rows += [(f"epochs {e}", {"epochs": e})
             for e in ([base_ep * 2] if args.quick else [30, 120, 240])]
    rows += [(f"rul_K {k}", {"K": k}) for k in (8, 16, 24)]
    rows += [(f"tau bins {b}", {"bins": b}) for b in (50, 100)]
    rows += [("tau_where root", {"tau_where": "root"})]
    rows += [("leaf mixture (3 comp.)", {"leaf_components": 3})]

    print(f"\nC-MAPSS {args.subset} · task held at window={TASK['window']}, "
          f"stride={TASK['stride']}, censoring={TASK['censor_frac']} "
          f"(literature-comparable)\nseeds {seeds}\n")

    ref = [ridge_reference(make_task(args.subset, 25, s)) for s in seeds]
    r_last = np.mean([r["rmse_last"] for r in ref])
    print(f"{'RIDGE REFERENCE (same split)':<28}"
          f"{r_last:>10.2f}{np.mean([r['rmse_all'] for r in ref]):>12.2f}"
          f"{'—':>10}{'—':>12}\n")
    hdr = (f"{'circuit variant':<28}{'RMSE last':>10}{'RMSE all':>12}"
           f"{'CRPS':>10}{'NLL':>12}{'vs ridge':>11}{'fit s':>8}")
    print(hdr); print("-" * len(hdr))

    tasks: Dict[Tuple[int, int], Any] = {}
    for label, over in rows:
        cfg = {**BASE, **over}
        got, secs = [], []
        for s in seeds:
            key = (cfg["bins"], s)
            if key not in tasks:
                t = make_task(args.subset, cfg["bins"], s)
                if args.max_train and len(t.X_train) > args.max_train:
                    sel = torch.randperm(len(t.X_train))[:args.max_train]
                    t.X_train = t.X_train[sel]
                    t.tau_train = t.tau_train[sel]
                    t.delta_train = t.delta_train[sel]
                    t.regime_train = t.regime_train[sel]
                    if t.unit_train is not None:
                        t.unit_train = t.unit_train[sel]
                    if t.rul_train is not None:
                        t.rul_train = t.rul_train[sel]
                tasks[key] = t
            try:
                m, dt = fit_circuit(tasks[key], s, cfg["K"], cfg["tau_where"],
                                    cfg["epochs"], args.device,
                                    cfg.get("leaf_components", 1))
                got.append(m); secs.append(dt)
            except DegenerateModelError as exc:
                print(f"{label:<28}   DEGENERATE — {str(exc)[:60]}")
                got = []
                break
        if not got:
            continue
        f = lambda k: np.mean([g[k] for g in got])
        print(f"{label:<28}{f('rmse_last'):>10.2f}{f('rmse_all'):>12.2f}"
              f"{f('crps'):>10.2f}{f('nll'):>12.1f}"
              f"{f('rmse_last') - r_last:>+11.2f}{np.mean(secs):>8.0f}")

    print("\nInterpretation:")
    print("  * a knob that closes the 'vs ridge' column is the deficit;")
    print("  * if `epochs` keeps helping, it is UNDER-TRAINING, and the rest of")
    print("    this table was measured on models that had not converged;")
    print("  * if NLL improves while RMSE does not, the objective is the problem")
    print("    (better density != better conditional mean) and the fix is a")
    print("    discriminative term, not more capacity.\n")


if __name__ == "__main__":
    main()
