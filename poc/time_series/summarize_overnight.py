"""
Read the overnight batch's JSON outputs and print ONE page that answers the
questions the batch was run to answer.

    PYTHONPATH=. python -m poc.time_series.summarize_overnight logs/overnight
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, Optional

import numpy as np


def load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def rul_row(d: dict, key: str) -> Optional[Dict[str, float]]:
    for k, v in d.get("summary", {}).items():
        if key in k:
            return v
    return None


def rul_verdict(path: str, label: str) -> Optional[str]:
    d = load(path)
    if not d:
        return None
    drop = rul_row(d, "drop censored")
    cens = rul_row(d, "exact censored")
    if not drop or not cens:
        return None
    dc, cc = drop.get("crps"), cens.get("crps")
    if dc is None or cc is None:
        return None
    delta = dc - cc                      # positive => censored term HELPS
    verdict = "HELPS" if delta > 0 else "HURTS"
    return (f"  {label:<26} CRPS  drop {dc:7.3f} | censored {cc:7.3f} "
            f"| Δ {delta:+7.3f}  → censoring {verdict}\n"
            f"  {'':26} RMSE  drop {drop.get('rmse', float('nan')):7.3f} | "
            f"censored {cens.get('rmse', float('nan')):7.3f}   "
            f"PICP {cens.get('picp', float('nan')):.3f} (nominal 0.90)")


def main(argv=None) -> None:
    root = (argv or sys.argv[1:] or ["logs/overnight"])[0]

    print("\n" + "=" * 78)
    print(" OVERNIGHT SUMMARY —", root)
    print("=" * 78)

    # ── the gate ────────────────────────────────────────────────────────
    print("\n[1] GATE: does the exact censored likelihood earn its place?\n")
    gates = [
        (f"{root}/rul_00_chain_fair.json", "chain, 35% censored"),
        (f"{root}/rul_01_heavy_censor.json", "chain, 70% censored *"),
        (f"{root}/rul_02_censor_0.2.json", "chain, 20% censored"),
        (f"{root}/rul_02_censor_0.5.json", "chain, 50% censored"),
    ]
    heavy_helps = None
    for path, label in gates:
        line = rul_verdict(path, label)
        print(line if line else f"  {label:<26} (missing / failed)")
        if "70%" in label and line:
            heavy_helps = "HELPS" in line
    print("\n  * the pre-registered kill condition: at 70% censoring the")
    print("    drop-censored arm is starved, so the exact term MUST win here.")
    if heavy_helps is True:
        print("\n  >>> GATE PASSED — T1 survives; continue the RUL line.")
    elif heavy_helps is False:
        print("\n  >>> GATE FAILED — per the pre-registered condition, T1 becomes")
        print("      a limitations paragraph and the paper ships on AD/XAI alone.")
    else:
        print("\n  >>> GATE INCONCLUSIVE — the 70% run did not produce a result.")

    # ── RUL probes ──────────────────────────────────────────────────────
    print("\n[2] RUL capacity probes (is the K×K coupling the bottleneck?)\n")
    for name, label in [("rul_03_tau_deep", "tau deep, K=12"),
                        ("rul_04_finebins", "40 bins, K=12"),
                        ("rul_05_delta", "first differences"),
                        ("rul_06_time_vtree", "balanced time vtree")]:
        d = load(f"{root}/{name}.json")
        r = rul_row(d, "exact censored") if d else None
        if r:
            print(f"  {label:<26} CRPS {r.get('crps', float('nan')):7.3f}  "
                  f"RMSE {r.get('rmse', float('nan')):7.3f}  "
                  f"PICP {r.get('picp', float('nan')):.3f}")
        else:
            print(f"  {label:<26} (missing / failed)")
    diag = f"{root}/rul_07_diag_clustering.txt"
    if os.path.exists(diag):
        print("\n  clustering diagnostic (few distinct values ⇒ coupling-limited):")
        for ln in open(diag).read().strip().splitlines()[-6:]:
            print("   ", ln)

    # ── AD ──────────────────────────────────────────────────────────────
    print("\n[3] AD detection (current generator)\n")
    d = load(f"{root}/ad_10_chain_full.json")
    if d:
        rows = [(k, v["auroc"][0], v["auroc"][1], v["ap"][0])
                for k, v in d["summary"].items()
                if v.get("auroc") and not k.strip().startswith("↳")]
        for k, a, s, p in sorted(rows, key=lambda r: -r[1])[:6]:
            print(f"  {k:>42} AUROC {a:.4f}±{s:.4f}  AP {p:.4f}")
    else:
        print("  (missing / failed)")

    # ── explanation ─────────────────────────────────────────────────────
    print("\n[4] EXPLANATION quality vs ground truth (the AD contribution)\n")
    d = load(f"{root}/xai_20_full.json")
    if d:
        for k, v in sorted(d["summary"].items(),
                           key=lambda kv: -kv[1].get("auroc", 0)):
            print(f"  {k:>36} AUROC {v['auroc']:.4f}±{v.get('auroc_sd', 0):.4f}"
                  f"  prec@k {v['prec_at_k']:.4f}")
        c = d.get("completeness", {})
        print(f"\n  completeness residual: max {c.get('max_residual_nats', float('nan')):.2e} nats"
              "  (float32 round-off ⇒ exact)")
    else:
        print("  (missing / failed)")

    # ── failures ────────────────────────────────────────────────────────
    print("\n[5] Run status\n")
    for f in sorted(glob.glob(f"{root}/*.txt")):
        name = os.path.basename(f)[:-4]
        txt = open(f, errors="ignore").read()
        bad = ("Traceback" in txt) or ("Error" in txt and "wrote" not in txt)
        print(f"  {'FAILED ' if bad else 'ok     '} {name}")
    print()


if __name__ == "__main__":
    main()
