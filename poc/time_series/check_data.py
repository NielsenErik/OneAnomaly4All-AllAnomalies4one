"""
Report which datasets are present, and how to get the ones that are not.

    PYTHONPATH=. python -m poc.time_series.check_data
    PYTHONPATH=. python -m poc.time_series.check_data --load       # parse them too
    PYTHONPATH=. python -m poc.time_series.check_data --only esa opssat
    PYTHONPATH=. python -m poc.time_series.check_data --caveats    # print all caveats

`--load` actually reads each available dataset and prints its shape, which is
the fastest way to find out that a file downloaded but is truncated, that h5py
is missing, or that a converted CSV has the wrong column names — BEFORE an
overnight batch discovers it at 2 a.m.

The table comes from `catalog.SOURCES`, so a source that exists to the runner
always appears here, with the same expected path and the same caveats.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from .catalog import SOURCES, Source

# Per-source variants worth probing individually: one C-MAPSS subset being
# present says nothing about the other three.
VARIANTS: Dict[str, List[Dict[str, Any]]] = {
    "cmapss": [{"subset": s} for s in ("FD001", "FD002", "FD003", "FD004")],
    "ncmapss": [{"dataset": d} for d in ("DS01", "DS02", "DS03", "DS04", "DS05",
                                         "DS06", "DS07", "DS08a", "DS08c", "DS08d")],
    "esa": [{"mission": m} for m in ("Mission1", "Mission2", "Mission3")],
    "ims": [{"ims_test": t} for t in ("1st_test", "2nd_test", "3rd_test")],
    "pcoe": [{"preset": p} for p in ("igbt", "capacitor", "fatigue")],
    "smapmsl": [{"spacecraft": s} for s in ("SMAP", "MSL")],
}

# Loader kwargs that keep `--load` cheap: parsing all of ESA-Mission1 or every
# IMS snapshot to answer "is it readable?" would take minutes.
PROBE_SPEC: Dict[str, Dict[str, Any]] = {
    "ncmapss": {"max_units": 3},
    "esa": {"max_train_samples": 20_000, "max_test_samples": 20_000, "cache": False},
    "ims": {"max_files": 40, "cache": False},
    "opssat": {"cache": False},
    "smapmsl": {"cache": False},
    "battery": {"cache": False},
    "milling": {"cache": False},
}


def _spec(src: Source, variant: Dict[str, Any]) -> Dict[str, Any]:
    return {"source": src.name, **PROBE_SPEC.get(src.name, {}), **variant}


def _label(variant: Dict[str, Any]) -> str:
    return ",".join(str(v) for v in variant.values()) if variant else ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", action="store_true",
                    help="parse each available dataset and print its shape")
    ap.add_argument("--only", nargs="+", default=None, metavar="SOURCE",
                    help="restrict to these sources (default: all)")
    ap.add_argument("--caveats", action="store_true",
                    help="print every source's known defects and exit")
    args = ap.parse_args(argv)

    names = list(args.only) if args.only else list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"unknown source(s) {unknown}; known: {sorted(SOURCES)}")
        return 1

    if args.caveats:
        for name in names:
            src = SOURCES[name]
            print(f"\n=== {name} — {src.title} ===")
            for c in src.caveats:
                print(f"  - {c}")
        print()
        return 0

    present, absent = [], []
    for name in names:
        src = SOURCES[name]
        variants = VARIANTS.get(name, [{}])
        print(f"\n=== {name} — {src.title} ===")
        print(f"    tasks: {', '.join(src.tasks)}   path: {src.root}")
        any_here = False
        for v in variants:
            spec = _spec(src, v)
            try:
                ok = bool(src.probe(spec))
            except Exception as exc:
                ok = False
                print(f"  {_label(v) or name:<10} probe failed: {exc}")
                continue
            any_here |= ok
            print(f"  {_label(v) or name:<10} {'present' if ok else 'MISSING'}")
            if ok and args.load:
                try:
                    pair = src.loader(spec, 0)
                    print(f"             -> {pair}")
                    if pair.test is not None:
                        print(f"                test: {pair.test}")
                except Exception as exc:
                    print(f"             -> FAILED to load: "
                          f"{type(exc).__name__}: {exc}")
        (present if any_here else absent).append(name)
        if not any_here:
            print(f"    get it: {src.howto}")

    print("\n" + "=" * 74)
    print(f"present: {', '.join(present) or 'none'}")
    print(f"missing: {', '.join(absent) or 'none'}")

    optional = {"h5py": "N-C-MAPSS", "scipy": "battery / milling",
                "pandas": "ESA / OPSSAT / CALCE / PCoE", "openpyxl": "raw CALCE .xlsx"}
    print()
    for mod, why in optional.items():
        try:
            __import__(mod)
            print(f"  {mod:<9} installed")
        except ImportError:
            print(f"  {mod:<9} NOT installed  (needed for: {why})")

    print("\nThe synthetic fleet always works and needs nothing.\n"
          "Configs for real data set `skip_if_missing_data: true`, so a batch\n"
          "on a machine without the files skips them instead of failing.\n"
          "Run with --caveats before writing anything up.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
