"""
Report which datasets are present, and how to get the ones that are not.

    PYTHONPATH=. python -m poc.time_series.check_data
    PYTHONPATH=. python -m poc.time_series.check_data --load    # parse them too

`--load` actually reads each available dataset and prints its shape, which is
the fastest way to find out that a file downloaded but is truncated, or that
h5py is missing, BEFORE an overnight batch discovers it at 2 a.m.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .data_real import (
    CMAPSS_DIR,
    CMAPSS_SUBSETS,
    NCMAPSS_DATASETS,
    NCMAPSS_DIR,
    cmapss_available,
    load_cmapss,
    load_ncmapss,
    ncmapss_available,
    ncmapss_file,
)

CMAPSS_HELP = f"""
C-MAPSS (Turbofan Engine Degradation Simulation Data Set)
  source : NASA Prognostics Center of Excellence data repository
           (also mirrored on Kaggle as "NASA Turbofan Jet Engine Data Set")
  place  : {CMAPSS_DIR}/
  files  : train_FD00x.txt  test_FD00x.txt  RUL_FD00x.txt   for x in 1..4
  size   : ~13 MB total
"""

NCMAPSS_HELP = f"""
N-C-MAPSS (Turbofan Engine Degradation Simulation Data Set 2)
  source : NASA Prognostics Center of Excellence data repository
           (Arias Chao, Kulkarni, Goebel, Fink — Data 6(1):5, 2021)
  place  : {NCMAPSS_DIR}/
  files  : N-CMAPSS_DS01-005.h5, N-CMAPSS_DS02-006.h5, ... (any subset)
  size   : ~1-5 GB per file
  needs  : pip install h5py
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", action="store_true",
                    help="parse each available dataset and print its shape")
    args = ap.parse_args(argv)

    print("\n=== C-MAPSS ===")
    any_cmapss = False
    for sub in CMAPSS_SUBSETS:
        ok = cmapss_available(sub)
        any_cmapss |= ok
        mark = "present" if ok else "MISSING"
        extras = []
        for f in (f"test_{sub}.txt", f"RUL_{sub}.txt"):
            if not os.path.exists(os.path.join(CMAPSS_DIR, f)):
                extras.append(f"no {f}")
        note = f"  ({', '.join(extras)})" if ok and extras else ""
        print(f"  {sub:<8} {mark}{note}")
        if ok and args.load:
            try:
                pair = load_cmapss(sub)
                print(f"           -> {pair}")
            except Exception as exc:
                print(f"           -> FAILED to load: {exc}")
    if not any_cmapss:
        print(CMAPSS_HELP)

    print("=== N-C-MAPSS ===")
    any_n = False
    for ds in NCMAPSS_DATASETS:
        path = ncmapss_file(ds)
        if path:
            any_n = True
            size = os.path.getsize(path) / 2 ** 30
            print(f"  {ds:<8} present  ({os.path.basename(path)}, {size:.1f} GB)")
            if args.load:
                try:
                    pair = load_ncmapss(ds, max_units=3)
                    print(f"           -> {pair}")
                except Exception as exc:
                    print(f"           -> FAILED to load: {exc}")
    if not any_n:
        print("  none found")
        print(NCMAPSS_HELP)

    try:
        import h5py                                     # noqa: F401
        print("h5py: installed")
    except ImportError:
        print("h5py: NOT installed  (needed for N-C-MAPSS only: pip install h5py)")

    print("\nThe synthetic fleet always works and needs nothing.\n"
          "Configs for real data set `skip_if_missing_data: true`, so a batch\n"
          "on a machine without the files skips them instead of failing.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
