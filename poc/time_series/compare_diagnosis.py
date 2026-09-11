"""Paired engine-bootstrap comparison of two completed study artifacts.

    python -m poc.time_series.compare_diagnosis A/artifacts/diagnosis_mask0.npz \
        B/artifacts/diagnosis_mask0.npz --score relational_max --out comparison.json

    python -m poc.time_series.compare_diagnosis A/.../diagnosis_mask0.npz \
        B/.../diagnosis_gaussian_fitted_mask0.npz --metric loc_ap

Three endpoints, all paired on the same windows and all resampled by ENGINE:

    auroc                    detection
    loc_ap                   localisation average precision
    end_to_end_unique_top1   a unique top-1 hit AND the window's alarm firing

The pairing checks are the point of the file. Two artifacts are comparable only
when they were produced from the same inputs, labels, masks and corruptions;
anything else silently compares two different experiments, so a mismatch is an
error rather than a warning. Identical scores must give a delta of exactly
zero, which is what makes a small nonzero delta meaningful.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .diagnosis import (ATTRIBUTION_OF, paired_engine_bootstrap,
                        paired_localization_bootstrap)

PAIRED_KEYS = ("labels", "units", "kinds", "observed", "pair_index",
               "input_windows", "affected")


def _status_ok(path) -> None:
    status = json.loads((Path(path).parent.parent / "status.json").read_text())
    if status.get("status") != "ok" or status.get("stages", {}).get("diagnosis") != "ok":
        raise ValueError(f"{path}: comparison requires a completed diagnosis stage and run")


def compare(a_path, b_path, score="relational_max", reps=1000, seed=0,
            metric="auroc"):
    if metric not in ("auroc", "loc_ap", "end_to_end_unique_top1"):
        raise ValueError("metric must be auroc, loc_ap or end_to_end_unique_top1")
    for path in (a_path, b_path):
        _status_ok(path)
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for key in PAIRED_KEYS:
            if key not in a or key not in b:
                raise ValueError(f"artifact predates this comparison: {key} is absent")
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"unpaired artifacts: {key} differs; compare the same data seed and mask")
        head = {"a": str(a_path), "b": str(b_path), "score": score, "metric": metric,
                "interpretation": "exploratory paired engine-bootstrap difference; "
                                  "training-seed variability is a separate axis"}
        if metric == "auroc":
            return {**head, **paired_engine_bootstrap(a[score], b[score], a["labels"],
                                                      a["units"], reps, seed)}
        attr = f"attr_{ATTRIBUTION_OF[score]}"
        alarm = f"alarm_{score}"
        for key in (attr, alarm):
            if key not in a or key not in b:
                raise ValueError(
                    f"{key} is absent: localisation comparison needs artifacts written "
                    "by the current stage, which stores the attribution matrices and "
                    "the calibrated alarms beside the scores")
        return {**head, **paired_localization_bootstrap(
            a[attr], b[attr], a["affected"], a["observed"], a[alarm], b[alarm],
            a["units"], metric=metric, reps=reps, seed=seed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--score", default="relational_max")
    parser.add_argument("--metric", default="auroc",
                        choices=["auroc", "loc_ap", "end_to_end_unique_top1"])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = compare(args.a, args.b, args.score, args.reps, args.seed, args.metric)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
