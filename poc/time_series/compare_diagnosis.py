"""Paired engine-bootstrap AUROC comparison of two completed study artifacts.

python -m poc.time_series.compare_diagnosis A/artifacts/diagnosis_mask0.npz \
    B/artifacts/diagnosis_mask0.npz --score relational_max --out comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .diagnosis import paired_engine_bootstrap


def compare(a_path, b_path, score="relational_max", reps=1000, seed=0):
    for path in (a_path, b_path):
        status = json.loads((Path(path).parent.parent / "status.json").read_text())
        if status.get("status") != "ok" or status.get("stages", {}).get("diagnosis") != "ok":
            raise ValueError("comparison requires a completed diagnosis stage and run")
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for key in ("labels", "units", "kinds", "observed", "pair_index", "input_windows", "affected"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"unpaired artifacts: {key} differs; compare the same data seed and mask")
        return {"a": str(a_path), "b": str(b_path), "score": score,
                "interpretation": "exploratory paired engine-bootstrap AUROC difference; seed variability is separate",
                **paired_engine_bootstrap(a[score], b[score], a["labels"], a["units"], reps, seed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--score", default="relational_max")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = compare(args.a, args.b, args.score, args.reps, args.seed)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
