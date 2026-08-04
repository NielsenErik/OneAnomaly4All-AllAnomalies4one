"""
Experiment configuration: YAML in, fully-resolved run specs out.

A config file describes ONE experiment (a dataset, a model, a set of stages)
plus optionally a set of variations of it.  The runner expands it into
`variants × seeds` runs and executes them independently, so the same file is
both the documentation of what was run and the thing that reruns it.

Two ways to vary a config, and they compose:

    grid:                    cartesian product over dotted keys
      model.vtree: [chain, time, random]
      model.K:     [4, 8]

    variants:                explicit, named, non-cartesian
      - name: baseline
        model: {vtree: chain}
      - name: no_delta
        model: {vtree: chain, delta: false}

`grid` is right for an ablation where every combination is meaningful; a
matched-budget structure sweep is the canonical case.  `variants` is right when
the combinations are not a product (different datasets needing different
windows, say).  Using both gives variants × grid.

Everything is merged onto DEFAULTS, so a config only states what it changes and
every run records the FULL resolved dict — a five-line config never leaves a
reader guessing what the other forty parameters were.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = ("ad", "explain", "rul", "calibration", "scaling")

DEFAULTS: Dict[str, Any] = {
    "name": "unnamed",
    "description": "",
    "seeds": [0, 1, 2],
    "device": "auto",                 # auto | cpu | cuda | cuda:0 | mps
    # layered = the compiled layer-parallel evaluator (§6a); recursive = the
    # per-node reference.  Identical numbers, gated against each other at fit
    # time; "recursive" exists for the A/B and for debugging, not for results.
    "evaluator": "layered",           # layered | recursive
    "log_root": None,                 # default: logs/ts/<name>
    "stages": ["ad"],
    "skip_if_missing_data": True,     # real-data configs no-op when files absent

    "dataset": {
        "source": "synthetic",        # synthetic | cmapss | ncmapss
        # --- shared ---
        "window": 8,
        "stride": 2,
        "cap": None,                  # None -> source default (130 synth, 125 real)
        "healthy_frac": 0.35,
        "organic_frac": 0.85,
        "inject_rate": 0.12,
        "strength": 1.0,
        "train_units": 0.6,           # used only when there is no official test fleet
        "per_regime": "auto",         # normalise per operating regime
        "max_test_windows": None,
        # --- synthetic only ---
        "units": 60,
        "channels": 14,
        "regimes": 3,
        "groups": 3,
        "dead_channels": 2,
        "noise": 0.10,
        "group_noise": 0.25,
        "phi_ar": 0.85,
        # --- cmapss ---
        "subset": "FD001",
        "official_test": True,
        # --- ncmapss ---
        "dataset": "DS02",
        "aggregate": "cycle",         # cycle | raw
        "subsample": 10,
        "channels_groups": ["X_s"],   # X_s | W | X_v | T  (X_v/T are model internals)
        "max_units": None,
        "max_rows": None,
        "cache": True,
        "data_dir": None,
        # --- rul only ---
        "censor_frac": 0.0,
        "bins": 20,
        "rul_stride": 3,
        "rul_test_windows": "all",    # all | last  ("last" = literature protocol)
    },

    "model": {
        "vtree": "chain",
        "K": 6,
        "leaf_components": 1,
        "epochs": 40,
        "lr": 0.05,
        "batch_size": 256,
        "delta": False,
        "sos": False,
        "weight_jitter": 0.5,
        "tau_where": "deep",          # 'root' is degenerate — see hand-off §3
        "rul_K": None,                # default: model.K
        "rul_epochs": None,           # default: model.epochs
    },

    "eval": {
        "fast_baselines": False,      # skip IForest / GMM / conv-AE / Deep SVDD
        "baselines": True,
        "missing": True,              # dead-sensor query
        "n_dead": 3,
        "typed": True,                # marginal / conditional / structural split
        "shapley_orders": 8,
        "shap_samples": 32,
        "deletion": True,
        "n_complete": 64,             # windows used for the completeness check
        "max_explain_windows": 1500,  # attribution is O(C) passes; cap for real data
        "kinds": ["spike", "offset", "drift", "decouple", "desync"],
        "alpha": 0.10,
        # Evaluate ONE trained circuit at several miscoverage levels / test
        # protocols.  Neither affects training, so putting them in the config
        # grid retrains an identical model per value.  Leave empty to use the
        # single `alpha` / the dataset's `rul_test_windows`.
        "alphas": [],                 # e.g. [0.10, 0.20]
        "test_protocols": [],         # e.g. [all, last]
        "censoring_ablation": True,   # train twice: drop-censored vs exact term
        "conformal": True,
        "conformal_modes": ["cqr", "pit"],
        "cal_frac": 0.3,
        "survival_demo": True,
        "partial_evidence": True,
        "plots": True,
        "examples": True,
        "save_scores": True,
        "scaling_dims": [16, 32, 64, 112, 256],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Loading / merging
# ═══════════════════════════════════════════════════════════════════════════

def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML config and merge it onto DEFAULTS."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = deep_merge(DEFAULTS, {k: v for k, v in raw.items()
                                if k not in ("grid", "variants")})
    cfg["grid"] = raw.get("grid") or {}
    cfg["variants"] = raw.get("variants") or []
    cfg["config_path"] = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    if cfg.get("log_root") is None:
        cfg["log_root"] = os.path.join("logs", "ts", str(cfg["name"]))
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    bad = [s for s in cfg["stages"] if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; expected a subset of {list(STAGES)}")
    src = cfg["dataset"]["source"]
    if src not in ("synthetic", "cmapss", "ncmapss"):
        raise ValueError(f"unknown dataset.source {src!r}")
    if cfg["model"]["sos"] and cfg["model"]["vtree"].endswith("_multi"):
        raise ValueError(
            "multi-partition region graphs are not structured decomposable, so "
            "the squared/SOS construction is not exact on them — SquaredPC "
            "refuses this combination by design")
    if not cfg["seeds"]:
        raise ValueError("no seeds given")


# ═══════════════════════════════════════════════════════════════════════════
# Overrides and grid expansion
# ═══════════════════════════════════════════════════════════════════════════

def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _coerce(text: str) -> Any:
    """`--set model.K=8` should give an int, not the string "8"."""
    try:
        return json.loads(text)
    except Exception:
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """`["model.K=8", "eval.plots=false"]` applied in order."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(cfg, key.strip(), _coerce(raw.strip()))
    return cfg


def _slug(value: Any) -> str:
    s = str(value)
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", s)
    return s.strip("-") or "x"


def expand_variants(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Config -> list of fully-resolved variant configs, each carrying:
      `variant`        directory-safe name
      `variant_axes`   the values that distinguish it (for result rows)
    """
    base_variants: List[Tuple[str, Dict[str, Any]]] = []
    for v in cfg.get("variants") or []:
        v = dict(v)
        name = str(v.pop("name", f"v{len(base_variants)}"))
        base_variants.append((name, v))
    if not base_variants:
        base_variants = [("", {})]

    grid = cfg.get("grid") or {}
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]

    out: List[Dict[str, Any]] = []
    for vname, vpatch in base_variants:
        for combo in combos:
            merged = deep_merge(cfg, vpatch)
            axes: Dict[str, Any] = {}
            for k, val in zip(keys, combo):
                set_dotted(merged, k, val)
                axes[k] = val
            parts = [p for p in [vname] if p]
            parts += [f"{k.split('.')[-1]}-{_slug(v)}" for k, v in axes.items()]
            merged["variant"] = "_".join(parts) if parts else "default"
            merged["variant_axes"] = {**({"variant": vname} if vname else {}), **axes}
            merged.pop("grid", None)
            merged.pop("variants", None)
            out.append(merged)
    return out


def run_dir_for(cfg: Dict[str, Any], seed: int) -> str:
    return os.path.join(cfg["log_root"], cfg.get("variant", "default"), f"seed{seed}")


def resolved_for_hash(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    The part of the config that decides whether a finished run is still valid.

    Deliberately excludes cosmetics (`description`, `plots`) — flipping a plot
    flag should not invalidate twelve hours of compute — and excludes
    `log_root`, so moving a batch's output directory does not force a rerun.
    """
    keep = {k: v for k, v in cfg.items()
            if k not in ("description", "log_root", "config_path", "grid",
                         "variants", "seeds")}
    ev = dict(keep.get("eval", {}))
    ev.pop("plots", None)
    ev.pop("examples", None)
    keep["eval"] = ev
    return keep
