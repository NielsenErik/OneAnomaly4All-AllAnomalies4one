"""
Frozen confirmation — roadmap step 8.

    PYTHONPATH=. python -m poc.time_series.confirm_diagnosis freeze \
        config/ts/diagnosis_confirmation.yaml
    PYTHONPATH=. python -m poc.time_series.confirm_diagnosis run \
        config/ts/diagnosis_confirmation.yaml
    PYTHONPATH=. python -m poc.time_series.confirm_diagnosis evaluate \
        config/ts/diagnosis_confirmation.yaml --out release/confirmation

Everything else in this package is exploratory by construction: the tables are
generated exhaustively, the report header says so, and a reader may look at any
cell.  This file is the opposite, and the difference is mechanical rather than
a matter of discipline:

  FREEZE      `freeze` stamps the config with the sha256 of its own frozen
              block and the date.  `evaluate` recomputes that digest and
              REFUSES when it differs, so a margin, an alpha, a mask workload
              or a comparator edited after the confirmation data were seen
              cannot silently become the pre-registered one.  Re-freezing an
              already-frozen protocol needs `--refreeze`, which writes a new
              version and keeps the old digest in the history.
  ONE ENDPOINT  the protocol names exactly one primary endpoint before the
              data exist.  `evaluate` reports that endpoint and its interval
              first, and every other number it prints is explicitly secondary.
  NO RESCUE   a comparator that dominates is reported as a negative result in
              the same format as a positive one.  There is no branch here that
              searches for a mask, a score or a seed on which the candidate
              would have won.

The two admissible designs are the ones the September 9 test plan proposed:

  noninferiority   localisation AP delta >= -margin (proposed 0.02) with the
                   lower end of the paired interval above -margin, AND at
                   least `speed_factor`x (proposed 2x) the comparator's warm
                   query latency.
  superiority      localisation AP delta >= margin (proposed 0.03) with the
                   whole interval above zero, under a fixed resource ceiling.

Both additionally REQUIRE the operational false-alarm condition: on
independent calibration objects, the upper end of the repeated-draw
false-alarm interval must not exceed alpha.  A candidate that localises well
by alarming on everything fails, which is the point of making it a condition
rather than a column.

Sample size comes from the pilot variance recorded in the frozen block; fresh
initialisation seeds on data that were already inspected are not fresh
confirmation units, so the unit of independence stays the engine.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .ts_logging import provenance_report, read_results, run_status

DESIGNS = ("noninferiority", "superiority")
REQUIRED_FROZEN_KEYS = (
    "design", "candidate", "comparator", "primary_metric", "score",
    "margin", "alpha", "sampling_unit", "mask_workload", "fault_mechanisms",
    "n_units_planned", "pilot_sd", "bootstrap_reps", "resource_ceiling",
    "tie_rule", "preprocessing", "dataset",
)


class ProtocolError(RuntimeError):
    """The protocol is not in a state in which a confirmation may be read."""


# ═══════════════════════════════════════════════════════════════════════════
# Freezing
# ═══════════════════════════════════════════════════════════════════════════

def _canonical(block: Dict[str, Any]) -> str:
    """Digest input: the frozen block without its own stamp, key-sorted."""
    body = {k: v for k, v in block.items()
            if k not in ("frozen_at", "digest", "history", "version")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def digest(block: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(block).encode()).hexdigest()


def load_protocol(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "confirmation" not in cfg:
        raise ProtocolError(f"{path}: no `confirmation:` block; this is not a "
                            "confirmation protocol")
    block = cfg["confirmation"]
    missing = [k for k in REQUIRED_FROZEN_KEYS if k not in block]
    if missing:
        raise ProtocolError(f"{path}: the frozen block is incomplete: {missing}")
    if block["design"] not in DESIGNS:
        raise ProtocolError(f"design must be one of {DESIGNS}, not {block['design']!r}")
    if block["primary_metric"] not in ("loc_ap", "end_to_end_unique_top1", "auroc"):
        raise ProtocolError("primary_metric must be loc_ap, end_to_end_unique_top1 or auroc")
    return cfg


def freeze(path: str, refreeze: bool = False) -> Dict[str, Any]:
    """Stamp the protocol with its digest.  Must happen before any run."""
    cfg = load_protocol(path)
    block = cfg["confirmation"]
    new = digest(block)
    if block.get("digest"):
        if block["digest"] == new and not refreeze:
            return {"path": path, "digest": new, "changed": False,
                    "frozen_at": block.get("frozen_at")}
        if not refreeze:
            raise ProtocolError(
                f"{path}: already frozen as {block['digest'][:12]} but its content now "
                f"digests to {new[:12]}. Editing a frozen protocol invalidates it; pass "
                "--refreeze to publish a NEW version, which starts a new confirmation.")
        history = list(block.get("history") or [])
        history.append({"digest": block["digest"], "frozen_at": block.get("frozen_at"),
                        "version": block.get("version", 1)})
        block["history"] = history
        block["version"] = int(block.get("version", 1)) + 1
    block.setdefault("version", 1)
    block["digest"] = new
    block["frozen_at"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return {"path": path, "digest": new, "changed": True,
            "frozen_at": block["frozen_at"], "version": block["version"]}


def check_frozen(cfg: Dict[str, Any], path: str) -> Dict[str, Any]:
    block = cfg["confirmation"]
    stamped = block.get("digest")
    if not stamped:
        raise ProtocolError(
            f"{path}: never frozen. Run `confirm_diagnosis freeze {path}` BEFORE the "
            "confirmation data are produced; a protocol frozen afterwards is a "
            "description of the result, not a prediction of it.")
    actual = digest(block)
    if actual != stamped:
        raise ProtocolError(
            f"{path}: content changed after freezing (stamped {stamped[:12]}, now "
            f"{actual[:12]}). The confirmation is void: restore the frozen content, or "
            "--refreeze and collect new confirmation data.")
    return block


# ═══════════════════════════════════════════════════════════════════════════
# Evidence collection
# ═══════════════════════════════════════════════════════════════════════════

def _artifact_pairs(root: str, candidate: str, comparator: str
                    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Candidate/comparator artifact pairs from completed runs of `candidate`.

    Returned as (pairs, refusals).  A run whose status is not ok, or whose
    comparator artifact is absent, is a refusal that the release reports; it is
    never quietly skipped, because "the comparator failed on those engines" and
    "the comparator was not run" have to look different in the record.
    """
    pairs: List[Dict[str, str]] = []
    refusals: List[Dict[str, str]] = []
    if not os.path.isdir(root):
        raise ProtocolError(f"no such log root: {root}; run the confirmation first")
    for dirpath, _, files in sorted(os.walk(root)):
        if "status.json" not in files:
            continue
        rel = os.path.relpath(dirpath, root)
        variant = rel.split(os.sep)[0]
        if variant != candidate:
            continue
        st = run_status(dirpath) or {}
        if st.get("status") != "ok" or st.get("stages", {}).get("diagnosis") != "ok":
            refusals.append({"run_dir": rel, "reason": f"status={st.get('status')}"})
            continue
        art = os.path.join(dirpath, "artifacts")
        if not os.path.isdir(art):
            refusals.append({"run_dir": rel, "reason": "no artifacts directory"})
            continue
        names = sorted(os.listdir(art))
        circuits = [n for n in names
                    if n.startswith("diagnosis_mask") and n.endswith(".npz")]
        if not circuits:
            refusals.append({"run_dir": rel, "reason": "no circuit mask artifacts"})
        for name in circuits:
            index = name[len("diagnosis_mask"):-len(".npz")]
            other = f"diagnosis_{comparator}_mask{index}.npz"
            if other not in names:
                refusals.append({"run_dir": rel, "mask_index": index,
                                 "reason": f"comparator artifact {other} absent"})
                continue
            pairs.append({"run_dir": rel, "mask_index": index,
                          "a": os.path.join(art, name),
                          "b": os.path.join(art, other)})
    return pairs, refusals


def _mask_name(path: str) -> str:
    """The mask the artifact was produced under, from the artifact itself.

    Strict on purpose: an artifact that cannot name its own mask cannot be
    checked against a frozen workload, and inferring the name from the file
    index would mean trusting that two runs enumerated their masks in the same
    order.  Older artifacts predate this field and are refused rather than
    guessed at.
    """
    with np.load(path, allow_pickle=False) as z:
        if "mask_name" not in z:
            raise ProtocolError(
                f"{path}: artifact predates the frozen-workload check (no mask_name). "
                "Re-run the confirmation with the current stage.")
        value = z["mask_name"]
        return str(value.item() if value.ndim == 0 else value[0])


def primary_evidence(root: str, block: Dict[str, Any], reps: Optional[int] = None,
                     seed: int = 0) -> Dict[str, Any]:
    """The paired primary endpoint, on the frozen mask workload only."""
    from .compare_diagnosis import compare
    candidate = str(block["candidate"])
    comparator = str(block["comparator"])
    metric = str(block["primary_metric"])
    score = str(block["score"])
    reps = int(reps if reps is not None else block["bootstrap_reps"])
    workload = [str(m) for m in (block["mask_workload"] or [])]

    pairs, refusals = _artifact_pairs(root, candidate, comparator)
    used, rows = [], []
    for pair in pairs:
        try:
            name = _mask_name(pair["a"])
        except ProtocolError as exc:
            refusals.append({**pair, "refused": True, "reason": str(exc)[:300]})
            continue
        if workload and name not in workload:
            continue                       # outside the frozen workload, by design
        entry = {**{k: pair[k] for k in ("run_dir", "mask_index")}, "mask": name}
        try:
            entry.update(compare(pair["a"], pair["b"], score=score, reps=reps,
                                 seed=seed, metric=metric))
            entry["delta"] = entry.get(f"{metric}_delta", entry.get("delta"))
            used.append(entry)
        except Exception as exc:
            entry.update({"refused": True, "reason": str(exc)[:300]})
            refusals.append(entry)
        rows.append(entry)
    if not used:
        raise ProtocolError(
            f"{root}: no paired {candidate}-vs-{comparator} artifact survived the frozen "
            f"mask workload {workload}. {len(refusals)} refusal(s); first: "
            f"{refusals[0] if refusals else 'none'}")

    # The primary endpoint is ONE number.  Pooling over masks would let a
    # favourable mask carry an unfavourable one, so the frozen block names the
    # primary mask; with several, the endpoint is the WORST of them, which is
    # the conservative reading and cannot be gamed by adding easy masks.
    primary_mask = block.get("primary_mask")
    if primary_mask:
        chosen = [e for e in used if e.get("mask") == primary_mask]
        if not chosen:
            raise ProtocolError(f"{root}: the frozen primary mask {primary_mask!r} "
                                f"produced no comparable pair")
    else:
        chosen = used
    worst = min(chosen, key=lambda e: (float(e["ci_low"]), float(e["delta"])))
    return {"metric": metric, "score": score, "candidate": candidate,
            "comparator": comparator, "bootstrap_reps": reps,
            "n_pairs_used": len(used), "n_refused": len(refusals),
            "aggregation": ("frozen primary mask" if primary_mask else
                            "worst frozen-workload mask by interval lower end"),
            "primary": worst, "all_pairs": rows, "refusals": refusals}


def latency_evidence(root: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """Measured warm query latency, candidate versus comparator, same masks."""
    rows = [r for r in read_results(root, require_ok=True)
            if str(r.get("method", "")).startswith("query ")
            and r.get("variant") == block["candidate"]]
    workload = [str(m) for m in (block["mask_workload"] or [])]
    by: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        mask = str(r.get("mask"))
        if workload and mask not in workload:
            continue
        value = r.get("query_s")
        if value is None:
            continue
        by.setdefault(mask, {}).setdefault(str(r.get("method_name")), []).append(float(value))
    out = []
    comparator = str(block["comparator"])
    for mask, methods in sorted(by.items()):
        a = methods.get("circuit")
        b = methods.get(comparator)
        if not a or not b:
            continue
        out.append({"mask": mask, "circuit_query_s": float(np.mean(a)),
                    "comparator_query_s": float(np.mean(b)),
                    "speed_factor": float(np.mean(b) / max(np.mean(a), 1e-12)),
                    "n_circuit": len(a), "n_comparator": len(b)})
    factor = min((r["speed_factor"] for r in out), default=None)
    return {"per_mask": out, "worst_speed_factor": factor,
            "note": "warm query seconds, mean over confirmation runs; a single-window "
                    "latency claim and a throughput claim are different claims and this "
                    "is the workload the protocol froze"}


def operational_evidence(root: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """Repeated-draw false-alarm behaviour on independent calibration objects."""
    alpha = float(block["alpha"])
    score = str(block["score"])
    workload = [str(m) for m in (block["mask_workload"] or [])]
    rows = [r for r in read_results(root, require_ok=True)
            if r.get("variant") == block["candidate"] and r.get("score") == score
            and r.get("method_name") == "circuit" and r.get("mask")
            and (not workload or str(r.get("mask")) in workload)]
    q95 = [float(r["trial_fpr_q95"]) for r in rows if r.get("trial_fpr_q95") is not None]
    inf = [float(r["trial_infinite_threshold_frac"]) for r in rows
           if r.get("trial_infinite_threshold_frac") is not None]
    power = [float(r["trial_power_mean"]) for r in rows if r.get("trial_power_mean") is not None]
    observed = [float(r["fpr"]) for r in rows if r.get("fpr") is not None]
    return {"alpha": alpha, "n_rows": len(rows),
            "trial_fpr_q95_max": max(q95) if q95 else None,
            "trial_power_mean": float(np.mean(power)) if power else None,
            "infinite_threshold_frac_max": max(inf) if inf else None,
            "observed_fpr_max": max(observed) if observed else None,
            "measured": bool(q95),
            "note": "repeated independent calibration draws; when the honest threshold "
                    "was infinite it stayed infinite and that fraction is reported"}


# ═══════════════════════════════════════════════════════════════════════════
# The decision
# ═══════════════════════════════════════════════════════════════════════════

def decide(block: Dict[str, Any], evidence: Dict[str, Any],
           latency: Dict[str, Any], operational: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the frozen rule.  No branch here inspects an alternative endpoint."""
    design = str(block["design"])
    margin = float(block["margin"])
    primary = evidence["primary"]
    delta, low, high = (float(primary["delta"]), float(primary["ci_low"]),
                        float(primary["ci_high"]))
    conditions: List[Dict[str, Any]] = []

    if design == "noninferiority":
        conditions.append({"name": "primary_noninferiority",
                           "rule": f"paired {evidence['metric']} interval lower end > -{margin}",
                           "value": low, "threshold": -margin, "met": low > -margin})
        want = float(block.get("speed_factor", 2.0))
        got = latency.get("worst_speed_factor")
        conditions.append({"name": "resource_advantage",
                           "rule": f"worst-mask warm-query speed factor >= {want}x",
                           "value": got, "threshold": want,
                           "met": got is not None and got >= want})
    else:
        conditions.append({"name": "primary_superiority_point",
                           "rule": f"paired {evidence['metric']} delta >= {margin}",
                           "value": delta, "threshold": margin, "met": delta >= margin})
        conditions.append({"name": "primary_superiority_interval",
                           "rule": "paired interval excludes zero from above",
                           "value": low, "threshold": 0.0, "met": low > 0.0})
        ceiling = block.get("resource_ceiling") or {}
        cap = ceiling.get("max_query_s")
        got = max((r["circuit_query_s"] for r in latency["per_mask"]), default=None)
        conditions.append({"name": "resource_ceiling",
                           "rule": f"candidate warm query seconds <= {cap}",
                           "value": got, "threshold": cap,
                           "met": cap is None or (got is not None and got <= float(cap))})

    q95 = operational.get("trial_fpr_q95_max")
    conditions.append({"name": "operational_false_alarm",
                       "rule": f"upper end of repeated-draw false-alarm interval <= alpha "
                               f"({operational['alpha']})",
                       "value": q95, "threshold": operational["alpha"],
                       "met": q95 is not None and q95 <= operational["alpha"] + 1e-12})

    met = all(bool(c["met"]) for c in conditions)
    dominated = high < 0.0
    if met:
        verdict = "CONFIRMED"
        reading = (f"The frozen endpoint is met: {evidence['metric']} delta {delta:+.4f} "
                   f"[{low:+.4f}, {high:+.4f}] against {evidence['comparator']}.")
    elif dominated:
        verdict = "NEGATIVE"
        reading = (f"The comparator dominates: {evidence['metric']} delta {delta:+.4f} "
                   f"[{low:+.4f}, {high:+.4f}] lies entirely below zero. This is the "
                   f"result, and {evidence['candidate']} is not the better method on this "
                   "endpoint.")
    else:
        verdict = "NOT MET"
        failed = [c["name"] for c in conditions if not c["met"]]
        reading = (f"The frozen endpoint is not met ({', '.join(failed)}); the interval "
                   f"[{low:+.4f}, {high:+.4f}] does not exclude the comparator either. "
                   "Inconclusive is not a licence to re-run with a different endpoint.")
    return {"verdict": verdict, "design": design, "margin": margin,
            "conditions": conditions, "delta": delta, "ci_low": low, "ci_high": high,
            "n_units": primary.get("n_units"), "reading": reading}


# ═══════════════════════════════════════════════════════════════════════════
# Release bundle
# ═══════════════════════════════════════════════════════════════════════════

def render(report: Dict[str, Any]) -> str:
    b, d = report["protocol"], report["decision"]
    ev, lat, op = report["evidence"], report["latency"], report["operational"]
    lines = [f"# Confirmation — {b.get('name', 'diagnosis')} v{b.get('version')}", "",
             f"**{d['verdict']}** — {d['reading']}", "",
             f"Protocol digest `{b['digest'][:16]}` frozen {b['frozen_at']}; "
             f"design `{d['design']}`, margin {d['margin']}, alpha {op['alpha']}, "
             f"sampling unit `{b['sampling_unit']}`.", "",
             "## Primary endpoint", "",
             f"| candidate | comparator | metric | delta | 95% CI | units | reps |",
             f"|---|---|---|---|---|---|---|",
             f"| {ev['candidate']} | {ev['comparator']} | {ev['metric']} | "
             f"{d['delta']:+.4f} | [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
             f"{d['n_units']} | {ev['bootstrap_reps']} |", "",
             f"Aggregation: {ev['aggregation']}. Mask `{ev['primary'].get('mask')}`, "
             f"{ev['n_pairs_used']} comparable pair(s), {ev['n_refused']} refused.", "",
             "## Frozen conditions", "",
             "| condition | rule | value | threshold | met |", "|---|---|---|---|---|"]
    for c in d["conditions"]:
        value = "—" if c["value"] is None else (f"{c['value']:.4f}"
                                                if isinstance(c["value"], float) else c["value"])
        thr = "—" if c["threshold"] is None else c["threshold"]
        lines.append(f"| {c['name']} | {c['rule']} | {value} | {thr} | "
                     f"{'yes' if c['met'] else 'NO'} |")
    lines += ["", "## Run inventory", "",
              f"- rows used {report['provenance']['rows_used']} of "
              f"{report['provenance']['rows_total']}",
              f"- runs by status: {report['provenance']['runs_by_status']}",
              f"- incomplete: {report['provenance']['incomplete_runs'] or 'none'}",
              f"- refusals recorded: {ev['n_refused']}", ""]
    if ev["refusals"]:
        lines.append("Refusals:")
        for r in ev["refusals"][:10]:
            lines.append(f"  - {r.get('run_dir')} mask {r.get('mask_index')}: "
                         f"{r.get('reason')}")
        lines.append("")
    lines += ["## Secondary and ambiguous", "",
              "Every other paired mask in the frozen workload, reported whether or not "
              "it agrees with the primary endpoint:", "",
              "| mask | delta | 95% CI | units |", "|---|---|---|---|"]
    for e in ev["all_pairs"]:
        if e.get("refused"):
            continue
        lines.append(f"| {e.get('mask')} | {float(e['delta']):+.4f} | "
                     f"[{float(e['ci_low']):+.4f}, {float(e['ci_high']):+.4f}] | "
                     f"{e.get('n_units')} |")
    lines += ["", "## Operational false alarm", "",
              f"- alpha {op['alpha']}, worst repeated-draw upper end "
              f"{op['trial_fpr_q95_max']}",
              f"- mean power {op['trial_power_mean']}",
              f"- infinite-threshold fraction (max) {op['infinite_threshold_frac_max']}", "",
              "## Cost", "", "| mask | circuit s | comparator s | factor |",
              "|---|---|---|---|"]
    for r in lat["per_mask"]:
        lines.append(f"| {r['mask']} | {r['circuit_query_s']:.5f} | "
                     f"{r['comparator_query_s']:.5f} | {r['speed_factor']:.2f}x |")
    lines += ["", f"_{lat['note']}_", "",
              "## Scope", "",
              f"Dataset `{b['dataset']}`, fault mechanisms {b['fault_mechanisms']}, "
              f"mask workload {b['mask_workload']}, preprocessing `{b['preprocessing']}`. "
              f"Tie rule: {b['tie_rule']}. Planned {b['n_units_planned']} independent "
              f"unit(s) from pilot sd {b['pilot_sd']}. Claims outside this scope are not "
              "confirmed by this run."]
    return "\n".join(lines)


def evaluate(path: str, out_dir: Optional[str] = None, reps: Optional[int] = None,
             seed: int = 0) -> Dict[str, Any]:
    cfg = load_protocol(path)
    block = check_frozen(cfg, path)
    root = cfg.get("log_root") or block.get("log_root")
    if not root:
        raise ProtocolError(f"{path}: no log_root to read the confirmation runs from")
    evidence = primary_evidence(root, block, reps=reps, seed=seed)
    latency = latency_evidence(root, block)
    operational = operational_evidence(root, block)
    decision = decide(block, evidence, latency, operational)
    report = {"protocol": {**{k: block[k] for k in REQUIRED_FROZEN_KEYS},
                           "name": cfg.get("name"), "digest": block["digest"],
                           "frozen_at": block["frozen_at"],
                           "version": block.get("version", 1),
                           "primary_mask": block.get("primary_mask"),
                           "speed_factor": block.get("speed_factor")},
              "log_root": root, "protocol_path": path,
              "provenance": provenance_report(root),
              "evidence": evidence, "latency": latency,
              "operational": operational, "decision": decision}
    out_dir = out_dir or os.path.join(root, "confirmation")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(path, os.path.join(out_dir, "frozen_protocol.yaml"))
    with open(os.path.join(out_dir, "confirmation.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    markdown = render(report)
    with open(os.path.join(out_dir, "confirmation.md"), "w") as f:
        f.write(markdown + "\n")
    report["out_dir"] = out_dir
    report["markdown"] = markdown
    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _run(path: str, extra: Sequence[str]) -> int:
    """Execute the frozen config through the ordinary runner."""
    cfg = load_protocol(path)
    check_frozen(cfg, path)              # refuse to generate data for a loose protocol
    from .runner import main as run_main
    return run_main([path, *extra])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze", help="stamp the protocol before any data exist")
    f.add_argument("config")
    f.add_argument("--refreeze", action="store_true",
                   help="publish a NEW protocol version; starts a new confirmation")

    r = sub.add_parser("run", help="execute the frozen config through the runner")
    r.add_argument("config")
    r.add_argument("rest", nargs=argparse.REMAINDER)

    e = sub.add_parser("evaluate", help="read the frozen endpoint and release")
    e.add_argument("config")
    e.add_argument("--out", default=None)
    e.add_argument("--reps", type=int, default=None)
    e.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    if args.command == "freeze":
        info = freeze(args.config, args.refreeze)
        print(json.dumps(info, indent=2))
        return 0
    if args.command == "run":
        return _run(args.config, args.rest)
    report = evaluate(args.config, args.out, args.reps, args.seed)
    print(report["markdown"])
    print(f"\nwritten: {report['out_dir']}")
    return 0 if report["decision"]["verdict"] != "ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
