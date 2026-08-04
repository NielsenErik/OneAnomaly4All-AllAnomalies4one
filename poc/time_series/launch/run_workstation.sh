#!/usr/bin/env bash
# FULL EXPERIMENT BATCH for a single workstation (developed against 64 GB RAM +
# RTX 4080).  Ordered by decision value, not by cost: if the batch is
# interrupted after two tiers, the two tiers that ran are the ones that decide
# whether the work ships.
#
#   bash poc/time_series/launch/run_workstation.sh              # everything
#   TIERS="1 2"  bash poc/time_series/launch/run_workstation.sh # real-data core
#   JOBS=3 DEVICE=cpu bash poc/time_series/launch/run_workstation.sh
#   SEEDS="0 1 2 3 4" bash poc/time_series/launch/run_workstation.sh
#   DRY=1 bash poc/time_series/launch/run_workstation.sh         # print the plan
#   BENCH_DEVICE=1 bash poc/time_series/launch/run_workstation.sh  # time cpu vs gpu first
#
# DEVICE is a measurement, not a default — but the answer changed when the
# evaluator did.  With the compiled (layered) evaluator, now the default, the
# GPU wins above batch ~128 and by ~2× at 2048+; with the old per-node
# recursion it lost to the CPU at every batch size.  Run BENCH_DEVICE=1 once
# per machine (about a minute) and set DEVICE from what it prints; every
# run.log records device, evaluator and windows/s, so a bad choice is visible
# afterwards instead of being assumed away.
#
# Everything is RESUMABLE: a run that already finished with the same config is
# skipped, so re-launching after a crash, a reboot or a Ctrl-C costs nothing.
# Add FORCE=1 to redo completed runs.
#
# Tiers
#   1  real C-MAPSS detection + explanation      <- the credibility gap, and the
#                                                   contribution.  Run this first.
#   2  real C-MAPSS calibration + RUL            <- exact-vs-calibrated, prognosis
#   3  real N-C-MAPSS detection + RUL            <- the hardest real source
#   4  synthetic controls + layout scaling       <- the reference arm
#   5  ablations: structure, capacity            <- informative, not decisive
#
# Rough wall-clock on one workstation, 3 seeds, CPU: tier 1 ~4-6 h, tier 2
# ~4-6 h, tier 3 depends entirely on which N-C-MAPSS releases are present,
# tier 4 ~2 h, tier 5 ~5-8 h.  With JOBS=3 the whole thing fits in a long night.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TIERS="${TIERS:-1 2 3 4 5}"
[ -n "${FORCE:-}" ] && EXTRA="$EXTRA --force"
[ -n "${DRY:-}" ] && EXTRA="$EXTRA --dry-run"

banner "time-series experiment batch — tiers: $TIERS"
hostinfo

# Fail fast on data: a real-data tier on a machine without the files should say
# so now, not at 3 a.m.  (Configs also skip themselves, but this prints the
# download instructions once, up front.)
"$PY" -m poc.time_series.check_data || true

has_tier() { [[ " $TIERS " == *" $1 "* ]]; }

# ── device measurement, opt-in ─────────────────────────────────────────────
if [ -n "${BENCH_DEVICE:-}" ]; then
  banner "device benchmark — same circuit on every available device"
  "$PY" -m poc.time_series.bench_device --epochs 3 \
        | tee "$CONSOLE_DIR/bench_device_${STAMP}.log"
  echo "set DEVICE=... from the line above and re-launch."
fi

# ── tier 0: wiring check, always ───────────────────────────────────────────
banner "tier 0 — smoke test (wiring only; these numbers mean nothing)"
"$PY" -m poc.time_series.runner config/ts/smoke.yaml --device "$DEVICE" \
      --log-root "$OUT/smoke" --force $EXTRA \
      > "$CONSOLE_DIR/smoke_${STAMP}.log" 2>&1 \
  && echo "smoke OK" \
  || { echo "SMOKE FAILED — see $CONSOLE_DIR/smoke_${STAMP}.log"; exit 1; }

# ── tier 1: the headline, on real data ─────────────────────────────────────
if has_tier 1; then
  banner "tier 1 — real C-MAPSS: detection + explanation"
  queue_config config/ts/cmapss_ad.yaml
  queue_config config/ts/cmapss_explain.yaml
  flush_queue
fi

# ── tier 2: prognosis and the calibration question ─────────────────────────
if has_tier 2; then
  banner "tier 2 — real C-MAPSS: conformal calibration + RUL/survival"
  queue_config config/ts/cmapss_calibration.yaml
  queue_config config/ts/cmapss_rul.yaml
  flush_queue
fi

# ── tier 3: N-C-MAPSS ──────────────────────────────────────────────────────
if has_tier 3; then
  banner "tier 3 — real N-C-MAPSS: detection + RUL"
  queue_config config/ts/ncmapss_ad.yaml
  queue_config config/ts/ncmapss_rul.yaml
  flush_queue
fi

# ── tier 4: synthetic controls ─────────────────────────────────────────────
if has_tier 4; then
  banner "tier 4 — synthetic controls + layout scaling"
  queue_config config/ts/synthetic_ad.yaml
  queue_config config/ts/synthetic_explain.yaml
  queue_config config/ts/synthetic_rul.yaml
  queue_config config/ts/scaling.yaml
  flush_queue
fi

# ── tier 5: ablations ──────────────────────────────────────────────────────
if has_tier 5; then
  banner "tier 5 — ablations: structure (real data), capacity"
  queue_config config/ts/cmapss_structure.yaml
  queue_config config/ts/capacity_sweep.yaml
  flush_queue
fi

banner "aggregating everything under $OUT"
"$PY" -m poc.time_series.aggregate "$OUT" --recursive \
      | tee "$CONSOLE_DIR/summary_${STAMP}.log"

banner "batch finished $(date)"
echo "per-experiment tables : $OUT/*/summary.md"
echo "machine-readable      : $OUT/*/summary.csv"
echo "per-run logs          : $OUT/*/<variant>/seed<N>/run.log"
echo "failures              : grep -l failed $OUT/*/*/*/status.json"
