#!/usr/bin/env bash
# Wiring check: every stage, one seed, a few epochs.  Minutes, not hours.
#
#   bash poc/time_series/launch/run_smoke.sh
#   DEVICE=cuda bash poc/time_series/launch/run_smoke.sh   # check the GPU path
#
# Run this after ANY change to the circuit layer, the data layer or the config
# schema, and before committing a workstation to an overnight batch.  It also
# runs the test suite for the pipeline, which is where the invariants live
# (properties hold, degeneracy raises, conformal coverage is sane).
#
# The numbers it prints are meaningless at this budget and must never be
# reported.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "smoke: pipeline tests"
"$PY" -m pytest tests/test_ts_pipeline.py -q || {
  echo "PIPELINE TESTS FAILED — fix before running experiments"; exit 1; }

banner "smoke: every stage end to end"
"$PY" -m poc.time_series.runner config/ts/smoke.yaml \
      --device "$DEVICE" --log-root "$OUT/smoke" --force || exit 1

banner "smoke: data availability"
"$PY" -m poc.time_series.check_data

banner "smoke OK — $OUT/smoke/summary.md"
