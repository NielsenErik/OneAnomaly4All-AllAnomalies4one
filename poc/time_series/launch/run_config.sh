#!/usr/bin/env bash
# Run ONE config with the launcher's environment (thread caps, PYTHONPATH,
# interpreter resolution, console log).  Everything after the config path is
# forwarded to the runner.
#
#   bash poc/time_series/launch/run_config.sh config/ts/cmapss_ad.yaml
#   bash poc/time_series/launch/run_config.sh config/ts/cmapss_rul.yaml \
#        --only censor_frac-0.7 --seeds 0 1 2 3 4
#   DEVICE=cuda bash poc/time_series/launch/run_config.sh config/ts/cmapss_explain.yaml
#   bash poc/time_series/launch/run_config.sh config/ts/capacity_sweep.yaml --dry-run

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${1:-}"
[ -n "$CFG" ] || { echo "usage: $0 <config.yaml> [runner args...]"; exit 1; }
shift

NAME="$(basename "${CFG%.yaml}")"
LOG="$CONSOLE_DIR/${NAME}_${STAMP}.log"

banner "$NAME"
hostinfo
SEEDS_ARG=""
[ -n "${SEEDS:-}" ] && SEEDS_ARG="--seeds $SEEDS"

# tee: watch it live and keep the transcript
"$PY" -m poc.time_series.runner "$CFG" --device "$DEVICE" \
      --log-root "$OUT/$NAME" $SEEDS_ARG $EXTRA "$@" 2>&1 | tee "$LOG"

echo
echo "console log : $LOG"
echo "tables      : $OUT/$NAME/summary.md"
