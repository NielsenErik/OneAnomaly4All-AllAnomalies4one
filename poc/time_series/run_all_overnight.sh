#!/bin/bash
# Overnight batch for the time-series PoC.
#
# Ordered by decision value, not by cost: the two experiments that decide
# whether T1 (RUL as an exact censored survival query) lives or dies run FIRST,
# so a partial night still answers the question that matters.
#
#   usage:  bash poc/time_series/run_all_overnight.sh
#   output: logs/overnight/<name>.txt  + logs/overnight/<name>.json
#
# Each block is independent; a failure is logged and the batch continues.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

PY=~/miniconda3/envs/expllm_env/bin/python
[ -x "$PY" ] || PY=python3
OUT=logs/overnight
mkdir -p "$OUT"

STAMP() { date "+%H:%M:%S"; }
run() {                      # run <name> <args...>
  local name="$1"; shift
  echo "[$(STAMP)] START  $name"
  echo "    $*" > "$OUT/$name.cmd"
  if "$@" > "$OUT/$name.txt" 2>&1; then
    echo "[$(STAMP)] OK     $name"
  else
    echo "[$(STAMP)] FAILED $name  (see $OUT/$name.txt)"
  fi
}

echo "=============================================================="
echo " time-series PoC — overnight batch, started $(date)"
echo "=============================================================="

# ─────────────────────────────────────────────────────────────────────────
# GATE: does the exact censored likelihood earn its place?
# These two decide T1.  Everything after them is secondary.
# ─────────────────────────────────────────────────────────────────────────

# STEP 0 — the fair test that has never been run: chain structure + current
# (AR(1)) generator.  Every previously reported RUL number predates both.
run rul_00_chain_fair $PY -m poc.time_series.run_rul \
    --seeds 0 1 2 --vtree chain --epochs 60 --survival-demo \
    --out "$OUT/rul_00_chain_fair.json"

# STEP 1 — the sanity check that MUST pass.  At 70% censoring the
# drop-censored arm is starved of most of the fleet, so if the exact censored
# term does not win here it is not going to win anywhere.
run rul_01_heavy_censor $PY -m poc.time_series.run_rul \
    --seeds 0 1 2 --vtree chain --censor-frac 0.7 --epochs 60 --no-partial \
    --out "$OUT/rul_01_heavy_censor.json"

# censoring sweep: is there a crossover point? (H2)
for CF in 0.2 0.5; do
  run "rul_02_censor_$CF" $PY -m poc.time_series.run_rul \
      --seeds 0 1 --vtree chain --censor-frac $CF --epochs 60 --no-partial \
      --out "$OUT/rul_02_censor_$CF.json"
done

# ─────────────────────────────────────────────────────────────────────────
# RUL capacity / coupling probes (H3, H5)
# ─────────────────────────────────────────────────────────────────────────

run rul_03_tau_deep $PY -m poc.time_series.run_rul \
    --seeds 0 1 --vtree chain --tau-where deep --K 12 --epochs 60 --no-partial \
    --out "$OUT/rul_03_tau_deep.json"

run rul_04_finebins $PY -m poc.time_series.run_rul \
    --seeds 0 1 --vtree chain --bins 40 --K 12 --epochs 60 --no-partial \
    --out "$OUT/rul_04_finebins.json"

run rul_05_delta $PY -m poc.time_series.run_rul \
    --seeds 0 1 --vtree chain --delta --epochs 60 --no-partial \
    --out "$OUT/rul_05_delta.json"

run rul_06_time_vtree $PY -m poc.time_series.run_rul \
    --seeds 0 1 --vtree time --epochs 60 --no-partial \
    --out "$OUT/rul_06_time_vtree.json"

# H3 diagnostic: if the predictive takes <= K distinct values, the K x K
# coupling is the bottleneck and no amount of training will fix it.
run rul_07_diag_clustering $PY - <<'PYEOF'
import numpy as np, torch
from poc.time_series.circuits import SurvivalPC
from poc.time_series.data import make_rul_task
for where, K in [("root", 6), ("root", 12), ("deep", 12)]:
    t = make_rul_task(window=8, stride=3, seed=0, n_units=60)
    m = SurvivalPC(t.window, t.n_channels, t.n_bins, t.cap, vtree_method="chain",
                   tau_where=where, n_sum_components=K, seed=0)
    m.fit(t.X_train, t.tau_train, t.delta_train, epochs=40)
    pred = m.predict(t.X_test)["mean"].numpy()
    u = len(np.unique(pred.round(2)))
    print(f"tau@{where:5s} K={K:2d}: {u:4d} distinct predicted RUL values "
          f"over {len(pred)} windows  (K^2 = {K*K})")
    print(f"           spread: min {pred.min():.1f} max {pred.max():.1f} "
          f"sd {pred.std():.1f}; true sd {t.rul_test.numpy().std():.1f}")
PYEOF

# ─────────────────────────────────────────────────────────────────────────
# AD side — re-run on the CURRENT generator so every number is comparable
# ─────────────────────────────────────────────────────────────────────────

run ad_10_chain_full $PY -m poc.time_series.run_ad \
    --seeds 0 1 2 --vtree chain --epochs 40 --typed --missing \
    --out "$OUT/ad_10_chain_full.json"

run ad_11_structure_ablation $PY -m poc.time_series.run_ad \
    --seeds 0 1 2 --epochs 40 --vtree-ablation \
    --ablation-methods random time channel chain chow_liu spectral orc_rg forman_rg \
    --out "$OUT/ad_11_structure_ablation.json"

run ad_12_sos $PY -m poc.time_series.run_ad \
    --seeds 0 1 2 --sos --K 2 --epochs 30 --fast \
    --out "$OUT/ad_12_sos.json"

run ad_13_orc_rg_multi $PY -m poc.time_series.run_ad \
    --seeds 0 1 2 --vtree orc_rg_multi --epochs 40 --fast \
    --out "$OUT/ad_13_orc_rg_multi.json"

run ad_14_delta $PY -m poc.time_series.run_ad \
    --seeds 0 1 2 --vtree chain --delta --epochs 40 --fast \
    --out "$OUT/ad_14_delta.json"

# ─────────────────────────────────────────────────────────────────────────
# Explainability — the AD half's actual contribution
# ─────────────────────────────────────────────────────────────────────────

run xai_20_full $PY -m poc.time_series.run_explain \
    --seeds 0 1 2 --epochs 40 --shapley 8 --plots --examples \
    --fig-dir "$OUT/figs" --out "$OUT/xai_20_full.json"

run xai_21_shap_budget $PY -m poc.time_series.run_explain \
    --seeds 0 --epochs 40 --shap-samples 128 --no-deletion \
    --out "$OUT/xai_21_shap_budget.json"

# ─────────────────────────────────────────────────────────────────────────
# Scaling evidence for the DAG rebuild
# ─────────────────────────────────────────────────────────────────────────

run bench_30_scaling $PY -m poc.time_series.bench_scaling --K 4
run bench_31_scaling_k6 $PY -m poc.time_series.bench_scaling --K 6 \
    --dims 8 16 32 64 112 256

# ─────────────────────────────────────────────────────────────────────────
echo
echo "=============================================================="
echo " batch finished $(date)"
echo "=============================================================="
$PY -m poc.time_series.summarize_overnight "$OUT" 2>/dev/null || \
  echo "(summary script unavailable; read $OUT/*.txt directly)"
