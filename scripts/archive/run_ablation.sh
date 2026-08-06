#!/bin/bash
# ────────────────────────────────────────────────────────────
# AstraKV-W Shared-Prefix Ablation Experiment Runner
# ────────────────────────────────────────────────────────────
# Runs four experimental groups (A/B/C/D) to quantify the
# contribution of each module:
#   A = pure LMCache CPU (baseline)
#   B = + Memory Pressure Controller
#   C = + Chunk Scorer
#   D = + Selective Prefetch (full system)
#
# Usage:
#   export ASTRAKV_MODEL=Qwen/Qwen2.5-7B-Instruct
#   bash scripts/archive/run_ablation.sh
# ────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${ASTRAKV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BASE_DIR="results/gpu/ablation"
CONFIG="configs/shared_prefix_workload.yaml"
LAUNCH_SCRIPT="scripts/launch/launch_lmcache_vllm.sh"

echo "============================================"
echo " AstraKV-W Ablation Experiment"
echo " Model:  $MODEL"
echo " Config: $CONFIG"
echo " Output: $BASE_DIR"
echo "============================================"

mkdir -p "$BASE_DIR"

# ── helper: launch server, wait, run benchmark, kill ──
run_group() {
    local group_label="$1"
    local output_dir="$BASE_DIR/$group_label"
    local extra_args="${2:-}"

    echo ""
    echo ">>> Group $group_label <<<"
    mkdir -p "$output_dir"

    echo "  Launching vLLM + LMCache server ..."
    bash "$LAUNCH_SCRIPT" cpu &
    local server_pid=$!
    sleep 30

    echo "  Running benchmark ..."
    # shellcheck disable=SC2086
    python scripts/benchmark/run_real_benchmark.py \
        --config "$CONFIG" \
        --output-dir "$output_dir" \
        $extra_args \
        2>&1 | tee "$output_dir/benchmark.log"

    echo "  Stopping server (PID $server_pid) ..."
    pkill -f vllm 2>/dev/null || true
    sleep 5
}

# ── Group A: pure baseline ──
run_group "A_baseline" \
    "--disable-pressure-controller --disable-chunk-scorer"

# ── Group B: + Memory Pressure Controller ──
run_group "B_pressure" \
    "--enable-pressure-controller --disable-chunk-scorer"

# ── Group C: + Chunk Scorer ──
run_group "C_scorer" \
    "--enable-pressure-controller --enable-chunk-scorer"

# ── Group D: full system (+ Selective Prefetch) ──
echo ""
echo ">>> Group D_full <<<"
mkdir -p "$BASE_DIR/D_full"

echo "  Launching vLLM + LMCache server ..."
bash "$LAUNCH_SCRIPT" cpu &
sleep 30

echo "  Running selective prefetch benchmark ..."
python scripts/benchmark/run_selective_prefetch_real.py \
    --config "$CONFIG" \
    --output-dir "$BASE_DIR/D_full" \
    --enable-pressure-controller \
    --enable-chunk-scorer \
    2>&1 | tee "$BASE_DIR/D_full/benchmark.log"

echo "  Stopping server ..."
pkill -f vllm 2>/dev/null || true
sleep 5

# ── Summarize ──
echo ""
echo "============================================"
echo " Ablation experiments complete."
echo " Generating comparison report ..."
echo "============================================"

python scripts/reporting/compare_real_runs.py \
    --run baseline="$BASE_DIR/A_baseline" \
    --run pressure="$BASE_DIR/B_pressure" \
    --run scorer="$BASE_DIR/C_scorer" \
    --run full="$BASE_DIR/D_full" \
    --output-dir "$BASE_DIR/ablation_summary"

echo ""
echo "Report: $BASE_DIR/ablation_summary/comparison_report.md"
echo "Done."
