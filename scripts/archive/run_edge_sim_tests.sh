#!/bin/bash
# ────────────────────────────────────────────────────────────
# AstraKV-W Edge Device Simulation Tests
# ────────────────────────────────────────────────────────────
# Runs AstraKV-W benchmarks under cgroup memory constraints
# to simulate three edge device configurations:
#   16 GB — low-end inference card / industrial PC
#   24 GB — mid-range edge server
#   32 GB — high-end edge server
#
# Prerequisites:
#   sudo bash scripts/archive/setup_edge_sim.sh 16   (or 24/32)
#
# Usage:
#   bash scripts/archive/run_edge_sim_tests.sh
# ────────────────────────────────────────────────────────────
set -euo pipefail

CGROUP_NAME="astrakv_edge_sim"
CGROUP_BASE="/sys/fs/cgroup"
BASE_OUT="results/gpu/edge_sim"
LAUNCH_SCRIPT="scripts/launch/launch_lmcache_vllm.sh"

# detect cgroup version
if [ -f "$CGROUP_BASE/cgroup.controllers" ]; then
    CGROUP_PATH="$CGROUP_BASE/$CGROUP_NAME"
else
    CGROUP_PATH="$CGROUP_BASE/memory/$CGROUP_NAME"
fi

mkdir -p "$BASE_OUT"

run_edge_config() {
    local config_label="$1"
    local mem_gb="$2"
    local model="$3"
    local max_context="$4"
    local output_dir="$BASE_OUT/$config_label"

    echo ""
    echo "============================================"
    echo " Edge Config: $config_label"
    echo "   Memory:    ${mem_gb} GB"
    echo "   Model:     $model"
    echo "   Max ctx:   $max_context"
    echo "============================================"

    mkdir -p "$output_dir"

    # Set memory limit
    local mem_bytes=$((mem_gb * 1024 * 1024 * 1024))
    if [ -f "$CGROUP_PATH/memory.max" ]; then
        echo "$mem_bytes" | sudo tee "$CGROUP_PATH/memory.max" > /dev/null
    elif [ -f "$CGROUP_PATH/memory.limit_in_bytes" ]; then
        echo "$mem_bytes" | sudo tee "$CGROUP_PATH/memory.limit_in_bytes" > /dev/null
    fi

    # Launch server under cgroup constraint
    echo "  Launching server under cgroup constraint ..."
    sudo cgexec -g memory:"$CGROUP_NAME" \
        bash -c "
            export ASTRAKV_MODEL=$model
            export ASTRAKV_MAX_MODEL_LEN=$max_context
            export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
            bash $LAUNCH_SCRIPT cpu &
            SERVER_PID=\$!
            sleep 30

            # Run benchmark
            python scripts/benchmark/run_real_benchmark.py \
                --config configs/dgx_spark_lmcache_cpu.yaml \
                --output-dir $output_dir \
                --context-lengths 1024 2048 4096 \
                --batch-sizes 1 2 \
                2>&1 | tee $output_dir/benchmark.log

            kill \$SERVER_PID 2>/dev/null
            wait \$SERVER_PID 2>/dev/null
        " 2>&1 | tee "$output_dir/edge_sim.log"

    echo "  Done: $config_label"
}

# ── Scenario 1: 16 GB + 1.5B model (closest to real edge device) ──
run_edge_config "edge_16gb_1.5b" 16 "Qwen/Qwen2.5-1.5B-Instruct" 4096

# ── Scenario 2: 24 GB + 7B model (entry-level inference card) ──
run_edge_config "edge_24gb_7b"   24 "Qwen/Qwen2.5-7B-Instruct"   4096

# ── Scenario 3: 32 GB + 7B long context (better edge config) ──
run_edge_config "edge_32gb_7b"   32 "Qwen/Qwen2.5-7B-Instruct"   8192

# ── Summarize ──
echo ""
echo "============================================"
echo " Edge simulation tests complete."
echo " Generating summary ..."
echo "============================================"

python scripts/reporting/compare_real_runs.py \
    --run edge_16gb_1.5b="$BASE_OUT/edge_16gb_1.5b" \
    --run edge_24gb_7b="$BASE_OUT/edge_24gb_7b" \
    --run edge_32gb_7b="$BASE_OUT/edge_32gb_7b" \
    --output-dir "$BASE_OUT/summary"

echo ""
echo "Report: $BASE_OUT/summary/comparison_report.md"
echo "Done."
