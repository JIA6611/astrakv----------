#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3.6-35B-A3B}"
PORT="${ASTRAKV_PORT:-8000}"
OUTPUT_ROOT="${OUTPUT_DIR:-$ROOT/results/moe-prepare-demo-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKLOAD="$OUTPUT_ROOT/workload/moe_prepare_workload.jsonl"
PAIR_ID="moe-prepare-$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2).read()" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

run_arm() {
  local role="$1"
  local prepare_enabled="$2"
  local arm_root="$OUTPUT_ROOT/$role"
  local cache_dir="$arm_root/lmcache_store"
  local cache_config="$arm_root/lmcache.yaml"
  local events="$arm_root/lmcache047_events.jsonl"
  local server_log="$arm_root/server.log"
  mkdir -p "$cache_dir"
  : > "$events"
  cat > "$cache_config" <<EOF
local_cpu: false
max_local_cpu_size: 2.0
local_disk: $cache_dir
max_local_disk_size: 80.0
EOF

  ASTRAKV_PYTHON="$PYTHON" \
  ASTRAKV_MODEL="$MODEL" \
  ASTRAKV_PORT="$PORT" \
  ASTRAKV_MAX_MODEL_LEN=16384 \
  ASTRAKV_GPU_MEMORY_UTILIZATION=0.70 \
  ASTRAKV_TENSOR_PARALLEL_SIZE=1 \
  ASTRAKV_PREFIX_CACHING=true \
  ASTRAKV_MOE_MODE=true \
  ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_LMCACHE047_EVENTS="$events" \
  LMCACHE_CONFIG_FILE="$cache_config" \
  LMCACHE_DISK_PATH="$cache_dir" \
  bash scripts/launch/launch_lmcache_vllm.sh disk > "$server_log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready; then
    tail -n 120 "$server_log" >&2 || true
    echo "MoE server failed to become ready for $role" >&2
    return 1
  fi

  ASTRAKV_MODEL="$MODEL" \
  ASTRAKV_MOE_PREPARE_ENABLED="$prepare_enabled" \
  ASTRAKV_MAX_MODEL_LEN=16384 \
  ASTRAKV_GPU_MEMORY_UTILIZATION=0.70 \
  ASTRAKV_PREFIX_CACHING=true \
  ASTRAKV_VLLM_SEED=0 \
  LMCACHE_CONFIG_FILE="$cache_config" \
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --config configs/dgx_spark_moe_prepare_prepared.yaml \
    --model "$MODEL" \
    --workload-jsonl "$WORKLOAD" \
    --output-dir "$arm_root/runs" \
    --pair-id "$PAIR_ID" \
    --pair-role "$role" \
    --online-artifact "events=$events"

  find "$arm_root/runs" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2- > "$arm_root/run_dir.txt"
  stop_server
}

mkdir -p "$OUTPUT_ROOT/workload"
"$PYTHON" scripts/benchmark/materialize_moe_prepare_workload.py \
  --model "$MODEL" \
  --output "$WORKLOAD" \
  --context-lengths 2048 8192 \
  --visits 4 \
  --output-tokens 8

run_arm baseline false
run_arm variant true

BASELINE_DIR="$(<"$OUTPUT_ROOT/baseline/run_dir.txt")"
PREPARED_DIR="$(<"$OUTPUT_ROOT/variant/run_dir.txt")"
"$PYTHON" scripts/reporting/analyze_moe_prepare_ablation.py \
  --baseline-dir "$BASELINE_DIR" \
  --prepared-dir "$PREPARED_DIR" \
  --output-dir "$OUTPUT_ROOT/report"

echo "MoE prepare demo completed: $OUTPUT_ROOT"
echo "Report: $OUTPUT_ROOT/report/moe_prepare_ablation.md"
