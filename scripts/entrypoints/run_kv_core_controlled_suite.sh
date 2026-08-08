#!/usr/bin/env bash
set -Eeuo pipefail

# E0--E4 runner for the version-locked Qwen3-8B KV-Core experiment.  It does
# not send warmup HTTP requests: every request is part of the supplied,
# content-addressed workload and cache state is explicit in its manifest.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
WORKLOAD_DIR=""
OUTPUT_DIR="$ROOT/results/kv-core-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
TIMEOUT="900"
GPU_MEMORY_UTILIZATION="0.72"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_kv_core_controlled_suite.sh --workload-dir DIR --patch-manifest FILE --callback-smoke FILE [options]

DIR must contain these canonical JSONL workloads: repeated_long_prefix,
random_no_reuse, constrained_kv_churn, queued_concurrency.  Each input must
already fix request order, seed, sampling parameters, output length, and cold
or warm cache-state cases.  The deployed connector patch must write
kv_core_native_receipts.jsonl and uma_resource_samples.jsonl to the runtime
state directory supplied in ASTRAKV_RUNTIME_CONTROL_STATE_DIR.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload-dir) WORKLOAD_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$WORKLOAD_DIR" && -d "$WORKLOAD_DIR" ]] || { echo "--workload-dir is required" >&2; exit 2; }
[[ -f "$PATCH_MANIFEST" && -f "$CALLBACK_SMOKE" ]] || { echo "verified deployment inputs are required" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "only a loopback server is supported" >&2; exit 2; }
for workload in repeated_long_prefix random_no_reuse constrained_kv_churn queued_concurrency; do
  [[ -f "$WORKLOAD_DIR/$workload.jsonl" ]] || { echo "Missing $workload.jsonl" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
"$PYTHON" scripts/runtime/verify_kv_core_connector_patch.py \
  --deployment-manifest "$PATCH_MANIFEST" --callback-smoke "$CALLBACK_SMOKE" \
  --output "$OUTPUT_DIR/connector_patch_verification.json"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 120); do
    curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null && return 0
    sleep 2
  done
  tail -n 160 "$log_path" >&2 || true
  return 1
}

run_one() {
  local label="$1" phase="$2" role="$3" workload="$4" cache_state="$5" baseline_label="$6"
  local run_id="kv-core-${phase}-${workload}-${cache_state}-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$label/$workload/$cache_state/$role"
  local state_dir="$run_dir/state"
  local log_path="$run_dir/server.log"
  local backend="disk" topology="gpu_ssd" mode="off" admission="false" prefetch="false" partial="false"
  mkdir -p "$state_dir"
  case "$phase" in
    E0) mode="off" ;;
    E1) mode="shadow" ;;
    E2) mode="active"; admission="true" ;;
    E3) mode="active"; admission="true"; prefetch="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    E4) mode="active"; admission="true"; prefetch="true"; partial="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    *) echo "invalid phase: $phase" >&2; return 2 ;;
  esac
  cleanup
  ASTRAKV_MODEL="$MODEL" ASTRAKV_HOST="$HOST" ASTRAKV_PORT="$PORT" \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_MAX_MODEL_LEN="32768" \
  LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:?pair-scoped LMCache config is required}" \
  # Disable vLLM's in-process prefix cache for this external-KV experiment.
  # Both pair members have the same setting; the only source of a cross-worker
  # reuse is the pair-scoped LMCache disk store populated by the baseline.
  ASTRAKV_PREFIX_CACHING=false ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE=engine_child ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" \
  ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" ASTRAKV_RUNTIME_CONTROL_ENGINE_ID="$run_id-engine" \
  ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0 ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
  ASTRAKV_RUNTIME_CONTROL_SESSION_ID="$run_id-session" ASTRAKV_RUNTIME_CONTROL_SECRET_HEX="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')" \
  ASTRAKV_KV_CORE_MODE="$mode" ASTRAKV_KV_CORE_TOPOLOGY="$topology" \
  ASTRAKV_KV_CORE_LOCAL_CPU="$([[ "$topology" == gpu_cpu_ssd ]] && echo true || echo false)" \
  ASTRAKV_KV_CORE_PATCH_VERIFICATION="$OUTPUT_DIR/connector_patch_verification.json" \
  ASTRAKV_KV_CORE_ADMISSION_ENABLED="$admission" ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED="$prefetch" \
  ASTRAKV_KV_CORE_PARTIAL_PREFIX_UPPER_BOUND_ENABLED="$partial" \
  nohup bash scripts/launch/launch_lmcache_vllm.sh "$backend" > "$log_path" 2>&1 < /dev/null &
  SERVER_PID="$!"
  wait_for_server "$log_path"
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" --model "$MODEL" --backend "vllm-lmcache-kv-core" \
    --output-dir "$run_dir" --workload-jsonl "$WORKLOAD_DIR/$workload.jsonl" \
    --run-id "$run_id" --workload-id "$workload" --model-revision local-qwen3-8b \
    --tokenizer-revision local-qwen3-8b --dtype bfloat16 --quantization unquantized \
    --random-seed 0 --cache-state "$cache_state" --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "${label}-${workload}-${cache_state}" --pair-role "$role" --claim-scope kv_core \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "$run_id-session" --timeout "$TIMEOUT" --output-tokens 128
  # These artifacts are emitted by the version-locked connector patch after
  # native events.  Do not synthesize estimated receipts or block capacity.
  cp "$state_dir/kv_core_native_receipts.jsonl" "$run_dir/kv_core_native_receipts.jsonl"
  cp "$state_dir/uma_resource_samples.jsonl" "$run_dir/uma_resource_samples.jsonl"
  cp "$state_dir/kv_core_run_metadata.json" "$run_dir/kv_core_run_metadata.json"
  cleanup
}

run_pair() {
  local label="$1" baseline_phase="$2" variant_phase="$3" workload="$4" cache_state="$5"
  local cache_dir="$OUTPUT_DIR/lmcache-store/$label/$workload/$cache_state"
  local cache_config="$OUTPUT_DIR/lmcache-config/$label/$workload/$cache_state.yaml"
  mkdir -p "$cache_dir" "$(dirname "$cache_config")"
  # Baseline and variant share only this explicit store.  A new output root
  # therefore yields a cold first member without mutating unrelated cache data.
  cat > "$cache_config" <<EOF
local_cpu: false
max_local_cpu_size: 0.0
local_disk: $cache_dir
max_local_disk_size: 80.0
EOF
  LMCACHE_CONFIG_FILE="$cache_config" run_one "$label" "$baseline_phase" baseline "$workload" "$cache_state" "$label"
  LMCACHE_CONFIG_FILE="$cache_config" run_one "$label" "$variant_phase" variant "$workload" "$cache_state" "$label"
  "$PYTHON" scripts/reporting/validate_kv_core_acceptance.py \
    --baseline "$OUTPUT_DIR/$label/$workload/$cache_state/baseline" \
    --variant "$OUTPUT_DIR/$label/$workload/$cache_state/variant" \
    --phase "$variant_phase" --output "$OUTPUT_DIR/$label/$workload/$cache_state/acceptance.json"
}

for workload in repeated_long_prefix random_no_reuse constrained_kv_churn queued_concurrency; do
  for cache_state in cold warm; do
    run_pair E1 E0 E1 "$workload" "$cache_state"
    run_pair E2 E0 E2 "$workload" "$cache_state"
    run_pair E3 E2 E3 "$workload" "$cache_state"
    run_pair E4 E3 E4 "$workload" "$cache_state"
  done
done

echo "KV-Core controlled suite completed: $OUTPUT_DIR"
