#!/usr/bin/env bash
set -Eeuo pipefail

# E3-P is an exploratory, single-prefix SSD-to-CPU prefetch measurement. It
# intentionally does not certify E3 correctness, E4, or capacity claims. Both
# arms have LocalCPUBackend, LocalDiskBackend, native admission, and identical
# target CPU invalidation. Only CPU prefetch changes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
SOURCE_WORKLOAD=""
OUTPUT_DIR="$ROOT/results/e3-prefetch-performance-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
SOURCE_REQUEST_ID=""
REPEATS="1"
REVISITS="16"
PREFETCH_LEAD_S="0.25"
OUTPUT_TOKENS="16"
HOST="127.0.0.1"
PORT="18200"
CONTEXT_PORT="18190"
TIMEOUT="900"
GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.72}"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_e3_prefetch_performance_suite.sh \
  --source-workload FILE --patch-manifest FILE --callback-smoke FILE [options]

The source must be a canonical exact-prefix runtime workload. The suite builds
one seed and 16 identical revisits by default, then compares CPU prefetch off
against SSD->LocalCPUBackend prefetch on. Results are exploratory only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-workload) SOURCE_WORKLOAD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --source-request-id) SOURCE_REQUEST_ID="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --revisits) REVISITS="$2"; shift 2 ;;
    --prefetch-lead-s) PREFETCH_LEAD_S="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
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

[[ -f "$SOURCE_WORKLOAD" ]] || { echo "--source-workload is required" >&2; exit 2; }
[[ -f "$PATCH_MANIFEST" ]] || { echo "--patch-manifest is required" >&2; exit 2; }
[[ -f "$CALLBACK_SMOKE" ]] || { echo "--callback-smoke is required" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "ASTRAKV_PYTHON is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "only loopback host is supported" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
[[ "$REVISITS" =~ ^[1-9][0-9]*$ ]] || { echo "--revisits must be positive" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to overwrite output directory: $OUTPUT_DIR" >&2; exit 2; }

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
  for _ in $(seq 1 300); do
    curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null && return 0
    sleep 2
  done
  tail -n 160 "$log_path" >&2 || true
  return 1
}

assert_lmcache_healthy() {
  local log_path="$1"
  if grep -Eq "LMCacheEngine marked as init failed|No eviction candidates found in local cpu backend|Memory allocation failed during (async )?disk load" "$log_path"; then
    echo "LMCache did not provide a healthy CPU staging path" >&2
    tail -n 180 "$log_path" >&2 || true
    return 1
  fi
}

write_control() {
  local path="$1" role="$2" prefetch="$3"
  cat > "$path" <<EOF
{
  "schema": "astrakv-e3-prefetch-control-v1",
  "role": "$role",
  "model": "Qwen3-8B",
  "dtype": "bfloat16",
  "mode": "active",
  "topology": "gpu_cpu_ssd",
  "local_cpu_enabled": true,
  "local_disk_enabled": true,
  "admission_enabled": true,
  "prefill_online_calibration_enabled": true,
  "invalidate_disk_backed_cpu_on_prefetch_lead": true,
  "cpu_prefetch_enabled": $prefetch,
  "output_tokens": $OUTPUT_TOKENS,
  "prefetch_lead_s": $PREFETCH_LEAD_S
}
EOF
}

run_role() {
  local repeat_dir="$1" role="$2" prefetch="$3"
  local run_dir="$repeat_dir/$role"
  local state_dir="$run_dir/state"
  local cache_dir="$repeat_dir/lmcache-store/$role"
  local config="$repeat_dir/lmcache-$role.yaml"
  local log_path="$run_dir/server.log"
  local run_id="e3p-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local secret_hex
  mkdir -p "$state_dir" "$cache_dir"
cat > "$config" <<EOF
local_cpu: true
max_local_cpu_size: ${ASTRAKV_LOCAL_CPU_SIZE_GB:-5.0}
local_disk: $cache_dir
max_local_disk_size: 80.0
EOF
  write_control "$run_dir/e3_prefetch_control.json" "$role" "$prefetch"
  secret_hex="$($PYTHON -c 'import secrets; print(secrets.token_hex(32))')"
  cleanup
  ASTRAKV_MODEL="$MODEL" ASTRAKV_HOST="$HOST" ASTRAKV_PORT="$PORT" \
  ASTRAKV_PYTHON="$PYTHON" \
  PYTHONHASHSEED=0 ASTRAKV_VLLM_SEED=0 \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_MAX_MODEL_LEN=32768 \
  LMCACHE_CONFIG_FILE="$config" ASTRAKV_PREFIX_CACHING=false \
  ASTRAKV_ENABLE_LMCACHE047_HOOKS=true ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE=engine_child \
  ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" \
  ASTRAKV_RUNTIME_CONTROL_ENGINE_ID="$run_id-engine" ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0 \
  ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" ASTRAKV_RUNTIME_CONTROL_SESSION_ID="$run_id-session" \
  ASTRAKV_RUNTIME_CONTROL_SECRET_HEX="$secret_hex" ASTRAKV_KV_CORE_VENDOR_PATCH=true \
  ASTRAKV_MODEL_ID=Qwen3-8B ASTRAKV_MODEL_REVISION=local-qwen3-8b \
  ASTRAKV_TOKENIZER_REVISION=local-qwen3-8b ASTRAKV_CHAT_TEMPLATE_REVISION=qwen3-default \
  ASTRAKV_REQUIRE_EXACT_TOKEN_IDS=true ASTRAKV_KV_CORE_MODE=active \
  ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd ASTRAKV_KV_CORE_LOCAL_CPU=true \
  ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true \
  ASTRAKV_KV_CORE_ADMISSION_ENABLED=true ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED="$prefetch" \
  ASTRAKV_KV_CORE_PREFILL_ONLINE_CALIBRATION=true ASTRAKV_KV_CORE_PREFILL_SAMPLE_MIN_TOKENS=32 \
  ASTRAKV_KV_CORE_PREFILL_SAMPLE_MAX_MS_PER_TOKEN=5.0 ASTRAKV_KV_CORE_PREFILL_EMA_ALPHA=0.25 \
  ASTRAKV_KV_CORE_PATCH_VERIFICATION="$OUTPUT_DIR/connector_patch_verification.json" \
  ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP=8192 ASTRAKV_KV_CORE_BOOTSTRAP_LOADS=2 \
  ASTRAKV_KV_CORE_SSD_READ_GBPS=3.0 ASTRAKV_KV_CORE_PREFETCH_DEADLINE_NS=5000000000 \
  ASTRAKV_KV_CORE_PREFETCH_TTL_NS=30000000000 VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  nohup bash scripts/launch/launch_lmcache_vllm.sh cpu > "$log_path" 2>&1 < /dev/null &
  SERVER_PID="$!"
  wait_for_server "$log_path"
  assert_lmcache_healthy "$log_path"
  # ``run_real_benchmark`` records the native KV runtime contract under its
  # existing ``kv_core`` manifest scope. The E3-P validator, not this generic
  # manifest enum, is responsible for keeping the outcome exploratory.
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_VLLM_SEED=0 \
  ASTRAKV_MAX_MODEL_LEN=32768 ASTRAKV_PREFIX_CACHING=false \
  ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  LMCACHE_CONFIG_FILE="$config" "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" --model "$MODEL" --backend vllm-lmcache-kv-core \
    --output-dir "$run_dir" --workload-jsonl "$OUTPUT_DIR/workload/e3_prefetch_single_exact_prefix.jsonl" \
    --run-id "$run_id" --workload-id e3_prefetch_single_exact_prefix \
    --model-revision local-qwen3-8b --tokenizer-revision local-qwen3-8b --dtype bfloat16 \
    --quantization unquantized --tokenizer-path "$MODEL" --chat-template-revision qwen3-default \
    --random-seed 0 --cache-state cold --connector-version lmcache-vllm-v1-0.4.7 \
    --pair-id "e3p-repeat-${repeat_dir##*/}" --pair-role "$role" --claim-scope kv_core \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "$run_id-session" --request-context-secret-hex "$secret_hex" \
    --timeout "$TIMEOUT" --output-tokens "$OUTPUT_TOKENS"
  assert_lmcache_healthy "$log_path"
  for artifact in callback-smoke.json kv_core_native_callbacks.jsonl kv_core_native_receipts.jsonl kv_core_request_accounting.jsonl request_context_associations.jsonl kv_core_prefetch_tickets.jsonl kv_core_policy_decisions.jsonl kv_core_cost_observations.jsonl kv_core_run_metadata.json uma_resource_samples.jsonl; do
    [[ -f "$state_dir/$artifact" ]] && cp "$state_dir/$artifact" "$run_dir/$artifact"
  done
  cleanup
}

mkdir -p "$OUTPUT_DIR/workload"
"$PYTHON" scripts/runtime/verify_kv_core_connector_patch.py \
  --deployment-manifest "$PATCH_MANIFEST" --callback-smoke "$CALLBACK_SMOKE" \
  --output "$OUTPUT_DIR/connector_patch_verification.json"
"$PYTHON" scripts/benchmark/materialize_e3_prefetch_performance_workload.py \
  --source-workload "$SOURCE_WORKLOAD" --output-dir "$OUTPUT_DIR/workload" \
  --source-request-id "$SOURCE_REQUEST_ID" --revisits "$REVISITS" \
  --prefetch-lead-s "$PREFETCH_LEAD_S" --output-tokens "$OUTPUT_TOKENS" \
  > "$OUTPUT_DIR/workload_materialization.json"

for repeat in $(seq 1 "$REPEATS"); do
  repeat_dir="$OUTPUT_DIR/repeat-$(printf '%03d' "$repeat")"
  run_role "$repeat_dir" baseline false
  run_role "$repeat_dir" variant true
  "$PYTHON" scripts/reporting/validate_e3_prefetch_performance.py \
    --baseline "$repeat_dir/baseline" --variant "$repeat_dir/variant" \
    --output "$repeat_dir/e3_prefetch_performance.json"
done

"$PYTHON" - "$OUTPUT_DIR" "$REPEATS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("repeat-*/e3_prefetch_performance.json")):
    records.append({"path": str(path.relative_to(root)), **json.loads(path.read_text(encoding="utf-8"))})
payload = {
    "schema": "astrakv-e3-prefetch-performance-suite-v1",
    "status": "exploratory_performance_only",
    "repeat_count": int(sys.argv[2]),
    "measurement_valid": bool(records) and all(row.get("measurement_valid") is True for row in records),
    "correctness_accepted": False,
    "eligible_for_e4": False,
    "eligible_for_capacity_claim": False,
    "repeat_results": records,
}
(root / "e3_prefetch_performance_suite.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
echo "E3-P exploratory prefetch suite completed: $OUTPUT_DIR"
