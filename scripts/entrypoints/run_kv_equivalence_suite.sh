#!/usr/bin/env bash
set -Eeuo pipefail

# P3a: a single vLLM engine first writes an exact prefix, then compares its
# request-owned native external load with a one-shot scheduler-declined native
# recompute of the same complete token sequence. This is a correctness probe,
# not a latency or capacity benchmark.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
SOURCE_WORKLOAD=""
PROMPTS_FILE=""
OUTPUT_DIR="$ROOT/results/kv-equivalence-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
HOST="127.0.0.1"
PORT="18100"
CONTEXT_PORT="18090"
TIMEOUT="900"
GPU_MEMORY_UTILIZATION="0.72"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_kv_equivalence_suite.sh \
  (--source-workload FILE | --prompts-file FILE) \
  --patch-manifest FILE --callback-smoke FILE [options]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-workload) SOURCE_WORKLOAD="$2"; shift 2 ;;
    --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
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

if [[ -n "$SOURCE_WORKLOAD" && -n "$PROMPTS_FILE" ]]; then
  echo "choose exactly one of --source-workload or --prompts-file" >&2; exit 2
fi
if [[ -n "$SOURCE_WORKLOAD" ]]; then
  [[ -f "$SOURCE_WORKLOAD" ]] || { echo "--source-workload not found: $SOURCE_WORKLOAD" >&2; exit 2; }
elif [[ -n "$PROMPTS_FILE" ]]; then
  [[ -f "$PROMPTS_FILE" ]] || { echo "--prompts-file not found: $PROMPTS_FILE" >&2; exit 2; }
else
  echo "--source-workload or --prompts-file is required" >&2; exit 2
fi
[[ -f "$PATCH_MANIFEST" ]] || { echo "--patch-manifest is required" >&2; exit 2; }
[[ -f "$CALLBACK_SMOKE" ]] || { echo "--callback-smoke is required" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "only loopback host is supported" >&2; exit 2; }

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
wait_for_server() {
  for _ in $(seq 1 300); do
    curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null && return 0
    sleep 2
  done
  tail -n 160 "$OUTPUT_DIR/server.log" >&2 || true
  return 1
}

mkdir -p "$OUTPUT_DIR/workload" "$OUTPUT_DIR/state" "$OUTPUT_DIR/lmcache-store"
export ASTRAKV_EQUIVALENCE_OUTPUT_DIR="$OUTPUT_DIR"
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path.cwd()
paths = [
    root / "astrakv/runtime/vendor_callback_bridge.py",
    root / "astrakv/runtime/lmcache047_runtime_patch.py",
    root / "scripts/benchmark/run_real_benchmark.py",
    root / "scripts/benchmark/materialize_kv_equivalence_workload.py",
    root / "scripts/reporting/validate_kv_equivalence.py",
    root / "scripts/entrypoints/run_kv_equivalence_suite.sh",
]
payload = {
    "schema": "astrakv-kv-equivalence-runtime-source-v1",
    "files": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
}
(Path(os.environ["ASTRAKV_EQUIVALENCE_OUTPUT_DIR"]) / "runtime_source_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
"$PYTHON" scripts/runtime/verify_kv_core_connector_patch.py \
  --deployment-manifest "$PATCH_MANIFEST" --callback-smoke "$CALLBACK_SMOKE" \
  --output "$OUTPUT_DIR/connector_patch_verification.json"
if [[ -n "$PROMPTS_FILE" ]]; then
  "$PYTHON" scripts/benchmark/materialize_kv_equivalence_workload.py \
    --prompts-file "$PROMPTS_FILE" --output-dir "$OUTPUT_DIR/workload" \
    > "$OUTPUT_DIR/workload_materialization.json"
else
  "$PYTHON" scripts/benchmark/materialize_kv_equivalence_workload.py \
    --source-workload "$SOURCE_WORKLOAD" --output-dir "$OUTPUT_DIR/workload" \
    > "$OUTPUT_DIR/workload_materialization.json"
fi

cat > "$OUTPUT_DIR/lmcache.yaml" <<EOF
local_cpu: false
max_local_cpu_size: 2.0
local_disk: $OUTPUT_DIR/lmcache-store
max_local_disk_size: 80.0
EOF
SECRET_HEX="$($PYTHON -c 'import secrets; print(secrets.token_hex(32))')"
RUN_ID="kv-equivalence-$(date -u +%Y%m%dT%H%M%SZ)"
ASTRAKV_PYTHON="$PYTHON" \
ASTRAKV_MODEL="$MODEL" ASTRAKV_HOST="$HOST" ASTRAKV_PORT="$PORT" \
PYTHONHASHSEED=0 ASTRAKV_VLLM_SEED=0 ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
ASTRAKV_MAX_MODEL_LEN=32768 LMCACHE_CONFIG_FILE="$OUTPUT_DIR/lmcache.yaml" \
ASTRAKV_PREFIX_CACHING=false ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE=engine_child ASTRAKV_RUNTIME_CONTROL_RUN_ID="$RUN_ID" \
ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$OUTPUT_DIR/state" ASTRAKV_RUNTIME_CONTROL_ENGINE_ID="$RUN_ID-engine" \
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0 ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
ASTRAKV_RUNTIME_CONTROL_SESSION_ID="$RUN_ID-session" ASTRAKV_RUNTIME_CONTROL_SECRET_HEX="$SECRET_HEX" \
ASTRAKV_KV_CORE_VENDOR_PATCH=true ASTRAKV_MODEL_ID=Qwen3-8B \
ASTRAKV_MODEL_REVISION=local-qwen3-8b ASTRAKV_TOKENIZER_REVISION=local-qwen3-8b \
ASTRAKV_CHAT_TEMPLATE_REVISION=qwen3-default ASTRAKV_REQUIRE_EXACT_TOKEN_IDS=true \
ASTRAKV_KV_CORE_MODE=active ASTRAKV_KV_CORE_TOPOLOGY=gpu_ssd ASTRAKV_KV_CORE_LOCAL_CPU=false \
ASTRAKV_KV_CORE_PATCH_VERIFICATION="$OUTPUT_DIR/connector_patch_verification.json" \
ASTRAKV_KV_CORE_ADMISSION_ENABLED=true ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=false \
ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP=8192 ASTRAKV_KV_CORE_BOOTSTRAP_LOADS=2 \
ASTRAKV_KV_CORE_SSD_READ_GBPS=3.0 ASTRAKV_KV_CORE_EQUIVALENCE_TEST=true \
VLLM_ENGINE_READY_TIMEOUT_S=1200 \
nohup bash scripts/launch/launch_lmcache_vllm.sh disk > "$OUTPUT_DIR/server.log" 2>&1 < /dev/null &
SERVER_PID="$!"
wait_for_server
if grep -q "LMCacheEngine marked as init failed" "$OUTPUT_DIR/server.log"; then
  echo "LMCache initialization failed" >&2
  exit 1
fi
ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_VLLM_SEED=0 \
ASTRAKV_MAX_MODEL_LEN=32768 ASTRAKV_PREFIX_CACHING=false \
ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
LMCACHE_CONFIG_FILE="$OUTPUT_DIR/lmcache.yaml" "$PYTHON" scripts/benchmark/run_real_benchmark.py \
  --base-url "http://${HOST}:${PORT}/v1" --model "$MODEL" --backend vllm-lmcache-kv-core \
  --output-dir "$OUTPUT_DIR/run" --workload-jsonl "$OUTPUT_DIR/workload/kv_equivalence_single_prefix.jsonl" \
  --run-id "$RUN_ID" --workload-id kv_equivalence_single_prefix \
  --model-revision local-qwen3-8b --tokenizer-revision local-qwen3-8b --dtype bfloat16 \
  --quantization unquantized --tokenizer-path "$MODEL" --chat-template-revision qwen3-default \
  --random-seed 0 --cache-state cold --connector-version lmcache-vllm-v1-0.4.7 \
  --pair-id "$RUN_ID" --pair-role variant --claim-scope kv_core --runtime-state-dir "$OUTPUT_DIR/state" \
  --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
  --request-context-session-id "$RUN_ID-session" --request-context-secret-hex "$SECRET_HEX" \
  --timeout "$TIMEOUT" --output-tokens 8
for artifact in callback-smoke.json kv_core_native_callbacks.jsonl kv_core_native_receipts.jsonl kv_core_request_accounting.jsonl request_context_associations.jsonl kv_core_policy_decisions.jsonl kv_core_run_metadata.json; do
  [[ -f "$OUTPUT_DIR/state/$artifact" ]] && cp "$OUTPUT_DIR/state/$artifact" "$OUTPUT_DIR/run/$artifact"
done
"$PYTHON" scripts/reporting/validate_kv_equivalence.py \
  --run-dir "$OUTPUT_DIR/run" --output "$OUTPUT_DIR/kv_equivalence.json"
echo "KV equivalence suite completed: $OUTPUT_DIR"
