#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

pick_python() {
  local candidate=""
  for candidate in \
    "${ASTRAKV_PYTHON:-}" \
    "$ROOT/.venv_from_szl/bin/python" \
    "$ROOT/.venv/bin/python"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  printf '%s\n' "$ROOT/.venv_from_szl/bin/python"
}

PYTHON="$(pick_python)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$ROOT/results/policy-ab-ttft-suite-$TIMESTAMP"
MODEL="/opt/models/Qwen3-8B-AWQ"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="16384"
GPU_MEMORY_UTILIZATION="0.60"
KV_CACHE_MEMORY_BYTES="2G"
TIMEOUT="900"
ANCHOR_COUNT="6"
CHURN_VARIANTS="12"
PROMPT_TOKENS="8192"
WARMUP_CYCLES="5"
SAMPLE_CYCLES="30"
IDLE_SECONDS="1.5"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_policy_ab_ttft_suite.sh [options]

Runs a strategy-level AstraKV ON/OFF A/B experiment for prefill-only TTFT:
  anchor seed -> churn primary -> churn secondary -> idle -> anchor revisit

Options:
  --output-dir PATH             Result directory root
  --model PATH                  Local model path (default /opt/models/Qwen3-8B-AWQ)
  --host HOST                   Loopback bind host (default 127.0.0.1)
  --port PORT                   vLLM port reused serially (default 18000)
  --context-port PORT           Runtime context port reused serially (default 17900)
  --max-model-len N             vLLM maximum sequence length (default 16384)
  --gpu-memory-utilization N    vLLM GPU memory fraction (default 0.60)
  --kv-cache-memory-bytes SIZE  Explicit KV budget, e.g. 2G (default 2G)
  --anchor-count N              Anchor prompt count (default 6)
  --churn-variants N            B/C churn prompt variants per family (default 12)
  --prompt-tokens N             Approximate prompt length target (default 8192)
  --warmup-cycles N             Warm-up cycles included before sampling (default 5)
  --sample-cycles N             Counted revisit cycles per role (default 30)
  --idle-seconds N              Sleep inserted before revisit (default 1.5)
  --timeout SECONDS             Per-request timeout (default 900)
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --kv-cache-memory-bytes) KV_CACHE_MEMORY_BYTES="$2"; shift 2 ;;
    --anchor-count) ANCHOR_COUNT="$2"; shift 2 ;;
    --churn-variants) CHURN_VARIANTS="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --warmup-cycles) WARMUP_CYCLES="$2"; shift 2 ;;
    --sample-cycles) SAMPLE_CYCLES="$2"; shift 2 ;;
    --idle-seconds) IDLE_SECONDS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Model path is missing: $MODEL" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "Only loopback hosts are allowed" >&2; exit 2; }
"$PYTHON" -c "import vllm" >/dev/null 2>&1 || { echo "Selected Python cannot import vllm: $PYTHON" >&2; exit 2; }

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

require_port_free() {
  if curl --max-time 2 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Refusing to start: endpoint already responds on http://${HOST}:${PORT}/v1/models" >&2
    exit 2
  fi
}

wait_for_endpoint() {
  local log_path="$1"
  for _ in $(seq 1 180); do
    if curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -n 200 "$log_path" >&2 || true
  return 1
}

write_runtime_env() {
  local run_id="$1" state_dir="$2" runtime_env="$3" secret="$4" role="$5"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
ASTRAKV_ENABLE_ONLINE_POLICY=false
ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=true
EOF
  if [[ "$role" == "variant" ]]; then
    cat > "$state_dir/offline_gate.json" <<'EOF'
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["policy_ab_ttft","policy_ab_ttft_suite"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=true\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\n' "$state_dir/offline_gate.json" >> "$runtime_env"
  fi
  chmod 600 "$runtime_env"
}

start_server() {
  local run_id="$1" state_dir="$2" runtime_env="$3" cache_dir="$4" log_path="$5"
  cleanup
  require_port_free
  mkdir -p "$cache_dir"
  sed "s|^local_disk:.*|local_disk: $cache_dir|" configs/lmcache_disk_example.yaml > "$state_dir/lmcache.yaml"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  ASTRAKV_MODEL="$MODEL" \
  ASTRAKV_HOST="$HOST" \
  ASTRAKV_PORT="$PORT" \
  ASTRAKV_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  ASTRAKV_KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
  ASTRAKV_PREFIX_CACHING=true \
  ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_LMCACHE047_EVENTS="$state_dir/hook_raw.jsonl" \
  ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" \
  ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" \
  ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
  LMCACHE_CONFIG_FILE="$state_dir/lmcache.yaml" \
  ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  nohup bash scripts/launch/launch_lmcache_vllm.sh disk > "$log_path" 2>&1 < /dev/null &
  SERVER_PID="$!"
  wait_for_endpoint "$log_path"
}

run_condition() {
  local role="$1" workload_jsonl="$2" pair_id="$3"
  local run_id="policy-ab-ttft-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$role"
  local state_dir="$OUTPUT_DIR/${role}-state"
  local cache_dir="$OUTPUT_DIR/${role}-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$OUTPUT_DIR/${role}-server.log"
  local secret=""
  mkdir -p "$run_dir" "$state_dir"
  secret="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  write_runtime_env "$run_id" "$state_dir" "$runtime_env" "$secret" "$role"
  start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" \
    --backend "vllm-lmcache047" \
    --output-dir "$run_dir" \
    --workload-jsonl "$workload_jsonl" \
    --run-id "$run_id" \
    --workload-id "policy_ab_ttft" \
    --model-revision "local-qwen3-8b-awq" \
    --tokenizer-revision "local-qwen3-8b-awq" \
    --dtype "bfloat16" \
    --quantization "awq" \
    --random-seed "0" \
    --cache-state cold \
    --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "$pair_id" \
    --pair-role "$role" \
    --claim-scope online_control \
    --runtime-state-dir "$state_dir" \
    --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "${run_id}-session" \
    --request-context-secret-hex "$secret" \
    --enable-samples \
    --metrics-interval 1.0 \
    --timeout "$TIMEOUT" \
    --output-tokens 1
  cleanup
}

mkdir -p "$OUTPUT_DIR/workload"
printf 'suite_started_at=%s\nmodel=%s\nkv_cache_memory_bytes=%s\ngpu_memory_utilization=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$KV_CACHE_MEMORY_BYTES" "$GPU_MEMORY_UTILIZATION" \
  > "$OUTPUT_DIR/suite_metadata.txt"

"$PYTHON" scripts/benchmark/generate_policy_ab_ttft_workload.py \
  --output-dir "$OUTPUT_DIR/workload" \
  --anchor-count "$ANCHOR_COUNT" \
  --churn-variants "$CHURN_VARIANTS" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --warmup-cycles "$WARMUP_CYCLES" \
  --sample-cycles "$SAMPLE_CYCLES" \
  --idle-seconds "$IDLE_SECONDS" \
  --expected-output-tokens 1 \
  --context-length "$PROMPT_TOKENS"

WORKLOAD_JSONL="$OUTPUT_DIR/workload/policy_ab_ttft_workload.jsonl"
PAIR_ID="policy-ab-ttft-$TIMESTAMP"
run_condition baseline "$WORKLOAD_JSONL" "$PAIR_ID"
run_condition variant "$WORKLOAD_JSONL" "$PAIR_ID"

mkdir -p "$OUTPUT_DIR/comparison"
"$PYTHON" scripts/reporting/compare_real_runs.py \
  --run "baseline=$OUTPUT_DIR/baseline" \
  --run "variant=$OUTPUT_DIR/variant" \
  --output-dir "$OUTPUT_DIR/comparison"

"$PYTHON" scripts/reporting/build_policy_ab_ttft_report.py \
  --suite-dir "$OUTPUT_DIR"

printf 'suite_completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_DIR/suite_metadata.txt"
echo "Policy A/B TTFT suite completed: $OUTPUT_DIR"
