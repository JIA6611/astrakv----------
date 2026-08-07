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
export ASTRAKV_PYTHON="$PYTHON"
TASK1_DIR="$ROOT/datasets/task1_qasper"
OUTPUT_DIR="$ROOT/results/qasper-prefix-prefetch-suite-$(date -u +%Y%m%dT%H%M%SZ)"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen2.5-7B-Instruct}"
HOST="${ASTRAKV_HOST:-127.0.0.1}"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="24576"
GPU_MEMORY_UTILIZATION="0.72"
TIMEOUT="900"
LIMIT="0"
ARRIVAL_GAP_MS="25"
REQUIRED_WINDOW_MS="50"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task1-dir) TASK1_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --arrival-gap-ms) ARRIVAL_GAP_MS="$2"; shift 2 ;;
    --required-window-ms) REQUIRED_WINDOW_MS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_for_endpoint() {
  local log_path="$1"
  for _ in $(seq 1 90); do
    if curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  tail -n 120 "$log_path" >&2 || true
  return 1
}

start_server() {
  local run_id="$1" state_dir="$2" runtime_env="$3" cache_dir="$4" log_path="$5"
  cleanup
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
  local workload_label="$1" role="$2" canonical="$3" scheduler_hints="$4" pair_id="$5"
  local pair_role="variant"
  if [[ "$role" == "disabled" ]]; then
    pair_role="baseline"
  fi
  local run_id="qasper-prefix-prefetch-${workload_label}-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$workload_label/$role"
  local state_dir="$OUTPUT_DIR/$workload_label/${role}-state"
  local cache_dir="$OUTPUT_DIR/$workload_label/${role}-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$OUTPUT_DIR/$workload_label/${role}-server.log"
  mkdir -p "$run_dir" "$state_dir"
  local secret
  secret="$($PYTHON -c 'import secrets; print(secrets.token_hex(32))')"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
ASTRAKV_ENABLE_ONLINE_POLICY=true
ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=true
ASTRAKV_ONLINE_OFFLINE_GATE_PATH=$state_dir/offline_gate.json
ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH=true
ASTRAKV_ONLINE_PREFETCH_MODE=${role%%_*}
EOF
  if [[ -n "$scheduler_hints" ]]; then
    printf 'ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH=%s\n' "$scheduler_hints" >> "$runtime_env"
  fi
  cat > "$state_dir/offline_gate.json" <<'EOF'
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["qasper"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
  chmod 600 "$runtime_env"
  start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" --backend "vllm-lmcache047" \
    --output-dir "$run_dir" --workload-jsonl "$canonical" \
    --run-id "$run_id" --workload-id "qasper-${workload_label}" \
    --model-revision "local-qwen2.5-7b-instruct" --tokenizer-revision "local-qwen2.5-7b-instruct" \
    --dtype "bfloat16" --quantization "unquantized" --random-seed "0" --cache-state cold \
    --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "$pair_id" --pair-role "$pair_role" --claim-scope online_control \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "${run_id}-session" --request-context-secret-hex "$secret" \
    --enable-samples --metrics-interval 1.0 --timeout "$TIMEOUT" --output-tokens 128
  cleanup
}

mkdir -p "$OUTPUT_DIR/grouped/materialized" "$OUTPUT_DIR/grouped/analysis" "$OUTPUT_DIR/grouped/hints"

"$PYTHON" scripts/benchmark/materialize_task1_qasper_workload.py \
  --task1-dir "$TASK1_DIR" --task1-workload grouped --output-dir "$OUTPUT_DIR/grouped/materialized"

CANONICAL="$OUTPUT_DIR/grouped/materialized/qasper_grouped_canonical_workload.jsonl"
if [[ "$LIMIT" -gt 0 ]]; then
  LIMITED="$OUTPUT_DIR/grouped/materialized/qasper_grouped_limit_${LIMIT}.jsonl"
  head -n "$LIMIT" "$CANONICAL" > "$LIMITED"
  CANONICAL="$LIMITED"
fi

"$PYTHON" scripts/reporting/build_qasper_prefix_prefetch_hints.py \
  --workload-jsonl "$CANONICAL" --output-dir "$OUTPUT_DIR/grouped/hints"

"$PYTHON" scripts/reporting/analyze_qasper_prefetch_window.py \
  --workload-jsonl "$CANONICAL" --output-dir "$OUTPUT_DIR/grouped/analysis" \
  --required-window-ms "$REQUIRED_WINDOW_MS" --arrival-gap-ms "$ARRIVAL_GAP_MS"

PACED_CANONICAL="$OUTPUT_DIR/grouped/analysis/qasper_grouped_prefetch_friendly_workload.jsonl"
PAIR_ID="qasper-prefix-prefetch-grouped-$(date -u +%Y%m%dT%H%M%SZ)"
HINTS="$OUTPUT_DIR/grouped/hints/prefix_prefetch_hints.jsonl"

run_condition grouped disabled "$PACED_CANONICAL" "$HINTS" "$PAIR_ID"
run_condition grouped prefix_only "$PACED_CANONICAL" "$HINTS" "$PAIR_ID"
run_condition grouped hybrid "$PACED_CANONICAL" "$HINTS" "$PAIR_ID"
run_condition grouped hybrid_no_hints "$PACED_CANONICAL" "" "$PAIR_ID-no-hints"

mkdir -p "$OUTPUT_DIR/grouped/comparison"
"$PYTHON" scripts/reporting/compare_real_runs.py \
  --run "baseline=$OUTPUT_DIR/grouped/disabled" \
  --run "variant=$OUTPUT_DIR/grouped/hybrid" \
  --output-dir "$OUTPUT_DIR/grouped/comparison"

echo "QASPER prefix prefetch suite completed: $OUTPUT_DIR"
