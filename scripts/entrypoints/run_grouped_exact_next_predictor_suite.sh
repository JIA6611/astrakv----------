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
OUTPUT_DIR="${ASTRAKV_ROOT:-$ROOT}/results/grouped-exact-next-predictor-suite-$TIMESTAMP"
GROUPED_ROOT=""
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
HOST="${ASTRAKV_HOST:-127.0.0.1}"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="24576"
GPU_MEMORY_UTILIZATION="0.72"
TIMEOUT="900"
LIMIT="0"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_grouped_exact_next_predictor_suite.sh [options]

Runs paired baseline/variant grouped exact-next benchmarks for qasper and
multifieldqa_en, with sidecar-generated predictor advisories enabled only for
the variant run.

Options:
  --grouped-root PATH          Root containing */grouped_prompts.jsonl
  --output-dir PATH            Result directory root
  --model PATH                 Local model path
  --host HOST                  Loopback bind host (default 127.0.0.1)
  --port PORT                  vLLM port reused serially (default 18000)
  --context-port PORT          Runtime context port reused serially (default 17900)
  --max-model-len N            vLLM maximum sequence length (default 24576)
  --gpu-memory-utilization N   vLLM GPU memory fraction (default 0.72)
  --limit N                    Optional request limit per dataset
  --timeout SECONDS            Per-request timeout (default 900)
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Model path is missing: $MODEL" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "Only loopback hosts are allowed" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
if [[ -z "$GROUPED_ROOT" ]]; then
  echo "--grouped-root is required because grouped prompts are not committed under the repo." >&2
  exit 2
fi

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
  for _ in $(seq 1 120); do
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

write_runtime_env() {
  local run_id="$1" state_dir="$2" runtime_env="$3" secret="$4" role="$5" sidecar_path="$6"
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
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["grouped_exact_next"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=true\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\nASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH=%s\n' \
      "$state_dir/offline_gate.json" "$sidecar_path" >> "$runtime_env"
  fi
  chmod 600 "$runtime_env"
}

materialize_dataset() {
  local dataset="$1"
  local materialized_dir="$OUTPUT_DIR/$dataset/materialized"
  mkdir -p "$materialized_dir"
  "$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
    --grouped-prompts-jsonl "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" \
    --output-dir "$materialized_dir" \
    --dataset "$dataset" \
    --task "$dataset" \
    --limit "$LIMIT"
}

build_analysis_artifacts() {
  local dataset="$1"
  local analysis_dir="$OUTPUT_DIR/$dataset/analysis"
  mkdir -p "$analysis_dir"
  "$PYTHON" scripts/reporting/build_unified_reuse_analysis.py \
    --grouped-root "$GROUPED_ROOT" \
    --results-root "$OUTPUT_DIR" \
    --output-dir "$analysis_dir"
  "$PYTHON" scripts/reporting/build_predictor_candidate_report.py \
    --analysis-jsonl "$analysis_dir/unified_reuse_analysis.jsonl" \
    --output-dir "$analysis_dir/candidates" \
    --source-name "$dataset" \
    --predicted-class exact-next
}

run_condition() {
  local dataset="$1" role="$2" canonical="$3" pair_id="$4" candidate_report="$5"
  local run_id="grouped-exact-next-${dataset}-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$dataset/$role"
  local state_dir="$OUTPUT_DIR/$dataset/${role}-state"
  local cache_dir="$OUTPUT_DIR/$dataset/${role}-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$OUTPUT_DIR/$dataset/${role}-server.log"
  local sidecar_dir="$state_dir/sidecar"
  mkdir -p "$run_dir" "$state_dir" "$sidecar_dir"
  local secret
  secret="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  local sidecar_path="$sidecar_dir/sidecar_prediction.jsonl"
  if [[ "$role" == "variant" ]]; then
    "$PYTHON" scripts/reporting/build_sidecar_prediction.py \
      --candidate-report "$candidate_report" \
      --output-dir "$sidecar_dir" \
      --run-id "$run_id" \
      --lead-time-ms 250 \
      --predicted-class exact-next
  fi
  write_runtime_env "$run_id" "$state_dir" "$runtime_env" "$secret" "$role" "$sidecar_path"
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
    --workload-jsonl "$canonical" \
    --run-id "$run_id" \
    --workload-id "${dataset}_grouped_exact_next" \
    --model-revision "local-qwen3-8b" \
    --tokenizer-revision "local-qwen3-8b" \
    --dtype "bfloat16" \
    --quantization "unquantized" \
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
    --output-tokens 128
  cleanup
}

mkdir -p "$OUTPUT_DIR"
printf 'suite_started_at=%s\nmodel=%s\ngrouped_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$GROUPED_ROOT" > "$OUTPUT_DIR/suite_metadata.txt"

for dataset in qasper multifieldqa_en; do
  materialize_dataset "$dataset"
  build_analysis_artifacts "$dataset"
  canonical="$OUTPUT_DIR/$dataset/materialized/${dataset}_grouped_exact_next_canonical_workload.jsonl"
  candidate_report="$OUTPUT_DIR/$dataset/analysis/candidates/predictor_candidate_report.jsonl"
  pair_id="grouped-exact-next-${dataset}-$TIMESTAMP"
  run_condition "$dataset" baseline "$canonical" "$pair_id" "$candidate_report"
  run_condition "$dataset" variant "$canonical" "$pair_id" "$candidate_report"
  comparison_dir="$OUTPUT_DIR/$dataset/comparison"
  mkdir -p "$comparison_dir"
  set +e
  "$PYTHON" scripts/reporting/compare_real_runs.py \
    --run "baseline=$OUTPUT_DIR/$dataset/baseline" \
    --run "variant=$OUTPUT_DIR/$dataset/variant" \
    --output-dir "$comparison_dir"
  comparison_status=$?
  set -e
  printf '%s\n' "$comparison_status" > "$comparison_dir/compare_exit_status.txt"
  "$PYTHON" scripts/reporting/build_observation_feasibility_report.py \
    --candidate-report "$candidate_report" \
    --request-results "$OUTPUT_DIR/$dataset/variant/request_results.jsonl" \
    --runtime-events-raw "$OUTPUT_DIR/$dataset/variant-state/runtime_events_raw.jsonl" \
    --runtime-command-receipts "$OUTPUT_DIR/$dataset/variant-state/runtime_command_receipts.jsonl" \
    --runtime-structured-events "$OUTPUT_DIR/$dataset/variant-state/runtime_structured_events.jsonl" \
    --backend-binding-events "$OUTPUT_DIR/$dataset/variant-state/backend_binding_events.jsonl" \
    --online-profile-checkpoint "$OUTPUT_DIR/$dataset/variant-state/online_profile_checkpoint.json" \
    --output-dir "$OUTPUT_DIR/$dataset/observation"
done

printf 'suite_completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_DIR/suite_metadata.txt"
echo "Grouped exact-next predictor suite completed: $OUTPUT_DIR"
