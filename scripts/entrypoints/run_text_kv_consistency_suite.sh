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
OUTPUT_DIR="$ROOT/results/text-kv-consistency-suite-$TIMESTAMP"
MODEL="/opt/models/Qwen3-8B"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="16384"
GPU_MEMORY_UTILIZATION="0.60"
TIMEOUT="900"
BLOCK_SIZE_TOKENS="16"
KV_BYTES_PER_TOKEN="1"
CONTEXT_LENGTHS="8192,16384"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_text_kv_consistency_suite.sh [options]

Runs the target-1 text/KV consistency suite with three warmup conditions:
  cold, warm, hot

Options:
  --output-dir PATH             Result directory root
  --model PATH                  Local model path
  --host HOST                   Loopback bind host (default 127.0.0.1)
  --port PORT                   vLLM port reused serially (default 18000)
  --context-port PORT           Runtime context port reused serially (default 17900)
  --context-lengths CSV         Context lengths to generate/run (default 8192,16384)
  --max-model-len N             vLLM maximum sequence length (default 16384)
  --gpu-memory-utilization N    vLLM GPU memory fraction (default 0.60)
  --block-size-tokens N         Tokenizer block size for text observation (default 16)
  --kv-bytes-per-token N        Positive placeholder for workflow observation (default 1)
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
    --context-lengths) CONTEXT_LENGTHS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --block-size-tokens) BLOCK_SIZE_TOKENS="$2"; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2 ;;
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
  local run_id="$1" state_dir="$2" runtime_env="$3" secret="$4"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
ASTRAKV_ENABLE_ONLINE_POLICY=false
EOF
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

run_benchmark() {
  local run_id="$1" run_dir="$2" workload_jsonl="$3"
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" \
    --backend "vllm-lmcache047" \
    --output-dir "$run_dir" \
    --workload-jsonl "$workload_jsonl" \
    --run-id "$run_id" \
    --workload-id "text_kv_consistency" \
    --model-revision "local-qwen3-8b" \
    --tokenizer-revision "local-qwen3-8b" \
    --dtype "bfloat16" \
    --quantization "unquantized" \
    --random-seed "0" \
    --cache-state cold \
    --connector-version "lmcache-vllm-v1-0.4.7" \
    --runtime-state-dir "$(dirname "$run_dir")/state" \
    --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "${run_id}-session" \
    --request-context-secret-hex "$RUNTIME_SECRET" \
    --enable-samples \
    --metrics-interval 1.0 \
    --timeout "$TIMEOUT" \
    --output-tokens 64
}

validate_generated_bundle() {
  local analysis_path="$1" warmup_path="$2" replay_path="$3"
  "$PYTHON" - "$analysis_path" "$warmup_path" "$replay_path" <<'PY'
import json
import sys
from pathlib import Path

analysis = Path(sys.argv[1])
warmup = Path(sys.argv[2])
replay = Path(sys.argv[3])

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

analysis_rows = load(analysis)
warmup_rows = load(warmup)
replay_rows = load(replay)
if len(analysis_rows) != 3:
    raise SystemExit(f"analysis workload must contain exactly 3 rows, got {len(analysis_rows)}")
if len(warmup_rows) != 3:
    raise SystemExit(f"warmup workload must contain exactly 3 rows, got {len(warmup_rows)}")
if len(replay_rows) != 6:
    raise SystemExit(f"replay workload must contain exactly 6 rows, got {len(replay_rows)}")
request_ids = [row["request_id"] for row in analysis_rows]
if request_ids != ["exact-probe", "sim90-probe", "sim80-probe"]:
    raise SystemExit(f"unexpected analysis request_ids: {request_ids}")
PY
}

extract_cache_events_for_condition() {
  local condition_root="$1" server_log="$2"
  local run_dir="$condition_root/run"
  local out_dir="$condition_root/cache_events"
  mkdir -p "$out_dir"
  "$PYTHON" scripts/benchmark/extract_cache_events.py \
    --server-log "$server_log" \
    --request-results "$run_dir/request_results.jsonl" \
    --benchmark-results "$run_dir/benchmark_results.csv" \
    --output-dir "$out_dir"
}

run_condition() {
  local condition="$1" warmup_repeats="$2" context_label="$3" analysis_workload="$4" warmup_workload="$5"
  local condition_root="$OUTPUT_DIR/$condition/$context_label"
  local state_dir="$condition_root/state"
  local run_dir="$condition_root/run"
  local cache_dir="$condition_root/lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$condition_root/server.log"
  mkdir -p "$state_dir" "$run_dir"

  local run_id="text-kv-consistency-${condition}-${context_label}-$(date -u +%Y%m%dT%H%M%SZ)"
  RUNTIME_SECRET="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  write_runtime_env "$run_id" "$state_dir" "$runtime_env" "$RUNTIME_SECRET"
  if ! start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"; then
    echo "start_server_failed context=${context_label} condition=${condition}" > "$condition_root/failure.txt"
    cleanup
    return 1
  fi

  if [[ "$warmup_repeats" -gt 0 ]]; then
    for repeat_index in $(seq 1 "$warmup_repeats"); do
      local warmup_run_dir="$condition_root/warmup-pass-$repeat_index"
      mkdir -p "$warmup_run_dir"
      if ! run_benchmark "$run_id" "$warmup_run_dir" "$warmup_workload"; then
        echo "warmup_failed context=${context_label} condition=${condition} repeat=${repeat_index}" > "$condition_root/failure.txt"
        cleanup
        return 1
      fi
    done
  fi

  if ! run_benchmark "$run_id" "$run_dir" "$analysis_workload"; then
    echo "analysis_failed context=${context_label} condition=${condition}" > "$condition_root/failure.txt"
    cleanup
    return 1
  fi
  cleanup
  if ! extract_cache_events_for_condition "$condition_root" "$server_log"; then
    echo "cache_extract_failed context=${context_label} condition=${condition}" >> "$condition_root/failure.txt"
    return 1
  fi
  return 0
}

mkdir -p "$OUTPUT_DIR/workload" "$OUTPUT_DIR/text_observation"
printf 'suite_started_at=%s\nmodel=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" > "$OUTPUT_DIR/suite_metadata.txt"

"$PYTHON" scripts/benchmark/generate_text_kv_consistency_workload.py \
  --output-dir "$OUTPUT_DIR/workload" \
  --context-lengths "$CONTEXT_LENGTHS" \
  --block-size-tokens "$BLOCK_SIZE_TOKENS" \
  --model-path "$MODEL"

IFS=',' read -r -a CONTEXT_LENGTH_ARRAY <<< "$CONTEXT_LENGTHS"
for raw_context_length in "${CONTEXT_LENGTH_ARRAY[@]}"; do
  context_length="$(echo "$raw_context_length" | xargs)"
  [[ -n "$context_length" ]] || continue
  if [[ "$context_length" == "8192" ]]; then
    context_label="ctx8k"
  elif [[ "$context_length" == "16384" ]]; then
    context_label="ctx16k"
  else
    context_label="ctx$((context_length / 1024))k"
  fi

  ANALYSIS_WORKLOAD="$OUTPUT_DIR/workload/$context_label/analysis_workload.jsonl"
  WARMUP_WORKLOAD="$OUTPUT_DIR/workload/$context_label/warmup_workload.jsonl"
  REPLAY_WORKLOAD="$OUTPUT_DIR/workload/$context_label/pairwise_reference_replay.jsonl"
  validate_generated_bundle "$ANALYSIS_WORKLOAD" "$WARMUP_WORKLOAD" "$REPLAY_WORKLOAD"

  "$PYTHON" scripts/benchmark/observe_workflow_reuse.py \
    --replay-jsonl "$REPLAY_WORKLOAD" \
    --output-dir "$OUTPUT_DIR/text_observation/$context_label" \
    --model-path "$MODEL" \
    --block-size-tokens "$BLOCK_SIZE_TOKENS" \
    --kv-bytes-per-token "$KV_BYTES_PER_TOKEN"
  # The tokenizer-backed observation stage is the source of truth for shared prefix ratio.

  run_condition cold 0 "$context_label" "$ANALYSIS_WORKLOAD" "$WARMUP_WORKLOAD" || true
  run_condition warm 1 "$context_label" "$ANALYSIS_WORKLOAD" "$WARMUP_WORKLOAD" || true
  run_condition hot 2 "$context_label" "$ANALYSIS_WORKLOAD" "$WARMUP_WORKLOAD" || true

  mkdir -p "$OUTPUT_DIR/comparison/$context_label/cold_vs_warm" "$OUTPUT_DIR/comparison/$context_label/cold_vs_hot"
  "$PYTHON" scripts/reporting/compare_real_runs.py \
    --run "cold=$OUTPUT_DIR/cold/$context_label/run" \
    --run "warm=$OUTPUT_DIR/warm/$context_label/run" \
    --output-dir "$OUTPUT_DIR/comparison/$context_label/cold_vs_warm" \
    --unpaired || true
  "$PYTHON" scripts/reporting/compare_real_runs.py \
    --run "cold=$OUTPUT_DIR/cold/$context_label/run" \
    --run "hot=$OUTPUT_DIR/hot/$context_label/run" \
    --output-dir "$OUTPUT_DIR/comparison/$context_label/cold_vs_hot" \
    --unpaired || true
done

"$PYTHON" scripts/reporting/build_text_kv_consistency_report.py \
  --suite-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/report"

printf 'suite_completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_DIR/suite_metadata.txt"
echo "Text/KV consistency suite completed: $OUTPUT_DIR"
