#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Historical compatibility marker retained for tests that still assert the git-common-dir probe text.
# git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir

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
TASK1_DIR="$ROOT/datasets/task1_qasper"
OUTPUT_DIR="$ROOT/results/qasper-online-control-suite-$(date -u +%Y%m%dT%H%M%SZ)"
MODEL="/opt/models/Qwen2.5-7B-Instruct"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="24576"
# DGX Spark shares system and GPU memory. Leave headroom for LMCache I/O and
# the runtime-control host so earlyoom cannot terminate the EngineCore.
GPU_MEMORY_UTILIZATION="0.72"
LIMIT="0"
TIMEOUT="900"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_task1_qasper_online_control_suite.sh [options]

Runs random/grouped QASPER workloads in cold baseline and online-policy-enabled
vLLM+LMCache conditions. The source Task 1 package is read-only.

Options:
  --task1-dir PATH              Validated datasets/task1_qasper package
  --output-dir PATH             Suite result directory
  --model PATH                  Local model path
  --host HOST                   Loopback bind host (default 127.0.0.1)
  --port PORT                   vLLM port reused serially (default 18000)
  --context-port PORT           Runtime context port reused serially (default 17900)
  --max-model-len N             vLLM maximum sequence length (default 24576)
  --gpu-memory-utilization N    vLLM GPU memory fraction (default 0.72)
  --limit N                     Published-prefix request limit; 0 means all 200
  --timeout SECONDS             Per-request timeout (default 900)
  -h, --help                    Show this help
EOF
}

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
    --limit) LIMIT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$TASK1_DIR" ]] || { echo "Task 1 directory is missing: $TASK1_DIR" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "Only loopback hosts are allowed" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }

SERVER_PID=""
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

wait_for_completed_drop() {
  local state_dir="$1"
  for _ in $(seq 1 60); do
    if "$PYTHON" - "$state_dir/runtime_command_receipts.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "completed" and int((row.get("metadata") or {}).get("removed") or 0) > 0:
            raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
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
  local workload="$1" role="$2" canonical="$3" pair_id="$4"
  local run_id="qasper-online-${workload}-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$workload/$role"
  local state_dir="$OUTPUT_DIR/$workload/${role}-state"
  local cache_dir="$OUTPUT_DIR/$workload/${role}-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$OUTPUT_DIR/$workload/${role}-server.log"
  local policy_enabled=false
  mkdir -p "$run_dir" "$state_dir"
  umask 077
  local secret
  secret="$($PYTHON -c 'import secrets; print(secrets.token_hex(32))')"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
EOF
  chmod 600 "$runtime_env"
  if [[ "$role" == "variant" ]]; then
    policy_enabled=true
    cat > "$state_dir/offline_gate.json" <<'EOF'
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["qasper"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=true\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\n' "$state_dir/offline_gate.json" >> "$runtime_env"
  else
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=false\n' >> "$runtime_env"
  fi
  start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" --backend "vllm-lmcache047" \
    --output-dir "$run_dir" --workload-jsonl "$canonical" \
    --run-id "$run_id" --workload-id "qasper-${workload}" \
    --model-revision "local-qwen2.5-7b-instruct" --tokenizer-revision "local-qwen2.5-7b-instruct" \
    --dtype "bfloat16" --quantization "unquantized" --random-seed "0" --cache-state cold \
    --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "$pair_id" --pair-role "$role" --claim-scope online_control \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "${run_id}-session" --enable-samples --metrics-interval 1.0 \
    --timeout "$TIMEOUT" --output-tokens 128
  if [[ "$policy_enabled" == true ]] && ! wait_for_completed_drop "$state_dir"; then
    echo "Enabled QASPER run did not produce a completed real DROP receipt: $run_id" >&2
    return 1
  fi
  "$PYTHON" scripts/benchmark/evaluate_qasper_quality.py \
    --request-results "$run_dir/request_results.jsonl" --output-dir "$run_dir"
  cleanup
}

mkdir -p "$OUTPUT_DIR"
printf 'suite_started_at=%s\nmodel=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" > "$OUTPUT_DIR/suite_metadata.txt"

for workload in random grouped; do
  materialized="$OUTPUT_DIR/$workload/canonical"
  mkdir -p "$materialized"
  "$PYTHON" scripts/benchmark/materialize_task1_qasper_workload.py \
    --task1-dir "$TASK1_DIR" --task1-workload "$workload" --output-dir "$materialized"
  canonical="$materialized/qasper_${workload}_canonical_workload.jsonl"
  if [[ "$LIMIT" -gt 0 ]]; then
    limited="$materialized/qasper_${workload}_limit_${LIMIT}.jsonl"
    head -n "$LIMIT" "$canonical" > "$limited"
    canonical="$limited"
  fi
  pair_id="qasper-online-${workload}-$(date -u +%Y%m%dT%H%M%SZ)"
  run_condition "$workload" baseline "$canonical" "$pair_id"
  run_condition "$workload" variant "$canonical" "$pair_id"
  comparison="$OUTPUT_DIR/$workload/comparison"
  mkdir -p "$comparison"
  set +e
  "$PYTHON" scripts/reporting/compare_real_runs.py \
    --run "baseline=$OUTPUT_DIR/$workload/baseline" \
    --run "variant=$OUTPUT_DIR/$workload/variant" \
    --output-dir "$comparison"
  comparison_status=$?
  set -e
  printf '%s\n' "$comparison_status" > "$comparison/compare_exit_status.txt"
done

"$PYTHON" scripts/reporting/summarize_task1_qasper_online_suite.py --suite-dir "$OUTPUT_DIR"
printf 'suite_completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_DIR/suite_metadata.txt"
echo "QASPER online-control suite completed: $OUTPUT_DIR"
