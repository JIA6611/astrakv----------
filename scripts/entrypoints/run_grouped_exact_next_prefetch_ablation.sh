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
OUTPUT_DIR="${ASTRAKV_ROOT:-$ROOT}/results/grouped-exact-next-prefetch-ablation-$TIMESTAMP"
GROUPED_ROOT=""
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
HOST="${ASTRAKV_HOST:-127.0.0.1}"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="24576"
GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.72}"
TIMEOUT="900"
LIMIT="0"
SERVER_PID=""
SIDECAR_MODE="build"
EXTERNAL_SIDECAR=""
DATASETS="qasper,multifieldqa_en"
WARMUP_PASSES="${ASTRAKV_ABLATION_WARMUP_PASSES:-0}"
WARMUP_LIMIT="${ASTRAKV_ABLATION_WARMUP_LIMIT:-8}"
EVICTION_FILL_GROUPS="${ASTRAKV_ABLATION_EVICTION_FILL_GROUPS:-0}"
MEASURE_PHASES="${ASTRAKV_ABLATION_MEASURE_PHASES:-}"
PREDICTION_PHASES="${ASTRAKV_ABLATION_PREDICTION_PHASES:-}"
INTERLEAVE="false"
INTERLEAVE_PATTERN="${ASTRAKV_INTERLEAVE_PATTERN:-round-robin}"
ROLES="${ASTRAKV_ABLATION_ROLES:-baseline,variant}"
PREFETCH_LEAD_S="${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-0.0}"

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
    --no-sidecar) SIDECAR_MODE="none"; shift ;;
    --sidecar-path) EXTERNAL_SIDECAR="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --interleave) INTERLEAVE="true"; shift ;;
    --interleave-pattern) INTERLEAVE_PATTERN="$2"; shift 2 ;;
    --roles) ROLES="$2"; shift 2 ;;
    --prefetch-lead-s) PREFETCH_LEAD_S="$2"; shift 2 ;;
    --eviction-fill-groups) EVICTION_FILL_GROUPS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Model path is missing: $MODEL" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "Only loopback hosts are allowed" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
[[ "$EVICTION_FILL_GROUPS" =~ ^[0-9]+$ ]] || { echo "--eviction-fill-groups must be a non-negative integer" >&2; exit 2; }
[[ -n "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
IFS=',' read -r -a SELECTED_DATASETS <<< "$DATASETS"
[[ "${#SELECTED_DATASETS[@]}" -gt 0 ]] || { echo "--datasets must not be empty" >&2; exit 2; }
IFS=',' read -r -a SELECTED_ROLES <<< "$ROLES"
for role in "${SELECTED_ROLES[@]}"; do
  case "$role" in
    baseline|variant) ;;
    *) echo "invalid --roles entry: $role (expected baseline|variant)" >&2; exit 2 ;;
  esac
done
for dataset in "${SELECTED_DATASETS[@]}"; do
  [[ -f "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" ]] || {
    echo "missing $GROUPED_ROOT/$dataset/grouped_prompts.jsonl" >&2; exit 2; }
done

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_for_endpoint() {
  local log_path="$1"
  for _ in $(seq 1 300); do
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
  # CPU==GPU on UMA: force the CPU hot cache on and size it equal to the GPU
  # KV tier (default 5.0 GiB standalone; the full pipeline sets it to 71 GiB).
  sed -e "s|^local_cpu:.*|local_cpu: true|" \
      -e "s|^max_local_cpu_size:.*|max_local_cpu_size: ${ASTRAKV_LOCAL_CPU_SIZE_GB:-5.0}|" \
      -e "s|^max_local_disk_size:.*|max_local_disk_size: ${ASTRAKV_LOCAL_DISK_SIZE_GB:-80.0}|" \
      -e "s|^local_disk:.*|local_disk: $cache_dir|" \
      configs/lmcache_disk_example.yaml > "$state_dir/lmcache.yaml"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  ASTRAKV_MODEL="$MODEL" \
  ASTRAKV_PYTHON="$PYTHON" \
  ASTRAKV_HOST="$HOST" \
  ASTRAKV_PORT="$PORT" \
  ASTRAKV_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  ASTRAKV_PREFIX_CACHING="${ASTRAKV_PREFIX_CACHING:-true}" \
  ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_LMCACHE047_EVENTS="$state_dir/hook_raw.jsonl" \
  ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" \
  ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" \
  ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
  ASTRAKV_REQUIRE_EXACT_TOKEN_IDS=true \
  LMCACHE_CONFIG_FILE="$state_dir/lmcache.yaml" \
  ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  setsid nohup bash scripts/launch/launch_lmcache_vllm.sh disk > "$log_path" 2>&1 < /dev/null &
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
ASTRAKV_ENABLE_ONLINE_POLICY=true
ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=true
EOF
  cat > "$state_dir/offline_gate.json" <<'EOF'
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["grouped_exact_next"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
  if [[ "$role" == "baseline" ]]; then
    printf 'ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH=false\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\n' \
      "$state_dir/offline_gate.json" >> "$runtime_env"
  else
    printf 'ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH=true\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\n' \
      "$state_dir/offline_gate.json" >> "$runtime_env"
    # B (predictive prefetch) also requires a prefetch mode; the bootstrap
    # default is "disabled", which gates out every prefetch decision even when
    # dispatch is enabled.  Honoring the environment lets the 2x2/transfer
    # wrappers set prefix_only/hybrid per cell.
    printf 'ASTRAKV_ONLINE_PREFETCH_MODE=%s\n' \
      "${ASTRAKV_ONLINE_PREFETCH_MODE:-hybrid}" >> "$runtime_env"
    if [[ -n "$sidecar_path" ]]; then
      printf 'ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH=%s\n' "$sidecar_path" >> "$runtime_env"
    fi
  fi
  chmod 600 "$runtime_env"
}

materialize_dataset() {
  local dataset="$1"
  local materialized_dir="$OUTPUT_DIR/$dataset/materialized"
  mkdir -p "$materialized_dir"
  local extra_args=()
  if [[ "$INTERLEAVE" == "true" ]]; then
    extra_args+=(--interleave --interleave-pattern "$INTERLEAVE_PATTERN")
    if [[ "$INTERLEAVE_PATTERN" == "fire-consume" && "$EVICTION_FILL_GROUPS" -gt 0 ]]; then
      extra_args+=(--eviction-fill-groups "$EVICTION_FILL_GROUPS")
    fi
  fi
  "$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
    --grouped-prompts-jsonl "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" \
    --output-dir "$materialized_dir" \
    --dataset "$dataset" \
    --task "$dataset" \
    --limit "$LIMIT" \
    --prefetch-lead-s "$PREFETCH_LEAD_S" \
    "${extra_args[@]}"

  if [[ -n "$MEASURE_PHASES" ]]; then
    local canonical="$materialized_dir/${dataset}_grouped_exact_next_canonical_workload.jsonl"
    local measured="$materialized_dir/${dataset}_grouped_exact_next_measured_workload.jsonl"
    local warmup="$materialized_dir/${dataset}_grouped_exact_next_warmup.jsonl"
    "$PYTHON" - "$canonical" "$measured" "$warmup" "$MEASURE_PHASES" <<'PY'
import json, sys
source, measured_path, warmup_path, phase_text = sys.argv[1:]
phases = {item.strip() for item in phase_text.split(",") if item.strip()}
measured, warmup = [], []
with open(source, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("prefetch_phase") or "") in phases:
            measured.append(row)
        else:
            warmup.append(row)
if not measured:
    raise SystemExit(f"no measured rows matched phases={sorted(phases)}")
if not warmup:
    raise SystemExit("phase-filtered workload has no warmup rows")
for path, rows in ((measured_path, measured), (warmup_path, warmup)):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"phase-filtered workload: measured={len(measured)} warmup={len(warmup)} phases={sorted(phases)}")
PY
  fi
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
  secret="$($PYTHON -c 'import secrets; print(secrets.token_hex(32))')"
  local sidecar_path=""
  if [[ "$role" == "variant" ]]; then
    if [[ -n "$EXTERNAL_SIDECAR" ]]; then
      sidecar_path="$EXTERNAL_SIDECAR"
    elif [[ "$SIDECAR_MODE" == "build" ]]; then
      local sidecar_candidate_report="$candidate_report"
      if [[ -n "$PREDICTION_PHASES" ]]; then
        sidecar_candidate_report="$sidecar_dir/predictor_candidate_report.phases.jsonl"
        "$PYTHON" - "$candidate_report" "$canonical" "$sidecar_candidate_report" "$PREDICTION_PHASES" <<'PY'
import json, sys
source, workload, target, phase_text = sys.argv[1:]
phases = {item.strip() for item in phase_text.split(",") if item.strip()}
allowed = set()
with open(workload, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("prefetch_phase") or "") in phases:
            allowed.add(str(row.get("request_id") or ""))
count = 0
with open(source, encoding="utf-8") as source_handle, open(target, "w", encoding="utf-8") as target_handle:
    for line in source_handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("request_id") or "") in allowed:
            target_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
print(f"phase-filtered predictor candidates: {count} rows, phases={sorted(phases)}")
PY
      fi
      "$PYTHON" scripts/reporting/build_sidecar_prediction.py \
        --candidate-report "$sidecar_candidate_report" \
        --output-dir "$sidecar_dir" \
        --run-id "$run_id" \
        --lead-time-ms 250 \
        --predicted-class exact-next
      sidecar_path="$sidecar_dir/sidecar_prediction.jsonl"
    fi
  fi
  write_runtime_env "$run_id" "$state_dir" "$runtime_env" "$secret" "$role" "$sidecar_path"
  start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  if [[ "$WARMUP_PASSES" =~ ^[0-9]+$ && "$WARMUP_PASSES" -gt 0 ]]; then
    local warmup_file="$OUTPUT_DIR/$dataset/materialized/${dataset}_grouped_exact_next_warmup.jsonl"
    if [[ ! -f "$warmup_file" ]]; then
      "$PYTHON" - "$canonical" "$warmup_file" "$WARMUP_LIMIT" <<'PY'
import json, sys
source, target, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(source, encoding="utf-8") as handle:
    for line in handle:
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
with open(target, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"warmup workload written to {target} ({len(rows)} rows)")
PY
    fi
    for pass_index in $(seq 1 "$WARMUP_PASSES"); do
      echo "[$dataset/$role] warmup pass $pass_index/$WARMUP_PASSES (same server/cache)"
      "$PYTHON" scripts/benchmark/run_real_benchmark.py \
        --base-url "http://${HOST}:${PORT}/v1" \
        --model "$MODEL" \
        --backend "vllm-lmcache047" \
        --output-dir "$run_dir/warmup-$pass_index" \
        --workload-jsonl "$warmup_file" \
        --run-id "${run_id}-warmup-${pass_index}" \
        --workload-id "${dataset}_grouped_exact_next_warmup" \
        --model-revision "local-qwen3-8b" \
        --tokenizer-revision "local-qwen3-8b" \
        --dtype "bfloat16" \
        --quantization "unquantized" \
        --random-seed "0" \
        --cache-state warm \
        --connector-version "lmcache-vllm-v1-0.4.7" \
        --pair-id "$pair_id" \
        --pair-role "$role" \
        --claim-scope online_control_warmup \
        --runtime-state-dir "$state_dir" \
        --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
        --request-context-session-id "${run_id}-session" \
        --request-context-secret-hex "$secret" \
        --metrics-interval 1.0 \
        --timeout "$TIMEOUT" \
        --output-tokens 128
    done
  fi
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" \
    --backend "vllm-lmcache-kv-core" \
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
    --claim-scope kv_core \
    --runtime-state-dir "$state_dir" \
    --tokenizer-path "$MODEL" \
    --chat-template-revision qwen3-default \
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

for dataset in "${SELECTED_DATASETS[@]}"; do
  materialize_dataset "$dataset"
  build_analysis_artifacts "$dataset"
  canonical="$OUTPUT_DIR/$dataset/materialized/${dataset}_grouped_exact_next_canonical_workload.jsonl"
  measured_canonical="$OUTPUT_DIR/$dataset/materialized/${dataset}_grouped_exact_next_measured_workload.jsonl"
  if [[ -f "$measured_canonical" ]]; then
    canonical="$measured_canonical"
  fi
  candidate_report="$OUTPUT_DIR/$dataset/analysis/candidates/predictor_candidate_report.jsonl"
  pair_id="grouped-exact-next-prefetch-${dataset}-$TIMESTAMP"
  for role in "${SELECTED_ROLES[@]}"; do
    run_condition "$dataset" "$role" "$canonical" "$pair_id" "$candidate_report"
  done
done

echo "Prefetch ablation suite completed: $OUTPUT_DIR"
