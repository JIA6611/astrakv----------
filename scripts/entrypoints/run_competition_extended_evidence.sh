#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ONLY="all"
OUTPUT_ROOT=""
EXISTING_E2E_ROOT=""
MODEL="${ASTRAKV_MODEL:-/home/szl/Desktop/Inference-OS/models/Qwen2.5-7B-Instruct}"
GPU_UTIL_OFFICIAL="0.60"
GPU_UTIL_BOUNDARY="0.25"
BOUNDARY_MAX_MODEL_LEN="32768"
BOUNDARY_CONTEXT_LENGTHS="16384 24576 32768"
BOUNDARY_BATCH_SIZES="1 2 4 8"
BOUNDARY_OUTPUT_TOKENS="128"
BOUNDARY_REPEAT="1"
BOUNDARY_TIMEOUT="1800"
CONTINUE_ON_FAILURE=0
SKIP_INSTALL=0
HOST="${ASTRAKV_HOST:-127.0.0.1}"
PORT="${ASTRAKV_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER_PID=""
WORKLOAD_JSONL=""
RUN_ID=""
RUNTIME_MODE="auto"
OFFLINE_WORKLOAD_SET=""
OFFLINE_OUTPUT_DIR=""
OFFLINE_SAFETY_GATE=""
OFFLINE_GPU_CAPACITY_BYTES=0
OFFLINE_CPU_CAPACITY_BYTES=0
OFFLINE_SSD_CAPACITY_BYTES=0
OFFLINE_DEFAULT_OBJECT_BYTES=0
STRUCTURED_EVENTS=""
STRUCTURED_HOOK_VERIFICATION=""
RUN_DIAGNOSTIC=0
DIAGNOSTIC_PID=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/entrypoints/run_competition_extended_evidence.sh [options]

Options:
  --only all|e2e|boundary|cache|vm|quality|policy|report|archive
  --output-root PATH
  --existing-e2e-root PATH
  --model PATH_OR_HF_ID
  --gpu-util-official FLOAT
  --gpu-util-boundary FLOAT
  --boundary-max-model-len INT
  --boundary-context-lengths "INT INT ..."
  --boundary-batch-sizes "INT INT ..."
  --boundary-output-tokens INT
  --boundary-repeat INT
  --boundary-timeout SECONDS
  --continue-on-failure
  --skip-install
  --workload-jsonl PATH
  --run-id ID
  --runtime-mode local|dgx|auto
  --offline-workload-set PATH --offline-output-dir PATH --offline-safety-gate PATH
  --offline-gpu-capacity-bytes INT --offline-cpu-capacity-bytes INT --offline-ssd-capacity-bytes INT --offline-default-object-bytes INT
  --structured-events PATH --structured-hook-verification PATH
  --run-diagnostic [--diagnostic-pid PID]
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --existing-e2e-root)
      EXISTING_E2E_ROOT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --gpu-util-official)
      GPU_UTIL_OFFICIAL="$2"
      shift 2
      ;;
    --gpu-util-boundary)
      GPU_UTIL_BOUNDARY="$2"
      shift 2
      ;;
    --boundary-max-model-len)
      BOUNDARY_MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --boundary-context-lengths)
      BOUNDARY_CONTEXT_LENGTHS="$2"
      shift 2
      ;;
    --boundary-batch-sizes)
      BOUNDARY_BATCH_SIZES="$2"
      shift 2
      ;;
    --boundary-output-tokens)
      BOUNDARY_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --boundary-repeat)
      BOUNDARY_REPEAT="$2"
      shift 2
      ;;
    --boundary-timeout)
      BOUNDARY_TIMEOUT="$2"
      shift 2
      ;;
    --continue-on-failure)
      CONTINUE_ON_FAILURE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --workload-jsonl)
      WORKLOAD_JSONL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --runtime-mode)
      RUNTIME_MODE="$2"
      shift 2
      ;;
    --offline-workload-set) OFFLINE_WORKLOAD_SET="$2"; shift 2 ;;
    --offline-output-dir) OFFLINE_OUTPUT_DIR="$2"; shift 2 ;;
    --offline-safety-gate) OFFLINE_SAFETY_GATE="$2"; shift 2 ;;
    --offline-gpu-capacity-bytes) OFFLINE_GPU_CAPACITY_BYTES="$2"; shift 2 ;;
    --offline-cpu-capacity-bytes) OFFLINE_CPU_CAPACITY_BYTES="$2"; shift 2 ;;
    --offline-ssd-capacity-bytes) OFFLINE_SSD_CAPACITY_BYTES="$2"; shift 2 ;;
    --offline-default-object-bytes) OFFLINE_DEFAULT_OBJECT_BYTES="$2"; shift 2 ;;
    --structured-events) STRUCTURED_EVENTS="$2"; shift 2 ;;
    --structured-hook-verification) STRUCTURED_HOOK_VERIFICATION="$2"; shift 2 ;;
    --run-diagnostic) RUN_DIAGNOSTIC=1; shift ;;
    --diagnostic-pid) DIAGNOSTIC_PID="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ONLY" in
  all|e2e|boundary|cache|vm|quality|policy|report|archive) ;;
  *)
    echo "--only must be one of all, e2e, boundary, cache, vm, quality, policy, report, archive" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/extended_evidence_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$RUN_ID" ]]; then RUN_ID="$(basename "$OUTPUT_ROOT")"; fi
case "$RUNTIME_MODE" in local|dgx|auto) ;; *) echo "--runtime-mode must be local, dgx, or auto" >&2; exit 2 ;; esac

mkdir -p "$OUTPUT_ROOT"
COMMAND_LOG="$OUTPUT_ROOT/commands.log"
STATE_DIR="$OUTPUT_ROOT/state"
mkdir -p "$STATE_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$COMMAND_LOG"
}

state_write() {
  local key="$1"
  local value="$2"
  printf '%s\n' "$value" > "$STATE_DIR/${key}"
}

state_read() {
  local key="$1"
  local path="$STATE_DIR/${key}"
  if [[ -f "$path" ]]; then
    cat "$path"
  fi
  return 0
}

state_from_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    cat "$path"
  fi
  return 0
}

run_cmd() {
  local name="$1"
  shift
  local log_file="$OUTPUT_ROOT/${name}.log"
  log "RUN ${name}: $*"
  set +e
  "$@" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2)
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAIL ${name}: exit=${status}"
    if [[ "$CONTINUE_ON_FAILURE" -eq 0 ]]; then
      exit "$status"
    fi
  else
    log "OK ${name}"
  fi
  return "$status"
}

run_cmd_allow_failure() {
  local name="$1"
  shift
  local old_continue="$CONTINUE_ON_FAILURE"
  CONTINUE_ON_FAILURE=1
  run_cmd "$name" "$@"
  local status=$?
  CONTINUE_ON_FAILURE="$old_continue"
  log "ALLOW_FAILURE ${name}: exit=${status}"
  return 0
}

ensure_python() {
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [[ -f configs/dgx_spark_env.sh ]]; then
    # shellcheck disable=SC1091
    source configs/dgx_spark_env.sh
  fi
  export ASTRAKV_PYTHON="${ASTRAKV_PYTHON:-python}"
  export ASTRAKV_MODEL="$MODEL"
  export PATH="$ROOT/.venv/bin:$PATH"
  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    run_cmd install python -m pip install --upgrade pip
    run_cmd install_requirements python -m pip install -r requirements.txt pytest tabulate
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    log "Stopping server pid=${SERVER_PID}"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
}

trap stop_server EXIT

wait_for_server() {
  local timeout_seconds="${1:-1200}"
  local deadline=$((SECONDS + timeout_seconds))
  local response_file="$STATE_DIR/models_response.json"
  while (( SECONDS < deadline )); do
    if curl -fsS "${BASE_URL}/v1/models" -o "$response_file" >/dev/null 2>&1; then
      if python - "$response_file" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    payload = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if payload.get("object") == "list" and "error" not in payload:
    raise SystemExit(0)
raise SystemExit(1)
PY
      then
        log "Server ready at ${BASE_URL}/v1/models"
        return 0
      fi
    fi
    sleep 5
  done
  log "Server readiness timed out after ${timeout_seconds}s"
  return 1
}

start_server() {
  local backend="$1"
  local gpu_util="$2"
  local max_model_len="$3"
  local log_file="$4"
  local lmcache_config="${5:-}"
  stop_server
  export ASTRAKV_MODEL="$MODEL"
  export ASTRAKV_HOST="$HOST"
  export ASTRAKV_PORT="$PORT"
  export ASTRAKV_GPU_MEMORY_UTILIZATION="$gpu_util"
  export ASTRAKV_MAX_MODEL_LEN="$max_model_len"
  if [[ -n "$lmcache_config" ]]; then
    export LMCACHE_CONFIG_FILE="$lmcache_config"
  else
    unset LMCACHE_CONFIG_FILE || true
  fi
  if [[ "$backend" == "disk" ]]; then
    export LMCACHE_LOCAL_DISK="$OUTPUT_ROOT/02_boundary_32k/lmcache_disk_store"
    export LMCACHE_DISK_PATH="$LMCACHE_LOCAL_DISK"
  fi
  log "Starting ${backend} server gpu_util=${gpu_util} max_model_len=${max_model_len} log=${log_file}"
  mkdir -p "$(dirname "$log_file")"
  if [[ "$backend" == "vllm" ]]; then
    bash scripts/launch/launch_vllm_server.sh > "$log_file" 2>&1 &
  else
    bash scripts/launch/launch_lmcache_vllm.sh "$backend" > "$log_file" 2>&1 &
  fi
  SERVER_PID=$!
  echo "$SERVER_PID" > "$STATE_DIR/${backend}_server.pid"
  wait_for_server 1200
}

capture_benchmark_dir() {
  local stdout_file="$1"
  local output_parent="$2"
  local captured
  captured="$(sed -n 's/^Benchmark outputs written to //p' "$stdout_file" | tail -n 1 || true)"
  if [[ -n "$captured" && -d "$captured" ]]; then
    echo "$captured"
    return
  fi
  find "$output_parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true
}

run_boundary_benchmark() {
  local name="$1"
  local config="$2"
  local output_parent="$3"
  local backend_label="$4"
  mkdir -p "$output_parent"
  local stdout_file="$OUTPUT_ROOT/${name}.stdout.log"
  local -a context_args batch_args
  local workload_args=()
  if [[ -n "$WORKLOAD_JSONL" ]]; then workload_args+=(--workload-jsonl "$WORKLOAD_JSONL" --run-id "$RUN_ID"); fi
  read -r -a context_args <<< "$BOUNDARY_CONTEXT_LENGTHS"
  read -r -a batch_args <<< "$BOUNDARY_BATCH_SIZES"
  log "RUN ${name}: run_real_benchmark.py ${config}"
  set +e
  python scripts/benchmark/run_real_benchmark.py \
    --config "$config" \
    --output-dir "$output_parent" \
    --context-lengths "${context_args[@]}" \
    --batch-sizes "${batch_args[@]}" \
    --output-tokens "$BOUNDARY_OUTPUT_TOKENS" \
    --repeat "$BOUNDARY_REPEAT" \
    --timeout "$BOUNDARY_TIMEOUT" \
    --model "$MODEL" \
    --backend "$backend_label" \
    "${workload_args[@]}" \
    --enable-samples \
    > >(tee "$stdout_file" | tee -a "$OUTPUT_ROOT/${name}.log") \
    2> >(tee -a "$OUTPUT_ROOT/${name}.log" >&2)
  local status=$?
  set -e
  local run_dir
  run_dir="$(capture_benchmark_dir "$stdout_file" "$output_parent")"
  state_write "${name}.dir" "$run_dir"
  if [[ "$status" -ne 0 ]]; then
    log "FAIL ${name}: exit=${status} run_dir=${run_dir}"
  else
    log "OK ${name}: run_dir=${run_dir}"
  fi
  return 0
}

run_e2e() {
  log "Starting extended e2e stage"
  if [[ -n "$EXISTING_E2E_ROOT" ]]; then
    if [[ ! -d "$EXISTING_E2E_ROOT" ]]; then
      log "Existing E2E root not found: $EXISTING_E2E_ROOT"
      if [[ "$CONTINUE_ON_FAILURE" -eq 0 ]]; then
        exit 1
      fi
    fi
    state_write e2e_root.path "$EXISTING_E2E_ROOT"
    log "Using existing E2E root: $EXISTING_E2E_ROOT"
    return 0
  fi

  local args=(scripts/entrypoints/run_competition_e2e.sh --output-root "$OUTPUT_ROOT/01_e2e" --model "$MODEL" --gpu-util-official "$GPU_UTIL_OFFICIAL" --gpu-util-extreme "$GPU_UTIL_BOUNDARY")
  if [[ -n "$WORKLOAD_JSONL" ]]; then args+=(--workload-jsonl "$WORKLOAD_JSONL" --run-id "$RUN_ID" --runtime-mode "$RUNTIME_MODE"); fi
  if [[ "$SKIP_INSTALL" -eq 1 ]]; then args+=(--skip-install); fi
  if [[ "$CONTINUE_ON_FAILURE" -eq 1 ]]; then args+=(--continue-on-failure); fi
  run_cmd e2e bash "${args[@]}"
  state_write e2e_root.path "$OUTPUT_ROOT/01_e2e"
}

run_boundary() {
  log "Starting boundary stage gpu_util=${GPU_UTIL_BOUNDARY} max_model_len=${BOUNDARY_MAX_MODEL_LEN}"
  mkdir -p "$OUTPUT_ROOT/02_boundary_32k"

  local vllm_log="$OUTPUT_ROOT/02_boundary_32k/vllm_server.log"
  state_write boundary_vllm_server.log "$vllm_log"
  if start_server vllm "$GPU_UTIL_BOUNDARY" "$BOUNDARY_MAX_MODEL_LEN" "$vllm_log"; then
    run_boundary_benchmark boundary_vllm configs/stress_vllm_extreme_memory_constrained.yaml "$OUTPUT_ROOT/02_boundary_32k/vllm" vllm_boundary_32k
  else
    log "Boundary vLLM server failed readiness; server log retained"
  fi
  stop_server

  local cpu_log="$OUTPUT_ROOT/02_boundary_32k/lmcache_cpu_server.log"
  state_write boundary_lmcache_cpu_server.log "$cpu_log"
  if start_server cpu "$GPU_UTIL_BOUNDARY" "$BOUNDARY_MAX_MODEL_LEN" "$cpu_log" configs/lmcache_cpu_constrained.yaml; then
    run_boundary_benchmark boundary_lmcache_cpu configs/stress_lmcache_cpu_extreme_memory_constrained.yaml "$OUTPUT_ROOT/02_boundary_32k/lmcache_cpu" lmcache_cpu_boundary_32k
  else
    log "Boundary LMCache CPU server failed readiness; server log retained"
  fi
  stop_server

  local disk_log="$OUTPUT_ROOT/02_boundary_32k/lmcache_disk_server.log"
  state_write boundary_lmcache_disk_server.log "$disk_log"
  if start_server disk "$GPU_UTIL_BOUNDARY" "$BOUNDARY_MAX_MODEL_LEN" "$disk_log" configs/lmcache_disk_constrained.yaml; then
    run_boundary_benchmark boundary_lmcache_disk configs/stress_lmcache_disk_extreme_memory_constrained.yaml "$OUTPUT_ROOT/02_boundary_32k/lmcache_disk" lmcache_disk_boundary_32k
  else
    log "Boundary LMCache Disk server failed readiness; server log retained"
  fi
  stop_server

  local stress_args=()
  for item in \
    "vllm=$(state_read boundary_vllm.dir)" \
    "lmcache_cpu=$(state_read boundary_lmcache_cpu.dir)" \
    "lmcache_disk=$(state_read boundary_lmcache_disk.dir)"; do
    local path="${item#*=}"
    if [[ -n "$path" && -f "$path/benchmark_results.csv" ]]; then
      stress_args+=(--run "$item")
    fi
  done
  if [[ "${#stress_args[@]}" -gt 0 ]]; then
    run_cmd_allow_failure boundary_stress_analysis python scripts/reporting/analyze_stress_results.py \
      "${stress_args[@]}" \
      --output-dir "$OUTPUT_ROOT/02_boundary_32k/stress_analysis"
  else
    log "Skipping boundary stress analysis: no benchmark_results.csv found"
  fi
}

extract_events_if_possible() {
  local label="$1"
  local run_dir="$2"
  local server_log="$3"
  local out_dir="$4"
  if [[ -z "$server_log" || ! -f "$server_log" ]]; then
    log "Skipping cache events for ${label}: missing server log"
    return 0
  fi
  local args=(scripts/benchmark/extract_cache_events.py --server-log "$server_log" --output-dir "$out_dir")
  if [[ -n "$run_dir" && -f "$run_dir/benchmark_results.csv" ]]; then
    args+=(--benchmark-results "$run_dir/benchmark_results.csv")
  fi
  if [[ -n "$run_dir" && -f "$run_dir/request_results.jsonl" ]]; then
    args+=(--request-results "$run_dir/request_results.jsonl")
  fi
  run_cmd_allow_failure "cache_${label}" python "${args[@]}"
  if [[ -f "$out_dir/cache_events.jsonl" ]]; then
    state_write "cache_events_${label}.path" "$out_dir/cache_events.jsonl"
  fi
}

run_cache() {
  log "Starting cache event stage"
  mkdir -p "$OUTPUT_ROOT/03_cache_events"
  extract_events_if_possible boundary_lmcache_cpu \
    "$(state_read boundary_lmcache_cpu.dir)" \
    "$(state_read boundary_lmcache_cpu_server.log)" \
    "$OUTPUT_ROOT/03_cache_events/lmcache_cpu_boundary"
  extract_events_if_possible boundary_lmcache_disk \
    "$(state_read boundary_lmcache_disk.dir)" \
    "$(state_read boundary_lmcache_disk_server.log)" \
    "$OUTPUT_ROOT/03_cache_events/lmcache_disk_boundary"

  local e2e_root
  e2e_root="$(state_read e2e_root.path)"
  if [[ -n "$e2e_root" && -d "$e2e_root" ]]; then
    extract_events_if_possible e2e_lmcache_disk \
      "$(state_from_file "$e2e_root/state/step4_lmcache_disk.dir")" \
      "$e2e_root/step4_lmcache_disk_server.log" \
      "$OUTPUT_ROOT/03_cache_events/lmcache_disk_e2e"
    if [[ -f "$e2e_root/step4_lmcache_disk_server.log" ]]; then
      run_cmd_allow_failure cache_e2e_prefetch python scripts/benchmark/extract_cache_events.py \
        --server-log "$e2e_root/step4_lmcache_disk_server.log" \
        --output-dir "$OUTPUT_ROOT/03_cache_events/prefetch_e2e"
    fi
  fi
}

run_vm() {
  log "Starting OS VM evidence stage"
  mkdir -p "$OUTPUT_ROOT/04_os_vm"
  run_cmd_allow_failure vm_dgx_spark python scripts/vm/run_dgx_spark_vm_evidence.py \
    --output-dir "$OUTPUT_ROOT/04_os_vm/dgx_spark_vm" \
    --chunks 16 \
    --total-blocks 128 \
    --block-size-mb 1
  run_cmd_allow_failure vm_mmap_kv python scripts/vm/run_mmap_kv_cache.py \
    --output-dir "$OUTPUT_ROOT/04_os_vm/mmap_kv_cache" \
    --blocks 256 \
    --block-size-mb 1
  run_cmd_allow_failure vm_cli_mmap python cli.py vm mmap \
    --blocks 128 \
    --block-size-mb 1 \
    --output-dir "$OUTPUT_ROOT/04_os_vm/cli_mmap"
}

first_existing_run_pair() {
  local vllm_run=""
  local disk_run=""
  vllm_run="$(state_read boundary_vllm.dir)"
  disk_run="$(state_read boundary_lmcache_disk.dir)"
  if [[ -n "$vllm_run" && -f "$vllm_run/request_results.jsonl" && -n "$disk_run" && -f "$disk_run/request_results.jsonl" ]]; then
    printf '%s\n%s\n' "$vllm_run" "$disk_run"
    return 0
  fi
  local e2e_root
  e2e_root="$(state_read e2e_root.path)"
  if [[ -n "$e2e_root" && -d "$e2e_root" ]]; then
    vllm_run="$(state_from_file "$e2e_root/state/step3_vllm.dir")"
    disk_run="$(state_from_file "$e2e_root/state/step4_lmcache_disk.dir")"
    if [[ -n "$vllm_run" && -f "$vllm_run/request_results.jsonl" && -n "$disk_run" && -f "$disk_run/request_results.jsonl" ]]; then
      printf '%s\n%s\n' "$vllm_run" "$disk_run"
      return 0
    fi
  fi
  return 1
}

run_quality() {
  log "Starting quality consistency stage"
  mkdir -p "$OUTPUT_ROOT/05_quality"
  local pair vllm_run disk_run
  if ! pair="$(first_existing_run_pair)"; then
    log "Skipping quality: missing vLLM/LMCache Disk request_results.jsonl pair"
    return 0
  fi
  vllm_run="$(printf '%s\n' "$pair" | sed -n '1p')"
  disk_run="$(printf '%s\n' "$pair" | sed -n '2p')"
  run_cmd_allow_failure quality_lmcache_disk_vs_vllm python scripts/research/evaluate_quality.py \
    --baseline-jsonl "$vllm_run/request_results.jsonl" \
    --variant-jsonl "$disk_run/request_results.jsonl" \
    --output-dir "$OUTPUT_ROOT/05_quality/lmcache_disk_vs_vllm"
  if [[ -f "$OUTPUT_ROOT/05_quality/lmcache_disk_vs_vllm/quality_results.csv" ]]; then
    state_write quality_lmcache_disk.path "$OUTPUT_ROOT/05_quality/lmcache_disk_vs_vllm/quality_results.csv"
  fi
}

run_policy() {
  log "Starting policy chain stage"
  mkdir -p "$OUTPUT_ROOT/06_policy_chain"
  local trace_args=()
  local e2e_root
  e2e_root="$(state_read e2e_root.path)"
  for cache_path in \
    "$OUTPUT_ROOT/03_cache_events/lmcache_cpu_boundary/cache_events.jsonl" \
    "$OUTPUT_ROOT/03_cache_events/lmcache_disk_boundary/cache_events.jsonl" \
    "$OUTPUT_ROOT/03_cache_events/lmcache_disk_e2e/cache_events.jsonl" \
    "$OUTPUT_ROOT/03_cache_events/prefetch_e2e/cache_events.jsonl"; do
    if [[ -f "$cache_path" ]]; then
      trace_args+=(--cache-events "$cache_path")
    fi
  done
  if [[ -n "$e2e_root" && -f "$e2e_root/step5_prefetch/prefetch_events.jsonl" ]]; then
    trace_args+=(--prefetch-events "$e2e_root/step5_prefetch/prefetch_events.jsonl")
  fi
  for sample_dir in \
    "$(state_read boundary_vllm.dir)/samples" \
    "$(state_read boundary_lmcache_cpu.dir)/samples" \
    "$(state_read boundary_lmcache_disk.dir)/samples"; do
    if [[ -d "$sample_dir" ]]; then
      trace_args+=(--samples "$sample_dir")
    fi
  done
  if [[ -n "$e2e_root" && -d "$e2e_root" ]]; then
    for state_file in step3_vllm.dir step4_lmcache_cpu.dir step4_lmcache_disk.dir; do
      local run_dir
      run_dir="$(state_from_file "$e2e_root/state/$state_file")"
      if [[ -d "$run_dir/samples" ]]; then
        trace_args+=(--samples "$run_dir/samples")
      fi
    done
  fi

  if [[ "${#trace_args[@]}" -eq 0 ]]; then
    log "Skipping policy chain: no trace inputs found"
    return 0
  fi

  if [[ -n "$WORKLOAD_JSONL" ]]; then trace_args+=(--workload-manifest "$WORKLOAD_JSONL" --run-id "$RUN_ID"); fi
  run_cmd_allow_failure policy_trace_store python scripts/policy/build_trace_store.py \
    "${trace_args[@]}" \
    --output-dir "$OUTPUT_ROOT/06_policy_chain/trace_store"
  local trace_events="$OUTPUT_ROOT/06_policy_chain/trace_store/trace_events.jsonl"
  if [[ ! -f "$trace_events" ]]; then
    log "Skipping ProfileDB: missing trace_events.jsonl"
    return 0
  fi

  run_cmd_allow_failure policy_profile_db python scripts/policy/build_profile_db.py \
    --trace-events "$trace_events" \
    --workload-id extended_evidence \
    --output-dir "$OUTPUT_ROOT/06_policy_chain/profile_db"
  local profile_db="$OUTPUT_ROOT/06_policy_chain/profile_db/profile_db.json"
  if [[ ! -f "$profile_db" ]]; then
    log "Skipping chunk score chain: missing profile_db.json"
    return 0
  fi
  state_write profile_db.path "$profile_db"

  run_cmd_allow_failure policy_chunk_scores python scripts/policy/score_chunks.py \
    --profile-db "$profile_db" \
    --output-dir "$OUTPUT_ROOT/06_policy_chain/chunk_scores"
  local chunk_scores="$OUTPUT_ROOT/06_policy_chain/chunk_scores/chunk_scores.csv"
  if [[ -f "$chunk_scores" ]]; then
    state_write chunk_scores.path "$chunk_scores"
  fi

  run_cmd_allow_failure policy_load_recompute python scripts/policy/decide_load_vs_recompute.py \
    --profile-db "$profile_db" \
    --output-dir "$OUTPUT_ROOT/06_policy_chain/load_recompute"
  local load_decisions="$OUTPUT_ROOT/06_policy_chain/load_recompute/load_recompute_decisions.csv"

  local object_args=(scripts/policy/run_unified_object_scheduler.py --profile-db "$profile_db" --output-dir "$OUTPUT_ROOT/06_policy_chain/object_scheduler")
  if [[ -f "$chunk_scores" ]]; then
    object_args+=(--chunk-scores "$chunk_scores")
  fi
  if [[ -f "$load_decisions" ]]; then
    object_args+=(--load-recompute-decisions "$load_decisions")
  fi
  run_cmd_allow_failure policy_object_scheduler python "${object_args[@]}"

  local schedule_path="$OUTPUT_ROOT/06_policy_chain/object_scheduler/object_schedule_decisions.csv"
  local disk_cache="$OUTPUT_ROOT/03_cache_events/lmcache_disk_boundary/cache_events.jsonl"
  local offline_dir="${OFFLINE_OUTPUT_DIR:-$OUTPUT_ROOT/08_offline_eviction}"
  local effective_gate="$OFFLINE_SAFETY_GATE"
  local current_trace="$trace_events"
  local current_results="$(state_read boundary_lmcache_disk.dir)"
  if [[ -n "$OFFLINE_WORKLOAD_SET" && -f "$OFFLINE_WORKLOAD_SET" && -n "$WORKLOAD_JSONL" && -f "$current_trace" && -f "$profile_db" && "$OFFLINE_GPU_CAPACITY_BYTES" -gt 0 && "$OFFLINE_CPU_CAPACITY_BYTES" -gt 0 && "$OFFLINE_SSD_CAPACITY_BYTES" -gt 0 && "$OFFLINE_DEFAULT_OBJECT_BYTES" -gt 0 ]]; then
    local current_entry="$offline_dir/current_workload.json"
    mkdir -p "$offline_dir"
    python - "$current_entry" "$RUN_ID" "$WORKLOAD_JSONL" "$current_trace" "$profile_db" "$schedule_path" "$current_results" "$OFFLINE_GPU_CAPACITY_BYTES" "$OFFLINE_CPU_CAPACITY_BYTES" "$OFFLINE_SSD_CAPACITY_BYTES" "$OFFLINE_DEFAULT_OBJECT_BYTES" <<'PY'
import json, sys
from pathlib import Path
path, run_id, workload, trace, profile, schedule, result_dir, gpu, cpu, ssd, obj = sys.argv[1:]
json.dump({"workload_id": f"current-{run_id}", "workload_manifest": workload, "trace": trace, "profile_db": profile,
           "scheduler_decisions": schedule if Path(schedule).exists() else "",
           "request_results": str(Path(result_dir) / "request_results.jsonl") if result_dir else "",
           "gpu_capacity_bytes": int(gpu), "cpu_capacity_bytes": int(cpu), "ssd_capacity_bytes": int(ssd),
           "default_object_bytes": int(obj), "profile_source": "separate_profiling_run"}, open(path, "w", encoding="utf-8"), indent=2)
PY
    run_cmd_allow_failure offline_eviction_pipeline python scripts/policy/run_offline_eviction_pipeline.py --workload-set "$OFFLINE_WORKLOAD_SET" --append-current "$current_entry" --run-id "$RUN_ID" --output-dir "$offline_dir"
    [[ -z "$effective_gate" ]] && effective_gate="$offline_dir/offline_safety_gate.json"
  else
    mkdir -p "$offline_dir"
    python - "$offline_dir/offline_pipeline_status.json" <<'PY'
import json, sys
json.dump({"schema":"astrakv-offline-pipeline-status-v1","status":"not_configured","reason":"offline workload set, trace/profile, and explicit tier capacities are required"}, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
PY
  fi
  if [[ -n "$WORKLOAD_JSONL" && -f "$schedule_path" ]]; then
    local runtime_args=(bash scripts/entrypoints/run_runtime_eviction_validation.sh --mode "$RUNTIME_MODE" --workload-jsonl "$WORKLOAD_JSONL" --run-id "$RUN_ID" --offline-decisions "$schedule_path" --output-dir "$OUTPUT_ROOT/08_runtime_eviction")
    if [[ -f "$disk_cache" ]]; then runtime_args+=(--cache-events "$disk_cache"); fi
    local disk_run="$(state_read boundary_lmcache_disk.dir)"
    if [[ -n "$disk_run" && -f "$disk_run/request_results.jsonl" ]]; then runtime_args+=(--request-results "$disk_run/request_results.jsonl"); fi
    if [[ -f "$OUTPUT_ROOT/02_boundary_32k/lmcache_disk_server.log" ]]; then runtime_args+=(--server-log "$OUTPUT_ROOT/02_boundary_32k/lmcache_disk_server.log"); fi
    if [[ -n "$effective_gate" && -f "$effective_gate" ]]; then runtime_args+=(--offline-safety-gate "$effective_gate"); fi
    if [[ -n "$STRUCTURED_EVENTS" ]]; then runtime_args+=(--structured-events "$STRUCTURED_EVENTS"); fi
    if [[ -n "$STRUCTURED_HOOK_VERIFICATION" ]]; then runtime_args+=(--structured-hook-verification "$STRUCTURED_HOOK_VERIFICATION"); fi
    run_cmd_allow_failure runtime_eviction "${runtime_args[@]}"
  fi
  if [[ "$RUN_DIAGNOSTIC" -eq 1 ]]; then
    run_cmd_allow_failure runtime_diagnostic python scripts/benchmark/diagnose_runtime.py --pid "$DIAGNOSTIC_PID" --collect-tools --output-dir "$OUTPUT_ROOT/runtime_diagnostic"
  fi
}

run_report() {
  log "Starting extended final report stage"
  mkdir -p "$OUTPUT_ROOT/07_final_report"
  local e2e_root vllm_run cpu_run disk_run
  e2e_root="$(state_read e2e_root.path)"
  vllm_run="$(state_read boundary_vllm.dir)"
  cpu_run="$(state_read boundary_lmcache_cpu.dir)"
  disk_run="$(state_read boundary_lmcache_disk.dir)"
  if [[ -z "$vllm_run" && -n "$e2e_root" ]]; then vllm_run="$(state_from_file "$e2e_root/state/step3_vllm.dir")"; fi
  if [[ -z "$cpu_run" && -n "$e2e_root" ]]; then cpu_run="$(state_from_file "$e2e_root/state/step4_lmcache_cpu.dir")"; fi
  if [[ -z "$disk_run" && -n "$e2e_root" ]]; then disk_run="$(state_from_file "$e2e_root/state/step4_lmcache_disk.dir")"; fi

  local report_args=(scripts/reporting/build_competition_report.py --output-dir "$OUTPUT_ROOT/07_final_report" --title "AstraKV-W Extended Evidence Report")
  report_args+=(--command "bash scripts/entrypoints/run_competition_extended_evidence.sh --only ${ONLY} --output-root ${OUTPUT_ROOT} --model ${MODEL}")

  if [[ -n "$vllm_run" && -f "$vllm_run/benchmark_results.csv" ]]; then report_args+=(--benchmark "vllm=${vllm_run}"); fi
  if [[ -n "$cpu_run" && -f "$cpu_run/benchmark_results.csv" ]]; then report_args+=(--benchmark "lmcache_cpu=${cpu_run}"); fi
  if [[ -n "$disk_run" && -f "$disk_run/benchmark_results.csv" ]]; then report_args+=(--benchmark "lmcache_disk=${disk_run}"); fi
  if [[ -f "$OUTPUT_ROOT/02_boundary_32k/stress_analysis/stress_summary.csv" ]]; then report_args+=(--stress "$OUTPUT_ROOT/02_boundary_32k/stress_analysis/stress_summary.csv"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step6_stress_analysis/stress_summary.csv" ]]; then report_args+=(--artifact "e2e_official_stress:stress=$e2e_root/step6_stress_analysis/stress_summary.csv"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/extreme_stress_analysis/stress_summary.csv" ]]; then report_args+=(--artifact "e2e_extreme_stress:stress=$e2e_root/extreme_stress_analysis/stress_summary.csv"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step7_comparison/comparison_results.csv" ]]; then report_args+=(--comparison "$e2e_root/step7_comparison/comparison_results.csv"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step7_policy_ablation/policy_ablation_results.csv" ]]; then report_args+=(--policy-ablation "$e2e_root/step7_policy_ablation/policy_ablation_results.csv"); fi
  if [[ -f "$OUTPUT_ROOT/03_cache_events/lmcache_disk_boundary/cache_events.jsonl" ]]; then report_args+=(--cache-events "boundary_disk=$OUTPUT_ROOT/03_cache_events/lmcache_disk_boundary/cache_events.jsonl"); fi
  if [[ -f "$OUTPUT_ROOT/03_cache_events/lmcache_cpu_boundary/cache_events.jsonl" ]]; then report_args+=(--cache-events "boundary_cpu=$OUTPUT_ROOT/03_cache_events/lmcache_cpu_boundary/cache_events.jsonl"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/cache_events/step4_lmcache_disk/cache_events.jsonl" ]]; then report_args+=(--cache-events "e2e_lmcache_disk=$e2e_root/cache_events/step4_lmcache_disk/cache_events.jsonl"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/cache_events/step5_prefetch/cache_events.jsonl" ]]; then report_args+=(--cache-events "e2e_prefetch=$e2e_root/cache_events/step5_prefetch/cache_events.jsonl"); fi
  if [[ -f "$OUTPUT_ROOT/04_os_vm/dgx_spark_vm/dgx_spark_vm_evidence_summary.json" ]]; then report_args+=(--vm-evidence "dgx_vm=$OUTPUT_ROOT/04_os_vm/dgx_spark_vm/dgx_spark_vm_evidence_summary.json"); fi
  if [[ -f "$OUTPUT_ROOT/04_os_vm/mmap_kv_cache/mmap_kv_demo_summary.json" ]]; then report_args+=(--vm-evidence "mmap_kv=$OUTPUT_ROOT/04_os_vm/mmap_kv_cache/mmap_kv_demo_summary.json"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step2_vm_evidence/dgx_spark_vm_evidence_summary.json" ]]; then report_args+=(--vm-evidence "e2e_dgx_vm=$e2e_root/step2_vm_evidence/dgx_spark_vm_evidence_summary.json"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step2_mmap_smoke/mmap_kv_demo_summary.json" ]]; then report_args+=(--vm-evidence "e2e_mmap_smoke=$e2e_root/step2_mmap_smoke/mmap_kv_demo_summary.json"); fi
  if [[ -f "$OUTPUT_ROOT/05_quality/lmcache_disk_vs_vllm/quality_results.csv" ]]; then report_args+=(--quality "$OUTPUT_ROOT/05_quality/lmcache_disk_vs_vllm/quality_results.csv"); fi
  if [[ -f "$OUTPUT_ROOT/06_policy_chain/trace_store/trace_summary.md" ]]; then report_args+=(--trace-summary "$OUTPUT_ROOT/06_policy_chain/trace_store/trace_summary.md"); fi
  if [[ -f "$OUTPUT_ROOT/06_policy_chain/profile_db/profile_db.json" ]]; then report_args+=(--profile-db "$OUTPUT_ROOT/06_policy_chain/profile_db/profile_db.json"); fi
  if [[ -f "$OUTPUT_ROOT/06_policy_chain/chunk_scores/chunk_scores.csv" ]]; then report_args+=(--chunk-scores "$OUTPUT_ROOT/06_policy_chain/chunk_scores/chunk_scores.csv"); fi
  if [[ ! -f "$OUTPUT_ROOT/06_policy_chain/trace_store/trace_summary.md" && -n "$e2e_root" && -f "$e2e_root/step7_trace_store/trace_summary.md" ]]; then report_args+=(--trace-summary "$e2e_root/step7_trace_store/trace_summary.md"); fi
  if [[ ! -f "$OUTPUT_ROOT/06_policy_chain/profile_db/profile_db.json" && -n "$e2e_root" && -f "$e2e_root/step7_profile_db/profile_db.json" ]]; then report_args+=(--profile-db "$e2e_root/step7_profile_db/profile_db.json"); fi
  if [[ ! -f "$OUTPUT_ROOT/06_policy_chain/chunk_scores/chunk_scores.csv" && -n "$e2e_root" && -f "$e2e_root/step7_chunk_scores/chunk_scores.csv" ]]; then report_args+=(--chunk-scores "$e2e_root/step7_chunk_scores/chunk_scores.csv"); fi
  if [[ -f "$OUTPUT_ROOT/06_policy_chain/load_recompute/load_recompute_decisions.csv" ]]; then report_args+=(--load-recompute "$OUTPUT_ROOT/06_policy_chain/load_recompute/load_recompute_decisions.csv"); fi
  if [[ -f "$OUTPUT_ROOT/06_policy_chain/object_scheduler/object_schedule_decisions.csv" ]]; then report_args+=(--object-schedule "$OUTPUT_ROOT/06_policy_chain/object_scheduler/object_schedule_decisions.csv"); fi
  if [[ -n "$e2e_root" && -f "$e2e_root/step5_prefetch/prefetch_results.csv" ]]; then report_args+=(--prefetch "e2e_selective_prefetch=$e2e_root/step5_prefetch/prefetch_results.csv"); fi
  if [[ -f "$OUTPUT_ROOT/08_runtime_eviction/runtime_agreement/eviction_agreement_manifest.json" ]]; then report_args+=(--eviction-agreement "runtime=$OUTPUT_ROOT/08_runtime_eviction/runtime_agreement/eviction_agreement_manifest.json"); fi
  if [[ -f "$OUTPUT_ROOT/08_runtime_eviction/vm_poc_agreement/eviction_agreement_manifest.json" ]]; then report_args+=(--eviction-agreement "vm_poc=$OUTPUT_ROOT/08_runtime_eviction/vm_poc_agreement/eviction_agreement_manifest.json"); fi
  local report_offline_dir="${OFFLINE_OUTPUT_DIR:-$OUTPUT_ROOT/08_offline_eviction}"
  if [[ -f "$report_offline_dir/offline_safety_gate.json" ]]; then report_args+=(--offline-safety-gate "offline_gate=$report_offline_dir/offline_safety_gate.json"); fi
  if [[ -f "$report_offline_dir/offline_pipeline_status.json" ]]; then report_args+=(--artifact "offline_pipeline:artifact=$report_offline_dir/offline_pipeline_status.json"); fi
  if [[ -f "$OUTPUT_ROOT/runtime_diagnostic/diagnostic_manifest.json" ]]; then report_args+=(--diagnostic "runtime=$OUTPUT_ROOT/runtime_diagnostic/diagnostic_manifest.json"); fi
  while IFS= read -r experiment_manifest; do report_args+=(--experiment-manifest "$(basename "$(dirname "$experiment_manifest")")=$experiment_manifest"); done < <(find "$OUTPUT_ROOT" -name experiment_manifest.json -type f)
  shopt -s nullglob
  for policy_summary in "$report_offline_dir"/workloads/*/offline_eviction_policy_summary.csv; do report_args+=(--offline-policy-summary "$(basename "$(dirname "$policy_summary")")=$policy_summary"); done
  shopt -u nullglob

  shopt -s nullglob
  for log_file in "$OUTPUT_ROOT"/02_boundary_32k/*server.log; do
    report_args+=(--server-log "$(basename "$log_file" .log)=$log_file")
  done
  if [[ -n "$e2e_root" && -d "$e2e_root" ]]; then
    for log_file in "$e2e_root"/*server.log; do
      report_args+=(--server-log "e2e_$(basename "$log_file" .log)=$log_file")
    done
  fi
  shopt -u nullglob
  run_cmd_allow_failure extended_report python "${report_args[@]}"
}

run_archive() {
  log "Starting archive stage"
  mkdir -p "$OUTPUT_ROOT/archive"
  local manifest="$OUTPUT_ROOT/archive/artifact_manifest.txt"
  local inventory="$OUTPUT_ROOT/archive/artifact_inventory.csv"
  find "$OUTPUT_ROOT" -type f \
    ! -path "$OUTPUT_ROOT/archive/*" \
    ! -name "*.bin" \
    ! -path "*/lmcache_disk_store/*" \
    | sort > "$manifest"
  {
    echo "path,size_bytes"
    while IFS= read -r file; do
      printf '%s,%s\n' "$file" "$(stat -c '%s' "$file" 2>/dev/null || echo 0)"
    done < "$manifest"
  } > "$inventory"

  local archive_name
  archive_name="$(basename "$OUTPUT_ROOT").tar.gz"
  local parent base
  parent="$(dirname "$OUTPUT_ROOT")"
  base="$(basename "$OUTPUT_ROOT")"
  tar -czf "$OUTPUT_ROOT/archive/$archive_name" \
    --exclude='*.bin' \
    --exclude='*lmcache_disk_store*' \
    --exclude="$base/archive" \
    -C "$parent" "$base"
  log "Archive written to $OUTPUT_ROOT/archive/$archive_name"
}

main() {
  log "Output root: $OUTPUT_ROOT"
  log "Model: $MODEL"
  log "Boundary: gpu_util=${GPU_UTIL_BOUNDARY} max_model_len=${BOUNDARY_MAX_MODEL_LEN} contexts='${BOUNDARY_CONTEXT_LENGTHS}' batches='${BOUNDARY_BATCH_SIZES}' output=${BOUNDARY_OUTPUT_TOKENS} repeat=${BOUNDARY_REPEAT}"
  ensure_python
  if [[ -n "$EXISTING_E2E_ROOT" ]]; then
    state_write e2e_root.path "$EXISTING_E2E_ROOT"
  fi
  case "$ONLY" in
    e2e)
      run_e2e
      ;;
    boundary)
      run_boundary
      ;;
    cache)
      run_cache
      ;;
    vm)
      run_vm
      ;;
    quality)
      run_quality
      ;;
    policy)
      run_policy
      ;;
    report)
      run_report
      ;;
    archive)
      run_archive
      ;;
    all)
      run_e2e
      run_boundary
      run_cache
      run_vm
      run_quality
      run_policy
      run_report
      run_archive
      ;;
  esac
  log "Extended evidence artifacts written to $OUTPUT_ROOT"
}

main "$@"
