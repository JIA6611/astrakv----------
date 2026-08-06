#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ONLY="all"
OUTPUT_ROOT=""
MODEL="${ASTRAKV_MODEL:-/home/szl/Desktop/Inference-OS/models/Qwen2.5-7B-Instruct}"
GPU_UTIL_OFFICIAL="0.60"
GPU_UTIL_EXTREME="0.20"
WITH_CGROUP=0
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
  bash scripts/entrypoints/run_competition_e2e.sh [options]

Options:
  --only smoke|official|extreme|report|all
  --output-root PATH
  --model PATH_OR_HF_ID
  --gpu-util-official FLOAT
  --gpu-util-extreme FLOAT
  --with-cgroup
  --continue-on-failure
  --skip-install
  --workload-jsonl PATH
  --run-id ID
  --runtime-mode local|dgx|auto
  --offline-workload-set PATH
  --offline-output-dir PATH
  --offline-safety-gate PATH
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
    --model)
      MODEL="$2"
      shift 2
      ;;
    --gpu-util-official)
      GPU_UTIL_OFFICIAL="$2"
      shift 2
      ;;
    --gpu-util-extreme)
      GPU_UTIL_EXTREME="$2"
      shift 2
      ;;
    --with-cgroup)
      WITH_CGROUP=1
      shift
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
  smoke|official|extreme|report|all) ;;
  *)
    echo "--only must be one of smoke, official, extreme, report, all" >&2
    exit 2
    ;;
esac

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/competition_e2e_$(date +%Y%m%d_%H%M%S)"
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
  local timeout_seconds="${1:-900}"
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
  find "$output_parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_benchmark_capture() {
  local name="$1"
  local config="$2"
  local output_parent="$3"
  shift 3
  mkdir -p "$output_parent"
  local stdout_file="$OUTPUT_ROOT/${name}.stdout.log"
  log "RUN ${name}: run_real_benchmark.py ${config}"
  set +e
  local workload_args=()
  if [[ -n "$WORKLOAD_JSONL" ]]; then workload_args+=(--workload-jsonl "$WORKLOAD_JSONL" --run-id "$RUN_ID"); fi
  python scripts/benchmark/run_real_benchmark.py --config "$config" --output-dir "$output_parent" "${workload_args[@]}" "$@" \
    > >(tee "$stdout_file" | tee -a "$OUTPUT_ROOT/${name}.log") \
    2> >(tee -a "$OUTPUT_ROOT/${name}.log" >&2)
  local status=$?
  set -e
  local run_dir
  run_dir="$(capture_benchmark_dir "$stdout_file" "$output_parent")"
  echo "$run_dir" > "$STATE_DIR/${name}.dir"
  if [[ "$status" -ne 0 ]]; then
    log "FAIL ${name}: exit=${status} run_dir=${run_dir}"
    if [[ "$CONTINUE_ON_FAILURE" -eq 0 ]]; then
      exit "$status"
    fi
  else
    log "OK ${name}: run_dir=${run_dir}"
  fi
  return "$status"
}

run_benchmark_capture_allow_failure() {
  local old_continue="$CONTINUE_ON_FAILURE"
  CONTINUE_ON_FAILURE=1
  run_benchmark_capture "$@"
  local status=$?
  CONTINUE_ON_FAILURE="$old_continue"
  log "ALLOW_FAILURE benchmark: exit=${status}"
  return 0
}

read_state_path() {
  local name="$1"
  local file="$STATE_DIR/${name}.dir"
  if [[ -f "$file" ]]; then
    cat "$file"
  fi
  return 0
}

extract_cache_events_for_run() {
  local name="$1"
  local run_dir="$2"
  local server_log="$3"
  local out_dir="$OUTPUT_ROOT/cache_events/${name}"
  if [[ -z "$run_dir" || ! -e "$run_dir/benchmark_results.csv" ]]; then
    log "Skipping cache events for ${name}: missing benchmark_results.csv"
    return 0
  fi
  mkdir -p "$out_dir"
  local args=(scripts/benchmark/extract_cache_events.py --benchmark-results "$run_dir/benchmark_results.csv" --request-results "$run_dir/request_results.jsonl" --output-dir "$out_dir")
  if [[ -f "$server_log" ]]; then
    args+=(--server-log "$server_log")
  fi
  run_cmd_allow_failure "cache_events_${name}" python "${args[@]}"
}

run_smoke() {
  log "Starting smoke workflow"
  run_cmd smoke_validation bash scripts/entrypoints/run_dgx_spark_validation.sh --skip-install --output-dir "$OUTPUT_ROOT/smoke_validation"
}

run_official() {
  log "Starting official workflow"
  run_cmd vm_evidence python scripts/vm/run_dgx_spark_vm_evidence.py --output-dir "$OUTPUT_ROOT/step2_vm_evidence"
  run_cmd_allow_failure mmap_smoke python cli.py vm mmap --blocks 16 --block-size-mb 1 --output-dir "$OUTPUT_ROOT/step2_mmap_smoke"

  local vllm_log="$OUTPUT_ROOT/step3_vllm_server.log"
  start_server vllm "$GPU_UTIL_OFFICIAL" 8192 "$vllm_log"
  run_benchmark_capture step3_vllm configs/dgx_spark_vllm_qwen7b.yaml "$OUTPUT_ROOT/step3_vllm_benchmark"
  local vllm_dir
  vllm_dir="$(read_state_path step3_vllm)"
  extract_cache_events_for_run step3_vllm "$vllm_dir" "$vllm_log"
  stop_server

  local cpu_log="$OUTPUT_ROOT/step4_lmcache_cpu_server.log"
  start_server cpu "$GPU_UTIL_OFFICIAL" 8192 "$cpu_log" configs/lmcache_cpu_constrained.yaml
  run_benchmark_capture step4_lmcache_cpu configs/dgx_spark_lmcache_cpu.yaml "$OUTPUT_ROOT/step4_lmcache_cpu_benchmark"
  local cpu_dir
  cpu_dir="$(read_state_path step4_lmcache_cpu)"
  extract_cache_events_for_run step4_lmcache_cpu "$cpu_dir" "$cpu_log"
  stop_server

  local disk_log="$OUTPUT_ROOT/step4_lmcache_disk_server.log"
  start_server disk "$GPU_UTIL_OFFICIAL" 8192 "$disk_log" configs/lmcache_disk_constrained.yaml
  run_benchmark_capture step4_lmcache_disk configs/dgx_spark_lmcache_disk.yaml "$OUTPUT_ROOT/step4_lmcache_disk_benchmark"
  local disk_dir
  disk_dir="$(read_state_path step4_lmcache_disk)"
  extract_cache_events_for_run step4_lmcache_disk "$disk_dir" "$disk_log"

  run_cmd_allow_failure step5_prefetch python scripts/benchmark/run_selective_prefetch_real.py \
    --config configs/astrakv_real_selective_prefetch.yaml \
    --output-dir "$OUTPUT_ROOT/step5_prefetch" \
    --model "$MODEL" \
    --backend vllm_lmcache_selective_prefetch \
    --cache-events "$OUTPUT_ROOT/cache_events/step4_lmcache_disk/cache_events.jsonl"
  run_cmd_allow_failure cache_events_step5_prefetch python scripts/benchmark/extract_cache_events.py \
    --server-log "$disk_log" \
    --output-dir "$OUTPUT_ROOT/cache_events/step5_prefetch"
  stop_server

  start_server vllm "$GPU_UTIL_OFFICIAL" 8192 "$OUTPUT_ROOT/step6_stress_vllm_server.log"
  run_benchmark_capture_allow_failure step6_stress_vllm configs/stress_vllm_memory_constrained.yaml "$OUTPUT_ROOT/step6_stress_vllm"
  stop_server
  start_server cpu "$GPU_UTIL_OFFICIAL" 8192 "$OUTPUT_ROOT/step6_stress_lmcache_cpu_server.log" configs/lmcache_cpu_constrained.yaml
  run_benchmark_capture_allow_failure step6_stress_lmcache_cpu configs/stress_lmcache_cpu_memory_constrained.yaml "$OUTPUT_ROOT/step6_stress_lmcache_cpu"
  stop_server
  start_server disk "$GPU_UTIL_OFFICIAL" 8192 "$OUTPUT_ROOT/step6_stress_lmcache_disk_server.log" configs/lmcache_disk_constrained.yaml
  run_benchmark_capture_allow_failure step6_stress_lmcache_disk configs/stress_lmcache_disk_memory_constrained.yaml "$OUTPUT_ROOT/step6_stress_lmcache_disk"
  stop_server
}

run_extreme() {
  log "Starting extreme workflow with gpu_util=${GPU_UTIL_EXTREME}"
  if start_server vllm "$GPU_UTIL_EXTREME" 16384 "$OUTPUT_ROOT/extreme_vllm_server.log"; then
    run_benchmark_capture_allow_failure extreme_vllm configs/stress_vllm_extreme_memory_constrained.yaml "$OUTPUT_ROOT/extreme_vllm"
  else
    log "Extreme vLLM server failed to become ready; retaining server log as boundary evidence"
  fi
  stop_server
  if start_server cpu "$GPU_UTIL_EXTREME" 16384 "$OUTPUT_ROOT/extreme_lmcache_cpu_server.log" configs/lmcache_cpu_constrained.yaml; then
    run_benchmark_capture_allow_failure extreme_lmcache_cpu configs/stress_lmcache_cpu_extreme_memory_constrained.yaml "$OUTPUT_ROOT/extreme_lmcache_cpu"
  else
    log "Extreme LMCache CPU server failed to become ready; retaining server log as boundary evidence"
  fi
  stop_server
  if start_server disk "$GPU_UTIL_EXTREME" 16384 "$OUTPUT_ROOT/extreme_lmcache_disk_server.log" configs/lmcache_disk_constrained.yaml; then
    run_benchmark_capture_allow_failure extreme_lmcache_disk configs/stress_lmcache_disk_extreme_memory_constrained.yaml "$OUTPUT_ROOT/extreme_lmcache_disk"
  else
    log "Extreme LMCache disk server failed to become ready; retaining server log as boundary evidence"
  fi
  stop_server
}

existing_benchmark_arg() {
  local label="$1"
  local dir="$2"
  if [[ -n "$dir" && -f "$dir/benchmark_results.csv" ]]; then
    echo "--benchmark" "${label}=${dir}"
  fi
}

existing_artifact_arg() {
  local label="$1"
  local kind="$2"
  local path="$3"
  if [[ -n "$path" && -e "$path" ]]; then
    echo "--artifact" "${label}:${kind}=${path}"
  fi
}

run_report() {
  log "Starting report workflow"
  local vllm_dir cpu_dir disk_dir stress_vllm stress_cpu stress_disk extreme_vllm extreme_cpu extreme_disk
  vllm_dir="$(read_state_path step3_vllm)"
  cpu_dir="$(read_state_path step4_lmcache_cpu)"
  disk_dir="$(read_state_path step4_lmcache_disk)"
  stress_vllm="$(read_state_path step6_stress_vllm)"
  stress_cpu="$(read_state_path step6_stress_lmcache_cpu)"
  stress_disk="$(read_state_path step6_stress_lmcache_disk)"
  extreme_vllm="$(read_state_path extreme_vllm)"
  extreme_cpu="$(read_state_path extreme_lmcache_cpu)"
  extreme_disk="$(read_state_path extreme_lmcache_disk)"

  if [[ -n "$vllm_dir" && -n "$cpu_dir" && -n "$disk_dir" ]]; then
    run_cmd_allow_failure step7_compare python scripts/reporting/compare_real_runs.py \
      --run "vllm=${vllm_dir}" \
      --run "lmcache_cpu=${cpu_dir}" \
      --run "lmcache_disk=${disk_dir}" \
      --output-dir "$OUTPUT_ROOT/step7_comparison"
  fi

  local stress_args=()
  for item in \
    "vllm=${stress_vllm}" \
    "lmcache_cpu=${stress_cpu}" \
    "lmcache_disk=${stress_disk}"; do
    local path="${item#*=}"
    if [[ -n "$path" && -f "$path/benchmark_results.csv" ]]; then
      stress_args+=(--run "$item")
    fi
  done
  if [[ "${#stress_args[@]}" -gt 0 ]]; then
    run_cmd_allow_failure step6_stress_analysis python scripts/reporting/analyze_stress_results.py "${stress_args[@]}" --output-dir "$OUTPUT_ROOT/step6_stress_analysis"
  fi

  local extreme_args=()
  for item in \
    "vllm_extreme=${extreme_vllm}" \
    "lmcache_cpu_extreme=${extreme_cpu}" \
    "lmcache_disk_extreme=${extreme_disk}"; do
    local path="${item#*=}"
    if [[ -n "$path" && -f "$path/benchmark_results.csv" ]]; then
      extreme_args+=(--run "$item")
    fi
  done
  if [[ "${#extreme_args[@]}" -gt 0 ]]; then
    run_cmd_allow_failure extreme_stress_analysis python scripts/reporting/analyze_stress_results.py "${extreme_args[@]}" --output-dir "$OUTPUT_ROOT/extreme_stress_analysis"
  fi

  local trace_events="$OUTPUT_ROOT/step7_trace_store/trace_events.jsonl"
  local trace_summary="$OUTPUT_ROOT/step7_trace_store/trace_summary.md"
  local profile_db="$OUTPUT_ROOT/step7_profile_db/profile_db.json"
  local chunk_scores="$OUTPUT_ROOT/step7_chunk_scores/chunk_scores.csv"

  local trace_args=()
  for cache_path in \
    "$OUTPUT_ROOT/cache_events/step3_vllm/cache_events.jsonl" \
    "$OUTPUT_ROOT/cache_events/step4_lmcache_cpu/cache_events.jsonl" \
    "$OUTPUT_ROOT/cache_events/step4_lmcache_disk/cache_events.jsonl" \
    "$OUTPUT_ROOT/cache_events/step5_prefetch/cache_events.jsonl"; do
    if [[ -f "$cache_path" ]]; then
      trace_args+=(--cache-events "$cache_path")
    fi
  done
  if [[ -f "$OUTPUT_ROOT/step5_prefetch/prefetch_events.jsonl" ]]; then
    trace_args+=(--prefetch-events "$OUTPUT_ROOT/step5_prefetch/prefetch_events.jsonl")
  fi
  for sample_dir in \
    "$vllm_dir/samples" \
    "$cpu_dir/samples" \
    "$disk_dir/samples" \
    "$stress_vllm/samples" \
    "$stress_cpu/samples" \
    "$stress_disk/samples" \
    "$extreme_vllm/samples" \
    "$extreme_cpu/samples" \
    "$extreme_disk/samples"; do
    if [[ -d "$sample_dir" ]]; then
      trace_args+=(--samples "$sample_dir")
    fi
  done

  if [[ ! -f "$trace_events" && "${#trace_args[@]}" -gt 0 ]]; then
    if [[ -n "$WORKLOAD_JSONL" ]]; then trace_args+=(--workload-manifest "$WORKLOAD_JSONL" --run-id "$RUN_ID"); fi
    run_cmd_allow_failure step7_trace_store python scripts/policy/build_trace_store.py \
      "${trace_args[@]}" \
      --output-dir "$OUTPUT_ROOT/step7_trace_store"
  elif [[ "${#trace_args[@]}" -eq 0 ]]; then
    log "Skipping step7_trace_store: no cache, prefetch, or sample artifacts found"
  fi

  if [[ ! -f "$profile_db" && -f "$trace_events" ]]; then
    run_cmd_allow_failure step7_profile_db python scripts/policy/build_profile_db.py \
      --trace-events "$trace_events" \
      --workload-id competition_e2e \
      --output-dir "$OUTPUT_ROOT/step7_profile_db"
  fi

  if [[ ! -f "$profile_db" && -f "$OUTPUT_ROOT/profile_db.json" ]]; then
    profile_db="$OUTPUT_ROOT/profile_db.json"
  fi

  if [[ ! -f "$chunk_scores" && -f "$profile_db" ]]; then
    mkdir -p "$OUTPUT_ROOT/step7_chunk_scores"
    run_cmd_allow_failure step7_chunk_scores python scripts/policy/score_chunks.py \
      --profile-db "$profile_db" \
      --output-dir "$OUTPUT_ROOT/step7_chunk_scores"
  elif [[ ! -f "$profile_db" ]]; then
    log "Skipping step7_chunk_scores: missing ProfileDB"
  fi

  local policy_args=(scripts/policy/analyze_policy_ablation.py --output-dir "$OUTPUT_ROOT/step7_policy_ablation")
  if [[ -n "$vllm_dir" ]]; then
    policy_args+=(--benchmark-run "no_prefetch=${vllm_dir}")
  fi
  if [[ -f "$OUTPUT_ROOT/step5_prefetch/prefetch_benchmark_results.csv" ]]; then
    policy_args+=(--benchmark-run "astrakv_combined=$OUTPUT_ROOT/step5_prefetch/prefetch_benchmark_results.csv")
  elif [[ -n "$disk_dir" ]]; then
    policy_args+=(--benchmark-run "astrakv_combined=${disk_dir}")
  fi
  if [[ -f "$OUTPUT_ROOT/step5_prefetch/prefetch_results.csv" ]]; then
    policy_args+=(--prefetch-run "astrakv_combined=$OUTPUT_ROOT/step5_prefetch/prefetch_results.csv")
  fi
  if [[ -f "$chunk_scores" ]]; then
    policy_args+=(--chunk-scores "astrakv_combined=$chunk_scores")
  fi
  run_cmd_allow_failure step7_policy_ablation python "${policy_args[@]}"

  local load_decisions="$OUTPUT_ROOT/step7_load_recompute/load_recompute_decisions.csv"
  if [[ -f "$profile_db" ]]; then
    run_cmd_allow_failure step7_load_recompute python scripts/policy/decide_load_vs_recompute.py \
      --profile-db "$profile_db" --output-dir "$OUTPUT_ROOT/step7_load_recompute"
    local object_args=(scripts/policy/run_unified_object_scheduler.py --profile-db "$profile_db" --output-dir "$OUTPUT_ROOT/step7_object_scheduler")
    if [[ -f "$chunk_scores" ]]; then object_args+=(--chunk-scores "$chunk_scores"); fi
    if [[ -f "$load_decisions" ]]; then object_args+=(--load-recompute-decisions "$load_decisions"); fi
    run_cmd_allow_failure step7_object_scheduler python "${object_args[@]}"
  fi

  local runtime_eviction_dir="$OUTPUT_ROOT/runtime_eviction"
  local schedule_path="$OUTPUT_ROOT/step7_object_scheduler/object_schedule_decisions.csv"
  local disk_cache="$OUTPUT_ROOT/cache_events/step4_lmcache_disk/cache_events.jsonl"
  local offline_dir="${OFFLINE_OUTPUT_DIR:-$OUTPUT_ROOT/offline_eviction}"
  local effective_gate="$OFFLINE_SAFETY_GATE"
  if [[ -n "$OFFLINE_WORKLOAD_SET" && -f "$OFFLINE_WORKLOAD_SET" && -n "$WORKLOAD_JSONL" && -f "$trace_events" && -f "$profile_db" && "$OFFLINE_GPU_CAPACITY_BYTES" -gt 0 && "$OFFLINE_CPU_CAPACITY_BYTES" -gt 0 && "$OFFLINE_SSD_CAPACITY_BYTES" -gt 0 && "$OFFLINE_DEFAULT_OBJECT_BYTES" -gt 0 ]]; then
    local current_entry="$offline_dir/current_workload.json"
    mkdir -p "$offline_dir"
    python - "$current_entry" "$RUN_ID" "$WORKLOAD_JSONL" "$trace_events" "$profile_db" "$schedule_path" "$disk_dir" "$OFFLINE_GPU_CAPACITY_BYTES" "$OFFLINE_CPU_CAPACITY_BYTES" "$OFFLINE_SSD_CAPACITY_BYTES" "$OFFLINE_DEFAULT_OBJECT_BYTES" <<'PY'
import json, sys
path, run_id, workload, trace, profile, schedule, disk_dir, gpu, cpu, ssd, obj = sys.argv[1:]
json.dump({"workload_id": f"current-{run_id}", "workload_manifest": workload, "trace": trace, "profile_db": profile,
           "scheduler_decisions": schedule if __import__('pathlib').Path(schedule).exists() else "",
           "request_results": str(__import__('pathlib').Path(disk_dir) / "request_results.jsonl") if disk_dir else "",
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
    local runtime_args=(bash scripts/entrypoints/run_runtime_eviction_validation.sh --mode "$RUNTIME_MODE" --workload-jsonl "$WORKLOAD_JSONL" --run-id "$RUN_ID" --offline-decisions "$schedule_path" --output-dir "$runtime_eviction_dir")
    if [[ -f "$disk_cache" ]]; then runtime_args+=(--cache-events "$disk_cache"); fi
    if [[ -n "$disk_dir" && -f "$disk_dir/request_results.jsonl" ]]; then runtime_args+=(--request-results "$disk_dir/request_results.jsonl"); fi
    if [[ -f "$OUTPUT_ROOT/step4_lmcache_disk_server.log" ]]; then runtime_args+=(--server-log "$OUTPUT_ROOT/step4_lmcache_disk_server.log"); fi
    if [[ -n "$effective_gate" && -f "$effective_gate" ]]; then runtime_args+=(--offline-safety-gate "$effective_gate"); fi
    if [[ -n "$STRUCTURED_EVENTS" ]]; then runtime_args+=(--structured-events "$STRUCTURED_EVENTS"); fi
    if [[ -n "$STRUCTURED_HOOK_VERIFICATION" ]]; then runtime_args+=(--structured-hook-verification "$STRUCTURED_HOOK_VERIFICATION"); fi
    run_cmd_allow_failure runtime_eviction "${runtime_args[@]}"
  fi
  if [[ "$RUN_DIAGNOSTIC" -eq 1 ]]; then
    run_cmd_allow_failure runtime_diagnostic python scripts/benchmark/diagnose_runtime.py --pid "$DIAGNOSTIC_PID" --collect-tools --output-dir "$OUTPUT_ROOT/runtime_diagnostic"
  fi

  local report_args=(scripts/reporting/build_competition_report.py --output-dir "$OUTPUT_ROOT/competition_report" --title "AstraKV-W Competition E2E Report")
  report_args+=(--command "bash scripts/entrypoints/run_competition_e2e.sh --only ${ONLY} --output-root ${OUTPUT_ROOT} --model ${MODEL}")
  if [[ -n "$vllm_dir" && -f "$vllm_dir/benchmark_results.csv" ]]; then report_args+=(--benchmark "vllm=${vllm_dir}"); fi
  if [[ -n "$cpu_dir" && -f "$cpu_dir/benchmark_results.csv" ]]; then report_args+=(--benchmark "lmcache_cpu=${cpu_dir}"); fi
  if [[ -n "$disk_dir" && -f "$disk_dir/benchmark_results.csv" ]]; then report_args+=(--benchmark "lmcache_disk=${disk_dir}"); fi
  if [[ -f "$OUTPUT_ROOT/step2_vm_evidence/dgx_spark_vm_evidence_summary.json" ]]; then report_args+=(--vm-evidence "dgx_vm=$OUTPUT_ROOT/step2_vm_evidence/dgx_spark_vm_evidence_summary.json"); fi
  if [[ -f "$OUTPUT_ROOT/step2_mmap_smoke/mmap_kv_demo_summary.json" ]]; then report_args+=(--vm-evidence "mmap_smoke=$OUTPUT_ROOT/step2_mmap_smoke/mmap_kv_demo_summary.json"); fi
  if [[ -f "$OUTPUT_ROOT/step5_prefetch/prefetch_results.csv" ]]; then report_args+=(--prefetch "selective_prefetch=$OUTPUT_ROOT/step5_prefetch/prefetch_results.csv"); fi
  if [[ -f "$OUTPUT_ROOT/runtime_eviction/runtime_agreement/eviction_agreement_manifest.json" ]]; then report_args+=(--eviction-agreement "runtime=$OUTPUT_ROOT/runtime_eviction/runtime_agreement/eviction_agreement_manifest.json"); fi
  if [[ -f "$OUTPUT_ROOT/runtime_eviction/vm_poc_agreement/eviction_agreement_manifest.json" ]]; then report_args+=(--eviction-agreement "vm_poc=$OUTPUT_ROOT/runtime_eviction/vm_poc_agreement/eviction_agreement_manifest.json"); fi
  local report_offline_dir="${OFFLINE_OUTPUT_DIR:-$OUTPUT_ROOT/offline_eviction}"
  if [[ -f "$report_offline_dir/offline_safety_gate.json" ]]; then report_args+=(--offline-safety-gate "offline_gate=$report_offline_dir/offline_safety_gate.json"); fi
  if [[ -f "$report_offline_dir/offline_pipeline_status.json" ]]; then report_args+=(--artifact "offline_pipeline:artifact=$report_offline_dir/offline_pipeline_status.json"); fi
  if [[ -f "$OUTPUT_ROOT/runtime_diagnostic/diagnostic_manifest.json" ]]; then report_args+=(--diagnostic "runtime=$OUTPUT_ROOT/runtime_diagnostic/diagnostic_manifest.json"); fi
  shopt -s nullglob
  for policy_summary in "$report_offline_dir"/workloads/*/offline_eviction_policy_summary.csv; do report_args+=(--offline-policy-summary "$(basename "$(dirname "$policy_summary")")=$policy_summary"); done
  while IFS= read -r experiment_manifest; do report_args+=(--experiment-manifest "$(basename "$(dirname "$experiment_manifest")")=$experiment_manifest"); done < <(find "$OUTPUT_ROOT" -name experiment_manifest.json -type f)
  shopt -u nullglob
  if [[ -f "$OUTPUT_ROOT/cache_events/step4_lmcache_cpu/cache_events.jsonl" ]]; then report_args+=(--cache-events "lmcache_cpu=$OUTPUT_ROOT/cache_events/step4_lmcache_cpu/cache_events.jsonl"); fi
  if [[ -f "$OUTPUT_ROOT/cache_events/step4_lmcache_disk/cache_events.jsonl" ]]; then report_args+=(--cache-events "lmcache_disk=$OUTPUT_ROOT/cache_events/step4_lmcache_disk/cache_events.jsonl"); fi
  if [[ -f "$OUTPUT_ROOT/cache_events/step5_prefetch/cache_events.jsonl" ]]; then report_args+=(--cache-events "prefetch=$OUTPUT_ROOT/cache_events/step5_prefetch/cache_events.jsonl"); fi
  if [[ -f "$OUTPUT_ROOT/step6_stress_analysis/stress_summary.csv" ]]; then report_args+=(--artifact "official_stress:stress=$OUTPUT_ROOT/step6_stress_analysis/stress_summary.csv"); fi
  if [[ -f "$OUTPUT_ROOT/extreme_stress_analysis/stress_summary.csv" ]]; then report_args+=(--artifact "extreme_stress:stress=$OUTPUT_ROOT/extreme_stress_analysis/stress_summary.csv"); fi
  if [[ -f "$OUTPUT_ROOT/step7_comparison/comparison_results.csv" ]]; then report_args+=(--comparison "$OUTPUT_ROOT/step7_comparison/comparison_results.csv"); fi
  if [[ -f "$OUTPUT_ROOT/step7_policy_ablation/policy_ablation_results.csv" ]]; then report_args+=(--policy-ablation "$OUTPUT_ROOT/step7_policy_ablation/policy_ablation_results.csv"); fi
  if [[ -f "$trace_summary" ]]; then report_args+=(--trace-summary "$trace_summary"); fi
  if [[ -f "$profile_db" ]]; then report_args+=(--profile-db "$profile_db"); fi
  if [[ -f "$chunk_scores" ]]; then report_args+=(--chunk-scores "$chunk_scores"); fi
  shopt -s nullglob
  for log_file in "$OUTPUT_ROOT"/*server.log; do
    if [[ -f "$log_file" ]]; then
      report_args+=(--server-log "$(basename "$log_file" .log)=$log_file")
    fi
  done
  shopt -u nullglob
  run_cmd_allow_failure competition_report python "${report_args[@]}"
}

main() {
  log "Output root: $OUTPUT_ROOT"
  log "Model: $MODEL"
  if [[ "$WITH_CGROUP" -eq 1 ]]; then
    log "--with-cgroup requested. This script does not create privileged cgroups by default; apply host limits externally or wrap this script in a cgroup runner."
  fi
  ensure_python
  case "$ONLY" in
    smoke)
      run_smoke
      ;;
    official)
      run_official
      ;;
    extreme)
      run_extreme
      ;;
    report)
      run_report
      ;;
    all)
      run_official
      run_extreme
      run_report
      ;;
  esac
  log "E2E artifacts written to $OUTPUT_ROOT"
}

main "$@"
