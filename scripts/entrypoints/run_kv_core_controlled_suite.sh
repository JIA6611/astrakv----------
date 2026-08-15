#!/usr/bin/env bash
set -Eeuo pipefail

# E0--E4 runner for the version-locked Qwen3-8B KV-Core experiment.  E3C is
# a CPU-tier A/A correctness control, not a performance phase. It does
# not send warmup HTTP requests: every request is part of the supplied,
# content-addressed workload and cache state is explicit in its manifest.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
WORKLOAD_DIR=""
OUTPUT_DIR="$ROOT/results/kv-core-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
WARM_STORE_DIR=""
PHASES="E1"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
TIMEOUT="900"
GPU_MEMORY_UTILIZATION="0.72"
OUTPUT_TOKENS="${ASTRAKV_CONTROL_OUTPUT_TOKENS:-128}"
# E1 is a lifecycle gate, not a performance sweep. Its default is deliberately
# one cold repeated-prefix workload that exercises all seven native callbacks.
WORKLOADS="repeated_long_prefix"
CACHE_STATES="cold"
ALLOW_INELIGIBLE=false
OFFLINE_PROFILE=""
VARIANT_ONLY=false

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_kv_core_controlled_suite.sh --workload-dir DIR [--phases E1,E2,E3,E3C,E4,E5,E5C] [--patch-manifest FILE --callback-smoke FILE] [options]

DIR must contain these canonical JSONL workloads: repeated_long_prefix,
random_no_reuse, constrained_kv_churn, queued_concurrency.  Each input must
already fix request order, seed, sampling parameters, output length, and cold
or warm cache-state cases.  Active phases additionally require verified
connector evidence and real runtime accounting.
--page-cache-evidence additionally samples mincore residency of the LMCache
disk store alongside every run (OS virtual-memory evidence for the report).
--variant-only skips the within-phase control member.  It is intended only
for cross-cell regime matrices whose report compares the variant runs to a
separate recompute-only cell; the default paired acceptance path is unchanged.
--output-tokens N fixes generation length for every paired request (default
128, or ASTRAKV_CONTROL_OUTPUT_TOKENS).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload-dir) WORKLOAD_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --warm-store-dir) WARM_STORE_DIR="$2"; shift 2 ;;
    --phases) PHASES="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --workloads) WORKLOADS="$2"; shift 2 ;;
    --cache-states) CACHE_STATES="$2"; shift 2 ;;
    --allow-ineligible) ALLOW_INELIGIBLE=true; shift ;;
    --variant-only) VARIANT_ONLY=true; shift ;;
    --offline-profile) OFFLINE_PROFILE="$2"; shift 2 ;;
    --page-cache-evidence) PAGE_CACHE_EVIDENCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$WORKLOAD_DIR" && -d "$WORKLOAD_DIR" ]] || { echo "--workload-dir is required" >&2; exit 2; }
[[ "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "--output-tokens must be positive" >&2; exit 2; }
[[ -z "$OFFLINE_PROFILE" || -f "$OFFLINE_PROFILE" ]] || { echo "invalid --offline-profile" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "only a loopback server is supported" >&2; exit 2; }
IFS=',' read -r -a SELECTED_PHASES <<< "$PHASES"
[[ "${#SELECTED_PHASES[@]}" -gt 0 ]] || { echo "--phases must not be empty" >&2; exit 2; }
ACTIVE_PHASE_SELECTED=false
for phase in "${SELECTED_PHASES[@]}"; do
  [[ "$phase" =~ ^(E0|E1|E2|E2R|E3|E3C|E4|E5|E5C)$ ]] || { echo "invalid phase: $phase" >&2; exit 2; }
  [[ "$phase" == E1 || "$phase" == E0 ]] || ACTIVE_PHASE_SELECTED=true
done
if [[ "$ACTIVE_PHASE_SELECTED" == true ]]; then
  [[ -f "$PATCH_MANIFEST" && -f "$CALLBACK_SMOKE" ]] || { echo "verified deployment inputs are required for E2-E4 and E3C" >&2; exit 2; }
fi
IFS=',' read -r -a SELECTED_WORKLOADS <<< "$WORKLOADS"
IFS=',' read -r -a SELECTED_CACHE_STATES <<< "$CACHE_STATES"
for workload in "${SELECTED_WORKLOADS[@]}"; do
  [[ "$workload" =~ ^(repeated_long_prefix|random_no_reuse|constrained_kv_churn|queued_concurrency)$ ]] || {
    echo "invalid workload: $workload" >&2; exit 2;
  }
  [[ -f "$WORKLOAD_DIR/$workload.jsonl" ]] || { echo "Missing $workload.jsonl" >&2; exit 2; }
done
for cache_state in "${SELECTED_CACHE_STATES[@]}"; do
  [[ "$cache_state" =~ ^(cold|warm)$ ]] || { echo "invalid cache state: $cache_state" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
if [[ "$ACTIVE_PHASE_SELECTED" == true ]]; then
  "$PYTHON" scripts/runtime/verify_kv_core_connector_patch.py \
    --deployment-manifest "$PATCH_MANIFEST" --callback-smoke "$CALLBACK_SMOKE" \
    --output "$OUTPUT_DIR/connector_patch_verification.json"
fi

SERVER_PID=""
PAGE_CACHE_PID=""
PAGE_CACHE_EVIDENCE=false
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$PAGE_CACHE_PID" ]] && kill -0 "$PAGE_CACHE_PID" 2>/dev/null; then
    kill -TERM "$PAGE_CACHE_PID" 2>/dev/null || true
    wait "$PAGE_CACHE_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  PAGE_CACHE_PID=""
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 120); do
    curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null && return 0
    sleep 2
  done
  tail -n 160 "$log_path" >&2 || true
  return 1
}

assert_lmcache_healthy() {
  local log_path="$1"
  if grep -q "LMCacheEngine marked as init failed" "$log_path"; then
    echo "LMCache initialization failed; refusing to benchmark degraded recompute mode." >&2
    tail -n 180 "$log_path" >&2 || true
    return 1
  fi
}

assert_lmcache_runtime_healthy() {
  local log_path="$1"
  if grep -Eq "No eviction candidates found in local cpu backend|Memory allocation failed during (async )?disk load" "$log_path"; then
    echo "LMCache CPU staging pool was exhausted; refusing to accept a stalled or partial disk restore." >&2
    tail -n 180 "$log_path" >&2 || true
    return 1
  fi
}

run_one() {
  local label="$1" phase="$2" role="$3" workload="$4" cache_state="$5" baseline_label="$6"
  local run_id="kv-core-${phase}-${workload}-${cache_state}-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$OUTPUT_DIR/$label/$workload/$cache_state/$role"
  local state_dir="$run_dir/state"
  local log_path="$run_dir/server.log"
  local backend="disk" topology="gpu_ssd" mode="off" admission="false" prefetch="false" partial="false" reap="false"
  mkdir -p "$state_dir"
  case "$phase" in
    E0) mode="off" ;;
    E1) mode="shadow" ;;
    E2) mode="active"; admission="true" ;;
    # E2R: active admission with every revisit forced to scheduler-declined
    # native recompute (test-only per-request override). Seed rows still write
    # the LMCache object; the arm is the clean "recompute-only" baseline for
    # the load-vs-recompute regime matrix.
    E2R) mode="active"; admission="true" ;;
    E3) mode="active"; admission="true"; prefetch="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    # A/A control for CPU-tier correctness. Both members use active native
    # admission and demand load/recompute, but no SSD->CPU prefetch is allowed.
    E3C) mode="active"; admission="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    E4) mode="active"; admission="true"; prefetch="true"; partial="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    # E5 isolates the cold external-copy reaper on top of the E4 configuration
    # (admission + CPU prefetch + partial prefix). E5C is the A/A control with
    # the reaper disabled and every other runtime knob identical.
    E5C) mode="active"; admission="true"; prefetch="true"; partial="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    E5) mode="active"; admission="true"; prefetch="true"; partial="true"; reap="true"; topology="gpu_cpu_ssd"; backend="cpu" ;;
    *) echo "invalid phase: $phase" >&2; return 2 ;;
  esac
  if [[ -n "${ASTRAKV_CONTROL_TOPOLOGY:-}" ]]; then
    topology="$ASTRAKV_CONTROL_TOPOLOGY"
    [[ "$topology" == gpu_cpu_ssd ]] && backend="cpu"
  fi
  local runtime_secret_hex
  runtime_secret_hex="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  cleanup
  # Disable vLLM's in-process prefix cache for this external-KV experiment.
  # Both pair members have the same setting; reuse comes only from the
  # pair-scoped LMCache disk store.
  ASTRAKV_PYTHON="$PYTHON" \
  ASTRAKV_MODEL="$MODEL" ASTRAKV_HOST="$HOST" ASTRAKV_PORT="$PORT" \
  PYTHONHASHSEED=0 ASTRAKV_VLLM_SEED=0 \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_MAX_MODEL_LEN="32768" \
  LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:?pair-scoped LMCache config is required}" \
  ASTRAKV_PREFIX_CACHING=false ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE=engine_child ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" \
  ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" ASTRAKV_RUNTIME_CONTROL_ENGINE_ID="$run_id-engine" \
  ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0 ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
  ASTRAKV_RUNTIME_CONTROL_SESSION_ID="$run_id-session" ASTRAKV_RUNTIME_CONTROL_SECRET_HEX="$runtime_secret_hex" \
  ASTRAKV_KV_CORE_VENDOR_PATCH=true ASTRAKV_MODEL_ID="Qwen3-8B" \
  ASTRAKV_MODEL_REVISION="local-qwen3-8b" ASTRAKV_TOKENIZER_REVISION="local-qwen3-8b" \
  ASTRAKV_CHAT_TEMPLATE_REVISION="qwen3-default" \
  ASTRAKV_REQUIRE_EXACT_TOKEN_IDS=true \
  ASTRAKV_KV_CORE_MODE="$mode" ASTRAKV_KV_CORE_TOPOLOGY="$topology" \
  ASTRAKV_KV_CORE_LOCAL_CPU="$([[ "$topology" == gpu_cpu_ssd ]] && echo true || echo false)" \
  ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD="$([[ "$topology" == gpu_cpu_ssd ]] && echo true || echo false)" \
  ASTRAKV_KV_CORE_PATCH_VERIFICATION="$OUTPUT_DIR/connector_patch_verification.json" \
  ASTRAKV_KV_CORE_ADMISSION_ENABLED="$admission" ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED="$prefetch" \
  ASTRAKV_KV_CORE_EQUIVALENCE_TEST="$([[ "$phase" == E2R ]] && echo true || echo false)" \
  ASTRAKV_KV_CORE_PARTIAL_PREFIX_UPPER_BOUND_ENABLED="$partial" \
  ASTRAKV_KV_CORE_COLD_REAP_ENABLED="$reap" \
  ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP="8192" ASTRAKV_KV_CORE_PARTIAL_PREFIX_TOKENS="2048" \
  ASTRAKV_KV_CORE_BOOTSTRAP_LOADS="2" ASTRAKV_KV_CORE_SSD_READ_GBPS="3.0" \
  ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION=0.2 \
  ASTRAKV_KV_CORE_RELEASE_CPU_STAGING_ON_CONSUME=true \
  ASTRAKV_KV_CORE_PREFILL_ONLINE_CALIBRATION=true ASTRAKV_KV_CORE_PREFILL_SAMPLE_MIN_TOKENS=32 \
  ASTRAKV_KV_CORE_PREFILL_SAMPLE_MAX_MS_PER_TOKEN=5.0 ASTRAKV_KV_CORE_PREFILL_EMA_ALPHA=0.25 \
  ASTRAKV_KV_CORE_OFFLINE_PROFILE="$OFFLINE_PROFILE" \
  nohup bash scripts/launch/launch_lmcache_vllm.sh "$backend" > "$log_path" 2>&1 < /dev/null &
  SERVER_PID="$!"
  wait_for_server "$log_path"
  assert_lmcache_healthy "$log_path"
  if [[ "$PAGE_CACHE_EVIDENCE" == "true" ]]; then
    local disk_dir
    disk_dir="$(sed -n 's/^local_disk: *//p' "$LMCACHE_CONFIG_FILE")"
    if [[ -n "$disk_dir" && -d "$disk_dir" ]]; then
      mkdir -p "$OUTPUT_DIR/page_cache_evidence/$label"
      nohup "$PYTHON" scripts/runtime/collect_page_cache_evidence.py \
        --path "$disk_dir" \
        --output "$OUTPUT_DIR/page_cache_evidence/$label/$workload-$cache_state.jsonl" \
        --interval-s 1.0 --duration-s "$TIMEOUT" \
        > "$OUTPUT_DIR/page_cache_evidence/$label/collector.log" 2>&1 < /dev/null &
      PAGE_CACHE_PID="$!"
    fi
  fi
  # The benchmark process is separate from the server child. Repeat only
  # immutable, non-secret controls here so the paired manifest fingerprints
  # the configuration actually used by the server rather than blank values.
  if ! ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" ASTRAKV_VLLM_SEED=0 \
    ASTRAKV_MAX_MODEL_LEN="32768" ASTRAKV_PREFIX_CACHING=false \
    ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:?pair-scoped LMCache config is required}" \
    "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" --model "$MODEL" --backend "vllm-lmcache-kv-core" \
    --output-dir "$run_dir" --workload-jsonl "$WORKLOAD_DIR/$workload.jsonl" \
    --run-id "$run_id" --workload-id "$workload" --model-revision local-qwen3-8b \
    --tokenizer-revision local-qwen3-8b --dtype bfloat16 --quantization unquantized \
    --tokenizer-path "$MODEL" --chat-template-revision qwen3-default \
    --random-seed 0 --cache-state "$cache_state" --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "${label}-${workload}-${cache_state}" --pair-role "$role" --claim-scope kv_core \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "$run_id-session" --request-context-secret-hex "$runtime_secret_hex" \
    --timeout "$TIMEOUT" --output-tokens "$OUTPUT_TOKENS"; then
    assert_lmcache_runtime_healthy "$log_path" || true
    return 1
  fi
  assert_lmcache_runtime_healthy "$log_path"
  # These artifacts are emitted by the version-locked connector patch after
  # native events.  Do not synthesize estimated receipts or block capacity.
  for artifact in callback-smoke.json kv_core_native_callbacks.jsonl kv_core_native_receipts.jsonl kv_core_request_accounting.jsonl request_context_associations.jsonl kv_core_prefetch_tickets.jsonl kv_core_policy_decisions.jsonl kv_core_cost_observations.jsonl uma_resource_samples.jsonl kv_core_external_reaps.jsonl kv_core_run_metadata.json; do
    if [[ -f "$state_dir/$artifact" ]]; then
      cp "$state_dir/$artifact" "$run_dir/$artifact"
    elif [[ "$phase" =~ ^(E[2-5]R?|E3C|E5C)$ && "$artifact" == kv_core_request_accounting.jsonl ]]; then
      echo "Missing required active-phase accounting artifact: $state_dir/$artifact" >&2
      return 1
    fi
  done
  if [[ -d "$state_dir/native_receipts" ]]; then
    cp -a "$state_dir/native_receipts" "$run_dir/native_receipts"
  fi
  cleanup
}

run_pair() {
  local label="$1" baseline_phase="$2" variant_phase="$3" workload="$4" cache_state="$5"
  local baseline_cache_dir="$OUTPUT_DIR/lmcache-store/$label/$workload/$cache_state/baseline"
  local variant_cache_dir="$OUTPUT_DIR/lmcache-store/$label/$workload/$cache_state/variant"
  local baseline_cache_config="$OUTPUT_DIR/lmcache-config/$label/$workload/$cache_state-baseline.yaml"
  local variant_cache_config="$OUTPUT_DIR/lmcache-config/$label/$workload/$cache_state-variant.yaml"
  # LMCache 0.4.7 requires a LocalCPUBackend object whenever LocalDiskBackend
  # is configured. Keep it non-hot for gpu_ssd: staging only, not prefetch.
  # A Qwen3-8B 256-token chunk is about 36 MiB; 0.1 GiB holds only two chunks
  # and makes the synchronous disk-restore allocator busy-loop indefinitely.
  # Two GiB is the LMCache gpu_ssd staging budget, shared by both pair members.
  local local_cpu="false" local_cpu_size="2.0" control_topology="gpu_ssd"
  if [[ "$variant_phase" == E3 || "$variant_phase" == E3C || "$variant_phase" == E4 ]]; then
    local_cpu="true"
    control_topology="gpu_cpu_ssd"
    # Budget is intentionally explicit and bounded; the runtime must still
    # prove the observed LocalCPUBackend capacity before using prefetch.
    local_cpu_size="5.0"
  fi
  mkdir -p "$baseline_cache_dir" "$variant_cache_dir" "$(dirname "$baseline_cache_config")"
  if [[ "$cache_state" == warm ]]; then
    [[ -n "$WARM_STORE_DIR" && -d "$WARM_STORE_DIR/$workload" ]] || {
      echo "Warm pair requires --warm-store-dir/$workload; refusing an unseeded warm claim" >&2
      return 1
    }
    cp -a "$WARM_STORE_DIR/$workload/." "$baseline_cache_dir/"
    cp -a "$WARM_STORE_DIR/$workload/." "$variant_cache_dir/"
  fi
  cat > "$baseline_cache_config" <<EOF
local_cpu: $local_cpu
max_local_cpu_size: $local_cpu_size
local_disk: $baseline_cache_dir
max_local_disk_size: 80.0
EOF
  cat > "$variant_cache_config" <<EOF
local_cpu: $local_cpu
max_local_cpu_size: $local_cpu_size
local_disk: $variant_cache_dir
max_local_disk_size: 80.0
EOF
  if [[ "$VARIANT_ONLY" != true ]]; then
    ASTRAKV_CONTROL_TOPOLOGY="$control_topology" \
      LMCACHE_CONFIG_FILE="$baseline_cache_config" run_one "$label" "$baseline_phase" baseline "$workload" "$cache_state" "$label"
  fi
  # Both arms learn their admission cost from their own native scheduler
  # callbacks. Do not derive variant policy inputs from baseline HTTP TTFT:
  # TTFT includes queueing and I/O and would violate the paired control.
  ASTRAKV_CONTROL_TOPOLOGY="$control_topology" \
    LMCACHE_CONFIG_FILE="$variant_cache_config" run_one "$label" "$variant_phase" variant "$workload" "$cache_state" "$label"
  if [[ "$VARIANT_ONLY" == true ]]; then
    return 0
  fi
  if [[ "$variant_phase" == E0 ]]; then
    # E0 is a raw off-mode baseline cell for the regime matrix (TTFT/UMA
    # reference only); it carries no phase-level eligibility claim.
    return 0
  fi
  if ! "$PYTHON" scripts/reporting/validate_kv_core_acceptance.py \
      --baseline "$OUTPUT_DIR/$label/$workload/$cache_state/baseline" \
      --variant "$OUTPUT_DIR/$label/$workload/$cache_state/variant" \
      --phase "$variant_phase" --output "$OUTPUT_DIR/$label/$workload/$cache_state/acceptance.json"; then
    [[ "$ALLOW_INELIGIBLE" == true ]] || return 1
  fi
}

for workload in "${SELECTED_WORKLOADS[@]}"; do
  for cache_state in "${SELECTED_CACHE_STATES[@]}"; do
    for phase in "${SELECTED_PHASES[@]}"; do
      case "$phase" in
        E0) run_pair E0 E0 E0 "$workload" "$cache_state" ;;
        E1) run_pair E1 E0 E1 "$workload" "$cache_state" ;;
        E2) run_pair E2 E0 E2 "$workload" "$cache_state" ;;
        E2R) run_pair E2R E0 E2R "$workload" "$cache_state" ;;
        # Keep the CPU tier, request-scoped native admission, and the
        # symmetric disk-backed CPU invalidation fixed. E3 then isolates only
        # the SSD->CPU prefetch switch instead of conflating it with E2's
        # gpu_ssd topology.
        E3) run_pair E3 E3C E3 "$workload" "$cache_state" ;;
        E3C) run_pair E3C E3C E3C "$workload" "$cache_state" ;;
        E4) run_pair E4 E3 E4 "$workload" "$cache_state" ;;
        # E5's baseline is E5C: identical admission/prefetch/partial knobs with
        # only the cold external-copy reaper toggled.
        E5) run_pair E5 E5C E5 "$workload" "$cache_state" ;;
        E5C) run_pair E5C E5C E5C "$workload" "$cache_state" ;;
      esac
    done
  done
done

echo "KV-Core controlled suite completed: $OUTPUT_DIR"
