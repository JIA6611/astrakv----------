#!/usr/bin/env bash
set -Eeuo pipefail

# evict-B vs LMCache built-in LRU: real-machine two-arm ablation on DGX.
#
# Each repeat runs the grouped exact-next prefetch ablation per dataset:
#   arm-evict-b: online policy + evict-B dispatch enabled (pressure-gated)
#   arm-lru:     online policy + evict dispatch disabled, LMCACHE_CACHE_POLICY=LRU
#
# Capacities are auto-scaled to each dataset's footprint (CPU 15% / SSD 45%)
# so that the evict pressure gate (default 0.8) actually fires and LMCache's
# built-in eviction is exercised in the LRU arm.  All other controls
# (workload, prefetch/sidecar settings, warmup) are identical between arms;
# only the evict decision source differs.
#
# After the runs, per-arm metrics are aggregated and the per-request mainline
# evidence report (ingress prefetch-A -> lookup -> release evict -> prefetch-B)
# is built for every state directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${ASTRAKV_ROOT:-$ROOT}/results/evict-b-vs-lru-$TIMESTAMP"
GROUPED_ROOT=""
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
LIMIT="50"
DATASETS="qasper,multifieldqa_en"
REPEATS="3"
KV_BYTES_PER_TOKEN="1600"
CPU_FRACTION="0.15"
SSD_FRACTION="0.45"
CPU_CAPACITY_BYTES=""
SSD_CAPACITY_BYTES=""
PRESSURE_TRIGGER="0.8"
COLD_SCORE_THRESHOLD="0.35"
GLOBAL_SCAN_INTERVAL_S="5.0"
GLOBAL_SCAN_MAX_VICTIMS="4"
WARMUP_PASSES="1"
WARMUP_LIMIT="8"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}"
EXTERNAL_SIDECAR=""
KV_CORE_MODE="on"
SKIP_AGGREGATE="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_evict_b_vs_lru_suite.sh --grouped-root DIR [options]

Options:
  --grouped-root DIR          Directory containing <dataset>/grouped_prompts.jsonl
  --output-root PATH          Result root (default results/evict-b-vs-lru-<ts>)
  --model PATH                Local model path
  --limit N                   Per-dataset request limit (default 50)
  --datasets LIST             Comma-separated datasets (default qasper,multifieldqa_en)
  --repeats N                 Paired repeats per arm (default 3)
  --kv-bytes-per-token N      KV bytes per token for footprint estimate (default 1600)
  --cpu-fraction N            CPU tier as footprint fraction (default 0.15)
  --ssd-fraction N            SSD tier as footprint fraction (default 0.45)
  --cpu-capacity-bytes N      Explicit CPU tier bytes (overrides fraction/floor)
  --ssd-capacity-bytes N      Explicit SSD tier bytes (overrides fraction/floor)
  --pressure-trigger N        Pressure trigger fraction (default 0.8)
  --cold-score-threshold N    evict coldness score threshold (default 0.35)
  --global-scan-interval-s N  Global evict scan min interval (default 5.0)
  --global-scan-max-victims N Max victims per scan (default 4)
  --warmup-passes N           Warmup passes before the measured run
                              (same server/cache, builds online profile; default 1)
  --warmup-limit N            Warmup request count per pass (default 8)
  --patch-verification PATH   KV-Core connector patch verification JSON
                              (required when --no-kv-core is not set; defaults
                              to $ASTRAKV_KV_CORE_PATCH_VERIFICATION)
  --sidecar-path PATH         Reuse an external sidecar instead of building
  --no-kv-core                Disable KV-core prefetch-A flags
  --skip-aggregate            Skip aggregation and mainline reports
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2 ;;
    --cpu-fraction) CPU_FRACTION="$2"; shift 2 ;;
    --ssd-fraction) SSD_FRACTION="$2"; shift 2 ;;
    --cpu-capacity-bytes) CPU_CAPACITY_BYTES="$2"; shift 2 ;;
    --ssd-capacity-bytes) SSD_CAPACITY_BYTES="$2"; shift 2 ;;
    --pressure-trigger) PRESSURE_TRIGGER="$2"; shift 2 ;;
    --cold-score-threshold) COLD_SCORE_THRESHOLD="$2"; shift 2 ;;
    --global-scan-interval-s) GLOBAL_SCAN_INTERVAL_S="$2"; shift 2 ;;
    --global-scan-max-victims) GLOBAL_SCAN_MAX_VICTIMS="$2"; shift 2 ;;
    --warmup-passes) WARMUP_PASSES="$2"; shift 2 ;;
    --warmup-limit) WARMUP_LIMIT="$2"; shift 2 ;;
    --patch-verification) PATCH_VERIFICATION="$2"; shift 2 ;;
    --sidecar-path) EXTERNAL_SIDECAR="$2"; shift 2 ;;
    --no-kv-core) KV_CORE_MODE="off"; shift ;;
    --skip-aggregate) SKIP_AGGREGATE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GROUPED_ROOT" && -d "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be a positive integer" >&2; exit 2; }
IFS=',' read -r -a SELECTED_DATASETS <<< "$DATASETS"
[[ "${#SELECTED_DATASETS[@]}" -gt 0 ]] || { echo "--datasets must not be empty" >&2; exit 2; }
for dataset in "${SELECTED_DATASETS[@]}"; do
  [[ -f "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" ]] || {
    echo "missing $GROUPED_ROOT/$dataset/grouped_prompts.jsonl" >&2; exit 2; }
done

SIDECAR_ARGS=()
if [[ -n "$EXTERNAL_SIDECAR" ]]; then
  SIDECAR_ARGS+=(--sidecar-path "$EXTERNAL_SIDECAR")
fi

KV_CORE_ENV=()
if [[ "$KV_CORE_MODE" == "on" ]]; then
  [[ -n "$PATCH_VERIFICATION" ]] || {
    echo "KV-Core active mode requires --patch-verification (or ASTRAKV_KV_CORE_PATCH_VERIFICATION)" >&2
    exit 2
  }
  [[ -f "$PATCH_VERIFICATION" ]] || {
    echo "patch verification file missing: $PATCH_VERIFICATION" >&2
    exit 2
  }
  KV_CORE_ENV+=(
    "ASTRAKV_KV_CORE_MODE=active"
    "ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd"
    "ASTRAKV_KV_CORE_LOCAL_CPU=true"
    "ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true"
    "ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true"
    "ASTRAKV_KV_CORE_PATCH_VERIFICATION=$PATCH_VERIFICATION"
  )
fi

materialize_dataset() {
  local dataset="$1"
  local prepare_dir="$OUTPUT_ROOT/prepare/$dataset"
  mkdir -p "$prepare_dir"
  "$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
    --grouped-prompts-jsonl "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" \
    --output-dir "$prepare_dir" \
    --dataset "$dataset" \
    --task "$dataset" \
    --limit "$LIMIT" >&2
  printf '%s\n' "$prepare_dir/${dataset}_grouped_exact_next_canonical_workload.jsonl"
}

footprint_bytes() {
  local canonical="$1"
  "$PYTHON" - "$canonical" "$KV_BYTES_PER_TOKEN" <<'PY'
import json, sys
total = 0
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    total += int(row.get("context_length") or 0) * int(sys.argv[2])
print(total)
PY
}

run_arm() {
  local arm_name="$1" evict_dispatch="$2" dataset="$3" rep="$4" \
        cpu_bytes="$5" ssd_bytes="$6"
  local arm_out="$OUTPUT_ROOT/$arm_name/rep-$rep"
  local cpu_gb ssd_gb
  cpu_gb="$("$PYTHON" -c "print(round($cpu_bytes / 2**30, 3))")"
  ssd_gb="$("$PYTHON" -c "print(round($ssd_bytes / 2**30, 3))")"
  echo "=== [$arm_name] rep $rep dataset=$dataset evict_dispatch=$evict_dispatch cpu=${cpu_gb}GB ssd=${ssd_gb}GB ==="
  # launch_vllm_server.sh defaults VENDOR_PATCH to true; legacy-hooks mode
  # (evict-B online policy) must explicitly disable it so hook events flow.
  local vendor_patch_env="ASTRAKV_KV_CORE_VENDOR_PATCH=false"
  local -a scope_env=()
  if [[ "$KV_CORE_MODE" == "on" ]]; then
    vendor_patch_env="ASTRAKV_KV_CORE_VENDOR_PATCH=true"
  else
    # Legacy-hooks mode needs install_from_environment() to run in the vLLM
    # EngineCore process only.  prepend scripts/runtime (sitecustomize) and
    # scope host creation to the engine child.
    scope_env+=(
      "ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE=engine_child"
      "PYTHONPATH=$ROOT/scripts/runtime${PYTHONPATH:+:$PYTHONPATH}"
      "ASTRAKV_EVICT_DISPATCH_INDEPENDENT_OF_MODE=true"
    )
  fi
  local -a periodic_env=()
  if [[ "$evict_dispatch" == "true" ]]; then
    # evict-B mirrors LMCache's native watermark loop: periodic pressure scan.
    periodic_env+=(
      "ASTRAKV_EVICT_PERIODIC_SCAN_ENABLED=true"
      "ASTRAKV_EVICT_PERIODIC_SCAN_INTERVAL_S=1.0"
      "ASTRAKV_EVICT_GLOBAL_SCAN_MIN_INTERVAL_S=1.0"
    )
  fi
  env \
    "ASTRAKV_ENABLE_ONLINE_POLICY=true" \
    "ASTRAKV_ONLINE_EVICT_DISPATCH_ENABLED=$evict_dispatch" \
    "ASTRAKV_EVICT_PRESSURE_GATE_ENABLED=true" \
    "ASTRAKV_EVICT_PRESSURE_TRIGGER=$PRESSURE_TRIGGER" \
    "ASTRAKV_EVICT_CPU_CAPACITY_BYTES=$cpu_bytes" \
    "ASTRAKV_EVICT_SSD_CAPACITY_BYTES=$ssd_bytes" \
    "ASTRAKV_EVICT_COLD_SCORE_THRESHOLD=$COLD_SCORE_THRESHOLD" \
    "ASTRAKV_EVICT_GLOBAL_SCAN_ENABLED=true" \
    "ASTRAKV_EVICT_GLOBAL_SCAN_MIN_INTERVAL_S=$GLOBAL_SCAN_INTERVAL_S" \
    "ASTRAKV_EVICT_GLOBAL_SCAN_MAX_VICTIMS=$GLOBAL_SCAN_MAX_VICTIMS" \
    "${periodic_env[@]}" \
    "$vendor_patch_env" \
    "${scope_env[@]}" \
    "ASTRAKV_ABLATION_WARMUP_PASSES=$WARMUP_PASSES" \
    "ASTRAKV_ABLATION_WARMUP_LIMIT=$WARMUP_LIMIT" \
    "ASTRAKV_LOCAL_CPU_SIZE_GB=$cpu_gb" \
    "ASTRAKV_LOCAL_DISK_SIZE_GB=$ssd_gb" \
    "LMCACHE_CACHE_POLICY=LRU" \
    "${KV_CORE_ENV[@]}" \
    bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
      --grouped-root "$GROUPED_ROOT" \
      --model "$MODEL" \
      --limit "$LIMIT" \
      --datasets "$dataset" \
      --output-dir "$arm_out" \
      "${SIDECAR_ARGS[@]}"
}

mkdir -p "$OUTPUT_ROOT"

for dataset in "${SELECTED_DATASETS[@]}"; do
  canonical="$(materialize_dataset "$dataset")"
  footprint="$(footprint_bytes "$canonical")"
  if [[ -n "$CPU_CAPACITY_BYTES" ]]; then
    cpu_bytes="$CPU_CAPACITY_BYTES"
  else
    # Floor at 128 MiB so the pool can hold several KV chunks (Qwen3-8B chunk
    # ~36 MiB); a pool smaller than one chunk makes LMCache store 0 chunks.
    cpu_bytes="$("$PYTHON" -c "print(max(int($footprint * $CPU_FRACTION), 134217728))")"
  fi
  if [[ -n "$SSD_CAPACITY_BYTES" ]]; then
    ssd_bytes="$SSD_CAPACITY_BYTES"
  else
    ssd_bytes="$("$PYTHON" -c "print(max(int($footprint * $SSD_FRACTION), 134217728))")"
  fi
  echo "[$dataset] footprint=$footprint cpu_bytes=$cpu_bytes ssd_bytes=$ssd_bytes"
  for rep in $(seq 1 "$REPEATS"); do
    run_arm "arm-evict-b" "true" "$dataset" "$rep" "$cpu_bytes" "$ssd_bytes"
    run_arm "arm-lru" "false" "$dataset" "$rep" "$cpu_bytes" "$ssd_bytes"
  done
done

if [[ "$SKIP_AGGREGATE" == "true" ]]; then
  echo "evict-B vs LRU suite runs completed (aggregation skipped): $OUTPUT_ROOT"
  exit 0
fi

for arm in arm-evict-b arm-lru; do
  for rep in $(seq 1 "$REPEATS"); do
    # Artifacts (receipts/commands/tickets) live in the run dir (baseline|variant),
    # not the *-state dir; the ablation copies them over during run_condition.
    for run_dir in "$OUTPUT_ROOT/$arm/rep-$rep"/*/{baseline,variant}; do
      [[ -d "$run_dir" ]] || continue
      "$PYTHON" scripts/reporting/build_evict_mainline_report.py --state-dir "$run_dir" >/dev/null || true
    done
  done
  "$PYTHON" scripts/reporting/aggregate_evict_ablation.py \
    --arm-root "$OUTPUT_ROOT/$arm" \
    --output "$OUTPUT_ROOT/$arm/arm_metrics.json" 2>/dev/null || true
done

echo ""
echo "evict-B vs LRU suite completed: $OUTPUT_ROOT"
echo "Arm metrics:"
for arm in arm-evict-b arm-lru; do
  if [[ -f "$OUTPUT_ROOT/$arm/arm_metrics.json" ]]; then
    echo "--- $arm ---"
    python3 -c "import json,sys; d=json.load(open('$OUTPUT_ROOT/$arm/arm_metrics.json', encoding='utf-8'))['merged']; print(json.dumps(d, ensure_ascii=False, indent=2))"
  fi
done
