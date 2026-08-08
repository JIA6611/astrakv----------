#!/usr/bin/env bash
set -Eeuo pipefail

# Independent fixed-SLO capacity sweep.  Every point runs the same canonical
# workload with only vLLM's GPU memory budget changed.  E2 is compared with E0;
# E4 is compared with E3 so partial-prefix admission remains the sole feature
# difference inside each point.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-python3}"
WORKLOAD_DIR=""
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
WARM_STORE_DIR=""
OUTPUT_DIR="$ROOT/results/kv-capacity-$(date -u +%Y%m%dT%H%M%SZ)"
PHASE="E2"
WORKLOAD="constrained_kv_churn"
CACHE_STATE="cold"
UTILIZATIONS="0.72,0.68,0.64,0.60"
SLO_TTFT_MS=""
SLO_THROUGHPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload-dir) WORKLOAD_DIR="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --warm-store-dir) WARM_STORE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --cache-state) CACHE_STATE="$2"; shift 2 ;;
    --gpu-memory-utilizations) UTILIZATIONS="$2"; shift 2 ;;
    --slo-ttft-p95-ms) SLO_TTFT_MS="$2"; shift 2 ;;
    --slo-throughput-tokens-s) SLO_THROUGHPUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$PHASE" == E2 || "$PHASE" == E4 ]] || { echo "--phase must be E2 or E4" >&2; exit 2; }
[[ -d "$WORKLOAD_DIR" && -f "$PATCH_MANIFEST" && -f "$CALLBACK_SMOKE" ]] || {
  echo "workload directory and verified patch inputs are required" >&2; exit 2;
}
[[ -n "$SLO_TTFT_MS" && -n "$SLO_THROUGHPUT" ]] || {
  echo "both fixed SLO thresholds are required" >&2; exit 2;
}

mkdir -p "$OUTPUT_DIR"
IFS=',' read -r -a POINTS <<< "$UTILIZATIONS"
BASELINE_ARGS=()
VARIANT_ARGS=()
for utilization in "${POINTS[@]}"; do
  point_dir="$OUTPUT_DIR/gpu-util-${utilization}"
  command=(
    bash scripts/entrypoints/run_kv_core_controlled_suite.sh
    --workload-dir "$WORKLOAD_DIR"
    --output-dir "$point_dir"
    --patch-manifest "$PATCH_MANIFEST"
    --callback-smoke "$CALLBACK_SMOKE"
    --phases "$PHASE"
    --workloads "$WORKLOAD"
    --cache-states "$CACHE_STATE"
    --gpu-memory-utilization "$utilization"
    --allow-ineligible
  )
  if [[ -n "$WARM_STORE_DIR" ]]; then
    command+=(--warm-store-dir "$WARM_STORE_DIR")
  fi
  "${command[@]}"
  BASELINE_ARGS+=(--baseline-run "$point_dir/$PHASE/$WORKLOAD/$CACHE_STATE/baseline")
  VARIANT_ARGS+=(--variant-run "$point_dir/$PHASE/$WORKLOAD/$CACHE_STATE/variant")
done

"$PYTHON" scripts/reporting/build_kv_core_capacity_sweep.py \
  "${BASELINE_ARGS[@]}" "${VARIANT_ARGS[@]}" \
  --phase "$PHASE" \
  --slo-ttft-p95-ms "$SLO_TTFT_MS" \
  --slo-throughput-tokens-s "$SLO_THROUGHPUT" \
  --output "$OUTPUT_DIR/capacity_sweep_manifest.json"

echo "KV-Core capacity sweep completed: $OUTPUT_DIR"
