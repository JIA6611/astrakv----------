#!/usr/bin/env bash
set -Eeuo pipefail

# Offline eviction layer on KV-core canonical workloads (interleaved reuse).
#
# The grouped exact-next workloads have reuse distance 1 (adjacent revisits),
# so eviction choice never affects hits.  The KV-core workloads
# (repeated_long_prefix / constrained_kv_churn / random_no_reuse /
# queued_concurrency) spread revisits across the trace and therefore exercise
# eviction policy differences: LRU / FIFO / AstraKV / Belady on identical
# tier capacities (auto-scaled to footprint fractions so eviction fires).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${ASTRAKV_ROOT:-$ROOT}/results/kv-core-offline-eviction-$TIMESTAMP"
WORKLOADS_DIR="$ROOT/results/kv-core-workloads"
WORKLOADS="repeated_long_prefix,constrained_kv_churn,random_no_reuse,queued_concurrency"
RUN_ID="kv-core-offline-$TIMESTAMP"
KV_BYTES_PER_TOKEN="1600"
GPU_FRACTION="0.05"
CPU_FRACTION="0.20"
SSD_FRACTION="0.55"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_kv_core_offline_eviction_suite.sh [options]

Options:
  --workloads-dir PATH        Directory with canonical workload jsonl files
                              (default results/kv-core-workloads)
  --workloads LIST            Comma-separated workload names
                              (default repeated_long_prefix,constrained_kv_churn,random_no_reuse,queued_concurrency)
  --output-dir PATH           Result root
  --run-id ID                 Run id recorded in artifacts
  --kv-bytes-per-token N      Estimated KV bytes per token (default 1600)
  --capacity-fractions G,C,S  Tier capacities as footprint fractions
                              (default 0.05,0.20,0.55)
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workloads-dir) WORKLOADS_DIR="$2"; shift 2 ;;
    --workloads) WORKLOADS="$2"; shift 2 ;;
    --output-dir) OUTPUT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2 ;;
    --capacity-fractions) IFS=',' read -r GPU_FRACTION CPU_FRACTION SSD_FRACTION <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$WORKLOADS_DIR" ]] || { echo "--workloads-dir missing: $WORKLOADS_DIR" >&2; exit 2; }
IFS=',' read -r -a SELECTED_WORKLOADS <<< "$WORKLOADS"
for workload in "${SELECTED_WORKLOADS[@]}"; do
  [[ -f "$WORKLOADS_DIR/$workload.jsonl" ]] || {
    echo "missing $WORKLOADS_DIR/$workload.jsonl" >&2; exit 2; }
done

mkdir -p "$OUTPUT"

for workload in "${SELECTED_WORKLOADS[@]}"; do
  base="$OUTPUT/$workload"
  mkdir -p "$base"
  canonical="$WORKLOADS_DIR/$workload.jsonl"
  total="$("$PYTHON" - "$canonical" "$KV_BYTES_PER_TOKEN" <<'PY'
import json, sys
total = 0
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    total += int(row.get("context_length") or 0) * int(sys.argv[2])
print(total)
PY
)"
  gpu_bytes="$("$PYTHON" -c "print(int($total * $GPU_FRACTION))")"
  cpu_bytes="$("$PYTHON" -c "print(int($total * $CPU_FRACTION))")"
  ssd_bytes="$("$PYTHON" -c "print(int($total * $SSD_FRACTION))")"
  echo "[$workload] footprint=$total gpu=$gpu_bytes cpu=$cpu_bytes ssd=$ssd_bytes"
  "$PYTHON" scripts/policy/run_task1_causal_profile_replay.py \
    --workload-manifest "$canonical" \
    --workload-id "$workload" \
    --run-id "$RUN_ID" \
    --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" \
    --gpu-capacity-bytes "$gpu_bytes" \
    --default-object-bytes "$KV_BYTES_PER_TOKEN" \
    --output-dir "$base/causal_profile"
  "$PYTHON" scripts/policy/run_offline_eviction_simulator.py \
    --workload-manifest "$canonical" \
    --trace "$base/causal_profile/causal_logical_trace.jsonl" \
    --profile-db "$base/causal_profile/causal_profile_db.json" \
    --workload-id "$workload" \
    --run-id "$RUN_ID" \
    --gpu-capacity-bytes "$gpu_bytes" \
    --cpu-capacity-bytes "$cpu_bytes" \
    --ssd-capacity-bytes "$ssd_bytes" \
    --default-object-bytes "$KV_BYTES_PER_TOKEN" \
    --scheduler-decisions "$base/causal_profile/causal_object_schedule_decisions.csv" \
    --profile-source causal_same_workload_history \
    --output-dir "$base/offline_eviction"
done

echo "KV-core offline eviction suite completed: $OUTPUT"
find "$OUTPUT" -name "offline_eviction_manifest.json" -print
