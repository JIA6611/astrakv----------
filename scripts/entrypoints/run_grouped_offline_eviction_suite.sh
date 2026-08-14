#!/usr/bin/env bash
set -Eeuo pipefail

# Offline eviction layer for the evict-B experiment: replay the same grouped
# exact-next workload under LRU / FIFO / AstraKV / Belady with identical
# capacities.  Belady is explicitly labeled an offline oracle (future access).
#
# AstraKV is the offline stand-in for evict-B (profile-guided victim choice);
# the gap (AstraKV - LRU) is the profile-gain estimate and
# (Belady - AstraKV) is the remaining headroom.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${ASTRAKV_ROOT:-$ROOT}/results/grouped-offline-eviction-$TIMESTAMP"
GROUPED_ROOT=""
RUN_ID="grouped-offline-$TIMESTAMP"
KV_BYTES_PER_TOKEN="1600"
GPU_CAPACITY_BYTES="2147483648"
CPU_CAPACITY_BYTES="5368709120"
SSD_CAPACITY_BYTES="85899345920"
GPU_FRACTION="0.05"
CPU_FRACTION="0.20"
SSD_FRACTION="0.55"
AUTO_CAPACITY="true"
DATASETS="qasper,multifieldqa_en"
LIMIT="50"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_grouped_offline_eviction_suite.sh --grouped-root DIR [options]

Options:
  --grouped-root DIR          Directory containing <dataset>/grouped_prompts.jsonl
  --output-dir PATH           Result root (default results/grouped-offline-eviction-<ts>)
  --run-id ID                 Run id recorded in artifacts
  --kv-bytes-per-token N      Estimated KV bytes per token (default 1600)
  --gpu-capacity-bytes N      GPU tier bytes (default 2 GiB)
  --cpu-capacity-bytes N      CPU tier bytes (default 5 GiB)
  --ssd-capacity-bytes N      SSD tier bytes (default 80 GiB)
  --auto-capacity             Scale tier capacities to fractions of the
                              workload footprint so eviction actually fires
                              (default on; explicit --*-capacity-bytes wins)
  --no-auto-capacity          Use explicit capacity bytes only
  --capacity-fractions G,C,S  Footprint fractions (default 0.05,0.20,0.55)
  --datasets LIST             Comma-separated datasets (default qasper,multifieldqa_en)
  --limit N                   Per-dataset request limit (default 50)
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2 ;;
    --gpu-capacity-bytes) GPU_CAPACITY_BYTES="$2"; shift 2 ;;
    --cpu-capacity-bytes) CPU_CAPACITY_BYTES="$2"; shift 2 ;;
    --ssd-capacity-bytes) SSD_CAPACITY_BYTES="$2"; shift 2 ;;
    --auto-capacity) AUTO_CAPACITY="true"; shift ;;
    --no-auto-capacity) AUTO_CAPACITY="false"; shift ;;
    --capacity-fractions) IFS=',' read -r GPU_FRACTION CPU_FRACTION SSD_FRACTION <<< "$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GROUPED_ROOT" && -d "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
[[ "$KV_BYTES_PER_TOKEN" =~ ^[0-9]+$ ]] || { echo "--kv-bytes-per-token must be an integer" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
IFS=',' read -r -a SELECTED_DATASETS <<< "$DATASETS"
for dataset in "${SELECTED_DATASETS[@]}"; do
  [[ -f "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" ]] || {
    echo "missing $GROUPED_ROOT/$dataset/grouped_prompts.jsonl" >&2; exit 2; }
done

mkdir -p "$OUTPUT"

compute_footprint_capacities() {
  local canonical="$1"
  local total
  total="$("$PYTHON" - "$canonical" "$KV_BYTES_PER_TOKEN" <<'PY'
import json, sys
total = 0
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    total += int(row.get("context_length") or 0) * int(sys.argv[2])
print(total)
PY
)"
  GPU_CAPACITY_BYTES="$("$PYTHON" -c "print(int($total * $GPU_FRACTION))")"
  CPU_CAPACITY_BYTES="$("$PYTHON" -c "print(int($total * $CPU_FRACTION))")"
  SSD_CAPACITY_BYTES="$("$PYTHON" -c "print(int($total * $SSD_FRACTION))")"
  echo "  auto capacities (footprint=$total): gpu=$GPU_CAPACITY_BYTES cpu=$CPU_CAPACITY_BYTES ssd=$SSD_CAPACITY_BYTES"
}

for dataset in "${SELECTED_DATASETS[@]}"; do
  base="$OUTPUT/$dataset"
  mkdir -p "$base/materialized"
  "$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
    --grouped-prompts-jsonl "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" \
    --output-dir "$base/materialized" \
    --dataset "$dataset" \
    --task "$dataset" \
    --limit "$LIMIT"
  canonical="$base/materialized/${dataset}_grouped_exact_next_canonical_workload.jsonl"
  if [[ "$AUTO_CAPACITY" == "true" ]]; then
    compute_footprint_capacities "$canonical"
  fi
  "$PYTHON" scripts/policy/run_task1_causal_profile_replay.py \
    --workload-manifest "$canonical" \
    --workload-id "${dataset}_grouped_exact_next" \
    --run-id "$RUN_ID" \
    --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" \
    --gpu-capacity-bytes "$GPU_CAPACITY_BYTES" \
    --default-object-bytes "$KV_BYTES_PER_TOKEN" \
    --output-dir "$base/causal_profile"
  "$PYTHON" scripts/policy/run_offline_eviction_simulator.py \
    --workload-manifest "$canonical" \
    --trace "$base/causal_profile/causal_logical_trace.jsonl" \
    --profile-db "$base/causal_profile/causal_profile_db.json" \
    --workload-id "${dataset}_grouped_exact_next" \
    --run-id "$RUN_ID" \
    --gpu-capacity-bytes "$GPU_CAPACITY_BYTES" \
    --cpu-capacity-bytes "$CPU_CAPACITY_BYTES" \
    --ssd-capacity-bytes "$SSD_CAPACITY_BYTES" \
    --default-object-bytes "$KV_BYTES_PER_TOKEN" \
    --scheduler-decisions "$base/causal_profile/causal_object_schedule_decisions.csv" \
    --profile-source causal_same_workload_history \
    --output-dir "$base/offline_eviction"
done

echo "Grouped offline eviction suite completed: $OUTPUT"
find "$OUTPUT" -name "offline_eviction_manifest.json" -print
