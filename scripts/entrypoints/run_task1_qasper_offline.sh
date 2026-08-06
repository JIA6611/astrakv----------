#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ZIP=""; WORKLOAD=""; RUN_ID=""; KV_BYTES_PER_TOKEN=""
GPU_BYTES=""; CPU_BYTES=""; SSD_BYTES=""; OUTPUT="results/task1_qasper_offline"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task1-zip) ZIP="$2"; shift 2 ;;
    --task1-workload) WORKLOAD="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2 ;;
    --gpu-capacity-bytes) GPU_BYTES="$2"; shift 2 ;;
    --cpu-capacity-bytes) CPU_BYTES="$2"; shift 2 ;;
    --ssd-capacity-bytes) SSD_BYTES="$2"; shift 2 ;;
    --output-dir) OUTPUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ZIP" && ( "$WORKLOAD" == "random" || "$WORKLOAD" == "grouped" ) && -n "$RUN_ID" && -n "$KV_BYTES_PER_TOKEN" && -n "$GPU_BYTES" && -n "$CPU_BYTES" && -n "$SSD_BYTES" ]] || {
  echo "Required: --task1-zip --task1-workload random|grouped --run-id --kv-bytes-per-token --gpu-capacity-bytes --cpu-capacity-bytes --ssd-capacity-bytes" >&2; exit 2;
}
BASE="$OUTPUT/$WORKLOAD"
python scripts/benchmark/materialize_task1_qasper_workload.py --task1-zip "$ZIP" --task1-workload "$WORKLOAD" --output-dir "$BASE/workload"
MANIFEST="$BASE/workload/qasper_${WORKLOAD}_canonical_workload.jsonl"
python scripts/policy/run_task1_causal_profile_replay.py --workload-manifest "$MANIFEST" --workload-id "qasper_${WORKLOAD}_200" --run-id "$RUN_ID" --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" --gpu-capacity-bytes "$GPU_BYTES" --default-object-bytes "$KV_BYTES_PER_TOKEN" --output-dir "$BASE/causal_profile"
python scripts/policy/run_offline_eviction_simulator.py --workload-manifest "$MANIFEST" --trace "$BASE/causal_profile/causal_logical_trace.jsonl" --profile-db "$BASE/causal_profile/causal_profile_db.json" --workload-id "qasper_${WORKLOAD}_200" --run-id "$RUN_ID" --gpu-capacity-bytes "$GPU_BYTES" --cpu-capacity-bytes "$CPU_BYTES" --ssd-capacity-bytes "$SSD_BYTES" --default-object-bytes "$KV_BYTES_PER_TOKEN" --scheduler-decisions "$BASE/causal_profile/causal_object_schedule_decisions.csv" --profile-source causal_same_workload_history --output-dir "$BASE/offline_eviction"
echo "Task-one offline artifacts written to $BASE"
