#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="auto"
WORKLOAD=""
TASK1_ZIP=""
TASK1_WORKLOAD=""
RUN_ID=""
OFFLINE_DECISIONS=""
CACHE_EVENTS=""
REQUEST_RESULTS=""
SERVER_LOG=""
BACKING_FILE=""
OFFLINE_SAFETY_GATE=""
STRUCTURED_EVENTS=""
STRUCTURED_HOOK_VERIFICATION=""
MODEL_REVISION="unknown"
TOKENIZER_REVISION="unknown"
DTYPE="unknown"
QUANTIZATION="unknown"
CACHE_STATE="unknown"
OUTPUT_DIR="results/runtime_eviction_validation"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_runtime_eviction_validation.sh (--workload-jsonl PATH | --task1-zip PATH --task1-workload random|grouped) --run-id ID [options]
  --task1-zip PATH --task1-workload random|grouped
  --mode local|dgx|auto
  --offline-decisions PATH
  --cache-events PATH
  --request-results PATH
  --server-log PATH
  --backing-file PATH
  --offline-safety-gate PATH
  --structured-events PATH
  --structured-hook-verification PATH
  --output-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --workload-jsonl) WORKLOAD="$2"; shift 2 ;;
    --task1-zip) TASK1_ZIP="$2"; shift 2 ;;
    --task1-workload) TASK1_WORKLOAD="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --offline-decisions) OFFLINE_DECISIONS="$2"; shift 2 ;;
    --cache-events) CACHE_EVENTS="$2"; shift 2 ;;
    --request-results) REQUEST_RESULTS="$2"; shift 2 ;;
    --server-log) SERVER_LOG="$2"; shift 2 ;;
    --backing-file) BACKING_FILE="$2"; shift 2 ;;
    --offline-safety-gate) OFFLINE_SAFETY_GATE="$2"; shift 2 ;;
    --structured-events) STRUCTURED_EVENTS="$2"; shift 2 ;;
    --structured-hook-verification) STRUCTURED_HOOK_VERIFICATION="$2"; shift 2 ;;
    --model-revision) MODEL_REVISION="$2"; shift 2 ;;
    --tokenizer-revision) TOKENIZER_REVISION="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --quantization) QUANTIZATION="$2"; shift 2 ;;
    --cache-state) CACHE_STATE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" ]] || { usage >&2; exit 2; }
if [[ -n "$TASK1_ZIP" || -n "$TASK1_WORKLOAD" ]]; then
  [[ -n "$TASK1_ZIP" && ( "$TASK1_WORKLOAD" == "random" || "$TASK1_WORKLOAD" == "grouped" ) ]] || { echo "--task1-zip requires --task1-workload random|grouped" >&2; exit 2; }
  mkdir -p "$OUTPUT_DIR/task1_workload"
  python scripts/benchmark/materialize_task1_qasper_workload.py --task1-zip "$TASK1_ZIP" --task1-workload "$TASK1_WORKLOAD" --output-dir "$OUTPUT_DIR/task1_workload"
  WORKLOAD="$OUTPUT_DIR/task1_workload/qasper_${TASK1_WORKLOAD}_canonical_workload.jsonl"
fi
[[ -n "$WORKLOAD" ]] || { usage >&2; exit 2; }
case "$MODE" in local|dgx|auto) ;; *) echo "--mode must be local, dgx, or auto" >&2; exit 2 ;; esac
if [[ "$MODE" == "auto" ]]; then
  if [[ "$(uname -s)" == "Linux" && -x "$(command -v nvidia-smi || true)" ]]; then MODE="dgx"; else MODE="local"; fi
fi

mkdir -p "$OUTPUT_DIR"
python scripts/benchmark/validate_runtime_workload.py --workload-jsonl "$WORKLOAD" --output "$OUTPUT_DIR/workload_validation.json"
if [[ -n "$STRUCTURED_EVENTS" ]]; then
  verification_output="${STRUCTURED_HOOK_VERIFICATION:-$OUTPUT_DIR/structured_hook_verification.json}"
  set +e
  python scripts/benchmark/verify_structured_eviction_hook.py --events "$STRUCTURED_EVENTS" --run-id "$RUN_ID" --output "$verification_output"
  structured_status=$?
  set -e
  STRUCTURED_HOOK_VERIFICATION="$verification_output"
else
  structured_status=2
fi
python scripts/benchmark/inspect_dgx_runtime.py --output "$OUTPUT_DIR/runtime_environment.json" --model "${ASTRAKV_MODEL:-}" --run-id "$RUN_ID" --workload-manifest "$WORKLOAD" --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" --dtype "$DTYPE" --quantization "$QUANTIZATION" --cache-state "$CACHE_STATE" --structured-hook-verification "$STRUCTURED_HOOK_VERIFICATION" --lmcache-config "${LMCACHE_CONFIG_FILE:-}" --disk-path "$OUTPUT_DIR"
cp "$WORKLOAD" "$OUTPUT_DIR/workload.jsonl"

python - "$OUTPUT_DIR/run_status.json" "$MODE" "$RUN_ID" <<'PY'
import json, sys
from datetime import datetime
path, mode, run_id = sys.argv[1:]
json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"), "mode": mode, "run_id": run_id,
           "runtime_agreement_status": "not_run", "vm_poc_status": "not_run"}, open(path, "w", encoding="utf-8"), indent=2)
PY

if [[ -n "$CACHE_EVENTS" || -n "$SERVER_LOG" || -n "$STRUCTURED_EVENTS" ]]; then
  normalizer=(python scripts/reporting/normalize_runtime_eviction_events.py --run-id "$RUN_ID" --workload-manifest "$WORKLOAD" --output "$OUTPUT_DIR/runtime_eviction_events.jsonl")
  [[ -n "$CACHE_EVENTS" ]] && normalizer+=(--cache-events "$CACHE_EVENTS")
  [[ -n "$SERVER_LOG" ]] && normalizer+=(--server-log "$SERVER_LOG")
  [[ -n "$REQUEST_RESULTS" ]] && normalizer+=(--request-results "$REQUEST_RESULTS")
  if [[ -n "$STRUCTURED_EVENTS" && "$structured_status" -eq 0 ]]; then
    normalizer+=(--structured-events "$STRUCTURED_EVENTS" --structured-hook-verification "$STRUCTURED_HOOK_VERIFICATION")
  fi
  "${normalizer[@]}"
fi

if [[ -n "$OFFLINE_DECISIONS" && -f "$OUTPUT_DIR/runtime_eviction_events.jsonl" ]]; then
  set +e
  python scripts/reporting/compare_offline_runtime_eviction.py --offline-decisions "$OFFLINE_DECISIONS" \
    --runtime-events "$OUTPUT_DIR/runtime_eviction_events.jsonl" --workload-manifest "$WORKLOAD" --run-id "$RUN_ID" \
    --comparison-scope runtime --output-dir "$OUTPUT_DIR/runtime_agreement"
  runtime_status=$?
  set -e
  python - "$OUTPUT_DIR/run_status.json" "$runtime_status" <<'PY'
import json, sys
path, status = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
data["runtime_agreement_status"] = "valid" if status == "0" else "insufficient_ground_truth"
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
fi

if [[ "$MODE" == "dgx" && -n "$OFFLINE_DECISIONS" && -n "$OFFLINE_SAFETY_GATE" ]]; then
  set +e
  python scripts/vm/run_workload_mmap_eviction.py --offline-decisions "$OFFLINE_DECISIONS" --workload-manifest "$WORKLOAD" \
    --run-id "$RUN_ID" --backing-file "${BACKING_FILE:-$OUTPUT_DIR/mmap_kv.bin}" --offline-safety-gate "$OFFLINE_SAFETY_GATE" --output "$OUTPUT_DIR/mmap_eviction_events.jsonl"
  vm_status=$?
  set -e
  if [[ "$vm_status" -eq 0 ]]; then
    set +e
    python scripts/reporting/compare_offline_runtime_eviction.py --offline-decisions "$OFFLINE_DECISIONS" \
      --runtime-events "$OUTPUT_DIR/mmap_eviction_events.jsonl" --workload-manifest "$WORKLOAD" --run-id "$RUN_ID" \
      --comparison-scope vm_poc --output-dir "$OUTPUT_DIR/vm_poc_agreement"
    set -e
    vm_value="executed"
  else
    vm_value="unsupported_or_failed"
  fi
  python - "$OUTPUT_DIR/run_status.json" "$vm_value" <<'PY'
import json, sys
path, value = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
data["vm_poc_status"] = value
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
fi

if [[ "$MODE" == "dgx" && -n "$OFFLINE_DECISIONS" && -z "$OFFLINE_SAFETY_GATE" ]]; then
  echo "Skipping VM-PoC actions: --offline-safety-gate is required" >&2
fi

echo "Runtime eviction validation artifacts written to $OUTPUT_DIR"
