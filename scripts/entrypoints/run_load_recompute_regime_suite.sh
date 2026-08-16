#!/usr/bin/env bash
set -Eeuo pipefail

# Load-vs-recompute regime matrix (P3c).
#
# Phase 1 runs the two decisive workloads across their arms:
#   repeated_long_prefix   x {E3 full-load, E4 partial, E2R recompute-only}
#   constrained_kv_churn   x {E0 off, E3 full-load, E4 partial, E2R recompute-only}
# Every arm reuses the identical canonical workload file (same request ids),
# so cells pair by sample_id.  The recompute-only arm (E2R) is a per-request
# forced scheduler-decline; E0 is the off-mode TTFT/UMA reference.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
WORKLOAD_DIR=""
OUTPUT_DIR="$ROOT/results/load-recompute-regime-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
PHASE="1"
SMOKE=false
WORKLOADS_FILTER=""
HOST="127.0.0.1"
PORT="18200"
CONTEXT_PORT="18190"
TIMEOUT="900"
GPU_MEMORY_UTILIZATION="0.72"
OUTPUT_TOKENS="${ASTRAKV_REGIME_OUTPUT_TOKENS:-8}"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_load_recompute_regime_suite.sh \
  --workload-dir DIR --patch-manifest FILE --callback-smoke FILE [options]

DIR must contain the canonical repeated_long_prefix.jsonl and
constrained_kv_churn.jsonl (phase 1) or queued_concurrency.jsonl and
random_no_reuse.jsonl (phase 2).
--output-tokens N defaults to 8 for this TTFT/UMA regime matrix; every cell
uses the same fixed value.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload-dir) WORKLOAD_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --workloads) WORKLOADS_FILTER="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$WORKLOAD_DIR" && -d "$WORKLOAD_DIR" ]] || { echo "--workload-dir is required" >&2; exit 2; }
[[ "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "--output-tokens must be positive" >&2; exit 2; }
[[ -f "$PATCH_MANIFEST" && -f "$CALLBACK_SMOKE" ]] || { echo "--patch-manifest and --callback-smoke are required" >&2; exit 2; }
[[ "$PHASE" == "1" || "$PHASE" == "2" ]] || { echo "--phase must be 1 or 2" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "only loopback host is supported" >&2; exit 2; }

if [[ "$PHASE" == "1" ]]; then
  PHASE_WORKLOADS="repeated_long_prefix constrained_kv_churn"
else
  PHASE_WORKLOADS="queued_concurrency random_no_reuse"
fi
if [[ -n "$WORKLOADS_FILTER" ]]; then
  WORKLOADS=""
  IFS=',' read -r -a FILTERED <<< "$WORKLOADS_FILTER"
  for workload in "${FILTERED[@]}"; do
    case " $PHASE_WORKLOADS " in
      *" $workload "*) WORKLOADS="$WORKLOADS $workload" ;;
      *) echo "workload $workload is not in phase $PHASE" >&2; exit 2 ;;
    esac
  done
  WORKLOADS="${WORKLOADS# }"
else
  WORKLOADS="$PHASE_WORKLOADS"
fi
for workload in $WORKLOADS; do
  [[ -f "$WORKLOAD_DIR/$workload.jsonl" ]] || { echo "Missing $WORKLOAD_DIR/$workload.jsonl" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
export ASTRAKV_REGIME_OUTPUT_DIR="$OUTPUT_DIR"
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path.cwd()
paths = [
    root / "astrakv/runtime/vendor_callback_bridge.py",
    root / "astrakv/runtime/lmcache047_runtime_patch.py",
    root / "scripts/benchmark/materialize_smoke_regime_workloads.py",
    root / "scripts/benchmark/materialize_recompute_only_workload.py",
    root / "scripts/reporting/validate_load_recompute_regime_cell.py",
    root / "scripts/reporting/build_load_recompute_regime_report.py",
    root / "scripts/entrypoints/run_kv_core_controlled_suite.sh",
    root / "scripts/entrypoints/run_load_recompute_regime_suite.sh",
]
payload = {
    "schema": "astrakv-load-recompute-regime-runtime-source-v1",
    "files": [
        {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ],
}
(Path(os.environ["ASTRAKV_REGIME_OUTPUT_DIR"]) / "runtime_source_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
if [[ "$SMOKE" == true ]]; then
  SMOKE_WORKLOAD_DIR="$OUTPUT_DIR/smoke-workloads"
  "$PYTHON" scripts/benchmark/materialize_smoke_regime_workloads.py \
    --output-dir "$SMOKE_WORKLOAD_DIR" > "$OUTPUT_DIR/smoke_materialization.json"
  WORKLOAD_DIR="$SMOKE_WORKLOAD_DIR"
  echo "smoke mode: small workloads materialized under $SMOKE_WORKLOAD_DIR"
fi
CELLS_FILE="$OUTPUT_DIR/regime_cells.jsonl"
: > "$CELLS_FILE"

run_cell() {
  local workload="$1" phase="$2" arm="$3"
  local cell_dir="$OUTPUT_DIR/cells/$workload/$arm"
  local cell_workload_dir="$cell_dir/workload"
  local cell_run_dir="$cell_dir/run"
  local label="$workload-$arm"
  mkdir -p "$cell_workload_dir"
  cp "$WORKLOAD_DIR/$workload.jsonl" "$cell_workload_dir/$workload.jsonl"
  if [[ "$phase" == E2R ]]; then
    "$PYTHON" scripts/benchmark/materialize_recompute_only_workload.py \
      --source-workload "$cell_workload_dir/$workload.jsonl" \
      --output-dir "$cell_workload_dir" > "$cell_dir/materialization.json"
  fi
  local extra_args=""
  [[ "$SMOKE" == true ]] && extra_args="--allow-ineligible"
  # This report compares the variant members across cells (full/partial/
  # recompute-only/off).  A same-cell baseline is never consumed, so omit it
  # and halve model starts without weakening the cross-cell comparison.
  bash scripts/entrypoints/run_kv_core_controlled_suite.sh \
    --workload-dir "$cell_workload_dir" --workloads "$workload" --phases "$phase" \
    --output-dir "$cell_run_dir" --patch-manifest "$PATCH_MANIFEST" \
    --callback-smoke "$CALLBACK_SMOKE" --model "$MODEL" --host "$HOST" \
    --port "$PORT" --context-port "$CONTEXT_PORT" --timeout "$TIMEOUT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --output-tokens "$OUTPUT_TOKENS" --variant-only $extra_args
  local baseline_dir="$cell_run_dir/$phase/$workload/cold/baseline"
  local variant_dir="$cell_run_dir/$phase/$workload/cold/variant"
  [[ -d "$variant_dir" ]] || { echo "missing variant cell run: $label" >&2; return 1; }
  if ! "$PYTHON" scripts/reporting/validate_load_recompute_regime_cell.py \
      --run-dir "$variant_dir" --arm "$arm" --phase "$phase" --workload "$workload" \
      --output "$variant_dir/acceptance.json"; then
    [[ "$SMOKE" == true ]] || return 1
  fi
  echo "{\"workload\": \"$workload\", \"arm\": \"$arm\", \"phase\": \"$phase\", \"baseline_dir\": \"\", \"variant_dir\": \"$variant_dir\", \"variant_only\": true, \"output_tokens\": $OUTPUT_TOKENS, \"smoke\": $SMOKE}" >> "$CELLS_FILE"
}

if [[ "$PHASE" == "1" ]]; then
  for workload in $WORKLOADS; do
    case "$workload" in
      repeated_long_prefix)
        run_cell repeated_long_prefix E3 full
        run_cell repeated_long_prefix E4 partial
        run_cell repeated_long_prefix E2R recompute_only
        ;;
      constrained_kv_churn)
        run_cell constrained_kv_churn E0 off
        run_cell constrained_kv_churn E3 full
        run_cell constrained_kv_churn E4 partial
        run_cell constrained_kv_churn E2R recompute_only
        ;;
    esac
  done
else
  for workload in $WORKLOADS; do
    case "$workload" in
      queued_concurrency)
        run_cell queued_concurrency E3 full
        run_cell queued_concurrency E4 partial
        run_cell queued_concurrency E2R recompute_only
        ;;
      random_no_reuse)
        run_cell random_no_reuse E0 off
        run_cell random_no_reuse E3 full
        run_cell random_no_reuse E4 partial
        run_cell random_no_reuse E2R recompute_only
        ;;
    esac
  done
fi

"$PYTHON" scripts/reporting/build_load_recompute_regime_report.py \
  --cells "$CELLS_FILE" --output "$OUTPUT_DIR/load_recompute_regime_report.md"
echo "Load-vs-recompute regime suite completed: $OUTPUT_DIR"
