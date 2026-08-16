#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv2/results_full/workload_prompts}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/connector_patch_verification.json}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${ASTRAKV_ROOT:-$ROOT}/results/e11-slo-demo-$TIMESTAMP"
SLO_MS="1600"
LIMIT="9"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_e11_slo_demo.sh [options]

Runs one real Qasper scan-pollution pair (LRU then evict-B), then prints only
the TTFT-P50 comparison in the terminal. Expected duration is about 4-6 min.

Options:
  --grouped-root DIR          Source containing qasper/grouped_prompts.jsonl
  --output-root DIR           Result directory
  --model DIR                 Local Qwen3-8B model directory
  --patch-verification FILE   Compatible KV-Core patch verification JSON
  --slo-ms N                  Preselected TTFT P50 target (default 1600)
  --limit N                   Selected request limit (default 9)
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --patch-verification) PATCH_VERIFICATION="$2"; shift 2 ;;
    --slo-ms) SLO_MS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" ]] || { echo "qasper grouped workload missing under: $GROUPED_ROOT" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory missing: $MODEL" >&2; exit 2; }
[[ -f "$PATCH_VERIFICATION" ]] || { echo "patch verification missing: $PATCH_VERIFICATION" >&2; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "--limit must be positive" >&2; exit 2; }
"$PYTHON" -c "value=float('$SLO_MS'); assert value > 0" || { echo "--slo-ms must be positive" >&2; exit 2; }

started="$(date +%s)"
mkdir -p "$OUTPUT_ROOT"
SUITE_LOG="$OUTPUT_ROOT/suite.log"
echo "Running real LRU vs evict-B demo (about 4-6 minutes)..."

if ! bash scripts/entrypoints/run_evict_b_vs_lru_suite.sh \
  --grouped-root "$GROUPED_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model "$MODEL" \
  --patch-verification "$PATCH_VERIFICATION" \
  --datasets qasper \
  --regimes scan_pollution_past_observed \
  --repeats 1 \
  --limit "$LIMIT" \
  --output-tokens 1 \
  --warmup-passes 0 \
  --skip-aggregate >"$SUITE_LOG" 2>&1; then
  echo "Demo failed. Last 60 log lines:" >&2
  tail -n 60 "$SUITE_LOG" >&2
  exit 1
fi

"$PYTHON" scripts/reporting/build_e11_slo_demo.py \
  --root "$OUTPUT_ROOT" \
  --slo-ms "$SLO_MS"

elapsed="$(( $(date +%s) - started ))"
if (( elapsed > 360 )); then
  echo "NOTE: runtime exceeded the 6-minute target; details are in $SUITE_LOG" >&2
fi
