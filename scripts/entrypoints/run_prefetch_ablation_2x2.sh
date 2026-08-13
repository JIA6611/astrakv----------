#!/usr/bin/env bash
set -Eeuo pipefail

# Run the Prefetch-A / Prefetch-B 2x2 ablation in one invocation.
#
# The existing grouped exact-next ablation script toggles B
# (ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH) per arm and leaves A off by
# default.  This wrapper runs that script twice -- once with the KV-Core A
# flags absent, once with them exported -- producing all four cells:
#
#   run 1: (A off, B off) + (A off, B on)
#   run 2: (A on,  B off) + (A on,  B on)
#
# It then aggregates receipts/tickets/decisions per cell with
# scripts/reporting/validate_prefetch_2x2_ablation.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GROUPED_ROOT=""
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
OUTPUT_ROOT="${ASTRAKV_ROOT:-$ROOT}/results/prefetch-ablation-2x2-$TIMESTAMP"
LIMIT="50"
SKIP_VALIDATE="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_prefetch_ablation_2x2.sh --grouped-root DIR [options]

Options:
  --grouped-root DIR   Directory containing <dataset>/grouped_prompts.jsonl
                       (qasper, multifieldqa_en). Required.
  --model PATH         Local model path (default $ASTRAKV_MODEL or repo models).
  --output-root PATH   Root for the two ablation runs (default results/...).
  --limit N            Per-dataset request limit for both runs (default 50).
  --skip-validate      Skip the final 4-cell aggregation step.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip-validate) SKIP_VALIDATE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GROUPED_ROOT" && -d "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
for dataset in qasper multifieldqa_en; do
  [[ -f "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" ]] || {
    echo "missing $GROUPED_ROOT/$dataset/grouped_prompts.jsonl" >&2
    exit 2
  }
done

AOFF_DIR="$OUTPUT_ROOT/a-off"
AON_DIR="$OUTPUT_ROOT/a-on"

echo "=== [2x2] Run 1/2: A off (cells: A0B0, A0B1) ==="
bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --output-dir "$AOFF_DIR"

echo "=== [2x2] Run 2/2: A on (cells: A1B0, A1B1) ==="
ASTRAKV_KV_CORE_MODE=active ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd \
ASTRAKV_KV_CORE_LOCAL_CPU=true ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true \
ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --output-dir "$AON_DIR"

if [[ "$SKIP_VALIDATE" == "true" ]]; then
  echo "2x2 ablation runs completed (validation skipped): $OUTPUT_ROOT"
  exit 0
fi

PYTHON="${ASTRAKV_PYTHON:-python3}"
"$PYTHON" scripts/reporting/validate_prefetch_2x2_ablation.py \
  --a-off "$AOFF_DIR" --a-on "$AON_DIR" \
  --output "$OUTPUT_ROOT/prefetch_2x2_validation.json"

echo "2x2 ablation completed: $OUTPUT_ROOT"
echo "Validation summary: $OUTPUT_ROOT/prefetch_2x2_validation.json"
