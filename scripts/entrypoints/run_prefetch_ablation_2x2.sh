#!/usr/bin/env bash
set -Eeuo pipefail

# Run the Prefetch-A / Prefetch-B ablation in one invocation.
#
# The existing grouped exact-next ablation script toggles B
# (ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH) per arm and leaves A off by
# default.  This wrapper runs that script twice -- once with the KV-Core mode
# active but A's CPU prefetch flag off, once with A's flag exported --
# producing the three cells we report on (the A+B combined cell is
# intentionally NOT run):
#
#   run 1: (A off, B off)  -> pure baseline   (mode stays off; prefetch uses
#                            its independent dispatch channel)
#          (A off, B on)   -> B-only
#   run 2: (A on,  B off)  -> A-only   (variant/both skipped)
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
DATASETS="qasper,multifieldqa_en"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}"
INTERLEAVE="false"
PREFETCH_LEAD_S="${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-0.25}"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_prefetch_ablation_2x2.sh --grouped-root DIR [options]

Options:
  --grouped-root DIR   Directory containing <dataset>/grouped_prompts.jsonl
                       (qasper, multifieldqa_en). Required.
  --model PATH         Local model path (default $ASTRAKV_MODEL or repo models).
  --output-root PATH   Root for the two ablation runs (default results/...).
  --limit N            Per-dataset request limit for both runs (default 50).
  --datasets LIST      Comma-separated datasets (default qasper,multifieldqa_en).
  --patch-verification PATH  connector_patch_verification.json for active (A-on) servers.
  --prefetch-lead-s SECONDS  Lead window before each HTTP submit; with the
                             invalidate-on-lead flag Prefetch-A frees then
                             repopulates the request's CPU copy from SSD.
  --interleave               Round-robin reuse groups so A/B can actually fire
                             (objects become SSD-resident-but-CPU-evicted between
                             visits).  Without it every object's visits are
                             consecutive and neither prefetch fires.
  --skip-validate      Skip the final 4-cell aggregation step.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --patch-verification) PATCH_VERIFICATION="$2"; shift 2 ;;
    --interleave) INTERLEAVE="true"; shift ;;
    --skip-validate) SKIP_VALIDATE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GROUPED_ROOT" && -d "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be a non-negative integer" >&2; exit 2; }
IFS=',' read -r -a SELECTED_DATASETS <<< "$DATASETS"
[[ "${#SELECTED_DATASETS[@]}" -gt 0 ]] || { echo "--datasets must not be empty" >&2; exit 2; }
for dataset in "${SELECTED_DATASETS[@]}"; do
  [[ -f "$GROUPED_ROOT/$dataset/grouped_prompts.jsonl" ]] || {
    echo "missing $GROUPED_ROOT/$dataset/grouped_prompts.jsonl" >&2
    exit 2
  }
done

if [[ -n "$PATCH_VERIFICATION" ]]; then
  export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
fi

AOFF_DIR="$OUTPUT_ROOT/a-off"
AON_DIR="$OUTPUT_ROOT/a-on"

echo "=== [2x2] Run 1/2: A off (cells: A0B0, A0B1) ==="
# KV-Core mode stays off (A and every other strategy remain inert); only the
# prefetch decision/execution channel is unlocked through the independent
# dispatch switch.
ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true \
bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --datasets "$DATASETS" \
  $([ "$INTERLEAVE" == "true" ] && echo --interleave) \
  --prefetch-lead-s "$PREFETCH_LEAD_S" \
  --output-dir "$AOFF_DIR"

echo "=== [2x2] Run 2/2: A on (cells: A1B0, A1B1) ==="
ASTRAKV_KV_CORE_MODE=active ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd \
ASTRAKV_KV_CORE_LOCAL_CPU=true ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true \
ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --datasets "$DATASETS" \
  $([ "$INTERLEAVE" == "true" ] && echo --interleave) \
  --prefetch-lead-s "$PREFETCH_LEAD_S" \
  --roles baseline \
  --output-dir "$AON_DIR"

if [[ "$SKIP_VALIDATE" == "true" ]]; then
  echo "2x2 ablation runs completed (validation skipped): $OUTPUT_ROOT"
  exit 0
fi

PYTHON="${ASTRAKV_PYTHON:-python3}"
export ASTRAKV_PYTHON="$PYTHON"
"$PYTHON" scripts/reporting/validate_prefetch_2x2_ablation.py \
  --a-off "$AOFF_DIR" --a-on "$AON_DIR" \
  --datasets "$DATASETS" \
  --output "$OUTPUT_ROOT/prefetch_2x2_validation.json"

echo "2x2 ablation completed: $OUTPUT_ROOT"
echo "Validation summary: $OUTPUT_ROOT/prefetch_2x2_validation.json"
