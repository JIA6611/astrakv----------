#!/usr/bin/env bash
set -Eeuo pipefail

# One-arm, one-dataset real-machine smoke for E12 control-chain audit.
# This is intentionally independent from E11's baseline-only performance-
# boundary matrix: A and B are both enabled here, while native/external E11
# eviction selectors are disabled so they cannot change the E12 conclusion.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GROUPED_ROOT=""
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
OUTPUT_ROOT="${ASTRAKV_ROOT:-$ROOT}/results/e12-mainline-audit-$TIMESTAMP"
DATASET="qasper"
LIMIT="${ASTRAKV_E12_LIMIT:-0}"
PREFETCH_LEAD_S="${ASTRAKV_E12_PREFETCH_LEAD_S:-5.0}"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_e12_mainline_audit_smoke.sh --grouped-root DIR [options]

Options:
  --grouped-root DIR          Directory containing <dataset>/grouped_prompts.jsonl
  --model PATH                Local model path
  --output-root PATH          Output bundle root
  --dataset NAME              One dataset only (default qasper)
  --limit N                   Scheduled-row cap after fire-consume construction
                              (default 0 = all complete qasper pairs)
  --prefetch-lead-s SECONDS   A/B SSD-to-CPU lead window (default 5.0)
  --patch-verification PATH   Verified connector patch manifest (required)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grouped-root) GROUPED_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --prefetch-lead-s) PREFETCH_LEAD_S="$2"; shift 2 ;;
    --patch-verification) PATCH_VERIFICATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GROUPED_ROOT" && -d "$GROUPED_ROOT" ]] || { echo "--grouped-root is required" >&2; exit 2; }
[[ -f "$GROUPED_ROOT/$DATASET/grouped_prompts.jsonl" ]] || {
  echo "missing $GROUPED_ROOT/$DATASET/grouped_prompts.jsonl" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "--limit must be non-negative" >&2; exit 2; }
[[ -n "$PATCH_VERIFICATION" && -f "$PATCH_VERIFICATION" ]] || {
  echo "--patch-verification is required for the active KV-Core E12 smoke" >&2; exit 2; }

export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
export ASTRAKV_KV_CORE_MODE=active
export ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd
export ASTRAKV_KV_CORE_LOCAL_CPU=true
export ASTRAKV_KV_CORE_ADMISSION_ENABLED=true
export ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true
export ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true
export ASTRAKV_KV_CORE_RELEASE_CPU_STAGING_ON_CONSUME=true
export ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION=1.0
export ASTRAKV_KV_CORE_PREFETCH_DEADLINE_NS=15000000000
export ASTRAKV_KV_CORE_PREFETCH_TTL_NS=60000000000
export ASTRAKV_PREFIX_CACHING=false
export ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true
# Keep prefetched A objects resident until their following native lookup.  The
# smoke is a control-chain audit, so capacity pressure must not race the
# consume proof; capacity eviction remains outside E12 acceptance.
export ASTRAKV_LOCAL_CPU_SIZE_GB="${ASTRAKV_E12_CPU_SIZE_GB:-32.0}"
export ASTRAKV_ABLATION_OUTPUT_TOKENS="${ASTRAKV_E12_OUTPUT_TOKENS:-8}"
export ASTRAKV_ABLATION_WARMUP_OUTPUT_TOKENS="${ASTRAKV_E12_OUTPUT_TOKENS:-8}"
export ASTRAKV_ABLATION_WARMUP_PASSES=1
export ASTRAKV_ABLATION_WARMUP_SAME_SERVER=true
export ASTRAKV_ABLATION_MEASURE_PHASES=far,near
export ASTRAKV_ABLATION_PREDICTION_PHASES=far
export ASTRAKV_ONLINE_PREFETCH_MODE=hybrid

# E12 isolates control-chain wiring. E11 native selector and the historical
# external evict paths must not influence eligibility.
export ASTRAKV_E11_CPU_EVICTION_POLICY=disabled
export ASTRAKV_ONLINE_EVICT_DISPATCH_ENABLED=false
export ASTRAKV_EVICT_DISPATCH_INDEPENDENT_OF_MODE=false
export ASTRAKV_EVICT_PERIODIC_SCAN_ENABLED=false
export ASTRAKV_EVICT_GLOBAL_SCAN_ENABLED=false

bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_ROOT" \
  --datasets "$DATASET" \
  --limit "$LIMIT" \
  --roles variant \
  --interleave --interleave-pattern fire-consume \
  --prefetch-lead-s "$PREFETCH_LEAD_S"

RUN_DIR="$OUTPUT_ROOT/$DATASET/variant"
STATE_DIR="$OUTPUT_ROOT/$DATASET/variant-state"
"$PYTHON" scripts/reporting/validate_e12_mainline_smoke.py \
  --run-dir "$RUN_DIR" \
  --state-dir "$STATE_DIR" \
  --output "$OUTPUT_ROOT/e12_mainline_audit.md" \
  --json-output "$OUTPUT_ROOT/e12_mainline_audit.json" \
  --require-eligible

echo "E12 mainline audit smoke completed: $OUTPUT_ROOT"
