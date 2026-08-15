#!/usr/bin/env bash
set -Eeuo pipefail

# Real DGX mini demo: one canonical exact-prefix prompt, one seed request and
# a small number of revisits. It runs the E3-P baseline/variant pair so the
# comparison isolates Prefetch-A (SSD->CPU arrival promotion) from the
# otherwise identical vLLM+LMCache backend.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ASTRAKV_PYTHON:-python3}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
SOURCE_WORKLOAD=""
PATCH_MANIFEST=""
CALLBACK_SMOKE=""
REVISITS="3"
OUTPUT_TOKENS="8"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/entrypoints/run_mainline_single_prompt_demo.sh \
    --source-workload results/kv-core-workloads/repeated_long_prefix.jsonl \
    --patch-manifest <deployment_manifest.json> \
    --callback-smoke <E1_callback-smoke.json> \
    [--revisits 3] [--output-tokens 8]

This is a small real-GPU demo, not the formal E3 acceptance suite.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-workload) SOURCE_WORKLOAD="$2"; shift 2 ;;
    --patch-manifest) PATCH_MANIFEST="$2"; shift 2 ;;
    --callback-smoke) CALLBACK_SMOKE="$2"; shift 2 ;;
    --revisits) REVISITS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$SOURCE_WORKLOAD" ]] || { echo "--source-workload is required" >&2; exit 2; }
[[ -f "$PATCH_MANIFEST" ]] || { echo "--patch-manifest is required" >&2; exit 2; }
[[ -f "$CALLBACK_SMOKE" ]] || { echo "--callback-smoke is required" >&2; exit 2; }
[[ "$REVISITS" =~ ^[1-9][0-9]*$ ]] || { echo "--revisits must be positive" >&2; exit 2; }

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/mainline-single-prompt-demo-$(date -u +%Y%m%dT%H%M%SZ)}"

ASTRAKV_PYTHON="$PYTHON" ASTRAKV_MODEL="$MODEL" \
  bash scripts/entrypoints/run_e3_prefetch_performance_suite.sh \
    --source-workload "$SOURCE_WORKLOAD" \
    --patch-manifest "$PATCH_MANIFEST" \
    --callback-smoke "$CALLBACK_SMOKE" \
    --repeats 1 \
    --revisits "$REVISITS" \
    --output-tokens "$OUTPUT_TOKENS" \
    --output-dir "$OUTPUT_DIR"

echo "Mainline single-prompt demo completed: $OUTPUT_DIR"
echo "Read $OUTPUT_DIR/repeat-001/e3_prefetch_performance.json"
