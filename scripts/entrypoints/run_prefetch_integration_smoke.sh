#!/usr/bin/env bash
set -Eeuo pipefail

# Quick DGX integration smoke: verify Prefetch-A and Prefetch-B actually fire
# with the current code BEFORE committing to the full multi-hour pipeline.
#
# Runs ONE ablation invocation on qasper with --limit 3 in active
# (A-on style) mode, then greps the state artifacts:
#   - A: prefetch_ssd_to_cpu decisions + consumed tickets
#   - B: action=prefetch receipts (online policy dispatched)
#
# Exit 0 only if both A and B produced activity.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv-W/results/dgx_prefetch_validation_bundle/workloads}"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/connector_patch_verification.json}"
OUTPUT_DIR="${ASTRAKV_SMOKE_OUT:-/home/zyx/astrakv-W/results/prefetch-integration-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${ASTRAKV_SMOKE_LIMIT:-3}"

[[ -d "$MODEL" ]] || { echo "model dir missing: $MODEL" >&2; exit 2; }
[[ -f "$PATCH_VERIFICATION" ]] || { echo "patch verification missing: $PATCH_VERIFICATION" >&2; exit 2; }
[[ -f "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" ]] || { echo "grouped prompts missing under $GROUPED_ROOT" >&2; exit 2; }

export ASTRAKV_PYTHON="$PYTHON"
export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
export ASTRAKV_KV_CORE_MODE=active
export ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd
export ASTRAKV_KV_CORE_LOCAL_CPU=true
export ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true
export ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true
export ASTRAKV_KV_CORE_ADMISSION_ENABLED=true
export ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP=8192

echo "== integration smoke: limit=$LIMIT model=$MODEL"
echo "== output: $OUTPUT_DIR"
bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$GROUPED_ROOT" \
  --datasets qasper \
  --limit "$LIMIT" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR"

A_DECISIONS=$(grep -h prefetch_ssd_to_cpu "$OUTPUT_DIR"/qasper/*-state/kv_core_policy_decisions.jsonl 2>/dev/null | wc -l)
A_CONSUMED=$(grep -h '"status": "consumed"' "$OUTPUT_DIR"/qasper/*-state/kv_core_prefetch_tickets.jsonl 2>/dev/null | wc -l)
B_RECEIPTS=$(grep -h '"action": "prefetch"' "$OUTPUT_DIR"/qasper/*-state/runtime_command_receipts.jsonl 2>/dev/null | wc -l)
B_COMPLETED=$(grep -h '"action": "prefetch"' "$OUTPUT_DIR"/qasper/*-state/runtime_command_receipts.jsonl 2>/dev/null | grep -c '"prefetched": 1')

echo "== A: prefetch_ssd_to_cpu decisions=$A_DECISIONS, consumed tickets=$A_CONSUMED"
echo "== B: prefetch receipts=$B_RECEIPTS, completed(prefetched=1)=$B_COMPLETED"

if [[ "$A_DECISIONS" -gt 0 && "$A_CONSUMED" -gt 0 && "$B_RECEIPTS" -gt 0 && "$B_COMPLETED" -gt 0 ]]; then
  echo "SMOKE PASS: A and B both fire end-to-end"
  exit 0
fi
echo "SMOKE FAIL: expected A decisions>0, A consumed>0, B receipts>0, B completed>0" >&2
exit 1
