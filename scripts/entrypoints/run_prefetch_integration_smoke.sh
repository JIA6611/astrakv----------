#!/usr/bin/env bash
set -Eeuo pipefail

# Quick DGX integration smoke: verify Prefetch-A (arrival fill) and
# Prefetch-B (predictive) each fire end-to-end BEFORE committing to the full
# multi-hour pipeline.  Per user decision we validate A-only and B-only, not
# the A+B combined cell.
#
# The smoke runs the standard 2x2 wrapper once and checks two cells:
#   - A-only:  a-on/qasper/baseline      (A on,  B off) ->
#              prefetch_ssd_to_cpu decisions > 0 AND consumed tickets > 0
#   - B-only:  a-off/qasper/variant      (A off, B on)  ->
#              action=prefetch receipts with prefetched=1 > 0
#
# Workload is INTERLEAVED across reuse groups and the CPU hot cache is kept
# small, so objects become SSD-resident-but-CPU-evicted between visits -- the
# precondition for both strategies to have anything to do.  A non-interleaved
# grouped workload visits every object consecutively and neither fires.
#
# Exit 0 only if both cells produced activity.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv-W/results/dgx_prefetch_validation_bundle/workloads}"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/connector_patch_verification.json}"
OUTPUT_DIR="${ASTRAKV_SMOKE_OUT:-/home/zyx/astrakv-W/results/prefetch-integration-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${ASTRAKV_SMOKE_LIMIT:-16}"
# Small CPU hot pool so earlier objects are fully evicted before their
# interleaved revisit (B needs SSD-resident objects at release).  Each qasper
# object is ~0.6 GiB and the limit-16 working set is ~3.6 GiB; 2.0 GiB keeps
# pressure without the allocator-failure noise seen at 1.0 GiB.
CPU_GB="${ASTRAKV_SMOKE_CPU_GB:-2.0}"

[[ -d "$MODEL" ]] || { echo "model dir missing: $MODEL" >&2; exit 2; }
[[ -f "$PATCH_VERIFICATION" ]] || { echo "patch verification missing: $PATCH_VERIFICATION" >&2; exit 2; }
[[ -f "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" ]] || { echo "grouped prompts missing under $GROUPED_ROOT" >&2; exit 2; }

export ASTRAKV_PYTHON="$PYTHON"
export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
export ASTRAKV_LOCAL_CPU_SIZE_GB="$CPU_GB"
export ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION="${ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION:-1.0}"
export ASTRAKV_ONLINE_PREFETCH_MODE="${ASTRAKV_ONLINE_PREFETCH_MODE:-hybrid}"
export ASTRAKV_ABLATION_PREFETCH_LEAD_S="${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-0.25}"

echo "== integration smoke: limit=$LIMIT cpu_gb=$CPU_GB model=$MODEL"
echo "== output: $OUTPUT_DIR"
bash scripts/entrypoints/run_prefetch_ablation_2x2.sh \
  --grouped-root "$GROUPED_ROOT" \
  --datasets qasper \
  --limit "$LIMIT" \
  --model "$MODEL" \
  --output-root "$OUTPUT_DIR" \
  --interleave

A_DECISIONS=$(grep -h prefetch_ssd_to_cpu \
  "$OUTPUT_DIR"/a-on/qasper/baseline-state/kv_core_policy_decisions.jsonl 2>/dev/null | wc -l)
A_CONSUMED=$(grep -h '"status": "consumed"' \
  "$OUTPUT_DIR"/a-on/qasper/baseline-state/kv_core_prefetch_tickets.jsonl 2>/dev/null | wc -l)
B_RECEIPTS=$(grep -h '"action": "prefetch"' \
  "$OUTPUT_DIR"/a-off/qasper/variant-state/runtime_command_receipts.jsonl 2>/dev/null | wc -l)
B_COMPLETED=$(grep -h '"action": "prefetch"' \
  "$OUTPUT_DIR"/a-off/qasper/variant-state/runtime_command_receipts.jsonl 2>/dev/null | grep -c '"prefetched": 1')

echo "== A-only cell: prefetch_ssd_to_cpu decisions=$A_DECISIONS, consumed tickets=$A_CONSUMED"
echo "== B-only cell: prefetch receipts=$B_RECEIPTS, completed(prefetched=1)=$B_COMPLETED"

if [[ "$A_DECISIONS" -gt 0 && "$A_CONSUMED" -gt 0 && "$B_RECEIPTS" -gt 0 && "$B_COMPLETED" -gt 0 ]]; then
  echo "SMOKE PASS: A-only and B-only both fire end-to-end"
  exit 0
fi
echo "SMOKE FAIL: expected A-only decisions>0 & consumed>0, B-only receipts>0 & prefetched=1" >&2
exit 1
