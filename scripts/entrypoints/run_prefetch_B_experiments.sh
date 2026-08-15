#!/usr/bin/env bash
set -Eeuo pipefail

# Prefetch-B standalone experiments: functional acceptance (decisions +
# completed/prefetched receipts) and performance evidence (TTFT) for the two
# decision sources.  Experiment 1 (E3, Prefetch-A) is already complete and is
# NOT re-run here.
#
#   Exp 2 (B-sidecar): baseline (B off) vs B-only using the same-source
#                      exact-next sidecar (oracle upper bound).
#   Exp 3 (B-profile): train-trace -> offline ProfileDB + hints, then
#                      baseline vs B-only using the profile (honest, no
#                      test-set sidecar).
#
# Both use vLLM prefix caching OFF (ASTRAKV_PREFIX_CACHING=false), the same
# condition under which E3 proved Prefetch-A's benefit: revisits fall back to
# LMCache, so a promoted/prefetched CPU copy measurably saves a disk read.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv-W/results/dgx_prefetch_validation_bundle/workloads}"
OUT_ROOT="${ASTRAKV_B_OUT_ROOT:-/home/zyx/astrakv-W/results/prefetch-B-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${ASTRAKV_B_LIMIT:-50}"
CPU_GB="${ASTRAKV_B_CPU_GB:-4.0}"
LEAD="${ASTRAKV_B_LEAD_S:-1.5}"
RUN_SIDECAR="${ASTRAKV_B_RUN_SIDECAR:-1}"
RUN_PROFILE="${ASTRAKV_B_RUN_PROFILE:-1}"
export ASTRAKV_PYTHON="$PYTHON"

[[ -d "$MODEL" ]] || { echo "model dir missing: $MODEL" >&2; exit 2; }
[[ -f "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" ]] || { echo "grouped prompts missing: $GROUPED_ROOT/qasper/grouped_prompts.jsonl" >&2; exit 2; }

COMMON_ENV=(
  ASTRAKV_PREFIX_CACHING=false
  ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true
  ASTRAKV_LOCAL_CPU_SIZE_GB="$CPU_GB"
  ASTRAKV_ABLATION_PREFETCH_LEAD_S="$LEAD"
  ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION=1.0
)

echo "=================================================="
echo ">>> Prefetch-B experiments [sidecar + profile]"
echo ">>> model=$MODEL limit=$LIMIT cpu_gb=$CPU_GB lead=${LEAD}s"
echo ">>> run_sidecar=$RUN_SIDECAR run_profile=$RUN_PROFILE"
echo ">>> output=$OUT_ROOT"
echo ">>> (Exp 1 = E3 for Prefetch-A already complete; not re-run)"
echo "=================================================="
mkdir -p "$OUT_ROOT"

echo
echo "=================================================="
echo ">>> Exp 2/3: B-sidecar (baseline vs B-only + exact-next sidecar)"
echo "=================================================="
if [[ "$RUN_SIDECAR" == "1" ]]; then
  env "${COMMON_ENV[@]}" \
    bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
      --grouped-root "$GROUPED_ROOT" --datasets qasper --limit "$LIMIT" \
      --interleave --prefetch-lead-s "$LEAD" --model "$MODEL" \
      --output-dir "$OUT_ROOT/sidecar"
else
  echo ">>> B-sidecar skipped (ASTRAKV_B_RUN_SIDECAR=0)"
fi

echo
echo "=================================================="
echo ">>> Exp 3/3: B-profile (train trace -> ProfileDB, no test sidecar)"
echo "=================================================="
if [[ "$RUN_PROFILE" == "1" ]]; then
  # NOTE: the profile is built from a real run over the SAME qasper object set
  # (an interleaved warm-up with Prefetch-B active).  The test phase replays
  # the same interleaved workload with the offline ProfileDB + hints and no
  # test-set sidecar, so B's decisions come from the profile path.
  echo "--- [3a] train trace run (full qasper, Prefetch-B active, prefix caching off) ---"
  env "${COMMON_ENV[@]}" \
    bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
      --grouped-root "$GROUPED_ROOT" --datasets qasper --limit "$LIMIT" \
      --roles variant --interleave --prefetch-lead-s "$LEAD" --model "$MODEL" \
      --output-dir "$OUT_ROOT/train-trace"
  TRAIN_STATE="$OUT_ROOT/train-trace/qasper/variant-state"
  TRAIN_CANONICAL="$OUT_ROOT/train-trace/qasper/materialized/qasper_grouped_exact_next_canonical_workload.jsonl"

  echo "--- [3b] build offline ProfileDB + hints from the train trace ---"
  "$PYTHON" scripts/policy/build_profile_from_runtime_events.py \
    --native-callbacks "$TRAIN_STATE/kv_core_native_callbacks.jsonl" \
    --associations "$TRAIN_STATE/request_context_associations.jsonl" \
    --workload-manifest "$TRAIN_CANONICAL" \
    --binding-events "$TRAIN_STATE/backend_binding_events.jsonl" \
    --prefetch-receipts "$TRAIN_STATE/runtime_command_receipts.jsonl" \
    --run-id train-qasper --output "$OUT_ROOT/profile/train_trace.jsonl"
  "$PYTHON" scripts/policy/build_profile_db.py \
    --trace-events "$OUT_ROOT/profile/train_trace.jsonl" \
    --workload-id train-qasper --output-dir "$OUT_ROOT/profile"
  "$PYTHON" scripts/reporting/build_qasper_prefix_prefetch_hints.py \
    --workload-jsonl "$TRAIN_CANONICAL" --output-dir "$OUT_ROOT/hints"

  echo "--- [3c] test run (baseline vs B-only + offline profile, no sidecar) ---"
  env "${COMMON_ENV[@]}" \
    ASTRAKV_ONLINE_PROFILE_DB_PATH="$OUT_ROOT/profile/profile_db.json" \
    ASTRAKV_OFFLINE_PROFILE_WORKLOAD_ID=train-qasper \
    ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH="$OUT_ROOT/hints/prefix_prefetch_hints.jsonl" \
    ASTRAKV_ONLINE_PREFETCH_MODE=prefix_only \
    bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
      --grouped-root "$GROUPED_ROOT" --datasets qasper --limit "$LIMIT" \
      --interleave --prefetch-lead-s "$LEAD" --model "$MODEL" --no-sidecar \
      --output-dir "$OUT_ROOT/profile-test"
else
  echo ">>> B-profile skipped (ASTRAKV_B_RUN_PROFILE=0)"
fi

echo
echo "=================================================="
echo ">>> B experiments done: $OUT_ROOT"
echo "=================================================="
echo "Sidecar B completed receipts:"
grep -h '"action": "prefetch"' "$OUT_ROOT/sidecar/qasper/variant-state/runtime_command_receipts.jsonl" 2>/dev/null | grep -c '"prefetched": 1' || true
echo "Profile B completed receipts:"
grep -h '"action": "prefetch"' "$OUT_ROOT/profile-test/qasper/variant-state/runtime_command_receipts.jsonl" 2>/dev/null | grep -c '"prefetched": 1' || true
