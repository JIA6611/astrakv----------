#!/usr/bin/env bash
set -Eeuo pipefail

# Train/transfer evaluation for Prefetch-B (predictive prefetch).
#
# Methodology (anti-oracle):
#   - Train artifacts are derived from workload A only: prefix-prefetch
#     scheduler hints (structural) plus, optionally, an offline profile DB
#     built from a real trace run on A (--train-trace).
#   - The test workload B shares hot contexts/chunks with A but is not the
#     same request set (different dataset or a non-overlapping subset).
#   - The test 2x2 ablation runs with A-derived hints/profile exported and
#     WITHOUT the test-dataset self-built sidecar (--no-sidecar), so the
#     B-only cell measures Profile-B transfer, not an exact-next oracle.
#   - Pass --sidecar-path to evaluate Sidecar-B (exact-next upper bound)
#     instead.
#
# A training trace requires a real run on A with hooks enabled; its
# ``trace_events.jsonl`` artifact feeds build_profile_db.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-python3}"
export ASTRAKV_PYTHON="$PYTHON"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_ROOT=""
TEST_ROOT=""
TRAIN_DATASET="qasper"
TEST_DATASET="multifieldqa_en"
MODEL="${ASTRAKV_MODEL:-$ROOT/models/Qwen3-8B}"
OUTPUT_ROOT="${ASTRAKV_ROOT:-$ROOT}/results/prefetch-transfer-$TIMESTAMP"
TRAIN_LIMIT="50"
LIMIT="50"
TRAIN_TRACE=""
SIDECAR_PATH=""
PREFETCH_MODE="prefix_only"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}"
INTERLEAVE="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_prefetch_transfer_ablation.sh \
  --train-grouped-root DIR --test-grouped-root DIR [options]

Train on dataset A (hints from its canonical workload, optional profile from
--train-trace), then run the 2x2 ablation on dataset B with A-derived
profile/hints and no self-built sidecar (Profile-B). Pass --sidecar-path for
the exact-next upper bound (Sidecar-B).

Options:
  --train-grouped-root DIR   Root with <train-dataset>/grouped_prompts.jsonl.
  --test-grouped-root DIR    Root with qasper+multifieldqa_en grouped prompts
                             (the ablation contract runs both datasets).
  --train-dataset NAME       Train dataset name (default qasper).
  --test-dataset NAME        Test dataset name (default multifieldqa_en).
  --model PATH               Local model path.
  --output-root PATH         Result root (default results/prefetch-transfer-*).
  --train-limit N            Rows materialized from train dataset.
  --limit N                  Per-dataset limit for both test ablation runs.
  --train-trace FILE         Unified astra-trace-v1 JSONL from a real run on A;
                             builds the offline profile DB.
  --sidecar-path FILE        Optional train-derived exact-next sidecar (upper bound).
  --patch-verification PATH  connector_patch_verification.json for active (A-on) servers.
  --interleave               Round-robin the TEST workload's reuse groups so
                             objects become SSD-resident-but-CPU-evicted
                             between visits (required for A/B to fire).
  --prefetch-mode MODE       Online prefetch mode for the test phase:
                             prefix_only (default, Profile-B via hints/profile),
                             hybrid (adds online RuntimePrefixIndex adaptation),
                             disabled.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-grouped-root) TRAIN_ROOT="$2"; shift 2 ;;
    --test-grouped-root) TEST_ROOT="$2"; shift 2 ;;
    --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --train-trace) TRAIN_TRACE="$2"; shift 2 ;;
    --sidecar-path) SIDECAR_PATH="$2"; shift 2 ;;
    --patch-verification) PATCH_VERIFICATION="$2"; shift 2 ;;
    --interleave) INTERLEAVE="true"; shift ;;
    --prefetch-mode) PREFETCH_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TRAIN_ROOT" && -d "$TRAIN_ROOT" ]] || { echo "--train-grouped-root is required" >&2; exit 2; }
[[ -n "$TEST_ROOT" && -d "$TEST_ROOT" ]] || { echo "--test-grouped-root is required" >&2; exit 2; }
[[ -f "$TRAIN_ROOT/$TRAIN_DATASET/grouped_prompts.jsonl" ]] || {
  echo "missing $TRAIN_ROOT/$TRAIN_DATASET/grouped_prompts.jsonl" >&2; exit 2; }
[[ -f "$TEST_ROOT/$TEST_DATASET/grouped_prompts.jsonl" ]] || {
  echo "missing $TEST_ROOT/$TEST_DATASET/grouped_prompts.jsonl" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$TRAIN_LIMIT" =~ ^[0-9]+$ && "$LIMIT" =~ ^[0-9]+$ ]] || {
  echo "--train-limit and --limit must be non-negative integers" >&2; exit 2; }
case "$PREFETCH_MODE" in
  prefix_only|hybrid|combined|disabled) ;;
  *) echo "invalid --prefetch-mode: $PREFETCH_MODE" >&2; exit 2 ;;
esac
if [[ -n "$TRAIN_TRACE" ]]; then
  [[ -f "$TRAIN_TRACE" ]] || { echo "train trace is not a file: $TRAIN_TRACE" >&2; exit 2; }
fi

if [[ -n "$PATCH_VERIFICATION" ]]; then
  export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
fi

mkdir -p "$OUTPUT_ROOT"

echo "=== [transfer] Train phase on $TRAIN_DATASET ==="
"$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
  --grouped-prompts-jsonl "$TRAIN_ROOT/$TRAIN_DATASET/grouped_prompts.jsonl" \
  --output-dir "$OUTPUT_ROOT/train" \
  --dataset "$TRAIN_DATASET" --task "$TRAIN_DATASET" --limit "$TRAIN_LIMIT"
TRAIN_CANONICAL="$OUTPUT_ROOT/train/${TRAIN_DATASET}_grouped_exact_next_canonical_workload.jsonl"

"$PYTHON" scripts/reporting/build_qasper_prefix_prefetch_hints.py \
  --workload-jsonl "$TRAIN_CANONICAL" --output-dir "$OUTPUT_ROOT/train/hints"

export ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH="$OUTPUT_ROOT/train/hints/prefix_prefetch_hints.jsonl"
# The controller only consults profile/hints/runtime-learner prefetch branches
# when the mode is prefix_only/combined/hybrid; "disabled" (the default) gates
# them all out and leaves only the oracle sidecar branch.
export ASTRAKV_ONLINE_PREFETCH_MODE="$PREFETCH_MODE"
if [[ -n "$TRAIN_TRACE" ]]; then
  "$PYTHON" scripts/policy/build_profile_db.py \
    --trace-events "$TRAIN_TRACE" --workload-id "train-$TRAIN_DATASET" \
    --output-dir "$OUTPUT_ROOT/train/profile"
  export ASTRAKV_ONLINE_PROFILE_DB_PATH="$OUTPUT_ROOT/train/profile/profile_db.json"
  export ASTRAKV_OFFLINE_PROFILE_WORKLOAD_ID="train-$TRAIN_DATASET"
  # NOTE: do NOT point ASTRAKV_KV_CORE_OFFLINE_PROFILE at profile_db.json --
  # the vendor bridge expects the research profiler's
  # astrakv-kv-core-offline-profile-v2 schema and would reject ProfileDB.
  unset ASTRAKV_KV_CORE_OFFLINE_PROFILE
else
  unset ASTRAKV_ONLINE_PROFILE_DB_PATH ASTRAKV_OFFLINE_PROFILE_WORKLOAD_ID ASTRAKV_KV_CORE_OFFLINE_PROFILE
fi

if [[ -n "$SIDECAR_PATH" ]]; then
  export ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH="$SIDECAR_PATH"
  SIDECAR_ARGS=(--sidecar-path "$SIDECAR_PATH")
else
  unset ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH
  SIDECAR_ARGS=(--no-sidecar)
fi

echo "=== [transfer] Test phase on $TEST_DATASET (Profile-B) ==="
echo "=== [2x2] Run 1/2: A off (cells: A0B0, A0B1) ==="
# KV-Core mode stays off (A and every other strategy remain inert); only the
# prefetch decision/execution channel is unlocked through the independent
# dispatch switch.
ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true \
bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$TEST_ROOT" --model "$MODEL" --limit "$LIMIT" \
  --datasets "$TEST_DATASET" \
  $([ "$INTERLEAVE" == "true" ] && echo --interleave) \
  --prefetch-lead-s "${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-1.5}" \
  --output-dir "$OUTPUT_ROOT/test-a-off" "${SIDECAR_ARGS[@]}"

echo "=== [2x2] Run 2/2: A on (cells: A1B0, A1B1) ==="
ASTRAKV_KV_CORE_MODE=active ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd \
ASTRAKV_KV_CORE_LOCAL_CPU=true ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=true \
ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
  --grouped-root "$TEST_ROOT" --model "$MODEL" --limit "$LIMIT" \
  --datasets "$TEST_DATASET" \
  $([ "$INTERLEAVE" == "true" ] && echo --interleave) \
  --prefetch-lead-s "${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-1.5}" \
  --roles baseline \
  --output-dir "$OUTPUT_ROOT/test-a-on" "${SIDECAR_ARGS[@]}"

"$PYTHON" scripts/reporting/validate_prefetch_2x2_ablation.py \
  --a-off "$OUTPUT_ROOT/test-a-off" --a-on "$OUTPUT_ROOT/test-a-on" \
  --datasets "$TEST_DATASET" \
  --output "$OUTPUT_ROOT/prefetch_2x2_validation.json"

echo "Transfer ablation completed: $OUTPUT_ROOT"
echo "Validation summary: $OUTPUT_ROOT/prefetch_2x2_validation.json"
