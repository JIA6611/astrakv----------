#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal, auditable Profile-B transfer experiment:
#   train-only trace/profile/hints -> frozen profile -> unseen test questions.
# The selected train/test questions share contexts but have disjoint request
# identities and prompt hashes.  The test variant runs without a sidecar.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
SPLIT_ROOT="${ASTRAKV_TRANSFER_SPLIT_ROOT:-/home/zyx/astrakv-W/results/transfer-split-qasper-20260814T143956Z}"
OUT_ROOT="${ASTRAKV_TRANSFER_OUT_ROOT:-/home/zyx/astrakv-W/results/prefetch-B-profile-transfer-minimal-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${ASTRAKV_TRANSFER_LIMIT:-9}"
CPU_GB="${ASTRAKV_TRANSFER_CPU_GB:-8.0}"
LEAD="${ASTRAKV_TRANSFER_LEAD_S:-5.0}"
ROLES="${ASTRAKV_TRANSFER_ROLES:-variant,baseline}"
MIN_FREE_GPU_MIB="${ASTRAKV_TRANSFER_MIN_FREE_GPU_MIB:-90000}"
PATCH_VERIFICATION="${ASTRAKV_B_PATCH_VERIFICATION:-${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}}"

[[ -x "$PYTHON" ]] || { echo "python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "model directory is missing: $MODEL" >&2; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "ASTRAKV_TRANSFER_LIMIT must be positive" >&2; exit 2; }
[[ -f "$SPLIT_ROOT/train/qasper/grouped_prompts.jsonl" ]] || {
  echo "train split is missing under $SPLIT_ROOT" >&2; exit 2; }
[[ -f "$SPLIT_ROOT/test/qasper/grouped_prompts.jsonl" ]] || {
  echo "test split is missing under $SPLIT_ROOT" >&2; exit 2; }
[[ -n "$PATCH_VERIFICATION" && -f "$PATCH_VERIFICATION" ]] || {
  echo "compatible connector patch verification is required" >&2; exit 2; }

IMPORT_PATH="$($PYTHON -c 'import astrakv; print(astrakv.__file__)')"
[[ "$IMPORT_PATH" == "$ROOT/astrakv/__init__.py" ]] || {
  echo "wrong AstraKV import path: $IMPORT_PATH" >&2; exit 2; }
"$PYTHON" - "$PATCH_VERIFICATION" <<'PY'
import json, sys
from astrakv.runtime.third_party_patch import PATCH_ID
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("compatible") is not True or payload.get("patch_id") != PATCH_ID:
    raise SystemExit("connector patch verification is incompatible")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_GPU_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$FREE_GPU_MIB" =~ ^[0-9]+$ && "$FREE_GPU_MIB" -lt "$MIN_FREE_GPU_MIB" ]]; then
    echo "insufficient free GPU/UMA memory: ${FREE_GPU_MIB} MiB < ${MIN_FREE_GPU_MIB} MiB" >&2
    exit 2
  fi
fi

mkdir -p "$OUT_ROOT"
"$PYTHON" scripts/benchmark/materialize_profile_transfer_subset.py \
  --train-grouped-prompts "$SPLIT_ROOT/train/qasper/grouped_prompts.jsonl" \
  --test-grouped-prompts "$SPLIT_ROOT/test/qasper/grouped_prompts.jsonl" \
  --output-root "$OUT_ROOT/subset" --dataset qasper --limit "$LIMIT" --visits 3

COMMON_ENV=(
  ASTRAKV_PREFIX_CACHING=false
  ASTRAKV_KV_CORE_MODE=active
  ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
  ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd
  ASTRAKV_KV_CORE_LOCAL_CPU=true
  ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=false
  ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true
  ASTRAKV_ONLINE_EVICT_DISPATCH_ENABLED=false
  ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true
  ASTRAKV_INTERLEAVE_PATTERN=fire-consume
  ASTRAKV_ABLATION_MEASURE_PHASES=far,near
  ASTRAKV_ABLATION_PREDICTION_PHASES=far
  ASTRAKV_ABLATION_WARMUP_PASSES=1
  ASTRAKV_ABLATION_WARMUP_SAME_SERVER=true
  ASTRAKV_LOCAL_CPU_SIZE_GB="$CPU_GB"
  ASTRAKV_ABLATION_PREFETCH_LEAD_S="$LEAD"
  ASTRAKV_ABLATION_OUTPUT_TOKENS=8
  ASTRAKV_ABLATION_WARMUP_OUTPUT_TOKENS=8
  ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION=1.0
  ASTRAKV_KV_CORE_PREFETCH_DEADLINE_NS=15000000000
  ASTRAKV_KV_CORE_PREFETCH_TTL_NS=60000000000
)

unset ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH

echo "=================================================="
echo ">>> Profile-B minimal transfer"
echo ">>> train/test shared contexts=$LIMIT roles=$ROLES cpu_gb=$CPU_GB lead=${LEAD}s"
echo ">>> split=$SPLIT_ROOT"
echo ">>> output=$OUT_ROOT"
echo "=================================================="

echo ">>> [1/4] train-only trace (train-only sidecar is allowed during profiling)"
env "${COMMON_ENV[@]}" ASTRAKV_ABLATION_ROLES=variant \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
    --grouped-root "$OUT_ROOT/subset/train" --datasets qasper --limit 0 \
    --interleave --interleave-pattern fire-consume --prefetch-lead-s "$LEAD" \
    --model "$MODEL" --output-dir "$OUT_ROOT/train-run"

TRAIN_QASPER="$OUT_ROOT/train-run/qasper"
TRAIN_STATE="$TRAIN_QASPER/variant-state"
TRAIN_CANONICAL="$TRAIN_QASPER/materialized/qasper_grouped_exact_next_measured_workload.jsonl"
[[ -f "$TRAIN_CANONICAL" ]] || {
  echo "train measured workload is missing: $TRAIN_CANONICAL" >&2; exit 2; }

echo ">>> [2/4] freeze train-only ProfileDB and scheduler hints"
"$PYTHON" scripts/policy/build_profile_from_runtime_events.py \
  --native-callbacks "$TRAIN_STATE/kv_core_native_callbacks.jsonl" \
  --associations "$TRAIN_STATE/request_context_associations.jsonl" \
  --workload-manifest "$TRAIN_CANONICAL" \
  --binding-events "$TRAIN_STATE/backend_binding_events.jsonl" \
  --prefetch-receipts "$TRAIN_STATE/runtime_command_receipts.jsonl" \
  --run-id train-qasper-transfer --output "$OUT_ROOT/profile/train_trace.jsonl"
"$PYTHON" scripts/policy/build_profile_db.py \
  --trace-events "$OUT_ROOT/profile/train_trace.jsonl" \
  --workload-id train-qasper-transfer --output-dir "$OUT_ROOT/profile"
"$PYTHON" scripts/reporting/build_qasper_prefix_prefetch_hints.py \
  --workload-jsonl "$TRAIN_CANONICAL" --output-dir "$OUT_ROOT/hints"

echo ">>> [3/4] unseen test questions, Profile-B only, no test sidecar"
env "${COMMON_ENV[@]}" \
  ASTRAKV_ABLATION_ROLES="$ROLES" \
  ASTRAKV_ONLINE_PROFILE_DB_PATH="$OUT_ROOT/profile/profile_db.json" \
  ASTRAKV_OFFLINE_PROFILE_WORKLOAD_ID=train-qasper-transfer \
  ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH="$OUT_ROOT/hints/prefix_prefetch_hints.jsonl" \
  ASTRAKV_ONLINE_PREFETCH_MODE=prefix_only \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
    --grouped-root "$OUT_ROOT/subset/test" --datasets qasper --limit 0 \
    --interleave --interleave-pattern fire-consume --prefetch-lead-s "$LEAD" \
    --model "$MODEL" --no-sidecar --output-dir "$OUT_ROOT/profile-test"

echo ">>> [4/4] functional, performance, and anti-leakage acceptance"
"$PYTHON" scripts/reporting/summarize_prefetch_phase_ttft.py \
  --baseline "$OUT_ROOT/profile-test/qasper/baseline/request_results.jsonl" \
  --variant "$OUT_ROOT/profile-test/qasper/variant/request_results.jsonl" \
  --phase far \
  --output "$OUT_ROOT/profile-test/qasper/predicted_far_ttft_summary.json"
"$PYTHON" scripts/reporting/analyze_prefetch_B_experiment.py \
  --root "$OUT_ROOT" --exp profile-test \
  --output "$OUT_ROOT/profile_acceptance.md" --require-functional-pass
"$PYTHON" scripts/reporting/analyze_profile_transfer_experiment.py \
  --root "$OUT_ROOT" --output "$OUT_ROOT/profile_transfer_acceptance.md" \
  --require-pass

echo "=================================================="
echo ">>> Profile-B minimal transfer completed: $OUT_ROOT"
echo ">>> acceptance: $OUT_ROOT/profile_transfer_acceptance.md"
echo "=================================================="
