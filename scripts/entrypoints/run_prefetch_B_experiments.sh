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
# fire-consume emits a bounded measured set after scheduling all eligible
# groups; do not truncate the final near pair when the filler count changes.
LIMIT="${ASTRAKV_B_LIMIT:-0}"
# The nine formal qasper targets total about 5.74 GB of KV.  Eight GiB keeps
# the symmetric warmup set addressable while still requiring each measured
# far request to follow the explicit CPU-invalidate path before B promotion.
CPU_GB="${ASTRAKV_B_CPU_GB:-8.0}"
LEAD="${ASTRAKV_B_LEAD_S:-5.0}"
EVICTION_FILL="${ASTRAKV_B_EVICTION_FILL_GROUPS:-32}"
RUN_SIDECAR="${ASTRAKV_B_RUN_SIDECAR:-1}"
RUN_PROFILE="${ASTRAKV_B_RUN_PROFILE:-1}"
EXISTING_TRAIN_QASPER="${ASTRAKV_B_EXISTING_TRAIN_QASPER:-}"
MIN_FREE_GPU_MIB="${ASTRAKV_B_MIN_FREE_GPU_MIB:-90000}"
PATCH_VERIFICATION="${ASTRAKV_B_PATCH_VERIFICATION:-${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-}}"
export ASTRAKV_PYTHON="$PYTHON"

[[ -d "$MODEL" ]] || { echo "model dir missing: $MODEL" >&2; exit 2; }
[[ -f "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" ]] || { echo "grouped prompts missing: $GROUPED_ROOT/qasper/grouped_prompts.jsonl" >&2; exit 2; }
[[ -n "$PATCH_VERIFICATION" ]] || {
  echo "Prefetch-B active mode requires ASTRAKV_B_PATCH_VERIFICATION (or ASTRAKV_KV_CORE_PATCH_VERIFICATION)" >&2
  exit 2
}
[[ -f "$PATCH_VERIFICATION" ]] || {
  echo "connector patch verification file missing: $PATCH_VERIFICATION" >&2
  exit 2
}
IMPORT_PATH="$($PYTHON -c 'import astrakv; print(astrakv.__file__)')"
[[ "$IMPORT_PATH" == "$ROOT/astrakv/__init__.py" ]] || {
  echo "wrong AstraKV import path: $IMPORT_PATH (expected $ROOT/astrakv/__init__.py)" >&2
  exit 2
}
"$PYTHON" - "$PATCH_VERIFICATION" <<'PY'
import json
import sys

from astrakv.runtime.third_party_patch import PATCH_ID

path = sys.argv[1]
try:
    payload = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid connector patch verification JSON: {path}: {exc}")
if payload.get("compatible") is not True:
    raise SystemExit(f"connector patch verification is not compatible: {path}")
if payload.get("patch_id") != PATCH_ID:
    raise SystemExit(
        f"connector patch verification patch_id mismatch: "
        f"{payload.get('patch_id')!r} != {PATCH_ID!r}"
    )
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_GPU_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$FREE_GPU_MIB" =~ ^[0-9]+$ && "$FREE_GPU_MIB" -lt "$MIN_FREE_GPU_MIB" ]]; then
    echo "insufficient free GPU/UMA memory: ${FREE_GPU_MIB} MiB < ${MIN_FREE_GPU_MIB} MiB" >&2
    echo "stop stale vLLM/EngineCore processes or lower the explicitly audited memory target before rerunning" >&2
    exit 2
  fi
fi

COMMON_ENV=(
  ASTRAKV_PREFIX_CACHING=false
  # Formal B-only contract: keep the shared KV-Core execution substrate active
  # in both arms, but leave A's unconditional arrival-prefetch flag disabled.
  # Only a sidecar/profile authorization may request SSD->CPU promotion.
  ASTRAKV_KV_CORE_MODE=active
  ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
  ASTRAKV_KV_CORE_TOPOLOGY=gpu_cpu_ssd
  ASTRAKV_KV_CORE_LOCAL_CPU=true
  ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED=false
  # LMCache 0.4.7 cannot restore a prior process's LocalDiskBackend index.
  # Seed on the formal server, then symmetrically remove the target CPU copy
  # at measured ingress. Only an authenticated B authorization may promote it
  # back during the lead window; Prefetch-A remains disabled in both arms.
  ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD=true
  # Evict-B is a separate policy arm; it must not alter the B prefetch
  # comparison or remove a CPU copy before the near consumer arrives.
  ASTRAKV_ONLINE_EVICT_DISPATCH_ENABLED=false
  ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true
  ASTRAKV_INTERLEAVE_PATTERN=fire-consume
  ASTRAKV_ABLATION_EVICTION_FILL_GROUPS="$EVICTION_FILL"
  ASTRAKV_ABLATION_MEASURE_PHASES=far,near
  ASTRAKV_ABLATION_PREDICTION_PHASES=far
  ASTRAKV_ABLATION_WARMUP_PASSES=1
  ASTRAKV_ABLATION_WARMUP_SAME_SERVER=true
  ASTRAKV_LOCAL_CPU_SIZE_GB="$CPU_GB"
  ASTRAKV_ABLATION_PREFETCH_LEAD_S="$LEAD"
  # B measures TTFT and prefix movement. Eight decode tokens preserve those
  # measurements while avoiding hundreds of unnecessary generation tokens.
  ASTRAKV_ABLATION_OUTPUT_TOKENS=8
  ASTRAKV_ABLATION_WARMUP_OUTPUT_TOKENS=8
  ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION=1.0
  # Give SSD->CPU promotion enough time for the largest qasper prefix while
  # keeping the TTL finite and the near request inside the same window.
  ASTRAKV_KV_CORE_PREFETCH_DEADLINE_NS=15000000000
  ASTRAKV_KV_CORE_PREFETCH_TTL_NS=60000000000
)

echo "=================================================="
echo ">>> Prefetch-B experiments [sidecar + profile]"
echo ">>> model=$MODEL limit=$LIMIT cpu_gb=$CPU_GB lead=${LEAD}s"
echo ">>> patch_verification=$PATCH_VERIFICATION"
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
      --interleave --interleave-pattern fire-consume --prefetch-lead-s "$LEAD" --model "$MODEL" \
      --output-dir "$OUT_ROOT/sidecar"
else
  echo ">>> B-sidecar skipped (ASTRAKV_B_RUN_SIDECAR=0)"
fi

echo
echo "=================================================="
echo ">>> Exp 3/3: B-profile (train trace -> ProfileDB, no test sidecar)"
echo "=================================================="
if [[ "$RUN_PROFILE" == "1" ]]; then
  # NOTE: the profile is built from a real measured run over the SAME qasper
  # object set.  Its SSD is seeded by an independent LMCache-only warmup, then
  # the measured far->near phase runs with Prefetch-B active.  The test phase
  # replays that workload with the offline ProfileDB + hints and no test-set
  # sidecar, so B's decisions come from the profile path.
  if [[ -n "$EXISTING_TRAIN_QASPER" ]]; then
    echo "--- [3a] reuse completed real train trace: $EXISTING_TRAIN_QASPER ---"
    TRAIN_QASPER="$EXISTING_TRAIN_QASPER"
  else
    echo "--- [3a] train trace run (full qasper, Prefetch-B active, prefix caching off) ---"
    env "${COMMON_ENV[@]}" \
      bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
        --grouped-root "$GROUPED_ROOT" --datasets qasper --limit "$LIMIT" \
        --roles variant --interleave --interleave-pattern fire-consume --prefetch-lead-s "$LEAD" --model "$MODEL" \
        --output-dir "$OUT_ROOT/train-trace"
    TRAIN_QASPER="$OUT_ROOT/train-trace/qasper"
  fi
  TRAIN_STATE="$TRAIN_QASPER/variant-state"
  TRAIN_CANONICAL="$TRAIN_QASPER/materialized/qasper_grouped_exact_next_measured_workload.jsonl"
  if [[ ! -f "$TRAIN_CANONICAL" ]]; then
    TRAIN_CANONICAL="$TRAIN_QASPER/materialized/qasper_grouped_exact_next_canonical_workload.jsonl"
  fi
  [[ -f "$TRAIN_STATE/kv_core_native_callbacks.jsonl" ]] || {
    echo "training native callbacks missing: $TRAIN_STATE/kv_core_native_callbacks.jsonl" >&2; exit 2; }
  [[ -f "$TRAIN_STATE/request_context_associations.jsonl" ]] || {
    echo "training request associations missing: $TRAIN_STATE/request_context_associations.jsonl" >&2; exit 2; }
  [[ -f "$TRAIN_CANONICAL" ]] || {
    echo "training workload manifest missing: $TRAIN_CANONICAL" >&2; exit 2; }

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
      --interleave --interleave-pattern fire-consume --prefetch-lead-s "$LEAD" --model "$MODEL" --no-sidecar \
      --output-dir "$OUT_ROOT/profile-test"
else
  echo ">>> B-profile skipped (ASTRAKV_B_RUN_PROFILE=0)"
fi

summarize_predicted_ttft() {
  local experiment_root="$1"
  local baseline="$experiment_root/qasper/baseline/request_results.jsonl"
  local variant="$experiment_root/qasper/variant/request_results.jsonl"
  local output="$experiment_root/qasper/predicted_far_ttft_summary.json"
  if [[ -f "$baseline" && -f "$variant" ]]; then
    "$PYTHON" scripts/reporting/summarize_prefetch_phase_ttft.py \
      --baseline "$baseline" \
      --variant "$variant" \
      --phase far \
      --output "$output"
  fi
}

if [[ "$RUN_SIDECAR" == "1" ]]; then
  summarize_predicted_ttft "$OUT_ROOT/sidecar"
  "$PYTHON" scripts/reporting/analyze_prefetch_B_experiment.py \
    --root "$OUT_ROOT" --exp sidecar \
    --output "$OUT_ROOT/sidecar_acceptance.md" \
    --require-functional-pass
fi
if [[ "$RUN_PROFILE" == "1" ]]; then
  summarize_predicted_ttft "$OUT_ROOT/profile-test"
  "$PYTHON" scripts/reporting/analyze_prefetch_B_experiment.py \
    --root "$OUT_ROOT" --exp profile-test \
    --output "$OUT_ROOT/profile_acceptance.md" \
    --require-functional-pass
fi

echo
echo "=================================================="
echo ">>> B experiments done: $OUT_ROOT"
echo "=================================================="
echo "Sidecar B completed receipts:"
grep -h '"action": "prefetch"' "$OUT_ROOT/sidecar/qasper/variant-state/runtime_command_receipts.jsonl" 2>/dev/null | grep -c '"prefetched": 1' || true
echo "Profile B completed receipts:"
grep -h '"action": "prefetch"' "$OUT_ROOT/profile-test/qasper/variant-state/runtime_command_receipts.jsonl" 2>/dev/null | grep -c '"prefetched": 1' || true
