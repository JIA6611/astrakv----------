#!/usr/bin/env bash
set -Eeuo pipefail

# Full Prefetch-A/B experiment pipeline for DGX (run inside a tmux window).
#
#   E3 rerun -> standard 2x2 (Sidecar-B upper bound) -> strong-overlap split
#   -> transfer Profile-B (prefix_only) -> transfer hybrid (online adaptation)
#   -> final report + artifact bundle.
#
# The pipeline stops at the first failing stage (set -e); the log shows which
# stage failed.  Override defaults via env: ASTRAKV_PYTHON, ASTRAKV_MODEL,
# ASTRAKV_MANIFEST, ASTRAKV_SMOKE, ASTRAKV_GROUPED_ROOT, ASTRAKV_RESULTS_ROOT,
# ASTRAKV_LIMIT, ASTRAKV_TRAIN_LIMIT.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
export ASTRAKV_PYTHON="$PYTHON"
# CPU==GPU on the DGX (UMA shared memory): CPU and GPU tiers are partitions of
# the same 128 GiB pool, so equal sizes must fit together with the model
# (~15 GiB).  36 GiB each keeps the total ~90 GiB; GPU utilization is lowered
# to 0.45 so the GPU KV cache lands near the CPU tier size.  Override via env.
LOCAL_CPU_SIZE_GB="${ASTRAKV_LOCAL_CPU_SIZE_GB:-36}"
export ASTRAKV_LOCAL_CPU_SIZE_GB="$LOCAL_CPU_SIZE_GB"
# The grouped ablation stages need CPU pressure so objects are evicted from
# the hot layer between interleaved visits; the E3 stage keeps the larger
# CPU==GPU pool above.  Override via env.
ABLATION_CPU_SIZE_GB="${ASTRAKV_ABLATION_CPU_SIZE_GB:-4.0}"
export ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION="${ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION:-1.0}"
export ASTRAKV_ABLATION_PREFETCH_LEAD_S="${ASTRAKV_ABLATION_PREFETCH_LEAD_S:-1.5}"
export ASTRAKV_GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.45}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
MANIFEST="${ASTRAKV_MANIFEST:-/home/zyx/astrakv-W/deployments/kv-core-v3-e3-deterministic-recovery-20260809/deployment.manifest.json}"
SMOKE="${ASTRAKV_SMOKE:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/E1/repeated_long_prefix/cold/variant/callback-smoke.json}"
PATCH_VERIFICATION="${ASTRAKV_KV_CORE_PATCH_VERIFICATION:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/connector_patch_verification.json}"
export ASTRAKV_KV_CORE_PATCH_VERIFICATION="$PATCH_VERIFICATION"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv-W/results/dgx_prefetch_validation_bundle/workloads}"
RESULTS_ROOT="${ASTRAKV_RESULTS_ROOT:-/home/zyx/astrakv-W/results}"
LIMIT="${ASTRAKV_LIMIT:-50}"
TRAIN_LIMIT="${ASTRAKV_TRAIN_LIMIT:-50}"
RESUME="${ASTRAKV_RESUME:-0}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
E3_OUT="$RESULTS_ROOT/e3-prefetch-fixed-$TS"
ABLATION_OUT="$RESULTS_ROOT/prefetch-ablation-2x2-$TS"
TRANSFER_SPLIT="$RESULTS_ROOT/transfer-split-qasper-$TS"
TRANSFER_PROFILE="$RESULTS_ROOT/prefetch-transfer-profile-$TS"
TRANSFER_HYBRID="$RESULTS_ROOT/prefetch-transfer-hybrid-$TS"
BUNDLE="$RESULTS_ROOT/prefetch-pipeline-bundle-$TS"

if [[ "$RESUME" == "1" ]]; then
  E3_OUT="$(for d in "$RESULTS_ROOT"/e3-prefetch-fixed-*; do
    [[ -d "$d" && -f "$d/e3_prefetch_acceptance.json" ]] && echo "$d"
  done | sort | tail -n 1)"
  if [[ -z "$E3_OUT" ]]; then
    echo "resume requested but no completed E3 dir with e3_prefetch_acceptance.json found under $RESULTS_ROOT" >&2
    exit 2
  fi
  echo "==> RESUME=1: reusing E3 output $E3_OUT (stages 1-2 skipped)"
fi

CURRENT_STAGE=""
stage() {
  CURRENT_STAGE="$*"
  echo
  echo "=================================================="
  echo ">>> $*"
  echo "=================================================="
}

fail() {
  echo
  echo "!!! Pipeline FAILED at stage: ${CURRENT_STAGE:-unknown}"
  echo "!!! Re-run the same command after fixing; logs above show the failure."
  exit 1
}
trap fail ERR

echo "==> env: PYTHON=$PYTHON"
echo "==> env: MODEL=$MODEL"
echo "==> env: MANIFEST=$MANIFEST"
echo "==> env: SMOKE=$SMOKE"
echo "==> env: LIMIT=$LIMIT TRAIN_LIMIT=$TRAIN_LIMIT"
echo "==> env: RESUME=$RESUME E3_OUT=$E3_OUT"

if [[ "$RESUME" != "1" ]]; then
  stage "1/7 E3 source materialize"
  mkdir -p "$RESULTS_ROOT/e3-source-$TS"
  "$PYTHON" scripts/benchmark/materialize_grouped_exact_next_workload.py \
    --grouped-prompts-jsonl "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" \
    --output-dir "$RESULTS_ROOT/e3-source-$TS" \
    --dataset qasper --task qasper --limit "$TRAIN_LIMIT"
  E3_SOURCE="$RESULTS_ROOT/e3-source-$TS/qasper_grouped_exact_next_canonical_workload.jsonl"

  stage "2/7 E3 rerun"
  ASTRAKV_PYTHON="$PYTHON" bash scripts/entrypoints/run_e3_prefetch_performance_suite.sh \
    --source-workload "$E3_SOURCE" \
    --patch-manifest "$MANIFEST" --callback-smoke "$SMOKE" \
    --model "$MODEL" --output-dir "$E3_OUT"
  "$PYTHON" scripts/reporting/validate_e3_prefetch_acceptance.py \
    --e3-root "$E3_OUT" --output "$E3_OUT/e3_prefetch_acceptance.json"
fi

stage "3/7 standard 2x2 (Sidecar-B upper bound)"
export ASTRAKV_LOCAL_CPU_SIZE_GB="$ABLATION_CPU_SIZE_GB"
bash scripts/entrypoints/run_prefetch_ablation_2x2.sh \
  --grouped-root "$GROUPED_ROOT" --model "$MODEL" --limit "$LIMIT" \
  --interleave \
  --output-root "$ABLATION_OUT"
ABLATION_VALIDATION="$ABLATION_OUT/prefetch_2x2_validation.json"

stage "4/7 strong-overlap split (qasper train/test)"
"$PYTHON" scripts/benchmark/split_grouped_prompts_for_transfer.py \
  --grouped-prompts-jsonl "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" \
  --output-root "$TRANSFER_SPLIT" \
  --dataset qasper --train-ratio 0.6 --seed 0

stage "4b/7 train trace -> history profile (Profile-B)"
TRAIN_TRACE_DIR="$RESULTS_ROOT/train-trace-$TS"
# Real run on the TRAIN split with Prefetch-B active (interleaved so the
# trace carries prefetch hits), producing the runtime artifacts the profile
# converter joins onto canonical chunk identities.
ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE=true \
  bash scripts/entrypoints/run_grouped_exact_next_prefetch_ablation.sh \
    --grouped-root "$TRANSFER_SPLIT/train" \
    --datasets qasper \
    --limit "$TRAIN_LIMIT" \
    --model "$MODEL" \
    --roles variant \
    --interleave \
    --prefetch-lead-s "$ASTRAKV_ABLATION_PREFETCH_LEAD_S" \
    --output-dir "$TRAIN_TRACE_DIR"
TRAIN_CANONICAL="$TRAIN_TRACE_DIR/qasper/materialized/qasper_grouped_exact_next_canonical_workload.jsonl"
"$PYTHON" scripts/policy/build_profile_from_runtime_events.py \
  --native-callbacks "$TRAIN_TRACE_DIR/qasper/variant-state/kv_core_native_callbacks.jsonl" \
  --associations "$TRAIN_TRACE_DIR/qasper/variant-state/request_context_associations.jsonl" \
  --workload-manifest "$TRAIN_CANONICAL" \
  --binding-events "$TRAIN_TRACE_DIR/qasper/variant-state/backend_binding_events.jsonl" \
  --prefetch-receipts "$TRAIN_TRACE_DIR/qasper/variant-state/runtime_command_receipts.jsonl" \
  --run-id "train-trace-qasper" \
  --output "$TRAIN_TRACE_DIR/train_trace.jsonl"
TRAIN_TRACE="$TRAIN_TRACE_DIR/train_trace.jsonl"
[[ -s "$TRAIN_TRACE" ]] || { echo "train trace is empty: $TRAIN_TRACE" >&2; exit 2; }

stage "5/7 transfer Profile-B (prefix_only)"
bash scripts/entrypoints/run_prefetch_transfer_ablation.sh \
  --output-root "$TRANSFER_PROFILE" \
  --train-grouped-root "$TRANSFER_SPLIT/train" \
  --test-grouped-root "$TRANSFER_SPLIT/test" \
  --train-dataset qasper --test-dataset qasper \
  --model "$MODEL" --train-limit "$TRAIN_LIMIT" --limit "$LIMIT" \
  --train-trace "$TRAIN_TRACE" \
  --interleave
TRANSFER_PROFILE_VALIDATION="$TRANSFER_PROFILE/prefetch_2x2_validation.json"

stage "6/7 transfer hybrid (online adaptation)"
bash scripts/entrypoints/run_prefetch_transfer_ablation.sh \
  --output-root "$TRANSFER_HYBRID" \
  --train-grouped-root "$TRANSFER_SPLIT/train" \
  --test-grouped-root "$TRANSFER_SPLIT/test" \
  --train-dataset qasper --test-dataset qasper \
  --model "$MODEL" --train-limit "$TRAIN_LIMIT" --limit "$LIMIT" \
  --train-trace "$TRAIN_TRACE" \
  --interleave \
  --prefetch-mode hybrid
for role in baseline variant; do
  "$PYTHON" scripts/reporting/analyze_prefetch_adaptation.py \
    --role-dir "$TRANSFER_HYBRID/test-a-on/qasper/$role" \
    --state-dir "$TRANSFER_HYBRID/test-a-on/qasper/${role}-state" \
    --windows 5 \
    --output "$TRANSFER_HYBRID/test-a-on/qasper/$role/adaptation.json"
done
ADAPTATION="$TRANSFER_HYBRID/test-a-on/qasper/variant/adaptation.json"

stage "7/7 final report + bundle"
FINAL_REPORT="$RESULTS_ROOT/prefetch_final_report-$TS.md"
"$PYTHON" scripts/reporting/build_prefetch_final_report.py \
  --e3 "$E3_OUT/e3_prefetch_acceptance.json" \
  --ablation-2x2 "$ABLATION_VALIDATION" \
  --transfer "$TRANSFER_PROFILE_VALIDATION" \
  --adaptation "$ADAPTATION" \
  --output "$FINAL_REPORT"
mkdir -p "$BUNDLE"
cp "$E3_OUT/e3_prefetch_acceptance.json" "$BUNDLE/e3_acceptance.json"
cp "$ABLATION_VALIDATION" "$BUNDLE/ablation_2x2_validation.json"
cp "$TRANSFER_PROFILE_VALIDATION" "$BUNDLE/transfer_profile_validation.json"
cp "$ADAPTATION" "$BUNDLE/adaptation.json"
cp "$FINAL_REPORT" "$BUNDLE/final_report.md"

echo
echo "=================================================="
echo "ALL STAGES COMPLETED"
echo "  E3 acceptance:   $E3_OUT/e3_prefetch_acceptance.json"
echo "  2x2 validation:  $ABLATION_VALIDATION"
echo "  Profile-B:       $TRANSFER_PROFILE_VALIDATION"
echo "  Adaptation:      $ADAPTATION"
echo "  Final report:    $FINAL_REPORT"
echo "  Bundle (send):   $BUNDLE"
echo "=================================================="
