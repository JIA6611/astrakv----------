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
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv-W/models/Qwen3-8B}"
MANIFEST="${ASTRAKV_MANIFEST:-/home/zyx/astrakv-W/deployments/kv-core-v3-e3-deterministic-recovery-20260809/deployment.manifest.json}"
SMOKE="${ASTRAKV_SMOKE:-/home/zyx/astrakv-W/results/kv-core-e1-smoke-20260813T120659Z/E1/repeated_long_prefix/cold/variant/callback-smoke.json}"
GROUPED_ROOT="${ASTRAKV_GROUPED_ROOT:-/home/zyx/astrakv-W/results/dgx_prefetch_validation_bundle/workloads}"
RESULTS_ROOT="${ASTRAKV_RESULTS_ROOT:-/home/zyx/astrakv-W/results}"
LIMIT="${ASTRAKV_LIMIT:-50}"
TRAIN_LIMIT="${ASTRAKV_TRAIN_LIMIT:-50}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
E3_OUT="$RESULTS_ROOT/e3-prefetch-fixed-$TS"
ABLATION_OUT="$RESULTS_ROOT/prefetch-ablation-2x2-$TS"
TRANSFER_SPLIT="$RESULTS_ROOT/transfer-split-qasper-$TS"
TRANSFER_PROFILE="$RESULTS_ROOT/prefetch-transfer-profile-$TS"
TRANSFER_HYBRID="$RESULTS_ROOT/prefetch-transfer-hybrid-$TS"
BUNDLE="$RESULTS_ROOT/prefetch-pipeline-bundle-$TS"

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

stage "3/7 standard 2x2 (Sidecar-B upper bound)"
bash scripts/entrypoints/run_prefetch_ablation_2x2.sh \
  --grouped-root "$GROUPED_ROOT" --model "$MODEL" --limit "$LIMIT" \
  --output-root "$ABLATION_OUT"
ABLATION_VALIDATION="$ABLATION_OUT/prefetch_2x2_validation.json"

stage "4/7 strong-overlap split (qasper train/test)"
"$PYTHON" scripts/benchmark/split_grouped_prompts_for_transfer.py \
  --grouped-prompts-jsonl "$GROUPED_ROOT/qasper/grouped_prompts.jsonl" \
  --output-root "$TRANSFER_SPLIT" \
  --dataset qasper --train-ratio 0.6 --seed 0

stage "5/7 transfer Profile-B (prefix_only)"
bash scripts/entrypoints/run_prefetch_transfer_ablation.sh \
  --output-root "$TRANSFER_PROFILE" \
  --train-grouped-root "$TRANSFER_SPLIT/train" \
  --test-grouped-root "$TRANSFER_SPLIT/test" \
  --train-dataset qasper --test-dataset qasper \
  --model "$MODEL" --train-limit "$TRAIN_LIMIT" --limit "$LIMIT"
TRANSFER_PROFILE_VALIDATION="$TRANSFER_PROFILE/prefetch_2x2_validation.json"

stage "6/7 transfer hybrid (online adaptation)"
bash scripts/entrypoints/run_prefetch_transfer_ablation.sh \
  --output-root "$TRANSFER_HYBRID" \
  --train-grouped-root "$TRANSFER_SPLIT/train" \
  --test-grouped-root "$TRANSFER_SPLIT/test" \
  --train-dataset qasper --test-dataset qasper \
  --model "$MODEL" --train-limit "$TRAIN_LIMIT" --limit "$LIMIT" \
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
