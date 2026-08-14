#!/usr/bin/env bash
set -Eeuo pipefail

# Master prefetch experiment runner: Phase 1 = integration smoke (A-only /
# B-only functional gate), Phase 2 = full pipeline (E3 -> 2x2 Sidecar-B ->
# strong-overlap split -> transfer Profile-B -> transfer hybrid -> report).
#
# The full pipeline only starts if the smoke PASSes (both A-only and B-only
# fire end-to-end).  Run inside a tmux window:
#
#   cd /home/zyx/astrakv-W
#   tmux new-session -d -s astrakv-full \
#     "bash scripts/entrypoints/run_prefetch_master.sh 2>&1 | tee results/prefetch-master-live.log"
#   tmux attach -t astrakv-full
#
# All output is also mirrored to results/prefetch-master-<ts>.log.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${ASTRAKV_PYTHON:-/home/zyx/venv_for_zyx/bin/python3}"
MODEL="${ASTRAKV_MODEL:-/home/zyx/astrakv2/models/Qwen3-8B}"
export ASTRAKV_PYTHON="$PYTHON"
export ASTRAKV_MODEL="$MODEL"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="results/prefetch-master-${TS}.log"
mkdir -p results

echo "=================================================="
echo ">>> AstraKV prefetch master run [${TS}]"
echo ">>> model=$MODEL python=$PYTHON"
echo "=================================================="

echo
echo "=================================================="
echo ">>> PHASE 1/2: integration smoke (A-only / B-only)"
echo "=================================================="
set +e
bash scripts/entrypoints/run_prefetch_integration_smoke.sh
SMOKE_STATUS=$?
set -e
if [[ "$SMOKE_STATUS" -ne 0 ]]; then
  echo
  echo "!!! SMOKE FAILED (exit=$SMOKE_STATUS) — full pipeline aborted."
  echo "!!! Latest smoke dir: $(ls -dt results/prefetch-integration-smoke-* 2>/dev/null | head -1)"
  echo "!!! Send this log for analysis before re-running."
  exit 1
fi

echo
echo ">>> SMOKE PASS — proceeding to the full pipeline."
echo
echo "=================================================="
echo ">>> PHASE 2/2: full prefetch pipeline"
echo "=================================================="
bash scripts/entrypoints/run_prefetch_full_experiments.sh

echo
echo "=================================================="
echo ">>> MASTER DONE — bundle under results/prefetch-pipeline-bundle-*"
echo "=================================================="
