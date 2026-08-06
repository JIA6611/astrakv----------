#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=""
MAIN_EVIDENCE="results/extended_evidence_20260626_131728"
DEMO_EVIDENCE=""
BOUNDARY_PASS="results/extended_g016_ctx32k_b16_out256"
BOUNDARY_FAIL="results/extended_g015_ctx32k_b16_out256"
MODEL="${ASTRAKV_MODEL:-/home/szl/Desktop/Inference-OS/models/Qwen2.5-7B-Instruct}"
SKIP_INSTALL=0
CONTINUE_ON_FAILURE=0
WITH_BOUNDARY_RUNS=0
BOUNDARY_UTILS="0.15 0.16 0.20"
BOUNDARY_MAX_MODEL_LEN="32768"
BOUNDARY_CONTEXT_LENGTHS="24576 32768"
BOUNDARY_BATCH_SIZES="4 8 12 16"
BOUNDARY_OUTPUT_TOKENS="256"
BOUNDARY_REPEAT="1"
BOUNDARY_TIMEOUT="2400"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/entrypoints/run_experiment_figures.sh [options]

Default mode only builds figures from existing evidence.

Options:
  --output-root PATH
  --main-evidence PATH
  --demo-evidence PATH
  --boundary-pass PATH
  --boundary-fail PATH
  --model PATH_OR_HF_ID
  --with-boundary-runs
  --boundary-utils "0.15 0.16 0.20"
  --boundary-max-model-len INT
  --boundary-context-lengths "INT INT ..."
  --boundary-batch-sizes "INT INT ..."
  --boundary-output-tokens INT
  --boundary-repeat INT
  --boundary-timeout SECONDS
  --skip-install
  --continue-on-failure
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --main-evidence) MAIN_EVIDENCE="$2"; shift 2 ;;
    --demo-evidence) DEMO_EVIDENCE="$2"; shift 2 ;;
    --boundary-pass) BOUNDARY_PASS="$2"; shift 2 ;;
    --boundary-fail) BOUNDARY_FAIL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --with-boundary-runs) WITH_BOUNDARY_RUNS=1; shift ;;
    --boundary-utils) BOUNDARY_UTILS="$2"; shift 2 ;;
    --boundary-max-model-len) BOUNDARY_MAX_MODEL_LEN="$2"; shift 2 ;;
    --boundary-context-lengths) BOUNDARY_CONTEXT_LENGTHS="$2"; shift 2 ;;
    --boundary-batch-sizes) BOUNDARY_BATCH_SIZES="$2"; shift 2 ;;
    --boundary-output-tokens) BOUNDARY_OUTPUT_TOKENS="$2"; shift 2 ;;
    --boundary-repeat) BOUNDARY_REPEAT="$2"; shift 2 ;;
    --boundary-timeout) BOUNDARY_TIMEOUT="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --continue-on-failure) CONTINUE_ON_FAILURE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/experiment_figures_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_ROOT"
COMMAND_LOG="$OUTPUT_ROOT/commands.log"
: > "$COMMAND_LOG"

log() {
  echo "[$(date -Is)] $*" | tee -a "$COMMAND_LOG"
}

run_cmd() {
  local name="$1"
  shift
  local log_file="$OUTPUT_ROOT/${name}.log"
  log "RUN ${name}: $*"
  set +e
  "$@" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2)
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAIL ${name}: exit=${status}"
    if [[ "$CONTINUE_ON_FAILURE" -eq 0 ]]; then
      exit "$status"
    fi
  else
    log "OK ${name}"
  fi
  return "$status"
}

ensure_python() {
  if [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    run_cmd install_plot_deps python -m pip install -r requirements.txt
  fi
}

run_boundary_sweep() {
  [[ "$WITH_BOUNDARY_RUNS" -eq 1 ]] || return 0
  local util
  for util in $BOUNDARY_UTILS; do
    local tag
    tag="$(printf '%s' "$util" | tr -d '.')"
    local out_dir="$OUTPUT_ROOT/boundary_runs/g${tag}_ctx${BOUNDARY_MAX_MODEL_LEN}"
    local args=(
      scripts/entrypoints/run_competition_extended_evidence.sh
      --only boundary
      --gpu-util-boundary "$util"
      --boundary-max-model-len "$BOUNDARY_MAX_MODEL_LEN"
      --boundary-context-lengths "$BOUNDARY_CONTEXT_LENGTHS"
      --boundary-batch-sizes "$BOUNDARY_BATCH_SIZES"
      --boundary-output-tokens "$BOUNDARY_OUTPUT_TOKENS"
      --boundary-repeat "$BOUNDARY_REPEAT"
      --boundary-timeout "$BOUNDARY_TIMEOUT"
      --output-root "$out_dir"
      --model "$MODEL"
    )
    [[ "$SKIP_INSTALL" -eq 1 ]] && args+=(--skip-install)
    args+=(--continue-on-failure)
    run_cmd "boundary_g${tag}" bash "${args[@]}"
    if [[ "$util" == "0.15" ]]; then
      BOUNDARY_FAIL="$out_dir"
    elif [[ "$util" == "0.16" ]]; then
      BOUNDARY_PASS="$out_dir"
    fi
  done
}

build_figures() {
  local figure_args=(
    scripts/plotting/build_experiment_figures.py
    --output-dir "$OUTPUT_ROOT/figures"
    --main-evidence "$MAIN_EVIDENCE"
    --boundary-pass "$BOUNDARY_PASS"
    --boundary-fail "$BOUNDARY_FAIL"
  )
  if [[ -n "$DEMO_EVIDENCE" ]]; then
    figure_args+=(--demo-evidence "$DEMO_EVIDENCE")
  fi
  run_cmd build_figures python "${figure_args[@]}"
}

main() {
  log "Experiment figure output root: $OUTPUT_ROOT"
  log "Main evidence: $MAIN_EVIDENCE"
  log "Demo evidence: ${DEMO_EVIDENCE:-none}"
  log "Boundary pass: $BOUNDARY_PASS"
  log "Boundary fail: $BOUNDARY_FAIL"
  ensure_python
  run_cmd shell_syntax bash -n \
    scripts/entrypoints/run_experiment_figures.sh \
    scripts/entrypoints/run_competition_extended_evidence.sh
  run_boundary_sweep
  build_figures
  log "Figure report: $OUTPUT_ROOT/figures/figure_report.md"
}

main "$@"
