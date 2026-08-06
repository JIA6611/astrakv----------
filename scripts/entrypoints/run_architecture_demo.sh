#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=""
MAIN_EVIDENCE="results/extended_evidence_20260625_014917"
BOUNDARY_PASS="results/extended_g016_ctx32k_b16_out256"
BOUNDARY_FAIL="results/extended_g015_ctx32k_b16_out256"
SUMMARY_PATH="results/project_implementation_evidence_summary.md"
MODEL="${ASTRAKV_MODEL:-/home/szl/Desktop/Inference-OS/models/Qwen2.5-7B-Instruct}"
WITH_LIVE_SMOKE=0
SKIP_INSTALL=0
CONTINUE_ON_FAILURE=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/entrypoints/run_architecture_demo.sh [options]

Options:
  --output-root PATH
  --main-evidence PATH
  --boundary-pass PATH
  --boundary-fail PATH
  --summary PATH
  --model PATH_OR_HF_ID
  --with-live-smoke
  --skip-install
  --continue-on-failure
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --main-evidence)
      MAIN_EVIDENCE="$2"
      shift 2
      ;;
    --boundary-pass)
      BOUNDARY_PASS="$2"
      shift 2
      ;;
    --boundary-fail)
      BOUNDARY_FAIL="$2"
      shift 2
      ;;
    --summary)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --with-live-smoke)
      WITH_LIVE_SMOKE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --continue-on-failure)
      CONTINUE_ON_FAILURE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/architecture_demo_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_ROOT"
COMMAND_LOG="$OUTPUT_ROOT/commands.log"
VM_SMOKE_DIR="$OUTPUT_ROOT/vm_smoke"
LIVE_SMOKE_DIR="$OUTPUT_ROOT/live_smoke"
COMMAND_ARGS=("bash scripts/entrypoints/run_architecture_demo.sh")

log() {
  echo "[$(date -Is)] $*" | tee -a "$COMMAND_LOG"
}

record_command() {
  printf '%s\n' "$*" >> "$OUTPUT_ROOT/.demo_commands"
}

run_cmd() {
  local name="$1"
  shift
  local log_file="$OUTPUT_ROOT/${name}.log"
  log "RUN ${name}: $*"
  record_command "$*"
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

run_cmd_allow_failure() {
  local name="$1"
  shift
  local old_continue="$CONTINUE_ON_FAILURE"
  CONTINUE_ON_FAILURE=1
  run_cmd "$name" "$@"
  local status=$?
  CONTINUE_ON_FAILURE="$old_continue"
  log "ALLOW_FAILURE ${name}: exit=${status}"
  return 0
}

ensure_python() {
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [[ -f configs/dgx_spark_env.sh ]]; then
    # shellcheck disable=SC1091
    source configs/dgx_spark_env.sh
  fi
  export ASTRAKV_MODEL="$MODEL"
  export PATH="$ROOT/.venv/bin:$PATH"
  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    run_cmd install python -m pip install --upgrade pip
    run_cmd install_requirements python -m pip install -r requirements.txt pytest tabulate
  fi
}

build_report() {
  local report_args=(
    scripts/reporting/build_architecture_demo_report.py
    --output-dir "$OUTPUT_ROOT"
    --main-evidence "$MAIN_EVIDENCE"
    --boundary-pass "$BOUNDARY_PASS"
    --boundary-fail "$BOUNDARY_FAIL"
    --summary "$SUMMARY_PATH"
    --command-log "$COMMAND_LOG"
  )
  if [[ -d "$VM_SMOKE_DIR" ]]; then
    report_args+=(--include-vm-smoke "$VM_SMOKE_DIR")
  fi
  if [[ -d "$LIVE_SMOKE_DIR" ]]; then
    report_args+=(--include-live-smoke "$LIVE_SMOKE_DIR")
  fi
  if [[ -f "$OUTPUT_ROOT/.demo_commands" ]]; then
    while IFS= read -r command_line; do
      [[ -n "$command_line" ]] && report_args+=(--command "$command_line")
    done < "$OUTPUT_ROOT/.demo_commands"
  fi
  run_cmd build_demo_report python "${report_args[@]}"
}

main() {
  : > "$COMMAND_LOG"
  : > "$OUTPUT_ROOT/.demo_commands"
  log "Architecture demo output root: $OUTPUT_ROOT"
  log "Main evidence: $MAIN_EVIDENCE"
  log "Boundary pass: $BOUNDARY_PASS"
  log "Boundary fail: $BOUNDARY_FAIL"

  ensure_python

  run_cmd shell_syntax bash -n scripts/entrypoints/run_competition_e2e.sh scripts/entrypoints/run_competition_extended_evidence.sh scripts/entrypoints/run_architecture_demo.sh
  run_cmd reporting_tests python -m pytest \
    tests/test_reporting_tools.py \
    tests/test_policy_ablation.py \
    tests/test_competition_report.py \
    tests/test_quality_evaluation.py \
    -q
  run_cmd_allow_failure vm_smoke python scripts/vm/run_mmap_kv_cache.py \
    --output-dir "$VM_SMOKE_DIR" \
    --blocks 32 \
    --block-size-mb 1

  if [[ "$WITH_LIVE_SMOKE" -eq 1 ]]; then
    local e2e_args=(scripts/entrypoints/run_competition_e2e.sh --only smoke --output-root "$LIVE_SMOKE_DIR" --model "$MODEL")
    if [[ "$SKIP_INSTALL" -eq 1 ]]; then e2e_args+=(--skip-install); fi
    if [[ "$CONTINUE_ON_FAILURE" -eq 1 ]]; then e2e_args+=(--continue-on-failure); fi
    run_cmd_allow_failure live_smoke bash "${e2e_args[@]}"
  fi

  build_report
  rm -f "$OUTPUT_ROOT/.demo_commands"
  log "Architecture demo complete: $OUTPUT_ROOT/demo_report.md"
}

main "$@"
