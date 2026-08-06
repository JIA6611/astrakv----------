#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

pick_python() {
  local candidate=""
  for candidate in \
    "${ASTRAKV_PYTHON:-}" \
    "$ROOT/.venv_from_szl/bin/python" \
    "$ROOT/.venv/bin/python"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  printf '%s\n' "$ROOT/.venv_from_szl/bin/python"
}

PYTHON="$(pick_python)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$ROOT/results/runtime-action-validation-suite-$TIMESTAMP"
MODEL="/opt/models/Qwen3-8B"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="32768"
GPU_MEMORY_UTILIZATION="0.60"
TIMEOUT="900"
PREFIX_CACHING="false"
SCENARIOS="hot_load,cpu_offload,ssd_prefetch,cold_drop,recompute_bias,evict_cold_disk"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_runtime_action_validation_suite.sh [options]

Runs a DGX-oriented runtime-action validation suite covering:
  hot_load, cpu_offload, ssd_prefetch, cold_drop, recompute_bias, evict_cold_disk

Options:
  --output-dir PATH             Result directory root
  --model PATH                  Local model path
  --host HOST                   Loopback bind host (default 127.0.0.1)
  --port PORT                   vLLM port reused serially (default 18000)
  --context-port PORT           Runtime context port reused serially (default 17900)
  --max-model-len N             vLLM maximum sequence length (default 32768)
  --gpu-memory-utilization N    vLLM GPU memory fraction (default 0.60)
  --prefix-caching BOOL         Enable vLLM prefix caching (default false)
  --scenarios CSV               Comma-separated scenarios to run (default all six)
  --timeout SECONDS             Per-request timeout (default 900)
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --context-port) CONTEXT_PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --prefix-caching) PREFIX_CACHING="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a REQUESTED_SCENARIOS <<< "$SCENARIOS"
declare -A SCENARIO_ENABLED=()
for scenario in "${REQUESTED_SCENARIOS[@]}"; do
  case "$scenario" in
    hot_load|cpu_offload|ssd_prefetch|cold_drop|recompute_bias|evict_cold_disk)
      SCENARIO_ENABLED["$scenario"]=1
      ;;
    "")
      ;;
    *)
      echo "Unknown scenario: $scenario" >&2
      exit 2
      ;;
  esac
done
[[ "${#SCENARIO_ENABLED[@]}" -gt 0 ]] || {
  echo "At least one scenario must be selected" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Model path is missing: $MODEL" >&2; exit 2; }
[[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]] || { echo "Only loopback hosts are allowed" >&2; exit 2; }
"$PYTHON" -c "import vllm" >/dev/null 2>&1 || { echo "Selected Python cannot import vllm: $PYTHON" >&2; exit 2; }

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

require_port_free() {
  if curl --max-time 2 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Refusing to start: endpoint already responds on http://${HOST}:${PORT}/v1/models" >&2
    exit 2
  fi
}

wait_for_endpoint() {
  local log_path="$1"
  for _ in $(seq 1 180); do
    if curl --max-time 3 -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -n 200 "$log_path" >&2 || true
  return 1
}

generate_workloads() {
  local workload_dir="$1"
  local generated_dir="$workload_dir/generated"
  local generated_jsonl="$generated_dir/runtime_action_validation_workload.jsonl"
  mkdir -p "$generated_dir"
  "$PYTHON" scripts/benchmark/generate_runtime_action_workload.py --output-dir "$generated_dir"
  "$PYTHON" - "$generated_jsonl" "$workload_dir" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)
rows_by_scenario: dict[str, list[dict]] = {}
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    scenario = str((row.get("metadata") or {}).get("scenario") or "")
    rows_by_scenario.setdefault(scenario, []).append(row)
expected_counts = {
    "hot_load": 2,
    "cpu_offload": 1,
    "ssd_prefetch": 2,
    "cold_drop": 1,
    "recompute_bias": 1,
    "evict_cold_disk": 2,
}
for scenario, expected in expected_counts.items():
    rows = rows_by_scenario.get(scenario, [])
    rows.sort(key=lambda item: int(item.get("arrival_index") or 0))
    if len(rows) != expected:
        raise SystemExit(f"scenario {scenario} expected {expected} rows, got {len(rows)}")
    target = target_dir / f"{scenario}.jsonl"
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
PY
}

write_runtime_env() {
  local run_id="$1" state_dir="$2" runtime_env="$3" secret="$4" workload_id="$5"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
ASTRAKV_ENABLE_ONLINE_POLICY=true
ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=true
ASTRAKV_ONLINE_OFFLINE_GATE_PATH=$state_dir/offline_gate.json
EOF
  if [[ "$workload_id" == "runtime_action_ssd_prefetch" || "$workload_id" == "runtime_action_evict_cold_disk" ]]; then
    printf '%s\n' "ASTRAKV_RUNTIME_DISABLE_NATIVE_REQUEST_LOAD=true" >> "$runtime_env"
  fi
  cat > "$state_dir/offline_gate.json" <<EOF
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["$workload_id","runtime_action_validation_suite"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
  chmod 600 "$runtime_env"
}

start_server() {
  local scenario="$1" run_id="$2" state_dir="$3" runtime_env="$4" cache_dir="$5" log_path="$6"
  cleanup
  require_port_free
  mkdir -p "$cache_dir"
  # All runtime actions share one two-tier topology.  Objects are stored on
  # SSD first, while LocalCPUBackend remains enabled as the real prefetch
  # target.  The runtime owner and action endpoint are unchanged for every
  # scenario.
  local lmcache_config="configs/lmcache_runtime_action_validation.yaml"
  sed "s|^local_disk:.*|local_disk: $cache_dir|" "$lmcache_config" > "$state_dir/lmcache.yaml"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  ASTRAKV_MODEL="$MODEL" \
  ASTRAKV_HOST="$HOST" \
  ASTRAKV_PORT="$PORT" \
  ASTRAKV_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  ASTRAKV_PREFIX_CACHING="$PREFIX_CACHING" \
  ASTRAKV_ENABLE_LMCACHE047_HOOKS=true \
  ASTRAKV_LMCACHE047_EVENTS="$state_dir/hook_raw.jsonl" \
  ASTRAKV_RUNTIME_CONTROL_RUN_ID="$run_id" \
  ASTRAKV_RUNTIME_CONTROL_STATE_DIR="$state_dir" \
  ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="$CONTEXT_PORT" \
  LMCACHE_CONFIG_FILE="$state_dir/lmcache.yaml" \
  ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  nohup bash scripts/launch/launch_lmcache_vllm.sh disk > "$log_path" 2>&1 < /dev/null &
  SERVER_PID="$!"
  wait_for_endpoint "$log_path"
}

verify_variant_scenario() {
  local scenario="$1" run_dir="$2" state_dir="$3"
  "$PYTHON" - "$scenario" "$run_dir" "$state_dir" <<'PY'
import json
import sys
from pathlib import Path

scenario = sys.argv[1]
run_dir = Path(sys.argv[2])
state_dir = Path(sys.argv[3])

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

request_rows = read_jsonl(run_dir / "request_results.jsonl")
if not request_rows:
    raise SystemExit(f"{scenario}: missing request results")
if any(str(row.get("status") or "") != "ok" for row in request_rows):
    raise SystemExit(f"{scenario}: all requests must succeed")

commands = read_jsonl(state_dir / "astrakv_runtime_commands.jsonl")
receipts = read_jsonl(state_dir / "runtime_command_receipts.jsonl")
structured = read_jsonl(state_dir / "runtime_structured_events.jsonl")

expected = {
    "cpu_offload": ("offload", "offload"),
    "ssd_prefetch": ("prefetch", "prefetch"),
    "cold_drop": ("drop", "drop"),
    "evict_cold_disk": ("evict", "evict"),
}
evidence = {
    "schema": "astrakv-runtime-action-scenario-evidence-v1",
    "scenario": scenario,
    "request_results_path": str(run_dir / "request_results.jsonl"),
    "state_dir": str(state_dir),
    "request_count": len(request_rows),
    "successful_request_count": sum(1 for row in request_rows if str(row.get("status") or "") == "ok"),
    "command_count": len(commands),
    "receipt_count": len(receipts),
    "structured_event_count": len(structured),
}
if scenario in expected:
    receipt_action, structured_action = expected[scenario]
    matched_receipt = next((
        row for row in receipts
        if row.get("action") == receipt_action and row.get("status") == "completed"
    ), None)
    if matched_receipt is None:
        raise SystemExit(f"{scenario}: missing completed receipt for {receipt_action}")
    matched_structured = next((
        row for row in structured
        if row.get("actual_action") == structured_action and row.get("status") == "completed"
    ), None)
    if matched_structured is None:
        raise SystemExit(f"{scenario}: missing completed structured event for {structured_action}")
    evidence.update({
        "evidence_type": "completed_receipt",
        "expected_action": receipt_action,
        "receipt": matched_receipt,
        "structured_event": matched_structured,
    })
elif scenario == "recompute_bias":
    checkpoint = state_dir / "online_profile_checkpoint.json"
    if not checkpoint.is_file():
        raise SystemExit(f"{scenario}: missing online profile checkpoint")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    dispatches = payload.get("dispatches") or []
    matched = next((
        row for row in dispatches
        if row.get("predicted_action") == "recompute"
        and row.get("status") == "no_dispatch_required"
    ), None)
    if matched is None:
        raise SystemExit(f"{scenario}: missing recompute no-dispatch evidence")
    if any(str(row.get("action") or "") == "recompute" for row in commands):
        raise SystemExit(f"{scenario}: recompute must not dispatch a backend command")
    evidence.update({
        "evidence_type": "strategy_no_dispatch",
        "expected_action": "recompute",
        "dispatch": matched,
        "recompute_backend_command_count": sum(
            1 for row in commands if str(row.get("action") or "") == "recompute"
        ),
    })
else:
    raise SystemExit(f"unsupported scenario: {scenario}")

(state_dir / "scenario_evidence.json").write_text(
    json.dumps(evidence, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY
}

run_variant_scenario() {
  local scenario="$1" workload_path="$2" output_root="$3"
  local run_id="runtime-action-${scenario}-variant-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$output_root/$scenario/variant"
  local state_dir="$output_root/$scenario/variant-state"
  local cache_dir="$output_root/$scenario/variant-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$output_root/$scenario/variant-server.log"
  mkdir -p "$run_dir" "$state_dir"
  umask 077
  local secret
  secret="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  write_runtime_env "$run_id" "$state_dir" "$runtime_env" "$secret" "runtime_action_${scenario}"
  start_server "$scenario" "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  local -a benchmark_args=(
    "$PYTHON" scripts/benchmark/run_real_benchmark.py
    --base-url "http://${HOST}:${PORT}/v1"
    --model "$MODEL" --backend "vllm-lmcache047"
    --output-dir "$run_dir" --workload-jsonl "$workload_path"
    --run-id "$run_id" --workload-id "runtime_action_${scenario}"
    --model-revision "local-qwen3-8b" --tokenizer-revision "local-qwen3-8b"
    --dtype "bfloat16" --quantization "unquantized" --random-seed "0" --cache-state cold
    --connector-version "lmcache-vllm-v1-0.4.7" --claim-scope online_control
    --runtime-state-dir "$state_dir"
    --enable-samples --metrics-interval 1.0
    --timeout "$TIMEOUT" --output-tokens 64
  )
  # Prefetch validation keeps request identity and store/release lifecycle
  # context, while ASTRAKV_RUNTIME_DISABLE_NATIVE_REQUEST_LOAD suppresses
  # only the direct native load-target path.
  benchmark_args+=(
    --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context"
    --request-context-session-id "${run_id}-session"
    --request-context-secret-hex "$secret"
  )
  "${benchmark_args[@]}"
  cleanup
  verify_variant_scenario "$scenario" "$run_dir" "$state_dir"
}

run_hot_load_pair() {
  local output_root="$1"
  bash scripts/entrypoints/run_runtime_action_hot_load_smoke.sh \
    --output-dir "$output_root/hot_load" \
    --model "$MODEL" --host "$HOST" --port "$PORT" --context-port "$CONTEXT_PORT" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --prefix-caching "$PREFIX_CACHING" --timeout "$TIMEOUT"
  "$PYTHON" - "$output_root/hot_load" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "smoke_summary.json"
manifest_path = root / "comparison" / "paired_run_manifest.json"
if not summary_path.is_file():
    raise SystemExit(f"hot_load: missing smoke summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("classification") == "functional_fail":
    raise SystemExit("hot_load: smoke summary reports functional failure")
if not manifest_path.is_file():
    raise SystemExit(f"hot_load: missing paired run manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("eligible") is not True:
    raise SystemExit("hot_load: paired run manifest is not eligible")
PY
}

write_suite_summary() {
  local output_root="$1" selected_scenarios="$2"
  "$PYTHON" - "$output_root" "$selected_scenarios" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected = {
    item.strip() for item in str(sys.argv[2] or "").split(",") if item.strip()
}
scenarios = ["hot_load", "cpu_offload", "ssd_prefetch", "cold_drop", "recompute_bias", "evict_cold_disk"]
coverage = {}

if (root / "hot_load" / "smoke_summary.json").is_file():
    hot_summary = json.loads((root / "hot_load" / "smoke_summary.json").read_text(encoding="utf-8"))
    hot_manifest = json.loads((root / "hot_load" / "comparison" / "paired_run_manifest.json").read_text(encoding="utf-8"))
    coverage["hot_load"] = {
        "classification": hot_summary.get("classification"),
        "paired_claim_eligible": bool(hot_manifest.get("eligible") is True),
        "baseline_revisit_ttft_ms": hot_summary.get("baseline_revisit_ttft_ms"),
        "variant_revisit_ttft_ms": hot_summary.get("variant_revisit_ttft_ms"),
        "evidence_source": "existing_hot_load_smoke",
    }

for scenario in scenarios[1:]:
    if scenario not in selected and not (root / scenario / "variant-state").is_dir():
        continue
    state_dir = root / scenario / "variant-state"
    run_dir = root / scenario / "variant"
    evidence_path = state_dir / "scenario_evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    else:
        # Older completed runs predate scenario_evidence.json. Derive a
        # read-only evidence record from their immutable runtime artifacts.
        receipts_path = state_dir / "runtime_command_receipts.jsonl"
        structured_path = state_dir / "runtime_structured_events.jsonl"
        checkpoint_path = state_dir / "online_profile_checkpoint.json"
        def read_jsonl(path):
            if not path.is_file():
                return []
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        receipts = read_jsonl(receipts_path)
        structured = read_jsonl(structured_path)
        evidence = {
            "schema": "astrakv-runtime-action-scenario-evidence-v1",
            "scenario": scenario,
            "request_results_path": str(run_dir / "request_results.jsonl"),
            "state_dir": str(state_dir),
            "evidence_type": "legacy_runtime_artifacts",
            "request_count": 0,
            "successful_request_count": 0,
            "command_count": 0,
            "receipt_count": len(receipts),
            "structured_event_count": len(structured),
        }
        if scenario == "recompute_bias" and checkpoint_path.is_file():
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            dispatch = next((
                row for row in payload.get("dispatches") or []
                if row.get("predicted_action") == "recompute"
                and row.get("status") == "no_dispatch_required"
            ), None)
            commands = read_jsonl(state_dir / "astrakv_runtime_commands.jsonl")
            evidence.update({
                "evidence_type": "legacy_runtime_artifacts",
                "expected_action": "recompute",
                "dispatch": dispatch,
                "recompute_backend_command_count": sum(
                    1 for row in commands if str(row.get("action") or "") == "recompute"
                ),
            })
        elif scenario == "cpu_offload":
            receipt = next((
                row for row in receipts
                if row.get("action") == "offload"
                and row.get("status") == "completed"
            ), None)
            event = next((
                row for row in structured
                if row.get("actual_action") == "offload"
                and row.get("status") == "completed"
            ), None)
            evidence.update({
                "evidence_type": "legacy_runtime_artifacts",
                "expected_action": "offload",
                "receipt": receipt,
                "structured_event": event,
            })
        else:
            expected_legacy_action = {
                "ssd_prefetch": "prefetch",
                "cold_drop": "drop",
                "evict_cold_disk": "evict",
            }.get(scenario)
            receipt = next((
                row for row in receipts
                if row.get("action") == expected_legacy_action
                and row.get("status") == "completed"
            ), None)
            event = next((
                row for row in structured
                if row.get("actual_action") == expected_legacy_action
                and row.get("status") == "completed"
            ), None)
            evidence.update({
                "evidence_type": "legacy_runtime_artifacts",
                "expected_action": expected_legacy_action,
                "receipt": receipt,
                "structured_event": event,
            })
    expected_action = {
        "cpu_offload": "offload",
        "ssd_prefetch": "prefetch",
        "cold_drop": "drop",
        "evict_cold_disk": "evict",
        "recompute_bias": "recompute",
    }[scenario]
    if evidence.get("scenario") != scenario or evidence.get("expected_action") != expected_action:
        raise SystemExit(f"{scenario}: scenario evidence identity mismatch")
    if (
        evidence.get("request_count")
        and evidence.get("successful_request_count") != evidence.get("request_count")
    ):
        raise SystemExit(f"{scenario}: scenario evidence contains failed requests")
    if scenario == "recompute_bias":
        dispatch = evidence.get("dispatch") or {}
        if dispatch.get("predicted_action") != "recompute" or dispatch.get("status") != "no_dispatch_required":
            raise SystemExit(f"{scenario}: invalid recompute no-dispatch evidence")
        if evidence.get("recompute_backend_command_count") != 0:
            raise SystemExit(f"{scenario}: recompute backend command evidence is not empty")
    else:
        receipt = evidence.get("receipt") or {}
        structured_event = evidence.get("structured_event") or {}
        if receipt.get("action") != expected_action or receipt.get("status") != "completed":
            raise SystemExit(f"{scenario}: invalid completed receipt evidence")
        if structured_event.get("actual_action") != expected_action or structured_event.get("status") != "completed":
            raise SystemExit(f"{scenario}: invalid structured event evidence")
    coverage[scenario] = {
        "request_results": str(run_dir / "request_results.jsonl"),
        "state_dir": str(state_dir),
        "evidence_path": str(evidence_path),
        "evidence_type": evidence.get("evidence_type"),
        "expected_action": expected_action,
        "verified": True,
        "evidence": evidence,
    }

all_verified = all(
    bool(item.get("paired_claim_eligible", True))
    and item.get("classification") != "functional_fail"
    and item.get("verified") is True
    for item in coverage.values()
)
summary = {
    "schema": "astrakv-runtime-action-validation-suite-v1",
    "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    "all_verified": all_verified,
    "coverage": coverage,
}
summary_path = root / "suite_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
report_path = root / "suite_summary.md"
lines = [
    "# Runtime Action Validation Suite Summary",
    "",
    f"- All verified: `{all_verified}`",
]
hot_load = coverage.get("hot_load")
if hot_load:
    lines.append(f"- Hot load classification: `{hot_load.get('classification')}`")
    lines.append(f"- Hot load paired claim eligible: `{hot_load.get('paired_claim_eligible')}`")
else:
    lines.append("- Hot load classification: `not_run`")
    lines.append("- Hot load paired claim eligible: `false`")

for scenario in scenarios[1:]:
    if scenario in coverage:
        lines.append(f"- {scenario}: `verified`")
    else:
        lines.append(f"- {scenario}: `not_run`")
report_path.write_text("\n".join(lines), encoding="utf-8")
print(summary_path)
PY
}

main() {
  mkdir -p "$OUTPUT_DIR"
  local workload_dir="$OUTPUT_DIR/workload"
  generate_workloads "$workload_dir"

  if [[ -n "${SCENARIO_ENABLED[hot_load]:-}" ]]; then
    run_hot_load_pair "$OUTPUT_DIR"
  fi
  if [[ -n "${SCENARIO_ENABLED[cpu_offload]:-}" ]]; then
    run_variant_scenario "cpu_offload" "$workload_dir/cpu_offload.jsonl" "$OUTPUT_DIR"
  fi
  if [[ -n "${SCENARIO_ENABLED[ssd_prefetch]:-}" ]]; then
    run_variant_scenario "ssd_prefetch" "$workload_dir/ssd_prefetch.jsonl" "$OUTPUT_DIR"
  fi
  if [[ -n "${SCENARIO_ENABLED[cold_drop]:-}" ]]; then
    run_variant_scenario "cold_drop" "$workload_dir/cold_drop.jsonl" "$OUTPUT_DIR"
  fi
  if [[ -n "${SCENARIO_ENABLED[recompute_bias]:-}" ]]; then
    run_variant_scenario "recompute_bias" "$workload_dir/recompute_bias.jsonl" "$OUTPUT_DIR"
  fi
  if [[ -n "${SCENARIO_ENABLED[evict_cold_disk]:-}" ]]; then
    run_variant_scenario "evict_cold_disk" "$workload_dir/evict_cold_disk.jsonl" "$OUTPUT_DIR"
  fi

  local summary_path
  summary_path="$(write_suite_summary "$OUTPUT_DIR" "$SCENARIOS")"
  echo "Runtime action validation suite artifacts: $OUTPUT_DIR"
  echo "Suite summary: $summary_path"
}

main "$@"
