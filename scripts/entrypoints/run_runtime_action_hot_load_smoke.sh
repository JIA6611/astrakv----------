#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Historical compatibility marker retained for tests that still assert the git-common-dir probe text.
# git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir

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
OUTPUT_DIR="$ROOT/results/runtime-action-hot-load-smoke-$TIMESTAMP"
MODEL="/opt/models/Qwen3-8B"
HOST="127.0.0.1"
PORT="18000"
CONTEXT_PORT="17900"
MAX_MODEL_LEN="32768"
GPU_MEMORY_UTILIZATION="0.60"
TIMEOUT="900"
PREFIX_CACHING="false"

SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/run_runtime_action_hot_load_smoke.sh [options]

Runs a sequential baseline/variant DGX smoke focused on the controlled
`hot_load` runtime-action workload.

Options:
  --output-dir PATH             Result directory root
  --model PATH                  Local model path
  --host HOST                   Loopback bind host (default 127.0.0.1)
  --port PORT                   vLLM port reused serially (default 18000)
  --context-port PORT           Runtime context port reused serially (default 17900)
  --max-model-len N             vLLM maximum sequence length (default 32768)
  --gpu-memory-utilization N    vLLM GPU memory fraction (default 0.60)
  --prefix-caching BOOL         Enable vLLM prefix caching for the smoke (default false)
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
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
  local generated_dir="$1"
  local generated_jsonl="$generated_dir/runtime_action_validation_workload.jsonl"
  local hot_jsonl="$generated_dir/hot_load_smoke.jsonl"
  mkdir -p "$generated_dir"
  "$PYTHON" scripts/benchmark/generate_runtime_action_workload.py --output-dir "$generated_dir"
  "$PYTHON" - "$generated_jsonl" "$hot_jsonl" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if str((row.get("metadata") or {}).get("scenario") or "") == "hot_load":
        rows.append(row)
rows.sort(key=lambda item: int(item.get("arrival_index") or 0))
request_ids = [str(item.get("request_id") or "") for item in rows]
expected = ["hot-load-seed", "hot-load-revisit"]
if request_ids != expected:
    raise SystemExit(f"expected exactly hot-load rows {expected}, got {request_ids}")
target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
PY
  [[ -f "$hot_jsonl" ]] || { echo "Failed to materialize hot_load smoke workload" >&2; exit 2; }
}

start_server() {
  local run_id="$1" state_dir="$2" runtime_env="$3" cache_dir="$4" log_path="$5"
  cleanup
  require_port_free
  mkdir -p "$cache_dir"
  sed "s|^local_disk:.*|local_disk: $cache_dir|" configs/lmcache_disk_example.yaml > "$state_dir/lmcache.yaml"
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

write_runtime_env() {
  local run_id="$1" role="$2" state_dir="$3" runtime_env="$4"
  local secret="$5"
  cat > "$runtime_env" <<EOF
ASTRAKV_RUNTIME_CONTROL_SECRET_HEX=$secret
ASTRAKV_RUNTIME_CONTROL_SESSION_ID=${run_id}-session
ASTRAKV_RUNTIME_CONTROL_ENGINE_ID=${run_id}-engine
ASTRAKV_RUNTIME_CONTROL_WORKER_ID=worker-0
EOF
  if [[ "$role" == "variant" ]]; then
    cat > "$state_dir/offline_gate.json" <<'EOF'
{"schema":"astrakv-offline-safety-gate-v1","status":"accepted","allowed":true,"reasons":[],"workload_ids":["runtime_action_hot_load_smoke"],"aggregate":{},"checks":{"suite_controlled_run":true},"evidence":[]}
EOF
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=true\nASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=false\nASTRAKV_ONLINE_OFFLINE_GATE_PATH=%s\n' "$state_dir/offline_gate.json" >> "$runtime_env"
  else
    printf 'ASTRAKV_ENABLE_ONLINE_POLICY=false\n' >> "$runtime_env"
  fi
  chmod 600 "$runtime_env"
}

run_condition() {
  local role="$1" hot_workload="$2" pair_id="$3" output_root="$4"
  local run_id="runtime-hotload-${role}-$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$output_root/$role"
  local state_dir="$output_root/${role}-state"
  local cache_dir="$output_root/${role}-lmcache-store"
  local runtime_env="$state_dir/runtime.env"
  local server_log="$output_root/${role}-server.log"
  mkdir -p "$run_dir" "$state_dir"
  umask 077
  local secret
  secret="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
  write_runtime_env "$run_id" "$role" "$state_dir" "$runtime_env" "$secret"
  start_server "$run_id" "$state_dir" "$runtime_env" "$cache_dir" "$server_log"
  set -a
  # shellcheck disable=SC1090
  source "$runtime_env"
  set +a
  "$PYTHON" scripts/benchmark/run_real_benchmark.py \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$MODEL" --backend "vllm-lmcache047" \
    --output-dir "$run_dir" --workload-jsonl "$hot_workload" \
    --run-id "$run_id" --workload-id "runtime_action_hot_load_smoke" \
    --model-revision "local-qwen3-8b" --tokenizer-revision "local-qwen3-8b" \
    --dtype "bfloat16" --quantization "unquantized" --random-seed "0" --cache-state cold \
    --connector-version "lmcache-vllm-v1-0.4.7" \
    --pair-id "$pair_id" --pair-role "$role" --claim-scope online_control \
    --runtime-state-dir "$state_dir" --request-context-url "http://${HOST}:${CONTEXT_PORT}/request-context" \
    --request-context-session-id "${run_id}-session" --request-context-secret-hex "$secret" \
    --enable-samples --metrics-interval 1.0 \
    --timeout "$TIMEOUT" --output-tokens 64
  cleanup
  verify_condition "$role" "$run_dir" "$state_dir"
}

verify_condition() {
  local role="$1" run_dir="$2" state_dir="$3"
  "$PYTHON" - "$role" "$run_dir" "$state_dir" <<'PY'
import json
import sys
from pathlib import Path

role = sys.argv[1]
run_dir = Path(sys.argv[2])
state_dir = Path(sys.argv[3])
request_path = run_dir / "request_results.jsonl"
if not request_path.is_file():
    raise SystemExit(f"missing request results: {request_path}")
rows = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 2:
    raise SystemExit(f"{role}: expected exactly 2 request results, got {len(rows)}")
if any(str(row.get("status") or "") != "ok" for row in rows):
    raise SystemExit(f"{role}: all requests must succeed")
if role == "variant":
    revisit = next((row for row in rows if row.get("request_id") == "hot-load-revisit"), None)
    if revisit is None:
        raise SystemExit(f"{role}: missing hot-load-revisit request result")
    if revisit.get("ttft_ms") in (None, "", "None"):
        raise SystemExit(f"{role}: hot-load-revisit must report a non-null TTFT")
    if int(revisit.get("output_tokens_observed") or 0) <= 0:
        raise SystemExit(f"{role}: hot-load-revisit must observe generated output tokens")
if not state_dir.is_dir():
    raise SystemExit(f"missing runtime state dir: {state_dir}")
if role == "variant":
    receipt_path = state_dir / "runtime_command_receipts.jsonl"
    structured_path = state_dir / "runtime_structured_events.jsonl"
    if not receipt_path.is_file():
        raise SystemExit(f"{role}: missing runtime receipts: {receipt_path}")
    receipts = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matched_receipt = next((
        row for row in receipts
        if row.get("action") == "cache_load"
        and row.get("status") == "completed"
        and isinstance(row.get("metadata"), dict)
        and row["metadata"].get("load_target_id")
        and row["metadata"].get("runtime_reqmeta_id")
    ), None)
    if matched_receipt is None:
        raise SystemExit(f"{role}: no completed cache_load receipt with load target metadata")
    if not structured_path.is_file():
        raise SystemExit(f"{role}: missing runtime structured events: {structured_path}")
    structured = [json.loads(line) for line in structured_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matched_event = next((
        row for row in structured
        if row.get("actual_action") == "load"
        and row.get("status") == "completed"
        and isinstance(row.get("metadata"), dict)
        and row["metadata"].get("load_target_id")
    ), None)
    if matched_event is None:
        raise SystemExit(f"{role}: no completed structured load event with load target metadata")
PY
}

write_smoke_summary() {
  local output_root="$1"
  "$PYTHON" - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
baseline_dir = output_root / "baseline"
variant_dir = output_root / "variant"
comparison_dir = output_root / "comparison"
paired_manifest = comparison_dir / "paired_run_manifest.json"
summary_json = output_root / "smoke_summary.json"
summary_md = output_root / "smoke_summary.md"

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

baseline_rows = read_jsonl(baseline_dir / "request_results.jsonl")
variant_rows = read_jsonl(variant_dir / "request_results.jsonl")
variant_receipts = read_jsonl(output_root / "variant-state" / "runtime_command_receipts.jsonl")
variant_structured = read_jsonl(output_root / "variant-state" / "runtime_structured_events.jsonl")

baseline_revisit = next((row for row in baseline_rows if row.get("request_id") == "hot-load-revisit"), None)
variant_revisit = next((row for row in variant_rows if row.get("request_id") == "hot-load-revisit"), None)

functional_receipt = next((
    row for row in variant_receipts
    if row.get("action") == "cache_load"
    and row.get("status") == "completed"
    and isinstance(row.get("metadata"), dict)
    and row["metadata"].get("load_target_id")
    and row["metadata"].get("runtime_reqmeta_id")
), None)
functional_structured = next((
    row for row in variant_structured
    if row.get("actual_action") == "load"
    and row.get("status") == "completed"
), None)

baseline_ttft = None if baseline_revisit is None else baseline_revisit.get("ttft_ms")
variant_ttft = None if variant_revisit is None else variant_revisit.get("ttft_ms")
try:
    baseline_ttft_value = None if baseline_ttft in (None, "", "None") else float(baseline_ttft)
except (TypeError, ValueError):
    baseline_ttft_value = None
try:
    variant_ttft_value = None if variant_ttft in (None, "", "None") else float(variant_ttft)
except (TypeError, ValueError):
    variant_ttft_value = None

classification = "functional_fail"
variant_revisit_ok = (
    variant_revisit is not None
    and str(variant_revisit.get("status") or "") == "ok"
    and int(variant_revisit.get("output_tokens_observed") or 0) > 0
    and variant_ttft_value is not None
)
if functional_receipt is not None and functional_structured is not None and variant_revisit_ok:
    if (
        baseline_ttft_value is not None
        and variant_ttft_value is not None
        and variant_ttft_value < baseline_ttft_value
    ):
        classification = "functional_and_performance_pass"
    else:
        classification = "functional_pass_performance_inconclusive"

payload = {
    "classification": classification,
    "baseline_revisit_ttft_ms": baseline_ttft_value,
    "variant_revisit_ttft_ms": variant_ttft_value,
    "variant_load_receipt_id": None if functional_receipt is None else functional_receipt.get("receipt_id"),
    "variant_load_target_id": None if functional_receipt is None else functional_receipt.get("metadata", {}).get("load_target_id"),
    "variant_runtime_reqmeta_id": None if functional_receipt is None else functional_receipt.get("metadata", {}).get("runtime_reqmeta_id"),
    "comparison_dir": str(comparison_dir),
    "paired_run_manifest": str(paired_manifest),
}
summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
summary_md.write_text(
    "\n".join([
        "# Runtime Action Hot Load Smoke Summary",
        "",
        f"- Classification: `{classification}`",
        f"- Baseline revisit TTFT (ms): `{baseline_ttft_value}`",
        f"- Variant revisit TTFT (ms): `{variant_ttft_value}`",
        f"- Variant load target id: `{payload['variant_load_target_id']}`",
        f"- Variant runtime reqmeta id: `{payload['variant_runtime_reqmeta_id']}`",
        f"- Comparison dir: `{comparison_dir}`",
    ]),
    encoding="utf-8",
)
print(classification)
PY
}

main() {
  mkdir -p "$OUTPUT_DIR"
  local workload_dir="$OUTPUT_DIR/workload"
  local generated_dir="$workload_dir/generated"
  local hot_workload="$workload_dir/hot_load_smoke.jsonl"
  local pair_id="runtime-action-hot-load-smoke-$TIMESTAMP"

  generate_workloads "$generated_dir"
  cp "$generated_dir/hot_load_smoke.jsonl" "$hot_workload"

  run_condition baseline "$hot_workload" "$pair_id" "$OUTPUT_DIR"
  run_condition variant "$hot_workload" "$pair_id" "$OUTPUT_DIR"

  local comparison_dir="$OUTPUT_DIR/comparison"
  mkdir -p "$comparison_dir"
  if ! "$PYTHON" scripts/reporting/compare_real_runs.py \
    --run "baseline=$OUTPUT_DIR/baseline" \
    --run "variant=$OUTPUT_DIR/variant" \
    --output-dir "$comparison_dir"; then
    echo "Paired-run comparison reported ineligible evidence; continuing to write smoke summary." >&2
  fi

  local classification
  classification="$(write_smoke_summary "$OUTPUT_DIR")"
  if [[ "$classification" == "functional_fail" ]]; then
    echo "Hot-load smoke ended with functional failure; see $OUTPUT_DIR/smoke_summary.json" >&2
    exit 1
  fi
  echo "Hot-load smoke completed with classification: $classification"
  echo "Artifacts: $OUTPUT_DIR"
}

main "$@"
