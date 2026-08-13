#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DEFAULT_PYTHON="$ROOT/.venv/bin/python"
# Keep the historical git-common-dir probe in place for compatibility checks.
GIT_COMMON_DIR="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
PYTHON="${ASTRAKV_PYTHON:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
  if [[ -x "$ROOT/.venv_from_szl/bin/python" ]] && "$ROOT/.venv_from_szl/bin/python" -c "import vllm" >/dev/null 2>&1; then
    PYTHON="$ROOT/.venv_from_szl/bin/python"
  elif [[ -n "${ASTRAKV_PYTHON:-}" && -x "${ASTRAKV_PYTHON:-}" ]]; then
    PYTHON="${ASTRAKV_PYTHON}"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "import vllm" >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  fi
fi
# Ensure JIT subprocesses can resolve tools installed with the selected Python.
export PATH="$(dirname "$PYTHON"):${PATH}"
MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
FALLBACK_MODEL="${ASTRAKV_FALLBACK_MODEL:-/opt/models/Qwen2.5-7B-Instruct}"
HOST="${ASTRAKV_HOST:-127.0.0.1}"
PORT="${ASTRAKV_PORT:-8000}"
GPU_MEMORY="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.78}"
KV_CACHE_MEMORY_BYTES="${ASTRAKV_KV_CACHE_MEMORY_BYTES:-}"
MAX_MODEL_LEN="${ASTRAKV_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL="${ASTRAKV_TENSOR_PARALLEL_SIZE:-1}"
KV_TRANSFER_CONFIG="${ASTRAKV_KV_TRANSFER_CONFIG:-}"
PREFIX_CACHING="${ASTRAKV_PREFIX_CACHING:-true}"
VLLM_DEV_MODE="${ASTRAKV_VLLM_DEV_MODE:-false}"
LMCACHE047_HOOKS="${ASTRAKV_ENABLE_LMCACHE047_HOOKS:-false}"
RUNTIME_CONTROL_RUN_ID="${ASTRAKV_RUNTIME_CONTROL_RUN_ID:-}"
VLLM_SEED="${ASTRAKV_VLLM_SEED:-0}"

case "$PREFIX_CACHING" in
  true|false) ;;
  *) echo "ASTRAKV_PREFIX_CACHING must be true or false, got: $PREFIX_CACHING" >&2; exit 2 ;;
esac
case "$VLLM_DEV_MODE" in
  true|false) ;;
  *) echo "ASTRAKV_VLLM_DEV_MODE must be true or false, got: $VLLM_DEV_MODE" >&2; exit 2 ;;
esac
case "$LMCACHE047_HOOKS" in
  true|false) ;;
  *) echo "ASTRAKV_ENABLE_LMCACHE047_HOOKS must be true or false, got: $LMCACHE047_HOOKS" >&2; exit 2 ;;
esac
if [[ "$LMCACHE047_HOOKS" == "true" ]]; then
  # The vendor-patched connector imports AstraKV directly.  Do not prepend
  # scripts/runtime: it contains sitecustomize.py, whose legacy monkey patch
  # must never own KV-Core lifecycle or socket state.
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export ASTRAKV_LMCACHE047_EVENTS="${ASTRAKV_LMCACHE047_EVENTS:-$ROOT/results/lmcache047_events.jsonl}"
  export ASTRAKV_KV_CORE_VENDOR_PATCH="${ASTRAKV_KV_CORE_VENDOR_PATCH:-true}"
fi
if ! [[ "$VLLM_SEED" =~ ^[0-9]+$ ]]; then
  echo "ASTRAKV_VLLM_SEED must be a non-negative integer, got: $VLLM_SEED" >&2
  exit 2
fi
if [[ -n "$RUNTIME_CONTROL_RUN_ID" ]]; then
  if [[ "$LMCACHE047_HOOKS" != "true" ]]; then
    echo "ASTRAKV_RUNTIME_CONTROL_RUN_ID requires ASTRAKV_ENABLE_LMCACHE047_HOOKS=true" >&2
    exit 2
  fi
  for required in ASTRAKV_RUNTIME_CONTROL_STATE_DIR ASTRAKV_RUNTIME_CONTROL_SECRET_HEX ASTRAKV_RUNTIME_CONTROL_ENGINE_ID ASTRAKV_RUNTIME_CONTROL_WORKER_ID; do
    if [[ -z "${!required:-}" ]]; then
      echo "ASTRAKV_RUNTIME_CONTROL_RUN_ID requires ${required}" >&2
      exit 2
    fi
  done
  export ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE="engine_child"
fi
if [[ "$VLLM_DEV_MODE" == "true" ]]; then
  export VLLM_SERVER_DEV_MODE=1
fi

# Use HF mirror to bypass GFW block on huggingface.co and Xet CDN (cas-bridge.xethub.hf.co)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# LMCache falls back to Python's builtin hash on this deployment.  A fixed
# process-independent seed is required for the same token prefix to resolve
# to the same external LMCache key after the worker is restarted.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Python dev headers (python3.12-dev) for Triton/CUDA JIT compilation
# Installed locally at ~/.local/include to avoid sudo requirement
# Both the main headers and architecture-specific (aarch64) headers are needed
export C_INCLUDE_PATH="${HOME}/.local/include/python3.12:${HOME}/.local/include:${C_INCLUDE_PATH:-}"
export CPATH="${HOME}/.local/include/python3.12:${HOME}/.local/include:${CPATH:-}"

echo "Launching vLLM server"
echo "Model=${MODEL} Host=${HOST} Port=${PORT} MaxModelLen=${MAX_MODEL_LEN}"
echo "Explicit fallback model=${FALLBACK_MODEL}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "PrefixCaching=${PREFIX_CACHING} VllmDevMode=${VLLM_DEV_MODE}"
echo "VllmSeed=${VLLM_SEED}"
echo "LMCache047Hooks=${LMCACHE047_HOOKS}"
if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  echo "KvCacheMemoryBytes=${KV_CACHE_MEMORY_BYTES}"
fi
if [[ -n "$RUNTIME_CONTROL_RUN_ID" ]]; then
  echo "RuntimeControlRunId=${RUNTIME_CONTROL_RUN_ID}"
fi

CMD=(
  "$PYTHON" -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype auto
  --seed "$VLLM_SEED"
  --gpu-memory-utilization "$GPU_MEMORY"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --trust-remote-code
)

if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  CMD+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if [[ "$PREFIX_CACHING" == "true" ]]; then
  CMD+=(--enable-prefix-caching)
else
  CMD+=(--no-enable-prefix-caching)
fi

if [[ -n "$RUNTIME_CONTROL_RUN_ID" ]]; then
  CMD+=(--enable-request-id-headers)
fi

if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  echo "KV transfer config=${KV_TRANSFER_CONFIG}"
  CMD+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi

exec "${CMD[@]}"
