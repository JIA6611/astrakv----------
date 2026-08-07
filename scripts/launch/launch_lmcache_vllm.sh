#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BACKEND="${1:-disk}"
if [[ "$BACKEND" == "disk" ]]; then
  export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-configs/lmcache_disk_example.yaml}"
  export ASTRAKV_GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.72}"
  LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-results/lmcache_gpu_ssd_store}"
  mkdir -p "$LMCACHE_DISK_PATH"
elif [[ "$BACKEND" == "cpu" ]]; then
  export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-configs/lmcache_cpu_example.yaml}"
  export ASTRAKV_GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.72}"
else
  echo "Supported LMCache backends are 'disk' (GPU+SSD) and 'cpu' (GPU+CPU+SSD experiment topology)." >&2
  exit 2
fi
if [[ -z "${ASTRAKV_KV_TRANSFER_CONFIG:-}" ]]; then
  export ASTRAKV_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
fi

echo "Launching vLLM with LMCache backend intent: ${BACKEND}"
echo "LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE}"
if [[ -n "$LMCACHE_DISK_PATH" ]]; then
  echo "LMCache disk path=${LMCACHE_DISK_PATH}"
fi
echo "ASTRAKV_KV_TRANSFER_CONFIG=${ASTRAKV_KV_TRANSFER_CONFIG}"
echo "Verify the installed LMCache/vLLM versions accept this connector before official runs."

exec bash scripts/launch/launch_vllm_server.sh
