# DGX Spark validation defaults for AstraKV-W.
#
# Source this file before launching real vLLM/LMCache runs:
#   source configs/dgx_spark_env.sh

export ASTRAKV_MODEL="${ASTRAKV_MODEL:-/opt/models/Qwen3-8B}"
export ASTRAKV_HOST="${ASTRAKV_HOST:-127.0.0.1}"
export ASTRAKV_PORT="${ASTRAKV_PORT:-8000}"
export ASTRAKV_MAX_MODEL_LEN="${ASTRAKV_MAX_MODEL_LEN:-32768}"

# DGX Spark uses coherent unified memory. Start conservative so vLLM, LMCache,
# OS page cache, and benchmark clients are not squeezed during smoke tests.
export ASTRAKV_GPU_MEMORY_UTILIZATION="${ASTRAKV_GPU_MEMORY_UTILIZATION:-0.60}"
export ASTRAKV_TENSOR_PARALLEL_SIZE="${ASTRAKV_TENSOR_PARALLEL_SIZE:-1}"

# Disk-tier experiments should use local NVMe. Override this path when the
# official run stores LMCache artifacts on a dedicated mount.
export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK:-results/lmcache_gpu_ssd_store}"
export LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-$LMCACHE_LOCAL_DISK}"

# The scripts default to python when inside .venv. Override if your DGX Spark
# environment has a specific interpreter.
export ASTRAKV_PYTHON="${ASTRAKV_PYTHON:-python3}"
