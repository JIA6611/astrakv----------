#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WITH_REAL=0
SKIP_INSTALL=0
OUTPUT_DIR="results/dgx_spark_validation"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-real)
      WITH_REAL=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON="${ASTRAKV_PYTHON:-python}"

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt pytest tabulate
fi

ENV_REPORT="$OUTPUT_DIR/environment_report.txt"
{
  echo "AstraKV-W DGX Spark validation"
  echo "date=$(date -Is)"
  echo "uname=$(uname -a)"
  echo "python=$(python --version 2>&1)"
  echo "machine=$(python -c 'import platform; print(platform.machine())')"
  echo "model=${ASTRAKV_MODEL}"
  echo "max_model_len=${ASTRAKV_MAX_MODEL_LEN}"
  echo "gpu_memory_utilization=${ASTRAKV_GPU_MEMORY_UTILIZATION}"
  echo
  echo "nvidia-smi:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi unavailable"
  fi
  echo
  echo "Python package probes:"
  python - <<'PY'
import importlib
for name in ("numpy", "yaml", "psutil", "matplotlib", "pytest", "tabulate", "torch", "vllm", "lmcache"):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        extra = ""
        if name == "torch":
            extra = f" cuda_available={mod.cuda.is_available()}"
        print(f"{name}: ok version={version}{extra}")
    except Exception as exc:
        print(f"{name}: unavailable {type(exc).__name__}: {exc}")
PY
} | tee "$ENV_REPORT"

echo "[1/4] Running unit tests"
python -m pytest tests/ -q | tee "$OUTPUT_DIR/pytest.log"

echo "[2/4] Running DGX Spark mmap KV evidence"
python scripts/vm/run_dgx_spark_vm_evidence.py \
  --output-dir "$OUTPUT_DIR/vm_evidence" \
  --chunks 8 \
  --total-blocks 64 \
  --block-size-mb 1 | tee "$OUTPUT_DIR/vm_evidence.log"

echo "[3/4] Running legacy mmap smoke"
python cli.py vm mmap \
  --blocks 16 \
  --block-size-mb 1 \
  --output-dir "$OUTPUT_DIR/mmap_smoke" | tee "$OUTPUT_DIR/mmap_smoke.log"

if [[ "$WITH_REAL" -eq 1 ]]; then
  echo "[4/4] Running real endpoint smoke against already-running vLLM"
  python scripts/benchmark/run_real_benchmark.py \
    --config configs/dgx_spark_vllm_qwen7b.yaml \
    --output-dir "$OUTPUT_DIR/real_vllm_smoke" \
    --context-lengths 512 1024 \
    --batch-sizes 1 \
    --output-tokens 32 \
    --repeat 1 | tee "$OUTPUT_DIR/real_vllm_smoke.log"
else
  echo "[4/4] Skipping real endpoint smoke. Re-run with --with-real after vLLM is listening on ${ASTRAKV_HOST}:${ASTRAKV_PORT}."
fi

echo "DGX Spark validation artifacts written to $OUTPUT_DIR"
