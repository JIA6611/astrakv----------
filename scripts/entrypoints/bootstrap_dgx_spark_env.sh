#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ASTRAKV_VENV_DIR:-.venv}"
INSTALL_VLLM=1
INSTALL_LMCACHE=0
USE_UV=1

usage() {
  cat <<'EOF'
Usage: bash scripts/entrypoints/bootstrap_dgx_spark_env.sh [options]

Create a fresh local Python environment for DGX Spark real vLLM/LMCache runs.

Options:
  --with-lmcache       Also install LMCache for CPU/disk tier experiments.
  --skip-vllm          Install AstraKV-W helper dependencies only.
  --pip-only           Do not use uv for vLLM installation.
  --python-bin PATH    Python interpreter used to create the venv. Default: python3.
  --venv-dir PATH      Virtual environment directory. Default: .venv.
  -h, --help           Show this help.

Environment overrides:
  PYTHON_BIN                         Same as --python-bin.
  ASTRAKV_VENV_DIR                   Same as --venv-dir.
  ASTRAKV_TORCH_INSTALL_COMMAND      Custom torch install command run before vLLM.
                                    Example:
                                    ASTRAKV_TORCH_INSTALL_COMMAND='python -m pip install torch'

The script does not install system Python packages with sudo. If python3 or
python3-venv is missing, install them with your OS package manager first.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-lmcache)
      INSTALL_LMCACHE=1
      shift
      ;;
    --skip-vllm)
      INSTALL_VLLM=0
      shift
      ;;
    --pip-only)
      USE_UV=0
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  cat >&2 <<EOF
Python interpreter not found: $PYTHON_BIN

On Ubuntu/Debian, install Python first:
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip

Then rerun:
  bash scripts/entrypoints/bootstrap_dgx_spark_env.sh
EOF
  exit 127
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, got {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

echo "[1/7] GPU probe"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is unavailable. vLLM GPU serving will not work until NVIDIA drivers are visible."
fi

echo "[2/7] Creating virtual environment: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    cat >&2 <<'EOF'
Failed to create venv. On Ubuntu/Debian this usually means python3-venv is missing:
  sudo apt-get install -y python3-venv
EOF
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/7] Upgrading packaging tools"
# Keep setuptools below 82 because some torch builds still require it.
python -m pip install --upgrade pip wheel "setuptools<82"

echo "[4/7] Installing AstraKV-W package and helper dependencies"
python -m pip install -e .
python -m pip install -r requirements.txt pytest tabulate

if [[ -n "${ASTRAKV_TORCH_INSTALL_COMMAND:-}" ]]; then
  echo "[5/7] Installing torch with custom ASTRAKV_TORCH_INSTALL_COMMAND"
  bash -lc "$ASTRAKV_TORCH_INSTALL_COMMAND"
elif [[ "$INSTALL_VLLM" -eq 0 ]]; then
  echo "[5/7] Installing torch with pip because --skip-vllm was selected"
  python -m pip install torch
else
  echo "[5/7] No custom torch command set; vLLM installation will resolve torch."
fi

if [[ "$INSTALL_VLLM" -eq 1 ]]; then
  echo "[6/7] Installing vLLM"
  if [[ "$USE_UV" -eq 1 ]]; then
    python -m pip install --upgrade uv
    if ! uv pip install vllm --torch-backend=auto; then
      echo "uv vLLM installation failed; retrying with pip."
      python -m pip install vllm
    fi
  else
    python -m pip install vllm
  fi
else
  echo "[6/7] Skipping vLLM installation"
fi

if [[ "$INSTALL_LMCACHE" -eq 1 ]]; then
  echo "[7/7] Installing LMCache"
  python -m pip install lmcache
else
  echo "[7/7] Skipping LMCache installation. Re-run with --with-lmcache when needed."
fi

echo
echo "Environment verification"
python - <<'PY'
import importlib

for name in ("numpy", "yaml", "psutil", "matplotlib", "torch", "vllm", "lmcache"):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        extra = ""
        if name == "torch":
            extra = f" cuda_available={mod.cuda.is_available()}"
            if mod.cuda.is_available():
                extra += f" device_count={mod.cuda.device_count()}"
        print(f"{name}: ok version={version}{extra}")
    except Exception as exc:
        print(f"{name}: unavailable {type(exc).__name__}: {exc}")
PY

cat <<EOF

Done.

Use this environment with:
  source $VENV_DIR/bin/activate
  source configs/dgx_spark_env.sh
  export ASTRAKV_PYTHON=python

Start vLLM:
  bash scripts/launch/launch_vllm_server.sh
EOF
