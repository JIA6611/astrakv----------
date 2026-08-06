# Runtime Backend Compatibility

## Supported Configuration

The online runtime-control Hook is supported only for this exact backend tuple:

| Component | Required value |
| --- | --- |
| Python | 3.12.3 |
| vLLM | 0.23.0 |
| LMCache | 0.4.7 |
| KV connector | `LMCacheConnectorV1` / `lmcache-vllm-v1` 0.4.7 |
| PyTorch | 2.11.0+cu130 |
| FastAPI | 0.139.2 |
| Starlette | 1.3.1 |
| Prometheus FastAPI Instrumentator | 8.0.2 |
| CUDA driver runtime | CUDA 13.0 with NVIDIA driver 580.82.09 |
| Platform validated | Ubuntu 24.04.3 LTS, Linux aarch64, NVIDIA GB10 |

The Hook rejects a runtime whose measured vLLM or LMCache package version does
not match the supported tuple. The remaining fields are reproducibility
constraints for the validated DGX target; they are not claimed to be portable
across CPU architectures, CUDA major versions, or GPU families.

The immutable measured target is documented in
`docs/environments/dgx_runtime_manifest_20260720.md`.

## Dependency Constraints

`constraints/dgx-py312-cu130-20260720.txt` contains all 232 non-editable
packages returned by `pip freeze --all` from the validated DGX virtual
environment. Its SHA-256 is:

```text
c4660d9f24875b5ded9b001c04f388b267861237b8ceaaeb5608f4c4951463f5
```

It is a full version constraints lock, including the CUDA 13 wheel family. It
does not use pip `--require-hashes`: the measured environment was installed
from binary CUDA wheels and `pip freeze` does not recover their distribution
hashes. Use a trusted, immutable wheel mirror when stronger supply-chain
integrity is required; do not fabricate hashes from package names.

## Recreate the Validated Environment

Run the following on an aarch64 CUDA 13.0 host with compatible NVIDIA driver
and wheel index access. Start from a clean Python 3.12.3 virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install "pip==26.1.2"
.venv/bin/python -m pip install \
  -c constraints/dgx-py312-cu130-20260720.txt \
  -r requirements.txt
.venv/bin/python -m pip install \
  -c constraints/dgx-py312-cu130-20260720.txt \
  -e ".[gpu,report,dev]"
.venv/bin/python -m pip install \
  -c constraints/dgx-py312-cu130-20260720.txt \
  "vllm==0.23.0" "lmcache==0.4.7"
```

The repository is intentionally installed from the local checkout, not pinned
as an editable Git URL in the constraints file. Before accepting an E2E claim,
record `git rev-parse HEAD`, re-check the model hashes in the DGX manifest, and
preserve the output of `pip freeze --all` alongside run artifacts.

## Verify Before Launch

```bash
.venv/bin/python - <<'PY'
import lmcache
import torch
import vllm

assert vllm.__version__ == "0.23.0"
assert lmcache.__version__ == "0.4.7"
assert torch.__version__ == "2.11.0+cu130"
print("backend-version-tuple-ok")
PY

nvidia-smi
.venv/bin/python -m unittest discover -s tests
```

`pip check` on the validated aarch64 host reports exactly one known vendor
metadata exception: `nvidia-cusparselt-cu13 0.8.0 is not supported on this
platform`. That package is a direct dependency of the validated
`torch==2.11.0+cu130` distribution, and the runtime imports and serves on the
GB10 host. Do not suppress any additional `pip check` output: a changed error
is a failed compatibility check and requires a new target definition.

Launch only with `LMCacheConnectorV1` and the explicit runtime-control
environment documented in `docs/guides/online_runtime_control_cn.md`. The
runtime preflight must still be retained for every run; this document and lock
do not replace the per-run installation evidence, endpoint binding, or action
receipt requirements.

## Unsupported Changes

Treat any of the following as a new compatibility target requiring a new
manifest, constraints file, test run, and controlled E2E:

- vLLM, LMCache, PyTorch, CUDA, NVIDIA driver, OS, or CPU architecture change;
- a different KV connector or connector role;
- a model whose recorded config or tokenizer digest differs;
- a changed LMCache disk backend configuration or runtime Hook path.
