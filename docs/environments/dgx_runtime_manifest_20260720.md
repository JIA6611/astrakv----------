# DGX Runtime Manifest: 2026-07-20

This is a measured environment manifest for the version-locked online runtime
control E2E. It is an evidence record, not a hardware minimum specification.

## Source and Host

| Field | Measured value |
| --- | --- |
| Worktree revision | `a3ccccfce43c06f07d9231a7055e1828cc3224bc` |
| Operating system | Ubuntu 24.04.3 LTS |
| Kernel | Linux 6.11.0-1014-nvidia aarch64 |
| libc | glibc 2.39 |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.82.09 |
| CUDA reported by driver | 13.0 |
| Python | 3.12.3 |
| pip | 26.1.2 |

The GPU-memory field was unavailable from this driver's `nvidia-smi` query
interface and is intentionally recorded as unavailable rather than inferred.

## Runtime Tuple

| Component | Version |
| --- | --- |
| vLLM | 0.23.0 |
| LMCache | 0.4.7 |
| Connector | `lmcache-vllm-v1` 0.4.7 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.12.0 |
| FastAPI | 0.139.2 |
| Starlette | 1.3.1 |
| Prometheus FastAPI Instrumentator | 8.0.2 |
| NumPy | 2.2.6 |
| Pandas | 3.0.3 |

The complete Python inventory has 232 package pins in
`constraints/dgx-py312-cu130-20260720.txt`. The source checkout itself is not
present in that constraints file; install it separately in editable mode so the
checked-out revision, rather than an implicit remote editable URL, defines the
code under test.

The initial DGX snapshot had FastAPI 0.114.2 and Starlette 0.38.6, which did
not satisfy the installed vLLM dependency metadata. The lock records the
resolver-corrected FastAPI 0.139.2 and Starlette 1.3.1 pair. The runtime tuple
and CUDA packages were unchanged.

## Model Identity

The verified E2E used `/opt/models/Qwen3-8B` (16G on disk). Retain the following
file digests with any recreated environment:

| File | SHA-256 |
| --- | --- |
| `config.json` | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `generation_config.json` | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |

## Captured Evidence

The runtime validation on this host used the resolver-corrected controlled
`online-policy-locked-env-e2e-v2` run. Its state directory contains one
automatic `online-` command, one terminal `completed` receipt with `removed:
1`, and a run-bound `drop` completion event. No manual owner-UDS action was
used.

Run `nvidia-smi`, `uname -srmo`, `getconf GNU_LIBC_VERSION`,
`python --version`, and `python -m pip freeze --all` again whenever this
manifest is refreshed. A changed result requires a new dated manifest and
constraints file; do not overwrite this record.
