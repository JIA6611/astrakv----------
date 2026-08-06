# DGX Spark Setup

Status: P0 competition setup guide.

Date: 2026-06-08

## Assumptions

- Target OS for official numbers is Linux.
- `nvidia-smi` is available.
- vLLM and LMCache are installed in the active Python environment.
- The first official dense model is `Qwen/Qwen2.5-7B-Instruct`.
- Model files are already cached or can be downloaded before the run.
- `third_party/` repositories are read-only references for P0.

## Environment Check

Run before the first official benchmark:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))"
python -c "import yaml, psutil, matplotlib; print('helper deps ok')"
```

Install AstraKV-W helper dependencies if needed:

```bash
pip install -r requirements.txt
```

For a fresh DGX Spark Python environment, use the bootstrap guide:

```bash
bash scripts/entrypoints/bootstrap_dgx_spark_env.sh
```

See [dgx_spark_bootstrap.md](dgx_spark_bootstrap.md) for the full install
flow, including vLLM, PyTorch, and optional LMCache.

For a one-command local validation path:

```bash
bash scripts/entrypoints/run_dgx_spark_validation.sh
```

This creates or reuses `.venv`, installs helper dependencies, records an
environment report, runs unit tests, and generates mmap-backed DGX Spark KV
evidence under `results/dgx_spark_validation/`. After a vLLM server is already
listening on the configured endpoint, include the real endpoint smoke:

```bash
bash scripts/entrypoints/run_dgx_spark_validation.sh --with-real
```

The default DGX Spark environment is in `configs/dgx_spark_env.sh`. It starts
with `ASTRAKV_GPU_MEMORY_UTILIZATION=0.60` because DGX Spark uses coherent
unified memory; raise this gradually after the smoke run is stable.

Install vLLM and LMCache according to the CUDA stack on the target machine.
Record exact package versions in the final report.

## Common Environment Variables

```bash
export ASTRAKV_MODEL=Qwen/Qwen2.5-7B-Instruct
export ASTRAKV_HOST=127.0.0.1
export ASTRAKV_PORT=8000
export ASTRAKV_MAX_MODEL_LEN=8192
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
```

For disk-tier LMCache:

```bash
export LMCACHE_LOCAL_DISK=results/lmcache_disk_store
```

For constrained-memory stress runs:

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
```

## Launch vLLM Baseline

Linux:

```bash
bash scripts/launch/launch_vllm_server.sh
```

PowerShell:

```powershell
$env:ASTRAKV_MODEL="Qwen/Qwen2.5-7B-Instruct"
$env:ASTRAKV_PORT="8000"
$env:ASTRAKV_MAX_MODEL_LEN="8192"
.\scripts\archive\launch_vllm_server.ps1
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Launch LMCache Baselines

CPU tier:

```bash
bash scripts/launch/launch_lmcache_vllm.sh cpu
```

Disk tier:

```bash
export LMCACHE_LOCAL_DISK=results/lmcache_disk_store
bash scripts/launch/launch_lmcache_vllm.sh disk
```

PowerShell:

```powershell
.\scripts\archive\launch_lmcache_vllm.ps1 -Backend cpu
.\scripts\archive\launch_lmcache_vllm.ps1 -Backend disk
```

Validation cues:

- Launch output should include the selected backend.
- vLLM command should include `--kv-transfer-config` when LMCache is enabled.
- `LMCACHE_CONFIG_FILE` should point to the selected CPU or disk config.

## Run Benchmarks

vLLM-only:

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/p0_1_vllm
```

LMCache CPU:

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_cpu.yaml \
  --output-dir results/p0_2_lmcache_cpu
```

LMCache disk:

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_disk.yaml \
  --output-dir results/p0_3_lmcache_disk
```

Stress:

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/stress_vllm_memory_constrained.yaml \
  --output-dir results/p0_7_stress_vllm
```

Selective prefetch:

```bash
python scripts/benchmark/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/p0_8_selective_prefetch
```

## Notes

- Close unrelated GPU workloads before official runs.
- Keep server logs with `tee` for cache-event extraction.
- Keep synthetic `scripts/benchmark/benchmark_runner.py` results separate from real vLLM/LMCache results.
- Use `docs/reproduction.md` as the authoritative P0 run order.
