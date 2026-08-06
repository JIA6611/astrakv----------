# AstraKV-W Contest Solution Plan

Status: V0.1 real-link implementation plan.

Date: 2026-05-28

## Goal

AstraKV-W targets the OS challenge "Runtime Optimization of LLM Inference for
the Memory Constraint System" by building a measurable, reproducible runtime
control layer around existing LLM serving systems.

The V0.1 goal is not to rewrite vLLM, implement a scheduler, or write CUDA
kernels. The goal is to connect a real model, a real OpenAI-compatible endpoint,
DGX-level metrics, and baseline reports so later KV prefetch/offload policies
have a trustworthy evaluation loop.

## Current Technical Boundary

- vLLM is the first real serving baseline.
- LMCache CPU and disk backends are the first KV offload comparison targets.
- The existing `SelectiveKVPrefetchMVP` remains a policy simulator.
- Synthetic prefetch results must be reported separately from real endpoint
  results.
- `third_party/` source trees remain unmodified.
- vLLM core remains unmodified.
- No CUDA kernel is introduced in V0.1.

## V0.1 Deliverables

| Area | Deliverable |
| --- | --- |
| Real endpoint benchmark | `scripts/run_real_benchmark.py` |
| DGX metrics sampling | `scripts/dgx_metrics_collector.py` |
| vLLM launch wrappers | `scripts/launch_vllm_server.ps1`, `scripts/launch_vllm_server.sh` |
| LMCache launch wrappers | `scripts/launch_lmcache_vllm.ps1`, `scripts/launch_lmcache_vllm.sh` |
| DGX configs | `configs/dgx_spark_vllm_qwen7b.yaml`, `configs/dgx_spark_lmcache_cpu.yaml`, `configs/dgx_spark_lmcache_disk.yaml` |
| Policy simulator config | `configs/astrakv_selective_prefetch.yaml` |
| Reproduction docs | `docs/dgx_spark_setup.md`, `docs/reproduction.md` |

## Baseline Matrix

| Run | Purpose | Required for V0.1 |
| --- | --- | --- |
| vLLM only | Real serving baseline | Yes |
| vLLM + LMCache CPU | CPU-tier KV offload baseline | Yes, after LMCache integration is validated |
| vLLM + LMCache disk | Disk-tier KV offload baseline | Yes, after LMCache integration is validated |
| SelectiveKVPrefetchMVP | Policy simulator only | Yes, but reported separately |

## Metrics

V0.1 real endpoint reports:

- TTFT from streaming response start to first generated content event.
- TPOT from first generated content event to stream completion.
- Throughput in generated output tokens per second.
- CPU memory from matching runtime processes when `psutil` is available.
- GPU memory and GPU utilization from `nvidia-smi`.
- SSD read/write deltas from `/proc/diskstats` on Linux.
- Per-case metric samples under the result directory.

KV hit rate and prefetch hit/waste are intentionally blank in real endpoint V0.1
until an approved LMCache/vLLM adapter exposes real cache events.

## Contest Storyline

1. Establish a real serving baseline on DGX Spark with Qwen2.5-7B-Instruct.
2. Add LMCache CPU and disk runs to show the memory hierarchy baseline.
3. Use AstraKV-W metadata, policy simulation, and reports to explain selective
   KV prefetch without claiming unimplemented runtime control.
4. In V0.2, map real LMCache cache events into AstraKV-W chunk metadata.
5. In V0.3, enable a controlled prefetch adapter that calls public LMCache or
   vLLM extension points.

## Next Engineering Steps

1. Validate vLLM and Qwen2.5-7B-Instruct on DGX Spark.
2. Run `dgx_spark_vllm_qwen7b.yaml` and archive the first real report.
3. Validate the exact LMCache integration flags for the installed version.
4. Run CPU and disk LMCache baselines.
5. Add a read-only LMCache metrics adapter for real KV hit/offload counters.
6. Promote `SelectiveKVPrefetchMVP` from simulator to adapter-driven policy only
   after the cache event boundary is proven.
