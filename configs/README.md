# configs

DGX Spark oriented configuration files for real AstraKV-W V0.1 runs.

The files in this directory describe model identity, vLLM endpoint settings,
benchmark matrices, metric sampling intervals, and LMCache backend intent. They
are configuration inputs only; they do not modify vLLM, LMCache, or any
third-party source tree.

Main competition configs:

- `dgx_spark_vllm_qwen7b.yaml`: plain vLLM real endpoint baseline.
- `dgx_spark_lmcache_cpu.yaml`: legacy non-DGX reference only; do not use it in the GPU+SSD production matrix.
- `dgx_spark_lmcache_disk.yaml`: DGX vLLM GPU + LMCache SSD two-tier validation path.
- `astrakv_real_selective_prefetch.yaml`: endpoint-level AstraKV-W selective prefetch / warmup.
- `stress_*_memory_constrained.yaml`: official constrained-memory stress runs.
- `stress_*_extreme_memory_constrained.yaml`: stronger boundary stress runs.
- `lmcache_cpu_constrained.yaml` and `lmcache_disk_constrained.yaml`: reduced LMCache capacity for memory-pressure evidence.

Auxiliary configs:

- `astrakv_selective_prefetch.yaml`: synthetic policy simulator, not a real endpoint performance claim.
- `shared_prefix_workload.yaml`: shared-prefix workload helper.
- `policy_ablation_matrix.yaml`: policy ablation matrix helper.
- `lmcache_cpu_example.yaml` and `lmcache_disk_example.yaml`: examples kept for reference.

See `docs/guides/repository_organization_cn.md` for cleanup candidates and the
current recommended run order.
