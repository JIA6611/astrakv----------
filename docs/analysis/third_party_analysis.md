# Third-Party Repository Analysis

Status: local source tree verified on 2026-05-25.

Scope:

- Official GitHub repositories were cloned into `third_party/`.
- Each repository was inspected with read-only `rg` searches.
- No third-party source files were modified.
- No runtime, CUDA kernel, or benchmark was implemented or executed.

## Clone Verification

| Project | Official repository | Local path | HEAD | `git status --short` |
| --- | --- | --- | --- | --- |
| vLLM | `https://github.com/vllm-project/vllm.git` | `third_party/vllm/` | `d400445` | Clean |
| LMCache | `https://github.com/LMCache/LMCache.git` | `third_party/LMCache/` | `9793d8b` | Clean |
| FlashAttention | `https://github.com/Dao-AILab/flash-attention.git` | `third_party/flash-attention/` | `2d5d5a1` | Clean |
| llama.cpp | `https://github.com/ggml-org/llama.cpp.git` | `third_party/llama.cpp/` | `328874d` | Clean |
| SGLang | `https://github.com/sgl-project/sglang.git` | `third_party/sglang/` | `0801cc0` | Clean |
| TensorRT-LLM | `https://github.com/NVIDIA/TensorRT-LLM.git` | `third_party/TensorRT-LLM/` | `546a5b0` | Clean |

Note: TensorRT-LLM required `core.longpaths=true` during clone on Windows because the upstream repository contains very long kernel artifact paths. This changed only the local Git checkout behavior, not upstream source files.

## Verification Commands

Representative read-only commands used:

```powershell
git status --short
git rev-parse --short HEAD
rg --files | rg -i "(kv|cache|sched|offload|prefetch|mmap|paged|block)"
rg -n -i "kv|cache|scheduler|offload|prefetch|mmap|paged|block" <selected-source-roots>
```

## vLLM

### Project Purpose

vLLM is a high-throughput LLM inference and serving engine. Its relevant architecture for AstraKV-W is the V1 scheduler, KV cache manager, paged attention layout, KV transfer connectors, and experimental KV offload paths.

### Core Modules

- `vllm/v1/core/`: V1 core scheduling, request queues, block pool, and KV cache management.
- `vllm/v1/worker/`: Worker-side model runner, GPU worker, block table, and KV connector mixins.
- `vllm/distributed/kv_transfer/`: KV transfer connector framework.
- `vllm/v1/kv_offload/`: KV offload/tiering implementation paths.
- `vllm/model_executor/offloader/`: Weight/offload helper logic and prefetch helper paths.
- `vllm/v1/attention/ops/`: Paged attention and chunked-prefill/decode operators.
- `csrc/`: CUDA/C++ kernels and cache/attention kernels.

### KV Cache Related Code Locations

- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/kv_cache_metrics.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- `csrc/cache_kernels.cu`
- `csrc/cache_kernels_fused.cu`
- `csrc/attention/paged_attention_v1.cu`
- `csrc/attention/paged_attention_v2.cu`

### Scheduler Related Code Locations

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/async_scheduler.py`
- `vllm/v1/core/sched/interface.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/core/sched/output.py`
- `vllm/config/scheduler.py`

### Offload Related Code Locations

- `vllm/v1/kv_offload/base.py`
- `vllm/v1/kv_offload/factory.py`
- `vllm/v1/kv_offload/file_mapper.py`
- `vllm/v1/kv_offload/cpu/manager.py`
- `vllm/v1/kv_offload/cpu/gpu_worker.py`
- `vllm/v1/kv_offload/cpu/shared_offload_region.py`
- `vllm/v1/kv_offload/cpu/policies/lru.py`
- `vllm/v1/kv_offload/cpu/policies/arc.py`
- `vllm/v1/kv_offload/tiering/manager.py`
- `vllm/v1/kv_offload/tiering/fs/manager.py`
- `vllm/v1/kv_offload/tiering/fs/io.py`
- `vllm/config/offload.py`

### Prefetch Related Code Locations

- `vllm/model_executor/offloader/prefetch.py`
- `vllm/model_executor/offloader/prefetch_ops.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `examples/disaggregated/lmcache/`
- `examples/disaggregated/kv_load_failure_recovery_offline/`

### mmap / Paging Related Code Locations

- `vllm/v1/core/block_pool.py`
- `vllm/v1/kv_offload/file_mapper.py`
- `vllm/v1/kv_offload/tiering/fs/io.py`
- `vllm/v1/attention/ops/paged_attn.py`
- `docs/design/paged_attention.md`
- `docs/design/hybrid_kv_cache_manager.md`
- `docs/design/nixl_kv_cache_lease.md`

### Recommended Reuse Modules

- V1 scheduler interfaces and request queues.
- `block_pool` and KV cache manager metadata concepts.
- KV transfer connector framework, especially LMCache connector paths.
- KV offload file/tiering abstractions as references.

### Modules Not Recommended for Modification

- `csrc/` kernels.
- Attention backends and paged attention kernels.
- Model executor internals.
- Public serving APIs during the current analysis stage.

## LMCache

### Project Purpose

LMCache is a KV cache reuse, transfer, storage, and offload system for LLM serving. It is directly relevant to AstraKV-W because it already contains cache engine APIs, storage backends, prefetch controllers, offload server code, and integrations with vLLM, SGLang, and TensorRT-LLM.

### Core Modules

- `lmcache/v1/cache_engine.py`: Main cache engine entry.
- `lmcache/v1/storage_backend/`: Local, remote, filesystem, raw block, DAX, GDS, and connector-backed storage.
- `lmcache/v1/distributed/`: Distributed storage manager, L1/L2 management, prefetch and eviction controllers.
- `lmcache/v1/multiprocess/`: Multiprocess cache server, GPU/non-GPU context, and transfer protocols.
- `lmcache/integration/`: Runtime adapters for vLLM, SGLang, TensorRT-LLM, and telemetry.
- `lmcache/v1/offload_server/`: Offload server interfaces.

### KV Cache Related Code Locations

- `lmcache/v1/cache_engine.py`
- `lmcache/v1/cache_interface.py`
- `lmcache/v1/metadata.py`
- `lmcache/v1/manager.py`
- `lmcache/v1/memory_management.py`
- `lmcache/v1/kv_layer_groups.py`
- `lmcache/v1/multiprocess/gpu_context.py`
- `lmcache/v1/multiprocess/non_gpu_context.py`
- `lmcache/v1/multiprocess/transfer_context.py`
- `lmcache/python_ops_fallback.py`

### Scheduler Related Code Locations

- `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`
- `lmcache/v1/distributed/storage_controllers/store_controller.py`
- `lmcache/v1/distributed/storage_controllers/eviction_controller.py`
- `lmcache/v1/distributed/storage_controller.py`
- `lmcache/integration/vllm/`
- `lmcache/integration/sglang/`
- `lmcache/integration/tensorrt_llm/`

LMCache does not replace a serving runtime scheduler. It exposes cache-side controllers and runtime integration points.

### Offload Related Code Locations

- `lmcache/v1/offload_server/abstract_server.py`
- `lmcache/v1/offload_server/message.py`
- `lmcache/v1/offload_server/zmq_server.py`
- `lmcache/v1/storage_backend/local_cpu_backend.py`
- `lmcache/v1/storage_backend/local_disk_backend.py`
- `lmcache/v1/storage_backend/gds_backend.py`
- `lmcache/v1/storage_backend/dax/core.py`
- `lmcache/v1/storage_backend/connector/fs_adapter.py`
- `docs/source/kv_cache/storage_backends/cpu_ram.rst`
- `docs/source/getting_started/quickstart/offload_kv_cache.rst`

### Prefetch Related Code Locations

- `lmcache/v1/distributed/storage_controllers/prefetch_policy.py`
- `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`
- `lmcache/v1/multiprocess/blend_server_v2.py`
- `lmcache/v1/storage_backend/storage_manager.py`
- `tests/v1/distributed/test_prefetch_policy.py`
- `tests/v1/distributed/test_prefetch_controller.py`
- `docs/source/kv_cache/storage_backends/cpu_ram.rst`

### mmap / Paging Related Code Locations

- `lmcache/v1/storage_backend/local_disk_backend.py`
- `lmcache/v1/storage_backend/raw_block/core.py`
- `lmcache/v1/storage_backend/raw_block/key_codec.py`
- `lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py`
- `rust/raw_block/`
- `lmcache/v1/storage_backend/dax/core.py`

The local source has strong storage/page/block abstractions. Direct `mmap` use should be checked again if a future phase targets file-backed KV pages specifically.

### Recommended Reuse Modules

- Cache engine API.
- Storage backend abstraction.
- Prefetch controller and policy interfaces.
- Runtime adapters for vLLM/SGLang/TensorRT-LLM.
- Multiprocess transfer context.

### Modules Not Recommended for Modification

- Native/CUDA storage operations.
- Stable runtime adapters until AstraKV-W chooses an integration target.
- Existing benchmark tools.
- Operator/Kubernetes code unless deployment is in scope.

## FlashAttention

### Project Purpose

FlashAttention provides optimized attention kernels and interfaces. It is useful to AstraKV-W as a reference for paged KV layout constraints and attention kernel calling conventions, not as an early control-plane integration point.

### Core Modules

- `flash_attn/`: Python APIs and CUTE kernels.
- `flash_attn/cute/`: CUTE DSL attention implementations, paged KV utilities, scheduler utilities.
- `hopper/`: Hopper-specific attention kernels and paged KV headers.
- `csrc/`: C++/CUDA/CK extension bindings and kernels.

### KV Cache Related Code Locations

- `csrc/flash_attn_ck/mha_fwd_kvcache.cpp`
- `flash_attn/cute/paged_kv.py`
- `flash_attn/cute/topk_gather_kv.py`
- `flash_attn/cute/cache_utils.py`
- `hopper/paged_kv.h`
- `hopper/test_kvcache.py`
- `hopper/test_attn_kvcache.py`

### Scheduler Related Code Locations

- `flash_attn/cute/tile_scheduler.py`
- `hopper/tile_scheduler.hpp`
- `hopper/flash_prepare_scheduler.cu`

These are kernel tile schedulers, not request schedulers.

### Offload Related Code Locations

No serving-level offload subsystem was found. Offload is expected to be managed by the caller/runtime.

### Prefetch Related Code Locations

- `flash_attn/cute/tile_scheduler.py`
- `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`
- `flash_attn/cute/topk_gather_kv.py`

Observed prefetch is low-level kernel/tile prefetch, not policy-level KV prefetch.

### mmap / Paging Related Code Locations

- `flash_attn/cute/paged_kv.py`
- `hopper/paged_kv.h`
- `csrc/flash_attn_ck/mha_fwd_kvcache.cpp`

Paging here means paged KV block-table layout, not storage-backed memory mapping.

### Recommended Reuse Modules

- Paged KV API constraints.
- Tensor shape and block-table expectations.
- Tests documenting legal KV cache layouts.

### Modules Not Recommended for Modification

- CUDA/CUTE/Hopper kernels.
- Build scripts.
- Benchmark scripts.

## llama.cpp

### Project Purpose

llama.cpp is a portable C/C++ local inference runtime. It is useful for studying compact KV cache structures, GGUF model loading, memory mapping, server slot management, and CPU/GPU backend placement.

### Core Modules

- `src/`: Main llama runtime, model loading, context, KV cache, mmap, batching.
- `ggml/`: Tensor and backend abstraction.
- `common/`: Shared CLI/runtime helpers and prompt/HF cache utilities.
- `tools/server/`: HTTP server, slot management, prompt cache behavior, tests.

### KV Cache Related Code Locations

- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cells.h`
- `src/llama-kv-cache-iswa.h`
- `src/llama-kv-cache-iswa.cpp`
- `src/llama-context.*`
- `tools/server/tests/unit/test_kv_keep_only_active.py`

### Scheduler Related Code Locations

- `tools/server/`
- `tools/server/tests/utils.py`
- `src/llama-batch.*`
- `src/llama-context.*`

llama.cpp scheduler concepts are mostly server-slot and batch oriented.

### Offload Related Code Locations

- `src/llama-model.*`
- `src/llama-context.*`
- `ggml/src/ggml-cuda/`
- `ggml/src/ggml-vulkan/`
- `ggml/src/ggml-metal/`
- backend operation tables containing `offload_op` fields under `ggml/src/`.

### Prefetch Related Code Locations

- `src/llama-mmap.cpp`
- `src/llama-mmap.h`
- `tools/server/` prompt cache paths.
- CPU feature detection under `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`.

The project contains model/file memory mapping and CPU prefetch capability references, but not a standalone KV prefetch policy layer comparable to LMCache.

### mmap / Paging Related Code Locations

- `src/llama-mmap.h`
- `src/llama-mmap.cpp`
- GGUF loading paths under `src/` and `ggml/src/gguf.cpp`.

### Recommended Reuse Modules

- KV cache data-structure concepts.
- Memory-mapped file loading patterns.
- Server slot/prompt cache behavior as a lightweight scheduling reference.

### Modules Not Recommended for Modification

- GGML backend kernels.
- Quantization kernels.
- Server APIs until a precise adapter boundary is selected.

## SGLang

### Project Purpose

SGLang is a serving/runtime stack with request scheduling, radix-prefix cache, HiCache, memory pools, disaggregation, and KV transfer/storage support. It is highly relevant as a scheduler-plus-cache reference.

### Core Modules

- `python/sglang/srt/managers/`: Scheduler, schedule batches, request policies, tokenizer/detokenizer managers.
- `python/sglang/srt/mem_cache/`: Radix cache, unified radix cache, memory pools, HiCache, storage backends, sparsity.
- `python/sglang/srt/disaggregation/`: Prefill/decode disaggregation and KV event/transfer paths.
- `python/sglang/srt/utils/offloader.py`: Weight/module offload and prefetch behavior.
- `sgl-kernel/`: Low-level kernels and KV cache IO.

### KV Cache Related Code Locations

- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/kv_cache_builder.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/jit_kernel/kvcache.py`
- `sgl-kernel/python/sgl_kernel/kvcacheio.py`

### Scheduler Related Code Locations

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/prefill_delayer.py`
- `python/sglang/srt/managers/scheduler_components/`
- `python/sglang/srt/debug_utils/schedule_simulator/`
- `python/sglang/srt/ray/scheduler_actor.py`

### Offload Related Code Locations

- `python/sglang/srt/utils/offloader.py`
- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/storage/`
- `python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py`
- `python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py`

### Prefetch Related Code Locations

- `python/sglang/srt/managers/prefill_delayer.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/utils/offloader.py`
- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`

### mmap / Paging Related Code Locations

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/mem_cache/storage/hf3fs/`
- `python/sglang/srt/mem_cache/storage/nixl/`
- `python/sglang/srt/mem_cache/storage/mooncake_store/`
- `python/sglang/srt/mem_cache/storage/lmcache/`

### Recommended Reuse Modules

- Radix cache and unified radix cache design.
- Scheduler/cache interaction model.
- HiCache storage abstractions.
- Disaggregation KV event and transfer structure.

### Modules Not Recommended for Modification

- `sgl-kernel/` kernels.
- Runtime model executor internals.
- Serving API behavior.
- Registered/manual benchmark suites.

## TensorRT-LLM

### Project Purpose

TensorRT-LLM is NVIDIA's high-performance LLM inference stack. For AstraKV-W it is primarily a production-grade reference for paged KV cache management, block reuse, batch management, KV transfer, and overlap scheduling.

### Core Modules

- `cpp/include/tensorrt_llm/batch_manager/`: C++ batch manager, KV cache manager headers, capacity scheduler.
- `cpp/tensorrt_llm/batch_manager/`: C++ implementations for KV cache, cache transfer, and schedulers.
- `tensorrt_llm/runtime/kv_cache_manager_v2/`: Python KV cache manager v2.
- `tensorrt_llm/_torch/pyexecutor/`: PyTorch executor, scheduler, KV connectors, transceiver.
- `docs/source/`: Developer and feature documentation for scheduler, KV cache, and transfer.
- `cpp/tensorrt_llm/kernels/`: Low-level kernels and packaged kernel artifacts.

### KV Cache Related Code Locations

- `cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h`
- `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h`
- `cpp/include/tensorrt_llm/batch_manager/kvCacheType.h`
- `cpp/include/tensorrt_llm/batch_manager/kvCacheTransferManager.h`
- `cpp/include/tensorrt_llm/batch_manager/kvCacheConnector.h`
- `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`
- `cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp`
- `tensorrt_llm/runtime/kv_cache_manager.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_page.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_storage_manager.py`

### Scheduler Related Code Locations

- `cpp/include/tensorrt_llm/batch_manager/capacityScheduler.h`
- `cpp/include/tensorrt_llm/batch_manager/microBatchScheduler.h`
- `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp`
- `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`
- `tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py`
- `docs/source/features/overlap-scheduler.md`
- `docs/source/torch/scheduler.md`

### Offload Related Code Locations

- `examples/llm-api/llm_kv_cache_offloading.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_storage/`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py`
- `tensorrt_llm/_torch/disaggregation/resource/`
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- `tensorrt_llm/_torch/pyexecutor/connectors/kv_cache_connector.py`

### Prefetch Related Code Locations

- `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`
- `cpp/tensorrt_llm/batch_manager/cacheTransferLayer.cpp`
- `tensorrt_llm/_torch/pyexecutor/scheduler/`
- `tensorrt_llm/_torch/disaggregation/resource/cache_reuse.py`

Most observed prefetch references are kernel, transfer, or scheduler-side preparation mechanisms rather than a standalone research prefetch policy layer.

### mmap / Paging Related Code Locations

- `tensorrt_llm/runtime/kv_cache_manager_v2/_page.py`
- `tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py`
- `cpp/kernels/fmha_v2/src/fmha/paged_kv_cache.h`
- `cpp/tensorrt_llm/common/attentionOp.h`
- `docs/source/features/paged-attention-ifb-scheduler.md`
- `docs/source/features/kvcache.md`

### Recommended Reuse Modules

- KV cache manager architecture as a reference.
- Batch manager scheduler concepts.
- KV transfer documentation and connector abstractions.
- Paged KV manager v2 design.

### Modules Not Recommended for Modification

- CUDA kernels and packaged `.cubin.tar.zst` artifacts.
- TensorRT engine build internals.
- C++ runtime memory allocation internals during early AstraKV-W design.
- Benchmark/perf test suites.

## Cross-Project Findings

| Concern | Strongest references | Notes |
| --- | --- | --- |
| Runtime scheduler | vLLM, SGLang, TensorRT-LLM | Use as design references first; do not modify scheduler internals yet. |
| KV cache abstraction | LMCache, vLLM, SGLang | LMCache is the most reusable cache-system layer. |
| Prefix/radix reuse | SGLang, vLLM | SGLang radix cache is a strong design reference. |
| KV transfer | LMCache, vLLM, TensorRT-LLM | Prefer connector/adapters over core runtime patches. |
| Offload/storage | LMCache, vLLM, SGLang | LMCache has the clearest storage backend abstraction. |
| mmap/file-backed memory | llama.cpp, LMCache, vLLM | llama.cpp is best for model mmap patterns; LMCache/vLLM are better for KV tiers. |
| Paged attention constraints | vLLM, FlashAttention, TensorRT-LLM | Treat kernels as fixed constraints, not modification targets. |

## Recommended Next-Step Boundary

The next phase should produce design-only artifacts before implementation:

1. Interface boundary diagram across runtime, scheduler, KV cache, offload, and prefetch components.
2. Reusable module shortlist with risk levels.
3. Candidate integration path comparison: LMCache-first, vLLM-connector-first, or SGLang-cache-first.
4. Read-only trace plan for observing request, cache, and transfer events.

Do not implement Runtime, CUDA kernels, third-party patches, or benchmarks until a later stage explicitly authorizes that work.

