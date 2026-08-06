# Reusable Modules Shortlist

Status: design-only shortlist based on local read-only source inspection.

## High-Priority Candidates

| Project | Module | Why it matters | Reuse mode | Risk |
| --- | --- | --- | --- | --- |
| LMCache | `lmcache/v1/cache_engine.py` | Main KV cache API surface | Adapter or API-level integration | Medium |
| LMCache | `lmcache/v1/storage_backend/` | Local/remote storage abstraction | Backend reuse/reference | Medium |
| LMCache | `lmcache/v1/distributed/storage_controllers/prefetch_controller.py` | Existing prefetch controller structure | Design reference first | Medium |
| LMCache | `lmcache/integration/vllm/` | Existing vLLM connector path | Adapter study | Medium |
| vLLM | `vllm/v1/core/sched/` | V1 scheduler structure | Boundary/reference | High |
| vLLM | `vllm/v1/core/kv_cache_manager.py` | KV allocation and block management | Boundary/reference | High |
| vLLM | `vllm/distributed/kv_transfer/kv_connector/v1/` | KV transfer extension point | Connector-level study | Medium |
| SGLang | `python/sglang/srt/mem_cache/radix_cache.py` | Prefix/radix cache model | Design reference | Medium |
| SGLang | `python/sglang/srt/mem_cache/unified_radix_cache.py` | Unified cache plus HiCache interactions | Design reference | Medium |
| SGLang | `python/sglang/srt/managers/scheduler.py` | Scheduler/cache interaction | Boundary/reference | High |

## Reference-Only Candidates

| Project | Module | Use |
| --- | --- | --- |
| FlashAttention | `flash_attn/cute/paged_kv.py` | Paged KV tensor layout constraints |
| FlashAttention | `csrc/flash_attn_ck/mha_fwd_kvcache.cpp` | KV cache attention API expectations |
| llama.cpp | `src/llama-mmap.*` | Memory-mapped model loading patterns |
| llama.cpp | `src/llama-kv-cache.*` | Compact C++ KV cache structures |
| TensorRT-LLM | `cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h` | Production KV cache manager design |
| TensorRT-LLM | `tensorrt_llm/runtime/kv_cache_manager_v2/` | Python paged KV manager v2 reference |
| TensorRT-LLM | `docs/source/developer-guide/kv-transfer.md` | KV transfer architecture reference |

## Avoid-Modifying List

| Project | Avoid modifying | Reason |
| --- | --- | --- |
| vLLM | `csrc/`, attention kernels, model executor internals | High blast radius and fast upstream movement |
| LMCache | Native storage ops and runtime adapter internals | Keep integration stable until API boundary is selected |
| FlashAttention | CUDA/CUTE/Hopper kernels | Kernel changes are out of current scope |
| llama.cpp | GGML backend kernels and quantization code | Not needed for AstraKV-W control-plane design |
| SGLang | `sgl-kernel/` and model executor internals | Kernel/runtime internals are out of scope |
| TensorRT-LLM | C++ kernels, packaged cubins, TensorRT build path | Large, hardware-specific, not appropriate for early integration |

## Recommended Study Order

1. LMCache cache engine, storage backends, and prefetch controller.
2. vLLM KV transfer connector and V1 scheduler/KV cache boundaries.
3. SGLang radix/unified cache and scheduler interaction.
4. TensorRT-LLM KV cache manager v2 and KV transfer docs as production reference.
5. llama.cpp mmap/KV cache structures for lightweight C++ memory design reference.
6. FlashAttention paged KV layouts as kernel constraint reference only.

