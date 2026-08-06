# AstraKV-W Codebase Analysis

Status: local source-tree analysis only.

Date: 2026-05-25

Scope:

- Repositories analyzed under `third_party/`: vLLM, LMCache, FlashAttention, llama.cpp, SGLang, TensorRT-LLM.
- Focus areas: KV cache, paged attention, async loading, prefetch, offload, scheduler, mmap, and unified memory.
- No third-party source files were modified.
- No Runtime, CUDA kernel, optimization logic, or benchmark was implemented.

## Executive Summary

AstraKV-W should treat the cloned projects as read-only reference and integration targets. The most practical early path is not to alter kernels or runtime schedulers, but to define a thin analysis/adapter boundary around cache metadata, request scheduling hints, page/block identity, prefetch plans, and offload tier selection.

Recommended insertion order:

1. Use LMCache as the primary reusable cache/offload/prefetch reference because it already exposes cache-engine, storage-backend, async prefetch, and runtime-integration layers.
2. Use vLLM as the first serving-runtime boundary reference because its V1 scheduler, KV cache manager, block pool, and LMCache connector are explicit and mature enough for adapter design.
3. Use SGLang as the second runtime reference for radix-cache and unified-cache behavior, especially scheduler-cache coupling and host/device cache transitions.
4. Use TensorRT-LLM as a production-grade paged-KV, block-reuse, UVM, and multi-tier memory reference, but avoid modifying its C++/CUDA internals.
5. Use FlashAttention and llama.cpp as lower-level layout and system-memory references, not as initial modification targets.

## Cross-Repository Concept Map

| Concept | Best local reference | AstraKV-W use |
| --- | --- | --- |
| KV cache metadata | `vllm/v1/core/kv_cache_manager.py`, `lmcache/v1/cache_engine.py`, `python/sglang/srt/mem_cache/` | Define cache object identity, token spans, page/block mapping, hit/miss metadata |
| Paged attention layout | `vllm/v1/attention/ops/`, `flash_attn/cute/paged_kv.py`, TensorRT-LLM KV managers | Record kernel-facing constraints without writing kernels |
| Async loading | LMCache async storage manager, SGLang HiCache/offload paths, vLLM connector load/save waits | Define future load lifecycle states and observability |
| Prefetch | LMCache `PrefetchController`, SGLang `HiRadixCache`, vLLM connector scheduling paths | Define policy inputs/outputs before implementing policy |
| Offload | LMCache storage backends, vLLM `kv_offload`, SGLang HiCache/offloader, TensorRT-LLM host/disk tiers | Define tier abstraction and migration states |
| Scheduler | vLLM V1 scheduler, SGLang scheduler, TensorRT-LLM capacity schedulers | Add scheduler hints outside runtime internals |
| mmap | llama.cpp mmap layer, LMCache local disk/raw block/DAX storage | Reference file-backed loading and page lifecycle |
| Unified memory | TensorRT-LLM UVM paths | Reference only; avoid early dependency on UVM behavior |

## vLLM

### Project Role

vLLM is the best first runtime-side reference for AstraKV-W because it has explicit request scheduling, KV block management, paged-attention metadata, and KV transfer connector boundaries. Its existing LMCache connector makes it a natural bridge between a serving engine and external KV storage.

### Key Directories

- `third_party/vllm/vllm/v1/core/`: scheduler, block pool, KV cache manager, request state.
- `third_party/vllm/vllm/v1/core/sched/`: V1 scheduler and async scheduler.
- `third_party/vllm/vllm/v1/worker/`: worker block tables and model-runner integration.
- `third_party/vllm/vllm/v1/attention/ops/`: Python-facing paged attention operator wrappers.
- `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/`: KV connector boundary, including LMCache connector.
- `third_party/vllm/vllm/v1/kv_offload/`: experimental KV offload manager and CPU/tiering implementations.
- `third_party/vllm/vllm/model_executor/offloader/`: model parameter offload and prefetch helper logic.
- `third_party/vllm/csrc/`: CUDA/C++ kernels. Read-only reference for this phase.

### Core Classes

- `KVCacheManager` in `vllm/v1/core/kv_cache_manager.py`: owns request-to-block allocation, prefix cache hit detection, block freeing, event emission, and block-id lookup.
- `KVCacheBlocks` in `vllm/v1/core/kv_cache_manager.py`: grouped KV block-id container used across cache manager and scheduler paths.
- `BlockPool` in `vllm/v1/core/block_pool.py`: allocates, caches, touches, evicts, and frees `KVCacheBlock` objects.
- `BlockHashToBlockMap` in `vllm/v1/core/block_pool.py`: maps block hash plus group id to cached blocks.
- `Scheduler` in `vllm/v1/core/sched/scheduler.py`: controls waiting/running queues, request scheduling, preemption, connector metadata, and state updates after execution.
- `AsyncScheduler` in `vllm/v1/core/sched/async_scheduler.py`: extends V1 scheduler behavior for async output update.
- `BlockTable` and `MultiGroupBlockTable` in `vllm/v1/worker/block_table.py`: transform logical block ids into kernel-facing block tables and slot mappings.
- `LMCacheConnectorV1` in `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`: runtime connector for external KV load/save via LMCache.
- `OffloadingManager`, `GPULoadStoreSpec`, `CanonicalKVCaches` in `vllm/v1/kv_offload/base.py`: abstract KV offload contract.
- `CPUOffloadingManager` in `vllm/v1/kv_offload/cpu/manager.py`: CPU-backed offload manager with lookup/load/store/touch lifecycle.
- `PrefetchOffloader` in `vllm/model_executor/offloader/prefetch.py`: model-parameter prefetch/offload reference; useful for lifecycle design, not KV policy reuse.

### Core Functions And Methods

- `KVCacheManager.get_computed_blocks()`: checks prefix-cache reuse and returns matched block ids plus token count.
- `KVCacheManager.allocate_slots()`: allocates KV slots for new tokens and links them to request state.
- `KVCacheManager.cache_blocks()`: commits full blocks into prefix cache.
- `KVCacheManager.free()`, `evict_blocks()`, `take_events()`: free/evict/event lifecycle.
- `BlockPool.get_cached_block()`, `cache_full_blocks()`, `get_new_blocks()`, `touch()`, `free_blocks()`: core block-pool operations.
- `Scheduler.schedule()`: central scheduling step for prefill/decode and waiting/running transitions.
- `Scheduler._build_kv_connector_meta()`: builds connector metadata for KV transfer.
- `Scheduler._update_from_kv_xfer_finished()`: handles remote KV transfer completion.
- `Scheduler._handle_invalid_blocks()`: handles block invalidation from connector load failure.
- `BlockTable.compute_slot_mapping()`, `commit_block_table()`, `map_to_kernel_blocks()`: bridges logical block allocation to kernel metadata.
- `LMCacheConnectorV1.start_load_kv()`, `wait_for_layer_load()`, `save_kv_layer()`, `wait_for_save()`, `get_num_new_matched_tokens()`, `update_state_after_alloc()`, `request_finished()`: connector lifecycle.
- `CPUOffloadingManager.lookup()`, `prepare_load()`, `complete_load()`, `prepare_store()`, `complete_store()`: offload state transitions.

### Data Flow

1. Request enters scheduler waiting queue.
2. `Scheduler.schedule()` selects prefill/decode work and calls KV cache manager to compute reusable blocks and allocate slots.
3. `KVCacheManager` queries `BlockPool` for cached prefix blocks, allocates new blocks, and records request-to-block metadata.
4. Scheduler builds `KVConnectorMetadata` when external KV transfer is configured.
5. Worker receives scheduler output, constructs `BlockTable`/slot mapping, and passes block tables to attention execution.
6. KV connector loads or saves per-layer KV through `start_load_kv()`, `wait_for_layer_load()`, `save_kv_layer()`, and `wait_for_save()`.
7. After execution, scheduler updates request state, commits full blocks to prefix cache, emits KV events, and frees finished request blocks.

### Memory Flow

- GPU KV pages are represented as logical block ids in scheduler/cache-manager code and as tensor block tables near worker/kernel boundaries.
- Prefix cache maps block hashes to reusable `KVCacheBlock` instances.
- External KV transfer moves layer KV between runtime GPU buffers and connector-managed storage.
- vLLM KV offload introduces CPU/tiering managers that prepare load/store specs, but these paths should remain reference-only until AstraKV-W chooses a concrete integration target.
- Paged attention kernels consume block tables and slot mappings; AstraKV-W should not modify those kernels in this stage.

### Reusable Modules

- Scheduler metadata boundaries: `SchedulerOutput`, connector metadata, request queue behavior.
- KV cache metadata concepts: block ids, hashes, prefix-cache events, request-to-block maps.
- KV connector interface and LMCache connector lifecycle.
- Block table layout as a compatibility reference for future adapter design.
- Offload manager abstract lifecycle, but not its concrete internals yet.

### Modules Not Recommended For Modification

- `third_party/vllm/csrc/`
- `vllm/v1/attention/ops/` kernel-facing attention paths
- `vllm/model_executor/` model execution internals
- Existing vLLM scheduler internals
- Existing LMCache connector implementation before AstraKV-W has its own adapter contract

### Best AstraKV-W Insertion Point

Best insertion point: a read-only adapter/spec around `vllm/distributed/kv_transfer/kv_connector/v1/` and scheduler-produced KV connector metadata.

The first design artifact should describe how AstraKV-W would observe request id, token ids, block ids, matched prefix length, new allocation, and load/save completion without changing vLLM internals.

## LMCache

### Project Role

LMCache is the strongest direct reuse candidate. It already owns cache-engine APIs, token-to-cache metadata, memory allocators, CPU/disk/distributed storage backends, async lookup/prefetch, and runtime adapters for vLLM/SGLang/TensorRT-LLM.

### Key Directories

- `third_party/LMCache/lmcache/v1/`: main cache engine, metadata, memory management, token processing.
- `third_party/LMCache/lmcache/v1/storage_backend/`: CPU, disk, raw-block, DAX, GDS, and connector-backed storage.
- `third_party/LMCache/lmcache/v1/distributed/`: distributed controllers and storage coordination.
- `third_party/LMCache/lmcache/v1/distributed/storage_controllers/`: prefetch, eviction, and store controllers.
- `third_party/LMCache/lmcache/v1/multiprocess/`: multiprocess cache server/context/transfer machinery.
- `third_party/LMCache/lmcache/v1/offload_server/`: offload server interface and ZMQ server.
- `third_party/LMCache/lmcache/integration/vllm/`: vLLM connector implementation and request metadata.
- `third_party/LMCache/lmcache/integration/sglang/`: SGLang integration.
- `third_party/LMCache/lmcache/integration/tensorrt_llm/`: TensorRT-LLM integration.
- `third_party/LMCache/rust/raw_block/`: raw block storage implementation reference.

### Core Classes

- `LMCacheEngine` in `lmcache/v1/cache_engine.py`: central store/retrieve/lookup/prefetch/move/compress/decompress API.
- `LMCacheEngineBuilder` in `lmcache/v1/cache_engine.py`: constructs engine dependencies and allocators.
- `StorageManager` in `lmcache/v1/storage_backend/storage_manager.py`: multi-backend allocation, put/get, async lookup and prefetch.
- `WeightedSemaphore`, `AsyncMultiSerializer`, `AsyncSingleSerializer` in `storage_manager.py`: concurrency and async serialization helpers.
- `LocalCPUBackend` in `lmcache/v1/storage_backend/local_cpu_backend.py`: CPU memory allocator backend with pin/unpin and chunk accounting.
- `LocalDiskBackend` and `LocalDiskWorker` in `lmcache/v1/storage_backend/local_disk_backend.py`: disk backend with async task submission and batched IO.
- `PrefetchController` in `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`: async prefetch lifecycle controller.
- `PrefetchPolicy`, `DefaultPrefetchPolicy`, `RetainPrefetchPolicy` in `prefetch_policy.py`: pluggable policy boundary.
- `OffloadServerInterface` in `lmcache/v1/offload_server/abstract_server.py`: offload server contract.
- `LMCacheConnectorV1Impl` in `lmcache/integration/vllm/vllm_v1_adapter.py`: vLLM-side connector implementation.
- `RequestTracker`, `ReqMeta`, `LoadSpec`, `SaveSpec`, `DisaggSpec` in `vllm_v1_adapter.py`: vLLM request/cache metadata model.

### Core Functions And Methods

- `LMCacheEngine.store()`, `store_layer()`: write KV cache segments.
- `LMCacheEngine.retrieve()`, `retrieve_layer()`: load KV cache segments.
- `LMCacheEngine.lookup()`: detect reusable tokens/cache keys.
- `LMCacheEngine.async_lookup_and_prefetch()`: async lookup plus prefetch entry.
- `LMCacheEngine.move()`: move cache objects across locations.
- `LMCacheEngine.compress()`, `decompress()`: compression hooks.
- `StorageManager.allocate()`, `batched_allocate()`: memory allocation for storage objects.
- `StorageManager.put()`, `batched_put()`, `get()`, `batched_get()`, `layerwise_batched_get()`: storage backend read/write paths.
- `StorageManager.async_lookup_and_prefetch()`: async backend lookup/prefetch.
- `StorageManager.get_block_mapping()`, `contains()`, `batched_contains()`, `touch_cache()`: mapping and cache-state operations.
- `PrefetchController.submit_prefetch_request()`, `query_lookup_result()`, `query_prefetch_result()`: external prefetch API.
- `PrefetchController._prefetch_loop()`, `_start_lookup_phase()`, `_transition_to_load_phase()`, `_finalize_load()`: controller lifecycle.
- `DefaultPrefetchPolicy.select_load_plan()`, `select_l1_retentions()`: policy hooks.
- `LocalDiskBackend.batched_get_non_blocking()`, `batched_async_contains()`, `async_save_bytes_to_disk()`, `batched_async_load_bytes_from_disk()`: async disk operations.
- `LMCacheConnectorV1Impl.start_load_kv()`, `wait_for_layer_load()`, `save_kv_layer()`, `wait_for_save()`, `get_finished()`, `get_num_new_matched_tokens()`, `update_state_after_alloc()`, `build_connector_meta()`, `request_finished()`: vLLM integration lifecycle.

### Data Flow

1. Runtime adapter produces request metadata: token ids, request id, cacheable spans, and slot/block mapping.
2. `LMCacheEngine.lookup()` or `async_lookup_and_prefetch()` maps token spans to cache keys and existing memory objects.
3. `StorageManager` checks active backends, optionally pins objects, and returns hit/miss/load-plan data.
4. `PrefetchController` can split prefetch into lookup and load phases, using `PrefetchPolicy` to choose a plan.
5. Runtime connector waits for load completion and writes KV tensors into runtime-owned KV cache locations.
6. On request completion or layer completion, connector calls store methods to persist newly computed KV.

### Memory Flow

- Cache keys map to memory objects stored in CPU RAM, disk, raw block, DAX/GDS, or distributed backend.
- Local CPU backend owns chunk allocation and pin/unpin state.
- Local disk backend serializes objects to file paths, uses async task submission, and copies bytes between allocator backend and disk.
- Prefetch moves cache objects from lower tiers toward L1/L2 locations before runtime consumption.
- Runtime integrations copy between runtime KV tensor layout and LMCache memory objects.

### Reusable Modules

- `LMCacheEngine` public API as the leading cache engine interface reference.
- `StorageManager` and storage backend abstraction.
- `PrefetchController` and `PrefetchPolicy` as policy/control separation.
- vLLM integration request metadata model and connector lifecycle.
- CPU/disk backend accounting and async IO structure.
- Multiprocess transfer context for future multi-process design.

### Modules Not Recommended For Modification

- Runtime adapters before AstraKV-W fixes its own boundary contract.
- Native/raw-block/Rust/GDS/DAX internals.
- Benchmark and deployment/operator code.
- Compression internals unless a later phase explicitly studies compression.

### Best AstraKV-W Insertion Point

Best insertion point: design a thin AstraKV-W cache-control API modeled on `LMCacheEngine` plus `PrefetchPolicy`, while treating LMCache storage backends as reference or eventual adapters.

The initial AstraKV-W document should define request metadata, cache key identity, token-span granularity, async prefetch status, and backend tier descriptors without implementing storage logic.

## FlashAttention

### Project Role

FlashAttention is a kernel/layout reference for paged KV attention and tile scheduling. It should not be an early modification target. AstraKV-W should use it to understand page-table shape, copy granularity, and kernel-facing constraints.

### Key Directories

- `third_party/flash-attention/flash_attn/cute/`: CUTE DSL paged KV and tile scheduler logic.
- `third_party/flash-attention/csrc/flash_attn_ck/`: C++/CK kvcache attention entry points.
- `third_party/flash-attention/hopper/`: Hopper paged KV headers.
- `third_party/flash-attention/csrc/`: native kernels and bindings.

### Core Classes And Structs

- `PagedKVManager` in `flash_attn/cute/paged_kv.py`: page-table aware loader for K/V pages.
- `PagedKVManager` in `hopper/paged_kv.h`: C++/Hopper side paged-KV management reference.
- `SchedulingMode`, `ClcState`, `WorkTileInfo` in `flash_attn/cute/tile_scheduler.py`: tile scheduling metadata.
- `TileSchedulerProtocol` in `tile_scheduler.py`: scheduler protocol for work tiles.
- `SingleTileScheduler`, `StaticPersistentTileScheduler`, `SingleTileLPTScheduler`, `SingleTileVarlenScheduler`, `Sm100FmhaStaticTileScheduler`, `Sm100FmhaClcDynamicTileScheduler`: kernel tile scheduling variants.

### Core Functions And Methods

- `PagedKVManager.create()`: constructs paged KV parameters.
- `PagedKVManager.load_page_table()`: loads page-table entries for a KV block.
- `PagedKVManager.compute_X_ptr()`: computes K or V pointer from page metadata.
- `PagedKVManager.load_KV()`: loads KV data into shared memory.
- `PagedKVManager._copy_row_async()`: async row copy primitive in kernel DSL.
- `TileSchedulerProtocol.get_current_work()`, `prefetch_next_work()`, `advance_to_next_work()`: tile lifecycle.
- `compute_sm100_fmha_grid()`, `compute_sm100_fmha_grid_clc()`: grid shape helpers for SM100 FMHA.

### Data Flow

1. Kernel receives page-table metadata and sequence/tile coordinates.
2. `PagedKVManager` resolves page-table entries to K/V base addresses.
3. Tile scheduler assigns work tiles.
4. K/V rows are loaded into shared memory or kernel-local fragments.
5. Attention computation consumes paged K/V tiles.

### Memory Flow

- Runtime-owned KV cache is represented through page-table indirection.
- Kernel code computes physical K/V addresses from logical page ids.
- Async copy occurs at the row/tile level inside kernel execution.
- Page size, block layout, head dimension, and alignment constraints must be honored by any external cache/offload system.

### Reusable Modules

- Conceptual page table layout and pointer arithmetic.
- Tile scheduler vocabulary and prefetch-next-work pattern.
- Kernel-facing constraints for page/block shapes.

### Modules Not Recommended For Modification

- All CUDA/C++/CUTE kernels.
- `csrc/`, `hopper/`, and generated/build-related native paths.
- Attention math and tile scheduling internals.

### Best AstraKV-W Insertion Point

Best insertion point: none in code at this stage. Use FlashAttention only to document compatibility constraints for KV page shape, page-table mapping, alignment, and async-copy granularity.

## llama.cpp

### Project Role

llama.cpp is the strongest compact reference for mmap, model-file memory mapping, mlock, and a portable C++ KV cache lifecycle. It is less relevant as a direct high-throughput server integration target, but very useful for file-backed memory and cache-state serialization patterns.

### Key Directories

- `third_party/llama.cpp/src/llama-kv-cache.h`: C++ KV cache class and memory context interfaces.
- `third_party/llama.cpp/src/llama-kv-cache.cpp`: KV cache allocation, update, sequence operations, serialization.
- `third_party/llama.cpp/src/llama-mmap.h`: file/mmap/mlock public structs.
- `third_party/llama.cpp/src/llama-mmap.cpp`: platform-specific mmap, prefetch, mlock implementation.
- `third_party/llama.cpp/src/llama-context.cpp`: context-level encode/decode, async copies, state save/load, memory API wrappers.
- `third_party/llama.cpp/tools/server/`: HTTP server reference, not central for AstraKV-W.

### Core Classes And Structs

- `llama_kv_cache` in `src/llama-kv-cache.h`: KV cache implementation and sequence operations.
- `llama_kv_cache_context` in `src/llama-kv-cache.h`: memory context object used by graph execution.
- `llama_file` in `src/llama-mmap.h`: file abstraction with size, seek, read/write helpers.
- `llama_mmap` in `src/llama-mmap.h`: memory-mapped file abstraction.
- `llama_mlock` in `src/llama-mmap.h`: memory locking abstraction.
- `llama_context` in `src/llama-context.cpp`: runtime context with encode/decode/state APIs.
- `llama_io_write_host`, `llama_io_read_host`, `llama_io_write_file`, `llama_io_read_file`, `llama_io_write_device`, `llama_io_read_device` in `llama-context.cpp`: state IO paths.

### Core Functions And Methods

- `llama_kv_cache::clear()`, `seq_rm()`, `seq_cp()`, `seq_keep()`, `seq_add()`, `seq_div()`: sequence-level cache editing.
- `llama_kv_cache::init_full()`, `init_update()`: initializes memory context.
- `llama_kv_cache::prepare()`, `update()`, `apply_ubatch()`: per-batch cache lifecycle.
- `llama_kv_cache::set_input_k_idxs()`, `set_input_v_idxs()`, `set_input_kq_mask()`: graph input setup for KV access.
- `llama_kv_cache::state_write()`, `state_read()`, `state_write_meta()`, `state_write_data()`, `state_read_meta()`, `state_read_data()`: KV state serialization.
- `llama_mmap::llama_mmap()`, `addr()`, `size()`, `unmap_fragment()`: mmap lifecycle.
- Platform mmap calls in `llama-mmap.cpp`: `mmap`, `posix_madvise`, `MapViewOfFile`, `PrefetchVirtualMemory`, and `mlock`/platform lock equivalents.
- `llama_context::encode()`, `decode()`, `memory_update()`: runtime-level cache update flow.
- `llama_context::state_seq_save_file()`, `state_seq_load_file()`: sequence-state persistence.

### Data Flow

1. Model and context initialize memory structures.
2. Decode/encode converts incoming batches into `llama_ubatch` units.
3. KV cache prepares slot information and graph inputs.
4. Graph execution writes K/V data into cache-owned tensors.
5. Sequence operations can remove/copy/keep/shift cache state.
6. State IO APIs serialize or restore cache metadata and data.

### Memory Flow

- Model files may be backed by `llama_mmap`, using OS page cache and optional prefetch.
- On POSIX, mmap may use `MAP_POPULATE`, `posix_madvise(... WILLNEED)`, and random-access advice.
- On Windows, file mapping uses `CreateFileMappingA`, `MapViewOfFile`, and optional `PrefetchVirtualMemory`.
- `llama_mlock` can lock mapped memory to reduce paging.
- KV cache itself is managed through ggml backend buffers and sequence-aware cache cells, not through a separate paged-attention block table like vLLM/TensorRT-LLM.

### Reusable Modules

- mmap/mlock portability model.
- File prefetch and OS page-advice behavior.
- Cache state serialization design.
- Sequence-level cache editing concepts.

### Modules Not Recommended For Modification

- ggml backend internals.
- KV cache core implementation.
- mmap platform code.
- Server code unless AstraKV-W later targets llama.cpp serving.

### Best AstraKV-W Insertion Point

Best insertion point: documentation/reference only for file-backed tier design. If AstraKV-W later introduces disk-backed KV pages, llama.cpp can inform mmap, prefetch, mlock, and state serialization requirements, but it should not be modified.

## SGLang

### Project Role

SGLang is the best reference for scheduler-cache coupling, radix prefix cache, unified cache, HiCache, and runtime-side offload/prefetch behavior. It is more runtime-specific than LMCache, but valuable for how scheduling decisions interact with prefix matching and host/device cache movement.

### Key Directories

- `third_party/sglang/python/sglang/srt/managers/`: scheduler and runtime managers.
- `third_party/sglang/python/sglang/srt/mem_cache/`: radix cache, unified radix cache, memory pools, HiCache, storage backends.
- `third_party/sglang/python/sglang/srt/mem_cache/storage/`: LMCache, Mooncake, NIXL, HF3FS, EIC, SiMM, AIBrix storage backends.
- `third_party/sglang/python/sglang/srt/disaggregation/`: prefill/decode disaggregation, KV events, decode KV offload manager.
- `third_party/sglang/python/sglang/srt/utils/offloader.py`: model parameter offload and prefetch logic.
- `third_party/sglang/python/sglang/srt/mem_cache/sparsity/`: sparse KV offload/reload coordination.
- `third_party/sglang/python/sglang/srt/mem_cache/hybrid_cache/`: hybrid cache assembly and pool configuration.

### Core Classes

- `Scheduler` in `python/sglang/srt/managers/scheduler.py`: central event loop, request admission, prefill/decode scheduling, cache prefetch hooks, disaggregation setup.
- `RadixKey`, `TreeNode`, `RadixCache` in `python/sglang/srt/mem_cache/radix_cache.py`: prefix cache key and radix tree cache.
- `UnifiedTreeNode`, `UnifiedLRUList`, `UnifiedRadixCache` in `python/sglang/srt/mem_cache/unified_radix_cache.py`: unified cache tree with device/host state and componentized cache values.
- `ReqToTokenPool`, `HybridReqToTokenPool` in `memory_pool.py`: request-to-token index allocation.
- `KVCache`, `MHATokenToKVPool`, `MLATokenToKVPool`, `HybridLinearKVPool`, `MambaPool` in `memory_pool.py`: token-to-KV storage pools.
- `BaseOffloader`, `OffloaderV1`, `OffloaderV2`, `_ModuleOffloader`, `_CpuParamOffloader`, `_ShmCpuParamOffloader`, `_ShardedGpuParamOffloader` in `utils/offloader.py`: model offload/prefetch reference.
- `DecodeKVCacheOffloadManager` in `disaggregation/decode_kvcache_offload_manager.py`: decode-side KV offload lifecycle.
- `OffloadedState` in `disaggregation/kv_events.py`: KV offload event state.
- `HiRadixCache` and related HiCache storage modules in `mem_cache/`: storage-backed prefix cache and prefetch/offload behavior.

### Core Functions And Methods

- `Scheduler.run_event_loop()`, `event_loop_normal()`, `event_loop_overlap()`: scheduler loop modes.
- `Scheduler.process_input_requests()`, `handle_generate_request()`, `_add_request_to_queue()`: request ingestion.
- `Scheduler._prefetch_kvcache()`: request-level KV prefetch hook.
- `Scheduler.get_next_batch_to_run()`, `get_new_batch_prefill()`, `_get_new_batch_prefill_raw()`, `update_running_batch()`, `run_batch()`: scheduling and execution loop.
- `Scheduler.init_disaggregation()`, `update_cache_from_scheduler()`: disaggregated execution and cache update hooks.
- `RadixCache.match_prefix()`, `insert()`, `evict()`, `cache_finished_req()`, `cache_unfinished_req()`, `inc_lock_ref()`, `dec_lock_ref()`: prefix cache lifecycle.
- `UnifiedRadixCache.match_prefix()`, `insert()`, `evict()`, `evict_host()`, `write_backup()`, `load_back()`, `init_load_back()`, `check_hicache_events()`, `ready_to_load_host_cache()`: unified host/device cache lifecycle.
- `MHATokenToKVPool.get_kv_buffer()`, `set_kv_buffer()`, `move_kv_cache()`, `get_cpu_copy()`, `load_cpu_copy()`: KV pool movement.
- `DecodeKVCacheOffloadManager.offload_kv_cache()`, `check_offload_progress()`, `_check_offload_progress()`, `free_tail_kv_cache()`: decode-side offload.
- `OffloaderV2.wrap_modules()`, `post_init()`, `_hook_module_forward_for_offloader()`: parameter offload with prefetch step.

### Data Flow

1. Scheduler receives request objects and tokenizes/prepares them.
2. Scheduler calls prefix-cache matching paths to find reusable token spans.
3. Cache manager allocates request/token/KV pool indices.
4. Prefill/decode batch is built from queued and running requests.
5. Runtime executes batch and writes K/V into token-to-KV pools.
6. Finished or unfinished requests are cached through radix or unified radix cache.
7. HiCache/offload paths can write backup to host/storage and later load back.
8. Disaggregation paths transfer KV or staging data across prefill/decode roles.

### Memory Flow

- Request tokens map through `ReqToTokenPool` to token slots.
- Attention K/V data lives in `MHATokenToKVPool`/`MLATokenToKVPool` style buffers.
- Prefix cache nodes hold value references into token/KV pools and track lock/ref/LRU state.
- Unified radix cache distinguishes device-resident, host-backed, evicted, backuped, and loading states.
- HiCache storage backends can use zero-copy/RDMA-like semantics depending on backend.
- Decode KV offload asynchronously copies incremental KV from device to host/storage with page/stride alignment.
- Model parameter offload paths use pinned CPU memory and non-blocking CUDA transfers; useful as lifecycle reference, but separate from KV offload.

### Reusable Modules

- Radix prefix-cache data model.
- Unified cache state machine: device, host, backup, evicted, loading.
- Scheduler-cache interaction points and request lifecycle.
- HiCache storage backend abstraction and prefetch timeout concepts.
- Decode-side KV offload lifecycle and event tracking.

### Modules Not Recommended For Modification

- Scheduler core implementation.
- Attention backend and memory-pool kernel-facing logic.
- Storage backend native/RDMA integrations.
- Disaggregation transport internals.
- Model parameter offloader unless weight offload enters scope later.

### Best AstraKV-W Insertion Point

Best insertion point: design-level extraction of scheduler/cache signals from `Scheduler._prefetch_kvcache()`, `RadixCache.match_prefix()`, `UnifiedRadixCache.load_back()`, and HiCache storage events.

For AstraKV-W, SGLang should inform the cache-policy state machine rather than be the first code integration target.

## TensorRT-LLM

### Project Role

TensorRT-LLM is a production-grade reference for paged KV cache management, block reuse, capacity scheduling, host/offload tiering, UVM allocation, cache transfer, and Python KV cache manager v2. Its internals are sophisticated and should remain read-only until AstraKV-W has a clear adapter plan.

### Key Directories

- `third_party/TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h`: C++ KV cache manager declarations.
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`: C++ KV cache manager implementation.
- `third_party/TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/capacityScheduler.h`: C++ capacity scheduler.
- `third_party/TensorRT-LLM/tensorrt_llm/runtime/kv_cache_manager_v2/`: Python KV cache manager v2 with cache tiers, pages, storage, copy engine, radix tree.
- `third_party/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/scheduler/`: Python executor scheduler and KV-cache-aware routing.
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/runtime/tllmBuffers.h`: UVM allocator and buffer types.
- `third_party/TensorRT-LLM/cpp/include/tensorrt_llm/runtime/bufferManager.h`: managed/UVM buffer allocation API.
- `third_party/TensorRT-LLM/cpp/include/tensorrt_llm/common/cudaUtils.h`: device memory info including UVM behavior.
- `third_party/TensorRT-LLM/docs/source/developer-guide/kv-transfer.md`: KV transfer architecture reference.

### Core Classes

- `KVCacheBlock` in `kvCacheManager.h`: block metadata, tree links, ref counts, reuse state, priority.
- `GenerationRequest` in `kvCacheManager.h`: per-request token and block bookkeeping.
- `KVCacheBlockPool` in `kvCacheManager.h`: block pool abstraction.
- `WindowBlockManager` and `BlockManager` in `kvCacheManager.h`: window-aware block allocation, reuse, offload/onboard, pool allocation.
- `BaseKVCacheManager` and `KVCacheManager` in `kvCacheManager.h`: public KV cache manager interface and implementation.
- `BaseCapacityScheduler`, `MaxRequestsScheduler`, `MaxUtilizationScheduler`, `GuaranteedNoEvictScheduler`, `StaticBatchScheduler`, `CapacityScheduler` in `capacityScheduler.h`: scheduler policy layer.
- `KVCacheManager` in `runtime/kv_cache_manager_v2/__init__.pyi`: Python v2 cache manager API.
- `_KVCache` in `runtime/kv_cache_manager_v2/__init__.pyi`: per-sequence/cache handle with commit/suspend/resume/page index APIs.
- `GpuCacheLevelStorage`, `HostCacheLevelStorage`, `DiskCacheLevelStorage` in `runtime/kv_cache_manager_v2/_storage/_core.py`: tiered cache storage.
- `BlockRadixTree` and related block/radix classes in `runtime/kv_cache_manager_v2/_block_radix_tree.py`: reuse tree.
- `KVCacheV2Scheduler` in `_torch/pyexecutor/scheduler/scheduler_v2.py`: scheduler using KV cache manager v2.
- `PyCapacityScheduler`, `PyMicroBatchScheduler`, `MaxUtilizationPolicy`, `GuaranteedNoEvictPolicy` in `_torch/pyexecutor/scheduler/scheduler.py`: Python scheduler policies.
- `KVCacheAwareADPRouter` in `_torch/pyexecutor/scheduler/adp_router.py`: prefix-match-aware routing.
- `UVMAllocator`, `UVMBuffer`, `UVMTensor` in `cpp/tensorrt_llm/runtime/tllmBuffers.h`: unified memory allocation references.

### Core Functions And Methods

- `KVCacheManager::allocatePools()`, `addSequenceBatch()`, `addToken()`, `removeToken()`: C++ cache manager lifecycle.
- `BlockManager::allocatePools()`, `allocateBlock()`, `releaseBlocks()`, `pinBlocks()`, `offloadBlock()`, `onboardBlock()`, `storeContextBlocks()`, `storeNewBlock()`, `refreshBlocks()`: block/tier transitions.
- `WindowBlockManager::setOffsets()`, `updateLastCacheBlockOffsets()`, `adjustBlocksIfNeeded()`, `truncateBlocks()`: paged offsets and sliding-window behavior.
- `BaseKVCacheManager::getBlockOffsetsOfBatch()`, `copyBlockOffsets()`, `rewindKVCache()`, `storeContextBlocks()`, `syncTransferManagerWithBufferManager()`: public manager operations.
- `PyCapacityScheduler.schedule_request()`, `_prefill_contributed_blocks()`, `_beneficial_to_skip()`: Python capacity scheduling with prefix reuse awareness.
- `KVCacheV2Scheduler.schedule_request()`, `_schedule_loop()`, `_try_schedule_context()`, `_try_schedule_generation()`, `_try_evict_for_gen()`: scheduler v2 request admission and eviction.
- `KVCacheAwareADPRouter.gather_prefix_matches()`, `_score_rank()`, `route_requests()`: KV-aware distributed routing.
- `runtime/kv_cache_manager_v2.KVCacheManager.create_kv_cache()`, `probe_reuse()`, `resize()`, `get_aggregated_pages()`: Python v2 API.
- `_KVCache.commit()`, `suspend()`, `resume()`, `get_base_page_indices()`, `get_aggregated_page_indices()`: per-cache lifecycle.
- `UVMAllocator::allocate()` and `BufferManager::managed()`: UVM allocation references.

### Data Flow

1. Scheduler receives active/waiting requests and queries KV/cache capacity.
2. Cache manager maps each request to `GenerationRequest` state and per-window block ids.
3. Block manager allocates/reuses/pins/releases blocks and updates offsets.
4. Prefix reuse tree and block keys estimate reusable tokens.
5. Scheduler selects context/generation work based on capacity, block requirements, and eviction policy.
6. Runtime obtains block offsets/page indices for kernel execution.
7. Transfer manager can offload/onboard blocks and sync buffer metadata.
8. Python KV cache manager v2 exposes page/tier abstractions and per-cache commit/suspend/resume lifecycle.

### Memory Flow

- GPU KV cache is organized into fixed-size blocks/pages with per-layer and per-window pools.
- Block offsets are copied into tensors consumed by attention kernels.
- Host cache and disk tiers are represented in both C++ manager paths and Python v2 storage layers.
- UVM can be enabled for KV cache through `use_uvm` config and managed buffer allocation.
- Transfer paths move KV blocks between GPU/host/disk or remote transfer layers.
- Python v2 cache manager introduces cache tiers, pages, slots, pool groups, copy engines, and radix reuse metadata.

### Reusable Modules

- Capacity scheduler vocabulary and block-budget accounting.
- Paged KV block/page metadata and offset-table concepts.
- Tiered cache model: GPU, host, disk.
- Python v2 cache manager API shape for `create_kv_cache`, `probe_reuse`, `commit`, `suspend`, `resume`.
- UVM configuration and memory accounting as reference.
- KV-aware distributed routing concept.

### Modules Not Recommended For Modification

- C++ batch manager internals.
- CUDA kernels and TensorRT plugin paths.
- UVM allocator/runtime buffer internals.
- Python KV cache manager v2 implementation before boundary design is fixed.
- Capacity scheduler internals.

### Best AstraKV-W Insertion Point

Best insertion point: use TensorRT-LLM as a design reference for a future `CacheTier` and `PageIndex` abstraction. Do not integrate directly until AstraKV-W has a stable minimal interface that can represent GPU/host/disk tiers, page indices, block reuse, and suspend/resume semantics.

## End-To-End Data Flow For AstraKV-W Design

The following flow is the safest common denominator across vLLM, LMCache, SGLang, and TensorRT-LLM:

1. Request arrives with request id, token ids, sampling/runtime metadata, and optional session id.
2. Scheduler asks cache layer for prefix match or reusable blocks.
3. Cache layer returns matched token count, block/page ids, load requirements, and invalid/missing ranges.
4. Scheduler decides whether to run prefill/decode, wait for remote KV, prefetch, or evict.
5. KV allocator reserves GPU slots/pages/blocks.
6. Optional prefetch/offload controller moves KV from lower tier to runtime-consumable memory.
7. Worker/runtime builds kernel-facing block table/page index/slot mapping.
8. Attention execution reads/writes KV cache.
9. Cache layer commits full blocks/pages for reuse and emits events.
10. Finished requests free runtime blocks while reusable blocks remain in cache or are offloaded.

## End-To-End Memory Flow For AstraKV-W Design

1. GPU resident KV is the active execution tier.
2. Host pinned/CPU memory is the most common near-storage tier for async transfer.
3. Disk/mmap/raw-block storage is the durable or capacity-extension tier.
4. Remote/distributed storage can be represented as another lower tier with async lookup/load/store.
5. Runtime kernels consume page/block tables, not abstract cache keys.
6. AstraKV-W should bridge cache keys to page/block ids and transfer state, but should not own attention kernels.

## Reusable Module Shortlist

| Priority | Source | Module | Why reuse/reference |
| --- | --- | --- | --- |
| P0 | LMCache | `LMCacheEngine`, `StorageManager`, `PrefetchController`, `PrefetchPolicy` | Directly matches cache/offload/prefetch scope |
| P0 | vLLM | KV connector lifecycle and scheduler metadata | Best first serving-runtime boundary |
| P0 | SGLang | `RadixCache`, `UnifiedRadixCache`, HiCache state machine | Strong cache-policy and host/device lifecycle reference |
| P1 | TensorRT-LLM | Python KV cache manager v2 API and C++ block/page model | Production-grade page/tier abstraction reference |
| P1 | llama.cpp | `llama_mmap`, `llama_mlock`, cache state serialization | mmap and file-backed tier reference |
| P2 | FlashAttention | `PagedKVManager`, tile scheduler concepts | Kernel layout constraints only |

## Do-Not-Modify List

- `third_party/vllm/csrc/`
- `third_party/vllm/vllm/v1/attention/ops/`
- `third_party/flash-attention/csrc/`
- `third_party/flash-attention/hopper/`
- `third_party/TensorRT-LLM/cpp/`
- TensorRT-LLM CUDA/TensorRT plugin paths
- llama.cpp ggml backend internals
- SGLang attention backend and memory-pool kernel-facing code
- LMCache native/raw-block/GDS/DAX internals
- All third-party benchmark directories

## Recommended AstraKV-W Initial Interfaces

These are design recommendations only, not implementation tasks:

- `CacheKey`: token span, model id, layer/group id, page/block hash, optional multimodal key.
- `CacheMatchResult`: matched tokens, matched block/page ids, missing spans, invalid blocks, confidence/source.
- `KVAllocationView`: runtime block ids, page ids, slot mapping, layer group, device.
- `PrefetchPlan`: keys, target tier, deadline/priority, expected bytes, request association.
- `OffloadPlan`: source tier, destination tier, block/page ids, eviction priority, completion event.
- `SchedulerHint`: wait/continue/preempt/prefetch/load/store recommendation with reason.
- `TransferEvent`: submitted, in-flight, completed, failed, invalidated.

## Best Overall Insertion Point

The best first insertion point for AstraKV-W is an adapter-level analysis boundary between runtime scheduler output and external KV cache operations:

- For vLLM: observe and model `Scheduler._build_kv_connector_meta()`, `LMCacheConnectorV1`, and `KVCacheManager` block allocation events.
- For LMCache: model `LMCacheEngine` and `PrefetchController` as the cache/offload/prefetch backend contract.
- For SGLang: model radix/unified-cache state transitions and scheduler cache hooks.
- For TensorRT-LLM: model page/tier abstractions and UVM constraints, not the internal implementation.

This keeps AstraKV-W outside third-party kernels and schedulers while still giving it a clear future path to cache-aware scheduling and tier-aware KV movement.

## Next Analysis Tasks

1. Draw an interface boundary diagram from request scheduler to cache engine to storage tiers.
2. Build a canonical field mapping table across vLLM, LMCache, SGLang, and TensorRT-LLM.
3. Define read-only trace schemas for cache match, allocation, prefetch, offload, and completion events.
4. Decide whether the first integration study is LMCache-first, vLLM-connector-first, or SGLang-cache-first.
5. Keep third-party source trees clean until implementation is explicitly authorized.
