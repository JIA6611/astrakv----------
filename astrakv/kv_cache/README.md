# kv_cache

Runtime-agnostic KV cache skeletons.

## Files

- `metadata.py`: `MemoryTier` and `KVChunkMeta`.
- `block_table.py`: `KVBlockTable` and `KVBlockEntry`.
- `partial_load.py`: partial KV load request, token-span, decision, and
  summary records for adapter-facing partial-load plans.

## Scope

The package tracks logical KV metadata only. It does not own tensors, allocate
GPU pages, implement paged attention, or modify third-party cache managers.

Future runtime adapters can translate these records into vLLM block ids, SGLang
KV pool indices, LMCache keys, or TensorRT-LLM page descriptors.

Partial-load records describe which layer/token spans should be loaded, skipped,
or loaded fully. They are intent records only; real tensor movement must be
implemented and validated by a backend adapter.
