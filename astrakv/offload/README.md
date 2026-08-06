# offload

Tier placement skeletons.

## Files

- `tier_placement.py`: `TierPlacementManager` and `PlacementRecord`.

## Scope

The manager records current and target tiers for KV chunks. It does not copy
memory, open files, use CUDA, or implement an eviction/offload optimization
policy.

Future adapters can use this state to drive LMCache, vLLM, SGLang, or
TensorRT-LLM transfer mechanisms.
