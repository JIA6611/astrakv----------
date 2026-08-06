# prefetch

Async prefetch lifecycle skeletons and Selective KV Prefetch MVP.

## Files

- `async_engine.py`: `AsyncPrefetchEngine`, `PrefetchRequest`,
  `PrefetchResult`, and `PrefetchStatus`.
- `scorer.py`: profile-guided `ChunkScorer` that turns ProfileDB chunk
  statistics into `prefetch`, `keep`, `offload`, or `drop` policy hints.
- `selective_kv.py`: decode-stage Selective KV Prefetch MVP with GPU/CPU tiers,
  async prefetch queue, LRU eviction, and metrics.

## Scope

The async engine tracks request lifecycle and delegates real work to an adapter
callback. The Selective KV Prefetch MVP is a pure Python, runtime-independent
implementation for validating prefetch behavior.

The scorer is advisory only. It reads reusable profile statistics and explains
why a chunk should be prefetched, kept, offloaded, or dropped. It does not move
memory or submit backend requests by itself.

## Current Boundaries

- No CUDA kernel.
- No tensor allocation.
- No scheduler implementation.
- No vLLM core modification.
- No third-party source modification.

The MVP models KV block residency and movement between CPU and GPU tiers. It is
not a production runtime integration.
