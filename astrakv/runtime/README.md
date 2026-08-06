# runtime

Runtime adapter skeletons and object-management boundaries.

This package does not implement a serving runtime. It defines adapter-facing
objects that future vLLM, SGLang, LMCache, or TensorRT-LLM integrations can use
without modifying third-party core code.

## Files

- `adapters.py`: `RuntimeAdapter` protocol and `RuntimeRequest` metadata.
- `cache_events.py`: read-only parser for vLLM/LMCache cache-related logs and benchmark artifacts.
- `endpoint_prefetch.py`: endpoint-level OpenAI-compatible warmup/prefetch adapter for real backends.
- `failure_recovery.py`: read-only failure classification and passive fallback hint planner.
- `memory_pressure.py`: read-only memory pressure controller that converts benchmark/sample/trace artifacts into passive pressure decisions and scheduler hints.
- `moe_events.py`: read-only MoE router/expert event parser for token-to-expert activation traces.
- `object_manager.py`: `RuntimeObjectManager` composition layer for KV metadata,
  block table records, placement records, and prefetch lifecycle state.
- `profile_db.py`: lightweight reusable JSON ProfileDB built from unified trace events.
- `trace_schema.py`: unified trace schema and adapters for cache events, prefetch events, and memory samples.
- `vm_backend.py`: reusable file-backed `mmap` virtual-memory backend helpers for OS VM PoC experiments and future KV chunk/page mapping.

## Boundaries

- No vLLM core modification.
- No scheduler implementation.
- No CUDA kernel or tensor allocation.
- Runtime-specific behavior must live behind adapters.
- Experiment runners should call reusable runtime helpers instead of embedding backend logic directly.
