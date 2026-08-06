"""AstraKV-W: Memory-Constrained LLM Inference with Virtual Memory Techniques.

AstraKV-W is a research engineering framework for studying memory-constrained
LLM inference, focusing on KV cache management, prefetching, offloading,
scheduling, and OS-level virtual memory integration.

Quick start::

    from astrakv.runtime import VirtualMemoryDemoRunner, VMDemoConfig
    from astrakv.vm import MMapKVCache, MMapKVCacheConfig
    from astrakv.prefetch import SelectiveKVPrefetchMVP
"""

__version__ = "0.2.0"
