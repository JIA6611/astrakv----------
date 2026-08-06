"""Virtual memory integration module for AstraKV-W.

This module provides OS-level virtual-memory mechanisms applied to LLM inference
memory-management experiments. The current evidence path uses file-backed mmap
and DGX Spark UMA helpers. userfaultfd on-demand page loading and layer-level
model weight offloading are optional PoCs, not part of the default competition
benchmark path.

These implementations are independent from vLLM, LMCache, CUDA, and real
benchmark runners so they can be verified as standalone evidence and later
integrated through adapters.
"""

from .mmap_kv_cache import MMapKVCache, MMapKVCacheConfig, MMapKVCacheStats, vm_platform_available, vm_platform_reason, vm_numpy_available
from .uffd_kv_loader import UFDDemandLoader, UFFDLoaderConfig, uffd_platform_available


def _get_layer_offload():
    """Lazy import for the layer_offload module (requires torch)."""
    from .layer_offload import (  # noqa: E402
        LayerOffloadManager,
        LayerOffloadConfig,
        LayerOffloadResult,
    )
    return (LayerOffloadManager, LayerOffloadConfig, LayerOffloadResult)


def _get_dgx_adapter():
    """Lazy import so a non-NumPy host can still import the VM boundary."""
    from .dgx_spark_adapter import (  # noqa: E402
        DgxSparkChunkAction,
        DgxSparkChunkRecord,
        DgxSparkKVAdapter,
        DgxSparkKVAdapterConfig,
    )
    return (DgxSparkChunkAction, DgxSparkChunkRecord, DgxSparkKVAdapter, DgxSparkKVAdapterConfig)


def __getattr__(name: str):
    if name in ("LayerOffloadManager", "LayerOffloadConfig", "LayerOffloadResult"):
        module_attrs = _get_layer_offload()
        mapping = dict(zip(["LayerOffloadManager", "LayerOffloadConfig", "LayerOffloadResult"], module_attrs))
        return mapping[name]
    if name in ("DgxSparkChunkAction", "DgxSparkChunkRecord", "DgxSparkKVAdapter", "DgxSparkKVAdapterConfig"):
        module_attrs = _get_dgx_adapter()
        mapping = dict(zip(["DgxSparkChunkAction", "DgxSparkChunkRecord", "DgxSparkKVAdapter", "DgxSparkKVAdapterConfig"], module_attrs))
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MMapKVCache",
    "MMapKVCacheConfig",
    "MMapKVCacheStats",
    "vm_platform_available",
    "vm_platform_reason",
    "vm_numpy_available",
    "DgxSparkChunkAction",
    "DgxSparkChunkRecord",
    "DgxSparkKVAdapter",
    "DgxSparkKVAdapterConfig",
    "UFDDemandLoader",
    "UFFDLoaderConfig",
    "uffd_platform_available",
    "LayerOffloadManager",
    "LayerOffloadConfig",
    "LayerOffloadResult",
]
