"""KV cache metadata objects.

This module defines runtime-agnostic metadata. It does not allocate tensors,
touch CUDA kernels, or depend on vLLM internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryTier(str, Enum):
    """Storage tier for a KV chunk."""

    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"
    REMOTE = "remote"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class KVChunkMeta:
    """Metadata for one logical KV-cache chunk.

    The object describes ownership and placement only. Payload tensors remain
    owned by runtime adapters or third-party cache/storage systems.
    """

    request_id: str
    layer_id: int
    start_token: int
    end_token: int
    block_ids: tuple[int, ...] = field(default_factory=tuple)
    chunk_id: str = field(default_factory=lambda: uuid4().hex)
    tier: MemoryTier = MemoryTier.UNKNOWN
    dtype: str | None = None
    device: str | None = None
    size_bytes: int | None = None
    cache_key: str | None = None
    adapter_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token < self.start_token:
            raise ValueError("end_token must be greater than or equal to start_token")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not isinstance(self.tier, MemoryTier):
            self.tier = MemoryTier(str(self.tier))

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def with_tier(self, tier: MemoryTier, device: str | None = None) -> "KVChunkMeta":
        return KVChunkMeta(
            request_id=self.request_id,
            layer_id=self.layer_id,
            start_token=self.start_token,
            end_token=self.end_token,
            block_ids=self.block_ids,
            chunk_id=self.chunk_id,
            tier=tier,
            dtype=self.dtype,
            device=device if device is not None else self.device,
            size_bytes=self.size_bytes,
            cache_key=self.cache_key,
            adapter_name=self.adapter_name,
            metadata=dict(self.metadata),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
            "block_ids": list(self.block_ids),
            "tier": self.tier.value,
            "dtype": self.dtype,
            "device": self.device,
            "size_bytes": self.size_bytes,
            "cache_key": self.cache_key,
            "adapter_name": self.adapter_name,
            "metadata": dict(self.metadata),
        }
