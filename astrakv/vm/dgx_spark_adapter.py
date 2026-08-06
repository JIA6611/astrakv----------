"""DGX Spark-oriented mmap KV chunk adapter.

This adapter bridges runtime-agnostic ``KVChunkMeta`` records to the
file-backed ``MMapKVCache`` implementation. It is still independent from vLLM
internals, but it turns the DGX Spark claim into executable evidence:

- KV chunks are mapped to mmap-backed blocks.
- NVMe/local disk is the backing store.
- ``MADV_WILLNEED`` is used as chunk prefetch.
- ``MADV_DONTNEED`` is used as chunk eviction.
- ``mincore`` reports chunk residency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # The adapter is optional outside a Linux/NumPy VM-PoC host.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - import-only behavior.
    np = None  # type: ignore[assignment]

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.vm.mmap_kv_cache import MMapKVCache, MMapKVCacheConfig


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("missing_optional_dependency: numpy is required for DgxSparkKVAdapter")
    return np


@dataclass(frozen=True, slots=True)
class DgxSparkKVAdapterConfig:
    """Configuration for the DGX Spark mmap-backed KV adapter."""

    backing_file: str
    total_blocks: int = 128
    block_size_bytes: int = 1024 * 1024
    dtype_str: str = "float16"
    target_tier: MemoryTier = MemoryTier.SSD
    keep_backing_file: bool = False

    @property
    def dtype(self) -> np.dtype:
        return _require_numpy().dtype(self.dtype_str)


@dataclass(frozen=True, slots=True)
class DgxSparkChunkRecord:
    """Mmap placement record for one logical KV chunk."""

    chunk: KVChunkMeta
    block_ids: tuple[int, ...]
    size_bytes: int
    dtype_str: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "request_id": self.chunk.request_id,
            "layer_id": self.chunk.layer_id,
            "token_span": [self.chunk.start_token, self.chunk.end_token],
            "block_ids": list(self.block_ids),
            "size_bytes": self.size_bytes,
            "dtype": self.dtype_str,
            "tier": self.chunk.tier.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DgxSparkChunkAction:
    """Trace event emitted by the DGX Spark adapter."""

    action: str
    chunk_id: str
    block_ids: tuple[int, ...]
    ok: bool
    latency_us: float
    resident_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "chunk_id": self.chunk_id,
            "block_ids": list(self.block_ids),
            "ok": self.ok,
            "latency_us": round(self.latency_us, 3),
            "resident_ratio": round(self.resident_ratio, 6),
            "metadata": dict(self.metadata),
        }


class DgxSparkKVAdapter:
    """Map ``KVChunkMeta`` objects onto an mmap-backed KV store."""

    def __init__(self, config: DgxSparkKVAdapterConfig) -> None:
        _require_numpy()
        self.config = config
        self.cache = MMapKVCache(
            MMapKVCacheConfig(
                total_blocks=config.total_blocks,
                block_size_bytes=config.block_size_bytes,
                backing_file=config.backing_file,
                dtype_str=config.dtype_str,
            )
        )
        self._records: dict[str, DgxSparkChunkRecord] = {}
        self._next_block = 0
        self.events: list[DgxSparkChunkAction] = []

    def close(self) -> None:
        self.cache.close()
        if not self.config.keep_backing_file:
            try:
                Path(self.config.backing_file).unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self) -> "DgxSparkKVAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def register_chunk(self, chunk: KVChunkMeta) -> DgxSparkChunkRecord:
        """Register a logical chunk and allocate mmap blocks for it."""

        size_bytes = int(chunk.size_bytes or self.config.block_size_bytes)
        block_count = max(1, math.ceil(size_bytes / self.config.block_size_bytes))
        if chunk.block_ids:
            block_ids = tuple(int(item) for item in chunk.block_ids)
            if len(block_ids) < block_count:
                raise ValueError("chunk.block_ids does not cover chunk size_bytes")
            block_ids = block_ids[:block_count]
        else:
            block_ids = tuple(range(self._next_block, self._next_block + block_count))
            self._next_block += block_count

        for block_id in block_ids:
            if block_id < 0 or block_id >= self.config.total_blocks:
                raise IndexError(f"block_id {block_id} out of range")

        mapped = chunk.with_tier(self.config.target_tier, device="dgx-spark-mmap")
        mapped.size_bytes = size_bytes
        mapped.metadata.update(
            {
                "adapter": "dgx_spark_mmap",
                "backing_file": self.config.backing_file,
                "block_size_bytes": self.config.block_size_bytes,
            }
        )
        record = DgxSparkChunkRecord(
            chunk=mapped,
            block_ids=block_ids,
            size_bytes=size_bytes,
            dtype_str=self.config.dtype_str,
        )
        self._records[mapped.chunk_id] = record
        return record

    def write_chunk(self, chunk_id: str, data: np.ndarray) -> DgxSparkChunkAction:
        record = self._get_record(chunk_id)
        payload = data.astype(self.config.dtype, copy=False)
        expected_items = math.ceil(record.size_bytes / self.config.dtype.itemsize)
        if payload.size < expected_items:
            padded = np.zeros(expected_items, dtype=self.config.dtype)
            padded[: payload.size] = payload
            payload = padded
        payload = payload[:expected_items]

        block_items = self.config.block_size_bytes // self.config.dtype.itemsize
        start = time.perf_counter()
        for index, block_id in enumerate(record.block_ids):
            block = np.zeros(block_items, dtype=self.config.dtype)
            begin = index * block_items
            end = min(begin + block_items, payload.size)
            if begin < end:
                block[: end - begin] = payload[begin:end]
            self.cache.write_block(block_id, block)
        return self._record_event("write_chunk", record, True, start)

    def read_chunk(self, chunk_id: str) -> tuple[np.ndarray, DgxSparkChunkAction]:
        record = self._get_record(chunk_id)
        start = time.perf_counter()
        blocks = [self.cache.read_block(block_id) for block_id in record.block_ids]
        data = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        expected_items = math.ceil(record.size_bytes / self.config.dtype.itemsize)
        event = self._record_event("read_chunk", record, True, start)
        return data[:expected_items], event

    def prefetch_chunk(self, chunk_id: str) -> DgxSparkChunkAction:
        record = self._get_record(chunk_id)
        start = time.perf_counter()
        ok = self.cache.prefetch_batch(list(record.block_ids)) == len(record.block_ids)
        return self._record_event("prefetch_chunk", record, ok, start)

    def evict_chunk(self, chunk_id: str) -> DgxSparkChunkAction:
        record = self._get_record(chunk_id)
        start = time.perf_counter()
        ok = self.cache.evict_batch(list(record.block_ids)) == len(record.block_ids)
        return self._record_event("evict_chunk", record, ok, start)

    def evict_chunks(self, chunk_ids: Iterable[str]) -> list[DgxSparkChunkAction]:
        return [self.evict_chunk(chunk_id) for chunk_id in chunk_ids]

    def resident_ratio(self, chunk_id: str) -> float:
        record = self._get_record(chunk_id)
        resident = self.cache.get_resident_blocks()
        values = [resident.get(block_id, 0.0) for block_id in record.block_ids]
        return sum(values) / max(1, len(values))

    def chunk_records(self) -> list[dict[str, Any]]:
        return [record.to_record() for record in self._records.values()]

    def _get_record(self, chunk_id: str) -> DgxSparkChunkRecord:
        try:
            return self._records[chunk_id]
        except KeyError as exc:
            raise KeyError(f"unknown chunk_id: {chunk_id}") from exc

    def _record_event(
        self,
        action: str,
        record: DgxSparkChunkRecord,
        ok: bool,
        started: float,
        metadata: dict[str, Any] | None = None,
    ) -> DgxSparkChunkAction:
        event = DgxSparkChunkAction(
            action=action,
            chunk_id=record.chunk.chunk_id,
            block_ids=record.block_ids,
            ok=ok,
            latency_us=(time.perf_counter() - started) * 1_000_000,
            resident_ratio=self.resident_ratio(record.chunk.chunk_id),
            metadata=metadata or {},
        )
        self.events.append(event)
        return event
