"""Selective KV prefetch MVP.

This module implements a small runtime-agnostic MVP for decode-stage KV
prefetch. It models two KV tiers, GPU and CPU, with an asynchronous prefetch
queue and GPU-side LRU eviction.

It intentionally does not use CUDA, does not allocate tensors, and does not
modify any third-party runtime.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from astrakv.kv_cache.metadata import MemoryTier


class KVAccessSource(str, Enum):
    GPU_HIT = "gpu_hit"
    PREFETCH_HIT = "prefetch_hit"
    CPU_MISS = "cpu_miss"


@dataclass(frozen=True, slots=True)
class KVBlockRef:
    block_id: str
    size_bytes: int


@dataclass(slots=True)
class KVResidentBlock:
    ref: KVBlockRef
    tier: MemoryTier
    prefetched: bool = False
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class KVAccessResult:
    block_id: str
    source: KVAccessSource
    latency_ms: float


@dataclass(slots=True)
class SelectivePrefetchMetrics:
    demand_lookups: int = 0
    gpu_hits: int = 0
    prefetch_hits: int = 0
    cpu_misses: int = 0
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_dropped: int = 0
    prefetch_wasted: int = 0
    evictions: int = 0
    gpu_blocks_peak: int = 0
    gpu_bytes_peak: int = 0

    @property
    def prefetch_hit_rate(self) -> float:
        return self.prefetch_hits / max(1, self.demand_lookups)

    @property
    def prefetch_waste_rate(self) -> float:
        return self.prefetch_wasted / max(1, self.prefetch_completed)

    def to_record(self) -> dict[str, float | int]:
        return {
            "demand_lookups": self.demand_lookups,
            "gpu_hits": self.gpu_hits,
            "prefetch_hits": self.prefetch_hits,
            "cpu_misses": self.cpu_misses,
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_completed": self.prefetch_completed,
            "prefetch_dropped": self.prefetch_dropped,
            "prefetch_wasted": self.prefetch_wasted,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "prefetch_waste_rate": self.prefetch_waste_rate,
            "evictions": self.evictions,
            "gpu_blocks_peak": self.gpu_blocks_peak,
            "gpu_bytes_peak": self.gpu_bytes_peak,
        }


@dataclass(frozen=True, slots=True)
class SelectiveKVPrefetchConfig:
    gpu_capacity_blocks: int = 8
    block_size_bytes: int = 16 * 1024 * 1024
    prefetch_window: int = 1
    max_queue_size: int = 16
    prefetch_latency_ms: float = 0.15
    cpu_miss_latency_ms: float = 1.2
    gpu_hit_latency_ms: float = 0.02
    prefetch_hit_latency_ms: float = 0.03


@dataclass(frozen=True, slots=True)
class PrefetchQueueItem:
    block_id: str
    reason: str


class SelectiveKVPrefetchMVP:
    """Two-tier selective KV prefetch MVP for decode-stage simulations."""

    def __init__(self, config: SelectiveKVPrefetchConfig | None = None) -> None:
        self.config = config or SelectiveKVPrefetchConfig()
        self.metrics = SelectivePrefetchMetrics()
        self.cpu_blocks: dict[str, KVBlockRef] = {}
        self.gpu_blocks: OrderedDict[str, KVResidentBlock] = OrderedDict()
        self._queue: asyncio.Queue[PrefetchQueueItem] = asyncio.Queue(
            maxsize=max(1, self.config.max_queue_size)
        )
        self._inflight: set[str] = set()
        self._closed = False
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        await self._queue.join()
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._mark_remaining_prefetch_waste()

    def add_cpu_blocks(self, blocks: Iterable[KVBlockRef]) -> None:
        for block in blocks:
            self.cpu_blocks[block.block_id] = block

    def predict_next_blocks(self, trace: list[str], position: int) -> list[str]:
        """Predict future block ids from the decode trace.

        The MVP intentionally uses a simple next-N predictor. It is transparent,
        deterministic, and easy to replace later.
        """

        window = max(0, self.config.prefetch_window)
        if window == 0:
            return []
        seen: set[str] = set()
        predicted: list[str] = []
        for block_id in trace[position + 1 : position + 1 + window]:
            if block_id in seen:
                continue
            seen.add(block_id)
            predicted.append(block_id)
        return predicted

    def submit_prefetch(self, block_id: str, reason: str = "next_step_prediction") -> bool:
        if block_id in self.gpu_blocks or block_id in self._inflight:
            return False
        if block_id not in self.cpu_blocks:
            self.cpu_blocks[block_id] = KVBlockRef(
                block_id=block_id,
                size_bytes=self.config.block_size_bytes,
            )
        try:
            self._queue.put_nowait(PrefetchQueueItem(block_id=block_id, reason=reason))
        except asyncio.QueueFull:
            self.metrics.prefetch_dropped += 1
            return False
        self._inflight.add(block_id)
        self.metrics.prefetch_submitted += 1
        return True

    async def submit_predictions(self, trace: list[str], position: int) -> int:
        count = 0
        for block_id in self.predict_next_blocks(trace, position):
            count += int(self.submit_prefetch(block_id))
        await asyncio.sleep(0)
        return count

    async def access(self, block_id: str) -> KVAccessResult:
        self.metrics.demand_lookups += 1

        resident = self.gpu_blocks.get(block_id)
        if resident is not None:
            self.gpu_blocks.move_to_end(block_id)
            if resident.prefetched and not resident.consumed:
                resident.consumed = True
                self.metrics.prefetch_hits += 1
                return KVAccessResult(
                    block_id=block_id,
                    source=KVAccessSource.PREFETCH_HIT,
                    latency_ms=self.config.prefetch_hit_latency_ms,
                )
            self.metrics.gpu_hits += 1
            return KVAccessResult(
                block_id=block_id,
                source=KVAccessSource.GPU_HIT,
                latency_ms=self.config.gpu_hit_latency_ms,
            )

        self.metrics.cpu_misses += 1
        await asyncio.sleep(max(0.0, self.config.cpu_miss_latency_ms) / 1000.0)
        self._move_to_gpu(block_id, prefetched=False)
        return KVAccessResult(
            block_id=block_id,
            source=KVAccessSource.CPU_MISS,
            latency_ms=self.config.cpu_miss_latency_ms,
        )

    def gpu_memory_bytes(self) -> int:
        return sum(block.ref.size_bytes for block in self.gpu_blocks.values())

    def to_metrics_record(self) -> dict[str, float | int]:
        return self.metrics.to_record()

    async def _worker_loop(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await asyncio.sleep(max(0.0, self.config.prefetch_latency_ms) / 1000.0)
                if item.block_id not in self.gpu_blocks:
                    self._move_to_gpu(item.block_id, prefetched=True)
                    self.metrics.prefetch_completed += 1
            finally:
                self._inflight.discard(item.block_id)
                self._queue.task_done()

    def _move_to_gpu(self, block_id: str, prefetched: bool) -> None:
        ref = self.cpu_blocks.get(block_id)
        if ref is None:
            ref = KVBlockRef(block_id=block_id, size_bytes=self.config.block_size_bytes)
            self.cpu_blocks[block_id] = ref
        self.gpu_blocks[block_id] = KVResidentBlock(
            ref=ref,
            tier=MemoryTier.GPU,
            prefetched=prefetched,
        )
        self.gpu_blocks.move_to_end(block_id)
        self._evict_if_needed()
        self._update_gpu_peak()

    def _evict_if_needed(self) -> None:
        capacity = max(1, self.config.gpu_capacity_blocks)
        while len(self.gpu_blocks) > capacity:
            _, evicted = self.gpu_blocks.popitem(last=False)
            self.metrics.evictions += 1
            if evicted.prefetched and not evicted.consumed:
                self.metrics.prefetch_wasted += 1

    def _update_gpu_peak(self) -> None:
        blocks = len(self.gpu_blocks)
        bytes_used = self.gpu_memory_bytes()
        self.metrics.gpu_blocks_peak = max(self.metrics.gpu_blocks_peak, blocks)
        self.metrics.gpu_bytes_peak = max(self.metrics.gpu_bytes_peak, bytes_used)

    def _mark_remaining_prefetch_waste(self) -> None:
        for resident in self.gpu_blocks.values():
            if resident.prefetched and not resident.consumed:
                resident.consumed = True
                self.metrics.prefetch_wasted += 1
