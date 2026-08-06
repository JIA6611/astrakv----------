"""Runtime object manager skeleton."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from astrakv.kv_cache.block_table import KVBlockTable
from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.offload.tier_placement import TierPlacementManager
from astrakv.prefetch.async_engine import AsyncPrefetchEngine, PrefetchRequest, PrefetchResult


@dataclass(slots=True)
class RuntimeObjectSnapshot:
    chunks: list[dict[str, Any]]
    block_table: list[dict[str, Any]]
    placements: list[dict[str, Any]]
    prefetch: list[dict[str, Any]]


class RuntimeObjectManager:
    """Coordinates metadata objects across KV, prefetch, and placement layers."""

    def __init__(
        self,
        block_table: KVBlockTable | None = None,
        placement_manager: TierPlacementManager | None = None,
        prefetch_engine: AsyncPrefetchEngine | None = None,
    ) -> None:
        self.block_table = block_table or KVBlockTable()
        self.placement_manager = placement_manager or TierPlacementManager()
        self.prefetch_engine = prefetch_engine or AsyncPrefetchEngine()
        self._chunks: dict[str, KVChunkMeta] = {}

    def register_chunk(self, meta: KVChunkMeta) -> KVChunkMeta:
        self._chunks[meta.chunk_id] = meta
        self.block_table.add_chunk(meta)
        self.placement_manager.register_chunk(meta)
        return meta

    def get_chunk(self, chunk_id: str) -> KVChunkMeta | None:
        return self._chunks.get(chunk_id)

    def list_request_chunks(self, request_id: str) -> list[KVChunkMeta]:
        entries = self.block_table.list_request_chunks(request_id)
        return [self._chunks[entry.chunk_id] for entry in entries if entry.chunk_id in self._chunks]

    def plan_offload(self, chunk_id: str, target_tier: MemoryTier, reason: str) -> None:
        self.placement_manager.plan_move(chunk_id, target_tier, reason)

    async def prefetch_chunk(
        self,
        chunk_id: str,
        target_tier: MemoryTier = MemoryTier.GPU,
        priority: int = 0,
    ) -> PrefetchResult:
        if chunk_id not in self._chunks:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        request = PrefetchRequest(chunk_id=chunk_id, target_tier=target_tier, priority=priority)
        return await self.prefetch_engine.submit(request)

    def prefetch_chunk_sync(
        self,
        chunk_id: str,
        target_tier: MemoryTier = MemoryTier.GPU,
        priority: int = 0,
    ) -> PrefetchResult:
        return asyncio.run(self.prefetch_chunk(chunk_id, target_tier, priority))

    def release_request(self, request_id: str) -> list[KVChunkMeta]:
        removed_entries = self.block_table.clear_request(request_id)
        removed: list[KVChunkMeta] = []
        for entry in removed_entries:
            meta = self._chunks.pop(entry.chunk_id, None)
            if meta is not None:
                removed.append(meta)
        return removed

    def snapshot(self) -> RuntimeObjectSnapshot:
        return RuntimeObjectSnapshot(
            chunks=[meta.to_record() for meta in sorted(self._chunks.values(), key=lambda item: item.chunk_id)],
            block_table=self.block_table.to_records(),
            placements=self.placement_manager.to_records(),
            prefetch=self.prefetch_engine.to_records(),
        )
