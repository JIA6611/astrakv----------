"""Tier placement manager skeleton.

The manager records desired and current placement. It does not perform data
movement; adapters are responsible for moving runtime-owned KV tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier


@dataclass(slots=True)
class PlacementRecord:
    chunk_id: str
    current_tier: MemoryTier
    target_tier: MemoryTier
    reason: str
    status: str = "planned"
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "current_tier": self.current_tier.value,
            "target_tier": self.target_tier.value,
            "reason": self.reason,
            "status": self.status,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


class TierPlacementManager:
    """Tracks placement intent for KV chunks across memory tiers."""

    def __init__(self, default_tier: MemoryTier = MemoryTier.GPU) -> None:
        self.default_tier = default_tier
        self._placements: dict[str, PlacementRecord] = {}

    def register_chunk(self, meta: KVChunkMeta) -> PlacementRecord:
        tier = meta.tier if meta.tier is not MemoryTier.UNKNOWN else self.default_tier
        record = PlacementRecord(
            chunk_id=meta.chunk_id,
            current_tier=tier,
            target_tier=tier,
            reason="registered",
            status="resident",
        )
        self._placements[meta.chunk_id] = record
        return record

    def plan_move(
        self,
        chunk_id: str,
        target_tier: MemoryTier,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> PlacementRecord:
        existing = self._placements.get(chunk_id)
        current = existing.current_tier if existing is not None else MemoryTier.UNKNOWN
        record = PlacementRecord(
            chunk_id=chunk_id,
            current_tier=current,
            target_tier=target_tier,
            reason=reason,
            status="planned",
            metadata=metadata or {},
        )
        self._placements[chunk_id] = record
        return record

    def mark_in_flight(self, chunk_id: str) -> PlacementRecord:
        return self._update_status(chunk_id, "in_flight")

    def mark_resident(self, chunk_id: str, tier: MemoryTier | None = None) -> PlacementRecord:
        record = self._require(chunk_id)
        target = tier if tier is not None else record.target_tier
        record.current_tier = target
        record.target_tier = target
        record.status = "resident"
        record.updated_at = datetime.now().isoformat(timespec="seconds")
        return record

    def mark_failed(self, chunk_id: str, reason: str) -> PlacementRecord:
        record = self._require(chunk_id)
        record.status = "failed"
        record.reason = reason
        record.updated_at = datetime.now().isoformat(timespec="seconds")
        return record

    def get(self, chunk_id: str) -> PlacementRecord | None:
        return self._placements.get(chunk_id)

    def to_records(self) -> list[dict[str, Any]]:
        return [
            record.to_record()
            for record in sorted(self._placements.values(), key=lambda item: item.chunk_id)
        ]

    def _update_status(self, chunk_id: str, status: str) -> PlacementRecord:
        record = self._require(chunk_id)
        record.status = status
        record.updated_at = datetime.now().isoformat(timespec="seconds")
        return record

    def _require(self, chunk_id: str) -> PlacementRecord:
        if chunk_id not in self._placements:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        return self._placements[chunk_id]
