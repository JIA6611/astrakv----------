"""Load-vs-recompute advisory decisions.

This module produces passive scheduling decisions from ProfileDB statistics and
optional partial KV load plans. It does not enqueue requests, run kernels, or
move memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from astrakv.scheduler.hints import SchedulerHint


class LoadRecomputeAction(str, Enum):
    LOAD = "load"
    RECOMPUTE = "recompute"
    DEFER = "defer"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class PartialPlanStats:
    chunk_id: str
    loaded_tokens: int = 0
    skipped_tokens: int = 0
    loaded_bytes: int = 0
    skipped_bytes: int = 0
    action: str = ""
    reason: str = ""

    @property
    def selected_tokens(self) -> int:
        return self.loaded_tokens

    @property
    def total_tokens(self) -> int:
        return self.loaded_tokens + self.skipped_tokens

    @property
    def byte_saving_rate(self) -> float:
        total = self.loaded_bytes + self.skipped_bytes
        return self.skipped_bytes / max(1, total)


@dataclass(frozen=True, slots=True)
class LoadRecomputeConfig:
    memory_pressure: float = 0.0
    deadline_ms: float = 120.0
    load_latency_fallback_ms: float = 80.0
    io_bandwidth_bytes_per_ms: float = 64 * 1024 * 1024 / 1000
    recompute_latency_per_token_ms: float = 0.08
    recompute_overhead_ms: float = 8.0
    recompute_penalty: float = 1.10
    load_preference_margin: float = 0.90
    low_reuse_drop_threshold: float = 0.03
    low_reuse_drop_memory_pressure: float = 0.65
    defer_deadline_slack: float = 0.20
    default_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class LoadRecomputeDecision:
    chunk_id: str
    workload_id: str
    action: LoadRecomputeAction
    request_id: str = ""
    case: str = ""
    cache_key: str = ""
    reuse_frequency: float = 0.0
    cache_hit_rate: float = 0.0
    prefetch_hit_rate: float = 0.0
    estimated_load_ms: float = 0.0
    estimated_recompute_ms: float = 0.0
    estimated_loaded_bytes: int = 0
    estimated_skipped_bytes: int = 0
    memory_pressure: float = 0.0
    deadline_ms: float = 0.0
    priority: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "workload_id": self.workload_id,
            "request_id": self.request_id,
            "case": self.case,
            "cache_key": self.cache_key,
            "action": self.action.value,
            "reuse_frequency": self.reuse_frequency,
            "cache_hit_rate": self.cache_hit_rate,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "estimated_load_ms": self.estimated_load_ms,
            "estimated_recompute_ms": self.estimated_recompute_ms,
            "estimated_loaded_bytes": self.estimated_loaded_bytes,
            "estimated_skipped_bytes": self.estimated_skipped_bytes,
            "memory_pressure": self.memory_pressure,
            "deadline_ms": self.deadline_ms,
            "priority": self.priority,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_hint(self) -> SchedulerHint:
        return SchedulerHint(
            request_id=self.request_id or self.case or self.chunk_id,
            action=self.action.value,
            reason=self.reason,
            priority=self.priority,
            metadata={
                "chunk_id": self.chunk_id,
                "workload_id": self.workload_id,
                "estimated_load_ms": self.estimated_load_ms,
                "estimated_recompute_ms": self.estimated_recompute_ms,
                "estimated_loaded_bytes": self.estimated_loaded_bytes,
                "estimated_skipped_bytes": self.estimated_skipped_bytes,
                **dict(self.metadata),
            },
        )


class LoadRecomputePlanner:
    def __init__(self, config: LoadRecomputeConfig | None = None) -> None:
        self.config = config or LoadRecomputeConfig()

    def decide_profile(
        self,
        profile: ChunkProfile,
        partial_plan: PartialPlanStats | None = None,
    ) -> LoadRecomputeDecision:
        cfg = self.config
        loaded_bytes = partial_plan.loaded_bytes if partial_plan else profile.bytes_loaded
        skipped_bytes = partial_plan.skipped_bytes if partial_plan else 0
        selected_tokens = partial_plan.selected_tokens if partial_plan and partial_plan.selected_tokens else cfg.default_tokens
        load_ms = self._estimate_load_ms(profile, loaded_bytes)
        recompute_ms = cfg.recompute_overhead_ms + selected_tokens * cfg.recompute_latency_per_token_ms
        action, reason = self._choose_action(profile, load_ms, recompute_ms, partial_plan)
        priority = priority_for(profile, action, cfg.memory_pressure)
        return LoadRecomputeDecision(
            chunk_id=profile.chunk_id,
            workload_id=profile.workload_id,
            request_id=profile.case,
            case=profile.case,
            cache_key=profile.cache_key,
            action=action,
            reuse_frequency=profile.reuse_frequency,
            cache_hit_rate=profile.cache_hit_rate,
            prefetch_hit_rate=profile.prefetch_hit_rate,
            estimated_load_ms=load_ms,
            estimated_recompute_ms=recompute_ms,
            estimated_loaded_bytes=int(loaded_bytes or 0),
            estimated_skipped_bytes=int(skipped_bytes or 0),
            memory_pressure=clamp(cfg.memory_pressure),
            deadline_ms=cfg.deadline_ms,
            priority=priority,
            reason=reason,
            metadata={
                "cache_loads": profile.cache_loads,
                "bytes_loaded_profile": profile.bytes_loaded,
                "avg_load_latency_ms": profile.avg_load_latency_ms,
                "partial_plan_action": partial_plan.action if partial_plan else "",
                "partial_byte_saving_rate": partial_plan.byte_saving_rate if partial_plan else "",
            },
        )

    def decide_db(
        self,
        db: ProfileDB,
        partial_plans: dict[str, PartialPlanStats] | None = None,
    ) -> list[LoadRecomputeDecision]:
        plan_by_chunk = partial_plans or {}
        return sorted(
            (
                self.decide_profile(profile, plan_by_chunk.get(profile.chunk_id))
                for profile in db.chunks.values()
            ),
            key=lambda item: (item.priority, item.chunk_id),
            reverse=True,
        )

    def _estimate_load_ms(self, profile: ChunkProfile, loaded_bytes: int) -> float:
        cfg = self.config
        if profile.avg_load_latency_ms > 0:
            if profile.bytes_loaded > 0 and loaded_bytes > 0:
                ratio = loaded_bytes / max(1, profile.bytes_loaded)
                return max(0.1, profile.avg_load_latency_ms * ratio)
            return profile.avg_load_latency_ms
        if loaded_bytes > 0:
            return loaded_bytes / max(1.0, cfg.io_bandwidth_bytes_per_ms)
        return cfg.load_latency_fallback_ms

    def _choose_action(
        self,
        profile: ChunkProfile,
        load_ms: float,
        recompute_ms: float,
        partial_plan: PartialPlanStats | None,
    ) -> tuple[LoadRecomputeAction, str]:
        cfg = self.config
        memory_pressure = clamp(cfg.memory_pressure)
        if partial_plan is not None and partial_plan.action == "skip":
            return LoadRecomputeAction.DROP, "drop: partial load plan skipped this chunk"
        if (
            profile.reuse_frequency <= cfg.low_reuse_drop_threshold
            and memory_pressure >= cfg.low_reuse_drop_memory_pressure
            and profile.prefetch_hit_rate <= 0.0
        ):
            return LoadRecomputeAction.DROP, "drop: low reuse under memory pressure"
        if load_ms > cfg.deadline_ms and recompute_ms > cfg.deadline_ms * (1.0 + cfg.defer_deadline_slack):
            return LoadRecomputeAction.DEFER, "defer: load and recompute both exceed deadline"
        if recompute_ms * cfg.recompute_penalty < load_ms and memory_pressure >= 0.35:
            return LoadRecomputeAction.RECOMPUTE, "recompute: cheaper than IO under memory pressure"
        if partial_plan is not None and partial_plan.byte_saving_rate >= 0.30 and load_ms <= cfg.deadline_ms:
            return LoadRecomputeAction.LOAD, "load: partial plan keeps IO within deadline"
        if load_ms <= recompute_ms * cfg.load_preference_margin or profile.cache_hit_rate >= 0.5:
            return LoadRecomputeAction.LOAD, "load: profile indicates IO is cheaper or cache-friendly"
        if recompute_ms <= cfg.deadline_ms:
            return LoadRecomputeAction.RECOMPUTE, "recompute: compute estimate fits deadline"
        return LoadRecomputeAction.DEFER, "defer: no safe immediate action"


def priority_for(profile: ChunkProfile, action: LoadRecomputeAction, memory_pressure: float) -> int:
    base = int(round(profile.reuse_frequency * 50 + profile.cache_hit_rate * 20 + profile.prefetch_hit_rate * 20))
    pressure = int(round(clamp(memory_pressure) * 10))
    action_bonus = {
        LoadRecomputeAction.LOAD: 20,
        LoadRecomputeAction.RECOMPUTE: 10,
        LoadRecomputeAction.DEFER: 0,
        LoadRecomputeAction.DROP: -10,
    }[action]
    return max(0, base + pressure + action_bonus)


def partial_plan_stats_from_records(records: Iterable[dict[str, Any]]) -> dict[str, PartialPlanStats]:
    stats: dict[str, PartialPlanStats] = {}
    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        if not chunk_id:
            continue
        stats[chunk_id] = PartialPlanStats(
            chunk_id=chunk_id,
            loaded_tokens=as_int(record.get("loaded_tokens")),
            skipped_tokens=as_int(record.get("skipped_tokens")),
            loaded_bytes=as_int(record.get("loaded_bytes")),
            skipped_bytes=as_int(record.get("skipped_bytes")),
            action=str(record.get("action", "")),
            reason=str(record.get("reason", "")),
        )
    return stats


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def as_int(value: Any) -> int:
    if value in (None, "", "None", "nan", "n/a"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
