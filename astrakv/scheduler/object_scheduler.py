"""Unified object scheduler MVP.

The scheduler arbitrates passive placement/prefetch/recompute hints for KV
objects under a GPU memory budget. It does not move tensors or modify a serving
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from astrakv.scheduler.decision import LoadRecomputeAction, LoadRecomputeDecision
from astrakv.scheduler.hints import SchedulerHint


class ObjectScheduleAction(str, Enum):
    PREFETCH = "prefetch"
    KEEP = "keep"
    OFFLOAD = "offload"
    DROP = "drop"
    RECOMPUTE = "recompute"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class ObjectSchedulerConfig:
    gpu_budget_bytes: int = 512 * 1024 * 1024
    default_object_bytes: int = 16 * 1024 * 1024
    memory_pressure: float = 0.0
    prefetch_score_bonus: float = 0.20
    keep_score_bonus: float = 0.10
    load_action_bonus: float = 0.15
    recompute_penalty: float = 0.10
    drop_threshold: float = 0.08
    offload_threshold: float = 0.24


@dataclass(frozen=True, slots=True)
class ObjectScheduleCandidate:
    chunk_id: str
    workload_id: str
    case: str = ""
    cache_key: str = ""
    size_bytes: int = 0
    current_tier: str = "unknown"
    chunk_score: float = 0.0
    chunk_action: str = ""
    load_action: str = ""
    load_priority: int = 0
    reuse_frequency: float = 0.0
    cache_hit_rate: float = 0.0
    prefetch_hit_rate: float = 0.0
    load_ms: float = 0.0
    recompute_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObjectScheduleDecision:
    chunk_id: str
    workload_id: str
    action: ObjectScheduleAction
    priority: float
    size_bytes: int
    gpu_budget_bytes: int
    gpu_bytes_after: int
    reason: str
    case: str = ""
    cache_key: str = ""
    source_chunk_action: str = ""
    source_load_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        prefix_id = str(self.metadata.get("prefix_id") or "")
        object_key = prefix_id or self.cache_key or self.chunk_id
        object_level = "prefix" if prefix_id else ("cache_key" if self.cache_key else "prefix")
        return {
            "chunk_id": self.chunk_id,
            "workload_id": self.workload_id,
            "case": self.case,
            "cache_key": self.cache_key,
            "run_id": str(self.metadata.get("run_id") or ""),
            "request_id": str(self.metadata.get("request_id") or self.case or self.chunk_id),
            "object_key": object_key,
            "object_level": object_level,
            "arrival_index": self.metadata.get("arrival_index"),
            "legacy_unlinked": bool(self.metadata.get("legacy_unlinked", True)),
            "action": self.action.value,
            "priority": self.priority,
            "size_bytes": self.size_bytes,
            "gpu_budget_bytes": self.gpu_budget_bytes,
            "gpu_bytes_after": self.gpu_bytes_after,
            "source_chunk_action": self.source_chunk_action,
            "source_load_action": self.source_load_action,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_hint(self) -> SchedulerHint:
        return SchedulerHint(
            request_id=self.case or self.chunk_id,
            action=self.action.value,
            reason=self.reason,
            priority=int(round(self.priority * 100)),
            metadata={
                "chunk_id": self.chunk_id,
                "workload_id": self.workload_id,
                "size_bytes": self.size_bytes,
                "gpu_budget_bytes": self.gpu_budget_bytes,
                "gpu_bytes_after": self.gpu_bytes_after,
                "source_chunk_action": self.source_chunk_action,
                "source_load_action": self.source_load_action,
                "run_id": self.metadata.get("run_id", ""),
                "request_id": self.metadata.get("request_id", self.case or self.chunk_id),
                "prefix_id": self.metadata.get("prefix_id", ""),
                "cache_key": self.cache_key,
                "arrival_index": self.metadata.get("arrival_index"),
                "legacy_unlinked": bool(self.metadata.get("legacy_unlinked", True)),
                **dict(self.metadata),
            },
        )


class UnifiedObjectScheduler:
    def __init__(self, config: ObjectSchedulerConfig | None = None) -> None:
        self.config = config or ObjectSchedulerConfig()

    def schedule(self, candidates: Iterable[ObjectScheduleCandidate]) -> list[ObjectScheduleDecision]:
        ordered = sorted(candidates, key=self._candidate_sort_key, reverse=True)
        gpu_used = 0
        decisions: list[ObjectScheduleDecision] = []
        for candidate in ordered:
            priority = self._priority(candidate)
            size = candidate.size_bytes or self.config.default_object_bytes
            if candidate.load_action == LoadRecomputeAction.DROP.value or candidate.chunk_action == "drop":
                decisions.append(self._decision(candidate, ObjectScheduleAction.DROP, priority, size, gpu_used, "drop requested by source policy"))
                continue
            if candidate.load_action == LoadRecomputeAction.DEFER.value:
                decisions.append(self._decision(candidate, ObjectScheduleAction.DEFER, priority, size, gpu_used, "load-vs-recompute deferred this chunk"))
                continue
            if candidate.load_action == LoadRecomputeAction.RECOMPUTE.value:
                decisions.append(self._decision(candidate, ObjectScheduleAction.RECOMPUTE, priority, size, gpu_used, "recompute is preferred over GPU residency"))
                continue

            wants_gpu = candidate.chunk_action in {"prefetch", "keep"} or candidate.load_action == "load"
            if wants_gpu and gpu_used + size <= self.config.gpu_budget_bytes:
                action = ObjectScheduleAction.PREFETCH if candidate.chunk_action == "prefetch" else ObjectScheduleAction.KEEP
                gpu_used += size
                reason = "scheduled within GPU budget"
                decisions.append(self._decision(candidate, action, priority, size, gpu_used, reason))
            elif priority <= self.config.drop_threshold:
                decisions.append(self._decision(candidate, ObjectScheduleAction.DROP, priority, size, gpu_used, "priority below drop threshold"))
            else:
                decisions.append(self._decision(candidate, ObjectScheduleAction.OFFLOAD, priority, size, gpu_used, "GPU budget exceeded; keep lower-tier copy"))
        return decisions

    def _candidate_sort_key(self, candidate: ObjectScheduleCandidate) -> tuple[float, int, str]:
        return (self._priority(candidate), candidate.load_priority, candidate.chunk_id)

    def _priority(self, candidate: ObjectScheduleCandidate) -> float:
        cfg = self.config
        priority = (
            candidate.chunk_score
            + candidate.reuse_frequency * 0.35
            + candidate.cache_hit_rate * 0.10
            + candidate.prefetch_hit_rate * 0.15
            + min(1.0, candidate.load_ms / 100.0) * 0.10
        )
        if candidate.chunk_action == "prefetch":
            priority += cfg.prefetch_score_bonus
        elif candidate.chunk_action == "keep":
            priority += cfg.keep_score_bonus
        elif candidate.chunk_action == "offload":
            priority -= 0.05
        if candidate.load_action == "load":
            priority += cfg.load_action_bonus
        elif candidate.load_action == "recompute":
            priority -= cfg.recompute_penalty
        priority -= clamp(cfg.memory_pressure) * 0.05
        return clamp(priority)

    def _decision(
        self,
        candidate: ObjectScheduleCandidate,
        action: ObjectScheduleAction,
        priority: float,
        size: int,
        gpu_used: int,
        reason: str,
    ) -> ObjectScheduleDecision:
        return ObjectScheduleDecision(
            chunk_id=candidate.chunk_id,
            workload_id=candidate.workload_id,
            action=action,
            priority=priority,
            size_bytes=size,
            gpu_budget_bytes=self.config.gpu_budget_bytes,
            gpu_bytes_after=gpu_used,
            reason=reason,
            case=candidate.case,
            cache_key=candidate.cache_key,
            source_chunk_action=candidate.chunk_action,
            source_load_action=candidate.load_action,
            metadata={
                "current_tier": candidate.current_tier,
                "load_ms": candidate.load_ms,
                "recompute_ms": candidate.recompute_ms,
                **dict(candidate.metadata),
            },
        )


def candidates_from_profile_db(
    db: ProfileDB,
    *,
    chunk_scores: dict[str, dict[str, Any]] | None = None,
    load_decisions: dict[str, LoadRecomputeDecision] | None = None,
    default_size_bytes: int = 16 * 1024 * 1024,
) -> list[ObjectScheduleCandidate]:
    score_by_chunk = chunk_scores or {}
    load_by_chunk = load_decisions or {}
    candidates: list[ObjectScheduleCandidate] = []
    for profile in db.chunks.values():
        score = score_by_chunk.get(profile.chunk_id, {})
        load = load_by_chunk.get(profile.chunk_id)
        size = profile.bytes_loaded or as_int(score.get("bytes_loaded")) or default_size_bytes
        candidates.append(
            ObjectScheduleCandidate(
                chunk_id=profile.chunk_id,
                workload_id=profile.workload_id,
                case=profile.case,
                cache_key=profile.cache_key,
                size_bytes=size,
                current_tier=infer_current_tier(profile),
                chunk_score=as_float(score.get("score")),
                chunk_action=str(score.get("action", "")),
                load_action=load.action.value if load is not None else "",
                load_priority=load.priority if load is not None else 0,
                reuse_frequency=profile.reuse_frequency,
                cache_hit_rate=profile.cache_hit_rate,
                prefetch_hit_rate=profile.prefetch_hit_rate,
                load_ms=load.estimated_load_ms if load is not None else profile.avg_load_latency_ms,
                recompute_ms=load.estimated_recompute_ms if load is not None else 0.0,
                metadata={
                    "cache_loads": profile.cache_loads,
                    "offloads": profile.offloads,
                    "prefetch_hits": profile.prefetch_hits,
                    "prefetch_waste": profile.prefetch_waste,
                    "run_id": profile.run_id,
                    "request_id": profile.request_id,
                    "prefix_id": profile.prefix_id,
                    "arrival_index": profile.arrival_index,
                    "reuse_ratio": profile.reuse_ratio,
                    "reuse_bucket": profile.reuse_bucket,
                    "legacy_unlinked": profile.legacy_unlinked,
                },
            )
        )
    return candidates


def load_decision_from_record(record: dict[str, Any]) -> LoadRecomputeDecision:
    return LoadRecomputeDecision(
        chunk_id=str(record.get("chunk_id", "")),
        workload_id=str(record.get("workload_id", "")),
        action=LoadRecomputeAction(str(record.get("action", "defer"))),
        request_id=str(record.get("request_id", "")),
        case=str(record.get("case", "")),
        cache_key=str(record.get("cache_key", "")),
        reuse_frequency=as_float(record.get("reuse_frequency")),
        cache_hit_rate=as_float(record.get("cache_hit_rate")),
        prefetch_hit_rate=as_float(record.get("prefetch_hit_rate")),
        estimated_load_ms=as_float(record.get("estimated_load_ms")),
        estimated_recompute_ms=as_float(record.get("estimated_recompute_ms")),
        estimated_loaded_bytes=as_int(record.get("estimated_loaded_bytes")),
        estimated_skipped_bytes=as_int(record.get("estimated_skipped_bytes")),
        memory_pressure=as_float(record.get("memory_pressure")),
        deadline_ms=as_float(record.get("deadline_ms")),
        priority=as_int(record.get("priority")),
        reason=str(record.get("reason", "")),
    )


def infer_current_tier(profile: ChunkProfile) -> str:
    if not profile.tier_counts:
        return "unknown"
    return max(profile.tier_counts.items(), key=lambda item: item[1])[0]


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def as_float(value: Any) -> float:
    if value in (None, "", "None", "nan", "n/a"):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any) -> int:
    if value in (None, "", "None", "nan", "n/a"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
