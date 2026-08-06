"""Profile-guided KV chunk scorer.

The scorer converts ProfileDB chunk statistics into policy-facing scores and
plain-language actions. It does not submit prefetch requests or move memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from astrakv.runtime.profile_db import ChunkProfile, ProfileDB


class ChunkAction(str, Enum):
    PREFETCH = "prefetch"
    KEEP = "keep"
    OFFLOAD = "offload"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class ChunkScorerConfig:
    reuse_weight: float = 0.35
    deadline_weight: float = 0.20
    load_latency_weight: float = 0.20
    prefetch_history_weight: float = 0.15
    memory_pressure_weight: float = 0.10
    size_penalty_weight: float = 0.15
    waste_penalty_weight: float = 0.20
    prefetch_threshold: float = 0.62
    keep_threshold: float = 0.38
    offload_threshold: float = 0.18
    deadline_ms: float = 80.0
    load_latency_reference_ms: float = 100.0
    size_reference_bytes: int = 16 * 1024 * 1024
    memory_pressure: float = 0.0


@dataclass(frozen=True, slots=True)
class ChunkScore:
    chunk_id: str
    workload_id: str
    case: str
    action: ChunkAction
    score: float
    reuse_score: float
    deadline_score: float
    load_latency_score: float
    prefetch_history_score: float
    memory_pressure_score: float
    size_penalty: float
    waste_penalty: float
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "workload_id": self.workload_id,
            "case": self.case,
            "action": self.action.value,
            "score": self.score,
            "reuse_score": self.reuse_score,
            "deadline_score": self.deadline_score,
            "load_latency_score": self.load_latency_score,
            "prefetch_history_score": self.prefetch_history_score,
            "memory_pressure_score": self.memory_pressure_score,
            "size_penalty": self.size_penalty,
            "waste_penalty": self.waste_penalty,
            "reason": self.reason,
            **dict(self.metrics),
        }


class ChunkScorer:
    def __init__(self, config: ChunkScorerConfig | None = None) -> None:
        self.config = config or ChunkScorerConfig()

    def score_profile(self, profile: ChunkProfile) -> ChunkScore:
        cfg = self.config
        reuse_score = clamp(profile.reuse_frequency)
        load_latency_score = clamp(profile.avg_load_latency_ms / max(1.0, cfg.load_latency_reference_ms))
        deadline_score = clamp(profile.avg_load_latency_ms / max(1.0, cfg.deadline_ms))
        prefetch_history_score = clamp(profile.prefetch_hit_rate)
        memory_pressure_score = clamp(cfg.memory_pressure)
        size_penalty = clamp(profile.bytes_loaded / max(1, cfg.size_reference_bytes))
        waste_penalty = clamp(profile.prefetch_waste / max(1, profile.prefetch_hits + profile.prefetch_waste))

        positive = (
            cfg.reuse_weight * reuse_score
            + cfg.deadline_weight * deadline_score
            + cfg.load_latency_weight * load_latency_score
            + cfg.prefetch_history_weight * prefetch_history_score
        )
        pressure_adjustment = cfg.memory_pressure_weight * memory_pressure_score
        negative = cfg.size_penalty_weight * size_penalty + cfg.waste_penalty_weight * waste_penalty
        score = clamp(positive - negative - pressure_adjustment)
        action = self._choose_action(profile, score, memory_pressure_score)
        return ChunkScore(
            chunk_id=profile.chunk_id,
            workload_id=profile.workload_id,
            case=profile.case,
            action=action,
            score=score,
            reuse_score=reuse_score,
            deadline_score=deadline_score,
            load_latency_score=load_latency_score,
            prefetch_history_score=prefetch_history_score,
            memory_pressure_score=memory_pressure_score,
            size_penalty=size_penalty,
            waste_penalty=waste_penalty,
            reason=explain_action(
                profile=profile,
                action=action,
                reuse_score=reuse_score,
                deadline_score=deadline_score,
                load_latency_score=load_latency_score,
                prefetch_history_score=prefetch_history_score,
                memory_pressure_score=memory_pressure_score,
                size_penalty=size_penalty,
                waste_penalty=waste_penalty,
            ),
            metrics={
                "reuse_frequency": profile.reuse_frequency,
                "cache_hit_rate": profile.cache_hit_rate,
                "prefetch_hit_rate": profile.prefetch_hit_rate,
                "avg_load_latency_ms": profile.avg_load_latency_ms,
                "bytes_loaded": profile.bytes_loaded,
                "cache_loads": profile.cache_loads,
                "prefetch_hits": profile.prefetch_hits,
                "prefetch_waste": profile.prefetch_waste,
                "tier_counts": ";".join(f"{key}:{value}" for key, value in sorted(profile.tier_counts.items())),
            },
        )

    def score_profiles(self, profiles: Iterable[ChunkProfile]) -> list[ChunkScore]:
        return sorted(
            (self.score_profile(profile) for profile in profiles),
            key=lambda item: (item.score, item.chunk_id),
            reverse=True,
        )

    def score_db(self, db: ProfileDB) -> list[ChunkScore]:
        return self.score_profiles(db.chunks.values())

    def _choose_action(
        self,
        profile: ChunkProfile,
        score: float,
        memory_pressure_score: float,
    ) -> ChunkAction:
        cfg = self.config
        if score >= cfg.prefetch_threshold:
            return ChunkAction.PREFETCH
        if score >= cfg.keep_threshold:
            return ChunkAction.KEEP
        if score >= cfg.offload_threshold or (memory_pressure_score > 0.65 and profile.reuse_frequency > 0.0):
            return ChunkAction.OFFLOAD
        return ChunkAction.DROP


def explain_action(
    *,
    profile: ChunkProfile,
    action: ChunkAction,
    reuse_score: float,
    deadline_score: float,
    load_latency_score: float,
    prefetch_history_score: float,
    memory_pressure_score: float,
    size_penalty: float,
    waste_penalty: float,
) -> str:
    signals: list[str] = []
    if reuse_score >= 0.5:
        signals.append("high reuse")
    elif reuse_score <= 0.1:
        signals.append("low reuse")
    if load_latency_score >= 0.5:
        signals.append("expensive load")
    if deadline_score >= 0.5:
        signals.append("deadline-sensitive")
    if prefetch_history_score >= 0.5:
        signals.append("prefetch has worked")
    if waste_penalty >= 0.5:
        signals.append("prefetch waste observed")
    if size_penalty >= 0.5:
        signals.append("large loaded bytes")
    if memory_pressure_score >= 0.5:
        signals.append("memory pressure")
    if not signals:
        signals.append("weak profile evidence")
    return (
        f"{action.value}: "
        f"{', '.join(signals)}; "
        f"reuse={profile.reuse_frequency:.3f}, "
        f"load_ms={profile.avg_load_latency_ms:.3f}, "
        f"prefetch_hit={profile.prefetch_hit_rate:.3f}"
    )


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
