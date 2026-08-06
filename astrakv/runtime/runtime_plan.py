"""Unified runtime action plans for AstraKV-W online control.

The canonical runtime-control path still dispatches public Hook commands, but
it now carries a stable action-plan schema so online policy, profile guards,
partial-load targets, and runtime audit state all speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


RUNTIME_ACTION_PLAN_SCHEMA = "astrakv-runtime-action-plan-v1"
RUNTIME_PROFILE_GUARD_SCHEMA = "astrakv-runtime-profile-guard-v1"


class RuntimeObjectKind(str, Enum):
    KV_OBJECT = "kv_object"
    KV_SEGMENT = "kv_segment"


class RuntimePlacementTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"
    NONE = "none"
    UNKNOWN = "unknown"
    RUNTIME = "runtime"


class RuntimeActionKind(str, Enum):
    LOAD = "load"
    PREFETCH = "prefetch"
    OFFLOAD = "offload"
    EVICT = "evict"
    DROP = "drop"
    RECOMPUTE = "recompute"
    KEEP = "keep"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class RuntimeLayerRange:
    start_layer: int
    end_layer: int

    def __post_init__(self) -> None:
        if self.start_layer < 0:
            raise ValueError("start_layer must be non-negative")
        if self.end_layer < self.start_layer:
            raise ValueError("end_layer must be greater than or equal to start_layer")

    def to_record(self) -> dict[str, int]:
        return {
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "layer_count": self.end_layer - self.start_layer,
        }


@dataclass(frozen=True, slots=True)
class RuntimeTokenRange:
    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token < self.start_token:
            raise ValueError("end_token must be greater than or equal to start_token")

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_record(self) -> dict[str, int]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class RuntimeProfileGuard:
    guard_id: str
    decision_source: str = "heuristic"
    profile_guard: str = "none"
    profile_guard_reason: str = ""
    partial_load_allowed: bool = True
    recompute_allowed: bool = True
    prefetch_priority_boost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_PROFILE_GUARD_SCHEMA,
            "guard_id": self.guard_id,
            "decision_source": self.decision_source,
            "profile_guard": self.profile_guard,
            "profile_guard_reason": self.profile_guard_reason,
            "partial_load_allowed": self.partial_load_allowed,
            "recompute_allowed": self.recompute_allowed,
            "prefetch_priority_boost": self.prefetch_priority_boost,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeActionPlan:
    plan_id: str
    object_id: str
    object_key: str
    object_kind: RuntimeObjectKind
    action: RuntimeActionKind
    request_id: str = ""
    object_level: str = "prefix"
    layer_range: RuntimeLayerRange | None = None
    token_range: RuntimeTokenRange | None = None
    source_tier: RuntimePlacementTier = RuntimePlacementTier.UNKNOWN
    target_tier: RuntimePlacementTier = RuntimePlacementTier.UNKNOWN
    allow_partial: bool = False
    allow_recompute_fallback: bool = False
    priority: int = 0
    decision_source: str = "heuristic"
    fallback_mode: str = "none"
    trigger_reason: str = ""
    profile_guard: RuntimeProfileGuard | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_ACTION_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "object_id": self.object_id,
            "object_key": self.object_key,
            "object_kind": self.object_kind.value,
            "action": self.action.value,
            "request_id": self.request_id,
            "object_level": self.object_level,
            "layer_range": None if self.layer_range is None else self.layer_range.to_record(),
            "token_range": None if self.token_range is None else self.token_range.to_record(),
            "source_tier": self.source_tier.value,
            "target_tier": self.target_tier.value,
            "allow_partial": self.allow_partial,
            "allow_recompute_fallback": self.allow_recompute_fallback,
            "priority": self.priority,
            "decision_source": self.decision_source,
            "fallback_mode": self.fallback_mode,
            "trigger_reason": self.trigger_reason,
            "profile_guard": None if self.profile_guard is None else self.profile_guard.to_record(),
            "metadata": dict(self.metadata),
        }
