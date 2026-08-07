"""Runtime prefix-level online statistics for hybrid prefetch learning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from astrakv.runtime.backend_hook import BackendHookEvent, HookAction


PREFIX_PROFILE_SCHEMA = "astrakv-runtime-prefix-profile-v1"


@dataclass(slots=True)
class RuntimePrefixProfile:
    prefix_key: str
    request_count: int = 0
    reuse_count: int = 0
    observation_count: int = 0
    last_release_or_offload_ns: int = 0
    last_next_submit_ns: int = 0
    inter_arrival_window_ms_ema: float = 0.0
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_hits: int = 0
    prefetch_waste: int = 0
    load_latency_ms_ema: float = 0.0
    last_seen_arrival_index: int = 0
    source_object_ids: set[str] = field(default_factory=set)
    last_request_id: str = ""
    last_runtime_reqmeta_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prefetch_hit_rate(self) -> float:
        return self.prefetch_hits / max(1, self.prefetch_hits + self.prefetch_waste)

    @property
    def runtime_confidence(self) -> float:
        reuse_signal = min(1.0, self.reuse_count / max(1, self.request_count))
        window_signal = 1.0 if self.inter_arrival_window_ms_ema > 0 else 0.0
        # Unknown prefetch benefit is not evidence. Cold-start exploration is
        # represented explicitly by observation/reuse signals instead of an
        # optimistic synthetic hit rate.
        benefit_signal = self.prefetch_hit_rate if (self.prefetch_hits or self.prefetch_waste) else 0.0
        return min(1.0, (0.45 * reuse_signal) + (0.25 * window_signal) + (0.30 * benefit_signal))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PREFIX_PROFILE_SCHEMA,
            "prefix_key": self.prefix_key,
            "request_count": self.request_count,
            "reuse_count": self.reuse_count,
            "observation_count": self.observation_count,
            "last_release_or_offload_ns": self.last_release_or_offload_ns,
            "last_next_submit_ns": self.last_next_submit_ns,
            "inter_arrival_window_ms_ema": self.inter_arrival_window_ms_ema,
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_completed": self.prefetch_completed,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_waste": self.prefetch_waste,
            "load_latency_ms_ema": self.load_latency_ms_ema,
            "last_seen_arrival_index": self.last_seen_arrival_index,
            "source_object_ids": sorted(self.source_object_ids),
            "last_request_id": self.last_request_id,
            "last_runtime_reqmeta_id": self.last_runtime_reqmeta_id,
            "runtime_confidence": self.runtime_confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RuntimePrefixProfile":
        return cls(
            prefix_key=str(record.get("prefix_key") or ""),
            request_count=int(record.get("request_count") or 0),
            reuse_count=int(record.get("reuse_count") or 0),
            observation_count=int(record.get("observation_count") or 0),
            last_release_or_offload_ns=int(record.get("last_release_or_offload_ns") or 0),
            last_next_submit_ns=int(record.get("last_next_submit_ns") or 0),
            inter_arrival_window_ms_ema=float(record.get("inter_arrival_window_ms_ema") or 0.0),
            prefetch_submitted=int(record.get("prefetch_submitted") or 0),
            prefetch_completed=int(record.get("prefetch_completed") or 0),
            prefetch_hits=int(record.get("prefetch_hits") or 0),
            prefetch_waste=int(record.get("prefetch_waste") or 0),
            load_latency_ms_ema=float(record.get("load_latency_ms_ema") or 0.0),
            last_seen_arrival_index=int(record.get("last_seen_arrival_index") or 0),
            source_object_ids=set(record.get("source_object_ids") or ()),
            last_request_id=str(record.get("last_request_id") or ""),
            last_runtime_reqmeta_id=str(record.get("last_runtime_reqmeta_id") or ""),
            metadata=dict(record.get("metadata") or {}),
        )


class RuntimePrefixIndex:
    def __init__(self, records: Iterable[RuntimePrefixProfile] = ()) -> None:
        self._profiles: dict[str, RuntimePrefixProfile] = {item.prefix_key: item for item in records if item.prefix_key}

    def observe(self, event: BackendHookEvent) -> None:
        prefix_key = prefix_key_from_event(event)
        if not prefix_key:
            return
        profile = self._profiles.setdefault(prefix_key, RuntimePrefixProfile(prefix_key=prefix_key))
        profile.observation_count += 1
        profile.last_request_id = event.request_id
        profile.last_runtime_reqmeta_id = str(event.metadata.get("runtime_reqmeta_id") or profile.last_runtime_reqmeta_id)
        profile.last_seen_arrival_index = max(profile.last_seen_arrival_index, _as_int(event.metadata.get("arrival_index")))
        if event.backend_object_id:
            profile.source_object_ids.add(event.backend_object_id)
        profile.metadata.update({
            "prefix_id": str(event.metadata.get("prefix_id") or profile.metadata.get("prefix_id") or ""),
            "cache_key": str(event.metadata.get("cache_key") or profile.metadata.get("cache_key") or ""),
            "prefix_hash": str(event.metadata.get("prefix_hash") or profile.metadata.get("prefix_hash") or ""),
        })

        if event.action is HookAction.CACHE_STORE and event.status == "submitted":
            previous_requests = profile.request_count
            profile.request_count += 1
            if previous_requests > 0:
                profile.reuse_count += 1
            if profile.last_release_or_offload_ns > 0 and event.timestamp_ns > profile.last_release_or_offload_ns:
                profile.last_next_submit_ns = event.timestamp_ns
                window_ms = (event.timestamp_ns - profile.last_release_or_offload_ns) / 1_000_000.0
                profile.inter_arrival_window_ms_ema = _ema(profile.inter_arrival_window_ms_ema, window_ms)
        elif event.action is HookAction.CACHE_HIT and event.status in {"completed", "ok", "executed"}:
            profile.reuse_count += 1
            if _is_causal_prefetch_consumption(event):
                profile.prefetch_hits += 1
        elif event.action is HookAction.CACHE_LOAD and event.status in {"completed", "ok", "executed"}:
            profile.reuse_count += 1
            load_latency_ns = _as_float(event.metadata.get("load_latency_ns"))
            if load_latency_ns > 0:
                profile.load_latency_ms_ema = _ema(profile.load_latency_ms_ema, load_latency_ns / 1_000_000)
            if _is_causal_prefetch_consumption(event):
                profile.prefetch_hits += 1
        elif event.action in {HookAction.PREFETCH, HookAction.PREFETCH_SSD_TO_CPU} and event.status == "submitted":
            profile.prefetch_submitted += 1
        elif event.action in {HookAction.PREFETCH, HookAction.PREFETCH_SSD_TO_CPU} and event.status in {"completed", "ok", "executed"}:
            profile.prefetch_completed += 1
        elif event.action is HookAction.RELEASE and event.status in {"completed", "ok", "executed"}:
            profile.last_release_or_offload_ns = max(profile.last_release_or_offload_ns, event.timestamp_ns)
        elif event.action in {HookAction.OFFLOAD, HookAction.EVICT, HookAction.DROP} and event.status in {"completed", "ok", "executed"}:
            profile.last_release_or_offload_ns = max(profile.last_release_or_offload_ns, event.timestamp_ns)
            if str(event.metadata.get("prefetch_id") or "") and not _is_causal_prefetch_consumption(event):
                profile.prefetch_waste += 1

    def profile_for(self, prefix_key: str) -> RuntimePrefixProfile | None:
        profile = self._profiles.get(prefix_key)
        if profile is None:
            return None
        return RuntimePrefixProfile.from_record(profile.to_record())

    def to_records(self) -> list[dict[str, Any]]:
        return [self._profiles[key].to_record() for key in sorted(self._profiles)]

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "RuntimePrefixIndex":
        return cls(RuntimePrefixProfile.from_record(record) for record in records)


def prefix_key_from_event(event: BackendHookEvent) -> str:
    metadata = dict(event.metadata or {})
    for field in ("prefix_id", "cache_key", "prefix_hash"):
        value = str(metadata.get(field) or "")
        if value:
            return value
    return str(event.object_key or "")


def prefix_key_from_binding(binding: Any, object_state: dict[str, Any]) -> str:
    for source in (binding.metadata, object_state):
        for field in ("prefix_id", "cache_key", "prefix_hash"):
            value = str(source.get(field) or "")
            if value:
                return value
    return str(binding.object_key or "")


def _ema(current: float, value: float, *, alpha: float = 0.2) -> float:
    if value <= 0:
        return current
    return value if current <= 0 else ((1.0 - alpha) * current) + (alpha * value)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_causal_prefetch_consumption(event: BackendHookEvent) -> bool:
    """Require the ticket/generation linkage before attributing a cache hit."""
    metadata = event.metadata or {}
    return bool(metadata.get("prefetch_id")) and bool(metadata.get("prefetch_consumed"))


__all__ = [
    "PREFIX_PROFILE_SCHEMA",
    "RuntimePrefixIndex",
    "RuntimePrefixProfile",
    "prefix_key_from_binding",
    "prefix_key_from_event",
]
