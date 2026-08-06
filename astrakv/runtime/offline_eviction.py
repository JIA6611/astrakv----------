"""Deterministic offline replay for logical KV-cache eviction policies.

The simulator is intentionally a control-plane evaluation tool.  It replays
canonical workload order and records modeled tier transitions; it never claims
to have moved a vLLM or LMCache tensor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class OfflinePolicy(str, Enum):
    LRU = "lru"
    FIFO = "fifo"
    ASTRAKV = "astrakv"
    BELADY = "belady_oracle"


TIERS = ("gpu", "cpu", "ssd")


@dataclass(frozen=True, slots=True)
class TierCapacities:
    gpu_bytes: int
    cpu_bytes: int
    ssd_bytes: int

    def __post_init__(self) -> None:
        if min(self.gpu_bytes, self.cpu_bytes, self.ssd_bytes) < 0:
            raise ValueError("tier capacities must be non-negative")

    def for_tier(self, tier: str) -> int:
        return {"gpu": self.gpu_bytes, "cpu": self.cpu_bytes, "ssd": self.ssd_bytes}[tier]

    def to_record(self) -> dict[str, int]:
        return {"gpu_bytes": self.gpu_bytes, "cpu_bytes": self.cpu_bytes, "ssd_bytes": self.ssd_bytes}


@dataclass(frozen=True, slots=True)
class ProxyCostModel:
    cpu_to_gpu_ms: float = 2.0
    ssd_to_gpu_ms: float = 12.0
    recompute_ms: float = 40.0

    def to_record(self) -> dict[str, float]:
        return {
            "cpu_to_gpu_ms": self.cpu_to_gpu_ms,
            "ssd_to_gpu_ms": self.ssd_to_gpu_ms,
            "recompute_ms": self.recompute_ms,
        }


@dataclass(frozen=True, slots=True)
class OfflineObject:
    object_key: str
    object_level: str
    size_bytes: int
    size_source: str
    observed_load_ms: float | None = None
    astrakv_score: float = 0.0


@dataclass(frozen=True, slots=True)
class OfflineAccess:
    request_id: str
    arrival_index: int
    object_key: str
    object_level: str
    size_bytes: int
    size_source: str
    base_ttft_ms: float | None = None
    base_tpot_ms: float | None = None
    observed_load_ms: float | None = None


@dataclass(frozen=True, slots=True)
class PrefetchHint:
    arrival_index: int
    object_key: str
    object_level: str
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class OfflineEvictionEvent:
    policy: str
    run_id: str
    workload_id: str
    request_id: str
    arrival_index: int
    object_key: str
    object_level: str
    event_type: str
    tier_before: str = "unknown"
    tier_after: str = "unknown"
    bytes: int = 0
    latency_proxy_ms: float = 0.0
    provenance: str = "offline_simulation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "request_id": self.request_id,
            "arrival_index": self.arrival_index,
            "object_key": self.object_key,
            "object_level": self.object_level,
            "event_type": self.event_type,
            "tier_before": self.tier_before,
            "tier_after": self.tier_after,
            "bytes": self.bytes,
            "latency_proxy_ms": self.latency_proxy_ms,
            "provenance": self.provenance,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class _ResidentObject:
    spec: OfflineObject
    tier: str | None = None
    inserted_at: int = -1
    last_access: int = -1
    prefetched_at: int | None = None
    prefetch_consumed: bool = False
    evicted_since_access: bool = False


@dataclass(slots=True)
class _Metrics:
    request_count: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    ssd_hits: int = 0
    misses: int = 0
    migration_bytes: int = 0
    ssd_read_bytes: int = 0
    ssd_write_bytes: int = 0
    prefetch_submitted: int = 0
    prefetch_hits: int = 0
    prefetch_waste: int = 0
    evictions: int = 0
    eviction_reaccesses: int = 0
    drops: int = 0
    recomputes: int = 0
    oom_attempted: int = 0
    oom_avoided: int = 0
    oom_unavoided: int = 0
    latency_proxy_ms: float = 0.0
    ttft_proxy_values: list[float] = field(default_factory=list)
    tpot_proxy_values: list[float] = field(default_factory=list)
    proxy_only: bool = True


@dataclass(frozen=True, slots=True)
class OfflinePolicyResult:
    policy: str
    metrics: dict[str, Any]
    events: tuple[OfflineEvictionEvent, ...]


class OfflineEvictionSimulator:
    """Replay one canonical workload under one tier-victim policy."""

    def __init__(
        self,
        *,
        policy: OfflinePolicy,
        capacities: TierCapacities,
        cost_model: ProxyCostModel,
        run_id: str,
        workload_id: str,
        objects: Iterable[OfflineObject],
        accesses: Iterable[OfflineAccess],
        prefetch_hints: Iterable[PrefetchHint] = (),
    ) -> None:
        self.policy = policy
        self.capacities = capacities
        self.cost_model = cost_model
        self.run_id = run_id
        self.workload_id = workload_id
        self.accesses = tuple(sorted(accesses, key=lambda item: item.arrival_index))
        self.objects: dict[str, _ResidentObject] = {
            item.object_key: _ResidentObject(item) for item in objects
        }
        for access in self.accesses:
            self.objects.setdefault(
                access.object_key,
                _ResidentObject(
                    OfflineObject(
                        object_key=access.object_key,
                        object_level=access.object_level,
                        size_bytes=access.size_bytes,
                        size_source=access.size_source,
                        observed_load_ms=access.observed_load_ms,
                    )
                ),
            )
        self.hints_by_index: dict[int, list[PrefetchHint]] = defaultdict(list)
        if policy == OfflinePolicy.ASTRAKV:
            for hint in prefetch_hints:
                self.hints_by_index[hint.arrival_index].append(hint)
        self.used = {tier: 0 for tier in TIERS}
        self.events: list[OfflineEvictionEvent] = []
        self.metrics = _Metrics()
        self.future: dict[str, list[int]] = defaultdict(list)
        for access in self.accesses:
            self.future[access.object_key].append(access.arrival_index)

    def run(self) -> OfflinePolicyResult:
        for access in self.accesses:
            self._consume_future(access)
            self._access(access)
            for hint in self.hints_by_index.get(access.arrival_index, []):
                self._prefetch(hint, access)
        self._finalize_prefetch_waste()
        return OfflinePolicyResult(self.policy.value, self._summary(), tuple(self.events))

    def _access(self, access: OfflineAccess) -> None:
        self.metrics.request_count += 1
        item = self.objects[access.object_key]
        penalty = 0.0
        if item.evicted_since_access:
            self.metrics.eviction_reaccesses += 1
            item.evicted_since_access = False
        if item.tier == "gpu":
            self.metrics.gpu_hits += 1
            self._event(access, "gpu_hit", item, "gpu", "gpu")
        elif item.tier == "cpu":
            self.metrics.cpu_hits += 1
            penalty = self._load_cost(item, "cpu")
            self._event(access, "cpu_hit", item, "cpu", "cpu", latency=penalty)
            if not self._promote(access, item, penalty):
                self.metrics.oom_unavoided += 1
                self._event(access, "oom_unavoided", item, "cpu", "cpu")
        elif item.tier == "ssd":
            self.metrics.ssd_hits += 1
            self.metrics.ssd_read_bytes += item.spec.size_bytes
            penalty = self._load_cost(item, "ssd")
            self._event(access, "ssd_hit", item, "ssd", "ssd", latency=penalty)
            if not self._promote(access, item, penalty):
                self.metrics.oom_unavoided += 1
                self._event(access, "oom_unavoided", item, "ssd", "ssd")
        else:
            self.metrics.misses += 1
            self.metrics.recomputes += 1
            penalty = self._load_cost(item, "recompute")
            self._event(access, "recompute_miss", item, "none", "none", latency=penalty)
            if not self._place_gpu(access, item):
                self.metrics.oom_unavoided += 1
                self._event(access, "oom_unavoided", item, "none", "none")
        item.last_access = access.arrival_index
        if item.prefetched_at is not None and not item.prefetch_consumed and item.tier == "gpu":
            item.prefetch_consumed = True
            self.metrics.prefetch_hits += 1
            self._event(access, "prefetch_hit", item, "gpu", "gpu")
        self.metrics.latency_proxy_ms += penalty
        if access.base_ttft_ms is not None:
            self.metrics.proxy_only = False
            self.metrics.ttft_proxy_values.append(access.base_ttft_ms + penalty)
        else:
            self.metrics.ttft_proxy_values.append(penalty)
        if access.base_tpot_ms is not None:
            self.metrics.proxy_only = False
            self.metrics.tpot_proxy_values.append(access.base_tpot_ms)

    def _prefetch(self, hint: PrefetchHint, source_access: OfflineAccess) -> None:
        item = self.objects.get(hint.object_key)
        if item is None or item.tier == "gpu":
            return
        self.metrics.prefetch_submitted += 1
        item.prefetched_at = source_access.arrival_index
        item.prefetch_consumed = False
        cost = self._load_cost(item, "recompute" if item.tier is None else item.tier)
        self._event(source_access, "prefetch_submit", item, item.tier or "none", "gpu", latency=cost)
        if item.tier is None:
            self.metrics.recomputes += 1
        self._promote(source_access, item, cost, prefetch=True)

    def _promote(self, access: OfflineAccess, item: _ResidentObject, penalty: float, prefetch: bool = False) -> bool:
        before = item.tier or "none"
        self._remove(item)
        if not self._place_gpu(access, item):
            # Preserve a lower-tier copy if promotion cannot be admitted.
            if before in TIERS:
                self._place_direct(item, before, access.arrival_index)
            return False
        self.metrics.migration_bytes += item.spec.size_bytes
        self._event(access, "prefetch_complete" if prefetch else "promote", item, before, "gpu", latency=penalty)
        return True

    def _place_gpu(self, access: OfflineAccess, item: _ResidentObject) -> bool:
        if item.spec.size_bytes > self.capacities.gpu_bytes:
            self.metrics.oom_attempted += 1
            return False
        needed = item.spec.size_bytes
        under_pressure = self.used["gpu"] + needed > self.capacities.gpu_bytes
        if under_pressure:
            self.metrics.oom_attempted += 1
        if not self._ensure_capacity("gpu", needed, access):
            return False
        if self.used["gpu"] + needed > self.capacities.gpu_bytes:
            return False
        if under_pressure:
            self.metrics.oom_avoided += 1
        self._place_direct(item, "gpu", access.arrival_index)
        return True

    def _ensure_capacity(self, tier: str, needed: int, access: OfflineAccess) -> bool:
        capacity = self.capacities.for_tier(tier)
        if needed > capacity:
            return False
        while self.used[tier] + needed > capacity:
            victim = self._victim(tier, access.arrival_index)
            if victim is None:
                return False
            self.metrics.evictions += 1
            victim.evicted_since_access = True
            before = victim.tier or tier
            self._remove(victim)
            next_tier = {"gpu": "cpu", "cpu": "ssd", "ssd": None}[tier]
            if next_tier is None or not self._place_lower(access, victim, next_tier):
                self.metrics.drops += 1
                self._event(access, "drop", victim, before, "none")
            else:
                self._event(access, "evict", victim, before, next_tier)
        return True

    def _place_lower(self, access: OfflineAccess, item: _ResidentObject, tier: str) -> bool:
        if item.spec.size_bytes > self.capacities.for_tier(tier):
            return False
        if not self._ensure_capacity(tier, item.spec.size_bytes, access):
            return False
        self._place_direct(item, tier, access.arrival_index)
        if tier == "ssd":
            self.metrics.ssd_write_bytes += item.spec.size_bytes
        return True

    def _place_direct(self, item: _ResidentObject, tier: str, index: int) -> None:
        item.tier = tier
        item.inserted_at = index
        self.used[tier] += item.spec.size_bytes

    def _remove(self, item: _ResidentObject) -> None:
        if item.tier in TIERS:
            self.used[item.tier] -= item.spec.size_bytes
        item.tier = None

    def _victim(self, tier: str, index: int) -> _ResidentObject | None:
        candidates = [item for item in self.objects.values() if item.tier == tier]
        if not candidates:
            return None
        if self.policy == OfflinePolicy.FIFO:
            return min(candidates, key=lambda item: (item.inserted_at, item.spec.object_key))
        if self.policy == OfflinePolicy.BELADY:
            return max(candidates, key=lambda item: (self._next_use(item.spec.object_key, index), item.spec.object_key))
        if self.policy == OfflinePolicy.ASTRAKV:
            return min(candidates, key=lambda item: (item.spec.astrakv_score, item.last_access, item.spec.object_key))
        return min(candidates, key=lambda item: (item.last_access, item.spec.object_key))

    def _next_use(self, object_key: str, index: int) -> int:
        values = self.future.get(object_key, [])
        for value in values:
            if value > index:
                return value
        return 10**18

    def _consume_future(self, access: OfflineAccess) -> None:
        values = self.future.get(access.object_key)
        if values and values[0] == access.arrival_index:
            values.pop(0)

    def _load_cost(self, item: _ResidentObject, source: str) -> float:
        if source == "cpu":
            return self.cost_model.cpu_to_gpu_ms
        if source == "ssd":
            return item.spec.observed_load_ms if item.spec.observed_load_ms is not None else self.cost_model.ssd_to_gpu_ms
        return item.spec.observed_load_ms if item.spec.observed_load_ms is not None else self.cost_model.recompute_ms

    def _event(
        self, access: OfflineAccess, event_type: str, item: _ResidentObject, before: str, after: str, *, latency: float = 0.0
    ) -> None:
        self.events.append(OfflineEvictionEvent(
            policy=self.policy.value, run_id=self.run_id, workload_id=self.workload_id,
            request_id=access.request_id, arrival_index=access.arrival_index,
            object_key=item.spec.object_key, object_level=item.spec.object_level,
            event_type=event_type, tier_before=before, tier_after=after,
            bytes=item.spec.size_bytes, latency_proxy_ms=latency,
            metadata={"size_source": item.spec.size_source, "cost_is_proxy": True},
        ))

    def _finalize_prefetch_waste(self) -> None:
        for item in self.objects.values():
            if item.prefetched_at is not None and not item.prefetch_consumed:
                self.metrics.prefetch_waste += 1

    def _summary(self) -> dict[str, Any]:
        m = self.metrics
        hits = m.gpu_hits + m.cpu_hits + m.ssd_hits
        request_count = max(1, m.request_count)
        reaccess_denominator = max(1, m.evictions)
        return {
            "policy": self.policy.value,
            "is_offline_oracle": self.policy == OfflinePolicy.BELADY,
            "request_count": m.request_count,
            "gpu_hits": m.gpu_hits,
            "cpu_hits": m.cpu_hits,
            "ssd_hits": m.ssd_hits,
            "misses": m.misses,
            "total_hits": hits,
            "total_hit_rate": hits / request_count,
            "migration_bytes": m.migration_bytes,
            "ssd_read_proxy_bytes": m.ssd_read_bytes,
            "ssd_write_proxy_bytes": m.ssd_write_bytes,
            "prefetch_submitted": m.prefetch_submitted,
            "prefetch_hits": m.prefetch_hits,
            "prefetch_waste": m.prefetch_waste,
            "prefetch_waste_rate": m.prefetch_waste / max(1, m.prefetch_submitted),
            "evictions": m.evictions,
            "eviction_reaccesses": m.eviction_reaccesses,
            "eviction_reaccess_rate": m.eviction_reaccesses / reaccess_denominator,
            "drops": m.drops,
            "recomputes": m.recomputes,
            "oom_attempted": m.oom_attempted,
            "oom_avoided": m.oom_avoided,
            "oom_unavoided": m.oom_unavoided,
            "latency_proxy_ms_total": m.latency_proxy_ms,
            "ttft_proxy_ms_mean": _mean(m.ttft_proxy_values),
            "tpot_proxy_ms_mean": _mean(m.tpot_proxy_values) if m.tpot_proxy_values else None,
            "timing_mode": "benchmark_plus_proxy" if not m.proxy_only else "proxy_only",
            "final_gpu_bytes": self.used["gpu"],
            "final_cpu_bytes": self.used["cpu"],
            "final_ssd_bytes": self.used["ssd"],
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
