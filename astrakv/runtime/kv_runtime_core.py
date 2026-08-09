"""Request-owned contracts for the AstraKV-W KV-Core runtime.

This module deliberately contains no vLLM or LMCache imports.  It defines the
identity, admission, tier, prefetch, and receipt contracts that the
version-locked connector integration must honor.  In particular, it never
models a background GPU KV write: a GPU load receipt can only describe work
performed by the native request-owned connector.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


KV_CORE_SCHEMA = "astrakv-kv-core-v1"


class RuntimeMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ACTIVE = "active"


class TierTopology(str, Enum):
    GPU_SSD = "gpu_ssd"
    GPU_CPU_SSD = "gpu_cpu_ssd"


class PrefetchStatus(str, Enum):
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    WASTED = "wasted"
    FAILED = "failed"


def _required(value: str, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _non_negative(value: int | float, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def exact_token_prefix_hash(token_ids: Iterable[int]) -> str:
    """Return a stable hash over token IDs, never prompt text or a Python repr."""
    normalized: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool):
            raise ValueError("token IDs must be integers")
        try:
            normalized.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("token IDs must be integers") from exc
    if not normalized:
        raise ValueError("token prefix must not be empty")
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def normalize_kv_dtype(value: Any) -> str:
    """Normalize torch/vLLM dtype spellings used in compatibility records."""
    text = str(value or "").strip().lower()
    if text.startswith("torch."):
        text = text[6:]
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    return aliases.get(text, text)


@dataclass(frozen=True, slots=True)
class KVCompatibilityKey:
    """All immutable inputs that make an exact KV prefix reusable."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_revision: str
    dtype: str
    rope_config: str
    adapter_namespace: str
    kv_layout: str
    block_size_tokens: int
    chunk_size_tokens: int
    layer_group: str
    prefix_hash: str
    # Deprecated request-local fields.  They remain readable for old artifact
    # consumers but are deliberately excluded from compatibility identity.
    engine_id: str = ""
    worker_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_kv_dtype(self.dtype))
        for name in (
            "model_id", "model_revision", "tokenizer_revision", "chat_template_revision", "dtype",
            "rope_config", "adapter_namespace", "kv_layout", "layer_group", "prefix_hash",
        ):
            _required(getattr(self, name), name)
        if self.block_size_tokens <= 0 or self.chunk_size_tokens <= 0:
            raise ValueError("block_size_tokens and chunk_size_tokens must be positive")
        if self.chunk_size_tokens % self.block_size_tokens:
            raise ValueError("chunk_size_tokens must be block aligned")

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.to_record(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": KV_CORE_SCHEMA,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_revision": self.chat_template_revision,
            "dtype": self.dtype,
            "rope_config": self.rope_config,
            "adapter_namespace": self.adapter_namespace,
            "kv_layout": self.kv_layout,
            "block_size_tokens": self.block_size_tokens,
            "chunk_size_tokens": self.chunk_size_tokens,
            "layer_group": self.layer_group,
            "prefix_hash": self.prefix_hash,
        }


@dataclass(frozen=True, slots=True)
class RequestKVBinding:
    """Request-local ownership for one immutable physical KV generation."""

    request_id: str
    physical_object_id: str
    binding_generation: int
    engine_id: str
    worker_id: str
    native_request_id: str
    cancellation_token: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id", "physical_object_id", "engine_id", "worker_id",
            "native_request_id",
        ):
            _required(getattr(self, name), name)
        if self.binding_generation <= 0:
            raise ValueError("binding_generation must be positive")

    @property
    def generation_key(self) -> tuple[str, int]:
        return self.physical_object_id, self.binding_generation


@dataclass(frozen=True, slots=True)
class PhysicalKVObject:
    native_key: str
    physical_object_id: str
    binding_generation: int
    compatibility_key: KVCompatibilityKey
    source_tier: str = "unknown"
    size_bytes: int = 0

    def __post_init__(self) -> None:
        _required(self.native_key, "native_key")
        _required(self.physical_object_id, "physical_object_id")
        if self.binding_generation <= 0:
            raise ValueError("binding_generation must be positive")
        _non_negative(self.size_bytes, "size_bytes")

    @property
    def generation_key(self) -> tuple[str, int]:
        return self.physical_object_id, self.binding_generation


@dataclass(frozen=True, slots=True)
class TierCapabilitySnapshot:
    topology: TierTopology
    local_cpu_enabled: bool
    local_disk_enabled: bool
    cpu_capacity_bytes: int = 0
    cpu_used_bytes: int = 0
    ssd_capacity_bytes: int = 0
    ssd_used_bytes: int = 0
    available_kv_blocks: int = 0
    external_token_cap: int = 0
    uma_available_bytes: int = 0
    memory_pressure: float = 0.0
    queue_depth: int = 0
    cpu_prefetch_budget_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "cpu_capacity_bytes", "cpu_used_bytes", "ssd_capacity_bytes", "ssd_used_bytes",
            "available_kv_blocks", "external_token_cap", "uma_available_bytes", "queue_depth",
        ):
            _non_negative(getattr(self, name), name)
        if not 0.0 <= float(self.memory_pressure) <= 1.0:
            raise ValueError("memory_pressure must be in [0, 1]")
        if not 0.0 < float(self.cpu_prefetch_budget_fraction) <= 1.0:
            raise ValueError("cpu_prefetch_budget_fraction must be in (0, 1]")
        if self.topology is TierTopology.GPU_SSD and self.local_cpu_enabled:
            raise ValueError("gpu_ssd topology cannot advertise a managed CPU tier")

    @property
    def cpu_prefetch_budget_bytes(self) -> int:
        return int(self.uma_available_bytes * self.cpu_prefetch_budget_fraction)

    @property
    def cpu_prefetch_used_budget_bytes(self) -> int:
        return max(0, self.cpu_used_bytes)

    def prefetch_block_reason(
        self,
        *,
        size_bytes: int,
        deadline_ns: int,
        now_ns: int | None = None,
        in_flight_reserved_bytes: int = 0,
    ) -> str | None:
        current = time.time_ns() if now_ns is None else int(now_ns)
        reserved = _non_negative(in_flight_reserved_bytes, "in_flight_reserved_bytes")
        if not self.local_cpu_enabled or self.topology is not TierTopology.GPU_CPU_SSD:
            return "cpu_tier_unavailable"
        if not self.local_disk_enabled:
            return "ssd_tier_unavailable"
        if deadline_ns <= current:
            return "deadline_expired"
        if self.memory_pressure >= 0.90:
            return "uma_memory_pressure"
        if self.cpu_prefetch_used_budget_bytes + reserved + size_bytes > self.cpu_prefetch_budget_bytes:
            return "cpu_prefetch_budget"
        if self.cpu_used_bytes + reserved + size_bytes > self.cpu_capacity_bytes:
            return "cpu_capacity"
        return None


@dataclass(frozen=True, slots=True)
class RequestKVIntent:
    request_id: str
    compatibility_key: KVCompatibilityKey
    physical_object: PhysicalKVObject
    max_external_tokens: int
    requested_prefix_tokens: int
    deadline_ns: int
    priority: int = 0
    cancellation_token: str = ""

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _non_negative(self.max_external_tokens, "max_external_tokens")
        _non_negative(self.requested_prefix_tokens, "requested_prefix_tokens")
        if self.max_external_tokens > self.requested_prefix_tokens:
            raise ValueError("max_external_tokens cannot exceed requested_prefix_tokens")
        if self.deadline_ns <= 0:
            raise ValueError("deadline_ns must be positive")
        if self.physical_object.compatibility_key != self.compatibility_key:
            raise ValueError("physical object compatibility key does not match request intent")


@dataclass(frozen=True, slots=True)
class PrefetchTicket:
    prefetch_id: str
    physical_object_id: str
    binding_generation: int
    prefix_hash: str
    source_tier: str
    target_tier: str
    requested_bytes: int
    deadline_ns: int
    expires_at_ns: int
    target_request_id: str = ""
    native_key: str = ""
    compatibility_identity: str = ""
    status: PrefetchStatus = PrefetchStatus.SUBMITTED
    completed_bytes: int = 0
    consumer_request_id: str = ""
    failure_reason: str = ""

    def __post_init__(self) -> None:
        for name in ("prefetch_id", "physical_object_id", "prefix_hash", "source_tier", "target_tier"):
            _required(getattr(self, name), name)
        if self.binding_generation <= 0:
            raise ValueError("binding_generation must be positive")
        _non_negative(self.requested_bytes, "requested_bytes")
        _non_negative(self.completed_bytes, "completed_bytes")
        if self.completed_bytes > self.requested_bytes:
            raise ValueError("completed_bytes cannot exceed requested_bytes")
        if self.deadline_ns <= 0 or self.expires_at_ns < self.deadline_ns:
            raise ValueError("ticket expiry must not precede its deadline")
        if self.target_tier != "cpu":
            raise ValueError("KV-Core prefetch tickets may only target the CPU tier")
        _required(self.native_key, "native_key")
        _required(self.compatibility_identity, "compatibility_identity")

    @property
    def generation_key(self) -> tuple[str, int]:
        return self.physical_object_id, self.binding_generation

    def expired(self, *, now_ns: int | None = None) -> bool:
        return (time.time_ns() if now_ns is None else int(now_ns)) >= self.expires_at_ns


class PrefetchTicketStore:
    """Generation-safe causal attribution for CPU prefetches."""

    def __init__(self) -> None:
        self._tickets: dict[str, PrefetchTicket] = {}
        self._lock = threading.RLock()

    def submit(self, ticket: PrefetchTicket) -> PrefetchTicket:
        with self._lock:
            if ticket.prefetch_id in self._tickets:
                raise ValueError("prefetch_id already exists")
            self._tickets[ticket.prefetch_id] = ticket
            return ticket

    def complete(self, prefetch_id: str, *, completed_bytes: int, now_ns: int | None = None) -> PrefetchTicket:
        with self._lock:
            ticket = self._require_open(prefetch_id, now_ns=now_ns)
            if completed_bytes != ticket.requested_bytes:
                return self._replace(ticket, PrefetchStatus.FAILED, completed_bytes=completed_bytes, failure_reason="partial_cpu_prefetch")
            return self._replace(ticket, PrefetchStatus.COMPLETED, completed_bytes=completed_bytes)

    def cancel(self, prefetch_id: str, *, reason: str, now_ns: int | None = None) -> PrefetchTicket:
        with self._lock:
            ticket = self._require_open(prefetch_id, now_ns=now_ns)
            return self._replace(ticket, PrefetchStatus.CANCELLED, failure_reason=_required(reason, "reason"))

    def mark_wasted(self, prefetch_id: str, *, reason: str, now_ns: int | None = None) -> PrefetchTicket:
        with self._lock:
            ticket = self._require_open(prefetch_id, now_ns=now_ns, allow_completed=True)
            return self._replace(
                ticket,
                PrefetchStatus.WASTED,
                failure_reason=_required(reason, "reason"),
            )

    def mark_wasted(self, prefetch_id: str, *, reason: str, now_ns: int | None = None) -> PrefetchTicket:
        with self._lock:
            ticket = self._require_open(prefetch_id, now_ns=now_ns, allow_completed=True)
            return self._replace(
                ticket,
                PrefetchStatus.WASTED,
                failure_reason=_required(reason, "reason"),
            )

    def consume(
        self,
        prefetch_id: str,
        *,
        request_id: str,
        physical_object_id: str,
        binding_generation: int,
        prefix_hash: str,
        native_key: str,
        compatibility_identity: str,
        now_ns: int | None = None,
    ) -> PrefetchTicket:
        with self._lock:
            ticket = self._require_open(prefetch_id, now_ns=now_ns, allow_completed=True)
            if ticket.status is not PrefetchStatus.COMPLETED:
                raise ValueError("prefetch ticket is not completed")
            if ticket.generation_key != (physical_object_id, binding_generation) or ticket.prefix_hash != prefix_hash:
                raise ValueError("prefetch ticket does not match physical object generation")
            if ticket.native_key != native_key or ticket.compatibility_identity != compatibility_identity:
                raise ValueError("prefetch ticket compatibility identity mismatch")
            if ticket.target_request_id and ticket.target_request_id != request_id:
                raise ValueError("prefetch ticket target request mismatch")
            return self._replace(ticket, PrefetchStatus.CONSUMED, consumer_request_id=_required(request_id, "request_id"))

    def expire(self, *, now_ns: int | None = None) -> tuple[PrefetchTicket, ...]:
        current = time.time_ns() if now_ns is None else int(now_ns)
        expired: list[PrefetchTicket] = []
        with self._lock:
            for ticket in tuple(self._tickets.values()):
                if ticket.status in {PrefetchStatus.SUBMITTED, PrefetchStatus.COMPLETED} and ticket.expired(now_ns=current):
                    terminal = self._replace(ticket, PrefetchStatus.EXPIRED, failure_reason="ticket_ttl_expired")
                    expired.append(terminal)
            return tuple(expired)

    def get(self, prefetch_id: str) -> PrefetchTicket | None:
        with self._lock:
            return self._tickets.get(prefetch_id)

    def snapshot(self, *, statuses: Iterable[PrefetchStatus] | None = None) -> tuple[PrefetchTicket, ...]:
        """Return an immutable ticket snapshot without exposing store internals."""
        allowed = None if statuses is None else frozenset(statuses)
        with self._lock:
            return tuple(
                ticket for ticket in self._tickets.values()
                if allowed is None or ticket.status in allowed
            )

    def in_flight_reserved_bytes(self) -> int:
        """Return bytes reserved only by asynchronous SSD->CPU promotions.

        A submitted ticket reserves capacity before LocalCPUBackend reports the
        promoted object as resident.  Once a promotion completes, its bytes
        must be charged to the backend's observed occupancy instead; retaining
        it here would double-count a consumed CPU-hot copy indefinitely.
        """
        with self._lock:
            return sum(
                ticket.requested_bytes
                for ticket in self._tickets.values()
                if ticket.status is PrefetchStatus.SUBMITTED
            )

    def _require_open(self, prefetch_id: str, *, now_ns: int | None, allow_completed: bool = False) -> PrefetchTicket:
        ticket = self._tickets.get(prefetch_id)
        if ticket is None:
            raise ValueError("unknown prefetch ticket")
        if ticket.expired(now_ns=now_ns):
            self._replace(ticket, PrefetchStatus.EXPIRED, failure_reason="ticket_ttl_expired")
            raise ValueError("prefetch ticket expired")
        allowed = {PrefetchStatus.SUBMITTED}
        if allow_completed:
            allowed.add(PrefetchStatus.COMPLETED)
        if ticket.status not in allowed:
            raise ValueError("prefetch ticket is terminal")
        return ticket

    def _replace(self, ticket: PrefetchTicket, status: PrefetchStatus, **changes: Any) -> PrefetchTicket:
        updated = replace(ticket, status=status, **changes)
        self._tickets[ticket.prefetch_id] = updated
        return updated


@dataclass(frozen=True, slots=True)
class NativeKVLoadReceipt:
    """A receipt emitted only after native request-owned connector work."""

    request_id: str
    physical_object_id: str
    binding_generation: int
    native_key: str
    compatibility_identity: str
    prefix_hash: str
    requested_prefix_tokens: int
    locally_cached_tokens: int
    lookup_hit_tokens: int
    allocated_external_tokens: int
    actual_loaded_tokens: int
    native_retrieved_tokens: int
    missing_tokens: int
    unallocated_recompute_tokens: int
    load_shortfall_tokens: int
    bytes_loaded: int
    load_latency_ns: int
    status: str
    native_request_id: str
    prefetch_id: str = ""

    @property
    def generation_key(self) -> tuple[str, int]:
        return self.physical_object_id, self.binding_generation

    def __post_init__(self) -> None:
        for name in (
            "request_id", "physical_object_id", "native_key", "compatibility_identity",
            "prefix_hash", "status", "native_request_id",
        ):
            _required(getattr(self, name), name)
        if self.binding_generation <= 0:
            raise ValueError("binding_generation must be positive")
        for name in (
            "requested_prefix_tokens", "locally_cached_tokens", "lookup_hit_tokens", "allocated_external_tokens", "actual_loaded_tokens",
            "missing_tokens", "unallocated_recompute_tokens", "load_shortfall_tokens",
            "native_retrieved_tokens", "bytes_loaded", "load_latency_ns",
        ):
            _non_negative(getattr(self, name), name)
        if not self.lookup_hit_tokens >= self.allocated_external_tokens >= self.actual_loaded_tokens:
            raise ValueError("lookup_hit_tokens >= allocated_external_tokens >= actual_loaded_tokens is required")
        if self.locally_cached_tokens + self.actual_loaded_tokens > self.requested_prefix_tokens:
            raise ValueError("local and external KV tokens exceed requested prefix")
        expected_missing = max(
            0,
            self.requested_prefix_tokens - self.locally_cached_tokens - self.actual_loaded_tokens,
        )
        if self.missing_tokens != expected_missing:
            raise ValueError("missing_tokens must cover every requested token not actually loaded")
        if self.unallocated_recompute_tokens != max(
            0,
            self.requested_prefix_tokens
            - self.locally_cached_tokens
            - self.allocated_external_tokens,
        ):
            raise ValueError("unallocated_recompute_tokens must cover the scheduler-declined suffix")
        if self.load_shortfall_tokens != self.allocated_external_tokens - self.actual_loaded_tokens:
            raise ValueError("load_shortfall_tokens must cover scheduler-credited KV not retrieved")
        if self.missing_tokens != self.unallocated_recompute_tokens + self.load_shortfall_tokens:
            raise ValueError("missing token classes must close exactly")
        if self.actual_loaded_tokens == 0 and self.bytes_loaded != 0:
            raise ValueError("bytes_loaded requires actual_loaded_tokens")
        if self.native_retrieved_tokens < self.actual_loaded_tokens:
            raise ValueError("native_retrieved_tokens cannot be less than actual_loaded_tokens")


@dataclass(frozen=True, slots=True)
class RequestKVAccounting:
    """Terminal request evidence closed only by the native request lifecycle."""

    request_id: str
    physical_object_id: str
    binding_generation: int
    native_key: str
    compatibility_identity: str
    prefix_hash: str
    requested_prefix_tokens: int
    locally_cached_tokens: int
    lookup_hit_tokens: int
    allocated_external_tokens: int
    actual_loaded_tokens: int
    missing_tokens: int
    unallocated_recompute_tokens: int
    load_shortfall_tokens: int
    scheduled_prefill_tokens: int
    recomputed_tokens: int
    recompute_confirmed: bool
    finish_status: str
    terminal_reason: str

    def __post_init__(self) -> None:
        for name in (
            "request_id", "physical_object_id", "native_key", "compatibility_identity",
            "prefix_hash", "finish_status", "terminal_reason",
        ):
            _required(getattr(self, name), name)
        if self.binding_generation <= 0:
            raise ValueError("binding_generation must be positive")
        for name in (
            "requested_prefix_tokens", "locally_cached_tokens", "lookup_hit_tokens", "allocated_external_tokens",
            "actual_loaded_tokens", "missing_tokens", "unallocated_recompute_tokens",
            "load_shortfall_tokens", "scheduled_prefill_tokens",
            "recomputed_tokens",
        ):
            _non_negative(getattr(self, name), name)
        if not self.lookup_hit_tokens >= self.allocated_external_tokens >= self.actual_loaded_tokens:
            raise ValueError("lookup_hit_tokens >= allocated_external_tokens >= actual_loaded_tokens is required")
        if self.locally_cached_tokens + self.actual_loaded_tokens > self.requested_prefix_tokens:
            raise ValueError("local and external KV tokens exceed requested prefix")
        if self.missing_tokens != max(
            0,
            self.requested_prefix_tokens - self.locally_cached_tokens - self.actual_loaded_tokens,
        ):
            raise ValueError("missing_tokens does not close the requested prefix")
        if self.unallocated_recompute_tokens != max(
            0,
            self.requested_prefix_tokens
            - self.locally_cached_tokens
            - self.allocated_external_tokens,
        ):
            raise ValueError("unallocated recompute suffix does not close")
        if self.load_shortfall_tokens != self.allocated_external_tokens - self.actual_loaded_tokens:
            raise ValueError("load shortfall does not close")
        if self.missing_tokens != self.unallocated_recompute_tokens + self.load_shortfall_tokens:
            raise ValueError("missing token classes must close exactly")
        if self.recompute_confirmed and self.load_shortfall_tokens:
            raise ValueError("native load shortfall cannot be reported as recomputed")
        if self.recompute_confirmed and self.recomputed_tokens != self.unallocated_recompute_tokens:
            raise ValueError("confirmed recompute must cover the scheduler-declined suffix")
        if not self.recompute_confirmed and self.recomputed_tokens != 0:
            raise ValueError("unconfirmed recompute cannot report recomputed tokens")


class NativeKVObjectRegistry:
    """Process-owned generation authority for native LMCache objects."""

    def __init__(self) -> None:
        self._generations: dict[str, int] = {}
        self._lock = threading.RLock()

    def generation(self, physical_object_id: str) -> int:
        object_id = _required(physical_object_id, "physical_object_id")
        with self._lock:
            return self._generations.setdefault(object_id, 1)

    def invalidate(self, physical_object_id: str) -> int:
        object_id = _required(physical_object_id, "physical_object_id")
        with self._lock:
            generation = self._generations.get(object_id, 1) + 1
            self._generations[object_id] = generation
            return generation

    def is_current(self, physical_object_id: str, binding_generation: int) -> bool:
        with self._lock:
            return self._generations.get(physical_object_id, 1) == binding_generation


@dataclass(frozen=True, slots=True)
class LoadVsRecomputeDecision:
    action: str
    load_cost_ms: float
    recompute_cost_ms: float
    reason: str


def choose_load_vs_recompute(
    *,
    intent: RequestKVIntent,
    capability: TierCapabilitySnapshot,
    queue_delay_ms: float,
    tier_read_ms: float,
    transfer_ms: float,
    materialization_ms: float,
    contention_ms: float,
    prefill_ms_per_token: float,
    now_ns: int | None = None,
) -> LoadVsRecomputeDecision:
    """Make a conservative admission decision; scheduler still has final authority."""
    current = time.time_ns() if now_ns is None else int(now_ns)
    if intent.deadline_ns <= current:
        return LoadVsRecomputeDecision("recompute", 0.0, 0.0, "deadline_expired")
    if capability.available_kv_blocks <= 0 or capability.external_token_cap <= 0:
        return LoadVsRecomputeDecision("recompute", 0.0, 0.0, "no_external_kv_capacity")
    if capability.memory_pressure >= 0.90:
        return LoadVsRecomputeDecision("recompute", 0.0, 0.0, "uma_memory_pressure")
    load_cost = max(0.0, queue_delay_ms) + max(0.0, tier_read_ms) + max(0.0, transfer_ms) + max(0.0, materialization_ms) + max(0.0, contention_ms)
    # Only the candidate external prefix differs between the two actions; the
    # suffix is recomputed in both cases and must not bias admission toward I/O.
    recompute_cost = max(0.0, prefill_ms_per_token) * intent.max_external_tokens + max(0.0, contention_ms)
    deadline_ms = (intent.deadline_ns - current) / 1_000_000.0
    if load_cost > deadline_ms:
        return LoadVsRecomputeDecision("recompute", load_cost, recompute_cost, "load_deadline_miss")
    if load_cost >= recompute_cost:
        return LoadVsRecomputeDecision("recompute", load_cost, recompute_cost, "recompute_cheaper")
    return LoadVsRecomputeDecision("admit_external_prefix", load_cost, recompute_cost, "native_load_cheaper")


__all__ = [
    "KV_CORE_SCHEMA", "KVCompatibilityKey", "LoadVsRecomputeDecision", "NativeKVLoadReceipt",
    "NativeKVObjectRegistry", "PhysicalKVObject", "PrefetchStatus", "PrefetchTicket",
    "PrefetchTicketStore", "RequestKVAccounting", "RequestKVBinding", "RequestKVIntent",
    "RuntimeMode", "TierCapabilitySnapshot", "TierTopology", "choose_load_vs_recompute", "exact_token_prefix_hash",
    "normalize_kv_dtype",
]
