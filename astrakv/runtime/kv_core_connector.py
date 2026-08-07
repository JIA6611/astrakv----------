"""Version-locked connector callback adapter for AstraKV-W KV-Core.

The vLLM/LMCache patch calls this adapter at scheduler lookup, allocation,
metadata, and native load completion.  This adapter never receives tensors and
never calls ``engine.retrieve``; the native connector remains the sole paged
KV writer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from astrakv.runtime.kv_runtime_core import (
    NativeKVLoadReceipt,
    PhysicalKVObject,
    PrefetchTicket,
    PrefetchTicketStore,
    RequestKVIntent,
    RuntimeMode,
    TierCapabilitySnapshot,
)


@dataclass(frozen=True, slots=True)
class SchedulerLookupObservation:
    request_id: str
    physical_object_id: str
    binding_generation: int
    lookup_hit_tokens: int
    native_request_id: str


@dataclass(frozen=True, slots=True)
class SchedulerAdmission:
    request_id: str
    physical_object_id: str
    binding_generation: int
    allocated_external_tokens: int
    status: str
    reason: str = ""


class KVCoreConnectorCallbacks:
    """In-memory state required to validate native connector callbacks."""

    def __init__(self, *, mode: RuntimeMode, capability: TierCapabilitySnapshot) -> None:
        self.mode = mode
        self.capability = capability
        self.tickets = PrefetchTicketStore()
        self._intents: dict[str, RequestKVIntent] = {}
        self._lookups: dict[str, SchedulerLookupObservation] = {}
        self._admissions: dict[str, SchedulerAdmission] = {}
        self._receipts: dict[str, NativeKVLoadReceipt] = {}

    def submit_intent(self, intent: RequestKVIntent) -> SchedulerAdmission:
        if self.mode is RuntimeMode.OFF:
            return SchedulerAdmission(
                intent.request_id, intent.physical_object.physical_object_id,
                intent.physical_object.binding_generation, 0, "off", "kv_core_off",
            )
        if intent.request_id in self._intents:
            raise ValueError("request already has a KV-Core intent")
        self._intents[intent.request_id] = intent
        return SchedulerAdmission(
            intent.request_id, intent.physical_object.physical_object_id,
            intent.physical_object.binding_generation, 0, "advisory",
        )

    def begin_cpu_prefetch(self, ticket: PrefetchTicket, physical: PhysicalKVObject, *, now_ns: int | None = None) -> str | None:
        """Validate an SSD->CPU request before the native storage API executes it."""
        if self.mode is not RuntimeMode.ACTIVE:
            return "kv_core_not_active"
        if ticket.generation_key != physical.generation_key or ticket.prefix_hash != physical.compatibility_key.prefix_hash:
            return "physical_generation_mismatch"
        if physical.source_tier != "ssd" or ticket.source_tier != "ssd":
            return "source_tier_not_ssd"
        reason = self.capability.prefetch_block_reason(
            size_bytes=ticket.requested_bytes, deadline_ns=ticket.deadline_ns, now_ns=now_ns,
        )
        if reason is not None:
            return reason
        self.tickets.submit(ticket)
        return None

    def complete_cpu_prefetch(self, prefetch_id: str, *, completed_bytes: int, now_ns: int | None = None) -> PrefetchTicket:
        return self.tickets.complete(prefetch_id, completed_bytes=completed_bytes, now_ns=now_ns)

    def record_scheduler_lookup(
        self, *, request_id: str, physical: PhysicalKVObject, lookup_hit_tokens: int, native_request_id: str,
    ) -> SchedulerLookupObservation:
        intent = self._intent(request_id, physical)
        if lookup_hit_tokens < 0 or lookup_hit_tokens > intent.requested_prefix_tokens:
            raise ValueError("lookup_hit_tokens must be within the requested prefix")
        if not native_request_id:
            raise ValueError("native_request_id is required")
        observed = SchedulerLookupObservation(
            request_id, physical.physical_object_id, physical.binding_generation,
            int(lookup_hit_tokens), native_request_id,
        )
        self._lookups[request_id] = observed
        return observed

    def record_scheduler_admission(self, *, request_id: str, physical: PhysicalKVObject, allocated_external_tokens: int) -> SchedulerAdmission:
        lookup = self._lookup(request_id, physical)
        intent = self._intent(request_id, physical)
        if allocated_external_tokens < 0 or allocated_external_tokens > lookup.lookup_hit_tokens:
            raise ValueError("scheduler allocation must not exceed lookup hit tokens")
        if allocated_external_tokens > intent.max_external_tokens:
            raise ValueError("scheduler allocation exceeds intent external-token upper bound")
        if allocated_external_tokens > self.capability.external_token_cap:
            raise ValueError("scheduler allocation exceeds external token cap")
        status = "accepted" if allocated_external_tokens else "recompute"
        admission = SchedulerAdmission(
            request_id, physical.physical_object_id, physical.binding_generation,
            int(allocated_external_tokens), status,
            "scheduler_allocated" if allocated_external_tokens else "scheduler_declined_external_prefix",
        )
        self._admissions[request_id] = admission
        return admission

    def record_native_load_completion(
        self,
        *,
        request_id: str,
        physical: PhysicalKVObject,
        actual_loaded_tokens: int,
        bytes_loaded: int,
        load_latency_ns: int,
        native_request_id: str,
        status: str,
        prefetch_id: str = "",
        now_ns: int | None = None,
    ) -> NativeKVLoadReceipt:
        intent = self._intent(request_id, physical)
        lookup = self._lookup(request_id, physical)
        admission = self._admissions.get(request_id)
        allocated = 0 if admission is None else admission.allocated_external_tokens
        receipt = NativeKVLoadReceipt(
            request_id=request_id,
            physical_object_id=physical.physical_object_id,
            binding_generation=physical.binding_generation,
            requested_prefix_tokens=intent.requested_prefix_tokens,
            lookup_hit_tokens=lookup.lookup_hit_tokens,
            allocated_external_tokens=allocated,
            actual_loaded_tokens=int(actual_loaded_tokens),
            recomputed_tokens=max(0, intent.requested_prefix_tokens - int(actual_loaded_tokens)),
            bytes_loaded=int(bytes_loaded),
            load_latency_ns=int(load_latency_ns),
            status=status,
            native_request_id=native_request_id,
            prefetch_id=prefetch_id,
        )
        if prefetch_id:
            self.tickets.consume(
                prefetch_id,
                request_id=request_id,
                physical_object_id=physical.physical_object_id,
                binding_generation=physical.binding_generation,
                prefix_hash=physical.compatibility_key.prefix_hash,
                now_ns=now_ns,
            )
        self._receipts[request_id] = receipt
        return receipt

    def receipt_for(self, request_id: str) -> NativeKVLoadReceipt | None:
        return self._receipts.get(request_id)

    def _intent(self, request_id: str, physical: PhysicalKVObject) -> RequestKVIntent:
        intent = self._intents.get(request_id)
        if intent is None:
            raise ValueError("native callback has no request intent")
        if intent.physical_object.generation_key != physical.generation_key:
            raise ValueError("native callback physical generation mismatch")
        if intent.compatibility_key != physical.compatibility_key:
            raise ValueError("native callback compatibility key mismatch")
        return intent

    def _lookup(self, request_id: str, physical: PhysicalKVObject) -> SchedulerLookupObservation:
        lookup = self._lookups.get(request_id)
        if lookup is None:
            raise ValueError("native callback has no scheduler lookup")
        if (lookup.physical_object_id, lookup.binding_generation) != physical.generation_key:
            raise ValueError("scheduler lookup physical generation mismatch")
        return lookup


__all__ = ["KVCoreConnectorCallbacks", "SchedulerAdmission", "SchedulerLookupObservation"]
