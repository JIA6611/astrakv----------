"""Version-locked connector callback adapter for AstraKV-W KV-Core.

The vLLM/LMCache patch calls this adapter at scheduler lookup, allocation,
metadata, and native load completion.  This adapter never receives tensors and
never calls ``engine.retrieve``; the native connector remains the sole paged
KV writer.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass

from astrakv.runtime.kv_runtime_core import (
    NativeKVLoadReceipt,
    PhysicalKVObject,
    PrefetchTicket,
    PrefetchTicketStore,
    RequestKVIntent,
    RequestKVAccounting,
    RuntimeMode,
    TierCapabilitySnapshot,
)


def native_key_prefix_ok(expected: str, observed: str) -> bool:
    """True when ``observed`` native-key JSON is a block-aligned prefix of ``expected``.

    Churn can evict the tail chunks of a long prefix between scheduler lookup
    and native load.  The worker then loads a shorter, block-aligned prefix of
    the very same object.  ``observed == expected[:len(observed)]`` is the
    safety condition that keeps the loaded KV inside the scheduler-declared
    prefix; the evicted tail is reconciled by ``missing_tokens`` recompute.
    """
    try:
        expected_keys = json.loads(expected)
        observed_keys = json.loads(observed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(expected_keys, list) or not isinstance(observed_keys, list):
        return False
    if not observed_keys or len(observed_keys) > len(expected_keys):
        return False
    return observed_keys == expected_keys[: len(observed_keys)]


def physical_identity_compatible(expected: PhysicalKVObject, observed: PhysicalKVObject) -> bool:
    """True when ``observed`` is the same binding as ``expected`` or a shorter
    block-aligned prefix of it (churn can evict tail chunks between scheduler
    lookup and native load)."""
    if expected.binding_generation != observed.binding_generation:
        return False
    if observed.native_key == expected.native_key:
        return True
    return native_key_prefix_ok(expected.native_key, observed.native_key)


@dataclass(frozen=True, slots=True)
class SchedulerLookupObservation:
    request_id: str
    physical_object_id: str
    binding_generation: int
    locally_cached_tokens: int
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
        self._lock = threading.RLock()
        self.tickets = PrefetchTicketStore()
        self._intents: dict[str, RequestKVIntent] = {}
        self._lookups: dict[str, SchedulerLookupObservation] = {}
        self._admissions: dict[str, SchedulerAdmission] = {}
        self._receipts: dict[str, NativeKVLoadReceipt] = {}
        self._scheduled_prefill_tokens: dict[str, int] = {}
        self._final_accounting: dict[str, RequestKVAccounting] = {}

    def submit_intent(self, intent: RequestKVIntent) -> SchedulerAdmission:
        with self._lock:
            if intent.request_id in self._intents:
                raise ValueError("request already has a KV-Core intent")
            self._intents[intent.request_id] = intent
        if self.mode is RuntimeMode.OFF:
            return SchedulerAdmission(
                intent.request_id, intent.physical_object.physical_object_id,
                intent.physical_object.binding_generation, 0, "off", "kv_core_off",
            )
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
        if physical.source_tier not in {"ssd", "mixed"} or ticket.source_tier != "ssd":
            return "source_tier_not_ssd"
        reason = self.capability.prefetch_block_reason(
            size_bytes=ticket.requested_bytes,
            deadline_ns=ticket.deadline_ns,
            now_ns=now_ns,
            in_flight_reserved_bytes=self.tickets.in_flight_reserved_bytes(),
        )
        if reason is not None:
            return reason
        self.tickets.submit(ticket)
        return None

    def complete_cpu_prefetch(self, prefetch_id: str, *, completed_bytes: int, now_ns: int | None = None) -> PrefetchTicket:
        return self.tickets.complete(prefetch_id, completed_bytes=completed_bytes, now_ns=now_ns)

    def record_scheduler_lookup(
        self, *, request_id: str, physical: PhysicalKVObject, lookup_hit_tokens: int,
        locally_cached_tokens: int = 0, native_request_id: str,
    ) -> SchedulerLookupObservation:
        intent = self._intent(request_id, physical)
        if lookup_hit_tokens < 0 or lookup_hit_tokens > intent.requested_prefix_tokens:
            raise ValueError("lookup_hit_tokens must be within the requested prefix")
        if locally_cached_tokens < 0 or locally_cached_tokens > intent.requested_prefix_tokens:
            raise ValueError("locally_cached_tokens must be within the requested prefix")
        if not native_request_id:
            raise ValueError("native_request_id is required")
        observed = SchedulerLookupObservation(
            request_id, physical.physical_object_id, physical.binding_generation,
            int(locally_cached_tokens), int(lookup_hit_tokens), native_request_id,
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
        if self.capability.external_token_cap > 0 and allocated_external_tokens > self.capability.external_token_cap:
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
        native_retrieved_tokens: int | None = None,
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
            native_key=physical.native_key,
            compatibility_identity=physical.compatibility_key.identity,
            prefix_hash=physical.compatibility_key.prefix_hash,
            requested_prefix_tokens=intent.requested_prefix_tokens,
            locally_cached_tokens=lookup.locally_cached_tokens,
            lookup_hit_tokens=lookup.lookup_hit_tokens,
            allocated_external_tokens=allocated,
            actual_loaded_tokens=int(actual_loaded_tokens),
            native_retrieved_tokens=(
                int(actual_loaded_tokens)
                if native_retrieved_tokens is None
                else int(native_retrieved_tokens)
            ),
            missing_tokens=max(
                0,
                intent.requested_prefix_tokens
                - lookup.locally_cached_tokens
                - int(actual_loaded_tokens),
            ),
            unallocated_recompute_tokens=max(
                0,
                intent.requested_prefix_tokens
                - lookup.locally_cached_tokens
                - allocated,
            ),
            load_shortfall_tokens=allocated - int(actual_loaded_tokens),
            bytes_loaded=int(bytes_loaded),
            load_latency_ns=int(load_latency_ns),
            status=status,
            native_request_id=native_request_id,
            prefetch_id=prefetch_id,
        )
        self._receipts[request_id] = receipt
        return receipt

    def record_scheduler_compute(self, *, request_id: str, scheduled_tokens: int) -> None:
        if request_id not in self._intents:
            return
        if scheduled_tokens < 0:
            raise ValueError("scheduled_tokens must be non-negative")
        self._scheduled_prefill_tokens[request_id] = (
            self._scheduled_prefill_tokens.get(request_id, 0) + int(scheduled_tokens)
        )

    def consume_cpu_prefetch(
        self,
        prefetch_id: str,
        *,
        request_id: str,
        physical: PhysicalKVObject,
        now_ns: int | None = None,
    ) -> PrefetchTicket:
        return self.tickets.consume(
            prefetch_id,
            request_id=request_id,
            physical_object_id=physical.physical_object_id,
            binding_generation=physical.binding_generation,
            prefix_hash=physical.compatibility_key.prefix_hash,
            native_key=physical.native_key,
            compatibility_identity=physical.compatibility_key.identity,
            now_ns=now_ns,
        )

    def update_capability(self, capability: TierCapabilitySnapshot) -> None:
        with self._lock:
            self.capability = capability

    def intent_for(self, request_id: str) -> RequestKVIntent | None:
        with self._lock:
            return self._intents.get(request_id)

    def lookup_for(self, request_id: str) -> SchedulerLookupObservation | None:
        with self._lock:
            return self._lookups.get(request_id)

    def admission_for(self, request_id: str) -> SchedulerAdmission | None:
        with self._lock:
            return self._admissions.get(request_id)

    def scheduled_prefill_for(self, request_id: str) -> int:
        with self._lock:
            return self._scheduled_prefill_tokens.get(request_id, 0)

    def import_native_load_receipt(
        self,
        receipt: NativeKVLoadReceipt,
        *,
        physical: PhysicalKVObject,
    ) -> None:
        """Import a worker receipt after validating scheduler-owned identity."""
        intent = self._intent(receipt.request_id, physical)
        lookup = self._lookup(receipt.request_id, physical)
        admission = self._admissions.get(receipt.request_id)
        allocated = 0 if admission is None else admission.allocated_external_tokens
        if receipt.binding_generation != physical.binding_generation:
            raise ValueError("worker receipt compatibility identity mismatch")
        keys_equal = receipt.native_key == physical.native_key
        keys_partial = native_key_prefix_ok(physical.native_key, receipt.native_key)
        if not (keys_equal or keys_partial):
            raise ValueError("worker receipt compatibility identity mismatch")
        if keys_equal and (
            receipt.compatibility_identity != physical.compatibility_key.identity
            or receipt.prefix_hash != physical.compatibility_key.prefix_hash
        ):
            raise ValueError("worker receipt compatibility identity mismatch")
        # ``keys_partial`` describes a block-aligned prefix of the same object
        # after churn evicted tail chunks; the shorter prefix legitimately
        # carries different derived identity fields.  ``missing_tokens``
        # reconciles the evicted tail through recompute.
        if receipt.requested_prefix_tokens != intent.requested_prefix_tokens:
            raise ValueError("worker receipt requested prefix mismatch")
        if receipt.locally_cached_tokens != lookup.locally_cached_tokens:
            raise ValueError("worker receipt local KV count mismatch")
        if receipt.lookup_hit_tokens != lookup.lookup_hit_tokens:
            raise ValueError("worker receipt lookup count mismatch")
        if receipt.allocated_external_tokens != allocated:
            raise ValueError("worker receipt scheduler allocation mismatch")
        with self._lock:
            self._receipts[receipt.request_id] = receipt

    def finalize_request(
        self,
        *,
        request_id: str,
        physical: PhysicalKVObject,
        finish_status: str,
        completed: bool,
        native_num_computed_tokens: int | None = None,
    ) -> RequestKVAccounting:
        intent = self._intent(request_id, physical)
        lookup = self._lookup(request_id, physical)
        admission = self._admissions.get(request_id)
        receipt = self._receipts.get(request_id)
        allocated = 0 if admission is None else admission.allocated_external_tokens
        loaded = 0 if receipt is None else receipt.actual_loaded_tokens
        load_shortfall = max(0, allocated - loaded)
        unallocated_recompute = max(
            0,
            intent.requested_prefix_tokens
            - lookup.locally_cached_tokens
            - allocated,
        )
        missing = max(
            0,
            intent.requested_prefix_tokens - lookup.locally_cached_tokens - loaded,
        )
        native_receipt_complete = allocated == 0 or receipt is not None
        native_compute_closes_prefix = (
            native_num_computed_tokens is not None
            and int(native_num_computed_tokens) >= intent.requested_prefix_tokens
        )
        recompute_confirmed = bool(
            completed
            and native_receipt_complete
            and native_compute_closes_prefix
            and load_shortfall == 0
        )
        if not completed:
            terminal_reason = "native_request_cancelled_or_failed"
        elif allocated > 0 and receipt is None:
            terminal_reason = "native_load_receipt_missing"
        elif load_shortfall > 0:
            terminal_reason = "native_load_shortfall_unsafe"
        elif not native_compute_closes_prefix:
            terminal_reason = "native_recompute_evidence_missing"
        elif allocated == 0:
            terminal_reason = "scheduler_declined_recompute"
        elif unallocated_recompute > 0:
            terminal_reason = "native_partial_prefix_load_recompute"
        else:
            terminal_reason = "native_load_completed"
        accounting = RequestKVAccounting(
            request_id=request_id,
            physical_object_id=physical.physical_object_id,
            binding_generation=physical.binding_generation,
            native_key=physical.native_key,
            compatibility_identity=physical.compatibility_key.identity,
            prefix_hash=physical.compatibility_key.prefix_hash,
            requested_prefix_tokens=intent.requested_prefix_tokens,
            locally_cached_tokens=lookup.locally_cached_tokens,
            lookup_hit_tokens=lookup.lookup_hit_tokens,
            allocated_external_tokens=allocated,
            actual_loaded_tokens=loaded,
            missing_tokens=missing,
            unallocated_recompute_tokens=unallocated_recompute,
            load_shortfall_tokens=load_shortfall,
            scheduled_prefill_tokens=self._scheduled_prefill_tokens.get(request_id, 0),
            recomputed_tokens=unallocated_recompute if recompute_confirmed else 0,
            recompute_confirmed=recompute_confirmed,
            finish_status=finish_status,
            terminal_reason=terminal_reason,
        )
        self._final_accounting[request_id] = accounting
        return accounting

    def final_accounting_for(self, request_id: str) -> RequestKVAccounting | None:
        return self._final_accounting.get(request_id)

    def receipt_for(self, request_id: str) -> NativeKVLoadReceipt | None:
        return self._receipts.get(request_id)

    def _intent(self, request_id: str, physical: PhysicalKVObject) -> RequestKVIntent:
        intent = self._intents.get(request_id)
        if intent is None:
            raise ValueError("native callback has no request intent")
        if not physical_identity_compatible(intent.physical_object, physical):
            raise ValueError("native callback physical identity mismatch")
        return intent

    def _lookup(self, request_id: str, physical: PhysicalKVObject) -> SchedulerLookupObservation:
        lookup = self._lookups.get(request_id)
        if lookup is None:
            raise ValueError("native callback has no scheduler lookup")
        if lookup.binding_generation != physical.binding_generation:
            raise ValueError("scheduler lookup physical generation mismatch")
        return lookup


__all__ = ["KVCoreConnectorCallbacks", "SchedulerAdmission", "SchedulerLookupObservation"]
