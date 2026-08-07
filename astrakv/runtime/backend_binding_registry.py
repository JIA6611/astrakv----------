"""Request-aware bindings for instrumented backend objects.

The registry deliberately does not infer a request from an LMCache key.  A key
without caller-provided request context is an observation only and can never be
used as a control-plane binding.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from astrakv.runtime.backend_hook import BackendHookEvent, BackendObjectBinding, HookAction
from astrakv.runtime.eviction import ObjectLevel


def _canonical_key(key: Any) -> str:
    """Produce a stable identity for the LMCache key rather than object repr/address."""
    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [normalize(item) for item in value]}
        if isinstance(value, list):
            return {"list": [normalize(item) for item in value]}
        if isinstance(value, dict):
            return {"dict": {str(name): normalize(item) for name, item in sorted(value.items(), key=lambda item: str(item[0]))}}
        if hasattr(value, "__dict__"):
            return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "fields": normalize(vars(value))}
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": str(value)}

    return json.dumps(normalize(key), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RequestContext:
    run_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.request_id or not self.object_key:
            raise ValueError("request context requires run_id, request_id, and object_key")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class BindingObservation:
    binding: BackendObjectBinding | None
    event: BackendHookEvent | None
    record: dict[str, Any]
    binding_record: dict[str, Any] | None = None

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        if self.binding is None or self.event is None:
            return (self.record,)
        return (self.binding_record or self.binding.to_record(), self.event.to_record())


@dataclass(slots=True)
class _PhysicalBinding:
    canonical_key: str
    backend_object_id: str
    binding_id: str
    binding_generation: int = 1
    logical_object_key: str = ""
    logical_object_level: ObjectLevel = ObjectLevel.PREFIX
    lifecycle: str = "created"
    active_request_ids: set[str] = field(default_factory=set)
    pin_count: int = 0
    pending_io: int = 0
    pending_operations: dict[str, tuple[int, str]] = field(default_factory=dict)
    action_reservation: str | None = None
    previous_binding_id: str = ""
    binding_replacement_reason: str = ""
    size_bytes: int = 0


@dataclass(slots=True)
class _ActionReservation:
    physical: _PhysicalBinding
    deadline_ns: int
    state: str = "reserved"
    command_id: str | None = None


class BackendBindingRegistry:
    """Own physical-object state while producing request-specific binding records."""

    def __init__(self, *, run_id: str, engine_instance_id: str, worker_id: str) -> None:
        if not run_id or not engine_instance_id or not worker_id:
            raise ValueError("run_id, engine_instance_id, and worker_id are required")
        self.run_id = run_id
        self.engine_instance_id = engine_instance_id
        self.worker_id = worker_id
        self._physical: dict[str, _PhysicalBinding] = {}
        self._by_binding_id: dict[str, _PhysicalBinding] = {}
        self._associations: dict[tuple[str, str, str, ObjectLevel], BackendObjectBinding] = {}
        self._retired_bindings: dict[str, dict[str, Any]] = {}
        self._event_index = 0
        self._operation_index = 0
        self._binding_index = 0
        self._reservation_index = 0
        self._reservations: dict[str, _ActionReservation] = {}
        self._reservation_terminal_states: dict[str, str] = {}
        self._lock = threading.RLock()

    def observe(
        self,
        key: Any,
        action: HookAction,
        status: str,
        context: RequestContext | None,
        *,
        timestamp_ns: int | None = None,
        tier_before: str = "unknown",
        tier_after: str = "unknown",
        bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        operation_lease: str | None = None,
    ) -> BindingObservation:
        with self._lock:
            return self._observe_locked(
                key, action, status, context, timestamp_ns=timestamp_ns,
                tier_before=tier_before, tier_after=tier_after, bytes=bytes,
                metadata=metadata, operation_lease=operation_lease,
            )

    def complete_operation(
        self, key: Any, action: HookAction, status: str, context: RequestContext, operation_lease: str,
        **kwargs: Any,
    ) -> BindingObservation:
        """Accept a terminal callback only for the exact lease issued at submit."""
        if not operation_lease:
            raise ValueError("operation lease is required")
        return self.observe(key, action, status, context, operation_lease=operation_lease, **kwargs)

    def _observe_locked(
        self, key: Any, action: HookAction, status: str, context: RequestContext | None,
        *, timestamp_ns: int | None, tier_before: str, tier_after: str, bytes: int | None,
        metadata: dict[str, Any] | None, operation_lease: str | None,
    ) -> BindingObservation:
        timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
        canonical_key = _canonical_key(key)
        base_metadata = {
            "engine_instance_id": self.engine_instance_id,
            "worker_id": self.worker_id,
            "lmcache_key_identity": canonical_key,
            **(dict(context.metadata) if context is not None else {}),
            **dict(metadata or {}),
        }
        physical = self._physical.get(canonical_key)
        if context is None:
            if physical is not None and physical.action_reservation is not None:
                return BindingObservation(
                    binding=None, event=None,
                    record={
                        "schema": "astrakv-backend-hook-v2", "record_type": "observation", "run_id": self.run_id,
                        "event_id": self._next_event_id(), "backend_object_id": physical.backend_object_id,
                        "action": action.value, "status": "deferred", "timestamp_ns": timestamp_ns,
                        "metadata": {
                            "observational_only": True, "bridge_eligible": False,
                            "deferred_by_reservation": True, "reservation_lease": physical.action_reservation,
                            **base_metadata,
                        },
                    },
                )
            return BindingObservation(
                binding=None, event=None,
                record={
                    "schema": "astrakv-backend-hook-v2", "record_type": "observation", "run_id": self.run_id,
                    "event_id": self._next_event_id(), "backend_object_id": self._backend_object_id(canonical_key),
                    "action": action.value, "status": status, "timestamp_ns": timestamp_ns,
                    "metadata": {"observational_only": True, "bridge_eligible": False, **base_metadata},
                },
            )
        if context.run_id != self.run_id:
            raise ValueError("request context run_id does not match registry run_id")
        if physical is None:
            physical = self._create_physical(
                canonical_key,
                object_key=context.object_key,
                object_level=context.object_level,
            )
        elif self._logical_identity_changed(physical, context):
            if not self._can_rotate_logical_identity(physical):
                return self._identity_conflict_observation(
                    physical,
                    action=action,
                    status=status,
                    context=context,
                    timestamp_ns=timestamp_ns,
                    base_metadata=base_metadata,
                    tier_before=tier_before,
                    tier_after=tier_after,
                    bytes=bytes,
                )
            self._replace_binding_identity(
                physical,
                object_key=context.object_key,
                object_level=context.object_level,
                replacement_reason="logical_object_changed",
            )
        elif physical.lifecycle in {"released", "dropped"} and action in {HookAction.CACHE_CREATE, HookAction.CACHE_STORE} and status == "submitted":
            self._replace_binding_identity(
                physical,
                object_key=context.object_key,
                object_level=context.object_level,
                replacement_reason="binding_recreated_after_release",
            )

        allow_reserved_io = bool(base_metadata.get("allow_reserved_io"))
        if (
            physical.action_reservation is not None
            and action in {HookAction.CACHE_CREATE, HookAction.CACHE_STORE, HookAction.CACHE_HIT, HookAction.CACHE_LOAD}
            and not allow_reserved_io
        ):
            raise ValueError("action reservation blocks new cache I/O")

        if bytes is not None:
            try:
                observed_bytes = int(bytes)
            except (TypeError, ValueError) as exc:
                raise ValueError("physical object bytes must be an integer") from exc
            if observed_bytes < 0:
                raise ValueError("physical object bytes must be non-negative")
            if action in {HookAction.CACHE_CREATE, HookAction.CACHE_STORE, HookAction.CACHE_HIT, HookAction.CACHE_LOAD}:
                physical.size_bytes = max(physical.size_bytes, observed_bytes)

        issued_lease = self._validate_or_issue_lease(physical, action, status, context.request_id, operation_lease)
        self._advance(physical, action, status, context.request_id, issued_lease)
        binding = BackendObjectBinding(
            run_id=self.run_id,
            request_id=context.request_id,
            object_key=context.object_key,
            object_level=context.object_level,
            backend_object_id=physical.backend_object_id,
            binding_id=physical.binding_id,
            binding_generation=physical.binding_generation,
            metadata={
                **self._binding_identity_metadata(physical),
                "engine_instance_id": self.engine_instance_id,
                "worker_id": self.worker_id,
                "lmcache_key_identity": canonical_key,
                "lifecycle": physical.lifecycle,
                "pin_count": physical.pin_count,
                "pending_io": physical.pending_io,
                "size_bytes": physical.size_bytes,
                **(
                    {"runtime_reqmeta_id": str(base_metadata["runtime_reqmeta_id"])}
                    if base_metadata.get("runtime_reqmeta_id")
                    else {}
                ),
            },
        )
        self._associations[(binding.binding_id, binding.request_id, binding.object_key, binding.object_level)] = binding
        event = BackendHookEvent(
            run_id=self.run_id,
            event_id=self._next_event_id(),
            request_id=context.request_id,
            object_key=context.object_key,
            object_level=context.object_level,
            backend_object_id=physical.backend_object_id,
            action=action,
            status=status,
            timestamp_ns=timestamp_ns,
            tier_before=tier_before,
            tier_after=tier_after,
            bytes=bytes,
            binding_generation=physical.binding_generation,
            metadata={
                "binding_id": physical.binding_id,
                **self._binding_identity_metadata(physical),
                "bridge_eligible": self.eligible_for_bridge(physical.binding_id, physical.binding_generation),
                "lifecycle": physical.lifecycle,
                "pin_count": physical.pin_count,
                "pending_io": physical.pending_io,
                **({"operation_lease": issued_lease} if issued_lease is not None else {}),
                **base_metadata,
            },
        )
        binding_record = binding.to_record()
        binding_record.update({
            "event_id": event.event_id,
            "action": event.action.value,
            "status": event.status,
            "timestamp_ns": event.timestamp_ns,
            "tier_before": event.tier_before,
            "tier_after": event.tier_after,
            "bytes": event.bytes,
        })
        return BindingObservation(
            binding=binding,
            event=event,
            record=event.to_record(),
            binding_record=binding_record,
        )

    def active_request_ids(self, binding_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._require_current(binding_id).active_request_ids))

    def is_active(self, binding_id: str) -> bool:
        with self._lock:
            return self._require_current(binding_id).lifecycle == "active"

    def eligible_for_bridge(self, binding_id: str, binding_generation: int) -> bool:
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            try:
                physical = self._require_current(binding_id)
            except ValueError:
                return False
            return bool(
                physical.binding_generation == binding_generation
                and physical.lifecycle in {"active", "released"}
                and not physical.active_request_ids
                and physical.pin_count == 0
                and physical.pending_io == 0
                and not physical.pending_operations
                and physical.action_reservation is None
            )

    def authorize_action(
        self, *, binding_id: str, binding_generation: int, backend_object_id: str,
        request_id: str, object_key: str, object_level: ObjectLevel,
    ) -> bool:
        """Return true only for a current, released logical association safe to drop."""
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            try:
                physical = self._require_current(binding_id)
            except ValueError:
                return False
            association = self._associations.get((binding_id, request_id, object_key, object_level))
            return bool(
                association is not None
                and physical.binding_id == binding_id
                and physical.binding_generation == binding_generation
                and physical.backend_object_id == backend_object_id
                and association.binding_generation == binding_generation
                and association.backend_object_id == backend_object_id
                and physical.lifecycle in {"active", "released"}
                and not physical.active_request_ids
                and physical.pin_count == 0
                and physical.pending_io == 0
                and not physical.pending_operations
                and physical.action_reservation is None
            )

    def reserve_action(
        self, *, binding_id: str, binding_generation: int, backend_object_id: str,
        request_id: str, object_key: str, object_level: ObjectLevel, deadline_ns: int | None = None,
    ) -> str | None:
        """Atomically reserve a safe physical object for one destructive action."""
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            if not self.authorize_action(
                binding_id=binding_id, binding_generation=binding_generation, backend_object_id=backend_object_id,
                request_id=request_id, object_key=object_key, object_level=object_level,
            ):
                return None
            physical = self._require_current(binding_id)
            self._reservation_index += 1
            lease = f"{binding_id}:action:{self._reservation_index}"
            physical.action_reservation = lease
            self._reservations[lease] = _ActionReservation(
                physical=physical,
                deadline_ns=deadline_ns if deadline_ns is not None else time.time_ns() + 30_000_000_000,
            )
            return lease

    def consume_action_reservation(
        self, reservation_lease: str | None, *, command_id: str, now_ns: int | None = None,
        binding_id: str | None = None, binding_generation: int | None = None, backend_object_id: str | None = None,
    ) -> bool:
        """Bind a bridge-issued lease to exactly one endpoint command."""
        if not reservation_lease or not command_id:
            return False
        with self._lock:
            self._reap_expired_locked(time.time_ns() if now_ns is None else now_ns)
            reservation = self._reservations.get(reservation_lease)
            if reservation is None or reservation.state != "reserved":
                return False
            physical = reservation.physical
            if (
                binding_id is not None and physical.binding_id != binding_id
                or binding_generation is not None and physical.binding_generation != binding_generation
                or backend_object_id is not None and physical.backend_object_id != backend_object_id
            ):
                return False
            reservation.state = "consumed"
            reservation.command_id = command_id
            return True

    def complete_action(
        self,
        reservation_lease: str | None,
        *,
        command_id: str,
        status: str,
        preserve_lifecycle: bool = False,
    ) -> bool:
        """Commit or cancel one consumed lease; terminal leases cannot be replayed."""
        if not reservation_lease or not command_id:
            return False
        with self._lock:
            reservation = self._reservations.get(reservation_lease)
            if (
                reservation is None or reservation.state != "consumed"
                or reservation.command_id != command_id
                or reservation.physical.action_reservation != reservation_lease
            ):
                return False
            physical = reservation.physical
            del self._reservations[reservation_lease]
            physical.action_reservation = None
            terminal_state = "committed" if status in {"completed", "ok", "executed"} else "cancelled"
            self._reservation_terminal_states[reservation_lease] = terminal_state
            if status in {"completed", "ok", "executed"} and not preserve_lifecycle:
                physical.lifecycle = "dropped"
                physical.active_request_ids.clear()
                physical.pin_count = 0
                physical.pending_io = 0
                physical.pending_operations.clear()
            return True

    def cancel_action_reservation(
        self,
        reservation_lease: str | None,
        *,
        command_id: str,
        status: str = "failed",
        preserve_lifecycle: bool = False,
    ) -> bool:
        return self.complete_action(
            reservation_lease,
            command_id=command_id,
            status=status,
            preserve_lifecycle=preserve_lifecycle,
        )

    def reap_expired_reservations(self, *, now_ns: int | None = None) -> tuple[str, ...]:
        with self._lock:
            return self._reap_expired_locked(time.time_ns() if now_ns is None else now_ns)

    def reservation_state(self, reservation_lease: str | None) -> str | None:
        if not reservation_lease:
            return None
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            reservation = self._reservations.get(reservation_lease)
            return reservation.state if reservation is not None else self._reservation_terminal_states.get(reservation_lease)

    def is_key_reserved(self, key: Any) -> bool:
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            physical = self._physical.get(_canonical_key(key))
            return physical is not None and physical.action_reservation is not None

    def current_binding(
        self, *, binding_id: str, binding_generation: int, request_id: str,
        object_key: str, object_level: ObjectLevel,
    ) -> BackendObjectBinding | None:
        with self._lock:
            binding = self._associations.get((binding_id, request_id, object_key, object_level))
            if binding is None:
                return None
            try:
                physical = self._require_current(binding_id)
            except ValueError:
                return None
            return binding if (
                physical.binding_id == binding_id
                and physical.binding_generation == binding_generation
                and binding.binding_generation == binding_generation
            ) else None

    def binding_status(self, binding_id: str) -> str:
        with self._lock:
            if binding_id in self._by_binding_id:
                return "current"
            if binding_id in self._retired_bindings:
                return "replaced"
            return "unknown"

    def snapshot(self, binding_id: str) -> dict[str, Any]:
        with self._lock:
            self._reap_expired_locked(time.time_ns())
            retired = self._retired_bindings.get(binding_id)
            if retired is not None:
                return dict(retired)
            physical = self._require_current(binding_id)
            return {
                "binding_id": physical.binding_id,
                "backend_object_id": physical.backend_object_id,
                "binding_generation": physical.binding_generation,
                "binding_identity_mode": "unique_binding_id",
                "lifecycle": physical.lifecycle,
                "active_request_ids": tuple(sorted(physical.active_request_ids)),
                "pin_count": physical.pin_count,
                "pending_io": physical.pending_io,
                "pending_operations": tuple(sorted(physical.pending_operations)),
                "action_reservation": physical.action_reservation,
                "size_bytes": physical.size_bytes,
            }

    def _create_physical(
        self,
        canonical_key: str,
        *,
        object_key: str,
        object_level: ObjectLevel,
    ) -> _PhysicalBinding:
        digest = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:20]
        physical = _PhysicalBinding(
            canonical_key=canonical_key,
            backend_object_id=self._backend_object_id(canonical_key),
            binding_id=self._new_binding_id(digest),
            logical_object_key=object_key,
            logical_object_level=object_level,
        )
        self._physical[canonical_key] = physical
        self._by_binding_id[physical.binding_id] = physical
        return physical

    def _replace_binding_identity(
        self,
        physical: _PhysicalBinding,
        *,
        object_key: str,
        object_level: ObjectLevel,
        replacement_reason: str,
    ) -> None:
        previous_binding_id = physical.binding_id
        digest = hashlib.sha256(physical.canonical_key.encode("utf-8")).hexdigest()[:20]
        physical.binding_id = self._new_binding_id(digest)
        physical.binding_generation += 1
        physical.logical_object_key = object_key
        physical.logical_object_level = object_level
        physical.lifecycle = "created"
        physical.active_request_ids.clear()
        physical.pin_count = 0
        physical.pending_io = 0
        physical.pending_operations.clear()
        physical.action_reservation = None
        physical.previous_binding_id = previous_binding_id
        physical.binding_replacement_reason = replacement_reason
        self._retired_bindings[previous_binding_id] = {
            "binding_id": previous_binding_id,
            "backend_object_id": physical.backend_object_id,
            "binding_generation": physical.binding_generation - 1,
            "binding_identity_mode": "unique_binding_id",
            "lifecycle": "replaced",
            "active_request_ids": (),
            "pin_count": 0,
            "pending_io": 0,
            "pending_operations": (),
            "action_reservation": None,
            "size_bytes": physical.size_bytes,
            "previous_binding_id": previous_binding_id,
            "replaced_by_binding_id": physical.binding_id,
            "binding_replacement_reason": replacement_reason,
        }
        self._by_binding_id.pop(previous_binding_id, None)
        self._by_binding_id[physical.binding_id] = physical

    @staticmethod
    def _binding_identity_metadata(physical: _PhysicalBinding) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "binding_identity_mode": "unique_binding_id",
        }
        if physical.previous_binding_id:
            metadata["previous_binding_id"] = physical.previous_binding_id
        if physical.binding_replacement_reason:
            metadata["binding_replacement_reason"] = physical.binding_replacement_reason
        return metadata

    @staticmethod
    def _logical_identity_changed(physical: _PhysicalBinding, context: RequestContext) -> bool:
        return (
            bool(physical.logical_object_key)
            and (
                physical.logical_object_key != context.object_key
                or physical.logical_object_level is not context.object_level
            )
        )

    @staticmethod
    def _can_rotate_logical_identity(physical: _PhysicalBinding) -> bool:
        return (
            physical.lifecycle in {"created", "released", "dropped"}
            and not physical.active_request_ids
            and physical.pin_count == 0
            and physical.pending_io == 0
            and not physical.pending_operations
            and physical.action_reservation is None
        )

    def _identity_conflict_observation(
        self,
        physical: _PhysicalBinding,
        *,
        action: HookAction,
        status: str,
        context: RequestContext,
        timestamp_ns: int,
        base_metadata: dict[str, Any],
        tier_before: str,
        tier_after: str,
        bytes: int | None,
    ) -> BindingObservation:
        """Fail closed when one physical key is concurrently claimed by two objects."""
        return BindingObservation(
            binding=None,
            event=None,
            record={
                "schema": "astrakv-backend-hook-v2",
                "record_type": "observation",
                "run_id": self.run_id,
                "event_id": self._next_event_id(),
                "request_id": context.request_id,
                "object_key": context.object_key,
                "object_level": context.object_level.value,
                "backend_object_id": physical.backend_object_id,
                "binding_id": physical.binding_id,
                "binding_generation": physical.binding_generation,
                "action": action.value,
                "status": "binding_identity_conflict",
                "timestamp_ns": timestamp_ns,
                "tier_before": tier_before,
                "tier_after": tier_after,
                "bytes": bytes,
                "metadata": {
                    **base_metadata,
                    "observational_only": True,
                    "bridge_eligible": False,
                    "binding_identity_conflict": True,
                    "binding_identity_conflict_reason": "physical_key_claimed_by_active_logical_object",
                    "current_object_key": physical.logical_object_key,
                    "current_object_level": physical.logical_object_level.value,
                    "requested_object_key": context.object_key,
                    "requested_object_level": context.object_level.value,
                },
            },
        )

    def _backend_object_id(self, canonical_key: str) -> str:
        digest = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:20]
        return f"lmcache:{self.engine_instance_id}:{self.worker_id}:{digest}"

    def _next_event_id(self) -> str:
        self._event_index += 1
        return f"{self.run_id}:binding-event:{self._event_index}"

    def _next_operation_lease(self, physical: _PhysicalBinding) -> str:
        self._operation_index += 1
        return f"{physical.binding_id}:lease:{self._operation_index}"

    def _new_binding_id(self, digest: str) -> str:
        self._binding_index += 1
        return f"lmcache-binding:{self.engine_instance_id}:{self.worker_id}:{digest}:{self._binding_index}"

    def _reap_expired_locked(self, now_ns: int) -> tuple[str, ...]:
        expired = tuple(
            lease for lease, reservation in self._reservations.items()
            if reservation.state == "reserved" and reservation.deadline_ns < now_ns
        )
        for lease in expired:
            reservation = self._reservations.pop(lease)
            if reservation.physical.action_reservation == lease:
                reservation.physical.action_reservation = None
            self._reservation_terminal_states[lease] = "expired"
        return expired

    def _validate_or_issue_lease(
        self, physical: _PhysicalBinding, action: HookAction, status: str, request_id: str, operation_lease: str | None,
    ) -> str | None:
        if action != HookAction.CACHE_STORE:
            return None
        if status == "submitted":
            if operation_lease is not None:
                raise ValueError("store submission must not supply an operation lease")
            lease = self._next_operation_lease(physical)
            physical.pending_operations[lease] = (physical.binding_id, request_id)
            return lease
        if status in {"completed", "ok", "executed", "failed", "error"}:
            if not operation_lease:
                raise ValueError("store completion requires operation lease")
            token = physical.pending_operations.get(operation_lease)
            if token is None:
                raise ValueError("stale or unknown operation lease")
            if token != (physical.binding_id, request_id):
                raise ValueError("stale operation lease")
            del physical.pending_operations[operation_lease]
            return operation_lease
        return None

    def _require_current(self, binding_id: str) -> _PhysicalBinding:
        try:
            return self._by_binding_id[binding_id]
        except KeyError as exc:
            raise ValueError(f"unknown binding_id: {binding_id}") from exc

    @staticmethod
    def _advance(physical: _PhysicalBinding, action: HookAction, status: str, request_id: str, operation_lease: str | None) -> None:
        completed = status in {"completed", "ok", "executed"}
        if action == HookAction.CACHE_STORE:
            if status == "submitted":
                physical.pending_io += 1
                physical.lifecycle = "pending"
                physical.active_request_ids.add(request_id)
                physical.pin_count = len(physical.active_request_ids)
                return
            if completed:
                physical.pending_io = max(0, physical.pending_io - 1)
                physical.lifecycle = "active"
                physical.active_request_ids.add(request_id)
            elif operation_lease is not None:
                physical.pending_io = max(0, physical.pending_io - 1)
        elif action in {HookAction.CACHE_CREATE, HookAction.CACHE_HIT, HookAction.CACHE_LOAD} and completed:
            physical.lifecycle = "active"
            physical.active_request_ids.add(request_id)
        elif action == HookAction.RELEASE and completed:
            physical.active_request_ids.discard(request_id)
            stale_leases = [lease for lease, (_, owner) in physical.pending_operations.items() if owner == request_id]
            for lease in stale_leases:
                del physical.pending_operations[lease]
                physical.pending_io = max(0, physical.pending_io - 1)
            if not physical.active_request_ids and physical.pending_io == 0:
                physical.lifecycle = "released"
        elif action == HookAction.DROP and completed:
            physical.lifecycle = "dropped"
            physical.active_request_ids.clear()
            physical.pending_io = 0
            physical.pending_operations.clear()
            physical.action_reservation = None
        physical.pin_count = len(physical.active_request_ids)
