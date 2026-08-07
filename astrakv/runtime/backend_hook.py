"""Verified public Hook records for online backend control.

The Hook, rather than AstraKV, owns binding to a third-party KV object.
These records make that boundary explicit and reject incomplete identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from astrakv.runtime.eviction import ObjectLevel


BACKEND_HOOK_SCHEMA = "astrakv-backend-hook-v2"
EXECUTION_SPEC_SCHEMA = "astrakv-runtime-execution-spec-v1"


@dataclass(frozen=True, slots=True)
class BackendExecutionSpec:
    spec_id: str
    binding_id: str
    binding_generation: int
    backend_object_id: str
    object_key: str
    object_level: ObjectLevel
    runtime_owner: str
    owner_channel: str
    key_identity: str
    lifecycle: str
    actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_generation", _binding_generation(self.binding_generation))
        if not all((
            self.spec_id,
            self.binding_id,
            self.backend_object_id,
            self.object_key,
            self.runtime_owner,
            self.owner_channel,
            self.key_identity,
            self.lifecycle,
        )):
            raise ValueError("execution spec requires non-empty identity fields")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BackendExecutionSpec":
        if str(record.get("schema") or EXECUTION_SPEC_SCHEMA) != EXECUTION_SPEC_SCHEMA:
            raise ValueError("execution_spec schema is invalid")
        actions = record.get("actions")
        if not isinstance(actions, dict):
            raise ValueError("execution_spec actions are required")
        return cls(
            spec_id=_required(record, "spec_id"),
            binding_id=_required(record, "binding_id"),
            binding_generation=_wire_binding_generation(record),
            backend_object_id=_required(record, "backend_object_id"),
            object_key=_required(record, "object_key"),
            object_level=_object_level(record),
            runtime_owner=_required(record, "runtime_owner"),
            owner_channel=_required(record, "owner_channel"),
            key_identity=_required(record, "key_identity"),
            lifecycle=_required(record, "lifecycle"),
            actions={str(name): dict(value) for name, value in actions.items() if isinstance(value, dict)},
            metadata=dict(record.get("metadata") or {}),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_SPEC_SCHEMA,
            "spec_id": self.spec_id,
            "binding_id": self.binding_id,
            "binding_generation": self.binding_generation,
            "backend_object_id": self.backend_object_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "runtime_owner": self.runtime_owner,
            "owner_channel": self.owner_channel,
            "key_identity": self.key_identity,
            "lifecycle": self.lifecycle,
            "actions": {str(name): dict(value) for name, value in self.actions.items()},
            "metadata": dict(self.metadata),
        }


class HookAction(str, Enum):
    CACHE_CREATE = "cache_create"
    CACHE_STORE = "cache_store"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_LOAD = "cache_load"
    PREFETCH = "prefetch"
    OFFLOAD = "offload"
    EVICT = "evict"
    DROP = "drop"
    RELEASE = "release"
    # KV-Core actions have explicit ownership.  The native connector is the
    # only component allowed to emit the GPU load and recompute outcomes.
    ADMIT_EXTERNAL_PREFIX = "admit_external_prefix"
    NATIVE_LOOKUP = "native_lookup"
    NATIVE_LOAD_TO_PAGED_GPU = "native_load_to_paged_gpu"
    RECOMPUTE_MISSING_SUFFIX = "recompute_missing_suffix"
    PREFETCH_SSD_TO_CPU = "prefetch_ssd_to_cpu"
    DEMOTE_CPU_COPY = "demote_cpu_copy"
    INVALIDATE_EXTERNAL_COPY = "invalidate_external_copy"
    PREFETCH_CANCELLED = "prefetch_cancelled"


def _required(record: dict[str, Any], name: str) -> str:
    value = str(record.get(name) or "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _object_level(record: dict[str, Any]) -> ObjectLevel:
    try:
        return ObjectLevel(_required(record, "object_level"))
    except ValueError as exc:
        raise ValueError("object_level is invalid") from exc


def _timestamp(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp_ns is required") from exc
    if result < 0:
        raise ValueError("timestamp_ns must be non-negative")
    return result


def _binding_generation(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("binding_generation must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("binding_generation must be positive") from exc
    if result <= 0:
        raise ValueError("binding_generation must be positive")
    return result


def _wire_binding_generation(record: dict[str, Any]) -> int:
    if "binding_generation" not in record:
        raise ValueError("binding_generation is required")
    return _binding_generation(record["binding_generation"])


@dataclass(frozen=True, slots=True)
class BackendObjectBinding:
    run_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    backend_object_id: str
    binding_id: str
    verified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    binding_generation: int = 1
    execution_spec: BackendExecutionSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_generation", _binding_generation(self.binding_generation))
        if self.execution_spec is not None:
            if (
                self.execution_spec.binding_id != self.binding_id
                or self.execution_spec.backend_object_id != self.backend_object_id
                or self.execution_spec.object_key != self.object_key
                or self.execution_spec.object_level != self.object_level
            ):
                raise ValueError("execution_spec does not match binding identity")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BackendObjectBinding":
        return cls(
            run_id=_required(record, "run_id"),
            request_id=_required(record, "request_id"),
            object_key=_required(record, "object_key"),
            object_level=_object_level(record),
            backend_object_id=_required(record, "backend_object_id"),
            binding_id=_required(record, "binding_id"),
            binding_generation=_wire_binding_generation(record),
            verified=bool(record.get("verified", True)),
            metadata=dict(record.get("metadata") or {}),
            execution_spec=(
                None if not isinstance(record.get("execution_spec"), dict)
                else BackendExecutionSpec.from_record(record["execution_spec"])
            ),
        )

    def matches_event(self, event: "BackendHookEvent") -> bool:
        event_binding_id = str(event.metadata.get("binding_id") or "").strip()
        return (
            self.verified
            and self.run_id == event.run_id
            and self.request_id == event.request_id
            and self.object_key == event.object_key
            and self.object_level == event.object_level
            and self.backend_object_id == event.backend_object_id
            and (not event_binding_id or self.binding_id == event_binding_id)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": BACKEND_HOOK_SCHEMA,
            "record_type": "binding",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "backend_object_id": self.backend_object_id,
            "binding_id": self.binding_id,
            "binding_generation": self.binding_generation,
            "verified": self.verified,
            "metadata": dict(self.metadata),
            "execution_spec": None if self.execution_spec is None else self.execution_spec.to_record(),
        }


@dataclass(frozen=True, slots=True)
class BackendHookEvent:
    run_id: str
    event_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    backend_object_id: str
    action: HookAction
    status: str
    timestamp_ns: int
    tier_before: str = "unknown"
    tier_after: str = "unknown"
    bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    binding_generation: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_generation", _binding_generation(self.binding_generation))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BackendHookEvent":
        try:
            action = HookAction(_required(record, "action"))
        except ValueError as exc:
            raise ValueError("action is invalid") from exc
        raw_bytes = record.get("bytes")
        try:
            size = None if raw_bytes in (None, "") else int(raw_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("bytes is invalid") from exc
        return cls(
            run_id=_required(record, "run_id"),
            event_id=_required(record, "event_id"),
            request_id=_required(record, "request_id"),
            object_key=_required(record, "object_key"),
            object_level=_object_level(record),
            backend_object_id=_required(record, "backend_object_id"),
            action=action,
            status=_required(record, "status"),
            timestamp_ns=_timestamp(record.get("timestamp_ns")),
            tier_before=str(record.get("tier_before") or "unknown"),
            tier_after=str(record.get("tier_after") or "unknown"),
            bytes=size,
            metadata=dict(record.get("metadata") or {}),
            binding_generation=_wire_binding_generation(record),
        )

    def with_backend_object_id(self, backend_object_id: str) -> "BackendHookEvent":
        return replace(self, backend_object_id=backend_object_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": BACKEND_HOOK_SCHEMA,
            "record_type": "event",
            "run_id": self.run_id,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "backend_object_id": self.backend_object_id,
            "action": self.action.value,
            "status": self.status,
            "timestamp_ns": self.timestamp_ns,
            "tier_before": self.tier_before,
            "tier_after": self.tier_after,
            "bytes": self.bytes,
            "binding_generation": self.binding_generation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BackendActionCommand:
    run_id: str
    command_id: str
    decision_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    binding_id: str
    backend_object_id: str
    action: HookAction
    issued_at_ns: int
    target_tier: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    binding_generation: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_generation", _binding_generation(self.binding_generation))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BackendActionCommand":
        try:
            action = HookAction(_required(record, "action"))
        except ValueError as exc:
            raise ValueError("action is invalid") from exc
        return cls(
            run_id=_required(record, "run_id"),
            command_id=_required(record, "command_id"),
            decision_id=_required(record, "decision_id"),
            request_id=_required(record, "request_id"),
            object_key=_required(record, "object_key"),
            object_level=_object_level(record),
            binding_id=_required(record, "binding_id"),
            backend_object_id=_required(record, "backend_object_id"),
            action=action,
            issued_at_ns=_timestamp(record.get("issued_at_ns")),
            target_tier=str(record.get("target_tier") or "unknown"),
            metadata=dict(record.get("metadata") or {}),
            binding_generation=_wire_binding_generation(record),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": BACKEND_HOOK_SCHEMA,
            "record_type": "command",
            "run_id": self.run_id,
            "command_id": self.command_id,
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "binding_id": self.binding_id,
            "backend_object_id": self.backend_object_id,
            "action": self.action.value,
            "issued_at_ns": self.issued_at_ns,
            "target_tier": self.target_tier,
            "metadata": dict(self.metadata),
            "binding_generation": self.binding_generation,
        }


@dataclass(frozen=True, slots=True)
class BackendActionReceipt:
    run_id: str
    command_id: str
    receipt_id: str
    binding_id: str
    backend_object_id: str
    action: HookAction
    status: str
    timestamp_ns: int
    tier_before: str = "unknown"
    tier_after: str = "unknown"
    bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    binding_generation: int = 1
    decision_id: str = ""
    request_id: str = ""
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_generation", _binding_generation(self.binding_generation))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BackendActionReceipt":
        try:
            action = HookAction(_required(record, "action"))
        except ValueError as exc:
            raise ValueError("action is invalid") from exc
        raw_bytes = record.get("bytes")
        try:
            size = None if raw_bytes in (None, "") else int(raw_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("bytes is invalid") from exc
        return cls(
            run_id=_required(record, "run_id"),
            command_id=_required(record, "command_id"),
            receipt_id=_required(record, "receipt_id"),
            binding_id=_required(record, "binding_id"),
            backend_object_id=_required(record, "backend_object_id"),
            action=action,
            status=_required(record, "status"),
            timestamp_ns=_timestamp(record.get("timestamp_ns")),
            tier_before=str(record.get("tier_before") or "unknown"),
            tier_after=str(record.get("tier_after") or "unknown"),
            bytes=size,
            metadata=dict(record.get("metadata") or {}),
            binding_generation=_wire_binding_generation(record),
            decision_id=str(record.get("decision_id") or ""),
            request_id=str(record.get("request_id") or ""),
            rejection_reason=str(record.get("rejection_reason") or ""),
        )

    def matches_command(self, command: BackendActionCommand) -> bool:
        return (
            self.run_id == command.run_id
            and self.command_id == command.command_id
            and self.binding_id == command.binding_id
            and self.backend_object_id == command.backend_object_id
            and self.binding_generation == command.binding_generation
            and self.action == command.action
            and (not self.decision_id or self.decision_id == command.decision_id)
            and (not self.request_id or self.request_id == command.request_id)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": BACKEND_HOOK_SCHEMA,
            "record_type": "receipt",
            "run_id": self.run_id,
            "command_id": self.command_id,
            "receipt_id": self.receipt_id,
            "binding_id": self.binding_id,
            "backend_object_id": self.backend_object_id,
            "action": self.action.value,
            "status": self.status,
            "timestamp_ns": self.timestamp_ns,
            "tier_before": self.tier_before,
            "tier_after": self.tier_after,
            "bytes": self.bytes,
            "metadata": dict(self.metadata),
            "binding_generation": self.binding_generation,
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "rejection_reason": self.rejection_reason,
        }
