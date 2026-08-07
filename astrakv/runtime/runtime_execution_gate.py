"""Admission-only safety gate for commands destined for a backend action service."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry
from astrakv.runtime.backend_capabilities import BackendCapabilityPreflight
from astrakv.runtime.backend_hook import BackendActionCommand
from astrakv.runtime.circuit_breaker import CircuitBreaker


RUNTIME_GATE_SCHEMA = "astrakv-runtime-execution-gate-v1"


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    max_bytes_per_command: int = 1 << 30
    max_bytes_per_window: int = 1 << 32
    max_commands_per_window: int = 100
    window_ns: int = 1_000_000_000
    max_concurrent: int = 1

    def __post_init__(self) -> None:
        if min(self.max_bytes_per_command, self.max_bytes_per_window, self.max_commands_per_window, self.window_ns, self.max_concurrent) <= 0:
            raise ValueError("execution budget values must be positive")


@dataclass(frozen=True, slots=True)
class GateDecision:
    allowed: bool
    reason: str


class RuntimeExecutionGate:
    """Serialize command admission. This class never invokes a backend client."""

    def __init__(self, *, run_id: str, endpoint: str, capabilities: BackendCapabilityPreflight | None,
                 binding_registry: BackendBindingRegistry, budget: ExecutionBudget | None = None,
                 breaker: CircuitBreaker | None = None, state_path: Path | str | None = None) -> None:
        self.run_id = run_id
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.binding_registry = binding_registry
        self.budget = budget or ExecutionBudget()
        self.breaker = breaker
        self.state_path = None if state_path is None else Path(state_path)
        self._lock = threading.RLock()
        self._command_ids: set[str] = set()
        self._inflight: set[str] = set()
        self._window_started_ns: int | None = None
        self._window_commands = 0
        self._window_bytes = 0
        if self.state_path is not None and self.state_path.exists():
            self._restore()

    def authorize(self, command: BackendActionCommand, *, now_ns: int) -> GateDecision:
        with self._lock:
            reason = self._validate(command, now_ns)
            if reason is not None:
                return GateDecision(False, reason)
            self._rotate_window(now_ns)
            self._command_ids.add(command.command_id)
            self._inflight.add(command.command_id)
            self._window_commands += 1
            snapshot = self.binding_registry.snapshot(command.binding_id)
            amount = _command_bytes(snapshot)
            if amount is None:  # _validate already rejects this, retain fail-closed behavior.
                self._command_ids.remove(command.command_id)
                self._inflight.remove(command.command_id)
                self._window_commands -= 1
                return GateDecision(False, "physical_size_unknown")
            self._window_bytes += amount
            self._persist()
            return GateDecision(True, "admitted")

    def complete(self, command_id: str) -> bool:
        with self._lock:
            if command_id not in self._inflight:
                return False
            self._inflight.remove(command_id)
            self._persist()
            return True

    def _validate(self, command: BackendActionCommand, now_ns: int) -> str | None:
        if command.run_id != self.run_id:
            return "run_mismatch"
        if self.capabilities is None or not self.capabilities.validate_for_execution(self.run_id, self.endpoint):
            return "capability_preflight"
        if self.breaker is not None and not self.breaker.allow_dispatch(now_ns=now_ns):
            return "circuit_open"
        deadline = command.metadata.get("deadline_ns")
        try:
            deadline_ns = int(deadline)
        except (TypeError, ValueError):
            return "deadline_missing"
        if deadline_ns < now_ns:
            return "deadline_expired"
        try:
            snapshot = self.binding_registry.snapshot(command.binding_id)
        except ValueError:
            return "binding_not_found"
        if snapshot.get("lifecycle") == "replaced":
            return "binding_replaced"
        current = self.binding_registry.current_binding(
            binding_id=command.binding_id, binding_generation=command.binding_generation,
            request_id=command.request_id, object_key=command.object_key, object_level=command.object_level,
        )
        if current is None:
            status = getattr(self.binding_registry, "binding_status", None)
            if callable(status) and status(command.binding_id) == "replaced":
                return "binding_replaced"
            return "binding_not_found"
        if current.backend_object_id != command.backend_object_id:
            return "binding_mismatch"
        if snapshot["pending_io"] or snapshot["pending_operations"] or snapshot["action_reservation"]:
            return "pending_action_conflict"
        if snapshot["active_request_ids"]:
            return "active_binding_conflict"
        if snapshot["pin_count"]:
            return "pinned_binding"
        if command.command_id in self._command_ids:
            return "duplicate_command"
        amount = _command_bytes(snapshot)
        if amount is None:
            return "physical_size_unknown"
        if amount > self.budget.max_bytes_per_command:
            return "byte_budget"
        self._rotate_window(now_ns)
        if self._window_commands >= self.budget.max_commands_per_window or self._window_bytes + amount > self.budget.max_bytes_per_window:
            return "rate_budget"
        if len(self._inflight) >= self.budget.max_concurrent:
            return "concurrency_budget"
        return None

    def _rotate_window(self, now_ns: int) -> None:
        if self._window_started_ns is None or now_ns - self._window_started_ns >= self.budget.window_ns:
            self._window_started_ns = now_ns
            self._window_commands = 0
            self._window_bytes = 0

    def _restore(self) -> None:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != RUNTIME_GATE_SCHEMA or payload.get("run_id") != self.run_id or payload.get("budget") != asdict(self.budget):
            raise ValueError("runtime gate checkpoint does not match configuration")
        self._command_ids = set(payload.get("command_ids") or [])
        self._inflight = set(payload.get("inflight") or [])
        self._window_started_ns = payload.get("window_started_ns")
        self._window_commands = int(payload.get("window_commands", 0))
        self._window_bytes = int(payload.get("window_bytes", 0))

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"schema": RUNTIME_GATE_SCHEMA, "run_id": self.run_id, "budget": asdict(self.budget),
            "command_ids": sorted(self._command_ids), "inflight": sorted(self._inflight),
            "window_started_ns": self._window_started_ns, "window_commands": self._window_commands, "window_bytes": self._window_bytes}
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.state_path)


def _command_bytes(snapshot: dict[str, Any]) -> int | None:
    """Use the observed physical binding size, never policy-provided metadata."""
    try:
        amount = int(snapshot.get("size_bytes"))
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None
