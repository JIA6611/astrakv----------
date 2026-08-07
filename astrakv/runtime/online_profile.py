"""Deterministic, checkpointable consumption of verified backend events."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from astrakv.runtime.backend_hook import BackendHookEvent, HookAction
from astrakv.runtime.eviction import OfflineEvictionDecision, RuntimeActionResult
from astrakv.runtime.prefix_learning import RuntimePrefixIndex


ONLINE_PROFILE_SCHEMA = "astrakv-online-profile-v1"


class OnlineProfileStore:
    """Materialize hook events once, preserving replay-safe online state."""

    def __init__(self, *, run_id: str, checkpoint_path: Path | str | None = None) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.run_id = run_id
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._lock = threading.RLock()
        self._event_fingerprints: dict[str, str] = {}
        self._dispatch_fingerprints: dict[str, str] = {}
        self._events: list[BackendHookEvent] = []
        self._dispatches: list[dict[str, Any]] = []
        self._objects: dict[str, dict[str, Any]] = {}
        self._prefix_index = RuntimePrefixIndex()
        self._last_event_id: str | None = None
        self._controller_state: dict[str, Any] = {
            "execution_enabled": False,
            "dispatch_count": 0,
            "dispatch_status_counts": {},
            "last_decision_id": None,
            "last_decision_source": "heuristic",
            "last_fallback_mode": "none",
            "last_profile_guard": "none",
            "last_profile_guard_reason": "",
            "last_resolved_action": None,
            "last_resolved_tier": "unknown",
            "last_dispatch_status": None,
            "last_dispatch_message": None,
            "last_command_id": None,
            "last_receipt_id": None,
            "last_runtime_event_id": None,
            "last_dispatch_timestamp_ns": None,
            "last_rejection_reason": None,
            "last_dispatch_origin": "",
            "last_prefetch_skip_reason": "",
            "breaker": {},
        }
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._restore()

    @property
    def events(self) -> tuple[BackendHookEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def consume(self, event: BackendHookEvent) -> bool:
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match online profile run_id")
        fingerprint = _fingerprint(event.to_record())
        with self._lock:
            previous = self._event_fingerprints.get(event.event_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError("conflicting replay for event_id")
                return False
            self._event_fingerprints[event.event_id] = fingerprint
            self._events.append(event)
            self._prefix_index.observe(event)
            self._last_event_id = event.event_id
            state = self._objects.setdefault(
                event.backend_object_id,
                _new_object_state(event.backend_object_id, binding_generation=event.binding_generation),
            )
            state["binding_generation"] = event.binding_generation
            state["event_count"] += 1
            state["bytes"] += max(0, event.bytes or 0)
            state["current_tier"] = _prefer_known_tier(event.tier_after, state["current_tier"])
            state["last_request_id"] = event.request_id
            state["last_object_key"] = event.object_key
            state["last_object_level"] = event.object_level.value
            state["last_timestamp_ns"] = max(state["last_timestamp_ns"], event.timestamp_ns)
            state["actions"][event.action.value] = state["actions"].get(event.action.value, 0) + 1
            state["last_action"] = event.action.value
            state["last_arrival_index"] = max(
                int(state["last_arrival_index"]),
                int(event.metadata.get("arrival_index") or 0),
            )
            if "reuse_ratio" in event.metadata:
                try:
                    state["last_reuse_ratio_hint"] = float(event.metadata.get("reuse_ratio"))
                except (TypeError, ValueError):
                    state["last_reuse_ratio_hint"] = None
            if event.metadata.get("load_target_id"):
                state["last_load_target_id"] = str(event.metadata["load_target_id"])
            if event.metadata.get("load_target_state"):
                state["last_load_target_state"] = str(event.metadata["load_target_state"])
            if event.metadata.get("load_target_consumed_at_ns") not in (None, "", "None"):
                try:
                    state["last_load_target_consumed_at_ns"] = int(event.metadata["load_target_consumed_at_ns"])
                except (TypeError, ValueError):
                    pass
            if event.metadata.get("runtime_reqmeta_id"):
                state["last_runtime_reqmeta_id"] = str(event.metadata["runtime_reqmeta_id"])
                if (
                    event.action in {HookAction.CACHE_STORE, HookAction.RELEASE, HookAction.CACHE_HIT, HookAction.CACHE_MISS}
                    and not str(state.get("owner_runtime_reqmeta_id") or "")
                ):
                    state["owner_runtime_reqmeta_id"] = str(event.metadata["runtime_reqmeta_id"])
            if event.action in {HookAction.CACHE_HIT, HookAction.CACHE_MISS, HookAction.CACHE_LOAD, HookAction.CACHE_STORE}:
                state["last_access_time_ns"] = max(state["last_access_time_ns"], event.timestamp_ns)
            if event.action == HookAction.CACHE_STORE and event.status == "submitted":
                state["last_request_submit_timestamp_ns"] = max(
                    int(state["last_request_submit_timestamp_ns"]),
                    event.timestamp_ns,
                )
            if event.action == HookAction.CACHE_HIT:
                state["cache_hits"] += 1
                state["reuse_count"] += 1
                if state["evicted_since_reaccess"]:
                    state["eviction_reaccess_count"] += 1
                    state["evicted_since_reaccess"] = False
                state["prefetched_since_access"] = False
                state["active_reference_count"] += 1
            elif event.action == HookAction.CACHE_MISS:
                state["cache_misses"] += 1
            elif event.action == HookAction.CACHE_STORE:
                state["cache_stores"] += 1
                if event.status == "submitted":
                    state["request_count"] += 1
                elif event.status in {"completed", "ok", "executed"}:
                    state["active_reference_count"] += 1
            elif event.action == HookAction.CACHE_LOAD and event.status in {"completed", "ok", "executed"}:
                state["reuse_count"] += 1
                state["active_reference_count"] += 1
                if str(event.metadata.get("load_target_id") or ""):
                    state["last_load_target_id"] = str(event.metadata["load_target_id"])
                    state["last_load_target_state"] = "consumed"
                    state["last_load_target_consumed_at_ns"] = event.timestamp_ns
                if state["evicted_since_reaccess"]:
                    state["eviction_reaccess_count"] += 1
                    state["evicted_since_reaccess"] = False
                state["prefetched_since_access"] = False
                load_latency_ns = event.metadata.get("load_latency_ns")
                try:
                    latency = float(load_latency_ns)
                except (TypeError, ValueError):
                    latency = 0.0
                if latency > 0:
                    current_ema = float(state["load_latency_ema"])
                    state["load_latency_ema"] = latency if current_ema <= 0 else ((0.8 * current_ema) + (0.2 * latency))
            elif event.action == HookAction.PREFETCH and event.status in {"completed", "ok", "executed"}:
                state["prefetch_success_count"] += 1
                state["prefetched_since_access"] = True
            elif event.action == HookAction.CACHE_LOAD and event.status not in {"completed", "ok", "executed"}:
                if str(event.metadata.get("load_target_id") or ""):
                    state["last_load_target_id"] = str(event.metadata["load_target_id"])
                load_target_state = str(event.metadata.get("load_target_state") or "")
                if load_target_state:
                    state["last_load_target_state"] = load_target_state
                elif str(event.metadata.get("failure_reason") or "") == "load_target_missing_or_already_consumed":
                    state["last_load_target_state"] = "unavailable"
                if event.metadata.get("load_target_consumed_at_ns") not in (None, "", "None"):
                    try:
                        state["last_load_target_consumed_at_ns"] = int(event.metadata["load_target_consumed_at_ns"])
                    except (TypeError, ValueError):
                        pass
            elif event.action in {HookAction.OFFLOAD, HookAction.EVICT, HookAction.DROP} and event.status in {"completed", "ok", "executed"}:
                state["evicted_since_reaccess"] = True
                if state["prefetched_since_access"]:
                    state["prefetch_waste_count"] += 1
                    state["prefetched_since_access"] = False
                if event.action is HookAction.EVICT and state["current_tier"] == "none":
                    state["active_reference_count"] = 0
            elif event.action == HookAction.RELEASE and event.status in {"completed", "ok", "executed"}:
                state["active_reference_count"] = max(0, int(state["active_reference_count"]) - 1)
            return True

    def record_dispatch(
        self,
        decision: OfflineEvictionDecision,
        result: RuntimeActionResult,
        *,
        execution_enabled: bool,
        breaker_state: dict[str, Any] | None = None,
    ) -> bool:
        event = result.event
        receipt = result.receipt
        command_id = (
            str(receipt.command_id)
            if receipt is not None
            else None if event is None else str(event.metadata.get("command_id") or "") or None
        )
        receipt_id = (
            str(receipt.receipt_id)
            if receipt is not None
            else None if event is None else str(event.metadata.get("receipt_id") or "") or None
        )
        backend_object_id = (
            str(receipt.backend_object_id)
            if receipt is not None
            else None if event is None else str(event.metadata.get("backend_object_id") or "") or None
        )
        if not backend_object_id:
            backend_object_id = str(decision.metadata.get("backend_object_id") or "") or None
        entry = {
            "decision_id": decision.decision_id,
            "request_id": decision.request_id,
            "object_key": decision.object_key,
            "object_level": decision.object_level.value,
            "predicted_action": decision.predicted_action,
            "target_tier": decision.target_tier,
            "decision_source": str(decision.metadata.get("decision_source") or "heuristic"),
            "fallback_mode": str(decision.metadata.get("fallback_mode") or "none"),
            "profile_guard": str(decision.metadata.get("profile_guard") or "none"),
            "profile_guard_reason": str(decision.metadata.get("profile_guard_reason") or ""),
            "resolved_action": (
                result.event.actual_action if event is not None else decision.predicted_action
            ),
            "resolved_tier": (
                event.tier_after if event is not None else decision.target_tier
            ),
            "status": result.status,
            "message": result.message,
            "execution_enabled": bool(execution_enabled),
            "breaker": dict(breaker_state or {}),
            "command_id": command_id,
            "receipt_id": receipt_id,
            "runtime_event_id": None if event is None else event.runtime_event_id,
            "backend_object_id": backend_object_id,
            "receipt_status": receipt.status if receipt is not None else None if event is None else event.metadata.get("receipt_status"),
            "rejection_reason": None if receipt is None else receipt.rejection_reason or None,
            "timestamp_ns": receipt.timestamp_ns if receipt is not None else None if event is None else event.timestamp_ns,
            "dispatch_origin": str(decision.metadata.get("dispatch_origin") or ""),
            "prefetch_skip_reason": str(decision.metadata.get("prefetch_skip_reason") or ""),
        }
        fingerprint = _fingerprint(entry)
        with self._lock:
            previous = self._dispatch_fingerprints.get(decision.decision_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError("conflicting replay for decision_id")
                return False
            self._dispatch_fingerprints[decision.decision_id] = fingerprint
            self._dispatches.append(entry)
            controller = self._controller_state
            controller["execution_enabled"] = bool(execution_enabled)
            controller["dispatch_count"] += 1
            counts = controller["dispatch_status_counts"]
            counts[result.status] = counts.get(result.status, 0) + 1
            controller["last_decision_id"] = decision.decision_id
            controller["last_decision_source"] = entry["decision_source"]
            controller["last_fallback_mode"] = entry["fallback_mode"]
            controller["last_profile_guard"] = entry["profile_guard"]
            controller["last_profile_guard_reason"] = entry["profile_guard_reason"]
            controller["last_resolved_action"] = entry["resolved_action"]
            controller["last_resolved_tier"] = entry["resolved_tier"]
            controller["last_dispatch_status"] = result.status
            controller["last_dispatch_message"] = result.message
            controller["last_command_id"] = command_id
            controller["last_receipt_id"] = receipt_id
            controller["last_runtime_event_id"] = entry["runtime_event_id"]
            controller["last_dispatch_timestamp_ns"] = entry["timestamp_ns"]
            controller["last_rejection_reason"] = entry["rejection_reason"]
            controller["last_dispatch_origin"] = entry["dispatch_origin"]
            controller["last_prefetch_skip_reason"] = entry["prefetch_skip_reason"]
            controller["breaker"] = dict(breaker_state or {})
            if backend_object_id:
                state = self._objects.setdefault(
                    backend_object_id,
                    _new_object_state(
                        backend_object_id,
                        binding_generation=(
                            receipt.binding_generation if receipt is not None
                            else (
                                decision.metadata.get("binding_generation")
                                if event is None
                                else event.metadata.get("binding_generation", decision.metadata.get("binding_generation"))
                            )
                        ),
                    ),
                )
                state["dispatch_count"] += 1
                status_counts = state["dispatch_status_counts"]
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
                state["last_decision_id"] = decision.decision_id
                state["last_action"] = decision.predicted_action
                state["last_resolved_action"] = entry["resolved_action"]
                state["last_resolved_tier"] = entry["resolved_tier"]
                state["last_receipt_status"] = entry["receipt_status"]
                state["last_rejection_reason"] = entry["rejection_reason"]
                state["last_decision_source"] = entry["decision_source"]
                state["last_fallback_mode"] = entry["fallback_mode"]
                state["last_profile_guard"] = entry["profile_guard"]
                state["last_profile_guard_reason"] = entry["profile_guard_reason"]
                state["last_command_id"] = command_id
                state["last_receipt_id"] = receipt_id
                state["last_runtime_event_id"] = entry["runtime_event_id"]
                state["last_dispatch_origin"] = entry["dispatch_origin"]
                state["last_prefetch_skip_reason"] = entry["prefetch_skip_reason"]
                state["last_request_id"] = decision.request_id
                state["last_object_key"] = decision.object_key
                state["last_object_level"] = decision.object_level.value
                if entry["timestamp_ns"] is not None:
                    state["last_timestamp_ns"] = max(state["last_timestamp_ns"], int(entry["timestamp_ns"]))
                if receipt is not None:
                    state["binding_generation"] = receipt.binding_generation
                elif event is not None:
                    state["binding_generation"] = event.metadata.get("binding_generation", state["binding_generation"])
                    state["current_tier"] = _prefer_known_tier(event.tier_after, state["current_tier"])
                    state["bytes"] = max(state["bytes"], max(0, event.bytes or 0))
                elif decision.metadata.get("binding_generation") is not None:
                    state["binding_generation"] = decision.metadata.get("binding_generation")
                if decision.decision_index is not None:
                    state["last_arrival_index"] = max(int(state["last_arrival_index"]), int(decision.decision_index))
            return True

    def object_state(self, backend_object_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._objects.get(backend_object_id)
            return None if value is None else json.loads(json.dumps(value, sort_keys=True))

    def controller_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._controller_state, sort_keys=True))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": ONLINE_PROFILE_SCHEMA,
                "run_id": self.run_id,
                "event_count": len(self._events),
                "dispatch_count": len(self._dispatches),
                "last_event_id": self._last_event_id,
                "objects": {key: self._objects[key] for key in sorted(self._objects)},
                "prefix_profiles": self._prefix_index.to_records(),
                "controller_state": json.loads(json.dumps(self._controller_state, sort_keys=True)),
            }

    def prefix_profile(self, prefix_key: str) -> dict[str, Any] | None:
        with self._lock:
            profile = self._prefix_index.profile_for(prefix_key)
            return None if profile is None else profile.to_record()

    def checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        with self._lock:
            payload = self.snapshot() | {
                "events": [event.to_record() for event in self._events],
                "event_fingerprints": dict(sorted(self._event_fingerprints.items())),
                "dispatches": list(self._dispatches),
                "dispatch_fingerprints": dict(sorted(self._dispatch_fingerprints.items())),
            }
            _atomic_json_write(self.checkpoint_path, payload)

    def _restore(self) -> None:
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("schema") != ONLINE_PROFILE_SCHEMA or payload.get("run_id") != self.run_id:
            raise ValueError("online profile checkpoint does not match run")
        events = [BackendHookEvent.from_record(record) for record in payload.get("events", [])]
        self._event_fingerprints = dict(payload.get("event_fingerprints") or {})
        if len(self._event_fingerprints) != len(events):
            raise ValueError("online profile checkpoint has invalid event index")
        self._events = events
        self._dispatches = list(payload.get("dispatches") or [])
        self._dispatch_fingerprints = dict(payload.get("dispatch_fingerprints") or {})
        if len(self._dispatch_fingerprints) != len(self._dispatches):
            raise ValueError("online profile checkpoint has invalid dispatch index")
        self._objects = dict(payload.get("objects") or {})
        self._prefix_index = RuntimePrefixIndex.from_records(payload.get("prefix_profiles") or ())
        self._last_event_id = payload.get("last_event_id")
        restored_controller = dict(payload.get("controller_state") or {})
        self._controller_state = {
            **self._controller_state,
            **restored_controller,
            "dispatch_status_counts": dict(restored_controller.get("dispatch_status_counts") or {}),
            "breaker": dict(restored_controller.get("breaker") or {}),
        }


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prefer_known_tier(candidate: Any, current: Any) -> str:
    normalized = str(candidate or "")
    if normalized and normalized != "unknown":
        return normalized
    return str(current or "unknown")


def _new_object_state(backend_object_id: str, *, binding_generation: Any) -> dict[str, Any]:
    return {
        "backend_object_id": backend_object_id,
        "binding_generation": binding_generation,
        "event_count": 0,
        "request_count": 0,
        "reuse_count": 0,
        "bytes": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_stores": 0,
        "current_tier": "unknown",
        "active_reference_count": 0,
        "last_access_time_ns": 0,
        "load_latency_ema": 0.0,
        "prefetch_success_count": 0,
        "prefetch_waste_count": 0,
        "eviction_reaccess_count": 0,
        "last_request_id": "",
        "last_object_key": "",
        "last_object_level": "",
        "last_arrival_index": 0,
        "dispatch_count": 0,
        "dispatch_status_counts": {},
        "last_decision_id": None,
        "last_action": None,
        "last_resolved_action": None,
        "last_resolved_tier": "unknown",
        "last_receipt_status": None,
        "last_rejection_reason": None,
        "last_dispatch_origin": "",
        "last_prefetch_skip_reason": "",
        "last_decision_source": "heuristic",
        "last_fallback_mode": "none",
        "last_profile_guard": "none",
        "last_profile_guard_reason": "",
        "last_reuse_ratio_hint": None,
        "last_load_target_id": "",
        "last_runtime_reqmeta_id": "",
        "last_load_target_state": "",
        "last_load_target_consumed_at_ns": 0,
        "owner_runtime_reqmeta_id": "",
        "last_command_id": None,
        "last_receipt_id": None,
        "last_runtime_event_id": None,
        "last_request_submit_timestamp_ns": 0,
        "actions": {},
        "last_timestamp_ns": 0,
        "prefetched_since_access": False,
        "evicted_since_reaccess": False,
    }


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)
