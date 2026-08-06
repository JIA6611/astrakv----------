"""Online profile ingestion and explicit, receipt-backed policy dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from astrakv.kv_cache.metadata import MemoryTier
from astrakv.kv_cache.partial_load import PartialKVLoadTarget, TokenSpan
from astrakv.runtime.backend_bridge import OnlineBackendBridge
from astrakv.runtime.backend_hook import BackendHookEvent, HookAction
from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision, RuntimeActionResult, RuntimeEvictionEvent
from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.runtime.online_profile import OnlineProfileStore
from astrakv.runtime.prediction_sidecar import PredictionSidecarIndex, SidecarPrediction
from astrakv.runtime.runtime_plan import (
    RuntimeActionKind,
    RuntimeActionPlan,
    RuntimeLayerRange,
    RuntimeObjectKind,
    RuntimePlacementTier,
    RuntimeProfileGuard,
    RuntimeTokenRange,
)
from astrakv.scheduler.decision import LoadRecomputeConfig, LoadRecomputePlanner
from astrakv.runtime.trace_schema import TraceEvent


@dataclass(frozen=True, slots=True)
class OnlinePolicyControllerConfig:
    hot_reuse_threshold: float = 0.50
    cold_reuse_threshold: float = 0.12
    prefetch_waste_tolerance: int = 1
    load_latency_hot_threshold_ms: float = 10.0
    memory_pressure: float = 0.0
    deadline_ms: float = 120.0


class OnlinePolicyController:
    """Keeps online observations, advisory decisions, and execution receipts separate."""

    def __init__(
        self, *, run_id: str, workload_id: str, bridge: OnlineBackendBridge,
        profile_store: OnlineProfileStore | None = None,
        config: OnlinePolicyControllerConfig | None = None,
        prediction_source: PredictionSidecarIndex | None = None,
        profile_db: ProfileDB | None = None,
        scheduler_hints: SchedulerHintIndex | None = None,
    ) -> None:
        self.run_id = run_id
        self.workload_id = workload_id
        self.bridge = bridge
        self.execution_enabled = False
        self.profile_store = profile_store or OnlineProfileStore(run_id=run_id)
        self.config = config or OnlinePolicyControllerConfig()
        self.prediction_source = prediction_source
        self.trace_events: list[TraceEvent] = []
        self.decisions: list[OfflineEvictionDecision] = []
        self.action_plans: list[RuntimeActionPlan] = []
        self.profile_db = profile_db or ProfileDB()
        self.observed_profile_db = ProfileDB()
        self.scheduler_hints = scheduler_hints
        self._load_planner = LoadRecomputePlanner(
            LoadRecomputeConfig(
                memory_pressure=self.config.memory_pressure,
                deadline_ms=self.config.deadline_ms,
            )
        )

    def ingest(self, event: BackendHookEvent) -> bool:
        if event.run_id != self.run_id or not self.bridge.observe_event(event):
            return False
        if not self.profile_store.consume(event):
            return False
        self.trace_events.append(TraceEvent(
            event_type=event.action.value,
            category="kv" if event.action.value.startswith("cache_") else "placement",
            source="verified_backend_hook",
            status=event.status,
            timestamp=str(event.timestamp_ns),
            request_id=event.request_id,
            chunk_id=event.backend_object_id,
            cache_key=event.object_key,
            tier=event.tier_after,
            bytes=event.bytes,
            metadata={
                "run_id": self.run_id,
                "event_id": event.event_id,
                "object_level": event.object_level.value,
                "backend_object_id": event.backend_object_id,
                "verified_backend_hook": True,
                **dict(event.metadata),
            },
            latency_ms=_load_latency_ms(event),
        ))
        self.observed_profile_db = ProfileDB.from_trace_events(self.trace_events, workload_id=self.workload_id)
        self.profile_store.checkpoint()
        return True

    def propose_for(
        self,
        object_key: str,
        object_level: ObjectLevel,
        *,
        request_id: str | None = None,
        binding_id: str | None = None,
    ) -> OfflineEvictionDecision:
        binding = self.bridge.binding_for(
            object_level,
            object_key,
            request_id=request_id,
            binding_id=binding_id,
        )
        if binding is None:
            raise ValueError("cannot propose action for an unbound object")
        object_state = self.profile_store.object_state(binding.backend_object_id) or {}
        current_tier = _resolve_current_tier(object_state, binding)
        profile = (
            self.observed_profile_db.get_chunk(binding.backend_object_id, workload_id=self.workload_id)
            or self.profile_db.get_chunk(binding.backend_object_id, workload_id=self.workload_id)
        )
        execution_actions = {} if binding.execution_spec is None else dict(binding.execution_spec.actions)
        active_refs = _as_int(object_state.get("active_reference_count"))
        request_count = _as_int(object_state.get("request_count"))
        reuse_frequency = _reuse_frequency(profile, object_state)
        policy_reuse_frequency = _policy_reuse_frequency(profile, object_state, reuse_frequency)
        prefetch_hit_rate = _prefetch_hit_rate(profile, object_state)
        prefetch_waste = _as_int(object_state.get("prefetch_waste_count"))
        load_latency_ms = _profile_load_latency_ms(profile, object_state)
        load_target_id = _load_target_id(binding, object_state)
        load_target_reqmeta_id = _load_target_runtime_reqmeta_id(binding)
        owner_reqmeta_id = str(object_state.get("owner_runtime_reqmeta_id") or "")
        same_request_load_target = bool(
            load_target_id
            and load_target_reqmeta_id
            and owner_reqmeta_id
            and load_target_reqmeta_id == owner_reqmeta_id
        )
        if same_request_load_target:
            # A request's own store target describes its current write, not a
            # future revisit. Treating it as load demand creates a load loop.
            load_target_id = ""
        load_ready = _action_ready(execution_actions, "load")
        prefetch_ready = _action_ready(execution_actions, "prefetch")
        offload_ready = _action_ready(execution_actions, "offload")
        evict_ready = _action_ready(execution_actions, "evict")
        drop_ready = _action_ready(execution_actions, "drop")
        prediction = _prediction_for_binding(
            self.prediction_source,
            binding=binding,
            now_ns=time.time_ns(),
        )
        scheduler_hint = _scheduler_hint_for_binding(
            self.scheduler_hints,
            binding=binding,
        )
        profile_guard = _build_profile_guard(
            self.profile_db,
            binding=binding,
            object_state=object_state,
            workload_id=self.workload_id,
        )
        prefix_prefetch_candidate = _prefix_prefetch_candidate(
            binding=binding,
            profile=profile,
            object_state=object_state,
            prefetch_ready=prefetch_ready,
            profile_guard=profile_guard,
        )
        prediction_gate_reason = _prediction_prefetch_gate_reason(
            bridge=self.bridge,
            binding=binding,
            prediction=prediction,
            prefetch_ready=prefetch_ready,
            same_request_load_target=same_request_load_target,
            prefetch_waste=prefetch_waste,
            prefetch_waste_tolerance=self.config.prefetch_waste_tolerance,
        )
        breaker = self.profile_store.controller_state().get("breaker") or {}
        breaker_state = str(breaker.get("state") or "closed")
        breaker_open = breaker_state in {"open", "half_open"}
        load_is_worthwhile = _load_is_worthwhile(
            request_count=request_count,
            load_target_present=bool(load_target_id),
            reuse_frequency=policy_reuse_frequency,
            load_latency_ms=load_latency_ms,
            hot_reuse_threshold=self.config.hot_reuse_threshold,
            load_latency_hot_threshold_ms=self.config.load_latency_hot_threshold_ms,
        )
        action = "drop"
        target_tier = "unknown"
        reason = "online profile: no stronger runtime action candidate was verified, so bookkeeping falls back to drop"

        scheduler_action = "" if scheduler_hint is None else str(scheduler_hint.action or "")

        if breaker_open:
            if current_tier == "gpu":
                action = "keep"
                target_tier = "gpu"
                reason = f"online profile: breaker is {breaker_state}; keep the GPU-resident object in place"
            else:
                planner = self._load_planner.decide_profile(_planner_profile(binding, object_state, profile))
                if planner.action.value == "recompute":
                    action = "recompute"
                    reason = f"online profile: breaker is {breaker_state}; recompute is safer than backend IO"
                else:
                    action = "defer"
                    reason = f"online profile: breaker is {breaker_state}; defer backend dispatch until the breaker closes"
        elif current_tier == "gpu":
            if scheduler_hint is not None and scheduler_action == "keep":
                action = "keep"
                target_tier = "gpu"
                reason = f"online policy: scheduler hint requested keep ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "drop" and drop_ready:
                action = "drop"
                reason = f"online policy: scheduler hint requested drop ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "defer":
                action = "defer"
                reason = f"online policy: scheduler hint requested defer ({scheduler_hint.reason})"
            elif active_refs > 0 or policy_reuse_frequency >= self.config.hot_reuse_threshold:
                action = "keep"
                target_tier = "gpu"
                reason = "online profile: GPU-resident object is still hot or actively referenced, so it should stay resident"
            elif drop_ready:
                action = "drop"
                reason = "online profile: GPU-resident object is cold and unreferenced, so a verified drop is preferred"
            else:
                action = "defer"
                reason = "online profile: GPU-resident object is cold but drop is not currently ready"
        elif current_tier == "cpu":
            if scheduler_hint is not None and scheduler_action == "keep":
                action = "keep"
                target_tier = "cpu"
                reason = f"online policy: scheduler hint requested keep ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "offload" and offload_ready:
                action = "offload"
                target_tier = "ssd"
                reason = f"online policy: scheduler hint requested offload ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "drop" and drop_ready:
                action = "drop"
                reason = f"online policy: scheduler hint requested drop ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "defer":
                action = "defer"
                reason = f"online policy: scheduler hint requested defer ({scheduler_hint.reason})"
            elif active_refs == 0 and offload_ready:
                action = "offload"
                target_tier = "ssd"
                reason = "online profile: CPU-resident object is released with zero active refs, so offload is ready"
            else:
                action = "keep"
                target_tier = "cpu"
                reason = "online profile: keep the CPU-resident object until a verified lower-tier action is ready"
        elif current_tier == "ssd":
            if scheduler_hint is not None and scheduler_action == "load" and load_target_id and load_ready:
                action = "load"
                target_tier = "gpu"
                reason = f"online policy: scheduler hint requested load ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "recompute" and profile_guard.recompute_allowed:
                action = "recompute"
                reason = f"online policy: scheduler hint requested recompute ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "prefetch" and prefix_prefetch_candidate and prefetch_ready:
                action = "prefetch"
                target_tier = "cpu"
                reason = f"online policy: scheduler hint requested prefix prefetch ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "drop" and drop_ready:
                action = "drop"
                reason = f"online policy: scheduler hint requested drop ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "evict" and evict_ready:
                action = "evict"
                target_tier = "ssd"
                reason = f"online policy: scheduler hint requested evict ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "defer":
                action = "defer"
                reason = f"online policy: scheduler hint requested defer ({scheduler_hint.reason})"
            elif load_target_id and load_ready and load_is_worthwhile:
                action = "load"
                target_tier = "gpu"
                reason = "online profile: SSD-resident object has a dynamic load target and load is more valuable than waiting for recompute"
            elif prediction_gate_reason is None and prediction is not None:
                action = "prefetch"
                target_tier = "cpu"
                reason = (
                    "online profile: sidecar exact-next advisory passed runtime gates, "
                    "so CPU prefetch is preferred before future demand"
                )
            elif prefix_prefetch_candidate and prefetch_ready:
                action = "prefetch"
                target_tier = "cpu"
                reason = "online profile: prefix reuse profile nominated this SSD object for prefix prefetch"
            elif prediction is not None:
                action = "defer"
                reason = (
                    "online profile: sidecar exact-next advisory is present but not runtime-ready, "
                    f"so dispatch stays advisory-only ({prediction_gate_reason})"
                )
            elif self.prediction_source is not None:
                action = "defer"
                reason = "online profile: no exact-next advisory matched this SSD object, so speculative dispatch stays advisory-only"
            elif prefetch_ready and not same_request_load_target and (
                active_refs > 0
                or
                policy_reuse_frequency >= self.config.hot_reuse_threshold
                or load_latency_ms >= self.config.load_latency_hot_threshold_ms
            ) and prefetch_waste < self.config.prefetch_waste_tolerance:
                action = "prefetch"
                target_tier = "cpu"
                reason = "online profile: SSD-resident object looks warm but is not ready for direct load, so prefetch is preferred"
            elif active_refs == 0 and evict_ready and (
                (policy_reuse_frequency <= self.config.cold_reuse_threshold and prefetch_hit_rate > 0.0)
                or prefetch_waste > 0
            ):
                action = "evict"
                target_tier = "ssd"
                reason = "online profile: SSD-resident object already served its warm revisit and now looks cold enough to evict"
            elif active_refs == 0 and drop_ready and (
                policy_reuse_frequency <= self.config.cold_reuse_threshold
                and prefetch_hit_rate <= 0.0
                and prefetch_waste == 0
            ):
                action = "drop"
                reason = "online profile: SSD-resident object is cold, unreferenced, and has no prefetch waste, so drop is preferred"
            elif active_refs == 0 and evict_ready and (
                policy_reuse_frequency <= self.config.cold_reuse_threshold
                or prefetch_waste > 0
            ):
                action = "evict"
                target_tier = "ssd"
                reason = "online profile: SSD-resident object is cold or has accumulated prefetch waste, so evict is preferred"
            elif active_refs == 0 and _recompute_is_preferred(
                profile=profile,
                load_target_present=bool(load_target_id),
                load_is_worthwhile=load_is_worthwhile,
                policy_reuse_frequency=policy_reuse_frequency,
                hot_reuse_threshold=self.config.hot_reuse_threshold,
                cold_reuse_threshold=self.config.cold_reuse_threshold,
                prefetch_waste=prefetch_waste,
            ) and profile_guard.recompute_allowed:
                action = "recompute"
                reason = "online profile: low nonzero reuse has no load target and recompute is preferable to backend IO"
            else:
                action = "defer"
                reason = "online profile: SSD-resident object has no immediately safe verified action"
        elif current_tier == "none":
            planner = self._load_planner.decide_profile(_planner_profile(binding, object_state, profile))
            if scheduler_hint is not None and scheduler_action == "load" and load_target_id and load_ready:
                action = "load"
                target_tier = "gpu"
                reason = f"online policy: scheduler hint requested load ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "recompute" and profile_guard.recompute_allowed:
                action = "recompute"
                reason = f"online policy: scheduler hint requested recompute ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "drop" and drop_ready:
                action = "drop"
                reason = f"online policy: scheduler hint requested drop ({scheduler_hint.reason})"
            elif scheduler_hint is not None and scheduler_action == "defer":
                action = "defer"
                reason = f"online policy: scheduler hint requested defer ({scheduler_hint.reason})"
            elif planner.action.value == "load" and load_target_id and load_ready:
                action = "load"
                target_tier = "gpu"
                reason = "online profile: planner prefers runtime load and a dynamic load target is available"
            elif planner.action.value == "recompute" and profile_guard.recompute_allowed:
                action = "recompute"
                reason = planner.reason
            elif planner.action.value == "defer":
                action = "defer"
                reason = planner.reason
            elif drop_ready:
                action = "drop"
                reason = planner.reason or "online profile: no verified residency remains; drop bookkeeping state"
            else:
                action = "defer"
                reason = "online profile: no residency remains and no verified terminal action is ready"
        else:
            action = "defer"
            reason = "online profile: tier is unknown after release; keep the state advisory-only until residency is observed again"
        decision = OfflineEvictionDecision(
            run_id=self.run_id,
            decision_id=f"online-{len(self.decisions)}-{binding.binding_id}",
            request_id=binding.request_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
            predicted_action=action,
            target_tier=target_tier,
            decision_time_ns=time.time_ns(),
            reason=reason,
            metadata={
                "binding_id": binding.binding_id,
                "binding_generation": binding.binding_generation,
                "backend_object_id": binding.backend_object_id,
                "profile_workload_id": self.workload_id,
                "current_tier": current_tier,
                "active_reference_count": active_refs,
                "request_count": request_count,
                "reuse_frequency": reuse_frequency,
                "policy_reuse_frequency": policy_reuse_frequency,
                "prefetch_hit_rate": prefetch_hit_rate,
                "prefetch_waste_count": prefetch_waste,
                "load_latency_ms": load_latency_ms,
                "breaker_state": breaker_state,
                "load_target_present": bool(load_target_id),
                "load_target_runtime_reqmeta_id": load_target_reqmeta_id,
                "owner_runtime_reqmeta_id": owner_reqmeta_id,
                "same_request_load_target": same_request_load_target,
                "load_ready": load_ready,
                "prefetch_ready": prefetch_ready,
                "offload_ready": offload_ready,
                "evict_ready": evict_ready,
                "drop_ready": drop_ready,
                "load_is_worthwhile": load_is_worthwhile,
                "last_access_time_ns": _as_int(object_state.get("last_access_time_ns")),
                "current_tier_source": _current_tier_source(object_state, binding),
                "prediction_present": prediction is not None,
                "prediction_reason": "" if prediction is None else prediction.reason,
                "prediction_lead_time_ms": None if prediction is None else prediction.recommended_lead_time_ms,
                "prediction_score": None if prediction is None else prediction.score,
                "prediction_runtime_ready": prediction is not None and prediction_gate_reason is None,
                "prediction_gate_reason": "" if prediction_gate_reason is None else prediction_gate_reason,
            },
        )
        if load_target_id:
            decision.metadata["load_target_id"] = load_target_id
        if prediction is not None:
            decision.metadata["prediction_candidate_object_id"] = prediction.candidate_object_id
            decision.metadata["prediction_evidence_source"] = prediction.evidence_source
            decision.metadata["prediction_expires_at_ns"] = prediction.expires_at_ns
            decision.metadata["predicted_class"] = prediction.predicted_class
        if scheduler_hint is not None:
            decision.metadata["scheduler_hint"] = {
                "action": scheduler_hint.action,
                "reason": scheduler_hint.reason,
                "priority": scheduler_hint.priority,
                "metadata": dict(scheduler_hint.metadata),
            }
        decision.metadata["prefix_prefetch_candidate"] = bool(prefix_prefetch_candidate)
        decision.metadata["prefetch_kind"] = _prefetch_kind(
            action=action,
            scheduler_hint=scheduler_hint,
            prediction=prediction,
            prefix_prefetch_candidate=prefix_prefetch_candidate,
        )
        decision.metadata["prefetch_source_tier"] = current_tier if action == "prefetch" and current_tier in {"cpu", "ssd"} else ""
        if binding.execution_spec is not None:
            decision.metadata["execution_spec_id"] = binding.execution_spec.spec_id
        partial_load_target = _extract_partial_load_target(
            binding=binding,
            object_state=object_state,
            allow_partial=profile_guard.partial_load_allowed,
        )
        if partial_load_target is not None:
            decision.metadata["partial_load_target"] = partial_load_target.to_record()
        action_plan = _build_runtime_action_plan(
            decision=decision,
            binding=binding,
            current_tier=current_tier,
            profile_guard=profile_guard,
            partial_load_target=partial_load_target,
        )
        decision.metadata.update({
            "decision_source": action_plan.decision_source,
            "fallback_mode": action_plan.fallback_mode,
            "profile_guard": action_plan.profile_guard.profile_guard if action_plan.profile_guard is not None else "none",
            "profile_guard_reason": (
                "" if action_plan.profile_guard is None else action_plan.profile_guard.profile_guard_reason
            ),
            "runtime_action_plan": action_plan.to_record(),
            "allow_recompute_fallback": action_plan.allow_recompute_fallback,
            "allow_partial": action_plan.allow_partial,
            "scheduler_hint_present": scheduler_hint is not None,
        })
        self.decisions.append(decision)
        self.action_plans.append(action_plan)
        return decision

    def dispatch(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        breaker = None if self.bridge.execution_gate is None else self.bridge.execution_gate.breaker
        if not self.execution_enabled:
            result = RuntimeActionResult("advisory_only", "online execution requires explicit enablement")
            self.profile_store.record_dispatch(
                decision,
                result,
                execution_enabled=self.execution_enabled,
                breaker_state=_breaker_snapshot(breaker),
            )
            self.profile_store.checkpoint()
            return result
        # Only speculatively executable actions are short-circuited by the live
        # revalidation guard. Non-speculative actions such as drop/offload still
        # need to reach the bridge so their normal rejection path remains visible.
        live_skip_reason = (
            _live_dispatch_skip_reason(self, decision)
            if decision.predicted_action in {"prefetch", "load"}
            else None
        )
        if live_skip_reason is not None:
            result = RuntimeActionResult("no_dispatch_required", live_skip_reason)
            self.profile_store.record_dispatch(
                decision,
                result,
                execution_enabled=self.execution_enabled,
                breaker_state=_breaker_snapshot(breaker),
            )
            self.profile_store.checkpoint()
            return result
        if decision.predicted_action in {"keep", "recompute", "defer"}:
            result = RuntimeActionResult(
                "no_dispatch_required",
                f"online policy kept {decision.predicted_action} as an advisory/no-op decision",
            )
            self.profile_store.record_dispatch(
                decision,
                result,
                execution_enabled=self.execution_enabled,
                breaker_state=_breaker_snapshot(breaker),
            )
            self.profile_store.checkpoint()
            return result
        result = self.bridge.dispatch(decision)
        if (
            result.status not in {"executed"}
            and bool(decision.metadata.get("allow_recompute_fallback"))
            and decision.predicted_action == "load"
        ):
            result = RuntimeActionResult(
                "recompute_fallback",
                "runtime load failed or was blocked; recompute fallback preserves correctness",
                event=RuntimeEvictionEvent(
                    run_id=self.run_id,
                    runtime_event_id=f"recompute-fallback:{decision.decision_id}",
                    request_id=decision.request_id,
                    object_key=decision.object_key,
                    object_level=decision.object_level,
                    actual_action="recompute",
                    tier_before=str(decision.metadata.get("current_tier") or "unknown"),
                    tier_after="runtime",
                    timestamp_ns=time.time_ns(),
                    arrival_index=decision.decision_index,
                    status="completed",
                    provenance="runtime_structured",
                    metadata={
                        "fallback_from_action": "load",
                        "fallback_mode": "recompute",
                        "decision_id": decision.decision_id,
                        "bridge_status": result.status,
                        "bridge_message": result.message,
                        "profile_guard_reason": str(decision.metadata.get("profile_guard_reason") or ""),
                        "backend_object_id": str(decision.metadata.get("backend_object_id") or ""),
                        "original_runtime_event_id": None if result.event is None else result.event.runtime_event_id,
                    },
                ),
                receipt=result.receipt,
            )
        if result.event is not None:
            self.trace_events.append(TraceEvent(
                event_type="backend_receipt",
                category="runtime",
                source="online_backend_bridge",
                status=result.status,
                timestamp=str(result.event.timestamp_ns or ""),
                request_id=result.event.request_id,
                chunk_id=str(result.event.metadata.get("backend_object_id") or ""),
                cache_key=result.event.object_key,
                tier=result.event.tier_after,
                bytes=result.event.bytes,
                metadata={"run_id": self.run_id, "decision_id": decision.decision_id, **result.event.metadata},
            ))
        self.profile_store.record_dispatch(
            decision,
            result,
            execution_enabled=self.execution_enabled,
            breaker_state=_breaker_snapshot(breaker),
        )
        self.profile_store.checkpoint()
        return result

    def records(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "trace": [item.to_record() for item in self.trace_events],
            "decisions": [item.to_record() for item in self.decisions],
            "action_plans": [item.to_record() for item in self.action_plans],
            "commands": [item.to_record() for item in self.bridge.commands],
            "receipts": [item.to_record() for item in self.bridge.receipts],
        }


def _breaker_snapshot(breaker: Any) -> dict[str, Any]:
    if breaker is None:
        return {}
    return {
        "state": getattr(breaker, "state", None),
        "opened_at_ns": getattr(breaker, "opened_at_ns", None),
        "failures": getattr(breaker, "failures", None),
        "timeouts": getattr(breaker, "timeouts", None),
        "pressures": getattr(breaker, "pressures", None),
        "health_restored": getattr(breaker, "health_restored", None),
    }


def _action_ready(actions: dict[str, dict[str, Any]], name: str) -> bool:
    return str(actions.get(name, {}).get("status") or "") == "ready"


def _known_tier(value: Any) -> str:
    tier = str(value or "unknown")
    return "unknown" if tier == "" else tier


def _reuse_frequency(profile: ChunkProfile | None, object_state: dict[str, Any]) -> float:
    state_requests = _as_int(object_state.get("request_count"))
    state_events = _as_int(object_state.get("event_count"))
    if state_events > 0:
        # Once the online store has observed this object, its counters are
        # authoritative for the current request.  ProfileDB also sees the
        # request's initial cache_store, which must not be counted as reuse.
        return _as_int(object_state.get("reuse_count")) / max(1, state_requests)
    if profile is not None:
        return float(profile.reuse_frequency)
    return 0.0


def _reuse_ratio_hint(profile: ChunkProfile | None) -> float:
    if profile is None or profile.reuse_ratio is None:
        return 0.0
    return max(0.0, float(profile.reuse_ratio))


def _policy_reuse_frequency(
    profile: ChunkProfile | None,
    object_state: dict[str, Any],
    observed_reuse_frequency: float,
) -> float:
    state_hint = object_state.get("last_reuse_ratio_hint")
    if state_hint not in (None, "", "None"):
        return max(0.0, _as_float(state_hint))
    profile_hint = _reuse_ratio_hint(profile)
    return profile_hint if profile_hint > 0 else observed_reuse_frequency


def _recompute_is_preferred(
    *,
    profile: ChunkProfile | None,
    load_target_present: bool,
    load_is_worthwhile: bool,
    policy_reuse_frequency: float,
    hot_reuse_threshold: float,
    cold_reuse_threshold: float,
    prefetch_waste: int,
) -> bool:
    reuse_hint = _reuse_ratio_hint(profile)
    return (
        profile is not None
        and not load_target_present
        and not load_is_worthwhile
        and prefetch_waste == 0
        and cold_reuse_threshold < reuse_hint < hot_reuse_threshold
        and policy_reuse_frequency < hot_reuse_threshold
    )


def _prefetch_hit_rate(profile: ChunkProfile | None, object_state: dict[str, Any]) -> float:
    hits = _as_int(object_state.get("prefetch_success_count"))
    waste = _as_int(object_state.get("prefetch_waste_count"))
    if hits > 0 or waste > 0:
        return hits / max(1, hits + waste)
    if profile is not None:
        return float(profile.prefetch_hit_rate)
    return 0.0


def _profile_load_latency_ms(profile: ChunkProfile | None, object_state: dict[str, Any]) -> float:
    if profile is not None and profile.avg_load_latency_ms > 0:
        return float(profile.avg_load_latency_ms)
    latency_ns = _as_float(object_state.get("load_latency_ema"))
    return latency_ns / 1_000_000 if latency_ns > 0 else 0.0


def _planner_profile(
    binding: Any,
    object_state: dict[str, Any],
    profile: ChunkProfile | None,
) -> ChunkProfile:
    if profile is not None:
        return profile
    synthetic = ChunkProfile(
        chunk_id=binding.backend_object_id,
        workload_id=str(object_state.get("profile_workload_id") or "online"),
        case=binding.request_id,
        cache_key=binding.object_key,
        request_id=binding.request_id,
        prefix_id=binding.object_key if binding.object_level is ObjectLevel.PREFIX else "",
        run_id=binding.run_id,
    )
    synthetic.request_count = _as_int(object_state.get("request_count"))
    synthetic.reuse_count = _as_int(object_state.get("reuse_count"))
    synthetic.cache_hits = _as_int(object_state.get("cache_hits"))
    synthetic.cache_misses = _as_int(object_state.get("cache_misses"))
    synthetic.cache_loads = _as_int(object_state.get("reuse_count"))
    synthetic.bytes_loaded = _as_int(object_state.get("bytes"))
    latency_ms = _profile_load_latency_ms(None, object_state)
    if latency_ms > 0:
        synthetic.load_latency_ms_total = latency_ms
        synthetic.load_latency_count = 1
    synthetic.prefetch_completed = _as_int(object_state.get("prefetch_success_count"))
    synthetic.prefetch_hits = _as_int(object_state.get("prefetch_success_count"))
    synthetic.prefetch_waste = _as_int(object_state.get("prefetch_waste_count"))
    current_tier = _known_tier(object_state.get("current_tier"))
    synthetic.tier_counts[current_tier] = 1
    return synthetic


def _layer_id_from_binding(binding: Any, object_state: dict[str, Any]) -> int | None:
    for source in (
        binding.metadata,
        object_state,
        {} if binding.execution_spec is None else binding.execution_spec.metadata,
    ):
        value = source.get("layer_id")
        parsed = _as_int(value)
        if parsed > 0 or value == 0:
            return parsed
    return None


def _build_profile_guard(
    profile_db: ProfileDB,
    *,
    binding: Any,
    object_state: dict[str, Any],
    workload_id: str,
) -> RuntimeProfileGuard:
    layer_id = _layer_id_from_binding(binding, object_state)
    sensitivity = None if layer_id is None else profile_db.get_layer_sensitivity(layer_id, workload_id=workload_id)
    quality = profile_db.get_quality_guard(
        workload_id=workload_id,
        chunk_id=binding.backend_object_id,
        layer_id=layer_id,
    )
    partial_allowed = True
    recompute_allowed = True
    prefetch_priority_boost = 0.0
    decision_source = "heuristic"
    guard_name = "none"
    reasons: list[str] = []
    if sensitivity is not None:
        decision_source = "offline-profile"
        partial_allowed = partial_allowed and sensitivity.partial_load_allowed
        recompute_allowed = recompute_allowed and sensitivity.recompute_allowed
        prefetch_priority_boost += sensitivity.prefetch_priority_boost
        if not sensitivity.partial_load_allowed:
            guard_name = "sensitivity_gate"
            reasons.append(f"layer_{sensitivity.layer_id}_blocks_partial_load")
        if not sensitivity.recompute_allowed:
            guard_name = "sensitivity_gate"
            reasons.append(f"layer_{sensitivity.layer_id}_blocks_recompute")
    if quality is not None:
        decision_source = "offline-profile"
        partial_allowed = partial_allowed and quality.partial_load_allowed
        recompute_allowed = recompute_allowed and quality.recompute_allowed
        prefetch_priority_boost += quality.prefetch_priority_boost
        if not quality.partial_load_allowed:
            guard_name = "quality_gate"
            reasons.append("quality_guard_blocks_partial_load")
        if not quality.recompute_allowed:
            guard_name = "quality_gate"
            reasons.append("quality_guard_blocks_recompute")
    return RuntimeProfileGuard(
        guard_id=f"{workload_id}:{binding.backend_object_id}",
        decision_source=decision_source,
        profile_guard=guard_name,
        profile_guard_reason=";".join(reasons),
        partial_load_allowed=partial_allowed,
        recompute_allowed=recompute_allowed,
        prefetch_priority_boost=prefetch_priority_boost,
        metadata={
            "backend_object_id": binding.backend_object_id,
            "layer_id": layer_id,
            "quality_guard_present": quality is not None,
            "layer_sensitivity_present": sensitivity is not None,
        },
    )


def _extract_partial_load_target(
    *,
    binding: Any,
    object_state: dict[str, Any],
    allow_partial: bool,
) -> PartialKVLoadTarget | None:
    if not allow_partial:
        return None
    for source in (
        {} if binding.execution_spec is None else binding.execution_spec.metadata,
        binding.metadata,
        object_state,
        {} if binding.execution_spec is None else dict(binding.execution_spec.actions.get("load") or {}),
    ):
        record = source.get("partial_load_target")
        if not isinstance(record, dict):
            continue
        token_span = record.get("token_span")
        if not isinstance(token_span, dict):
            continue
        start_token = _as_int(token_span.get("start_token"))
        end_token = _as_int(token_span.get("end_token"))
        if start_token != 0 or end_token <= start_token:
            continue
        return PartialKVLoadTarget(
            plan_id=str(record.get("plan_id") or f"partial:{binding.backend_object_id}"),
            request_id=str(record.get("request_id") or binding.request_id),
            chunk_id=str(record.get("chunk_id") or binding.backend_object_id),
            object_key=str(record.get("object_key") or binding.object_key),
            object_level=str(record.get("object_level") or binding.object_level.value),
            layer_id=record.get("layer_id"),
            token_span=TokenSpan(start_token, end_token),
            selected_tokens=_as_int(record.get("selected_tokens") or (end_token - start_token)),
            total_tokens=_as_int(record.get("total_tokens") or (end_token - start_token)),
            target_tier=MemoryTier(str(record.get("target_tier") or "gpu")),
            source_tier=MemoryTier(str(record.get("source_tier") or "unknown")),
            allow_partial=bool(record.get("allow_partial", True)),
            prefix_aligned=bool(record.get("prefix_aligned", True)),
            contiguous=bool(record.get("contiguous", True)),
            requires_recompute_fallback=bool(record.get("requires_recompute_fallback", True)),
            metadata=dict(record.get("metadata") or {}),
        )
    return None


def _build_runtime_action_plan(
    *,
    decision: OfflineEvictionDecision,
    binding: Any,
    current_tier: str,
    profile_guard: RuntimeProfileGuard,
    partial_load_target: PartialKVLoadTarget | None,
) -> RuntimeActionPlan:
    action = RuntimeActionKind(decision.predicted_action)
    token_range = None
    layer_range = None
    if partial_load_target is not None and partial_load_target.token_span is not None:
        token_range = RuntimeTokenRange(
            partial_load_target.token_span.start_token,
            partial_load_target.token_span.end_token,
        )
    if partial_load_target is not None and partial_load_target.layer_id is not None:
        layer_range = RuntimeLayerRange(partial_load_target.layer_id, partial_load_target.layer_id + 1)
    return RuntimeActionPlan(
        plan_id=f"plan:{decision.decision_id}",
        object_id=str(decision.metadata.get("backend_object_id") or binding.backend_object_id),
        object_key=decision.object_key,
        object_kind=RuntimeObjectKind.KV_SEGMENT if partial_load_target is not None else RuntimeObjectKind.KV_OBJECT,
        action=action,
        request_id=decision.request_id,
        object_level=decision.object_level.value,
        layer_range=layer_range,
        token_range=token_range,
        source_tier=RuntimePlacementTier(current_tier),
        target_tier=RuntimePlacementTier(decision.target_tier or "unknown"),
        allow_partial=partial_load_target is not None and profile_guard.partial_load_allowed,
        allow_recompute_fallback=action is RuntimeActionKind.LOAD,
        priority=max(0, _as_int(decision.metadata.get("request_count")) + int(float(decision.metadata.get("policy_reuse_frequency") or 0.0) * 100)),
        decision_source=profile_guard.decision_source,
        fallback_mode="recompute" if action is RuntimeActionKind.LOAD else "none",
        trigger_reason=decision.reason,
        profile_guard=profile_guard,
        metadata={
            "binding_id": str(decision.metadata.get("binding_id") or binding.binding_id),
            "prediction_present": bool(decision.metadata.get("prediction_present")),
            "load_target_id": str(decision.metadata.get("load_target_id") or ""),
        },
    )


def _load_target_id(binding: Any, object_state: dict[str, Any]) -> str:
    actions = {} if binding.execution_spec is None else dict(binding.execution_spec.actions)
    load_action = actions.get("load", {})
    candidate = ""
    for source in (
        load_action,
        load_action.get("metadata") if isinstance(load_action.get("metadata"), dict) else {},
        binding.metadata,
        object_state,
    ):
        for field in ("load_target_id", "last_load_target_id"):
            value = str(source.get(field) or "")
            if value:
                candidate = value
                break
        if candidate:
            break
    if not candidate:
        return ""
    current_target_id = str(object_state.get("last_load_target_id") or "")
    current_target_state = str(object_state.get("last_load_target_state") or "")
    current_target_consumed_at_ns = _as_int(object_state.get("last_load_target_consumed_at_ns"))
    if candidate == current_target_id and (
        current_target_state in {"consumed", "unavailable", "missing", "expired"}
        or current_target_consumed_at_ns > 0
    ):
        return ""
    return candidate


def _load_target_runtime_reqmeta_id(binding: Any) -> str:
    actions = {} if binding.execution_spec is None else dict(binding.execution_spec.actions)
    load_action = actions.get("load", {})
    sources = (
        load_action,
        load_action.get("metadata") if isinstance(load_action.get("metadata"), dict) else {},
        binding.metadata,
    )
    for source in sources:
        value = str(source.get("runtime_reqmeta_id") or "")
        if value:
            return value
    return ""


def _load_latency_ms(event: BackendHookEvent) -> float | None:
    raw = event.metadata.get("load_latency_ns")
    if raw in (None, "", "None"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return None if value <= 0 else value / 1_000_000


def _prediction_for_binding(
    prediction_source: PredictionSidecarIndex | None,
    *,
    binding: Any,
    now_ns: int,
) -> SidecarPrediction | None:
    if prediction_source is None:
        return None
    return prediction_source.advisory_for(
        request_id=binding.request_id,
        candidate_object_id=binding.object_key,
        object_level=binding.object_level,
        now_ns=now_ns,
    )


def _scheduler_hint_for_binding(
    scheduler_hints: SchedulerHintIndex | None,
    *,
    binding: Any,
):
    if scheduler_hints is None:
        return None
    return scheduler_hints.best_hint_for_object(
        request_id=binding.request_id,
        backend_object_id=binding.backend_object_id,
        object_key=binding.object_key,
    )


def _prefetch_kind(
    *,
    action: str,
    scheduler_hint: Any,
    prediction: SidecarPrediction | None,
    prefix_prefetch_candidate: bool,
) -> str:
    if action != "prefetch":
        return ""
    if scheduler_hint is not None and str(getattr(scheduler_hint, "action", "") or "") == "prefetch":
        return "prefix"
    if prediction is not None and prediction.predicted_class == "exact-next":
        return "next_use"
    if prefix_prefetch_candidate:
        return "prefix"
    return "heuristic"


def _prefix_prefetch_candidate(
    *,
    binding: Any,
    profile: ChunkProfile | None,
    object_state: dict[str, Any],
    prefetch_ready: bool,
    profile_guard: RuntimeProfileGuard,
) -> bool:
    if not prefetch_ready:
        return False
    if not profile_guard.partial_load_allowed and not profile_guard.recompute_allowed:
        return False
    prefix_id = str(binding.metadata.get("prefix_id") or "") or str(object_state.get("prefix_id") or "")
    if not prefix_id and binding.object_level is not ObjectLevel.PREFIX:
        return False
    if profile is None:
        return False
    if _as_int(object_state.get("prefetch_waste_count")) > 0:
        return False
    observed_reuse = _as_float(object_state.get("last_reuse_ratio_hint"))
    if observed_reuse > 0 and observed_reuse < 0.2:
        return False
    if profile.prefetch_hit_rate <= 0.0 and profile.cache_hit_rate < 0.5 and _reuse_ratio_hint(profile) < 0.5:
        return False
    reuse_hint = _reuse_ratio_hint(profile)
    return (
        profile.cache_hit_rate >= 0.5
        or profile.prefetch_hit_rate >= 0.5
        or reuse_hint >= 0.5
    )


def _prediction_prefetch_ready(
    *,
    bridge: OnlineBackendBridge,
    binding: Any,
    prediction: SidecarPrediction | None,
    prefetch_ready: bool,
    same_request_load_target: bool,
    prefetch_waste: int,
    prefetch_waste_tolerance: int,
) -> bool:
    return _prediction_prefetch_gate_reason(
        bridge=bridge,
        binding=binding,
        prediction=prediction,
        prefetch_ready=prefetch_ready,
        same_request_load_target=same_request_load_target,
        prefetch_waste=prefetch_waste,
        prefetch_waste_tolerance=prefetch_waste_tolerance,
    ) is None


def _prediction_prefetch_gate_reason(
    *,
    bridge: OnlineBackendBridge,
    binding: Any,
    prediction: SidecarPrediction | None,
    prefetch_ready: bool,
    same_request_load_target: bool,
    prefetch_waste: int,
    prefetch_waste_tolerance: int,
) -> str | None:
    if prediction is None or prediction.predicted_class != "exact-next":
        return "no_prediction"
    if not prefetch_ready or same_request_load_target:
        return "same_request_load_target" if same_request_load_target else "prefetch_not_ready"
    if prefetch_waste >= prefetch_waste_tolerance:
        return "prefetch_waste_exceeded"
    snapshot = bridge.binding_snapshot(binding.binding_id)
    if snapshot is None:
        return "binding_snapshot_unavailable"
    if bridge.binding_is_active(binding.binding_id):
        return "active_binding_conflict"
    if snapshot.get("active_request_ids"):
        return "active_binding_conflict"
    if snapshot.get("pending_io") or snapshot.get("pending_operations") or snapshot.get("action_reservation"):
        return "pending_action_conflict"
    if snapshot.get("pin_count"):
        return "pinned_binding"
    return None


def _live_dispatch_skip_reason(controller: OnlinePolicyController, decision: OfflineEvictionDecision) -> str | None:
    binding = controller.bridge.binding_for(
        decision.object_level,
        decision.object_key,
        request_id=decision.request_id,
        binding_id=str(decision.metadata.get("binding_id") or ""),
    )
    if binding is None:
        return "binding_missing"
    snapshot = controller.bridge.binding_snapshot(binding.binding_id)
    if snapshot is None:
        return "binding_snapshot_unavailable"
    if snapshot.get("active_request_ids"):
        return "active_binding_conflict"
    if snapshot.get("pending_io") or snapshot.get("pending_operations") or snapshot.get("action_reservation"):
        return "pending_action_conflict"
    if snapshot.get("pin_count"):
        return "pinned_binding"
    breaker = controller.profile_store.controller_state().get("breaker") or {}
    if str(breaker.get("state") or "closed") in {"open", "half_open"}:
        return "breaker_open"
    object_state = controller.profile_store.object_state(binding.backend_object_id) or {}
    same_request_load_target = bool(
        _load_target_id(binding, object_state)
        and _load_target_runtime_reqmeta_id(binding)
        and str(object_state.get("owner_runtime_reqmeta_id") or "")
        and _load_target_runtime_reqmeta_id(binding) == str(object_state.get("owner_runtime_reqmeta_id") or "")
    )
    if same_request_load_target:
        return "same_request_load_target"
    if decision.predicted_action == "prefetch":
        if _resolve_current_tier(object_state, binding) != "ssd":
            return "not_ssd_resident"
        try:
            prediction_expires_at_ns = int(decision.metadata.get("prediction_expires_at_ns") or 0)
        except (TypeError, ValueError):
            prediction_expires_at_ns = 0
        if prediction_expires_at_ns > 0 and time.time_ns() >= prediction_expires_at_ns:
            return "candidate_expired"
        if _as_int(object_state.get("prefetch_waste_count")) >= controller.config.prefetch_waste_tolerance:
            return "prefetch_waste_exceeded"
        if str(object_state.get("last_load_target_state") or "") in {"consumed", "unavailable", "missing", "expired"}:
            return "load_target_not_available"
        if _action_ready({} if binding.execution_spec is None else dict(binding.execution_spec.actions), "prefetch") is False:
            return "prefetch_not_ready"
    if decision.predicted_action == "load":
        load_target_id = _load_target_id(binding, object_state)
        if not load_target_id:
            return "load_target_not_available"
        if _action_ready({} if binding.execution_spec is None else dict(binding.execution_spec.actions), "load") is False:
            return "load_target_not_available"
    return None


def _load_is_worthwhile(
    *,
    request_count: int,
    load_target_present: bool,
    reuse_frequency: float,
    load_latency_ms: float,
    hot_reuse_threshold: float,
    load_latency_hot_threshold_ms: float,
) -> bool:
    return (
        (load_target_present and request_count > 0)
        or
        reuse_frequency >= hot_reuse_threshold
        or load_latency_ms >= load_latency_hot_threshold_ms
    )


def _as_int(value: Any) -> int:
    try:
        return 0 if value in (None, "", "None") else int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return 0.0 if value in (None, "", "None") else float(value)
    except (TypeError, ValueError):
        return 0.0


def _binding_tier_hint(binding: Any) -> str:
    execution_spec = getattr(binding, "execution_spec", None)
    if execution_spec is not None:
        metadata = dict(getattr(execution_spec, "metadata", {}) or {})
        tier = _known_tier(metadata.get("observed_resident_tier"))
        if tier != "unknown":
            return tier
    metadata = dict(getattr(binding, "metadata", {}) or {})
    return _known_tier(metadata.get("observed_resident_tier"))


def _resolve_current_tier(object_state: dict[str, Any], binding: Any) -> str:
    observed = _known_tier(object_state.get("current_tier"))
    if observed != "unknown":
        return observed
    return _binding_tier_hint(binding)


def _current_tier_source(object_state: dict[str, Any], binding: Any) -> str:
    observed = _known_tier(object_state.get("current_tier"))
    if observed != "unknown":
        return "online_profile"
    return "binding_hint" if _binding_tier_hint(binding) != "unknown" else "unknown"
