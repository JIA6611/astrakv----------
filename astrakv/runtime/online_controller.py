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
from astrakv.runtime.prefix_learning import prefix_key_from_binding
from astrakv.runtime.kv_runtime_core import RuntimeMode
from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.runtime.online_profile import OnlineProfileStore
from astrakv.runtime.prediction_sidecar import PredictionSidecarIndex, SidecarPrediction
from astrakv.runtime.runtime_plan import (
    RuntimeActionKind,
    RuntimeActionPlan,
    RuntimeLayerRange,
    RuntimeObjectKind,
    RuntimePlacementTier,
    RuntimePrefetchWindow,
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
    enable_prefetch_dispatch: bool = True
    online_prefetch_mode: str = "disabled"
    inter_arrival_required_window_ms: float = 50.0
    inter_arrival_borderline_ratio: float = 0.75
    runtime_prefix_min_reuse_count: int = 2
    runtime_prefix_min_observation_count: int = 3
    runtime_prefix_confidence_threshold: float = 0.55
    # evict-B control plane: pressure gate, coldness score, and global scan.
    evict_dispatch_enabled: bool = True
    evict_pressure_gate_enabled: bool = True
    evict_pressure_trigger: float = 0.8
    evict_cpu_capacity_bytes: int = 0
    evict_ssd_capacity_bytes: int = 0
    evict_cold_score_threshold: float = 0.35
    global_evict_scan_enabled: bool = True
    global_evict_scan_min_interval_s: float = 5.0
    global_evict_scan_max_victims: int = 4
    # Direct controller users retain the historic active default. The service
    # host explicitly supplies off/shadow/active and defaults to off.
    kv_core_mode: RuntimeMode = RuntimeMode.ACTIVE
    # When True (and kv_core_mode is OFF), a prefetch decision may bypass the
    # mode=off short-circuit and dispatch through the normal bridge, while
    # every other action stays gated by mode.  This lets Prefetch-B run as a
    # standalone strategy without dragging in A/evict/offload.  Defaults to
    # False so the fail-closed "mode=off means nothing executes" contract is
    # preserved.
    prefetch_dispatch_independent_of_mode: bool = False
    # Same mode=off bypass for evict decisions, so legacy-hooks runs can
    # execute evict-B through the action service without KV-Core active mode.
    evict_dispatch_independent_of_mode: bool = False
    # Offline ProfileDB workload id for cross-run profile lookups.  The
    # controller's own workload_id is the live run id, but a train-phase
    # profile is keyed by its own workload id (e.g. "train-qasper"); without
    # this override the offline history profile would never be found.
    offline_profile_workload_id: str = ""


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
        self._last_global_evict_scan_ns = 0

    def _offline_profile_for(
        self,
        binding: Any,
        object_state: dict[str, Any],
    ) -> ChunkProfile | None:
        """Resolve the offline history profile by a cross-run stable key.

        ``binding.backend_object_id`` embeds the live run id, so an offline
        profile built from a train run can never match it.  The prefix key
        (canonical object identity) is stable across runs; the optional
        ``offline_profile_workload_id`` aligns the train profile's workload
        bucket with this controller.
        """
        if not self.config.offline_profile_workload_id:
            return None
        workload_id = self.config.offline_profile_workload_id
        prefix_key = prefix_key_from_binding(binding, object_state)
        if prefix_key:
            chunk = self.profile_db.get_chunk(prefix_key, workload_id=workload_id)
            if chunk is not None:
                return chunk
        return self.profile_db.get_chunk(binding.backend_object_id, workload_id=workload_id)

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
            or self._offline_profile_for(binding, object_state)
        )
        prefix_key = prefix_key_from_binding(binding, object_state)
        runtime_prefix_profile = None if not prefix_key else self.profile_store.prefix_profile(prefix_key)
        execution_actions = {} if binding.execution_spec is None else dict(binding.execution_spec.actions)
        active_refs = _as_int(object_state.get("active_reference_count"))
        request_count = _as_int(object_state.get("request_count"))
        reuse_frequency = _reuse_frequency(profile, object_state)
        policy_reuse_frequency = _policy_reuse_frequency(profile, object_state, reuse_frequency)
        prefetch_hit_rate = _prefetch_hit_rate(profile, object_state)
        prefetch_waste = _as_int(object_state.get("prefetch_waste_count"))
        load_latency_ms = _profile_load_latency_ms(profile, object_state)
        pressure_snapshot = self._pressure_snapshot()
        pressure_over = bool(pressure_snapshot["over_pressure"]) if self.config.evict_pressure_gate_enabled else True
        cpu_pressure_over = True
        if self.config.evict_pressure_gate_enabled:
            cpu_pressure_over = bool(
                pressure_snapshot["available"]
                and pressure_snapshot["cpu_usage_fraction"] >= float(pressure_snapshot["trigger"])
            )
        ssd_pressure_over = True
        if self.config.evict_pressure_gate_enabled:
            ssd_pressure_over = bool(
                pressure_snapshot["available"]
                and pressure_snapshot["ssd_usage_fraction"] >= float(pressure_snapshot["trigger"])
            )
        runtime_confidence = (
            float(runtime_prefix_profile.get("runtime_confidence") or 0.0)
            if runtime_prefix_profile is not None
            else 0.0
        )
        evict_cold_score = _evict_cold_score(
            policy_reuse_frequency=policy_reuse_frequency,
            runtime_confidence=runtime_confidence,
            prefetch_waste=prefetch_waste,
            cold_reuse_threshold=self.config.cold_reuse_threshold,
            prefetch_waste_tolerance=self.config.prefetch_waste_tolerance,
        )
        load_target_id = _load_target_id(binding, object_state)
        load_target_reqmeta_id = _load_target_runtime_reqmeta_id(binding)
        owner_reqmeta_id = str(object_state.get("owner_runtime_reqmeta_id") or "")
        prefetch_mode = str(self.config.online_prefetch_mode or "disabled")
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
        evict_ready = _action_ready(execution_actions, "evict") and self.config.evict_dispatch_enabled
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
        prefetch_window = _prefetch_window_for_binding(
            bridge=self.bridge,
            binding=binding,
            object_state=object_state,
            runtime_prefix_profile=runtime_prefix_profile,
            required_window_ms=max(
                self.config.inter_arrival_required_window_ms,
                load_latency_ms if load_latency_ms > 0 else 0.0,
            ),
            borderline_ratio=self.config.inter_arrival_borderline_ratio,
        )
        runtime_prefix_candidate = _runtime_prefix_prefetch_candidate(
            runtime_prefix_profile=runtime_prefix_profile,
            current_tier=current_tier,
            prefetch_ready=prefetch_ready,
            same_request_load_target=same_request_load_target,
            prefetch_window=prefetch_window,
            min_reuse_count=self.config.runtime_prefix_min_reuse_count,
            min_observation_count=self.config.runtime_prefix_min_observation_count,
            confidence_threshold=self.config.runtime_prefix_confidence_threshold,
        )
        prefix_prefetch_candidate = _prefix_prefetch_candidate(
            binding=binding,
            profile=profile,
            object_state=object_state,
            prefetch_ready=prefetch_ready,
            profile_guard=profile_guard,
            scheduler_hint=scheduler_hint,
            prefetch_mode=prefetch_mode,
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
        prefix_prefetch_gate_reason = _prefix_prefetch_gate_reason(
            current_tier=current_tier,
            prefetch_ready=prefetch_ready,
            prefix_prefetch_candidate=prefix_prefetch_candidate,
            prefetch_mode=prefetch_mode,
            prefetch_window=prefetch_window,
            runtime_prefix_candidate=runtime_prefix_candidate,
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
            elif active_refs == 0 and evict_ready and cpu_pressure_over and (
                evict_cold_score >= float(self.config.evict_cold_score_threshold)
            ):
                action = "evict"
                target_tier = "cpu"
                reason = (
                    "evict-B: CPU-layer eviction removes the CPU copy only (SSD preserved), "
                    f"(cold_score={evict_cold_score:.3f})"
                )
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
            elif prefetch_mode == "hybrid" and runtime_prefix_candidate["eligible"]:
                action = "prefetch"
                target_tier = "cpu"
                reason = str(runtime_prefix_candidate["reason"])
            elif (
                scheduler_hint is not None
                and scheduler_action == "prefetch"
                and prefix_prefetch_gate_reason is None
            ):
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
            elif active_refs == 0 and evict_ready and ssd_pressure_over and (
                evict_cold_score >= float(self.config.evict_cold_score_threshold)
                or prefetch_waste > 0
            ):
                action = "evict"
                target_tier = "ssd"
                reason = (
                    "evict-B: SSD-layer eviction removes the SSD copy only "
                    f"(cold_score={evict_cold_score:.3f})"
                )
            elif prefetch_mode != "prefix_only" and prediction_gate_reason is None and prediction is not None:
                action = "prefetch"
                target_tier = "cpu"
                reason = (
                    "online profile: sidecar exact-next advisory passed runtime gates, "
                    "so CPU prefetch is preferred before future demand"
                )
            elif prefix_prefetch_gate_reason is None:
                action = "prefetch"
                target_tier = "cpu"
                reason = "online profile: prefix reuse profile nominated this SSD object for prefix prefetch"
            elif prefetch_mode != "prefix_only" and prediction is not None:
                action = "defer"
                reason = (
                    "online profile: sidecar exact-next advisory is present but not runtime-ready, "
                    f"so dispatch stays advisory-only ({prediction_gate_reason})"
                )
            elif prefetch_mode != "prefix_only" and self.prediction_source is not None:
                action = "defer"
                reason = "online profile: no exact-next advisory matched this SSD object, so speculative dispatch stays advisory-only"
            elif prefetch_mode != "prefix_only" and prefetch_ready and not same_request_load_target and (
                active_refs > 0
                or
                policy_reuse_frequency >= self.config.hot_reuse_threshold
                or load_latency_ms >= self.config.load_latency_hot_threshold_ms
            ) and prefetch_waste < self.config.prefetch_waste_tolerance:
                action = "prefetch"
                target_tier = "cpu"
                reason = "online profile: SSD-resident object looks warm but is not ready for direct load, so prefetch is preferred"
            elif active_refs == 0 and drop_ready and (
                policy_reuse_frequency <= self.config.cold_reuse_threshold
                and prefetch_hit_rate <= 0.0
                and prefetch_waste == 0
            ):
                action = "drop"
                reason = "online profile: SSD-resident object is cold, unreferenced, and has no prefetch waste, so drop is preferred"
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
                "evict_dispatch_enabled": self.config.evict_dispatch_enabled,
                "evict_pressure_gate_enabled": self.config.evict_pressure_gate_enabled,
                "evict_pressure_snapshot": dict(pressure_snapshot),
                "evict_pressure_over": bool(pressure_over),
                "evict_cpu_pressure_over": bool(cpu_pressure_over),
                "evict_ssd_pressure_over": bool(ssd_pressure_over),
                "runtime_confidence": runtime_confidence,
                "evict_cold_score": evict_cold_score,
                "load_is_worthwhile": load_is_worthwhile,
                "last_access_time_ns": _as_int(object_state.get("last_access_time_ns")),
                "current_tier_source": _current_tier_source(object_state, binding),
                "prediction_present": prediction is not None,
                "prediction_reason": "" if prediction is None else prediction.reason,
                "prediction_lead_time_ms": None if prediction is None else prediction.recommended_lead_time_ms,
                "prediction_score": None if prediction is None else prediction.score,
                "prediction_runtime_ready": prediction is not None and prediction_gate_reason is None,
                "prediction_gate_reason": "" if prediction_gate_reason is None else prediction_gate_reason,
                "prefix_prefetch_gate_reason": "" if prefix_prefetch_gate_reason is None else prefix_prefetch_gate_reason,
                "online_prefetch_mode": prefetch_mode,
                "prefix_key": prefix_key,
                "runtime_prefix_profile": {} if runtime_prefix_profile is None else dict(runtime_prefix_profile),
                "runtime_prefix_candidate": dict(runtime_prefix_candidate),
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
        decision.metadata["dispatch_origin"] = _dispatch_origin_from_object_state(object_state)
        decision.metadata["prefetch_candidate_source"] = _prefetch_candidate_source(
            action=action,
            prefetch_mode=prefetch_mode,
            scheduler_hint=scheduler_hint,
            runtime_prefix_candidate=runtime_prefix_candidate,
            prefix_prefetch_candidate=prefix_prefetch_candidate,
        )
        decision.metadata["prefetch_kind"] = _prefetch_kind(
            action=action,
            scheduler_hint=scheduler_hint,
            prediction=prediction,
            prefix_prefetch_candidate=prefix_prefetch_candidate,
        )
        decision.metadata["prefetch_skip_reason"] = _prefetch_skip_reason(
            action=action,
            prefix_prefetch_gate_reason=prefix_prefetch_gate_reason,
            prediction_gate_reason=prediction_gate_reason,
            scheduler_hint=scheduler_hint,
            same_request_load_target=same_request_load_target,
            prefetch_mode=prefetch_mode,
        )
        decision.metadata["prefetch_source_tier"] = current_tier if action == "prefetch" and current_tier in {"cpu", "ssd"} else ""
        decision.metadata["prefetch_window"] = prefetch_window.to_record()
        decision.metadata.update(prefetch_window.to_record())
        decision.metadata["runtime_prefix_confidence"] = float(runtime_prefix_candidate.get("runtime_prefix_confidence") or 0.0)
        decision.metadata["runtime_prefix_observation_count"] = int(runtime_prefix_candidate.get("runtime_prefix_observation_count") or 0)
        decision.metadata["cold_start_seed_used"] = bool(action == "prefetch" and not runtime_prefix_candidate["eligible"])
        if binding.execution_spec is not None:
            decision.metadata["execution_spec_id"] = binding.execution_spec.spec_id
        partial_load_target = _extract_partial_load_target(
            binding=binding,
            object_state=object_state,
            allow_partial=profile_guard.partial_load_allowed,
        )
        if partial_load_target is not None:
            decision.metadata["partial_load_target"] = partial_load_target.to_record()
        resolved_decision_source = _resolved_decision_source(
            action=action,
            prefetch_mode=prefetch_mode,
            runtime_prefix_candidate=runtime_prefix_candidate,
            profile_guard=profile_guard,
            scheduler_hint=scheduler_hint,
        )
        decision.metadata["decision_source"] = resolved_decision_source
        action_plan = _build_runtime_action_plan(
            decision=decision,
            binding=binding,
            current_tier=current_tier,
            profile_guard=profile_guard,
            partial_load_target=partial_load_target,
        )
        decision.metadata.update({
            "decision_source": resolved_decision_source,
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
        if self.config.kv_core_mode is RuntimeMode.OFF:
            prefetch_independent_channel = (
                self.config.prefetch_dispatch_independent_of_mode
                and decision.predicted_action == "prefetch"
                and self.config.enable_prefetch_dispatch
                and self.config.online_prefetch_mode != "disabled"
            )
            evict_independent_channel = (
                self.config.evict_dispatch_independent_of_mode
                and decision.predicted_action == "evict"
                and self.config.evict_dispatch_enabled
            )
            if not (prefetch_independent_channel or evict_independent_channel):
                result = RuntimeActionResult("kv_core_off", "KV-Core mode is off; no runtime command was issued")
                self.profile_store.record_dispatch(
                    decision, result, execution_enabled=False, breaker_state=_breaker_snapshot(breaker),
                )
                self.profile_store.checkpoint()
                return result
            # Independent channel: fall through so the dispatch gates
            # (execution_enabled, live revalidation, per-action config) decide;
            # non-prefetch/evict actions never reach here.
        if self.config.kv_core_mode is RuntimeMode.SHADOW:
            result = RuntimeActionResult("shadow_only", "KV-Core shadow mode recorded the decision without runtime execution")
            self.profile_store.record_dispatch(
                decision, result, execution_enabled=False, breaker_state=_breaker_snapshot(breaker),
            )
            self.profile_store.checkpoint()
            return result
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
        if decision.predicted_action == "load":
            result = RuntimeActionResult(
                "native_connector_required",
                "generic load dispatch is disabled; only a request-owned native connector may load paged KV",
            )
            self.profile_store.record_dispatch(
                decision, result, execution_enabled=self.execution_enabled, breaker_state=_breaker_snapshot(breaker),
            )
            self.profile_store.checkpoint()
            return result
        if decision.predicted_action == "prefetch" and not self.config.enable_prefetch_dispatch:
            result = RuntimeActionResult(
                "no_dispatch_required",
                "prefetch dispatch disabled by controller config",
            )
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

    def _pressure_snapshot(self) -> dict[str, Any]:
        """Estimate CPU/SSD usage from observed resident bytes and configured capacities."""
        states = self.profile_store.objects()
        cpu_used = sum(_resident_bytes(state) for state in states.values() if state.get("current_tier") == "cpu")
        ssd_used = sum(_resident_bytes(state) for state in states.values() if state.get("current_tier") == "ssd")
        cpu_cap = max(0, int(self.config.evict_cpu_capacity_bytes))
        ssd_cap = max(0, int(self.config.evict_ssd_capacity_bytes))
        cpu_frac = (cpu_used / cpu_cap) if cpu_cap > 0 else 0.0
        ssd_frac = (ssd_used / ssd_cap) if ssd_cap > 0 else 0.0
        trigger = max(0.0, min(1.0, float(self.config.evict_pressure_trigger)))
        available = cpu_cap > 0 or ssd_cap > 0
        return {
            "available": bool(available),
            "cpu_used_bytes": int(cpu_used),
            "cpu_capacity_bytes": int(cpu_cap),
            "cpu_usage_fraction": round(cpu_frac, 6),
            "ssd_used_bytes": int(ssd_used),
            "ssd_capacity_bytes": int(ssd_cap),
            "ssd_usage_fraction": round(ssd_frac, 6),
            "trigger": trigger,
            "over_pressure": bool(available and (cpu_frac >= trigger or ssd_frac >= trigger)),
        }

    def global_evict_scan(
        self,
        *,
        now_ns: int | None = None,
    ) -> list[tuple[OfflineEvictionDecision, RuntimeActionResult]]:
        """Under capacity pressure, dispatch evict for the coldest released objects.

        Runs at most once per ``global_evict_scan_min_interval_s``.  Victims are
        ranked by the evict coldness score (low reuse + low runtime confidence +
        high prefetch waste) and dispatched through the normal bridge so every
        evict-B action is receipt-backed.
        """
        now = time.time_ns() if now_ns is None else int(now_ns)
        if not self.execution_enabled or not self.config.evict_dispatch_enabled:
            return []
        if not self.config.global_evict_scan_enabled:
            return []
        interval_ns = int(max(0.0, float(self.config.global_evict_scan_min_interval_s)) * 1_000_000_000)
        if self._last_global_evict_scan_ns and (now - self._last_global_evict_scan_ns) < interval_ns:
            return []
        self._last_global_evict_scan_ns = now
        if not self.config.evict_pressure_gate_enabled:
            return []
        pressure = self._pressure_snapshot()
        trigger = float(pressure["trigger"])
        cpu_pressure_over = pressure["available"] and pressure["cpu_usage_fraction"] >= trigger
        ssd_pressure_over = pressure["available"] and pressure["ssd_usage_fraction"] >= trigger
        if not (cpu_pressure_over or ssd_pressure_over):
            return []
        candidates: list[tuple[float, str, Any]] = []
        for backend_object_id, state in self.profile_store.objects().items():
            tier = str(state.get("current_tier") or "")
            if tier == "cpu" and not cpu_pressure_over:
                continue
            if tier == "ssd" and not ssd_pressure_over:
                continue
            if tier not in {"cpu", "ssd"}:
                continue
            if _as_int(state.get("active_reference_count")) > 0:
                continue
            object_key = str(state.get("last_object_key") or "")
            level_raw = str(state.get("last_object_level") or "")
            request_id = str(state.get("last_request_id") or "")
            if not object_key or not level_raw or not request_id:
                continue
            try:
                object_level = ObjectLevel(level_raw)
            except ValueError:
                continue
            binding = self.bridge.binding_for(object_level, object_key, request_id=request_id)
            if binding is None:
                continue
            execution_actions = {} if binding.execution_spec is None else dict(binding.execution_spec.actions)
            if not _action_ready(execution_actions, "evict"):
                continue
            profile = (
                self.observed_profile_db.get_chunk(binding.backend_object_id, workload_id=self.workload_id)
                or self._offline_profile_for(binding, object_state)
            )
            reuse = _reuse_frequency(profile, state)
            policy_reuse = _policy_reuse_frequency(profile, state, reuse)
            prefetch_waste = _as_int(state.get("prefetch_waste_count"))
            prefix_key = prefix_key_from_binding(binding, state)
            prefix_profile = None if not prefix_key else self.profile_store.prefix_profile(prefix_key)
            confidence = (
                float(prefix_profile.get("runtime_confidence") or 0.0)
                if prefix_profile is not None
                else 0.0
            )
            score = _evict_cold_score(
                policy_reuse_frequency=policy_reuse,
                runtime_confidence=confidence,
                prefetch_waste=prefetch_waste,
                cold_reuse_threshold=self.config.cold_reuse_threshold,
                prefetch_waste_tolerance=self.config.prefetch_waste_tolerance,
            )
            if score < float(self.config.evict_cold_score_threshold):
                continue
            candidates.append(
                (
                    score,
                    backend_object_id,
                    (
                        state, binding, object_level, object_key, request_id,
                        policy_reuse, confidence, prefetch_waste, "cpu" if tier == "cpu" else "ssd",
                    ),
                )
            )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        results: list[tuple[OfflineEvictionDecision, RuntimeActionResult]] = []
        max_victims = max(0, int(self.config.global_evict_scan_max_victims))
        for score, backend_object_id, payload in candidates[:max_victims]:
            (
                state, binding, object_level, object_key, request_id,
                policy_reuse, confidence, prefetch_waste, target_tier,
            ) = payload
            decision = OfflineEvictionDecision(
                run_id=self.run_id,
                decision_id=f"online-global-evict-{len(self.decisions)}-{binding.binding_id}",
                request_id=request_id,
                object_key=object_key,
                object_level=object_level,
                predicted_action="evict",
                target_tier=target_tier,
                decision_time_ns=now,
                decision_index=_as_int(state.get("last_arrival_index")) or None,
                reason=(
                    "global pressure scan: "
                    f"cpu_frac={pressure['cpu_usage_fraction']:.3f} "
                    f"ssd_frac={pressure['ssd_usage_fraction']:.3f} "
                    f"cold_score={score:.3f}"
                ),
                metadata={
                    "binding_id": binding.binding_id,
                    "binding_generation": binding.binding_generation,
                    "backend_object_id": backend_object_id,
                    "decision_source": "global_pressure_scan",
                    "dispatch_origin": "global_evict_scan",
                    "policy_reuse_frequency": policy_reuse,
                    "runtime_confidence": confidence,
                    "prefetch_waste_count": prefetch_waste,
                    "evict_cold_score": score,
                    "evict_pressure_snapshot": dict(pressure),
                    "current_tier": str(state.get("current_tier") or "unknown"),
                    "active_reference_count": _as_int(state.get("active_reference_count")),
                },
            )
            self.decisions.append(decision)
            results.append((decision, self.dispatch(decision)))
        if results:
            self.profile_store.checkpoint()
        return results

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


def _resident_bytes(object_state: dict[str, Any]) -> int:
    """Best-effort resident size estimate: last placed bytes while in cpu/ssd."""
    if str(object_state.get("current_tier") or "") not in {"cpu", "ssd"}:
        return 0
    return max(0, _as_int(object_state.get("resident_bytes")))


def _evict_cold_score(
    *,
    policy_reuse_frequency: float,
    runtime_confidence: float,
    prefetch_waste: int,
    cold_reuse_threshold: float,
    prefetch_waste_tolerance: int,
) -> float:
    """0..1 coldness: low reuse + low runtime confidence + high prefetch waste."""
    reuse_denom = max(1e-9, float(cold_reuse_threshold))
    reuse_signal = min(1.0, max(0.0, 1.0 - (float(policy_reuse_frequency) / reuse_denom)))
    confidence_signal = min(1.0, max(0.0, 1.0 - float(runtime_confidence)))
    waste_signal = min(1.0, max(0, int(prefetch_waste)) / max(1, int(prefetch_waste_tolerance)))
    return round((0.5 * reuse_signal) + (0.3 * confidence_signal) + (0.2 * waste_signal), 6)


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
        decision_source=str(decision.metadata.get("decision_source") or profile_guard.decision_source),
        fallback_mode="recompute" if action is RuntimeActionKind.LOAD else "none",
        trigger_reason=decision.reason,
        profile_guard=profile_guard,
        metadata={
            "binding_id": str(decision.metadata.get("binding_id") or binding.binding_id),
            "prediction_present": bool(decision.metadata.get("prediction_present")),
            "load_target_id": str(decision.metadata.get("load_target_id") or ""),
            "dispatch_origin": str(decision.metadata.get("dispatch_origin") or ""),
            "prefetch_skip_reason": str(decision.metadata.get("prefetch_skip_reason") or ""),
            "prefetch_window": dict(decision.metadata.get("prefetch_window") or {}),
        },
    )


def _resolved_decision_source(
    *,
    action: str,
    prefetch_mode: str,
    runtime_prefix_candidate: dict[str, Any],
    profile_guard: RuntimeProfileGuard,
    scheduler_hint: Any,
) -> str:
    if action == "prefetch":
        if prefetch_mode == "hybrid" and bool(runtime_prefix_candidate.get("eligible")):
            return "runtime-observed"
        if scheduler_hint is not None and str(getattr(scheduler_hint, "action", "") or "") == "prefetch":
            return "offline-profile"
    return profile_guard.decision_source


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
    prefix_id = str(binding.metadata.get("prefix_id") or "")
    prefix_hash = str(binding.metadata.get("prefix_hash") or "")
    return scheduler_hints.best_hint_for_object(
        request_id=binding.request_id,
        backend_object_id=binding.backend_object_id,
        object_key=binding.object_key,
        prefix_id=prefix_id,
        prefix_hash=prefix_hash,
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
    scheduler_hint: Any,
    prefetch_mode: str,
) -> bool:
    if prefetch_mode not in {"prefix_only", "combined", "hybrid"}:
        return False
    if not prefetch_ready:
        return False
    if not profile_guard.partial_load_allowed and not profile_guard.recompute_allowed:
        return False
    prefix_id = str(binding.metadata.get("prefix_id") or "") or str(object_state.get("prefix_id") or "")
    if not prefix_id and binding.object_level is not ObjectLevel.PREFIX:
        return False
    if scheduler_hint is not None and str(getattr(scheduler_hint, "action", "") or "") == "prefetch":
        return True
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


def _prefetch_window_for_binding(
    *,
    bridge: OnlineBackendBridge,
    binding: Any,
    object_state: dict[str, Any],
    runtime_prefix_profile: dict[str, Any] | None,
    required_window_ms: float,
    borderline_ratio: float,
) -> RuntimePrefetchWindow:
    snapshot = bridge.binding_snapshot(binding.binding_id) or {}
    source_event = str(object_state.get("last_dispatch_origin") or "")
    source_timestamp_ns = _as_int(object_state.get("last_timestamp_ns"))
    next_request_submit_timestamp_ns = _as_int(snapshot.get("last_request_submit_timestamp_ns"))
    next_request_id = str(snapshot.get("last_request_id") or "")
    if source_timestamp_ns <= 0 and runtime_prefix_profile is not None:
        source_timestamp_ns = _as_int(runtime_prefix_profile.get("last_release_or_offload_ns"))
    if next_request_submit_timestamp_ns <= 0:
        next_request_submit_timestamp_ns = _as_int(object_state.get("last_request_submit_timestamp_ns"))
    if next_request_submit_timestamp_ns <= 0 and runtime_prefix_profile is not None:
        next_request_submit_timestamp_ns = _as_int(runtime_prefix_profile.get("last_next_submit_ns"))
    if next_request_submit_timestamp_ns <= source_timestamp_ns or source_timestamp_ns <= 0:
        return RuntimePrefetchWindow(
            source_event=source_event,
            source_timestamp_ns=source_timestamp_ns,
            next_request_id=next_request_id,
            next_request_submit_timestamp_ns=next_request_submit_timestamp_ns,
            inter_arrival_window_ms=0.0,
            required_window_ms=max(0.0, required_window_ms),
            window_feasibility="unknown",
            prefetch_completion_before_demand=None,
        )
    window_ms = max(0.0, (next_request_submit_timestamp_ns - source_timestamp_ns) / 1_000_000)
    required_ms = max(0.0, required_window_ms)
    if required_ms <= 0:
        feasibility = "window_sufficient"
    elif window_ms >= required_ms:
        feasibility = "window_sufficient"
    elif window_ms >= required_ms * max(0.0, borderline_ratio):
        feasibility = "window_borderline"
    else:
        feasibility = "window_insufficient"
    return RuntimePrefetchWindow(
        source_event=source_event,
        source_timestamp_ns=source_timestamp_ns,
        next_request_id=next_request_id,
        next_request_submit_timestamp_ns=next_request_submit_timestamp_ns,
        inter_arrival_window_ms=window_ms,
        required_window_ms=required_ms,
        window_feasibility=feasibility,
        prefetch_completion_before_demand=(feasibility == "window_sufficient"),
    )


def _prefix_prefetch_gate_reason(
    *,
    current_tier: str,
    prefetch_ready: bool,
    prefix_prefetch_candidate: bool,
    prefetch_mode: str,
    prefetch_window: RuntimePrefetchWindow,
    runtime_prefix_candidate: dict[str, Any],
) -> str | None:
    if prefetch_mode not in {"prefix_only", "combined", "hybrid"}:
        return "prefetch_mode_disabled"
    if current_tier != "ssd":
        return "tier_not_ssd"
    if not prefetch_ready:
        return "prefetch_not_ready"
    if prefetch_mode == "hybrid" and bool(runtime_prefix_candidate.get("eligible")):
        return None
    if not prefix_prefetch_candidate:
        return "prefix_hint_miss"
    if prefetch_window.window_feasibility == "window_insufficient":
        return "insufficient_inter_arrival_window"
    return None


def _runtime_prefix_prefetch_candidate(
    *,
    runtime_prefix_profile: dict[str, Any] | None,
    current_tier: str,
    prefetch_ready: bool,
    same_request_load_target: bool,
    prefetch_window: RuntimePrefetchWindow,
    min_reuse_count: int,
    min_observation_count: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    base = {
        "eligible": False,
        "candidate_source": "runtime-observed",
        "priority": 0,
        "reason": "",
        "runtime_prefix_confidence": 0.0,
        "runtime_prefix_observation_count": 0,
    }
    if runtime_prefix_profile is None:
        return base | {"reason": "runtime prefix learner has no profile for this prefix"}
    observation_count = _as_int(runtime_prefix_profile.get("observation_count"))
    reuse_count = _as_int(runtime_prefix_profile.get("reuse_count"))
    confidence = _as_float(runtime_prefix_profile.get("runtime_confidence"))
    hit_rate = _as_float(runtime_prefix_profile.get("prefetch_hit_rate"))
    waste = _as_int(runtime_prefix_profile.get("prefetch_waste"))
    base.update({
        "runtime_prefix_confidence": confidence,
        "runtime_prefix_observation_count": observation_count,
    })
    if current_tier != "ssd":
        return base | {"reason": "runtime prefix candidate requires SSD residency"}
    if not prefetch_ready:
        return base | {"reason": "runtime prefix candidate is not prefetch-ready"}
    if same_request_load_target:
        return base | {"reason": "candidate_suppressed_same_request"}
    if prefetch_window.window_feasibility == "window_insufficient":
        return base | {"reason": "candidate_suppressed_window_insufficient"}
    if observation_count < max(1, min_observation_count):
        return base | {"reason": "runtime prefix candidate lacks enough observations"}
    if reuse_count < max(1, min_reuse_count):
        return base | {"reason": "runtime prefix candidate lacks enough reuse"}
    if confidence < max(0.0, confidence_threshold):
        return base | {"reason": "runtime prefix confidence is below threshold"}
    if waste > 0 and hit_rate <= 0.0:
        return base | {"reason": "runtime prefix candidate shows only prefetch waste"}
    priority = max(1, min(100, int(round(confidence * 100)) + reuse_count))
    return base | {
        "eligible": True,
        "priority": priority,
        "reason": "online profile: runtime-observed prefix learner predicts this SSD prefix will be reused soon enough to prefetch",
    }


def _dispatch_origin_from_object_state(object_state: dict[str, Any]) -> str:
    action_name = str(object_state.get("last_action") or "")
    if action_name == HookAction.RELEASE.value:
        return "release_completed"
    if action_name == HookAction.OFFLOAD.value:
        return "offload_completed"
    if action_name == HookAction.CACHE_LOAD.value:
        return "cache_load_available"
    return ""


def _prefetch_skip_reason(
    *,
    action: str,
    prefix_prefetch_gate_reason: str | None,
    prediction_gate_reason: str | None,
    scheduler_hint: Any,
    same_request_load_target: bool,
    prefetch_mode: str,
) -> str:
    if action == "prefetch":
        return ""
    if same_request_load_target:
        return "same_request_load_target"
    if prefetch_mode == "prefix_only":
        return "" if prefix_prefetch_gate_reason is None else prefix_prefetch_gate_reason
    if prefetch_mode == "hybrid" and prefix_prefetch_gate_reason is not None:
        return prefix_prefetch_gate_reason
    if scheduler_hint is not None and str(getattr(scheduler_hint, "action", "") or "") == "prefetch":
        return "" if prefix_prefetch_gate_reason is None else prefix_prefetch_gate_reason
    if prefix_prefetch_gate_reason is not None:
        return prefix_prefetch_gate_reason
    return "" if prediction_gate_reason is None else prediction_gate_reason


def _prefetch_candidate_source(
    *,
    action: str,
    prefetch_mode: str,
    scheduler_hint: Any,
    runtime_prefix_candidate: dict[str, Any],
    prefix_prefetch_candidate: bool,
) -> str:
    if action != "prefetch":
        return ""
    if prefetch_mode == "hybrid" and bool(runtime_prefix_candidate.get("eligible")):
        return "runtime-observed"
    if scheduler_hint is not None and str(getattr(scheduler_hint, "action", "") or "") == "prefetch":
        return "offline-profile"
    if prefix_prefetch_candidate:
        return "heuristic"
    return "heuristic"


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
            return "tier_not_ssd"
        window = decision.metadata.get("prefetch_window") or {}
        if str(window.get("window_feasibility") or "") == "window_insufficient":
            return "insufficient_inter_arrival_window"
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
