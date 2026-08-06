import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_bridge import OnlineBackendBridge
from astrakv.runtime.backend_hook import (
    BackendActionReceipt,
    BackendExecutionSpec,
    BackendHookEvent,
    BackendObjectBinding,
    HookAction,
)
from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision, RuntimeActionResult
from astrakv.runtime.offline_safety import OfflineSafetyGate
from astrakv.runtime.online_controller import OnlinePolicyController, OnlinePolicyControllerConfig
from astrakv.runtime.online_profile import OnlineProfileStore
from astrakv.runtime.profile_db import LayerSensitivityRecord, ProfileDB, QualityGuardRecord
from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.scheduler.hints import SchedulerHint


def gate():
    base = {
        "schema": "astrakv-offline-eviction-v1", "simulation_status": "valid",
        "trace_sha256": "t", "profile_db_sha256": "p", "workload_sha256": "w",
        "profile_source": "separate_profiling_run", "capacities": {"gpu_bytes": 1, "cpu_bytes": 1, "ssd_bytes": 1},
        "policies": [{"policy": "astrakv", "request_count": 1, "total_hits": 1, "migration_bytes": 1, "oom_unavoided": 0},
                     {"policy": "lru", "request_count": 1, "total_hits": 0, "migration_bytes": 2},
                     {"policy": "fifo", "request_count": 1, "total_hits": 0, "migration_bytes": 3}],
    }
    return OfflineSafetyGate.evaluate([{**base, "workload_id": item} for item in ("a", "b", "c")])


class OnlineControllerTests(unittest.TestCase):
    def test_online_event_stays_advisory_without_runtime_capability_preflight(self):
        binding = BackendObjectBinding("run", "req", "prefix", ObjectLevel.PREFIX, "block-7", "bind")
        class Client:
            def submit(self, command):
                return BackendActionReceipt("run", command.command_id, "receipt", "bind", "block-7", command.action, "completed", 2, tier_before="gpu", tier_after="ssd")
        bridge = OnlineBackendBridge(run_id="run", bindings=[binding], hook_client=Client(), hook_url="http://127.0.0.1:7900/actions", gate=gate())
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        event = BackendHookEvent("run", "event", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_MISS, "completed", 1, tier_after="unknown")

        self.assertTrue(controller.ingest(event))
        self.assertFalse(controller.ingest(event))
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "defer")
        self.assertEqual(controller.dispatch(proposed).status, "advisory_only")

        controller.execution_enabled = True
        result = controller.dispatch(replace(proposed, decision_id="online-1-bind"))
        self.assertEqual(result.status, "no_dispatch_required")
        self.assertEqual(controller.trace_events[-1].event_type, "cache_miss")

    def test_live_revalidation_short_circuits_only_prefetch_and_load(self):
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "req", "prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("key", HookAction.CACHE_STORE, "submitted", context)
        completed = registry.complete_operation(
            "key",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        )
        registry.observe("key", HookAction.CACHE_HIT, "completed", context)
        binding = completed.binding
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
            binding_registry=registry,
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        controller.execution_enabled = True
        proposed = OfflineEvictionDecision(
            run_id="run",
            decision_id="decision-prefetch",
            request_id="req",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            predicted_action="prefetch",
        )

        result = controller.dispatch(proposed)

        self.assertEqual(result.status, "no_dispatch_required")
        self.assertEqual(result.message, "active_binding_conflict")

    def test_dispatch_feedback_updates_profile_checkpoint_for_advisory_and_blocked(self):
        binding = BackendObjectBinding("run", "req", "prefix", ObjectLevel.PREFIX, "block-7", "bind")
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "online-profile.json"
            controller = OnlinePolicyController(
                run_id="run",
                workload_id="w",
                bridge=bridge,
                profile_store=OnlineProfileStore(run_id="run", checkpoint_path=checkpoint),
            )
            event = BackendHookEvent("run", "event", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_MISS, "completed", 1, tier_after="unknown")
            self.assertTrue(controller.ingest(event))
            proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)

            advisory = controller.dispatch(proposed)
            self.assertEqual(advisory.status, "advisory_only")
            controller.execution_enabled = True
            blocked = controller.dispatch(
                type(proposed)(
                    run_id=proposed.run_id,
                    decision_id="online-1-bind",
                    request_id=proposed.request_id,
                    object_key=proposed.object_key,
                    object_level=proposed.object_level,
                    predicted_action=proposed.predicted_action,
                    target_tier=proposed.target_tier,
                    bytes=proposed.bytes,
                    decision_time_ns=proposed.decision_time_ns,
                    decision_index=proposed.decision_index,
                    reason=proposed.reason,
                    metadata=dict(proposed.metadata),
                )
            )
            self.assertEqual(blocked.status, "no_dispatch_required")

            restored = OnlineProfileStore(run_id="run", checkpoint_path=checkpoint)
            state = restored.controller_state()
            self.assertEqual(state["dispatch_count"], 2)
            self.assertEqual(state["dispatch_status_counts"]["advisory_only"], 1)
            self.assertEqual(state["dispatch_status_counts"]["no_dispatch_required"], 1)

    def test_propose_for_prefers_offload_for_cpu_resident_binding_with_ready_execution_spec(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "offload": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        store = OnlineProfileStore(run_id="run")
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge, profile_store=store)
        event = BackendHookEvent(
            "run",
            "event",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            HookAction.PREFETCH,
            "completed",
            1,
            tier_before="ssd",
            tier_after="cpu",
        )

        self.assertTrue(controller.ingest(event))
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "offload")
        self.assertEqual(proposed.target_tier, "ssd")
        self.assertEqual(proposed.metadata["current_tier"], "cpu")

    def test_keep_decision_stays_noop_even_when_execution_is_enabled(self):
        binding = BackendObjectBinding("run", "req", "prefix", ObjectLevel.PREFIX, "block-7", "bind")
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_STORE, "submitted", 1, tier_after="gpu")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-hit", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_HIT, "completed", 2, tier_after="gpu")
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "keep")
        controller.execution_enabled = True
        result = controller.dispatch(proposed)
        self.assertEqual(result.status, "no_dispatch_required")
        self.assertEqual(controller.profile_store.controller_state()["last_dispatch_status"], "no_dispatch_required")
        state = controller.profile_store.object_state("block-7")
        assert state is not None
        self.assertEqual(state["dispatch_count"], 1)
        self.assertEqual(state["last_action"], "keep")

    def test_propose_for_prefetches_hot_ssd_object_without_dynamic_load_target(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "load": {"status": "ready", "requires_dynamic_target": "load_target_id"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-hit", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_HIT, "completed", 2, tier_after="ssd")
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.target_tier, "cpu")

    def test_propose_for_loads_hot_ssd_object_when_dynamic_load_target_exists(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {"status": "ready", "load_target_id": "target-1"},
                "prefetch": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-hit", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_HIT, "completed", 2, tier_after="ssd")
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "load")
        self.assertEqual(proposed.target_tier, "gpu")
        self.assertEqual(proposed.metadata["load_target_id"], "target-1")
        self.assertEqual(proposed.metadata["fallback_mode"], "recompute")
        self.assertIn("runtime_action_plan", proposed.metadata)

    def test_profile_guard_blocks_partial_load_and_dispatch_can_fallback_to_recompute(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-guard",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-guard",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {"status": "ready", "load_target_id": "target-guard"},
                "drop": {"status": "ready"},
            },
            metadata={
                "layer_id": 4,
                "partial_load_target": {
                    "plan_id": "plan-guard",
                    "request_id": "req",
                    "chunk_id": "block-guard",
                    "token_span": {"start_token": 0, "end_token": 64},
                    "allow_partial": True,
                    "prefix_aligned": True,
                    "contiguous": True,
                    "target_tier": "gpu",
                    "source_tier": "ssd",
                },
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-guard",
            "bind",
            execution_spec=execution_spec,
            metadata={"layer_id": 4},
        )

        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        profile_db = ProfileDB()
        profile_db.put_layer_sensitivity(
            LayerSensitivityRecord(
                workload_id="w",
                layer_id=4,
                sensitivity_score=0.95,
                partial_load_allowed=False,
                recompute_allowed=True,
            )
        )
        profile_db.put_quality_guard(
            QualityGuardRecord(
                workload_id="w",
                chunk_id="block-guard",
                layer_id=4,
                quality_tier="strict",
                partial_load_allowed=False,
                recompute_allowed=True,
            )
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge, profile_db=profile_db)
        controller.execution_enabled = True
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-guard", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-hit", "req", "prefix", ObjectLevel.PREFIX, "block-guard", HookAction.CACHE_HIT, "completed", 2, tier_after="ssd")
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.metadata["decision_source"], "offline-profile")
        self.assertEqual(proposed.metadata["profile_guard"], "quality_gate")
        self.assertFalse(proposed.metadata["allow_partial"])
        self.assertNotIn("partial_load_target", proposed.metadata)
        self.assertIn("runtime_action_plan", proposed.metadata)

    def test_scheduler_hint_prefetch_unifies_prefix_prefetch_path(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-prefetch",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-prefetch",
            object_key="prefix-hot",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "load": {"status": "ready", "load_target_id": "target-1"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-hot",
            ObjectLevel.PREFIX,
            "block-prefetch",
            "bind",
            execution_spec=execution_spec,
            metadata={"prefix_id": "prefix-hot"},
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        scheduler_hints = SchedulerHintIndex.from_hints(
            [
                SchedulerHint(
                    request_id="req",
                    action="prefetch",
                    reason="prefix warm candidate",
                    priority=80,
                    metadata={"chunk_id": "block-prefetch", "prefix_id": "prefix-hot"},
                )
            ]
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=scheduler_hints,
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix-hot", ObjectLevel.PREFIX, "block-prefetch", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-hit", "req", "prefix-hot", ObjectLevel.PREFIX, "block-prefetch", HookAction.CACHE_HIT, "completed", 2, tier_after="ssd", metadata={"prefix_id": "prefix-hot"})
        ))

        proposed = controller.propose_for("prefix-hot", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["scheduler_hint"]["action"], "prefetch")
        self.assertTrue(proposed.metadata["prefix_prefetch_candidate"])
        self.assertEqual(proposed.metadata["prefetch_kind"], "prefix")
        self.assertEqual(proposed.metadata["prefetch_source_tier"], "ssd")

    def test_scheduler_hint_offload_unifies_cpu_runtime_decision(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-offload-hint",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-cpu",
            object_key="prefix-cpu",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "offload": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-cpu",
            ObjectLevel.PREFIX,
            "block-cpu",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        scheduler_hints = SchedulerHintIndex.from_hints(
            [
                SchedulerHint(
                    request_id="req",
                    action="offload",
                    reason="cpu object should cool to ssd",
                    priority=70,
                    metadata={"chunk_id": "block-cpu"},
                )
            ]
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=scheduler_hints,
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-prefetch", "req", "prefix-cpu", ObjectLevel.PREFIX, "block-cpu",
                HookAction.PREFETCH, "completed", 1, tier_before="ssd", tier_after="cpu",
            )
        ))

        proposed = controller.propose_for("prefix-cpu", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "offload")
        self.assertEqual(proposed.target_tier, "ssd")
        self.assertEqual(proposed.metadata["scheduler_hint"]["action"], "offload")

    def test_scheduler_hint_drop_is_consumed_for_none_tier(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-none-drop-hint",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-none",
            object_key="prefix-none",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "drop": {"status": "ready"},
                "load": {"status": "ready", "requires_dynamic_target": "load_target_id"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-none",
            ObjectLevel.PREFIX,
            "block-none",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        scheduler_hints = SchedulerHintIndex.from_hints(
            [
                SchedulerHint(
                    request_id="req",
                    action="drop",
                    reason="object unavailable, clean bookkeeping",
                    priority=40,
                    metadata={"chunk_id": "block-none"},
                )
            ]
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=scheduler_hints,
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-load", "req", "prefix-none", ObjectLevel.PREFIX, "block-none",
                HookAction.CACHE_LOAD, "completed", 1, tier_before="ssd", tier_after="none",
            )
        ))

        proposed = controller.propose_for("prefix-none", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "drop")
        self.assertEqual(proposed.metadata["scheduler_hint"]["action"], "drop")

    def test_propose_for_loads_first_revisit_when_dynamic_load_target_signals_live_demand(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-8",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {"status": "ready", "load_target_id": "target-live"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-8",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-store",
                "req",
                "prefix",
                ObjectLevel.PREFIX,
                "block-8",
                HookAction.CACHE_STORE,
                "submitted",
                1,
                tier_after="ssd",
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-load-ready",
                "req",
                "prefix",
                ObjectLevel.PREFIX,
                "block-8",
                HookAction.CACHE_LOAD,
                "available",
                2,
                tier_before="ssd",
                tier_after="ssd",
                metadata={"load_target_id": "target-live", "runtime_reqmeta_id": "reqmeta-live"},
            )
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "load")
        self.assertEqual(proposed.target_tier, "gpu")
        self.assertEqual(proposed.metadata["request_count"], 1)
        self.assertEqual(proposed.metadata["load_target_id"], "target-live")

    def test_same_request_load_target_is_not_treated_as_future_load_demand(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-same-request",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {
                    "status": "ready",
                    "load_target_id": "target-current",
                    "runtime_reqmeta_id": "reqmeta-current",
                    "metadata": {"runtime_reqmeta_id": "reqmeta-current"},
                },
                "offload": {"status": "ready"},
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "completed", 1, tier_after="ssd",
                metadata={"runtime_reqmeta_id": "reqmeta-current"},
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-release", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.RELEASE, "completed", 2, tier_before="ssd", tier_after="ssd",
                metadata={"runtime_reqmeta_id": "reqmeta-current"},
            )
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "load")
        self.assertNotEqual(proposed.predicted_action, "prefetch")
        self.assertFalse(proposed.metadata["load_target_present"])
        self.assertTrue(proposed.metadata["same_request_load_target"])

    def test_runtime_reqmeta_id_without_load_target_does_not_create_load_demand(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-no-target",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {"status": "ready", "requires_dynamic_target": "load_target_id"},
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "completed", 1, tier_after="ssd",
                metadata={"runtime_reqmeta_id": "reqmeta-current"},
            )
        ))
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertFalse(proposed.metadata["load_target_present"])

    def test_propose_for_evicts_cold_ssd_object_after_prefetch_waste(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-7",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix",
            ObjectLevel.PREFIX,
            "block-7",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-prefetch", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.PREFETCH, "completed", 2, tier_before="ssd", tier_after="cpu")
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-offload", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.OFFLOAD, "completed", 3, tier_before="cpu", tier_after="ssd")
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "evict")
        self.assertEqual(proposed.target_tier, "ssd")

    def test_propose_for_drops_cold_ssd_object_without_prefetch_waste(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-drop-cold-ssd",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-drop",
            object_key="prefix-drop",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-drop",
            ObjectLevel.PREFIX,
            "block-drop",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-store",
                "req",
                "prefix-drop",
                ObjectLevel.PREFIX,
                "block-drop",
                HookAction.CACHE_STORE,
                "completed",
                1,
                tier_after="ssd",
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-release",
                "req",
                "prefix-drop",
                ObjectLevel.PREFIX,
                "block-drop",
                HookAction.RELEASE,
                "completed",
                2,
                tier_before="ssd",
                tier_after="ssd",
            )
        ))

        proposed = controller.propose_for("prefix-drop", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "drop")
        self.assertEqual(proposed.metadata["reuse_frequency"], 0.0)
        self.assertEqual(proposed.metadata["prefetch_waste_count"], 0)

    def test_propose_for_recomputes_low_nonzero_reuse_without_load_target(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-recompute-ssd",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-recompute",
            object_key="prefix-recompute",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "load": {"status": "ready", "requires_dynamic_target": "load_target_id"},
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-recompute",
            ObjectLevel.PREFIX,
            "block-recompute",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        event_metadata = {"reuse_ratio": 0.15}
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-store",
                "req",
                "prefix-recompute",
                ObjectLevel.PREFIX,
                "block-recompute",
                HookAction.CACHE_STORE,
                "submitted",
                1,
                tier_after="ssd",
                metadata=event_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-release",
                "req",
                "prefix-recompute",
                ObjectLevel.PREFIX,
                "block-recompute",
                HookAction.RELEASE,
                "completed",
                2,
                tier_before="ssd",
                tier_after="ssd",
                metadata=event_metadata,
            )
        ))

        proposed = controller.propose_for("prefix-recompute", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "recompute")
        self.assertEqual(proposed.metadata["policy_reuse_frequency"], 0.15)
        controller.execution_enabled = True
        self.assertEqual(controller.dispatch(proposed).status, "no_dispatch_required")
        self.assertEqual(controller.bridge.commands, [])

    def test_propose_for_evicts_prefetched_object_after_low_value_revisit_returns_to_ssd(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-evict-after-hit",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-evict-hit",
            object_key="prefix-evict-hit",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "offload": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req",
            "prefix-evict-hit",
            ObjectLevel.PREFIX,
            "block-evict-hit",
            "bind",
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        seed_metadata = {"reuse_ratio": 0.70}
        followup_metadata = {"reuse_ratio": 0.05}
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", metadata=seed_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-prefetch", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.PREFETCH, "completed", 2, tier_before="ssd", tier_after="cpu", metadata=seed_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-hit", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.CACHE_HIT, "completed", 3, tier_before="cpu", tier_after="cpu", metadata=followup_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-release", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.RELEASE, "completed", 4, tier_before="cpu", tier_after="cpu", metadata=followup_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-offload", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.OFFLOAD, "completed", 5, tier_before="cpu", tier_after="ssd", metadata=followup_metadata,
            )
        ))

        proposed = controller.propose_for("prefix-evict-hit", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "evict")
        self.assertEqual(proposed.metadata["policy_reuse_frequency"], 0.05)
        self.assertGreater(proposed.metadata["prefetch_hit_rate"], 0.0)

    def test_breaker_open_prefers_recompute_when_io_is_expensive(self):
        binding = BackendObjectBinding("run", "req", "prefix", ObjectLevel.PREFIX, "block-7", "bind")
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        store = OnlineProfileStore(run_id="run")
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            profile_store=store,
            config=OnlinePolicyControllerConfig(memory_pressure=0.5),
        )
        decision = OfflineEvictionDecision(
            run_id="run",
            decision_id="breaker-state",
            request_id="req",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            predicted_action="drop",
        )
        self.assertTrue(store.record_dispatch(
            decision,
            RuntimeActionResult("advisory_only", "breaker checkpoint"),
            execution_enabled=True,
            breaker_state={"state": "open"},
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run",
                "event-load",
                "req",
                "prefix",
                ObjectLevel.PREFIX,
                "block-7",
                HookAction.CACHE_LOAD,
                "completed",
                1,
                tier_before="ssd",
                tier_after="none",
                metadata={"load_latency_ns": 200_000_000},
            )
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "recompute")


if __name__ == "__main__":
    unittest.main()
