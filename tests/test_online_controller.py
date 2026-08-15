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
from astrakv.runtime.kv_runtime_core import RuntimeMode
from astrakv.runtime.offline_safety import OfflineSafetyGate
from astrakv.runtime.online_controller import OnlinePolicyController, OnlinePolicyControllerConfig
from astrakv.runtime.online_profile import OnlineProfileStore
from astrakv.runtime.profile_db import ChunkProfile, LayerSensitivityRecord, ProfileDB, QualityGuardRecord
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
    def test_offline_profile_lookup_uses_stable_prefix_key_and_workload_id(self):
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        db = ProfileDB()
        chunk = ChunkProfile(
            chunk_id="sha256:prefix-23ff",
            workload_id="train-qasper",
            request_count=7,
            cache_hits=5,
        )
        db.chunks["train-qasper:sha256:prefix-23ff"] = chunk
        binding = BackendObjectBinding(
            "run", "req", "sha256:prefix-23ff", ObjectLevel.PREFIX,
            "lmcache:live-run-engine:worker-0:abc", "bind",
            metadata={"prefix_id": "sha256:prefix-23ff"},
        )

        controller = OnlinePolicyController(
            run_id="run",
            workload_id="live-run",
            bridge=bridge,
            profile_db=db,
            config=OnlinePolicyControllerConfig(offline_profile_workload_id="train-qasper"),
        )
        self.assertIs(controller._offline_profile_for(binding, {}), chunk)

        # Without the workload override the offline profile is out of reach
        # (backend_object_id embeds the live run id).
        controller2 = OnlinePolicyController(
            run_id="run",
            workload_id="live-run",
            bridge=bridge,
            profile_db=db,
        )
        self.assertIsNone(controller2._offline_profile_for(binding, {}))

    def test_prefetch_dispatch_independent_of_mode_only_unlocks_prefetch(self):
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        prefetch = OfflineEvictionDecision(
            run_id="run",
            decision_id="decision-independent-prefetch",
            request_id="req",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            predicted_action="prefetch",
        )
        keep = replace(prefetch, decision_id="decision-independent-keep", predicted_action="keep")

        # Fail-closed default: mode=off blocks even prefetch.
        closed = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                kv_core_mode=RuntimeMode.OFF,
                enable_prefetch_dispatch=True,
                online_prefetch_mode="hybrid",
            ),
        )
        self.assertEqual(closed.dispatch(prefetch).status, "kv_core_off")
        self.assertEqual(closed.dispatch(keep).status, "kv_core_off")

        # Independent channel: prefetch bypasses the mode=off gate (and then
        # hits the execution_enabled gate -> advisory_only, proving the
        # short-circuit was skipped), while non-prefetch stays gated.
        independent = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                kv_core_mode=RuntimeMode.OFF,
                enable_prefetch_dispatch=True,
                online_prefetch_mode="hybrid",
                prefetch_dispatch_independent_of_mode=True,
            ),
        )
        self.assertEqual(independent.dispatch(prefetch).status, "advisory_only")
        self.assertEqual(independent.dispatch(keep).status, "kv_core_off")

        # The channel is inert when prefetch execution itself is disabled.
        inert = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                kv_core_mode=RuntimeMode.OFF,
                enable_prefetch_dispatch=False,
                online_prefetch_mode="hybrid",
                prefetch_dispatch_independent_of_mode=True,
            ),
        )
        self.assertEqual(inert.dispatch(prefetch).status, "kv_core_off")

    def test_evict_dispatch_independent_of_mode_only_unlocks_evict(self):
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        evict = OfflineEvictionDecision(
            run_id="run",
            decision_id="decision-independent-evict",
            request_id="req",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            predicted_action="evict",
        )
        keep = replace(evict, decision_id="decision-independent-keep", predicted_action="keep")

        # Fail-closed default: mode=off blocks evict.
        closed = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(kv_core_mode=RuntimeMode.OFF),
        )
        self.assertEqual(closed.dispatch(evict).status, "kv_core_off")
        self.assertEqual(closed.dispatch(keep).status, "kv_core_off")

        # Independent channel: evict bypasses the mode=off gate (hits the
        # execution_enabled gate -> advisory_only), non-evict stays gated.
        independent = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                kv_core_mode=RuntimeMode.OFF,
                evict_dispatch_independent_of_mode=True,
            ),
        )
        self.assertEqual(independent.dispatch(evict).status, "advisory_only")
        self.assertEqual(independent.dispatch(keep).status, "kv_core_off")

        # Inert when evict execution itself is disabled.
        inert = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                kv_core_mode=RuntimeMode.OFF,
                evict_dispatch_independent_of_mode=True,
                evict_dispatch_enabled=False,
            ),
        )
        self.assertEqual(inert.dispatch(evict).status, "kv_core_off")

    def test_resident_bytes_combine_store_submitted_bytes_and_completed_tier(self):
        binding = self._evict_ready_binding(
            object_key="prefix", backend_object_id="block-7", spec_id="spec-resident",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(evict_cpu_capacity_bytes=1),
        )
        # Legacy hooks: bytes on store submitted, tier on store completed.
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "store-submitted", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "submitted", 1, tier_after="unknown", bytes=37748736,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "store-completed", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "completed", 2, tier_after="cpu", bytes=None,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "release", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.RELEASE, "completed", 3, tier_before="cpu", tier_after="cpu",
            )
        ))
        # Second observation cycle so request_count reaches the eviction
        # observation floor (2) and the object is eligible for eviction.
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "store-submitted-2", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "submitted", 4, tier_after="unknown", bytes=37748736,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "store-completed-2", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "completed", 5, tier_after="cpu", bytes=None,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "release-2", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.RELEASE, "completed", 6, tier_before="cpu", tier_after="cpu",
            )
        ))
        snapshot = controller._pressure_snapshot()
        self.assertGreater(snapshot["cpu_usage_fraction"], 0.8)
        # Single-object (release) eviction was removed: pressure alone no
        # longer evicts; eviction is decided by the global pressure scan.
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "evict")

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

    def test_propose_for_no_longer_evicts_cpu_object(self):
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
        store = OnlineProfileStore(run_id="run")
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            profile_store=store,
            config=OnlinePolicyControllerConfig(evict_cpu_capacity_bytes=1),
        )
        # Two store+release cycles so request_count reaches the observation
        # floor (2), then a prefetch places the object in CPU.
        for index, suffix in ((1, "a"), (4, "b")):
            self.assertTrue(controller.ingest(BackendHookEvent(
                "run", f"store-s-{suffix}", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "submitted", index, tier_after="unknown", bytes=1,
            )))
            self.assertTrue(controller.ingest(BackendHookEvent(
                "run", f"store-c-{suffix}", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.CACHE_STORE, "completed", index + 1, tier_after="cpu", bytes=None,
            )))
            self.assertTrue(controller.ingest(BackendHookEvent(
                "run", f"release-{suffix}", "req", "prefix", ObjectLevel.PREFIX, "block-7",
                HookAction.RELEASE, "completed", index + 2, tier_before="cpu", tier_after="cpu",
            )))
        self.assertTrue(controller.ingest(BackendHookEvent(
            "run", "prefetch", "req", "prefix", ObjectLevel.PREFIX, "block-7",
            HookAction.PREFETCH, "completed", 7, tier_before="ssd", tier_after="cpu", bytes=1,
        )))
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "evict")
        self.assertEqual(proposed.metadata["current_tier"], "cpu")
        self.assertTrue(proposed.metadata["evict_cpu_pressure_over"])

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
            config=OnlinePolicyControllerConfig(online_prefetch_mode="prefix_only"),
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

    def test_prefix_only_mode_prefetches_from_scheduler_hint_without_same_request_load(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-prefix-only",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-prefix-only",
            object_key="prefix-prefetch",
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
            "req-a",
            "prefix-prefetch",
            ObjectLevel.PREFIX,
            "block-prefix-only",
            "bind",
            metadata={"prefix_id": "prefix-prefetch", "prefix_hash": "prefix-prefetch"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        hints = SchedulerHintIndex.from_hints([
            SchedulerHint(
                request_id="",
                action="prefetch",
                reason="prefix reuse ready",
                priority=10,
                metadata={"prefix_id": "prefix-prefetch", "object_key": "prefix-prefetch"},
            )
        ])
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=hints,
            config=OnlinePolicyControllerConfig(online_prefetch_mode="prefix_only"),
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req-a", "prefix-prefetch", ObjectLevel.PREFIX,
                "block-prefix-only", HookAction.CACHE_STORE, "completed", 1, tier_after="ssd",
                metadata={"prefix_id": "prefix-prefetch", "prefix_hash": "prefix-prefetch"},
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-release", "req-a", "prefix-prefetch", ObjectLevel.PREFIX,
                "block-prefix-only", HookAction.RELEASE, "completed", 2, tier_before="ssd", tier_after="ssd",
                metadata={"prefix_id": "prefix-prefetch", "prefix_hash": "prefix-prefetch"},
            )
        ))

        proposed = controller.propose_for("prefix-prefetch", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["dispatch_origin"], "release_completed")
        self.assertEqual(proposed.metadata["prefetch_skip_reason"], "")
        self.assertEqual(proposed.metadata["online_prefetch_mode"], "prefix_only")

    def test_prefix_only_mode_marks_insufficient_inter_arrival_window(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-window",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-window",
            object_key="prefix-window",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-a",
            "prefix-window",
            ObjectLevel.PREFIX,
            "block-window",
            "bind",
            metadata={"prefix_id": "prefix-window", "prefix_hash": "prefix-window"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        hints = SchedulerHintIndex.from_hints([
            SchedulerHint(
                request_id="",
                action="prefetch",
                reason="prefix reuse ready",
                priority=10,
                metadata={"prefix_id": "prefix-window", "object_key": "prefix-window"},
            )
        ])
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=hints,
            config=OnlinePolicyControllerConfig(
                online_prefetch_mode="prefix_only",
                inter_arrival_required_window_ms=100.0,
            ),
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req-a", "prefix-window", ObjectLevel.PREFIX,
                "block-window", HookAction.CACHE_STORE, "submitted", 90, tier_after="ssd",
                metadata={"prefix_id": "prefix-window", "prefix_hash": "prefix-window"},
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-release", "req-a", "prefix-window", ObjectLevel.PREFIX,
                "block-window", HookAction.RELEASE, "completed", 100, tier_before="ssd", tier_after="ssd",
                metadata={"prefix_id": "prefix-window", "prefix_hash": "prefix-window"},
            )
        ))
        controller.profile_store._objects["block-window"]["last_request_submit_timestamp_ns"] = 120

        proposed = controller.propose_for("prefix-window", ObjectLevel.PREFIX)

        self.assertNotEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["prefetch_skip_reason"], "insufficient_inter_arrival_window")
        self.assertEqual(proposed.metadata["window_feasibility"], "window_insufficient")

    def test_hybrid_mode_uses_runtime_observed_candidate_without_hints(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-hybrid-runtime",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-hybrid",
            object_key="prefix-hybrid",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-2",
            "prefix-hybrid",
            ObjectLevel.PREFIX,
            "block-hybrid",
            "bind",
            metadata={"prefix_id": "prefix-hybrid", "prefix_hash": "prefix-hybrid"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                online_prefetch_mode="hybrid",
                runtime_prefix_min_reuse_count=1,
                runtime_prefix_min_observation_count=2,
                runtime_prefix_confidence_threshold=0.1,
            ),
        )
        meta1 = {"prefix_id": "prefix-hybrid", "prefix_hash": "prefix-hybrid", "cache_key": "prefix-hybrid", "arrival_index": 1}
        meta2 = {"prefix_id": "prefix-hybrid", "prefix_hash": "prefix-hybrid", "cache_key": "prefix-hybrid", "arrival_index": 2}
        self.assertTrue(controller.ingest(BackendHookEvent("run", "store-1", "req-2", "prefix-hybrid", ObjectLevel.PREFIX, "block-hybrid", HookAction.CACHE_STORE, "submitted", 100, tier_after="ssd", metadata=meta1)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "release-1", "req-2", "prefix-hybrid", ObjectLevel.PREFIX, "block-hybrid", HookAction.RELEASE, "completed", 200, tier_before="ssd", tier_after="ssd", metadata=meta1)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "store-2", "req-2", "prefix-hybrid", ObjectLevel.PREFIX, "block-hybrid", HookAction.CACHE_STORE, "submitted", 280, tier_after="ssd", metadata=meta2)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "release-2", "req-2", "prefix-hybrid", ObjectLevel.PREFIX, "block-hybrid", HookAction.RELEASE, "completed", 360, tier_before="ssd", tier_after="ssd", metadata=meta2)))

        proposed = controller.propose_for("prefix-hybrid", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["prefetch_candidate_source"], "runtime-observed")
        self.assertEqual(proposed.metadata["decision_source"], "runtime-observed")
        self.assertGreater(proposed.metadata["runtime_prefix_confidence"], 0.0)

    def test_hybrid_mode_runtime_candidate_overrides_offline_prefetch_seed(self):
        execution_spec = BackendExecutionSpec(
            spec_id="spec-hybrid-override",
            binding_id="bind",
            binding_generation=1,
            backend_object_id="block-override",
            object_key="prefix-override",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-2",
            "prefix-override",
            ObjectLevel.PREFIX,
            "block-override",
            "bind",
            metadata={"prefix_id": "prefix-override", "prefix_hash": "prefix-override"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        hints = SchedulerHintIndex.from_hints([
            SchedulerHint(request_id="", action="prefetch", reason="offline warm seed", priority=1, metadata={"prefix_id": "prefix-override", "object_key": "prefix-override"})
        ])
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            scheduler_hints=hints,
            config=OnlinePolicyControllerConfig(
                online_prefetch_mode="hybrid",
                runtime_prefix_min_reuse_count=1,
                runtime_prefix_min_observation_count=2,
                runtime_prefix_confidence_threshold=0.1,
            ),
        )
        meta1 = {"prefix_id": "prefix-override", "prefix_hash": "prefix-override", "cache_key": "prefix-override", "arrival_index": 1}
        meta2 = {"prefix_id": "prefix-override", "prefix_hash": "prefix-override", "cache_key": "prefix-override", "arrival_index": 2}
        self.assertTrue(controller.ingest(BackendHookEvent("run", "store-1", "req-2", "prefix-override", ObjectLevel.PREFIX, "block-override", HookAction.CACHE_STORE, "submitted", 100, tier_after="ssd", metadata=meta1)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "release-1", "req-2", "prefix-override", ObjectLevel.PREFIX, "block-override", HookAction.RELEASE, "completed", 200, tier_before="ssd", tier_after="ssd", metadata=meta1)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "store-2", "req-2", "prefix-override", ObjectLevel.PREFIX, "block-override", HookAction.CACHE_STORE, "submitted", 280, tier_after="ssd", metadata=meta2)))
        self.assertTrue(controller.ingest(BackendHookEvent("run", "release-2", "req-2", "prefix-override", ObjectLevel.PREFIX, "block-override", HookAction.RELEASE, "completed", 360, tier_before="ssd", tier_after="ssd", metadata=meta2)))

        proposed = controller.propose_for("prefix-override", ObjectLevel.PREFIX)

        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["prefetch_candidate_source"], "runtime-observed")
        self.assertEqual(proposed.metadata["decision_source"], "runtime-observed")

    def test_propose_for_no_longer_evicts_ssd_object(self):
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
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(evict_ssd_capacity_bytes=1),
        )
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-store", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", bytes=1)
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-prefetch", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.PREFETCH, "completed", 2, tier_before="ssd", tier_after="cpu", bytes=1)
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent("run", "event-offload", "req", "prefix", ObjectLevel.PREFIX, "block-7", HookAction.OFFLOAD, "completed", 3, tier_before="cpu", tier_after="ssd", bytes=1)
        ))

        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "evict")

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

    def test_propose_for_keeps_hit_object_with_observed_reuse(self):
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
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(evict_ssd_capacity_bytes=1),
        )
        seed_metadata = {"reuse_ratio": 0.70}
        followup_metadata = {"reuse_ratio": 0.05}
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-store", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", bytes=1, metadata=seed_metadata,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "event-prefetch", "req", "prefix-evict-hit", ObjectLevel.PREFIX, "block-evict-hit",
                HookAction.PREFETCH, "completed", 2, tier_before="ssd", tier_after="cpu", bytes=1, metadata=seed_metadata,
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
                HookAction.OFFLOAD, "completed", 5, tier_before="cpu", tier_after="ssd", bytes=1, metadata=followup_metadata,
            )
        ))

        proposed = controller.propose_for("prefix-evict-hit", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "evict")
        self.assertEqual(proposed.metadata["policy_reuse_frequency"], 0.05)

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


    def _evict_ready_binding(
        self,
        *,
        object_key: str,
        backend_object_id: str,
        spec_id: str,
        binding_id: str = "bind",
    ) -> BackendObjectBinding:
        execution_spec = BackendExecutionSpec(
            spec_id=spec_id,
            binding_id=binding_id,
            binding_generation=1,
            backend_object_id=backend_object_id,
            object_key=object_key,
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
        return BackendObjectBinding(
            "run",
            "req",
            object_key,
            ObjectLevel.PREFIX,
            backend_object_id,
            binding_id,
            execution_spec=execution_spec,
        )

    def _ingest_ssd_object(
        self,
        controller: OnlinePolicyController,
        *,
        object_key: str,
        backend_object_id: str,
        prefetch: bool,
        size: int,
    ) -> None:
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", f"store-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                backend_object_id, HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", bytes=size,
            )
        ))
        if prefetch:
            self.assertTrue(controller.ingest(
                BackendHookEvent(
                    "run", f"prefetch-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                    backend_object_id, HookAction.PREFETCH, "completed", 2,
                    tier_before="ssd", tier_after="cpu", bytes=size,
                )
            ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", f"offload-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                backend_object_id, HookAction.OFFLOAD, "completed", 3,
                tier_before="cpu" if prefetch else "ssd", tier_after="ssd", bytes=size,
            )
        ))

    def _ingest_cpu_object(
        self,
        controller: OnlinePolicyController,
        *,
        object_key: str,
        backend_object_id: str,
        prefetch_waste: bool,
        size: int,
    ) -> None:
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", f"store-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                backend_object_id, HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", bytes=size,
            )
        ))
        # Second store observation so request_count reaches the eviction floor.
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", f"store2-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                backend_object_id, HookAction.CACHE_STORE, "submitted", 4, tier_after="ssd", bytes=size,
            )
        ))
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", f"prefetch-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                backend_object_id, HookAction.PREFETCH, "completed", 2,
                tier_before="ssd", tier_after="cpu", bytes=size,
            )
        ))
        if prefetch_waste:
            self.assertTrue(controller.ingest(
                BackendHookEvent(
                    "run", f"drop-{backend_object_id}", "req", object_key, ObjectLevel.PREFIX,
                    backend_object_id, HookAction.DROP, "completed", 3, tier_before="cpu", bytes=size,
                )
            ))

    def test_evict_pressure_gate_blocks_below_trigger(self):
        binding = self._evict_ready_binding(
            object_key="prefix", backend_object_id="block-7", spec_id="spec-pressure-low",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(evict_cpu_capacity_bytes=100),
        )
        self._ingest_cpu_object(controller, object_key="prefix", backend_object_id="block-7", prefetch_waste=False, size=1)
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "keep")
        self.assertFalse(proposed.metadata["evict_cpu_pressure_over"])
        self.assertLess(proposed.metadata["evict_pressure_snapshot"]["cpu_usage_fraction"], 0.8)

    def test_evict_dispatch_switch_disables_evict(self):
        binding = self._evict_ready_binding(
            object_key="prefix", backend_object_id="block-7", spec_id="spec-switch-off",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                evict_cpu_capacity_bytes=1,
                evict_dispatch_enabled=False,
            ),
        )
        self._ingest_cpu_object(controller, object_key="prefix", backend_object_id="block-7", prefetch_waste=True, size=1)
        proposed = controller.propose_for("prefix", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "keep")
        self.assertFalse(proposed.metadata["evict_ready"])

    def test_layered_eviction_uses_layer_specific_pressure(self):
        binding_cpu = self._evict_ready_binding(
            object_key="prefix-cpu", backend_object_id="block-cpu", spec_id="spec-layer-cpu",
        )
        binding_ssd = self._evict_ready_binding(
            object_key="prefix-ssd", backend_object_id="block-ssd", spec_id="spec-layer-ssd", binding_id="bind-ssd",
        )

        def build(cpu_cap: int, ssd_cap: int) -> OnlinePolicyController:
            bridge = OnlineBackendBridge(
                run_id="run",
                bindings=[binding_cpu, binding_ssd],
                hook_client=object(),
                hook_url="http://127.0.0.1:7900/actions",
                gate=gate(),
            )
            controller = OnlinePolicyController(
                run_id="run",
                workload_id="w",
                bridge=bridge,
                config=OnlinePolicyControllerConfig(
                    evict_cpu_capacity_bytes=cpu_cap,
                    evict_ssd_capacity_bytes=ssd_cap,
                ),
            )
            self._ingest_cpu_object(controller, object_key="prefix-cpu", backend_object_id="block-cpu", prefetch_waste=False, size=1)
            self._ingest_ssd_object(controller, object_key="prefix-ssd", backend_object_id="block-ssd", prefetch=True, size=1)
            return controller

        # CPU layer over pressure, SSD layer not: the global scan only picks the
        # CPU object (target cpu); the SSD object is untouched.
        cpu_over = build(1, 100_000)
        cpu_over.execution_enabled = True
        cpu_results = cpu_over.global_evict_scan(now_ns=1_000)
        self.assertEqual([r[0].object_key for r in cpu_results], ["prefix-cpu"])
        self.assertTrue(all(r[0].target_tier == "cpu" for r in cpu_results))

        # SSD layer over pressure, CPU layer not: the scan only picks the SSD
        # object (target ssd); the CPU object is untouched.
        ssd_over = build(100_000, 1)
        ssd_over.execution_enabled = True
        ssd_results = ssd_over.global_evict_scan(now_ns=1_000)
        self.assertEqual([r[0].object_key for r in ssd_results], ["prefix-ssd"])
        self.assertTrue(all(r[0].target_tier == "ssd" for r in ssd_results))

    def test_pressure_snapshot_reports_usage_fractions(self):
        binding_a = self._evict_ready_binding(
            object_key="prefix-a", backend_object_id="block-a", spec_id="spec-pressure-a",
        )
        binding_b = self._evict_ready_binding(
            object_key="prefix-b", backend_object_id="block-b", spec_id="spec-pressure-b", binding_id="bind-b",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding_a, binding_b],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(evict_ssd_capacity_bytes=2),
        )
        self._ingest_ssd_object(controller, object_key="prefix-a", backend_object_id="block-a", prefetch=False, size=1)
        snapshot = controller._pressure_snapshot()
        self.assertAlmostEqual(snapshot["ssd_usage_fraction"], 0.5)
        self.assertFalse(snapshot["over_pressure"])
        self.assertTrue(controller.ingest(
            BackendHookEvent(
                "run", "store-block-b", "req", "prefix-b", ObjectLevel.PREFIX,
                "block-b", HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", bytes=2,
            )
        ))
        snapshot = controller._pressure_snapshot()
        self.assertGreaterEqual(snapshot["ssd_usage_fraction"], 1.0)
        self.assertTrue(snapshot["over_pressure"])

    def test_global_evict_scan_ranks_cold_victims_and_dispatches(self):
        binding_a = self._evict_ready_binding(
            object_key="prefix-a", backend_object_id="block-a", spec_id="spec-scan-a",
        )
        binding_b = self._evict_ready_binding(
            object_key="prefix-b", backend_object_id="block-b", spec_id="spec-scan-b", binding_id="bind-b",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding_a, binding_b],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                evict_cpu_capacity_bytes=1,
                global_evict_scan_min_interval_s=0.0,
            ),
        )
        controller.execution_enabled = True
        self._ingest_cpu_object(controller, object_key="prefix-a", backend_object_id="block-a", prefetch_waste=True, size=1)
        self._ingest_cpu_object(controller, object_key="prefix-b", backend_object_id="block-b", prefetch_waste=False, size=1)
        results = controller.global_evict_scan(now_ns=1_000)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            [item[0].object_key for item in results],
            ["prefix-a", "prefix-b"],
        )
        self.assertTrue(all(item[0].predicted_action == "evict" for item in results))
        self.assertTrue(all(item[0].metadata["decision_source"] == "global_pressure_scan" for item in results))
        self.assertGreater(
            results[0][0].metadata["evict_cold_score"],
            results[1][0].metadata["evict_cold_score"],
        )
        self.assertIn("evict_pressure_snapshot", results[0][0].metadata)

    def test_global_evict_scan_skips_without_pressure(self):
        binding = self._evict_ready_binding(
            object_key="prefix", backend_object_id="block-7", spec_id="spec-scan-nopressure",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(run_id="run", workload_id="w", bridge=bridge)
        controller.execution_enabled = True
        self._ingest_cpu_object(controller, object_key="prefix", backend_object_id="block-7", prefetch_waste=True, size=1)
        self.assertEqual(controller.global_evict_scan(now_ns=1_000), [])

    def test_global_evict_scan_deduplicates_completed_source_until_repopulated(self):
        binding = self._evict_ready_binding(
            object_key="prefix-a", backend_object_id="block-a", spec_id="spec-terminal",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                evict_cpu_capacity_bytes=1,
                global_evict_scan_min_interval_s=0.0,
            ),
        )
        controller.execution_enabled = True
        self._ingest_cpu_object(
            controller,
            object_key="prefix-a",
            backend_object_id="block-a",
            prefetch_waste=True,
            size=1,
        )
        state = controller.profile_store.object_state("block-a")
        assert state is not None
        generation = int(state.get("binding_generation") or 0)
        bridge.dispatch = lambda decision: RuntimeActionResult(
            "executed",
            "evicted",
            receipt=BackendActionReceipt(
                "run", "command-1", "receipt-1", "bind", "block-a",
                HookAction.EVICT, "completed", 10,
                tier_before="cpu", tier_after="ssd",
                binding_generation=generation,
                decision_id=decision.decision_id,
                request_id="req",
            ),
        )
        decision = OfflineEvictionDecision(
            run_id="run",
            decision_id="terminal-evict",
            request_id="req",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            predicted_action="evict",
            target_tier="cpu",
            metadata={
                "binding_id": "bind",
                "binding_generation": generation,
                "backend_object_id": "block-a",
            },
        )
        self.assertEqual(controller.dispatch(decision).status, "executed")
        self.assertIn(
            ("block-a", generation, "cpu"),
            controller._completed_evict_sources,
        )

        # A late release callback can carry the old CPU tier.  It must not
        # make the completed source eligible for another not_found retry.
        self.assertTrue(controller.ingest(BackendHookEvent(
            "run", "late-release", "req", "prefix-a", ObjectLevel.PREFIX,
            "block-a", HookAction.RELEASE, "completed", 11,
            tier_before="cpu", tier_after="cpu", binding_generation=generation,
        )))
        self.assertEqual(controller.global_evict_scan(now_ns=12), [])

        # A genuine later placement into CPU reopens that source tier.
        self.assertTrue(controller.ingest(BackendHookEvent(
            "run", "new-prefetch", "req", "prefix-a", ObjectLevel.PREFIX,
            "block-a", HookAction.PREFETCH, "completed", 13,
            tier_before="ssd", tier_after="cpu", bytes=1, binding_generation=generation,
        )))
        self.assertEqual(len(controller.global_evict_scan(now_ns=14)), 1)

    def test_global_evict_scan_is_deterministic(self):
        binding_a = self._evict_ready_binding(
            object_key="prefix-a", backend_object_id="block-a", spec_id="spec-scan-det-a",
        )
        binding_b = self._evict_ready_binding(
            object_key="prefix-b", backend_object_id="block-b", spec_id="spec-scan-det-b", binding_id="bind-b",
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding_a, binding_b],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            config=OnlinePolicyControllerConfig(
                evict_cpu_capacity_bytes=1,
                global_evict_scan_min_interval_s=0.0,
            ),
        )
        controller.execution_enabled = True
        self._ingest_cpu_object(controller, object_key="prefix-a", backend_object_id="block-a", prefetch_waste=True, size=1)
        self._ingest_cpu_object(controller, object_key="prefix-b", backend_object_id="block-b", prefetch_waste=False, size=1)
        first = controller.global_evict_scan(now_ns=1_000)
        second = controller.global_evict_scan(now_ns=2_000)
        self.assertEqual(
            [item[0].object_key for item in first],
            [item[0].object_key for item in second],
        )


if __name__ == "__main__":
    unittest.main()
