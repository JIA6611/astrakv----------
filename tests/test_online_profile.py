import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.backend_hook import BackendHookEvent, HookAction
from astrakv.runtime.backend_hook import BackendActionReceipt
from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision, RuntimeActionResult, RuntimeEvictionEvent
from astrakv.runtime.online_profile import OnlineProfileStore


def event(event_id: str, *, action: HookAction = HookAction.CACHE_HIT, timestamp_ns: int = 1) -> BackendHookEvent:
    return BackendHookEvent(
        run_id="run-a", event_id=event_id, request_id="request-a", object_key="prefix-a",
        object_level=ObjectLevel.PREFIX, backend_object_id="object-a", action=action,
        status="completed", timestamp_ns=timestamp_ns, bytes=64,
    )


class OnlineProfileStoreTests(unittest.TestCase):
    def test_replay_is_idempotent_and_checkpoint_restores_event_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "online-state.json"
            store = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            self.assertTrue(store.consume(event("event-1")))
            self.assertFalse(store.consume(event("event-1")))
            self.assertEqual(store.snapshot()["event_count"], 1)
            store.checkpoint()

            restored = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            self.assertEqual(restored.snapshot()["last_event_id"], "event-1")
            self.assertFalse(restored.consume(event("event-1")))
            self.assertTrue(restored.consume(event("event-2", action=HookAction.CACHE_STORE, timestamp_ns=2)))
            self.assertEqual(restored.object_state("object-a")["cache_stores"], 1)

    def test_replay_with_conflicting_event_payload_is_rejected(self) -> None:
        store = OnlineProfileStore(run_id="run-a")
        self.assertTrue(store.consume(event("event-1", timestamp_ns=1)))
        with self.assertRaisesRegex(ValueError, "conflicting replay"):
            store.consume(event("event-1", timestamp_ns=2))

    def test_dispatch_feedback_is_checkpointed_and_replay_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "online-state.json"
            store = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            store.consume(event("event-1", action=HookAction.CACHE_STORE, timestamp_ns=1))
            decision = OfflineEvictionDecision(
                run_id="run-a",
                decision_id="decision-1",
                request_id="request-a",
                object_key="prefix-a",
                object_level=ObjectLevel.PREFIX,
                predicted_action="drop",
            )
            result = RuntimeActionResult(
                "executed",
                "backend Hook acknowledged action",
                RuntimeEvictionEvent(
                    run_id="run-a",
                    runtime_event_id="runtime-1",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    actual_action="drop",
                    tier_before="gpu",
                    tier_after="ssd",
                    bytes=64,
                    timestamp_ns=2,
                    status="completed",
                    provenance="runtime_structured",
                    metadata={
                        "backend_object_id": "object-a",
                        "binding_generation": 1,
                        "command_id": "command-1",
                        "receipt_id": "receipt-1",
                        "receipt_status": "completed",
                    },
                ),
            )
            self.assertTrue(
                store.record_dispatch(
                    decision,
                    result,
                    execution_enabled=True,
                    breaker_state={"state": "closed", "failures": 0},
                )
            )
            self.assertFalse(
                store.record_dispatch(
                    decision,
                    result,
                    execution_enabled=True,
                    breaker_state={"state": "closed", "failures": 0},
                )
            )
            self.assertEqual(store.snapshot()["dispatch_count"], 1)
            self.assertEqual(store.controller_state()["last_receipt_id"], "receipt-1")
            self.assertEqual(store.object_state("object-a")["last_action"], "drop")
            store.checkpoint()

            restored = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            self.assertEqual(restored.snapshot()["dispatch_count"], 1)
            self.assertEqual(restored.controller_state()["breaker"]["state"], "closed")
            self.assertFalse(
                restored.record_dispatch(
                    decision,
                    result,
                    execution_enabled=True,
                    breaker_state={"state": "closed", "failures": 0},
                )
            )

    def test_dispatch_replay_with_conflicting_payload_is_rejected(self) -> None:
        store = OnlineProfileStore(run_id="run-a")
        decision = OfflineEvictionDecision(
            run_id="run-a",
            decision_id="decision-1",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            predicted_action="drop",
        )
        self.assertTrue(
            store.record_dispatch(
                decision,
                RuntimeActionResult("advisory_only", "advisory"),
                execution_enabled=False,
                breaker_state={"state": "closed"},
            )
        )
        with self.assertRaisesRegex(ValueError, "conflicting replay"):
            store.record_dispatch(
                decision,
                RuntimeActionResult("blocked_by_capability_preflight", "rejected"),
                execution_enabled=True,
                breaker_state={"state": "closed"},
            )

    def test_rejected_receipt_is_checkpointed_without_runtime_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "online-state.json"
            store = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            decision = OfflineEvictionDecision(
                run_id="run-a",
                decision_id="decision-1",
                request_id="request-a",
                object_key="prefix-a",
                object_level=ObjectLevel.PREFIX,
                predicted_action="drop",
            )
            result = RuntimeActionResult(
                "blocked_by_runtime_gate:byte_budget",
                "runtime execution gate rejected command",
                receipt=BackendActionReceipt(
                    "run-a",
                    "run-a:decision-1",
                    "run-a:decision-1:rejected",
                    "binding-a",
                    "object-a",
                    HookAction.DROP,
                    "rejected",
                    2,
                    binding_generation=3,
                    decision_id="decision-1",
                    request_id="request-a",
                    rejection_reason="runtime_execution_gate:byte_budget",
                ),
            )

            self.assertTrue(
                store.record_dispatch(
                    decision,
                    result,
                    execution_enabled=True,
                    breaker_state={"state": "closed"},
                )
            )
            store.checkpoint()

            restored = OnlineProfileStore(run_id="run-a", checkpoint_path=checkpoint)
            controller_state = restored.controller_state()
            object_state = restored.object_state("object-a")
            assert object_state is not None
            self.assertEqual(controller_state["last_receipt_id"], "run-a:decision-1:rejected")
            self.assertEqual(controller_state["last_rejection_reason"], "runtime_execution_gate:byte_budget")
            self.assertIsNone(controller_state["last_runtime_event_id"])
            self.assertEqual(object_state["binding_generation"], 3)
            self.assertEqual(object_state["last_receipt_status"], "rejected")
            self.assertEqual(object_state["last_rejection_reason"], "runtime_execution_gate:byte_budget")

    def test_receipt_only_completed_evict_updates_resident_tier(self) -> None:
        """Independent-channel evicts carry a receipt without a runtime event;
        the tracked resident tier must still move cpu -> ssd so the global
        evict scan does not re-select the same object (not_found churn)."""
        store = OnlineProfileStore(run_id="run-a")
        decision = OfflineEvictionDecision(
            run_id="run-a",
            decision_id="decision-evict-1",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            predicted_action="evict",
        )
        completed = RuntimeActionResult(
            "executed",
            "evicted",
            receipt=BackendActionReceipt(
                "run-a",
                "run-a:decision-evict-1",
                "run-a:decision-evict-1:terminal",
                "binding-a",
                "object-a",
                HookAction.EVICT,
                "completed",
                2,
                tier_before="cpu",
                tier_after="ssd",
                bytes=64,
                binding_generation=1,
                decision_id="decision-evict-1",
                request_id="request-a",
                metadata={
                    "source_location": "LocalCPUBackend",
                    "target_location": "LocalDiskBackend",
                },
            ),
        )
        self.assertTrue(
            store.record_dispatch(decision, completed, execution_enabled=True, breaker_state={})
        )
        state = store.object_state("object-a")
        assert state is not None
        self.assertEqual(state["current_tier"], "ssd")

        # A later not_found re-attempt (stale candidate) must not move the
        # tier back or churn further.
        not_found_decision = OfflineEvictionDecision(
            run_id="run-a",
            decision_id="decision-evict-2",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            predicted_action="evict",
        )
        not_found = RuntimeActionResult(
            "not_found",
            "object no longer resident in source tier",
            receipt=BackendActionReceipt(
                "run-a",
                "run-a:decision-evict-2",
                "run-a:decision-evict-2:terminal",
                "binding-a",
                "object-a",
                HookAction.EVICT,
                "not_found",
                3,
                tier_before="cpu",
                tier_after="ssd",
                bytes=0,
                binding_generation=1,
                decision_id="decision-evict-2",
                request_id="request-a",
                metadata={
                    "source_location": "LocalCPUBackend",
                    "target_location": "LocalDiskBackend",
                },
            ),
        )
        self.assertTrue(
            store.record_dispatch(
                not_found_decision,
                not_found,
                execution_enabled=True,
                breaker_state={},
            )
        )
        self.assertEqual(store.object_state("object-a")["current_tier"], "ssd")

    def test_profile_tracks_tier_transitions_prefetch_waste_and_reaccess(self) -> None:
        store = OnlineProfileStore(run_id="run-a")
        self.assertTrue(
            store.consume(
                BackendHookEvent(
                    run_id="run-a",
                    event_id="event-store",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="object-a",
                    action=HookAction.CACHE_STORE,
                    status="submitted",
                    timestamp_ns=1,
                    tier_after="ssd",
                    bytes=64,
                )
            )
        )
        self.assertTrue(
            store.consume(
                BackendHookEvent(
                    run_id="run-a",
                    event_id="event-prefetch",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="object-a",
                    action=HookAction.PREFETCH,
                    status="completed",
                    timestamp_ns=2,
                    tier_before="ssd",
                    tier_after="cpu",
                    bytes=64,
                )
            )
        )
        self.assertTrue(
            store.consume(
                BackendHookEvent(
                    run_id="run-a",
                    event_id="event-evict",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="object-a",
                    action=HookAction.EVICT,
                    status="completed",
                    timestamp_ns=3,
                    tier_before="ssd",
                    tier_after="none",
                    bytes=64,
                )
            )
        )
        self.assertTrue(
            store.consume(
                BackendHookEvent(
                    run_id="run-a",
                    event_id="event-load",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="object-a",
                    action=HookAction.CACHE_LOAD,
                    status="completed",
                    timestamp_ns=4,
                    tier_before="ssd",
                    tier_after="gpu",
                    bytes=64,
                    metadata={"load_latency_ns": 50, "load_target_id": "target-a", "runtime_reqmeta_id": "reqmeta-a"},
                )
            )
        )
        self.assertTrue(
            store.consume(
                BackendHookEvent(
                    run_id="run-a",
                    event_id="event-release",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="object-a",
                    action=HookAction.RELEASE,
                    status="completed",
                    timestamp_ns=5,
                    tier_before="gpu",
                    tier_after="unknown",
                    bytes=64,
                )
            )
        )
        state = store.object_state("object-a")
        assert state is not None
        self.assertEqual(state["request_count"], 1)
        self.assertEqual(state["prefetch_success_count"], 1)
        self.assertEqual(state["prefetch_waste_count"], 1)
        self.assertEqual(state["reuse_count"], 1)
        self.assertEqual(state["eviction_reaccess_count"], 1)
        self.assertEqual(state["current_tier"], "gpu")
        self.assertEqual(state["last_access_time_ns"], 4)
        self.assertEqual(state["last_load_target_id"], "target-a")
        self.assertEqual(state["last_runtime_reqmeta_id"], "reqmeta-a")
        self.assertEqual(state["load_latency_ema"], 50.0)


if __name__ == "__main__":
    unittest.main()
