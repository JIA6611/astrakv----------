import unittest

from astrakv.runtime.backend_binding_registry import (
    BackendBindingRegistry,
    RequestContext,
)
from astrakv.runtime.backend_hook import HookAction
from astrakv.runtime.eviction import ObjectLevel


class BackendBindingRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BackendBindingRegistry(
            run_id="run-a",
            engine_instance_id="engine-1",
            worker_id="worker-0",
        )

    def test_conflicting_logical_object_during_active_use_is_observational_only(self) -> None:
        first = self.registry.observe(
            key=("prefix", 17),
            action=HookAction.CACHE_STORE,
            status="submitted",
            context=RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX),
        )
        second = self.registry.observe(
            key=("prefix", 17),
            action=HookAction.CACHE_HIT,
            status="completed",
            context=RequestContext("run-a", "req-b", "prefix-b", ObjectLevel.PREFIX),
        )

        self.assertIsNone(second.binding)
        self.assertEqual(second.record["status"], "binding_identity_conflict")
        self.assertEqual(self.registry.active_request_ids(first.binding.binding_id), ("req-a",))
        self.assertEqual(first.event.action, HookAction.CACHE_STORE)
        self.assertEqual(first.event.status, "submitted")
        self.assertFalse(self.registry.eligible_for_bridge(first.binding.binding_id, first.binding.binding_generation))

    def test_idle_physical_key_allocates_a_new_binding_id_for_new_logical_object(self) -> None:
        first_context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        first = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", first_context)
        self.registry.complete_operation(
            "key-a",
            HookAction.CACHE_STORE,
            "completed",
            first_context,
            first.event.metadata["operation_lease"],
        )
        self.registry.observe("key-a", HookAction.RELEASE, "completed", first_context)

        second_context = RequestContext("run-a", "req-b", "prefix-b", ObjectLevel.PREFIX)
        second = self.registry.observe("key-a", HookAction.CACHE_HIT, "completed", second_context)

        self.assertIsNotNone(second.binding)
        self.assertNotEqual(second.binding.binding_id, first.binding.binding_id)
        self.assertEqual(second.binding.binding_generation, 1)
        self.assertEqual(second.binding.object_key, "prefix-b")
        self.assertIsNone(self.registry.current_binding(
            binding_id=first.binding.binding_id,
            binding_generation=first.binding.binding_generation,
            request_id="req-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
        ))
        self.assertEqual(self.registry.binding_status(first.binding.binding_id), "replaced")
        self.assertEqual(self.registry.snapshot(first.binding.binding_id)["lifecycle"], "replaced")

    def test_release_and_reuse_allocates_a_new_binding_id_and_rejects_the_old_one(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        first = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"])
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context)
        next_submit = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        second = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, next_submit.event.metadata["operation_lease"])

        self.assertNotEqual(second.binding.binding_id, first.binding.binding_id)
        self.assertEqual(second.binding.binding_generation, 1)
        self.assertFalse(self.registry.eligible_for_bridge(first.binding.binding_id, first.binding.binding_generation))
        self.assertFalse(self.registry.eligible_for_bridge(second.binding.binding_id, second.binding.binding_generation))

    def test_submitted_store_is_not_active_until_completed(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)

        self.assertFalse(self.registry.is_active(submitted.binding.binding_id))
        completed = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"])
        self.assertTrue(self.registry.is_active(completed.binding.binding_id))
        self.assertEqual(self.registry.snapshot(completed.binding.binding_id)["pending_io"], 0)

    def test_store_completion_requires_the_submitted_lease_and_rejects_a_stale_binding_lease(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        lease = submitted.event.metadata["operation_lease"]
        with self.assertRaisesRegex(ValueError, "lease"):
            self.registry.observe("key-a", HookAction.CACHE_STORE, "completed", context)
        completed = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, lease)
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context)
        next_submit = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)

        self.assertNotEqual(next_submit.binding.binding_id, completed.binding.binding_id)
        self.assertEqual(next_submit.binding.binding_generation, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, lease)

    def test_action_authorization_rejects_active_associations_pins_and_pending_io(self) -> None:
        context_a = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        context_b = RequestContext("run-a", "req-b", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context_a)
        binding = submitted.binding

        self.assertFalse(self.registry.authorize_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a",
            object_key="prefix-a", object_level=ObjectLevel.PREFIX,
        ))
        self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context_a, submitted.event.metadata["operation_lease"])
        self.registry.observe("key-a", HookAction.CACHE_HIT, "completed", context_b)
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context_a)
        self.assertFalse(self.registry.authorize_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a",
            object_key="prefix-a", object_level=ObjectLevel.PREFIX,
        ))
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context_b)
        self.assertTrue(self.registry.authorize_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a",
            object_key="prefix-a", object_level=ObjectLevel.PREFIX,
        ))

    def test_action_reservation_is_atomic_and_blocks_io_until_exact_terminal_lease(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        completed = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"])
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context)
        binding = completed.binding
        lease = self.registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a",
            object_key="prefix-a", object_level=ObjectLevel.PREFIX,
        )

        self.assertIsNotNone(lease)
        self.assertFalse(self.registry.eligible_for_bridge(binding.binding_id, binding.binding_generation))
        with self.assertRaisesRegex(ValueError, "reservation"):
            self.registry.observe("key-a", HookAction.CACHE_HIT, "completed", context)
        self.assertFalse(self.registry.complete_action("wrong", command_id="command-1", status="completed"))
        self.assertTrue(self.registry.consume_action_reservation(lease, command_id="command-1"))
        self.assertTrue(self.registry.complete_action(lease, command_id="command-1", status="completed"))
        self.assertEqual(self.registry.snapshot(binding.binding_id)["lifecycle"], "dropped")

    def test_reservation_is_consumed_once_and_expiry_restores_eligibility_without_drop(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        binding = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"]).binding
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context)
        lease = self.registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a", object_key="prefix-a",
            object_level=ObjectLevel.PREFIX, deadline_ns=100,
        )

        self.assertTrue(self.registry.consume_action_reservation(lease, command_id="command-1", now_ns=99))
        self.assertFalse(self.registry.consume_action_reservation(lease, command_id="command-1", now_ns=99))
        self.assertFalse(self.registry.complete_action(lease, command_id="wrong", status="completed"))
        self.assertTrue(self.registry.complete_action(lease, command_id="command-1", status="failed"))
        self.assertTrue(self.registry.eligible_for_bridge(binding.binding_id, binding.binding_generation))

        expiring = self.registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a", object_key="prefix-a",
            object_level=ObjectLevel.PREFIX, deadline_ns=100,
        )
        self.assertEqual(self.registry.reap_expired_reservations(now_ns=101), (expiring,))
        self.assertEqual(self.registry.reservation_state(expiring), "expired")
        self.assertEqual(self.registry.snapshot(binding.binding_id)["lifecycle"], "released")
        self.assertTrue(self.registry.eligible_for_bridge(binding.binding_id, binding.binding_generation))

    def test_contextless_hit_is_deferred_while_known_binding_is_reserved(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        submitted = self.registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        binding = self.registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"]).binding
        self.registry.observe("key-a", HookAction.RELEASE, "completed", context)
        self.registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="req-a", object_key="prefix-a", object_level=ObjectLevel.PREFIX,
        )

        observed = self.registry.observe("key-a", HookAction.CACHE_HIT, "completed", None)
        self.assertEqual(observed.record["status"], "deferred")
        self.assertTrue(observed.record["metadata"]["deferred_by_reservation"])
        self.assertEqual(self.registry.snapshot(binding.binding_id)["lifecycle"], "released")

    def test_unknown_key_stays_observational_and_has_no_bridge_binding(self) -> None:
        observed = self.registry.observe("unknown", HookAction.CACHE_MISS, "completed", None)

        self.assertIsNone(observed.binding)
        self.assertTrue(observed.record["metadata"]["observational_only"])
        self.assertFalse(observed.record["metadata"]["bridge_eligible"])

    def test_runtime_request_metadata_is_preserved_on_binding_records(self) -> None:
        context = RequestContext("run-a", "req-a", "prefix-a", ObjectLevel.PREFIX)
        observed = self.registry.observe(
            "key-a",
            HookAction.CACHE_STORE,
            "submitted",
            context,
            metadata={"runtime_reqmeta_id": "reqmeta-a"},
        )

        self.assertEqual(observed.binding.metadata["runtime_reqmeta_id"], "reqmeta-a")
        self.assertEqual(observed.binding_record["metadata"]["runtime_reqmeta_id"], "reqmeta-a")


if __name__ == "__main__":
    unittest.main()
