import unittest

from astrakv.runtime.backend_hook import (
    BackendActionCommand,
    BackendActionReceipt,
    BackendExecutionSpec,
    BackendHookEvent,
    BackendObjectBinding,
    HookAction,
)
from astrakv.runtime.eviction import ObjectLevel


class BackendHookContractTests(unittest.TestCase):
    def test_binding_generation_is_appended_after_legacy_binding_fields(self) -> None:
        binding = BackendObjectBinding(
            "run-1", "request-1", "prefix-1", ObjectLevel.PREFIX,
            "vllm:block:7", "binding-1", False, {"source": "legacy"},
        )

        self.assertFalse(binding.verified)
        self.assertEqual(binding.metadata, {"source": "legacy"})
        self.assertEqual(binding.binding_generation, 1)
    def test_verified_binding_matches_only_the_same_real_backend_object(self) -> None:
        execution_spec = BackendExecutionSpec(
            spec_id="execspec-1",
            binding_id="binding-1",
            binding_generation=1,
            backend_object_id="vllm:block:7",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="runtime-owner",
            owner_channel="owner-local",
            key_identity="key-identity-1",
            lifecycle="released",
            actions={"drop": {"status": "ready"}},
        )
        binding = BackendObjectBinding(
            run_id="run-1",
            request_id="request-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="vllm:block:7",
            binding_id="binding-1",
            execution_spec=execution_spec,
        )
        event = BackendHookEvent(
            run_id="run-1",
            event_id="event-1",
            request_id="request-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="vllm:block:7",
            action=HookAction.CACHE_STORE,
            status="completed",
            timestamp_ns=1,
            metadata={"binding_id": "binding-1"},
        )

        self.assertTrue(binding.matches_event(event))
        self.assertFalse(binding.matches_event(event.with_backend_object_id("vllm:block:8")))
        self.assertEqual(binding.to_record()["execution_spec"]["spec_id"], "execspec-1")

    def test_invalid_event_record_is_rejected_before_it_can_become_online_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend_object_id"):
            BackendHookEvent.from_record({
                "run_id": "run-1",
                "event_id": "event-1",
                "request_id": "request-1",
                "object_key": "prefix-1",
                "object_level": "prefix",
                "action": "cache_store",
                "status": "completed",
                "timestamp_ns": 1,
            })

    def test_binding_id_mismatch_rejects_an_event_from_a_replaced_binding(self) -> None:
        binding = BackendObjectBinding(
            run_id="run-1",
            request_id="request-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="vllm:block:7",
            binding_id="binding-1",
            binding_generation=2,
        )
        stale_event = BackendHookEvent(
            run_id="run-1",
            event_id="event-1",
            request_id="request-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="vllm:block:7",
            action=HookAction.CACHE_STORE,
            status="completed",
            timestamp_ns=1,
            binding_generation=1,
            metadata={"binding_id": "binding-2"},
        )

        self.assertFalse(binding.matches_event(stale_event))

    def test_binding_generation_defaults_to_one_and_must_be_positive(self) -> None:
        binding = BackendObjectBinding(
            run_id="run-1",
            request_id="request-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="vllm:block:7",
            binding_id="binding-1",
        )

        self.assertEqual(binding.binding_generation, 1)
        with self.assertRaisesRegex(ValueError, "binding_generation must be positive"):
            BackendObjectBinding(
                run_id="run-1",
                request_id="request-1",
                object_key="prefix-1",
                object_level=ObjectLevel.PREFIX,
                backend_object_id="vllm:block:7",
                binding_id="binding-1",
                binding_generation=0,
            )

    def test_command_and_receipt_ignore_binding_generation_for_runtime_identity(self) -> None:
        command = BackendActionCommand(
            "run-1", "command-1", "decision-1", "request-1", "prefix-1",
            ObjectLevel.PREFIX, "binding-1", "vllm:block:7", HookAction.DROP, 1,
            binding_generation=2,
        )
        receipt = BackendActionReceipt(
            "run-1", "command-1", "receipt-1", "binding-1", "vllm:block:7",
            HookAction.DROP, "completed", 2, binding_generation=1,
        )

        self.assertEqual(command.to_record()["binding_generation"], 2)
        self.assertEqual(BackendActionCommand.from_record(command.to_record()), command)
        self.assertEqual(BackendActionReceipt.from_record(receipt.to_record()), receipt)
        self.assertTrue(receipt.matches_command(command))

    def test_rejected_receipt_round_trips_audit_fields(self) -> None:
        receipt = BackendActionReceipt(
            "run-1",
            "command-1",
            "receipt-1",
            "binding-1",
            "object-1",
            HookAction.DROP,
            "rejected",
            1,
            binding_generation=2,
            decision_id="decision-1",
            request_id="request-1",
            rejection_reason="runtime_execution_gate:byte_budget",
        )

        record = receipt.to_record()
        self.assertEqual(record["decision_id"], "decision-1")
        self.assertEqual(record["request_id"], "request-1")
        self.assertEqual(record["rejection_reason"], "runtime_execution_gate:byte_budget")
        self.assertEqual(BackendActionReceipt.from_record(record), receipt)

    def test_wire_records_require_binding_generation(self) -> None:
        records = [
            (BackendObjectBinding, BackendObjectBinding(
                "run-1", "request-1", "prefix-1", ObjectLevel.PREFIX, "object-1", "binding-1",
            ).to_record()),
            (BackendHookEvent, BackendHookEvent(
                "run-1", "event-1", "request-1", "prefix-1", ObjectLevel.PREFIX, "object-1",
                HookAction.CACHE_STORE, "completed", 1,
            ).to_record()),
            (BackendActionCommand, BackendActionCommand(
                "run-1", "command-1", "decision-1", "request-1", "prefix-1", ObjectLevel.PREFIX,
                "binding-1", "object-1", HookAction.DROP, 1,
            ).to_record()),
            (BackendActionReceipt, BackendActionReceipt(
                "run-1", "command-1", "receipt-1", "binding-1", "object-1", HookAction.DROP,
                "completed", 1,
            ).to_record()),
        ]
        for record_type, record in records:
            record.pop("binding_generation")
            with self.subTest(record_type=record_type.__name__):
                with self.assertRaisesRegex(ValueError, "binding_generation is required"):
                    record_type.from_record(record)

    def test_execution_spec_round_trips_with_binding_identity(self) -> None:
        spec = BackendExecutionSpec(
            spec_id="execspec-1",
            binding_id="binding-1",
            binding_generation=2,
            backend_object_id="object-1",
            object_key="prefix-1",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="lmcache047-runtime-owner",
            owner_channel="lmcache047-owner-local",
            key_identity="lmcache-key-identity",
            lifecycle="released",
            actions={
                "drop": {"status": "ready", "executor": "DropExecutor"},
                "offload": {"status": "blocked"},
            },
            metadata={"connector_name": "lmcache-vllm-v1"},
        )
        binding = BackendObjectBinding(
            "run-1",
            "request-1",
            "prefix-1",
            ObjectLevel.PREFIX,
            "object-1",
            "binding-1",
            binding_generation=2,
            execution_spec=spec,
        )

        self.assertEqual(BackendObjectBinding.from_record(binding.to_record()), binding)
        self.assertEqual(BackendExecutionSpec.from_record(spec.to_record()), spec)


if __name__ == "__main__":
    unittest.main()
