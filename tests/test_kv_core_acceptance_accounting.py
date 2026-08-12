import json
import unittest

from scripts.reporting.validate_kv_core_acceptance import (
    _association_index,
    validate_prefetch_disabled,
    validate_variant_request_associations,
    validate_receipts,
    validate_request_accounting,
)


class KVCoreAcceptanceAccountingTests(unittest.TestCase):
    def _request_result(self):
        return {
            "request_id": "logical-request",
            "runtime_association_status": "linked",
            "runtime_request_id": "chatcmpl-logical-request-1",
            "runtime_event_id": "runtime-context:chatcmpl-logical-request-1",
        }

    def _association_row(self):
        return {
            "status": "associated",
            "run_id": "run",
            "request_id": "logical-request",
            "request_nonce": "nonce",
            "runtime_request_id": "chatcmpl-logical-request-1",
            "runtime_event_id": "runtime-context:chatcmpl-logical-request-1",
            "session_id": "session",
            "expires_at_ns": 10,
            "mac": "authenticated",
        }

    def _association(self):
        errors = []
        index = _association_index([self._association_row()], {"logical-request"}, errors)
        self.assertEqual(errors, [])
        return index

    def test_e1_requires_matched_linked_runtime_association(self):
        errors = []
        index = validate_variant_request_associations(
            [self._request_result()], [self._association_row()], errors,
            expected_run_id="run",
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            index["chatcmpl-logical-request-1"]["logical_request_id"],
            "logical-request",
        )

    def test_receipts_accept_partial_prefix_identity_after_churn(self) -> None:
        accounting = [{
            "request_id": "logical-request",
            "physical_object_id": "long-object",
            "binding_generation": 1,
            "native_key": json.dumps(["c0", "c1", "c2"]),
            "compatibility_identity": "long-id",
            "prefix_hash": "long-hash",
            "allocated_external_tokens": 16,
            "lookup_hit_tokens": 32,
            "actual_loaded_tokens": 16,
            "requested_prefix_tokens": 32,
            "locally_cached_tokens": 0,
            "missing_tokens": 16,
            "unallocated_recompute_tokens": 16,
            "load_shortfall_tokens": 0,
        }]
        receipt = {
            "request_id": "logical-request",
            "physical_object_id": "short-object",
            "binding_generation": 1,
            "native_key": json.dumps(["c0"]),
            "compatibility_identity": "short-id",
            "prefix_hash": "short-hash",
            "allocated_external_tokens": 16,
            "lookup_hit_tokens": 32,
            "actual_loaded_tokens": 16,
            "requested_prefix_tokens": 32,
            "locally_cached_tokens": 0,
            "missing_tokens": 16,
            "unallocated_recompute_tokens": 16,
            "load_shortfall_tokens": 0,
        }
        errors: list[str] = []
        validate_receipts([receipt], accounting, errors)
        self.assertEqual(errors, [])

    def test_receipts_reject_non_prefix_identity(self) -> None:
        accounting = [{
            "request_id": "logical-request",
            "physical_object_id": "long-object",
            "binding_generation": 1,
            "native_key": json.dumps(["c0", "c1"]),
            "compatibility_identity": "long-id",
            "prefix_hash": "long-hash",
            "allocated_external_tokens": 16,
            "lookup_hit_tokens": 32,
            "actual_loaded_tokens": 16,
            "requested_prefix_tokens": 32,
            "locally_cached_tokens": 0,
            "missing_tokens": 16,
            "unallocated_recompute_tokens": 16,
            "load_shortfall_tokens": 0,
        }]
        receipt = {
            "request_id": "logical-request",
            "physical_object_id": "other-object",
            "binding_generation": 1,
            "native_key": json.dumps(["c9"]),
            "compatibility_identity": "other-id",
            "prefix_hash": "other-hash",
            "allocated_external_tokens": 16,
            "lookup_hit_tokens": 32,
            "actual_loaded_tokens": 16,
            "requested_prefix_tokens": 32,
            "locally_cached_tokens": 0,
            "missing_tokens": 16,
            "unallocated_recompute_tokens": 16,
            "load_shortfall_tokens": 0,
        }
        errors: list[str] = []
        validate_receipts([receipt], accounting, errors)
        self.assertIn("receipt_identity_mismatch:native_key:logical-request", errors)

    def test_e1_rejects_unlinked_or_missing_runtime_association(self):
        errors = []
        result = self._request_result()
        result["runtime_association_status"] = "unlinked"
        index = validate_variant_request_associations(
            [result], [], errors, expected_run_id="run",
        )
        self.assertEqual(index, {})
        self.assertIn("request_association_artifact_missing", errors)
        self.assertIn("request_result_association_not_linked:logical-request", errors)

    def test_e1_rejects_event_id_not_emitted_by_runtime(self):
        errors = []
        result = self._request_result()
        result["runtime_event_id"] = "runtime-context:forged"
        validate_variant_request_associations(
            [result], [self._association_row()], errors, expected_run_id="run",
        )
        self.assertEqual(
            errors,
            ["request_result_association_event_mismatch:logical-request"],
        )

    def _accounting(self):
        return {
            "request_id": "chatcmpl-logical-request-1",
            "native_request_id": "chatcmpl-logical-request-1",
            "physical_object_id": "physical",
            "binding_generation": 1,
            "native_key": "native-key",
            "compatibility_identity": "compatibility",
            "prefix_hash": "prefix",
            "requested_prefix_tokens": 8,
            "locally_cached_tokens": 0,
            "lookup_hit_tokens": 8,
            "allocated_external_tokens": 8,
            "actual_loaded_tokens": 8,
            "missing_tokens": 0,
            "unallocated_recompute_tokens": 0,
            "load_shortfall_tokens": 0,
            "recomputed_tokens": 0,
            "recompute_confirmed": True,
            "finish_status": "FINISHED",
            "terminal_reason": "native_load_completed",
            "terminal": True,
        }

    def test_native_accounting_is_mapped_only_by_association_index(self):
        errors = []
        rows = validate_request_accounting(
            [self._accounting()], {"logical-request"}, errors,
            association_index=self._association(),
        )
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["request_id"], "logical-request")
        self.assertEqual(rows[0]["native_request_id"], "chatcmpl-logical-request-1")

    def test_unassociated_native_accounting_is_rejected(self):
        errors = []
        rows = validate_request_accounting(
            [self._accounting()], {"logical-request"}, errors,
            association_index={"other-native": self._association()["chatcmpl-logical-request-1"]},
        )
        self.assertEqual(rows, [])
        self.assertEqual(errors, [
            "accounting_association_missing:chatcmpl-logical-request-1",
            "accounting_request_coverage_mismatch",
        ])

    def test_receipt_uses_same_native_to_logical_binding(self):
        errors = []
        accounting = validate_request_accounting(
            [self._accounting()], {"logical-request"}, errors,
            association_index=self._association(),
        )
        receipt = dict(self._accounting())
        receipt["status"] = "completed"
        receipt["bytes_loaded"] = 1024
        receipt["load_latency_ns"] = 100
        receipt["native_retrieved_tokens"] = 8
        validate_receipts(
            [receipt], accounting, errors, association_index=self._association(),
        )
        self.assertEqual(errors, [])

    def test_e3c_rejects_any_ssd_to_cpu_ticket(self):
        errors = []
        validate_prefetch_disabled(
            [],
            [{"source_tier": "ssd", "target_tier": "cpu", "prefetch_id": "unexpected"}],
            errors,
        )
        self.assertEqual(errors, ["e3c_prefetch_ticket_emitted:variant"])

    def test_e3c_allows_empty_ticket_artifacts(self):
        errors = []
        validate_prefetch_disabled([], [], errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
