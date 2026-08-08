"""Tests for the request-owned connector callback adapter."""

from __future__ import annotations

import unittest

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey, PhysicalKVObject, PrefetchTicket, RequestKVIntent,
    RuntimeMode, TierCapabilitySnapshot, TierTopology, exact_token_prefix_hash,
)


class KVCoreConnectorCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = KVCompatibilityKey(
            "Qwen3-8B", "qwen3-8b-bf16", "qwen3-8b", "qwen3-default", "bfloat16", "default", "base",
            "vllm-paged-kv-v1", 16, 256, "all-layers", exact_token_prefix_hash((1, 2, 3, 4)), "engine", "worker",
        )
        self.physical = PhysicalKVObject("native", "physical", 1, self.key, source_tier="ssd", size_bytes=128)
        self.intent = RequestKVIntent("request", self.key, self.physical, 64, 64, deadline_ns=2_000_000_000)
        self.capability = TierCapabilitySnapshot(
            TierTopology.GPU_CPU_SSD, True, True, cpu_capacity_bytes=1024, ssd_capacity_bytes=1024,
            available_kv_blocks=8, external_token_cap=64, uma_available_bytes=10_000,
        )

    def test_native_receipt_requires_lookup_and_scheduler_admission(self) -> None:
        callbacks = KVCoreConnectorCallbacks(mode=RuntimeMode.ACTIVE, capability=self.capability)
        callbacks.submit_intent(self.intent)
        callbacks.record_scheduler_lookup(request_id="request", physical=self.physical, lookup_hit_tokens=64, native_request_id="native-request")
        callbacks.record_scheduler_admission(request_id="request", physical=self.physical, allocated_external_tokens=48)
        receipt = callbacks.record_native_load_completion(
            request_id="request", physical=self.physical, actual_loaded_tokens=32, bytes_loaded=128,
            load_latency_ns=100, native_request_id="native-request", status="completed",
        )
        self.assertEqual(receipt.missing_tokens, 32)
        self.assertEqual(receipt.unallocated_recompute_tokens, 16)
        self.assertEqual(receipt.load_shortfall_tokens, 16)

        accounting = callbacks.finalize_request(
            request_id="request", physical=self.physical,
            finish_status="FINISHED_STOPPED", completed=True,
            native_num_computed_tokens=64,
        )
        self.assertFalse(accounting.recompute_confirmed)
        self.assertEqual(accounting.recomputed_tokens, 0)
        self.assertEqual(accounting.terminal_reason, "native_load_shortfall_unsafe")

    def test_cpu_prefetch_never_implies_gpu_load(self) -> None:
        callbacks = KVCoreConnectorCallbacks(mode=RuntimeMode.ACTIVE, capability=self.capability)
        ticket = PrefetchTicket(
            "ticket", "physical", 1, self.key.prefix_hash, "ssd", "cpu", 128,
            deadline_ns=2_000_000_000, expires_at_ns=3_000_000_000,
            native_key=self.physical.native_key, compatibility_identity=self.key.identity,
        )
        self.assertIsNone(callbacks.begin_cpu_prefetch(ticket, self.physical, now_ns=1_000_000_000))
        callbacks.complete_cpu_prefetch("ticket", completed_bytes=128, now_ns=1_100_000_000)
        self.assertIsNone(callbacks.receipt_for("request"))

    def test_scheduler_keeps_true_lookup_count_but_enforces_intent_upper_bound(self) -> None:
        callbacks = KVCoreConnectorCallbacks(mode=RuntimeMode.ACTIVE, capability=self.capability)
        limited = RequestKVIntent("request", self.key, self.physical, 32, 64, deadline_ns=2_000_000_000)
        callbacks.submit_intent(limited)
        lookup = callbacks.record_scheduler_lookup(
            request_id="request", physical=self.physical, lookup_hit_tokens=64, native_request_id="native-request",
        )
        self.assertEqual(lookup.lookup_hit_tokens, 64)
        with self.assertRaisesRegex(ValueError, "upper bound"):
            callbacks.record_scheduler_admission(request_id="request", physical=self.physical, allocated_external_tokens=48)

    def test_shadow_mode_cannot_start_cpu_prefetch(self) -> None:
        callbacks = KVCoreConnectorCallbacks(mode=RuntimeMode.SHADOW, capability=self.capability)
        ticket = PrefetchTicket(
            "ticket", "physical", 1, self.key.prefix_hash, "ssd", "cpu", 128,
            deadline_ns=2_000_000_000, expires_at_ns=3_000_000_000,
            native_key=self.physical.native_key, compatibility_identity=self.key.identity,
        )
        self.assertEqual(callbacks.begin_cpu_prefetch(ticket, self.physical, now_ns=1_000_000_000), "kv_core_not_active")


if __name__ == "__main__":
    unittest.main()
