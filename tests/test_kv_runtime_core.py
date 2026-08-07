"""Unit tests for the request-owned KV-Core contracts."""

from __future__ import annotations

import unittest

from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey,
    NativeKVLoadReceipt,
    PhysicalKVObject,
    PrefetchStatus,
    PrefetchTicket,
    PrefetchTicketStore,
    RequestKVIntent,
    TierCapabilitySnapshot,
    TierTopology,
    choose_load_vs_recompute,
    exact_token_prefix_hash,
)


class KVRuntimeCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = KVCompatibilityKey(
            model_id="Qwen3-8B",
            model_revision="qwen3-8b-bf16",
            tokenizer_revision="qwen3-8b",
            chat_template_revision="qwen3-default",
            dtype="bfloat16",
            rope_config="default",
            adapter_namespace="base",
            kv_layout="vllm-paged-kv-v1",
            block_size_tokens=16,
            chunk_size_tokens=256,
            layer_group="all-layers",
            prefix_hash=exact_token_prefix_hash((1, 2, 3, 4)),
            engine_id="engine-0",
            worker_id="worker-0",
        )
        self.object = PhysicalKVObject("native-key", "physical-0", 2, self.key, source_tier="ssd", size_bytes=128)
        self.intent = RequestKVIntent("request-0", self.key, self.object, 64, 64, deadline_ns=2_000_000_000)

    def test_token_hash_uses_exact_token_sequence(self) -> None:
        self.assertNotEqual(exact_token_prefix_hash((1, 2, 3)), exact_token_prefix_hash((1, 3, 2)))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            exact_token_prefix_hash(())

    def test_compatibility_key_requires_block_aligned_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "block aligned"):
            KVCompatibilityKey(
                model_id="Qwen3-8B", model_revision="r", tokenizer_revision="t", chat_template_revision="c",
                dtype="bfloat16", rope_config="default", adapter_namespace="base", kv_layout="paged",
                block_size_tokens=16, chunk_size_tokens=20, layer_group="all", prefix_hash="x", engine_id="e", worker_id="w",
            )

    def test_gpu_ssd_topology_refuses_cpu_prefetch(self) -> None:
        capability = TierCapabilitySnapshot(
            TierTopology.GPU_SSD, local_cpu_enabled=False, local_disk_enabled=True,
            ssd_capacity_bytes=1024, uma_available_bytes=10_000,
        )
        self.assertEqual(
            capability.prefetch_block_reason(size_bytes=100, deadline_ns=2_000_000_000, now_ns=1_000_000_000),
            "cpu_tier_unavailable",
        )

    def test_prefetch_ticket_requires_exact_generation_for_consumption(self) -> None:
        store = PrefetchTicketStore()
        ticket = store.submit(PrefetchTicket(
            "prefetch-0", "physical-0", 2, self.key.prefix_hash, "ssd", "cpu", 128,
            deadline_ns=2_000_000_000, expires_at_ns=3_000_000_000, target_request_id="request-0",
        ))
        self.assertEqual(ticket.status, PrefetchStatus.SUBMITTED)
        store.complete("prefetch-0", completed_bytes=128, now_ns=1_000_000_000)
        with self.assertRaisesRegex(ValueError, "generation"):
            store.consume(
                "prefetch-0", request_id="request-0", physical_object_id="physical-0",
                binding_generation=3, prefix_hash=self.key.prefix_hash, now_ns=1_100_000_000,
            )
        consumed = store.consume(
            "prefetch-0", request_id="request-0", physical_object_id="physical-0",
            binding_generation=2, prefix_hash=self.key.prefix_hash, now_ns=1_100_000_000,
        )
        self.assertEqual(consumed.status, PrefetchStatus.CONSUMED)

    def test_load_receipt_requires_recompute_for_every_unloaded_token(self) -> None:
        receipt = NativeKVLoadReceipt(
            request_id="request-0", physical_object_id="physical-0", binding_generation=2,
            requested_prefix_tokens=64, lookup_hit_tokens=64, allocated_external_tokens=48,
            actual_loaded_tokens=32, recomputed_tokens=32, bytes_loaded=128, load_latency_ns=100,
            status="completed", native_request_id="native-request-0",
        )
        self.assertEqual(receipt.actual_loaded_tokens, 32)
        with self.assertRaisesRegex(ValueError, "recomputed_tokens"):
            NativeKVLoadReceipt(
                request_id="request-0", physical_object_id="physical-0", binding_generation=2,
                requested_prefix_tokens=64, lookup_hit_tokens=64, allocated_external_tokens=48,
                actual_loaded_tokens=32, recomputed_tokens=31, bytes_loaded=128, load_latency_ns=100,
                status="completed", native_request_id="native-request-0",
            )

    def test_cost_model_defers_when_recompute_is_cheaper(self) -> None:
        capability = TierCapabilitySnapshot(
            TierTopology.GPU_CPU_SSD, local_cpu_enabled=True, local_disk_enabled=True,
            cpu_capacity_bytes=1024, ssd_capacity_bytes=1024, available_kv_blocks=8,
            external_token_cap=128, uma_available_bytes=10_000,
        )
        decision = choose_load_vs_recompute(
            intent=self.intent, capability=capability, queue_delay_ms=1, tier_read_ms=200,
            transfer_ms=1, materialization_ms=1, contention_ms=1, prefill_ms_per_token=1,
            now_ns=1_000_000_000,
        )
        self.assertEqual(decision.action, "recompute")
        self.assertEqual(decision.reason, "recompute_cheaper")

    def test_cost_model_admits_only_when_native_load_is_cheaper(self) -> None:
        capability = TierCapabilitySnapshot(
            TierTopology.GPU_CPU_SSD, local_cpu_enabled=True, local_disk_enabled=True,
            cpu_capacity_bytes=1024, ssd_capacity_bytes=1024, available_kv_blocks=8,
            external_token_cap=128, uma_available_bytes=10_000,
        )
        decision = choose_load_vs_recompute(
            intent=self.intent, capability=capability, queue_delay_ms=1, tier_read_ms=1,
            transfer_ms=1, materialization_ms=1, contention_ms=1, prefill_ms_per_token=2,
            now_ns=1_000_000_000,
        )
        self.assertEqual(decision.action, "admit_external_prefix")


if __name__ == "__main__":
    unittest.main()
