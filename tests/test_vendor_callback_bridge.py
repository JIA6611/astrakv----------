import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey,
    PhysicalKVObject,
    PrefetchStatus,
    PrefetchTicket,
    RuntimeMode,
    TierCapabilitySnapshot,
    TierTopology,
    exact_token_prefix_hash,
)
from astrakv.runtime.request_context import RequestContextReceipt
from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge, _owns_runtime_control_host


class _Key:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_string(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Key) and other.value == self.value


class _TokenDatabase:
    def process_tokens(self, *, tokens, request_configs=None):
        del request_configs
        return [
            (start, start + 2, _Key(f"key-{start}-{tokens[start:start + 2]}"))
            for start in range(0, len(tokens) - 1, 2)
        ]


class _Disk:
    max_cache_size = 1 << 30
    current_cache_size = 4096

    def __init__(self) -> None:
        self.dict = {}
        self.disk_lock = None

    def contains(self, key, pin):
        del pin
        self.dict.setdefault(key, SimpleNamespace(size=1024))
        return True


class _SelectiveDisk(_Disk):
    def __init__(self, keys) -> None:
        super().__init__()
        self.keys = set(keys)

    def contains(self, key, pin):
        del pin
        return key in self.keys


class _CPU:
    use_hot = True

    def __init__(self) -> None:
        self.cpu_lock = threading.Lock()
        self.hot_cache = {}
        self.cache_policy = SimpleNamespace(update_on_force_evict=lambda _key: None)
        self.removed = []

    def contains(self, key, pin):
        del pin
        return key in self.hot_cache

    def remove(self, key, force=True) -> bool:
        if force:
            with self.cpu_lock:
                return self.remove(key, force=False)
        if key not in self.hot_cache:
            return False
        self.hot_cache.pop(key)
        self.removed.append(key)
        return True


class _MemoryObj:
    def __init__(self, size: int = 1024) -> None:
        self.is_pinned = False
        self._size = size

    def unpin(self) -> None:
        self.is_pinned = False

    def ref_count_down(self) -> None:
        return None

    def get_physical_size(self) -> int:
        return self._size


class _PromotionDisk(_SelectiveDisk):
    def __init__(self, keys) -> None:
        super().__init__(keys)
        self.reads = []
        self.unpins = []

    async def batched_get_non_blocking(self, lookup_id, keys, transfer_spec=None):
        del transfer_spec
        self.reads.append((lookup_id, list(keys)))
        return [_MemoryObj() for _ in keys]

    def unpin(self, key) -> None:
        self.unpins.append(key)


class _PromotionCPU(_CPU):
    def __init__(self) -> None:
        super().__init__()
        self.puts = []

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None, on_complete_callback=None):
        del transfer_spec, on_complete_callback
        self.puts.append((list(keys), list(memory_objs)))
        for key, memory_obj in zip(keys, memory_objs):
            self.hot_cache[key] = memory_obj


def _connector() -> SimpleNamespace:
    disk = _Disk()
    manager = SimpleNamespace(storage_backends={"LocalDiskBackend": disk})
    engine = SimpleNamespace(token_database=_TokenDatabase(), storage_manager=manager)
    model = SimpleNamespace(model="Qwen3-8B", served_model_name="Qwen3-8B", rope_scaling=None)
    config = SimpleNamespace(
        model_config=model,
        cache_config=SimpleNamespace(num_gpu_blocks=1024),
        dtype="bfloat16",
    )
    return SimpleNamespace(
        lmcache_engine=engine,
        _vllm_config=config,
        _block_size=1,
        _lmcache_chunk_size=2,
        _role=SimpleNamespace(name="WORKER"),
        kv_role="kv_both",
        worker_count=1,
        config=SimpleNamespace(local_cpu=False, local_disk=True, max_local_cpu_size=0.0),
    )


def _scheduler_connector() -> SimpleNamespace:
    connector = _connector()
    connector.lmcache_engine = None
    connector.lookup_client = SimpleNamespace(token_database=_TokenDatabase())
    connector._role = SimpleNamespace(name="SCHEDULER")
    return connector


class VendorCallbackBridgeTests(unittest.TestCase):
    def test_predictive_authorization_promotes_with_a_flag_disabled(self):
        with patch.dict(
            os.environ,
            {"ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED": "false"},
            clear=False,
        ):
            bridge = VendorCallbackBridge(SimpleNamespace())
            active_callbacks = SimpleNamespace(mode=RuntimeMode.ACTIVE)
            with (
                patch.object(bridge, "_callbacks", return_value=active_callbacks),
                patch.object(bridge, "_storage_manager", return_value=object()),
                patch.object(bridge, "_publish_tier_observation"),
                patch.object(bridge, "_schedule_cpu_promotion") as promote,
            ):
                bridge.ingress_request(SimpleNamespace(
                    request_id="request-far",
                    metadata={
                        "cache_key": "prefix-a",
                        "exact_token_ids": [1, 2, 3, 4],
                        "prefetch_lead_s": 5.0,
                        "predictive_prefetch_authorized": True,
                        "prefetch_origin": "sidecar_b",
                    },
                ))

            promote.assert_called_once()
            self.assertEqual(
                bridge._ingress_prefetch_origin["request-far"], "sidecar_b",
            )

    def test_connector_metadata_associates_real_native_request_through_host(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.SHADOW,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD, local_cpu_enabled=False, local_disk_enabled=True,
            ),
        )
        receipt = RequestContextReceipt(
            run_id="run", request_id="logical-request", request_nonce="nonce",
            runtime_request_id="chatcmpl-logical-request-native",
            runtime_event_id="runtime-context:chatcmpl-logical-request-native",
            status="associated", session_id="session", expires_at_ns=10, mac="authenticated",
        )
        host = SimpleNamespace(
            associate_runtime_request=lambda native_id: receipt if native_id == receipt.runtime_request_id else None,
            runtime_identity_for=lambda native_id: SimpleNamespace(request_id="logical-request") if native_id == receipt.runtime_request_id else None,
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_runtime_control_host",
            return_value=host,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            bridge.connector_metadata(
                request_id=receipt.runtime_request_id, metadata_present=True, can_load=True,
            )
            record = json.loads((Path(raw_tmp) / "kv_core_native_callbacks.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["native_request_id"], receipt.runtime_request_id)
            self.assertEqual(record["logical_request_id"], "logical-request")
            self.assertEqual(record["association_receipt_reference"], receipt.runtime_event_id)

    def test_single_worker_kv_both_worker_owns_control_host(self) -> None:
        self.assertTrue(_owns_runtime_control_host(_connector()))

    def test_distributed_worker_does_not_own_control_host(self) -> None:
        connector = _connector()
        connector.worker_count = 2
        self.assertFalse(_owns_runtime_control_host(connector))

    def test_scheduler_role_owns_control_host_independently_of_kv_role(self) -> None:
        connector = _connector()
        connector._role = SimpleNamespace(name="SCHEDULER")
        connector.kv_role = "kv_producer"
        connector.worker_count = 8
        self.assertTrue(_owns_runtime_control_host(connector))

    def test_existing_process_local_host_receives_late_connector_bridge(self) -> None:
        connector = _connector()
        connector.worker_count = 2
        registered = []
        host = SimpleNamespace(register_kv_runtime_bridge=registered.append)
        callbacks = SimpleNamespace(mode=RuntimeMode.ACTIVE)
        with (
            patch.dict(os.environ, {
                "ASTRAKV_KV_CORE_VENDOR_PATCH": "true",
                "ASTRAKV_TENSOR_PARALLEL_SIZE": "2",
            }, clear=False),
            patch(
                "astrakv.runtime.vendor_callback_bridge.patch_local_disk_remove_race_class",
            ),
            patch(
                "astrakv.runtime.vendor_callback_bridge.install_from_environment",
            ) as install,
            patch(
                "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
                return_value=callbacks,
            ),
            patch(
                "astrakv.runtime.vendor_callback_bridge.installed_runtime_control_host",
                return_value=host,
            ),
            patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"),
        ):
            bridge = VendorCallbackBridge.from_environment(connector)

        self.assertIsNotNone(bridge)
        self.assertEqual(registered, [bridge])
        install.assert_called_once_with(
            vendor_engine_child=True,
            start_runtime_host=False,
        )

    def test_scheduler_lookup_publishes_intent_before_using_it(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
                available_kv_blocks=1024,
                external_token_cap=8,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
            "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "8",
            "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            cap = bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
                num_computed_tokens=0,
            )
            self.assertEqual(cap, 2)
            intent = callbacks.intent_for("native-request")
            self.assertIsNotNone(intent)
            self.assertEqual(intent.max_external_tokens, 2)
            intent_files = list((Path(raw_tmp) / "native_intents").glob("*.json"))
            self.assertEqual(len(intent_files), 1)
            ledger = json.loads(intent_files[0].read_text(encoding="utf-8"))
            self.assertEqual(ledger["max_external_tokens"], 2)
            self.assertEqual(ledger["logical_request_id"], "native-request")

    def test_scheduler_progress_calibrates_prefill_cost_before_next_admission(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
                available_kv_blocks=1024,
                external_token_cap=128,
            ),
        )
        tokens = tuple(range(128))
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
            "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "128",
            "ASTRAKV_KV_CORE_SSD_READ_GBPS": "3.0",
            "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
            "ASTRAKV_KV_CORE_PREFILL_ONLINE_CALIBRATION": "true",
            "ASTRAKV_KV_CORE_PREFILL_SAMPLE_MIN_TOKENS": "32",
            "ASTRAKV_KV_CORE_PREFILL_EMA_ALPHA": "1.0",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="calibration-request",
                token_ids=tokens,
                request_configs=None,
                lookup_hit_tokens=0,
            )
            bridge.scheduler_compute_progress(
                request_id="calibration-request", scheduled_tokens=64,
            )
            bridge._pending_prefill_steps["calibration-request"] = (1_000_000_000, 64)
            with patch(
                "astrakv.runtime.vendor_callback_bridge.time.time_ns",
                return_value=1_064_000_000,
            ):
                bridge.scheduler_compute_progress(
                    request_id="calibration-request", scheduled_tokens=1,
                )
            admitted = bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="measured-request",
                token_ids=tokens,
                request_configs=None,
                lookup_hit_tokens=128,
            )
            observations = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_cost_observations.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        self.assertGreater(admitted, 0)
        self.assertEqual(bridge._bootstrap_loads, 0)
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0]["accepted"])
        self.assertEqual(observations[0]["prefill_tokens"], 64)
        self.assertAlmostEqual(observations[0]["observed_prefill_ms_per_token"], 1.0)

    def test_local_request_finish_marks_unconsumed_prefetch_wasted(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_CPU_SSD,
                local_cpu_enabled=True,
                local_disk_enabled=True,
            ),
        )
        connector = _connector()
        connector.lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"] = _CPU()
        tokens = tuple(range(8))
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(connector)
            physical, keys = bridge._physical(connector, tokens, None)
            self.assertIsNotNone(physical)
            assert physical is not None
            ticket = PrefetchTicket(
                prefetch_id="ticket-1",
                physical_object_id=physical.physical_object_id,
                binding_generation=physical.binding_generation,
                prefix_hash=physical.compatibility_key.prefix_hash,
                source_tier="ssd",
                target_tier="cpu",
                requested_bytes=1024,
                deadline_ns=9_000_000_000_000_000_000,
                expires_at_ns=9_000_000_000_000_000_001,
                target_request_id="target-request",
                native_key=physical.native_key,
                compatibility_identity=physical.compatibility_key.identity,
            )
            callbacks.tickets.submit(ticket)
            bridge._prefetch_keys[ticket.prefetch_id] = keys
            bridge._finalize_local_prefetch_for_terminal(
                logical_request_id="target-request", completed=True,
            )
            persisted = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_prefetch_tickets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        current = callbacks.tickets.get("ticket-1")
        self.assertIsNotNone(current)
        self.assertIs(current.status, PrefetchStatus.WASTED)
        self.assertEqual(persisted[-1]["status"], "wasted")

    def test_scheduler_lookup_does_not_associate_before_reqmeta_metadata(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.SHADOW,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
            ),
        )
        host = SimpleNamespace(
            associate_runtime_request=lambda native_id: self.fail(
                f"association started before ReqMeta metadata: {native_id}"
            ),
            runtime_identity_for=lambda native_id: None,
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_runtime_control_host",
            return_value=host,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
            )

    def test_request_finished_writes_terminal_accounting_record(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
                available_kv_blocks=1024,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=0,
            )
            bridge.scheduler_external_admission(
                request_id="native-request", allocated_external_tokens=0,
            )
            bridge.request_finished(
                request_id="native-request",
                finish_status="FINISHED_LENGTH_CAPPED",
                num_computed_tokens=4,
                num_tokens=5,
            )
            rows = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_request_accounting.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(rows[-1]["terminal"])
            self.assertEqual(rows[-1]["terminal_reason"], "scheduler_declined_recompute")

    def test_scheduler_lookup_uses_lmcache_lookup_client_token_database(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.SHADOW,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
                available_kv_blocks=1024,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "false",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_scheduler_connector())
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
                num_computed_tokens=0,
            )
            self.assertIsNotNone(callbacks.lookup_for("native-request"))

    def test_prefetch_lead_invalidates_only_disk_backed_target_cpu_chunks(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.OFF,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_CPU_SSD, local_cpu_enabled=True, local_disk_enabled=True,
            ),
        )
        connector = _connector()
        cpu = _CPU()
        keys = tuple(key for _start, _end, key in connector.lmcache_engine.token_database.process_tokens(tokens=[1, 2, 3, 4]))
        cpu.hot_cache[keys[0]] = SimpleNamespace(is_pinned=False)
        cpu.hot_cache[keys[1]] = SimpleNamespace(is_pinned=False)
        connector.lmcache_engine.storage_manager.storage_backends["LocalDiskBackend"] = _SelectiveDisk([keys[0]])
        connector.lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"] = cpu
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD": "true",
            "ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED": "false",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(connector)
            bridge.ingress_request(SimpleNamespace(
                request_id="target", metadata={
                    "exact_token_ids": [1, 2, 3, 4], "prefetch_lead_s": 0.05,
                },
            ))
            self.assertEqual(cpu.removed, [keys[0]])
            self.assertIn(keys[1], cpu.hot_cache)
            records = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(records[-1]["action"], "invalidate_external_copy")
            self.assertEqual(records[-1]["cpu_removed_chunk_count"], 1)
            self.assertEqual(records[-1]["cpu_only_chunk_count"], 1)

    def test_native_load_start_accepts_partial_prefix_after_churn(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_SSD,
                local_cpu_enabled=False,
                local_disk_enabled=True,
                available_kv_blocks=1024,
                external_token_cap=8,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
            "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "8",
            "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector())
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
                num_computed_tokens=0,
            )
            # Simulate the worker-side view after churn evicted tail chunks:
            # only the first block-aligned key remains available.
            short_key = KVCompatibilityKey(
                "Qwen3-8B", "local-qwen3-8b", "local-qwen3-8b", "qwen3-default", "bfloat16",
                "{}", "base", "vllm-paged-kv-v1", 1, 2, "all-kv-layers",
                exact_token_prefix_hash((1, 2)), "", "",
            )
            short_physical = PhysicalKVObject(
                json.dumps([key.to_string() for key in bridge._keys_by_request["native-request"][:1]]),
                "short-object", 1, short_key, source_tier="ssd", size_bytes=1,
            )
            bridge._physical_by_request["native-request"] = short_physical
            bridge.native_load_start(
                request_id="native-request",
                connector=bridge._connector,
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                compatibility_prefix_tokens=2,
                allocated_external_tokens=2,
            )
            records = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_native_callbacks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            partial = [
                row for row in records
                if row.get("callback") == "native_load_start" and row.get("status") == "accepted_partial_prefix"
            ]
            rejected = [
                row for row in records
                if row.get("callback") == "native_load_start" and row.get("status") == "rejected"
            ]
            self.assertEqual(len(partial), 1)
            self.assertEqual(rejected, [])

    def test_native_load_start_consumes_completed_prefetch_for_declared_partial_prefix(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_CPU_SSD,
                local_cpu_enabled=True,
                local_disk_enabled=True,
                available_kv_blocks=1024,
                external_token_cap=2,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
            "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "2",
            "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            connector = _connector()
            cpu = _CPU()
            connector.lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"] = cpu

            scheduler_bridge = VendorCallbackBridge(connector)
            scheduler_bridge.scheduler_exact_lookup(
                connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
                num_computed_tokens=0,
            )
            declared = scheduler_bridge._physical_by_request["native-request"]
            declared_keys = scheduler_bridge._keys_by_request["native-request"]
            now = time.time_ns()
            ticket = PrefetchTicket(
                prefetch_id="partial-prefix-ticket",
                physical_object_id=declared.physical_object_id,
                binding_generation=declared.binding_generation,
                prefix_hash=declared.compatibility_key.prefix_hash,
                source_tier="ssd",
                target_tier="cpu",
                requested_bytes=2048,
                deadline_ns=now + 1_000_000_000,
                expires_at_ns=now + 2_000_000_000,
                target_request_id="native-request",
                native_key=declared.native_key,
                compatibility_identity=declared.compatibility_key.identity,
            )
            callbacks.tickets.submit(ticket)
            callbacks.tickets.complete(ticket.prefetch_id, completed_bytes=2048)
            for key in declared_keys:
                cpu.hot_cache[key] = _MemoryObj()

            # A separate worker-side bridge sees only the admitted one-chunk
            # prefix, while the persisted intent and ticket describe both
            # scheduler-visible chunks.
            worker_bridge = VendorCallbackBridge(connector)
            worker_bridge.native_load_start(
                request_id="native-request",
                connector=connector,
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                compatibility_prefix_tokens=2,
                allocated_external_tokens=2,
            )

            consumed = callbacks.tickets.get(ticket.prefetch_id)
            self.assertIsNotNone(consumed)
            self.assertIs(consumed.status, PrefetchStatus.CONSUMED)
            self.assertEqual(consumed.consumer_request_id, "native-request")
            self.assertEqual(
                worker_bridge._prefetch_by_request["native-request"], ticket.prefetch_id,
            )
            records = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_native_callbacks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(any(
                row.get("callback") == "native_load_start"
                and row.get("status") == "accepted_partial_prefix"
                for row in records
            ))
            self.assertEqual(records[-1]["prefetch_id"], ticket.prefetch_id)

    @staticmethod
    def _roomy_capability() -> TierCapabilitySnapshot:
        return TierCapabilitySnapshot(
            topology=TierTopology.GPU_CPU_SSD,
            local_cpu_enabled=True,
            local_disk_enabled=True,
            uma_available_bytes=1 << 30,
            cpu_prefetch_budget_fraction=0.5,
            cpu_capacity_bytes=1 << 30,
        )

    def _promotion_connector(self, *, cpu, disk):
        connector = _connector()
        manager = SimpleNamespace(
            storage_backends={"LocalCPUBackend": cpu, "LocalDiskBackend": disk},
        )
        connector.lmcache_engine.storage_manager = manager
        return connector, manager

    @staticmethod
    def _chunk_keys(connector, tokens=(1, 2, 3, 4)):
        return tuple(
            key
            for _start, _end, key in connector.lmcache_engine.token_database.process_tokens(
                tokens=list(tokens),
            )
        )

    @staticmethod
    def _wait_for_ticket_record(state_dir: str, prefetch_id: str, status: str) -> None:
        """Wait for the async promote callback to persist its ticket record.

        The promote done-callback runs on the manager event loop after the
        future resolves; tearing down the temp dir before it writes the ticket
        file would race with cleanup.  Poll the JSONL until the terminal
        record appears (bounded, so a real failure still surfaces quickly).
        """
        ticket_path = Path(state_dir) / "kv_core_prefetch_tickets.jsonl"
        needle = f'"prefetch_id": "{prefetch_id}"'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            lines = (
                ticket_path.read_text(encoding="utf-8").splitlines()
                if ticket_path.exists()
                else []
            )
            if any(needle in line and f'"status": "{status}"' in line for line in lines):
                return
            time.sleep(0.01)
        raise AssertionError(
            f"ticket record {prefetch_id} with status {status!r} not persisted within 5s"
        )

    def test_schedule_cpu_promotion_happy_path_submits_ticket_and_promotes(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=self._roomy_capability(),
        )
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
                "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
                "ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED": "true",
            }, clear=False), patch(
                "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
                return_value=callbacks,
            ), patch.object(
                VendorCallbackBridge, "_runtime_capability", return_value=self._roomy_capability(),
            ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
                cpu = _PromotionCPU()
                disk = _PromotionDisk(())
                connector, manager = self._promotion_connector(cpu=cpu, disk=disk)
                keys = self._chunk_keys(connector)
                disk.keys = set(keys)
                for key in keys:
                    disk.dict[key] = SimpleNamespace(size=1024)
                manager.loop = loop
                bridge = VendorCallbackBridge(connector)
                bridge._schedule_cpu_promotion(
                    request_id="target", token_ids=(1, 2, 3, 4), request_configs=None,
                )
                tickets = callbacks.tickets.snapshot()
                self.assertEqual(len(tickets), 1)
                ticket = tickets[0]
                self.assertIs(ticket.status, PrefetchStatus.SUBMITTED)
                # The promotion can finish before this thread observes the
                # future; the done callback intentionally removes completed
                # futures.  Absence therefore means "already settled", not
                # "never scheduled".
                future = bridge._prefetch_futures.get(ticket.prefetch_id)
                if future is not None:
                    self.assertEqual(future.result(timeout=5), 2048)
                self._wait_for_ticket_record(raw_tmp, ticket.prefetch_id, "completed")
                self.assertIs(
                    callbacks.tickets.get(ticket.prefetch_id).status,
                    PrefetchStatus.COMPLETED,
                )
                decisions = [
                    json.loads(line)
                    for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(decisions[-1]["action"], "prefetch_ssd_to_cpu")
                self.assertEqual(decisions[-1]["status"], "submitted")
                self.assertEqual(len(disk.reads), 1)
                self.assertEqual(disk.reads[0][0], ticket.prefetch_id)
                self.assertEqual(len(cpu.puts), 1)
                self.assertEqual(cpu.puts[0][0], list(keys))
                self.assertTrue(all(key in cpu.hot_cache for key in keys))
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_prefetch_watcher_file_handoff_promotes_when_flag_set(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=self._roomy_capability(),
        )
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
                "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            }, clear=False), patch(
                "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
                return_value=callbacks,
            ), patch.object(
                VendorCallbackBridge, "_runtime_capability", return_value=self._roomy_capability(),
            ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
                cpu = _PromotionCPU()
                disk = _PromotionDisk(())
                connector, manager = self._promotion_connector(cpu=cpu, disk=disk)
                keys = self._chunk_keys(connector)
                disk.keys = set(keys)
                for key in keys:
                    disk.dict[key] = SimpleNamespace(size=1024)
                manager.loop = loop
                bridge = VendorCallbackBridge(connector)
                bridge._write_prefetch_request("target", (1, 2, 3, 4), None, promote=True)
                request_path = next((Path(raw_tmp) / "prefetch_requests").glob("*.json"))
                bridge._process_prefetch_request_file(request_path)
                tickets = callbacks.tickets.snapshot()
                self.assertEqual(len(tickets), 1)
                ticket = tickets[0]
                future = bridge._prefetch_futures.get(ticket.prefetch_id)
                if future is not None:
                    self.assertEqual(future.result(timeout=5), 2048)
                self._wait_for_ticket_record(raw_tmp, ticket.prefetch_id, "completed")
                self.assertIs(
                    callbacks.tickets.get(ticket.prefetch_id).status,
                    PrefetchStatus.COMPLETED,
                )
                decisions = [
                    json.loads(line)
                    for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(decisions[-1]["action"], "prefetch_ssd_to_cpu")
                self.assertEqual(decisions[-1]["status"], "submitted")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_prefetch_watcher_file_handoff_skips_promotion_when_flag_unset(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=self._roomy_capability(),
        )
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            cpu = _PromotionCPU()
            disk = _PromotionDisk(())
            connector, manager = self._promotion_connector(cpu=cpu, disk=disk)
            keys = self._chunk_keys(connector)
            disk.keys = set(keys)
            for key in keys:
                disk.dict[key] = SimpleNamespace(size=1024)
            bridge = VendorCallbackBridge(connector)
            bridge._write_prefetch_request("target", (1, 2, 3, 4), None, promote=False)
            request_path = next((Path(raw_tmp) / "prefetch_requests").glob("*.json"))
            bridge._process_prefetch_request_file(request_path)
            self.assertEqual(callbacks.tickets.snapshot(), ())
            self.assertEqual(cpu.puts, [])
            decisions_path = Path(raw_tmp) / "kv_core_policy_decisions.jsonl"
            self.assertFalse(
                decisions_path.exists()
                and any(
                    "prefetch_ssd_to_cpu" in line
                    for line in decisions_path.read_text(encoding="utf-8").splitlines()
                )
            )

    def test_schedule_cpu_promotion_rejected_when_cpu_budget_insufficient(self) -> None:
        callbacks = KVCoreConnectorCallbacks(
            mode=RuntimeMode.ACTIVE,
            capability=TierCapabilitySnapshot(
                topology=TierTopology.GPU_CPU_SSD,
                local_cpu_enabled=True,
                local_disk_enabled=True,
                uma_available_bytes=1,
                cpu_prefetch_budget_fraction=0.01,
                cpu_capacity_bytes=1,
            ),
        )
        tight = callbacks.capability
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_runtime_capability", return_value=tight), patch.object(
            VendorCallbackBridge, "_start_prefetch_watcher_if_worker",
        ):
            cpu = _PromotionCPU()
            disk = _PromotionDisk(())
            connector, manager = self._promotion_connector(cpu=cpu, disk=disk)
            keys = self._chunk_keys(connector)
            disk.keys = set(keys)
            for key in keys:
                disk.dict[key] = SimpleNamespace(size=1024)
            bridge = VendorCallbackBridge(connector)
            bridge._schedule_cpu_promotion(
                request_id="target", token_ids=(1, 2, 3, 4), request_configs=None,
            )
            self.assertEqual(callbacks.tickets.snapshot(), ())
            decisions = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            last = decisions[-1]
            self.assertEqual(last["action"], "prefetch_ssd_to_cpu")
            self.assertEqual(last["status"], "rejected")
            self.assertEqual(last["reason"], "cpu_prefetch_budget")
            tickets = [
                json.loads(line)
                for line in (Path(raw_tmp) / "kv_core_prefetch_tickets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(tickets[-1]["status"], PrefetchStatus.CANCELLED.value)
            self.assertEqual(tickets[-1]["failure_reason"], "cpu_prefetch_budget")


if __name__ == "__main__":
    unittest.main()
