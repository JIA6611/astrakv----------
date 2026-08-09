import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import RuntimeMode, TierCapabilitySnapshot, TierTopology
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


if __name__ == "__main__":
    unittest.main()
