import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import RuntimeMode, TierCapabilitySnapshot, TierTopology
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


class VendorCallbackBridgeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
