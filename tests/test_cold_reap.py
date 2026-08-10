"""Unit tests for the E5 cold external-copy reaper."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    PrefetchStatus,
    PrefetchTicket,
    RuntimeMode,
    TierCapabilitySnapshot,
    TierTopology,
)
from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge


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
    def __init__(self) -> None:
        self.dict = {}
        self.disk_lock = threading.Lock()

    def contains(self, key, pin):
        del pin
        return key in self.dict


class _CPU:
    use_hot = True

    def __init__(self) -> None:
        self.cpu_lock = threading.Lock()
        self.hot_cache = {}

    def contains(self, key, pin):
        del pin
        return key in self.hot_cache


class _Manager:
    def __init__(self, cpu: _CPU, disk: _Disk, *, removable: bool = True) -> None:
        self.storage_backends = {"LocalCPUBackend": cpu, "LocalDiskBackend": disk}
        if removable:
            self.remove = self._remove_impl

    def _remove_impl(self, key, locations) -> bool:
        removed = False
        for location in locations:
            backend = self.storage_backends[location]
            if location == "LocalCPUBackend":
                if key in backend.hot_cache:
                    backend.hot_cache.pop(key)
                    removed = True
            elif location == "LocalDiskBackend":
                if key in backend.dict:
                    backend.dict.pop(key)
                    removed = True
        return removed


def _connector(cpu: _CPU, disk: _Disk, *, removable: bool = True) -> SimpleNamespace:
    manager = _Manager(cpu, disk, removable=removable)
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
        config=SimpleNamespace(local_cpu=True, local_disk=True, max_local_cpu_size=5.0),
    )


def _callbacks() -> KVCoreConnectorCallbacks:
    return KVCoreConnectorCallbacks(
        mode=RuntimeMode.ACTIVE,
        capability=TierCapabilitySnapshot(
            topology=TierTopology.GPU_CPU_SSD,
            local_cpu_enabled=True,
            local_disk_enabled=True,
            available_kv_blocks=1024,
            external_token_cap=8,
        ),
    )


def time_now_ns() -> int:
    import time

    return time.time_ns()


class ColdReapTests(unittest.TestCase):
    def _reap_events(self, raw_tmp: str) -> list[dict[str, object]]:
        path = Path(raw_tmp) / "kv_core_external_reaps.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _scenario(
        self,
        *,
        enabled: bool = True,
        removable: bool = True,
        reuse_count: int = 0,
        request_count: int = 3,
        pending_ticket: bool = False,
    ) -> tuple[list[dict[str, object]], _CPU, _Disk]:
        cpu, disk, callbacks = _CPU(), _Disk(), _callbacks()
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
            "ASTRAKV_RUNTIME_CONTROL_RUN_ID": "reap-run",
            "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
            "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "8",
            "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
            "ASTRAKV_KV_CORE_COLD_REAP_ENABLED": "true" if enabled else "false",
            "ASTRAKV_KV_CORE_COLD_REAP_REUSE_THRESHOLD": "0.2",
            "ASTRAKV_KV_CORE_COLD_REAP_IDLE_MS": "0",
        }, clear=False), patch(
            "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
            return_value=callbacks,
        ), patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"):
            bridge = VendorCallbackBridge(_connector(cpu, disk, removable=removable))
            bridge.scheduler_exact_lookup(
                bridge._connector,
                request_id="native-request",
                token_ids=(1, 2, 3, 4),
                request_configs=None,
                lookup_hit_tokens=4,
                num_computed_tokens=0,
            )
            (object_id, state), = bridge._reap_state.items()
            state["ref_count"] = 0
            state["request_count"] = request_count
            state["reuse_count"] = reuse_count
            state["last_release_ns"] = time_now_ns() - 60_000_000_000
            for key in state["native_keys"]:
                cpu.hot_cache[key] = SimpleNamespace(get_physical_size=lambda: 2048)
                disk.dict[key] = SimpleNamespace(size=1024)
            if pending_ticket:
                now = time_now_ns()
                callbacks.tickets.submit(PrefetchTicket(
                    prefetch_id="prefetch:pending",
                    physical_object_id=object_id,
                    binding_generation=state["binding_generation"],
                    prefix_hash=state["prefix_hash"],
                    source_tier="ssd",
                    target_tier="cpu",
                    requested_bytes=1024,
                    deadline_ns=now + 60_000_000_000,
                    expires_at_ns=now + 120_000_000_000,
                    native_key="native-key",
                    compatibility_identity="identity",
                ))
            bridge._cold_reap_pass(callbacks)
            return self._reap_events(raw_tmp), cpu, disk

    def test_enabled_reaps_cold_cpu_and_disk_copies(self) -> None:
        events, cpu, disk = self._scenario(reuse_count=0, request_count=3)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["status"], "invalidated")
        self.assertEqual(event["target_tier"], "none")
        self.assertEqual(event["demoted_keys"], 2)
        self.assertEqual(event["invalidated_keys"], 2)
        self.assertEqual(event["freed_bytes"], 6144)
        self.assertEqual(event["run_id"], "reap-run")
        self.assertFalse(cpu.hot_cache)
        self.assertFalse(disk.dict)

    def test_disabled_reap_is_fail_closed(self) -> None:
        events, cpu, disk = self._scenario(enabled=False)
        self.assertEqual(events, [])
        self.assertEqual(len(cpu.hot_cache), 2)
        self.assertEqual(len(disk.dict), 2)

    def test_pending_prefetch_ticket_blocks_reap(self) -> None:
        events, cpu, _disk = self._scenario(pending_ticket=True)
        self.assertEqual(events, [])
        self.assertEqual(len(cpu.hot_cache), 2)

    def test_warm_reuse_object_is_not_reaped(self) -> None:
        events, cpu, _disk = self._scenario(reuse_count=3, request_count=3)
        self.assertEqual(events, [])
        self.assertEqual(len(cpu.hot_cache), 2)

    def test_manager_without_remove_records_delegation(self) -> None:
        events, _cpu, _disk = self._scenario(removable=False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "delegated_to_lmcache_eviction")


if __name__ == "__main__":
    unittest.main()
