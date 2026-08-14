"""Unit tests for the test-only per-request load-vs-recompute decision probes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey,
    PhysicalKVObject,
    RuntimeMode,
    TierCapabilitySnapshot,
    TierTopology,
    exact_token_prefix_hash,
)
from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge, _exact_token_sequence_hash


TOKENS = tuple(range(4096))
PREFIX_HASH = exact_token_prefix_hash(TOKENS)
SEQUENCE_HASH = _exact_token_sequence_hash(TOKENS)


def _physical() -> PhysicalKVObject:
    key = KVCompatibilityKey(
        "Qwen3-8B", "qwen3-8b-bf16", "qwen3-8b", "qwen3-default", "bfloat16", "default", "base",
        "vllm-paged-kv-v1", 16, 256, "all-layers", PREFIX_HASH, "engine", "worker",
    )
    # 2 GiB object over 4096 tokens: SSD read is seconds at 3 GB/s, tens of
    # seconds at 0.1 GB/s, and ~175 ms at 12 GB/s.
    return PhysicalKVObject("native", "physical", 1, key, source_tier="ssd", size_bytes=2_147_483_648)


def _callbacks() -> KVCoreConnectorCallbacks:
    return KVCoreConnectorCallbacks(
        mode=RuntimeMode.ACTIVE,
        capability=TierCapabilitySnapshot(
            topology=TierTopology.GPU_SSD,
            local_cpu_enabled=False,
            local_disk_enabled=True,
            available_kv_blocks=1024,
            external_token_cap=8192,
        ),
    )


def _bridge(tmp: str):
    connector = SimpleNamespace(
        _vllm_config=SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=1024)),
        config=SimpleNamespace(max_local_cpu_size=0.0),
    )
    return VendorCallbackBridge(connector)


def _probe_context(tmp: str, *, test_gate: bool = True):
    env = _probe_env(tmp)
    if not test_gate:
        env["ASTRAKV_KV_CORE_EQUIVALENCE_TEST"] = "false"
    stack = ExitStack()
    stack.enter_context(patch.dict(os.environ, env, clear=False))
    stack.enter_context(patch(
        "astrakv.runtime.vendor_callback_bridge.installed_kv_core_callbacks",
        return_value=_callbacks(),
    ))
    stack.enter_context(patch.object(VendorCallbackBridge, "_start_prefetch_watcher_if_worker"))
    return stack


def _probe_env(tmp: str) -> dict[str, str]:
    return {
        "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": tmp,
        "ASTRAKV_KV_CORE_ADMISSION_ENABLED": "true",
        "ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP": "8192",
        "ASTRAKV_KV_CORE_SSD_READ_GBPS": "3.0",
        "ASTRAKV_KV_CORE_PREFILL_MS_PER_TOKEN": "0.08",
        "ASTRAKV_KV_CORE_BOOTSTRAP_LOADS": "1",
        "ASTRAKV_KV_CORE_EQUIVALENCE_TEST": "true",
    }


def _last_decision(tmp: str) -> dict:
    lines = (Path(tmp) / "kv_core_policy_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


class DecisionProbeTests(unittest.TestCase):
    def test_ingress_stores_probe_only_under_test_gate(self) -> None:
        probe = {"memory_pressure": 0.95, "deadline_ns": 1, "force_recompute": True}
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-request",
                metadata={"exact_token_ids": TOKENS, "kv_core_decision_probe": probe},
            ))
            self.assertEqual(bridge._decision_probe_by_request.get("probe-request"), probe)
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp, test_gate=False):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-request",
                metadata={"exact_token_ids": TOKENS, "kv_core_decision_probe": probe},
            ))
            self.assertNotIn("probe-request", bridge._decision_probe_by_request)

    def test_per_request_force_recompute_returns_zero_and_logs_reason(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-force",
                metadata={
                    "exact_token_ids": TOKENS,
                    "kv_core_decision_probe": {"force_recompute": True},
                },
            ))
            cap = bridge._external_token_cap(
                physical=_physical(),
                requested_tokens=len(TOKENS),
                available_external_tokens=len(TOKENS),
                priority=0,
                logical_request_id="probe-force",
                token_sequence_hash=SEQUENCE_HASH,
            )
            self.assertEqual(cap, 0)
            decision = _last_decision(raw_tmp)
            self.assertEqual(decision["action"], "recompute")
            self.assertEqual(decision["reason"], "equivalence_probe_force_recompute")
            self.assertTrue(decision["test_only"])
            self.assertEqual(decision["candidate_external_tokens"], 0)
            self.assertEqual(decision["decision_probe"], {"force_recompute": True})

    def test_per_request_pressure_override_forces_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-pressure",
                metadata={
                    "exact_token_ids": TOKENS,
                    "kv_core_decision_probe": {"memory_pressure": 0.95},
                },
            ))
            cap = bridge._external_token_cap(
                physical=_physical(),
                requested_tokens=len(TOKENS),
                available_external_tokens=len(TOKENS),
                priority=0,
                logical_request_id="probe-pressure",
                token_sequence_hash=SEQUENCE_HASH,
            )
            self.assertEqual(cap, 0)
            self.assertEqual(_last_decision(raw_tmp)["reason"], "uma_memory_pressure")

    def test_per_request_tight_deadline_forces_load_deadline_miss(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-deadline",
                metadata={
                    "exact_token_ids": TOKENS,
                    "kv_core_decision_probe": {"deadline_ns": 50_000_000},
                },
            ))
            cap = bridge._external_token_cap(
                physical=_physical(),
                requested_tokens=len(TOKENS),
                available_external_tokens=len(TOKENS),
                priority=0,
                logical_request_id="probe-deadline",
                token_sequence_hash=SEQUENCE_HASH,
            )
            self.assertEqual(cap, 0)
            self.assertEqual(_last_decision(raw_tmp)["reason"], "load_deadline_miss")

    def test_per_request_high_bandwidth_low_pressure_admits(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe-load",
                metadata={
                    "exact_token_ids": TOKENS,
                    "kv_core_decision_probe": {"ssd_gbps": 12.0, "memory_pressure": 0.0},
                },
            ))
            cap = bridge._external_token_cap(
                physical=_physical(),
                requested_tokens=len(TOKENS),
                available_external_tokens=len(TOKENS),
                priority=0,
                logical_request_id="probe-load",
                token_sequence_hash=SEQUENCE_HASH,
            )
            self.assertGreater(cap, 0)
            self.assertEqual(_last_decision(raw_tmp)["action"], "admit_external_prefix")
            self.assertEqual(_last_decision(raw_tmp)["reason"], "native_load_cheaper")

    def test_probe_deadline_override_falls_back_to_env(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, _probe_context(raw_tmp):
            bridge = _bridge(raw_tmp)
            self.assertEqual(bridge._probe_deadline_offset_ns("unknown-request"), 60_000_000_000)
            bridge._decision_probe_by_request["known-request"] = {"deadline_ns": 250_000_000}
            self.assertEqual(bridge._probe_deadline_offset_ns("known-request"), 250_000_000)


if __name__ == "__main__":
    unittest.main()
