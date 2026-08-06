import unittest

from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision, RuntimeActionResult
from astrakv.runtime.offline_safety import GatedRuntimeAdapter, OfflineSafetyGate


def manifest(workload_id: str, *, source: str = "separate_profiling_run", hit: int = 8, migration: int = 10, oom: int = 0) -> dict:
    def row(policy: str, policy_hit: int, policy_migration: int, policy_oom: int = 0) -> dict:
        return {"policy": policy, "request_count": 10, "total_hits": policy_hit, "migration_bytes": policy_migration, "oom_unavoided": policy_oom}
    return {
        "schema": "astrakv-offline-eviction-v1", "simulation_status": "valid", "workload_id": workload_id,
        "workload_sha256": "w" + workload_id, "trace_sha256": "t", "profile_db_sha256": "p",
        "profile_source": source, "self_profile_leakage": source == "self_profile",
        "capacities": {"gpu_bytes": 1, "cpu_bytes": 1, "ssd_bytes": 1}, "legacy_unlinked_in_denominator": False,
        "policies": [row("astrakv", hit, migration, oom), row("lru", 7, 12), row("fifo", 6, 15)],
    }


class OfflineSafetyTests(unittest.TestCase):
    def test_three_valid_workloads_allow_action(self) -> None:
        gate = OfflineSafetyGate.evaluate([manifest("a"), manifest("b"), manifest("c")])
        self.assertTrue(gate.result.allowed)

    def test_self_profile_and_insufficient_workloads_reject(self) -> None:
        gate = OfflineSafetyGate.evaluate([manifest("a"), manifest("b", source="self_profile")])
        self.assertFalse(gate.result.allowed)
        self.assertTrue(any("minimum_three_distinct_workloads" in item for item in gate.result.reasons))

    def test_gated_adapter_does_not_call_rejected_backend(self) -> None:
        class Adapter:
            name = "fake"
            called = False
            def capabilities(self): return object()
            def collect_runtime_events(self): return []
            def apply_hint(self, decision):
                self.called = True
                return RuntimeActionResult("executed", "incorrect")
        adapter = Adapter()
        gate = OfflineSafetyGate.evaluate([manifest("a")])
        result = GatedRuntimeAdapter(adapter, gate).apply_hint(OfflineEvictionDecision(
            "run", "d", "r", "prefix", ObjectLevel.PREFIX, "offload"
        ))
        self.assertEqual(result.status, "blocked_by_offline_gate")
        self.assertFalse(adapter.called)


if __name__ == "__main__":
    unittest.main()
