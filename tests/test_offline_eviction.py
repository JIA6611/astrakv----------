import unittest

from astrakv.runtime.offline_eviction import (
    OfflineAccess,
    OfflineEvictionSimulator,
    OfflineObject,
    OfflinePolicy,
    PrefetchHint,
    ProxyCostModel,
    TierCapacities,
)


def access(request_id: str, index: int, key: str) -> OfflineAccess:
    return OfflineAccess(request_id, index, key, "prefix", 4, "fixture")


class OfflineEvictionTests(unittest.TestCase):
    def test_tier_replay_tracks_cpu_ssd_and_eviction_reaccess(self) -> None:
        result = OfflineEvictionSimulator(
            policy=OfflinePolicy.LRU,
            capacities=TierCapacities(4, 4, 4), cost_model=ProxyCostModel(),
            run_id="run", workload_id="fixture",
            objects=[OfflineObject("a", "prefix", 4, "fixture"), OfflineObject("b", "prefix", 4, "fixture"), OfflineObject("c", "prefix", 4, "fixture")],
            accesses=[access("r0", 0, "a"), access("r1", 1, "b"), access("r2", 2, "a"), access("r3", 3, "c")],
        ).run()
        self.assertEqual(result.metrics["cpu_hits"], 1)
        self.assertGreater(result.metrics["ssd_write_proxy_bytes"], 0)
        self.assertGreater(result.metrics["eviction_reaccesses"], 0)
        self.assertTrue(any(event.event_type == "evict" for event in result.events))

    def test_astrakv_prefetch_is_consumed_before_demand(self) -> None:
        result = OfflineEvictionSimulator(
            policy=OfflinePolicy.ASTRAKV,
            capacities=TierCapacities(8, 8, 8), cost_model=ProxyCostModel(),
            run_id="run", workload_id="fixture",
            objects=[OfflineObject("a", "prefix", 4, "fixture"), OfflineObject("b", "prefix", 4, "fixture", astrakv_score=1.0)],
            accesses=[access("r0", 0, "a"), access("r1", 1, "b")],
            prefetch_hints=[PrefetchHint(0, "b", "prefix")],
        ).run()
        self.assertEqual(result.metrics["prefetch_submitted"], 1)
        self.assertEqual(result.metrics["prefetch_hits"], 1)
        self.assertEqual(result.metrics["prefetch_waste"], 0)

    def test_belady_is_explicitly_labeled_offline_oracle(self) -> None:
        result = OfflineEvictionSimulator(
            policy=OfflinePolicy.BELADY,
            capacities=TierCapacities(4, 4, 4), cost_model=ProxyCostModel(),
            run_id="run", workload_id="fixture",
            objects=[OfflineObject("a", "prefix", 4, "fixture"), OfflineObject("b", "prefix", 4, "fixture")],
            accesses=[access("r0", 0, "a"), access("r1", 1, "b")],
        ).run()
        self.assertTrue(result.metrics["is_offline_oracle"])

    def test_proxy_mode_changes_when_benchmark_timing_exists(self) -> None:
        result = OfflineEvictionSimulator(
            policy=OfflinePolicy.FIFO,
            capacities=TierCapacities(4, 4, 4), cost_model=ProxyCostModel(),
            run_id="run", workload_id="fixture",
            objects=[OfflineObject("a", "prefix", 4, "fixture")],
            accesses=[OfflineAccess("r", 0, "a", "prefix", 4, "fixture", base_ttft_ms=8.0, base_tpot_ms=1.0)],
        ).run()
        self.assertEqual(result.metrics["timing_mode"], "benchmark_plus_proxy")
        self.assertEqual(result.metrics["tpot_proxy_ms_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
