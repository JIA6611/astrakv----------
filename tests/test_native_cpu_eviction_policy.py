from __future__ import annotations

from collections import OrderedDict
import time
import unittest

from astrakv.runtime.native_cpu_eviction_policy import (
    NativeCPUCachePolicy,
    NativeCPUScoreRegistry,
    install_native_cpu_eviction_policy,
    normalize_native_cpu_policy,
)


class _CacheEntry:
    def __init__(self, can_evict: bool = True):
        self.can_evict = can_evict


class _LRUDelegate:
    def __init__(self):
        self.hits: list[object] = []
        self.puts: list[object] = []
        self.forced: list[object] = []
        self.selection_calls = 0

    def init_mutable_mapping(self):
        return OrderedDict()

    def update_on_hit(self, key, cache_dict):
        self.hits.append(key)
        cache_dict.move_to_end(key)

    def update_on_put(self, key):
        self.puts.append(key)

    def update_on_force_evict(self, key):
        self.forced.append(key)

    def get_evict_candidates(self, cache_dict, num_candidates=1):
        self.selection_calls += 1
        result = []
        for key, value in cache_dict.items():
            if value.can_evict:
                result.append(key)
            if len(result) == num_candidates:
                break
        return result


class _CPUBackend:
    def __init__(self, policy):
        self.cache_policy = policy
        self.hot_cache = policy.init_mutable_mapping()
        self.use_hot = True

    def remove(self, key, force=True):
        return self.hot_cache.pop(key, None) is not None


class _DiskBackend:
    def __init__(self, policy):
        self.cache_policy = policy


class _Manager:
    def __init__(self, cpu, disk):
        self.storage_backends = {
            "LocalCPUBackend": cpu,
            "LocalDiskBackend": disk,
        }


def _wait_for(records, predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = [row for row in records if predicate(row)]
        if matches:
            return matches
        time.sleep(0.005)
    return []


class NativeCPUCachePolicyTest(unittest.TestCase):
    def make_policy(self, mode="astrakv"):
        records = []
        delegate = _LRUDelegate()
        registry = NativeCPUScoreRegistry()
        policy = NativeCPUCachePolicy(
            delegate,
            mode=mode,
            run_id="run-1",
            event_sink=records.append,
            score_registry=registry,
        )
        return policy, delegate, registry, records

    def test_lru_mode_delegates_exact_native_order(self):
        policy, delegate, _registry, records = self.make_policy("lru")
        cache = OrderedDict((key, _CacheEntry()) for key in ("old", "middle", "new"))
        self.assertEqual(policy.get_evict_candidates(cache, 2), ["old", "middle"])
        self.assertEqual(delegate.selection_calls, 1)
        selected = _wait_for(records, lambda row: row.get("status") == "selected")
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(selected[0]["selection_duration_ns"], 0)
        self.assertGreaterEqual(selected[0]["policy_scoring_duration_ns"], 0)
        self.assertEqual(selected[0]["candidate_scan_count"], 2)

    def test_astrakv_selects_colder_object_instead_of_oldest(self):
        policy, _delegate, registry, _records = self.make_policy()
        cache = OrderedDict((key, _CacheEntry()) for key in ("old-hot", "new-cold"))
        registry.observe("old-hot", metadata={"reuse_ratio": 1.0}, action="store", status="submitted")
        registry.observe("new-cold", metadata={"reuse_ratio": 0.0}, action="store", status="submitted")
        self.assertEqual(policy.get_evict_candidates(cache, 1), ["new-cold"])

    def test_e11_policy_hint_override_prevents_future_ratio_leakage(self):
        policy, _delegate, registry, _records = self.make_policy()
        cache = OrderedDict((key, _CacheEntry()) for key in ("reported-hot", "observed-hot"))
        registry.observe(
            "reported-hot",
            metadata={"reuse_ratio": 1.0, "e11_policy_reuse_ratio": 0.0},
            action="store",
            status="submitted",
        )
        registry.observe(
            "observed-hot",
            metadata={"reuse_ratio": 0.0, "e11_policy_reuse_ratio": 0.0},
            action="hit",
            status="completed",
        )
        self.assertEqual(policy.get_evict_candidates(cache, 1), ["reported-hot"])

    def test_pinned_entry_is_never_selected(self):
        policy, _delegate, registry, _records = self.make_policy()
        cache = OrderedDict((
            ("pinned-cold", _CacheEntry(False)),
            ("evictable-hot", _CacheEntry(True)),
        ))
        registry.observe("pinned-cold", metadata={"reuse_ratio": 0.0})
        registry.observe("evictable-hot", metadata={"reuse_ratio": 1.0})
        self.assertEqual(policy.get_evict_candidates(cache, 1), ["evictable-hot"])

    def test_unscored_candidates_fall_back_to_lru(self):
        policy, delegate, _registry, _records = self.make_policy()
        cache = OrderedDict((key, _CacheEntry()) for key in ("old", "new"))
        self.assertEqual(policy.get_evict_candidates(cache, 1), ["old"])
        self.assertGreaterEqual(delegate.selection_calls, 1)

    def test_successful_native_remove_emits_completed_pair(self):
        delegate = _LRUDelegate()
        cpu = _CPUBackend(delegate)
        disk_policy = _LRUDelegate()
        disk = _DiskBackend(disk_policy)
        manager = _Manager(cpu, disk)
        records = []
        wrapper = install_native_cpu_eviction_policy(
            manager,
            mode="astrakv",
            run_id="run-1",
            event_sink=records.append,
        )
        self.assertIsNotNone(wrapper)
        self.assertIs(cpu.cache_policy, wrapper)
        self.assertIs(disk.cache_policy, disk_policy)
        cpu.hot_cache["key"] = _CacheEntry()
        wrapper.score_registry.observe("key", metadata={"reuse_ratio": 0.0})
        self.assertEqual(wrapper.get_evict_candidates(cpu.hot_cache, 1), ["key"])
        self.assertTrue(cpu.remove("key", force=False))
        completed = _wait_for(
            records,
            lambda row: row.get("record_type") == "native_cache_policy_eviction"
            and row.get("status") == "completed",
        )
        self.assertEqual(len(completed), 1)
        selected = [
            row for row in records
            if row.get("record_type") == "native_cache_policy_eviction"
            and row.get("status") == "selected"
        ]
        self.assertEqual(selected[0]["selection_id"], completed[0]["selection_id"])
        self.assertEqual(completed[0]["terminal_condition"], "local_cpu_backend.remove:force=false")
        installation = _wait_for(
            records, lambda row: row.get("record_type") == "native_policy_installation"
        )
        self.assertEqual(installation[0]["cpu_requested_policy"], "astrakv")
        self.assertTrue(installation[0]["ssd_policy_unchanged"])

    def test_invalid_policy_fails_closed(self):
        with self.assertRaises(RuntimeError):
            normalize_native_cpu_policy("fifo")


if __name__ == "__main__":
    unittest.main()
