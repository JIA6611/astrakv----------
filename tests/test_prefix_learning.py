import unittest

from astrakv.runtime.backend_hook import BackendHookEvent, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.prefix_learning import RuntimePrefixIndex


class RuntimePrefixLearningTests(unittest.TestCase):
    def test_prefix_index_accumulates_runtime_observations(self) -> None:
        index = RuntimePrefixIndex()
        prefix_meta = {"prefix_id": "prefix-a", "cache_key": "prefix-a", "arrival_index": 1}
        index.observe(BackendHookEvent("run", "store-sub", "req-1", "prefix-a", ObjectLevel.PREFIX, "block-1", HookAction.CACHE_STORE, "submitted", 100, metadata=prefix_meta))
        index.observe(BackendHookEvent("run", "release", "req-1", "prefix-a", ObjectLevel.PREFIX, "block-1", HookAction.RELEASE, "completed", 200, metadata=prefix_meta))
        index.observe(BackendHookEvent("run", "store-sub-2", "req-2", "prefix-a", ObjectLevel.PREFIX, "block-2", HookAction.CACHE_STORE, "submitted", 260, metadata={**prefix_meta, "arrival_index": 2}))
        index.observe(BackendHookEvent("run", "prefetch", "req-2", "prefix-a", ObjectLevel.PREFIX, "block-2", HookAction.PREFETCH, "completed", 270, metadata={**prefix_meta, "arrival_index": 2}))
        index.observe(BackendHookEvent("run", "load", "req-2", "prefix-a", ObjectLevel.PREFIX, "block-2", HookAction.CACHE_LOAD, "completed", 320, metadata={**prefix_meta, "arrival_index": 2, "load_latency_ns": 5_000_000}))

        profile = index.profile_for("prefix-a")

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.request_count, 2)
        self.assertGreaterEqual(profile.reuse_count, 1)
        self.assertGreater(profile.inter_arrival_window_ms_ema, 0.0)
        self.assertGreater(profile.runtime_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
