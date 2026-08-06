from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.benchmark.snapshot_lmcache_ssd_store import compare_snapshots, snapshot_store


class LMCacheSSDStoreSnapshotTests(unittest.TestCase):
    def test_reports_file_and_byte_growth_without_claiming_cache_hits(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            store = Path(raw_tmp) / "store"
            store.mkdir()
            before = snapshot_store(store, label="before", run_id="run-1")
            (store / "chunk-001.pt").write_bytes(b"a" * 32)
            after = snapshot_store(store, label="after", run_id="run-1")
            comparison = compare_snapshots(before, after)

        self.assertEqual(after["file_count"], 1)
        self.assertEqual(after["pt_file_count"], 1)
        self.assertEqual(comparison["file_count_delta"], 1)
        self.assertEqual(comparison["byte_delta"], 32)
        self.assertTrue(comparison["disk_artifact_growth_observed"])
        self.assertIn("does not prove", comparison["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
