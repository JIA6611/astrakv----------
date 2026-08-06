import json
import sys
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.profile_db import ProfileDB
from scripts.policy import run_offline_eviction_simulator as simulator_cli


class OfflineEvictionCliTests(unittest.TestCase):
    def test_cli_writes_all_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            workload = root / "workload.jsonl"
            workload.write_text("\n".join(json.dumps(item) for item in [
                {"request_id": "r0", "prompt": "a", "prefix_id": "p0", "arrival_index": 0, "reuse_ratio": 0.5, "reuse_bucket": "medium"},
                {"request_id": "r1", "prompt": "b", "prefix_id": "p0", "arrival_index": 1, "reuse_ratio": 0.5, "reuse_bucket": "medium"},
            ]) + "\n", encoding="utf-8")
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({
                "request_id": "r0", "event_type": "cache_load", "bytes": 4096, "latency_ms": 3.0,
                "metadata": {"legacy_unlinked": False},
            }) + "\n", encoding="utf-8")
            profile = root / "profile.json"
            ProfileDB().save(profile)
            output = root / "output"
            previous = sys.argv
            try:
                sys.argv = ["run_offline_eviction_simulator.py", "--workload-manifest", str(workload), "--trace", str(trace),
                           "--profile-db", str(profile), "--workload-id", "fixture", "--run-id", "run", "--gpu-capacity-bytes", "4096",
                           "--cpu-capacity-bytes", "4096", "--ssd-capacity-bytes", "4096", "--default-object-bytes", "4096",
                           "--profile-source", "separate_profiling_run", "--output-dir", str(output)]
                self.assertEqual(simulator_cli.main(), 0)
            finally:
                sys.argv = previous
            manifest = json.loads((output / "offline_eviction_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "astrakv-offline-eviction-v1")
            self.assertFalse(manifest["self_profile_leakage"])
            self.assertTrue((output / "offline_eviction_events.jsonl").exists())
            self.assertTrue((output / "offline_eviction_policy_summary.csv").exists())
            self.assertTrue((output / "offline_eviction_report.md").exists())


if __name__ == "__main__":
    unittest.main()
