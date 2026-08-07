from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.uma_metrics import UMAResourceCollector


class UMAResourceCollectorTests(unittest.TestCase):
    def test_collects_cgroup_rss_and_logical_tier_evidence_without_gpu_hbm(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            cgroup = root / "memory.current"
            status = root / "status"
            cgroup.write_text("4096\n", encoding="utf-8")
            status.write_text("Name:\tpython\nVmRSS:\t12 kB\n", encoding="utf-8")
            snapshot = UMAResourceCollector(
                cgroup_memory_current_path=cgroup, process_status_path=status, topology="gpu_cpu_ssd",
            ).snapshot(
                timestamp_ns=7,
                vllm={"available_kv_blocks": 123},
                lmcache={"cpu_used_bytes": 80, "ssd_used_bytes": 160},
                disk_io={"read_bytes": 17, "write_bytes": 19},
            )
            record = snapshot.to_record()
            self.assertEqual(record["cgroup_memory_current_bytes"], 4096)
            self.assertEqual(record["process_rss_bytes"], 12 * 1024)
            self.assertEqual(record["vllm_available_kv_blocks"], 123)
            self.assertNotIn("gpu_hbm_bytes", record)

    def test_missing_proc_files_are_reported_as_unknown(self) -> None:
        root = Path(tempfile.gettempdir()) / "astrakv-not-present"
        snapshot = UMAResourceCollector(cgroup_memory_current_path=root / "cgroup", process_status_path=root / "status").snapshot(timestamp_ns=1)
        self.assertIsNone(snapshot.cgroup_memory_current_bytes)
        self.assertIsNone(snapshot.process_rss_bytes)


if __name__ == "__main__":
    unittest.main()
