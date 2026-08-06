import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.runtime.vm_backend import (
    VMDemoConfig,
    VirtualMemoryDemoRunner,
    build_access_order,
    write_access_trace_csv,
    write_summary_json,
)
from scripts.vm.run_vm_demo import write_manifest, write_report


class VirtualMemoryDemoTests(unittest.TestCase):
    def test_access_order_patterns_are_deterministic(self) -> None:
        self.assertEqual(build_access_order(4, 6, "sequential", seed=1), [0, 1, 2, 3, 0, 1])
        self.assertEqual(build_access_order(4, 4, "reverse", seed=1), [3, 2, 1, 0])
        self.assertEqual(
            build_access_order(8, 5, "random", seed=7),
            build_access_order(8, 5, "random", seed=7),
        )

    def test_demand_mode_counts_first_accesses(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = VirtualMemoryDemoRunner(
                VMDemoConfig(file_size_mb=1, page_size_bytes=4096, access_count=8, prefetch_window=0)
            ).run(tmp / "backing.bin")

            self.assertEqual(len(result.access_records), 8)
            self.assertEqual(result.summary.unique_pages_accessed, 8)
            self.assertEqual(result.summary.demand_fault_like_count, 8)
            self.assertEqual(result.summary.prefetched_page_count, 0)
            self.assertFalse((tmp / "backing.bin").exists())

    def test_prefetch_window_reduces_demand_fault_like_accesses(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            demand = VirtualMemoryDemoRunner(
                VMDemoConfig(file_size_mb=1, page_size_bytes=4096, access_count=8, prefetch_window=0)
            ).run(tmp / "demand.bin")
            prefetched = VirtualMemoryDemoRunner(
                VMDemoConfig(file_size_mb=1, page_size_bytes=4096, access_count=8, prefetch_window=1)
            ).run(tmp / "prefetch.bin")

            self.assertLess(prefetched.summary.demand_fault_like_count, demand.summary.demand_fault_like_count)
            self.assertGreater(prefetched.summary.prefetched_page_count, 0)
            self.assertLessEqual(prefetched.summary.prefetch_coverage_rate, 1.0)

    def test_outputs_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = VirtualMemoryDemoRunner(
                VMDemoConfig(file_size_mb=1, page_size_bytes=4096, access_count=4, prefetch_window=1)
            ).run(tmp / "backing.bin")
            trace_path = tmp / "vm_access_trace.csv"
            summary_path = tmp / "vm_summary.json"
            report_path = tmp / "vm_demo_report.md"
            manifest_path = tmp / "vm_demo_manifest.json"
            args = Namespace(
                file_size_mb=1,
                page_size_bytes=4096,
                access_count=4,
                pattern="sequential",
                prefetch_window=1,
                seed=3136859,
                keep_backing_file=False,
            )

            write_access_trace_csv(trace_path, result.access_records)
            write_summary_json(summary_path, result.summary)
            write_report(report_path, result, trace_path, summary_path)
            write_manifest(manifest_path, args, result, trace_path, summary_path, report_path)

            with trace_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["access_count"], 4)
            self.assertIn("# OS Virtual Memory Demonstration Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "astra-vm-demo-manifest-v1")


if __name__ == "__main__":
    unittest.main()
