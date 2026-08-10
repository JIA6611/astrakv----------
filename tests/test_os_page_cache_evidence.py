"""Unit tests for the OS page-cache residency evidence module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.os_page_cache_evidence import (  # noqa: E402
    PAGE_CACHE_EVIDENCE_SCHEMA,
    PageCacheEvidenceCollector,
    collect_sample,
    write_csv,
    write_jsonl,
)


class OsPageCacheEvidenceTests(unittest.TestCase):
    def test_sample_record_schema_and_graceful_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "backing.bin"
            path.write_bytes(os.urandom(8192))
            sample = collect_sample(path=path, page_size=4096, max_mapped_bytes=1 << 30)
            self.assertEqual(sample["schema"], PAGE_CACHE_EVIDENCE_SCHEMA)
            self.assertEqual(sample["file_size_bytes"], 8192)
            self.assertEqual(sample["mapped_bytes"], 8192)
            self.assertEqual(sample["total_pages"], 2)
            self.assertIn(sample["mincore_status"], {"valid", "unsupported"})
            if sample["mincore_status"] == "valid":
                self.assertIsNotNone(sample["resident_pages"])
                self.assertIsNotNone(sample["resident_fraction"])
            else:
                self.assertIsNone(sample["resident_pages"])
            self.assertIn("cgroup_memory_current_bytes", sample)
            self.assertIn("process_rss_bytes", sample)
            self.assertIn("timestamp_ns", sample)

    def test_dir_selection_picks_largest_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "small.bin").write_bytes(os.urandom(1024))
            (root / "large.bin").write_bytes(os.urandom(65536))
            collector = PageCacheEvidenceCollector(path=root, duration_s=0.0)
            self.assertEqual(collector.path.name, "large.bin")

    def test_collector_run_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "backing.bin").write_bytes(os.urandom(4096))
            collector = PageCacheEvidenceCollector(path=root, sample_interval_s=0.0, duration_s=0.0)
            samples = collector.run()
            self.assertGreaterEqual(len(samples), 1)
            jsonl_path = root / "samples.jsonl"
            write_jsonl(jsonl_path, samples)
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(samples))
            csv_path = root / "samples.csv"
            write_csv(csv_path, samples)
            self.assertIn("resident_pages", csv_path.read_text(encoding="utf-8").splitlines()[0])

    def test_cli_runs_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "backing.bin").write_bytes(os.urandom(4096))
            output = Path(raw) / "out.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "runtime" / "collect_page_cache_evidence.py"),
                    "--path",
                    str(root),
                    "--output",
                    str(output),
                    "--duration-s",
                    "0",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
