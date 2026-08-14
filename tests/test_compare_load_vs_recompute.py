"""Unit tests for the load-vs-recompute comparator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.compare_load_vs_recompute import compare


class CompareLoadVsRecomputeTests(unittest.TestCase):
    def test_computes_ttft_and_memory_deltas_between_arms(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            run_dir = Path(raw_tmp)

            def request(role: str, ttft: float, gpu_mb: float, cpu_mb: float) -> dict:
                return {
                    "case": f"kv_equivalence_{role}",
                    "status": "ok",
                    "ttft_ms": ttft,
                    "gpu_memory_mb_after": gpu_mb,
                    "cpu_memory_mb_after": cpu_mb,
                }

            rows = [
                request("seed", 2000.0, 15000.0, 3000.0),
                request("loaded", 150.0, 16000.0, 3200.0),
                request("loaded", 170.0, 16000.0, 3200.0),
                request("recompute", 250.0, 15500.0, 3100.0),
                request("recompute", 260.0, 15500.0, 3100.0),
            ]
            (run_dir / "request_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "kv_equivalence.json").write_text(
                json.dumps({"equivalent": True}), encoding="utf-8",
            )

            summary = compare(run_dir, run_dir / "state")

        self.assertEqual(summary["arms"]["loaded"]["ok_count"], 2)
        self.assertEqual(summary["arms"]["recompute"]["ok_count"], 2)
        self.assertEqual(summary["arms"]["loaded"]["ttft_p50_ms"], 150.0)
        self.assertEqual(summary["arms"]["recompute"]["ttft_p50_ms"], 250.0)
        self.assertAlmostEqual(summary["comparison"]["ttft_p50_delta_percent"], -40.0)
        self.assertTrue(summary["equivalence"]["equivalent"])


if __name__ == "__main__":
    unittest.main()
