from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SummarizePrefetchPhaseTTFTTests(unittest.TestCase):
    def test_near_phase_is_paired_and_reports_directional_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline = tmp / "baseline.jsonl"
            variant = tmp / "variant.jsonl"
            rows = [
                {"request_id": "first-1", "prefetch_phase": "first", "prefetch_pair_id": "g1", "ttft_ms": 100},
                {"request_id": "far-1", "prefetch_phase": "far", "prefetch_pair_id": "g1", "ttft_ms": 200},
                {"request_id": "near-1", "prefetch_phase": "near", "prefetch_pair_id": "g1", "ttft_ms": 100},
                {"request_id": "near-2", "prefetch_phase": "near", "prefetch_pair_id": "g2", "ttft_ms": 200},
            ]
            baseline.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            variant.write_text(
                "\n".join(
                    json.dumps({**row, "ttft_ms": value})
                    for row, value in zip(rows, (110, 210, 50, 150), strict=True)
                ) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/reporting/summarize_prefetch_phase_ttft.py"),
                    "--baseline", str(baseline), "--variant", str(variant), "--phase", "near",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["paired_count"], 2)
            self.assertEqual(record["variant_wins"], 2)
            self.assertLess(record["p50_delta_percent"], 0.0)
            self.assertEqual(record["baseline_request_count"], 2)


if __name__ == "__main__":
    unittest.main()
