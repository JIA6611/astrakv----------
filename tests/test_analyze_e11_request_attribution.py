from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.analyze_e11_request_attribution import analyze


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AnalyzeE11RequestAttributionTest(unittest.TestCase):
    def _arm(self, root: Path, arm: str, ttft: list[float], victims: list[str]) -> None:
        run = root / arm / "rep-1" / "qasper" / "baseline"
        run.mkdir(parents=True)
        _write_jsonl(run / "request_results.jsonl", [
            {
                "sample_id": f"sample-{index}",
                "request_id": f"request-{index}",
                "status": "ok",
                "ttft_ms": value,
                "arrival_index": index,
                "reuse_bucket": "high" if index else "none",
                "reuse_ratio": 0.5 if index else 0.0,
                "request_started_s": 1.0 + index,
                "request_ended_s": 1.5 + index,
            }
            for index, value in enumerate(ttft)
        ])
        rows = []
        for index, victim in enumerate(victims):
            rows.append({
                "selection_id": f"{arm}-{index}",
                "status": "selected",
                "backend_key_identity": victim,
                "timestamp_ns": int((1.75 + index) * 1_000_000_000),
                "selection_duration_ns": 1_000_000 + index,
                "policy_scoring_duration_ns": 500_000,
                "candidate_scan_count": 3,
                "fallback_candidate_count": 0,
                "signals": {"logical_object_key": victim},
            })
            rows.append({
                **rows[-1],
                "status": "completed",
                "timestamp_ns": int((1.8 + index) * 1_000_000_000),
            })
        _write_jsonl(run / "native_cache_policy_evictions.jsonl", rows)
        _write_jsonl(run / "kv_core_native_receipts.jsonl", [{
            "status": "completed",
            "logical_request_id": "request-1",
            "bytes_loaded": 1024,
            "load_latency_ns": 2_000_000,
        }])
        _write_jsonl(run / "runtime_events_raw.jsonl", [{
            "action": "cache_hit",
            "status": "completed",
            "object_key": victims[0],
            "request_id": "later-request",
            "timestamp_ns": 3_000_000_000,
        }])

    def test_localizes_requests_around_first_victim_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._arm(root, "arm-lru", [100.0, 100.0, 100.0], ["a", "b"])
            self._arm(root, "arm-evict-b", [130.0, 90.0, 80.0], ["a", "c"])

            result = analyze(root)
            cell = result["cells"]["rep-1/qasper"]

            self.assertEqual(cell["first_divergence_ordinal"], 2)
            self.assertEqual(cell["requests"][0]["phase"], "pre_divergence")
            self.assertEqual(cell["requests"][2]["phase"], "post_divergence")
            self.assertAlmostEqual(cell["requests"][0]["positive_regression_share"], 1.0)
            self.assertEqual(cell["native_load"]["arm-lru"]["bytes_loaded"], 1024)
            self.assertTrue(cell["selector_overhead"]["arm-evict-b"]["instrumented"])
            self.assertEqual(result["evidence_gaps"], [])

    def test_missing_optional_evidence_is_not_reported_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._arm(root, "arm-lru", [100.0], ["a"])
            self._arm(root, "arm-evict-b", [100.0], ["b"])
            for arm in ("arm-lru", "arm-evict-b"):
                run = root / arm / "rep-1" / "qasper" / "baseline"
                (run / "kv_core_native_receipts.jsonl").unlink()
                rows = [
                    {key: value for key, value in row.items() if not key.endswith("duration_ns")}
                    for row in (
                        json.loads(line)
                        for line in (run / "native_cache_policy_evictions.jsonl").read_text().splitlines()
                    )
                ]
                _write_jsonl(run / "native_cache_policy_evictions.jsonl", rows)

            result = analyze(root)
            gaps = result["evidence_gaps"]

            self.assertIn("native_load_receipts_missing:rep-1/qasper", gaps)
            self.assertIn("selector_timing_missing:rep-1/qasper", gaps)
            native = result["cells"]["rep-1/qasper"]["native_load"]["arm-lru"]
            self.assertIsNone(native["bytes_loaded"])
            overhead = result["cells"]["rep-1/qasper"]["selector_overhead"]["arm-lru"]
            self.assertIsNone(overhead["selection_total_ms"])


if __name__ == "__main__":
    unittest.main()
