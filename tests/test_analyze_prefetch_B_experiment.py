"""Unit tests for the Prefetch-B acceptance analyzer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reporting.analyze_prefetch_B_experiment import analyze, visit_buckets  # noqa: E402


PA = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PB = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AnalyzePrefetchBExperimentTests(unittest.TestCase):
    def test_visit_buckets_classifies_fire_consume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            canon = Path(tmp) / "canonical.jsonl"
            rows = [
                {"arrival_index": 0, "cache_key": PA, "request_id": "r0"},
                {"arrival_index": 1, "cache_key": PB, "request_id": "r3"},
                {"arrival_index": 2, "cache_key": PA, "request_id": "r1"},
                {"arrival_index": 3, "cache_key": PA, "request_id": "r2"},
                {"arrival_index": 4, "cache_key": PB, "request_id": "r4"},
                {"arrival_index": 5, "cache_key": PB, "request_id": "r5"},
            ]
            _write_jsonl(canon, rows)
            buckets, visits = visit_buckets(canon)
        self.assertEqual(buckets[0], "first")
        self.assertEqual(buckets[1], "first")
        self.assertEqual(buckets[2], "far")
        self.assertEqual(buckets[3], "near")
        self.assertEqual(buckets[4], "far")
        self.assertEqual(buckets[5], "near")
        self.assertEqual(visits[PA], [0, 2, 3])

    def test_analyze_produces_acceptance_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "profile-test" / "qasper"
            canon = exp / "materialized/qasper_grouped_exact_next_canonical_workload.jsonl"
            _write_jsonl(canon, [
                {"arrival_index": 0, "cache_key": PA, "request_id": "r0"},
                {"arrival_index": 1, "cache_key": PB, "request_id": "r3"},
                {"arrival_index": 2, "cache_key": PA, "request_id": "r1"},
                {"arrival_index": 3, "cache_key": PA, "request_id": "r2"},
                {"arrival_index": 4, "cache_key": PB, "request_id": "r4"},
                {"arrival_index": 5, "cache_key": PB, "request_id": "r5"},
            ])
            for role, ttft in (("baseline", [1000.0, 1000.0, 3000.0, 2900.0, 3100.0, 2900.0]),
                               ("variant", [1000.0, 1000.0, 3100.0, 500.0, 3200.0, 500.0])):
                csv_path = exp / role / "benchmark_results.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["case", "status", "ttft_ms", "disk_read_delta_mb"])
                    writer.writeheader()
                    for i, t in enumerate(ttft):
                        writer.writerow({"case": f"case_{i:05d}", "status": "ok", "ttft_ms": t, "disk_read_delta_mb": 100})
            state = exp / "variant-state"
            _write_jsonl(state / "runtime_command_receipts.jsonl", [
                {"action": "prefetch", "status": "completed", "metadata": {"prefetched": 1, "bytes": 641728512}},
                {"action": "prefetch", "status": "completed", "metadata": {"prefetched": 1, "bytes": 641728512}},
                {"action": "prefetch", "status": "rejected", "metadata": {"failure_reason": "x"}},
            ])
            _write_jsonl(state / "kv_core_native_callbacks.jsonl", [
                {"callback": "scheduler_exact_lookup", "request_id": "r2", "lookup_hit_tokens": 2048, "locally_cached_tokens": 0},
                {"callback": "scheduler_exact_lookup", "request_id": "r0", "lookup_hit_tokens": 0, "locally_cached_tokens": 0},
            ])
            data = analyze(root, "profile-test")
        self.assertEqual(data["prefetch_b"]["completed_prefetched_1"], 2)
        self.assertEqual(data["consumption"]["lmcache_external_hit_count"], 1)
        var_p50 = data["ttft"]["variant"]["ttft_p50_ms"]
        base_p50 = data["ttft"]["baseline"]["ttft_p50_ms"]
        self.assertEqual(base_p50, 2900.0)
        self.assertEqual(var_p50, 1000.0)
        self.assertLess(data["ttft"]["delta_p50_percent"], 0)
        # near bucket (index 3) must be faster than baseline's near (index 3).
        self.assertEqual(data["ttft"]["variant"]["by_bucket"]["near"]["p50_ms"], 500.0)


if __name__ == "__main__":
    unittest.main()
