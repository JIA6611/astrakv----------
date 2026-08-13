"""Unit tests for the temporal adaptation analyzer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.analyze_prefetch_adaptation import analyze_role


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AnalyzePrefetchAdaptationTests(unittest.TestCase):
    def test_windows_attribute_prefetch_evidence_to_request_time_spans(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            role_dir = root / "variant"
            state_dir = root / "variant" / "state"
            _write_jsonl(role_dir / "request_results.jsonl", [
                {"arrival_index": 0, "request_started_s": 100.0, "request_ended_s": 110.0,
                 "status": "ok", "ttft_ms": 500.0},
                {"arrival_index": 1, "request_started_s": 200.0, "request_ended_s": 210.0,
                 "status": "ok", "ttft_ms": 600.0},
                {"arrival_index": 2, "request_started_s": 300.0, "request_ended_s": 310.0,
                 "status": "ok", "ttft_ms": 700.0},
            ])
            _write_jsonl(state_dir / "kv_core_policy_decisions.jsonl", [
                {"action": "prefetch_ssd_to_cpu", "prefetch_id": "p1", "timestamp_ns": 105_000_000_000},
                {"action": "prefetch_ssd_to_cpu", "prefetch_id": "p2", "timestamp_ns": 205_000_000_000},
            ])
            _write_jsonl(state_dir / "kv_core_prefetch_tickets.jsonl", [
                {"prefetch_id": "p1", "status": "consumed"},
                {"prefetch_id": "p2", "status": "consumed"},
            ])
            _write_jsonl(state_dir / "runtime_command_receipts.jsonl", [
                {"action": "prefetch", "status": "completed",
                 "metadata": {"prefetched": 1}, "timestamp_ns": 106_000_000_000},
                {"action": "prefetch", "status": "completed",
                 "metadata": {"prefetched": 1}, "timestamp_ns": 206_000_000_000},
            ])

            summary = analyze_role(role_dir, state_dir, windows=3)

        self.assertEqual(summary["window_count"], 3)
        windows = {row["window"]: row for row in summary["windows"]}
        self.assertEqual(windows[0]["prefetch_a"]["decision_count"], 1)
        self.assertEqual(windows[0]["prefetch_b"]["completed_with_bytes"], 1)
        self.assertEqual(windows[1]["prefetch_a"]["decision_count"], 1)
        self.assertEqual(windows[1]["prefetch_b"]["completed_with_bytes"], 1)
        self.assertEqual(windows[2]["prefetch_a"]["decision_count"], 0)
        self.assertEqual(windows[0]["ttft_p50_ms"], 500.0)
        self.assertEqual(windows[2]["ttft_p50_ms"], 700.0)


if __name__ == "__main__":
    unittest.main()
