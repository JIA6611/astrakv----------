"""Unit tests for the Prefetch-A/B 2x2 ablation validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.validate_prefetch_2x2_ablation import _aggregate_cell


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Prefetch2x2ValidatorTests(unittest.TestCase):
    def test_cell_aggregation_counts_b_receipts_and_a_tickets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            run_dir = root / "qasper" / "variant"
            state_dir = root / "qasper" / "variant-state"
            _write_jsonl(run_dir / "runtime_command_receipts.jsonl", [
                {"action": "prefetch", "status": "completed",
                 "metadata": {"prefetched": 1, "failure_reason": None}},
                {"action": "prefetch", "status": "completed",
                 "metadata": {"prefetched": 0, "failure_reason": "already_cpu_resident"}},
            ])
            _write_jsonl(state_dir / "kv_core_policy_decisions.jsonl", [
                {"action": "prefetch_ssd_to_cpu", "status": "submitted", "reason": ""},
                {"action": "invalidate_external_copy", "cpu_removed_chunk_count": 3},
            ])
            _write_jsonl(state_dir / "kv_core_prefetch_tickets.jsonl", [
                {"prefetch_id": "p1", "status": "consumed"},
                {"prefetch_id": "p2", "status": "wasted"},
            ])
            (run_dir / "benchmark_results.csv").write_text(
                "case,status,ttft_ms\n"
                "c1,ok,100.0\n"
                "c2,ok,200.0\n"
                "c3,error,300.0\n",
                encoding="utf-8",
            )

            cell = _aggregate_cell(root, "qasper", "variant")

        self.assertEqual(cell["prefetch_b"]["completed_with_bytes"], 1)
        self.assertEqual(cell["prefetch_b"]["already_cpu_noop"], 1)
        self.assertEqual(cell["prefetch_a"]["submitted"], 1)
        self.assertEqual(cell["prefetch_a"]["tickets_consumed"], 1)
        self.assertEqual(cell["prefetch_a"]["tickets_wasted"], 1)
        self.assertEqual(cell["conflict_signals"]["invalidate_removed_chunk_count"], 3)
        self.assertTrue(cell["conflict_signals"]["dual_accounting_ticket_consumed_and_b_completed"])
        self.assertEqual(cell["benchmark"]["success_count"], 2)
        self.assertEqual(cell["benchmark"]["ttft_p50_ms"], 100.0)
        self.assertEqual(cell["benchmark"]["ttft_p95_ms"], 200.0)

    def test_missing_artifacts_aggregate_to_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            cell = _aggregate_cell(root, "qasper", "baseline")
        self.assertEqual(cell["prefetch_b"]["receipt_count"], 0)
        self.assertEqual(cell["prefetch_a"]["decision_count"], 0)
        self.assertEqual(cell["prefetch_a"]["ticket_statuses"], {})
        self.assertEqual(cell["conflict_signals"]["invalidate_external_copy_count"], 0)


if __name__ == "__main__":
    unittest.main()
