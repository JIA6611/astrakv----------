"""Unit tests for the E3 prefetch acceptance aggregator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.validate_e3_prefetch_acceptance import _aggregate_role


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class E3PrefetchAcceptanceTests(unittest.TestCase):
    def test_aggregates_a_consumed_ticket_and_b_receipts_with_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            role_dir = root / "variant"
            state_dir = root / "variant" / "state"
            _write_jsonl(role_dir / "kv_core_policy_decisions.jsonl", [
                {"action": "prefetch_ssd_to_cpu", "status": "submitted", "reason": ""},
            ])
            _write_jsonl(role_dir / "kv_core_prefetch_tickets.jsonl", [
                {"prefetch_id": "p1", "status": "consumed"},
            ])
            _write_jsonl(state_dir / "runtime_command_receipts.jsonl", [
                {"action": "prefetch", "status": "completed",
                 "metadata": {"prefetched": 1, "failure_reason": None}},
                {"action": "prefetch", "status": "failed",
                 "metadata": {"prefetched": 0, "failure_reason": "cpu_capacity",
                              "cpu_used_bytes": 42, "memory_pressure": 0.9}},
            ])

            row = _aggregate_role(role_dir, state_dir, "variant")

        self.assertEqual(row["prefetch_a"]["submitted"], 1)
        self.assertEqual(row["prefetch_a"]["tickets_consumed"], 1)
        self.assertEqual(row["prefetch_b"]["completed_with_bytes"], 1)
        self.assertEqual(row["prefetch_b"]["failed_count"], 1)
        self.assertEqual(row["prefetch_b"]["failure_reasons"]["cpu_capacity"], 1)
        self.assertEqual(row["prefetch_b"]["failed_with_diagnostics"], 1)

    def test_missing_artifacts_aggregate_to_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            row = _aggregate_role(root, root / "state", "baseline")
        self.assertEqual(row["prefetch_a"]["decision_count"], 0)
        self.assertEqual(row["prefetch_b"]["receipt_count"], 0)


if __name__ == "__main__":
    unittest.main()
