"""Unit tests for the medium-tier reorder helper."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.reorder_grouped_prompts_for_transfer import reorder_rows  # noqa: E402


class ReorderGroupedPromptsTests(unittest.TestCase):
    def test_reorder_preserves_rows_and_changes_order(self) -> None:
        rows = [
            {"request_id": f"r{i}", "reuse_group": f"g{i % 3}", "order": i, "prompt": f"p{i}"}
            for i in range(9)
        ]
        reordered = reorder_rows(rows, seed=1)
        self.assertEqual(len(reordered), 9)
        # Same multiset of prompts, deterministic permutation.
        self.assertEqual(
            sorted(r["request_id"] for r in reordered),
            sorted(r["request_id"] for r in rows),
        )
        self.assertEqual([r["order"] for r in reordered], list(range(9)))
        self.assertNotEqual([r["request_id"] for r in reordered], [r["request_id"] for r in rows])
        # Reuse groups travel with their rows.
        by_id = {r["request_id"]: r["reuse_group"] for r in rows}
        self.assertTrue(all(r["reuse_group"] == by_id[r["request_id"]] for r in reordered))
        # Same seed reproduces the same permutation.
        self.assertEqual(
            [r["request_id"] for r in reordered],
            [r["request_id"] for r in reorder_rows(rows, seed=1)],
        )


if __name__ == "__main__":
    unittest.main()
