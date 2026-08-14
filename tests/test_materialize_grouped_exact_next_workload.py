"""Unit tests for grouped exact-next workload materialization (interleave)."""

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

from scripts.benchmark.materialize_grouped_exact_next_workload import (  # noqa: E402
    interleave_groups,
)


def _row(group: str, order: int, prompt: str = "shared context") -> dict:
    return {
        "prompt": prompt,
        "reuse_group": group,
        "order": order,
        "request_id": f"req-{order:04d}",
        "context_hash": f"ctx-{group}",
        "shared_context": True,
        "max_tokens": 128,
    }


class InterleaveGroupsTests(unittest.TestCase):
    def test_round_robin_spaces_same_group_visits(self) -> None:
        rows = [
            _row("g1", 0), _row("g1", 1), _row("g1", 2),
            _row("g2", 3), _row("g2", 4),
            _row("g3", 5),
        ]
        interleaved = interleave_groups(rows)
        self.assertEqual(len(interleaved), 6)
        self.assertEqual({r["reuse_group"] for r in interleaved}, {"g1", "g2", "g3"})
        groups = [r["reuse_group"] for r in interleaved]
        # No two consecutive rows may belong to the same reuse group.
        for previous, current in zip(groups, groups[1:]):
            self.assertNotEqual(previous, current, f"consecutive {previous} visits")
        # Multiset of reuse groups is preserved.
        from collections import Counter
        self.assertEqual(Counter(groups), Counter(["g1", "g1", "g1", "g2", "g2", "g3"]))
        # Intra-group order is preserved (request ids ascending per group).
        g1_orders = [int(r["order"]) for r in interleaved if r["reuse_group"] == "g1"]
        self.assertEqual(g1_orders, [0, 1, 2])

    def test_cli_interleave_canonical_has_gapped_revisits(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source = tmp / "grouped_prompts.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                for row in [
                    _row("g1", 0), _row("g1", 1),
                    _row("g2", 2), _row("g2", 3),
                    _row("g3", 4), _row("g3", 5),
                ]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_dir = tmp / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/benchmark/materialize_grouped_exact_next_workload.py"),
                    "--grouped-prompts-jsonl", str(source),
                    "--output-dir", str(out_dir),
                    "--dataset", "qasper",
                    "--task", "qasper",
                    "--limit", "6",
                    "--interleave",
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            canonical = out_dir / "qasper_grouped_exact_next_canonical_workload.jsonl"
            rows = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 6)
            groups = [r["prefix_id"] for r in rows]
            for previous, current in zip(groups, groups[1:]):
                self.assertNotEqual(previous, current)
            # arrival_index follows the new execution order.
            self.assertEqual([r["arrival_index"] for r in rows], list(range(6)))


if __name__ == "__main__":
    unittest.main()
