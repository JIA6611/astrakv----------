from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark.materialize_profile_transfer_subset import materialize


def _row(context: str, request: str, prompt_hash: str, size: int) -> dict[str, object]:
    return {
        "context_hash": context,
        "reuse_group": context,
        "request_id": request,
        "sample_id": request,
        "prompt_hash": prompt_hash,
        "prompt": "x" * size,
        "shared_context": True,
    }


class MaterializeProfileTransferSubsetTests(unittest.TestCase):
    def test_builds_disjoint_repeated_visits_over_shared_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            train_rows = [
                _row("ctx-a", "train-a", "hash-train-a", 20),
                _row("ctx-b", "train-b", "hash-train-b", 10),
                _row("ctx-only-train", "train-only", "hash-only", 1),
            ]
            test_rows = [
                _row("ctx-a", "test-a", "hash-test-a", 30),
                _row("ctx-b", "test-b", "hash-test-b", 15),
                _row("ctx-only-test", "test-only", "hash-test-only", 1),
            ]
            for path, rows in ((train, train_rows), (test, test_rows)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            manifest = materialize(
                train, test, root / "out", dataset="qasper", limit=2, visits=3,
            )

            self.assertTrue(manifest["anti_leakage"]["passed"])
            self.assertEqual(manifest["anti_leakage"]["shared_prefix_count"], 2)
            self.assertEqual(manifest["outputs"]["train_rows"], 6)
            self.assertEqual(manifest["outputs"]["test_rows"], 6)
            selected = manifest["selected"]
            self.assertEqual([row["context_key"] for row in selected], ["ctx-b", "ctx-a"])
            output_train = [
                json.loads(line)
                for line in (root / "out/train/qasper/grouped_prompts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len({row["request_id"] for row in output_train}), 6)
            self.assertEqual({row["transfer_split"] for row in output_train}, {"train"})

    def test_rejects_cross_split_prompt_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            train.write_text(
                json.dumps(_row("ctx", "train", "same-hash", 10)) + "\n",
                encoding="utf-8",
            )
            test.write_text(
                json.dumps(_row("ctx", "test", "same-hash", 10)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only 0 disjoint"):
                materialize(
                    train, test, root / "out", dataset="qasper", limit=1, visits=3,
                )


if __name__ == "__main__":
    unittest.main()
