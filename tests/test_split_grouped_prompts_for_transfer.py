"""Unit tests for the shared-context train/test split tool."""

from __future__ import annotations

import unittest

from scripts.benchmark.split_grouped_prompts_for_transfer import split_rows


class SplitGroupedPromptsTests(unittest.TestCase):
    def test_shared_contexts_appear_on_both_sides_with_different_questions(self) -> None:
        rows = []
        for context in range(5):
            for question in range(3):
                rows.append({
                    "context_hash": f"c{context}",
                    "question_hash": f"q{context}-{question}",
                    "order": context * 3 + question,
                    "prompt": f"p{context}-{question}",
                })
        train, test, stats = split_rows(rows, train_ratio=0.6, seed=0)

        train_contexts = {row["context_hash"] for row in train}
        test_contexts = {row["context_hash"] for row in test}
        train_questions = {row["question_hash"] for row in train}
        test_questions = {row["question_hash"] for row in test}

        self.assertEqual(train_contexts, test_contexts)
        self.assertEqual(stats["contexts_shared_both_sides"], 5)
        self.assertEqual(stats["overlap_contexts"], 5)
        self.assertFalse(train_questions & test_questions)
        self.assertEqual(len(train) + len(test), len(rows))

    def test_single_question_contexts_go_to_one_side_only(self) -> None:
        rows = [
            {"context_hash": "c1", "question_hash": "q1", "order": 0},
            {"context_hash": "c2", "question_hash": "q2", "order": 1},
        ]
        train, test, stats = split_rows(rows, train_ratio=0.5, seed=0)

        self.assertEqual(stats["contexts_shared_both_sides"], 0)
        self.assertEqual(stats["contexts_single_side"], 2)
        self.assertEqual(len(train) + len(test), 2)


if __name__ == "__main__":
    unittest.main()
