from __future__ import annotations

import json
import unittest
from pathlib import Path

from astrakv.benchmarks.workflow_observer import (
    load_task1_prompt_records,
    observe_workflow_rows,
    task1_prompt_records_to_workflow_rows,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return [sum(map(ord, item["content"])) for item in messages] + [7]


class Task1WorkflowObservationTests(unittest.TestCase):
    def test_loads_original_prompt_records_without_reordering(self):
        root = Path("datasets/task1_qasper")
        source = [json.loads(line) for line in (root / "prompts/qasper_random_prompts.jsonl").read_text(encoding="utf-8").splitlines()]

        records = load_task1_prompt_records(root, workload_type="random", limit=2)

        self.assertEqual(records, source[:2])

    def test_preserves_task_one_messages_and_order(self):
        path = Path("datasets/task1_qasper/prompts/qasper_grouped_prompts.jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:3]]

        rows = task1_prompt_records_to_workflow_rows(records, workload_type="grouped")
        observations = observe_workflow_rows(
            rows, tokenizer=FakeTokenizer(), block_size_tokens=1, kv_bytes_per_token=8
        )

        self.assertEqual([row.arrival_index for row in rows], [item["order"] for item in records])
        self.assertEqual(rows[0].messages, tuple(records[0]["messages"]))
        self.assertEqual(rows[0].adapter, "single_request")
        self.assertEqual(len(observations), 3)


if __name__ == "__main__":
    unittest.main()
