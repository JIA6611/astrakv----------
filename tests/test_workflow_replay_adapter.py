from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from astrakv.benchmarks.workflow_observer import (
    load_replay_workflow_rows,
    observe_workflow_rows,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return [sum(map(ord, message["content"])) for message in messages] + [99]


FIXTURE = Path(__file__).with_name("fixtures") / "workflow_replay_v1.jsonl"


class WorkflowReplayAdapterTests(unittest.TestCase):
    def test_loads_callback_export_without_reconstructing_payloads(self):
        rows = load_replay_workflow_rows(FIXTURE)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].adapter, "replay_jsonl")
        self.assertEqual(rows[0].model_identifier, "Qwen3-8B")
        self.assertEqual(rows[0].tool_output_sha256, "a" * 64)
        self.assertEqual(rows[0].messages, ({"role": "user", "content": "Explain prefix reuse."},))
        observations = observe_workflow_rows(
            rows, tokenizer=FakeTokenizer(), block_size_tokens=1, kv_bytes_per_token=8
        )
        self.assertEqual(observations[1].historical_reused_tokens, 2)
        self.assertEqual(observations[0].to_record()["tool_output_sha256"], "a" * 64)

    def test_rejects_non_monotonic_callback_export(self):
        first_record, second_record = (
            json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        )
        first_record["arrival_index"] = 1
        second_record["arrival_index"] = 0
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "bad.jsonl"
            trace.write_text(
                json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_replay_workflow_rows(trace)

    def test_rejects_raw_tool_output_or_hidden_reasoning(self):
        record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
        record["tool_output"] = "private payload"
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "bad.jsonl"
            trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported replay fields"):
                load_replay_workflow_rows(trace)

    def test_rejects_non_string_message_content_without_coercion(self):
        record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
        record["messages"][0]["content"] = 7
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "bad.jsonl"
            trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content"):
                load_replay_workflow_rows(trace)


if __name__ == "__main__":
    unittest.main()
