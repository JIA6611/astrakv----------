from __future__ import annotations

import unittest

from astrakv.benchmarks.workflow_observer import WorkflowTraceRow, observe_workflow_rows


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_flags = (tokenize, add_generation_prompt)
        return [sum(map(ord, message["content"])) for message in messages] + [99]


class BatchEncodingLikeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return {"input_ids": [7, 8, 9]}


def row(index: int, content: str, *, workflow_id: str = "wf") -> WorkflowTraceRow:
    return WorkflowTraceRow(
        workflow_id=workflow_id,
        parent_request_id=workflow_id,
        subtask_index=0,
        arrival_index=index,
        messages=({"role": "user", "content": content},),
        dataset_id="fixture",
        workload_id="fixture-random",
        adapter="single_request",
    )


class WorkflowObserverTests(unittest.TestCase):
    def test_observes_only_prior_matching_blocks(self):
        tokenizer = FakeTokenizer()
        observations = observe_workflow_rows(
            [row(0, "aa", workflow_id="a"), row(1, "aa", workflow_id="b"), row(2, "bb", workflow_id="c")],
            tokenizer=tokenizer,
            block_size_tokens=1,
            kv_bytes_per_token=16,
        )

        self.assertEqual(tokenizer.last_flags, (True, True))
        self.assertEqual(observations[0].historical_reused_tokens, 0)
        self.assertEqual(observations[1].historical_reused_tokens, 2)
        self.assertEqual(observations[1].historical_reuse_count, 2)
        self.assertEqual(observations[1].potential_kv_bytes, 32)
        self.assertEqual(observations[2].historical_reused_tokens, 1)
        self.assertEqual(observations[0].block_hashes, observations[1].block_hashes)

    def test_rejects_duplicate_identity_and_non_monotonic_arrival(self):
        with self.assertRaisesRegex(ValueError, "duplicate workflow/subtask"):
            observe_workflow_rows([row(0, "x"), row(1, "y")], tokenizer=FakeTokenizer(), block_size_tokens=1, kv_bytes_per_token=1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            observe_workflow_rows([row(1, "x"), row(0, "y", workflow_id="other")], tokenizer=FakeTokenizer(), block_size_tokens=1, kv_bytes_per_token=1)

    def test_rejects_invalid_messages(self):
        invalid = WorkflowTraceRow("wf", "wf", 0, 0, ({"role": "tool", "content": "x"},), "d", "w", "single_request")
        with self.assertRaisesRegex(ValueError, "role"):
            observe_workflow_rows([invalid], tokenizer=FakeTokenizer(), block_size_tokens=1, kv_bytes_per_token=1)

    def test_accepts_input_ids_from_tokenizer_batch_encoding(self):
        observations = observe_workflow_rows(
            [row(0, "x")],
            tokenizer=BatchEncodingLikeTokenizer(),
            block_size_tokens=2,
            kv_bytes_per_token=1,
        )

        self.assertEqual(observations[0].token_count, 3)
        self.assertEqual(observations[0].historical_reused_tokens, 0)


if __name__ == "__main__":
    unittest.main()
