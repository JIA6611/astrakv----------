import unittest
from unittest.mock import patch

from scripts.benchmark.run_real_benchmark import (
    extract_choice_token_ids,
    normalize_exact_token_ids,
    resolve_workload_context_length,
    run_one_request,
)


class RealBenchmarkTokenEvidenceTests(unittest.TestCase):
    def test_context_length_uses_declared_or_tokenizer_profile_metadata(self) -> None:
        self.assertEqual(
            resolve_workload_context_length({"context_length": 128}),
            (128, "workload.context_length"),
        )
        self.assertEqual(
            resolve_workload_context_length({"metadata": {"context_token_estimate": 256}}),
            (256, "workload.metadata.context_token_estimate"),
        )
        self.assertEqual(resolve_workload_context_length({}), (0, "missing"))

    def test_extracts_vllm_logprob_token_ids_when_extension_field_is_absent(self) -> None:
        choice = {
            "delta": {"content": "hello"},
            "logprobs": {
                "content": [
                    {"token": "hello", "token_id": 123, "logprob": -0.1},
                    {"token": "!", "token_id": 456, "logprob": -0.2},
                ]
            },
        }
        self.assertEqual(extract_choice_token_ids(choice), [123, 456])

    def test_prefers_explicit_token_ids_over_logprob_fallback(self) -> None:
        choice = {
            "token_ids": [7, 8],
            "delta": {"token_ids": [9]},
            "logprobs": {"content": [{"token_id": 10}]},
        }
        self.assertEqual(extract_choice_token_ids(choice), [7, 8])

    def test_normalizes_qwen3_transformers_batch_encoding_input_ids(self) -> None:
        # Transformers 5.12 Qwen3 ``apply_chat_template`` returns a
        # BatchEncoding-like mapping instead of a bare integer list.
        self.assertEqual(
            normalize_exact_token_ids({"input_ids": [151644, 872, 198]}),
            (151644, 872, 198),
        )

    def test_rejects_batch_encoding_without_a_valid_input_id_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty integer sequence"):
            normalize_exact_token_ids({"input_ids": []})

    def test_kv_core_request_fails_closed_without_token_evidence(self) -> None:
        stream = iter([
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"finish_reason": "length", "delta": {}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model",
                backend="vllm-lmcache-kv-core", case="case", request_id="req-token-evidence",
                batch_size=1, context_length=32, output_tokens=1, timeout=1,
                temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt",
                request_metadata={"run_id": "run-token-evidence"},
                request_nonce="44444444-4444-4444-8444-444444444444",
            )

        self.assertEqual(result.status, "error")
        self.assertIn("token IDs", result.error)

    def test_kv_core_payload_enables_vllm_stream_token_ids(self) -> None:
        stream = iter([
            {"choices": [{"delta": {"content": "hello"}, "token_ids": [123]}]},
            {"choices": [{"finish_reason": "length", "delta": {}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream) as request:
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model",
                backend="vllm-lmcache-kv-core", case="case", request_id="req-payload",
                batch_size=1, context_length=32, output_tokens=1, timeout=1,
                temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt",
                request_metadata={"run_id": "run-payload"},
                request_nonce="55555555-5555-4555-8555-555555555555",
            )

        self.assertEqual(result.status, "ok")
        payload = request.call_args.args[1]
        self.assertTrue(payload["return_token_ids"])
        self.assertNotIn("return_tokens_as_token_ids", payload)


if __name__ == "__main__":
    unittest.main()
