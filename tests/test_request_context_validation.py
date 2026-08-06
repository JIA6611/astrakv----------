import unittest
from unittest.mock import patch
from uuid import uuid4

from astrakv.runtime.request_context import RequestContextReceipt
from scripts.benchmark.run_real_benchmark import run_one_request


class RequestContextValidationTests(unittest.TestCase):
    def test_receipt_parser_requires_its_exact_record_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "record_type"):
            RequestContextReceipt.from_record({
                "schema": "astrakv-request-context-v1",
                "record_type": "request_context",
                "run_id": "run-7",
                "request_id": "logical-request",
                "request_nonce": "nonce-7",
            })

    def test_direct_request_nonce_must_be_a_uuid_and_cannot_be_replayed(self) -> None:
        kwargs = {
            "base_url": "http://endpoint", "api_key": "empty", "model": "model", "backend": "backend",
            "case": "case-direct", "request_id": "logical-request", "batch_size": 1, "context_length": 4,
            "output_tokens": 2, "timeout": 1, "temperature": 0, "top_p": 1, "system_prompt": "system",
            "prompt_seed": "seed", "prompt_token_scale": 1, "request_metadata": {"run_id": "run-direct"},
        }
        with self.assertRaisesRegex(ValueError, "UUID"):
            run_one_request(**kwargs, request_nonce="not-a-uuid")

        nonce = str(uuid4())
        stream = iter([
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            run_one_request(**kwargs, request_nonce=nonce)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_one_request(**kwargs, request_nonce=nonce)


if __name__ == "__main__":
    unittest.main()
