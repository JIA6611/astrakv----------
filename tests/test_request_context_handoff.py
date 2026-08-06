import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from astrakv.runtime.request_context import (
    InMemoryLoopbackRequestContextClient,
    RequestContextJsonlArtifact,
    RequestContextReceipt,
    RuntimeRequestContext,
    RuntimeRequestContextAuthority,
    RuntimeRequestContextReceiver,
    RuntimeRequestIdentity,
)
from scripts.benchmark.run_real_benchmark import build_runtime_request_context_client, run_one_request


class RequestContextHandoffTests(unittest.TestCase):
    def test_runner_selects_authenticated_client_only_with_complete_runtime_secret(self) -> None:
        client = build_runtime_request_context_client(
            "http://127.0.0.1:9988/request-context", run_id="run-7", session_id="session-7", secret_hex="ab" * 32,
        )
        self.assertEqual(client.__class__.__name__, "AuthenticatedJsonHttpRequestContextClient")

    def test_runtime_receiver_records_signed_context_then_associates_only_at_reqmeta_lifecycle(self) -> None:
        authority = RuntimeRequestContextAuthority.install(
            run_id="run-7", session_id="session-7", secret=b"k" * 32, ttl_s=30,
        )
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        context = RuntimeRequestContext("run-7", "logical-request", "case-a", "nonce-7", 1.0)
        headers = authority.context_headers(context, receiver.endpoint_identity, now_ns=100)

        recorded = receiver.receive(context.to_record(), headers, now_ns=101)
        self.assertEqual(recorded.status, "recorded")
        self.assertEqual(recorded.runtime_request_id, "")
        self.assertEqual(recorded.to_record()["record_type"], "request_context_receipt")
        self.assertTrue(authority.verify_receipt(recorded, receiver.endpoint_identity))
        self.assertEqual(receiver.receive(context.to_record(), headers, now_ns=102).status, "recorded")

        associated = receiver.associate_runtime_request(
            "reqmeta-7", RuntimeRequestIdentity("run-7", "logical-request", "nonce-7"), now_ns=103,
        )
        self.assertEqual(associated.status, "associated")
        self.assertEqual(associated.runtime_request_id, "reqmeta-7")
        self.assertTrue(authority.verify_receipt(associated, receiver.endpoint_identity))
        self.assertEqual(receiver.associated_context("reqmeta-7").request_id, "logical-request")

    def test_runtime_receiver_rejects_unsigned_expired_and_conflicting_context_replay(self) -> None:
        authority = RuntimeRequestContextAuthority.install(
            run_id="run-7", session_id="session-7", secret=b"k" * 32, ttl_s=30,
        )
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        context = RuntimeRequestContext("run-7", "logical-request", "case-a", "nonce-7", 1.0)
        with self.assertRaisesRegex(ValueError, "authentication"):
            receiver.receive(context.to_record(), {}, now_ns=1)
        expired = authority.context_headers(context, receiver.endpoint_identity, now_ns=1)
        with self.assertRaisesRegex(ValueError, "expired"):
            receiver.receive(context.to_record(), expired, now_ns=31_000_000_002)

        headers = authority.context_headers(context, receiver.endpoint_identity, now_ns=100)
        receiver.receive(context.to_record(), headers, now_ns=101)
        conflicting = RuntimeRequestContext("run-7", "other-request", "case-a", "nonce-7", 1.0)
        conflicting_headers = authority.context_headers(conflicting, receiver.endpoint_identity, now_ns=102)
        with self.assertRaisesRegex(ValueError, "nonce replay conflict"):
            receiver.receive(conflicting.to_record(), conflicting_headers, now_ns=103)

    def test_runner_writes_versioned_context_and_marks_matching_loopback_association_linked(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            artifact_path = Path(raw_tmp) / "request_context.jsonl"
            published = []
            client = InMemoryLoopbackRequestContextClient(
                "http://127.0.0.1:9988/request-context",
                responder=lambda context: published.append(context) or RequestContextReceipt(
                    run_id=context.run_id,
                    request_id=context.request_id,
                    request_nonce=context.request_nonce,
                    runtime_request_id="runtime-request-7",
                    runtime_event_id="runtime-event-7",
                    status="associated",
                ),
            )
            stream = iter([
                {"choices": [{"delta": {"content": "ok"}}]},
                {"usage": {"completion_tokens": 1}, "choices": []},
            ])
            with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream) as stream_request:
                result = run_one_request(
                    base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                    case="case-a", request_id="logical-request", batch_size=1, context_length=4,
                    output_tokens=2, timeout=1, temperature=0, top_p=1, system_prompt="system",
                    prompt_seed="seed", prompt_token_scale=1, request_metadata={"run_id": "run-7"},
                    request_nonce="11111111-1111-4111-8111-111111111111", request_context_client=client,
                    request_context_artifact=RequestContextJsonlArtifact(artifact_path),
                )

            self.assertEqual(stream_request.call_args.args[1]["_astrakv_request_id"], "logical-request")

            self.assertEqual(len(published), 1)
            self.assertEqual(published[0].schema, "astrakv-request-context-v1")
            self.assertEqual(published[0].run_id, "run-7")
            self.assertEqual(published[0].request_id, "logical-request")
            self.assertEqual(published[0].case, "case-a")
            self.assertEqual(published[0].request_nonce, "11111111-1111-4111-8111-111111111111")
            self.assertEqual(result.runtime_association_status, "linked")
            self.assertEqual(result.runtime_request_id, "runtime-request-7")
            self.assertEqual(result.runtime_event_id, "runtime-event-7")
            record = json.loads(artifact_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["schema"], "astrakv-request-context-v1")
            self.assertEqual(record["request_nonce"], "11111111-1111-4111-8111-111111111111")
            self.assertEqual(RuntimeRequestContext.from_record(record), published[0])

    def test_unmatched_or_nonloopback_context_cannot_claim_a_runtime_link(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            InMemoryLoopbackRequestContextClient("http://192.168.1.7/context", lambda _context: None)
        with self.assertRaisesRegex(ValueError, "numeric"):
            InMemoryLoopbackRequestContextClient("http://localhost:9988/request-context", lambda _context: None)

        client = InMemoryLoopbackRequestContextClient(
            "http://127.0.0.1:9988/request-context",
            responder=lambda context: RequestContextReceipt(
                run_id=context.run_id,
                request_id=context.request_id,
                request_nonce="wrong-nonce",
                runtime_request_id="runtime-request-7",
                status="recorded",
            ),
        )
        stream = iter([
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", request_id="logical-request", batch_size=1, context_length=4,
                output_tokens=2, timeout=1, temperature=0, top_p=1, system_prompt="system",
                prompt_seed="seed", prompt_token_scale=1, request_metadata={"run_id": "run-7"},
                request_nonce="22222222-2222-4222-8222-222222222222", request_context_client=client,
            )

        self.assertEqual(result.runtime_association_status, "unlinked")
        self.assertEqual(result.runtime_request_id, "")

    def test_recorded_receipt_cannot_claim_runtime_association(self) -> None:
        client = InMemoryLoopbackRequestContextClient(
            "http://127.0.0.1:9988/request-context",
            responder=lambda context: RequestContextReceipt(
                run_id=context.run_id,
                request_id=context.request_id,
                request_nonce=context.request_nonce,
                runtime_request_id="runtime-request-7",
                runtime_event_id="runtime-event-7",
                status="recorded",
            ),
        )
        stream = iter([
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", request_id="logical-request", batch_size=1, context_length=4,
                output_tokens=2, timeout=1, temperature=0, top_p=1, system_prompt="system",
                prompt_seed="seed", prompt_token_scale=1, request_metadata={"run_id": "run-7"},
                request_nonce="33333333-3333-4333-8333-333333333333", request_context_client=client,
            )

        self.assertEqual(result.runtime_association_status, "unlinked")
        self.assertEqual(result.runtime_request_id, "")

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
