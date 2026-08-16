import base64
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from astrakv.runtime.moe_prepare import (
    MoePrepareClient,
    MoePrepareConfig,
    MoePrepareResult,
    decode_routed_experts,
    routed_experts_to_events,
)
from astrakv.runtime.request_context import RequestContextJsonlArtifact
from scripts.benchmark.run_real_benchmark import run_one_request


def encode_array(array: np.ndarray) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    np.save(buffer, array)
    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), raw


class StubMoePrepareClient(MoePrepareClient):
    def __init__(self, *, response=None, failure=None, **kwargs):
        super().__init__(**kwargs)
        self.response = response
        self.failure = failure
        self.payload = None

    def _post_json(self, url, payload, *, request_id):
        self.payload = payload
        if self.failure is not None:
            raise self.failure
        return self.response


class MoePrepareTests(unittest.TestCase):
    def test_decodes_valid_routed_experts(self) -> None:
        source = np.array([[[0, 3], [1, 2]], [[2, 1], [3, 0]]], dtype=np.uint8)
        encoded, raw = encode_array(source)

        decoded, decoded_raw = decode_routed_experts(
            encoded, expected_layers=2, expected_top_k=2, max_expert_id=3,
        )

        np.testing.assert_array_equal(decoded, source)
        self.assertEqual(decoded_raw, raw)

    def test_rejects_wrong_shape_dtype_and_corrupt_payload(self) -> None:
        wrong_shape, _ = encode_array(np.zeros((2, 3), dtype=np.uint8))
        wrong_dtype, _ = encode_array(np.zeros((2, 3, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_routed_experts(wrong_shape)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            decode_routed_experts(wrong_dtype)
        with self.assertRaisesRegex(ValueError, "valid base64"):
            decode_routed_experts("not base64")

    def test_events_preserve_token_layer_expert_and_rank(self) -> None:
        source = np.array([[[7, 3], [5, 1]]], dtype=np.uint8)
        events = list(
            routed_experts_to_events(source, request_id="request-a", token_start=255)
        )
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].request_id, "request-a")
        self.assertEqual(events[0].token_index, 255)
        self.assertEqual(events[0].layer_id, 0)
        self.assertEqual(events[0].expert_id, "7")
        self.assertEqual(events[0].expert_rank, 0)
        self.assertEqual(events[-1].expert_id, "1")

    def test_success_writes_receipt_events_summary_and_hashed_raw_array(self) -> None:
        array = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 0]]], dtype=np.uint8)
        encoded, raw = encode_array(array)
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            client = StubMoePrepareClient(
                base_url="http://endpoint",
                api_key="EMPTY",
                output_dir=root,
                config=MoePrepareConfig(
                    enabled=True, capture_window_tokens=2,
                    expected_layers=2, expected_top_k=2, max_expert_id=3,
                ),
                response={"choices": [{"routed_experts": encoded}]},
            )

            result = client.prepare(
                messages=[{"role": "user", "content": "hello"}],
                model="moe-model",
                request_id="request/a",
                exact_token_ids=(10, 11, 12, 13),
                context_published=True,
                runtime_association_status="linked",
                prefix_id="prefix-a",
            )

            self.assertEqual(result.status, "prepared")
            self.assertEqual(result.route_event_count, 8)
            self.assertEqual(result.unique_experts, 4)
            self.assertEqual(client.payload["routed_experts_prompt_start"], 2)
            raw_path = root / result.raw_routing_path
            self.assertEqual(raw_path.read_bytes(), raw)
            self.assertEqual(result.raw_routing_sha256, hashlib.sha256(raw).hexdigest())
            receipts = read_jsonl(root / "moe_prepare_receipts.jsonl")
            events = read_jsonl(root / "moe_route_events.jsonl")
            summary = json.loads((root / "moe_route_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (root / "moe_routed_experts_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipts[0]["runtime_association_status"], "linked")
            self.assertEqual(receipts[0]["route_token_start"], 2)
            self.assertEqual(len(events), 8)
            self.assertEqual(summary["successful_prepare_count"], 1)
            self.assertEqual(summary["route_event_count"], 8)
            self.assertEqual(manifest["entries"][0]["sha256"], result.raw_routing_sha256)

    def test_prepare_failure_is_recorded_and_fail_open_returns(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            client = StubMoePrepareClient(
                base_url="http://endpoint",
                api_key="EMPTY",
                output_dir=root,
                config=MoePrepareConfig(enabled=True, fail_open=True),
                failure=TimeoutError("prepare timed out"),
            )
            result = client.prepare(
                messages=[{"role": "user", "content": "hello"}],
                model="moe-model",
                request_id="request-failure",
                exact_token_ids=(1, 2),
                context_published=True,
                runtime_association_status="unlinked",
            )
            self.assertEqual(result.status, "prepare_failed")
            self.assertIn("prepare timed out", result.error)
            self.assertEqual(
                read_jsonl(root / "moe_prepare_receipts.jsonl")[0]["status"],
                "prepare_failed",
            )

    def test_run_one_request_prepares_before_measured_http_request(self) -> None:
        class RecordingClient:
            called_s = 0.0

            def prepare(self, **_kwargs):
                self.called_s = time.time()
                return MoePrepareResult(
                    status="prepared", latency_ms=4.5, route_event_count=16,
                    unique_experts=4, model_type="qwen3_5_moe",
                )

        stream = iter([
            {
                "id": "chatcmpl-a",
                "choices": [{
                    "delta": {"content": "ok"},
                    "token_ids": [42],
                    "logprobs": {"content": [{"token_id": 42, "logprob": -0.1}]},
                }],
            },
            {"choices": [{"finish_reason": "length", "delta": {}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        recorder = RecordingClient()
        with tempfile.TemporaryDirectory() as raw_tmp, patch(
            "scripts.benchmark.run_real_benchmark.stream_chat_completion",
            return_value=stream,
        ):
            result = run_one_request(
                base_url="http://endpoint", api_key="EMPTY", model="moe-model",
                backend="vllm-lmcache-kv-core", case="case", request_id="request-order",
                batch_size=1, context_length=3, output_tokens=1, timeout=1,
                temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt",
                request_metadata={
                    "run_id": "run-order",
                    "metadata": {"exact_token_ids": [1, 2, 3]},
                },
                request_nonce=str(uuid4()),
                request_context_artifact=RequestContextJsonlArtifact(
                    Path(raw_tmp) / "request_context.jsonl"
                ),
                moe_prepare_client=recorder,
            )

        self.assertEqual(result.status, "ok")
        self.assertLessEqual(recorder.called_s, result.request_started_s)
        self.assertEqual(result.moe_prepare_status, "prepared")
        self.assertEqual(result.moe_prepare_latency_ms, 4.5)
        self.assertEqual(result.moe_route_event_count, 16)
        self.assertEqual(result.moe_unique_experts, 4)
        self.assertEqual(result.moe_model_type, "qwen3_5_moe")

    def test_launcher_keeps_moe_flags_behind_explicit_mode(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "launch" / "launch_vllm_server.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('MOE_MODE="${ASTRAKV_MOE_MODE:-false}"', launcher)
        conditional = 'if [[ "$MOE_MODE" == "true" ]]; then'
        self.assertIn(conditional, launcher)
        self.assertIn("CMD+=(--language-model-only --enable-return-routed-experts)", launcher)
        self.assertLess(launcher.index("CMD=("), launcher.rindex(conditional))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
