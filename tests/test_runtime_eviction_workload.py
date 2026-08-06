import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astrakv.runtime.eviction import RuntimeEvictionEvent, write_runtime_events_jsonl
from scripts.reporting import compare_offline_runtime_eviction as compare_script
from scripts.benchmark.run_real_benchmark import (
    load_workload_rows,
    run_one_request,
    validate_configured_workload,
)
from scripts.reporting.normalize_runtime_eviction_events import load_request_objects
from scripts.vm.run_workload_mmap_eviction import load_workload_objects
from astrakv.runtime.eviction import ObjectLevel


class RuntimeEvictionWorkloadTests(unittest.TestCase):
    def test_workload_manifest_sorts_by_arrival_and_result_has_link_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "workload.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"request_id": "req-2", "prompt": "second", "arrival_index": 2, "prefix_id": "p-a", "reuse_ratio": 0.5, "reuse_bucket": "medium"}),
                    json.dumps({"request_id": "req-1", "prompt": "first", "arrival_index": 1, "prefix_id": "p-a", "reuse_ratio": 0.5, "reuse_bucket": "medium"}),
                ]) + "\n",
                encoding="utf-8",
            )
            rows = load_workload_rows(str(path))
            self.assertEqual([row["request_id"] for row in rows], ["req-1", "req-2"])

        stream = iter([
            {"id": "chatcmpl-1", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case", request_id="req-1", batch_size=1, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt",
                request_metadata={
                    "run_id": "run-a", "prefix_id": "p-a", "prefix_hash": "hash-a",
                    "arrival_index": 1, "reuse_ratio": 0.75, "reuse_bucket": "high", "cache_key": "cache-a", "prompt_hash": "prompt-hash",
                    "workflow_id": "wf-a", "parent_request_id": "parent-a", "subtask_index": 2,
                    "cache_state": "warm",
                },
            )
        self.assertEqual(result.request_id, "req-1")
        self.assertEqual(result.run_id, "run-a")
        self.assertEqual(result.prefix_id, "p-a")
        self.assertEqual(result.arrival_index, 1)
        self.assertEqual(result.reuse_bucket, "high")
        self.assertEqual(result.reuse_ratio, 0.75)
        self.assertEqual(result.cache_key, "cache-a")
        self.assertEqual(result.prompt_hash, "prompt-hash")
        self.assertEqual(result.workflow_id, "wf-a")
        self.assertEqual(result.parent_request_id, "parent-a")
        self.assertEqual(result.subtask_index, 2)
        self.assertEqual(result.cache_state, "warm")
        self.assertEqual(result.endpoint_response_id, "chatcmpl-1")
        self.assertIsNotNone(result.first_sse_ms)
        self.assertIsNotNone(result.ttft_ms)
        self.assertLessEqual(result.first_sse_ms, result.ttft_ms)

    def test_configured_run_requires_canonical_workload_path(self) -> None:
        with self.assertRaisesRegex(SystemExit, "canonical workload"):
            validate_configured_workload("", True)
        validate_configured_workload("tests/fixtures/runtime_workload_v1.jsonl", True)

    def test_request_without_generated_content_is_marked_as_error(self) -> None:
        stream = iter([
            {"id": "chatcmpl-empty", "choices": [{"delta": {"role": "assistant"}}]},
            {"usage": {"completion_tokens": 0}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case", request_id="req-empty", batch_size=1, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt", request_metadata={"run_id": "run-a"},
            )

        self.assertEqual(result.status, "error")
        self.assertIn("generated content tokens", result.error)
        self.assertEqual(result.output_tokens_observed, 0)
        self.assertIsNotNone(result.first_sse_ms)
        self.assertIsNone(result.ttft_ms)

    def test_request_with_content_keeps_positive_output_tokens_when_usage_reports_zero(self) -> None:
        stream = iter([
            {"id": "chatcmpl-usage-zero", "choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 0}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case", request_id="req-usage-zero", batch_size=1, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, prompt="prompt", request_metadata={"run_id": "run-a"},
            )

        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.output_tokens_observed, 1)
        self.assertIsNotNone(result.ttft_ms)

    def test_request_and_workload_mappings_are_built_without_runtime_imports(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "request_results.jsonl"
            path.write_text(json.dumps({
                "request_id": "req-1", "prefix_id": "p-a", "cache_key": "c-a",
                "prompt": "prompt", "arrival_index": 3, "reuse_ratio": 0.5, "reuse_bucket": "medium",
            }) + "\n", encoding="utf-8")

            mapping = load_request_objects(str(path))
            self.assertEqual(mapping["req-1"]["prefix_id"], "p-a")
            objects = load_workload_objects(str(path))
            self.assertIn((ObjectLevel.PREFIX, "p-a"), objects)
            self.assertIn((ObjectLevel.CACHE_KEY, "c-a"), objects)

    def test_comparison_cli_writes_fixture_agreement_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            decisions = tmp / "decisions.csv"
            events = tmp / "events.jsonl"
            workload = tmp / "workload.jsonl"
            output = tmp / "agreement"
            decisions.write_text(
                "chunk_id,case,cache_key,action,size_bytes,metadata\n"
                "chunk-a,req-1,,offload,64,\"{\"\"prefix_id\"\":\"\"prefix-a\"\",\"\"arrival_index\"\":1}\"\n",
                encoding="utf-8",
            )
            workload.write_text(json.dumps({"request_id": "req-2", "arrival_index": 2}) + "\n", encoding="utf-8")
            write_runtime_events_jsonl(events, [RuntimeEvictionEvent(
                run_id="run-a", runtime_event_id="runtime-1", request_id="req-2",
                object_key="prefix-a", object_level=ObjectLevel.PREFIX, actual_action="evict",
                tier_after="disk", bytes=64, timestamp_ns=2, arrival_index=2,
                status="completed", provenance="runtime_structured",
            )])
            with patch("sys.argv", [
                "compare_offline_runtime_eviction.py", "--offline-decisions", str(decisions),
                "--runtime-events", str(events), "--workload-manifest", str(workload),
                "--run-id", "run-a", "--output-dir", str(output),
            ]):
                self.assertEqual(compare_script.main(), 0)

            manifest = json.loads((output / "eviction_agreement_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["ground_truth_status"], "valid")
            self.assertEqual(manifest["summary"]["metrics"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
