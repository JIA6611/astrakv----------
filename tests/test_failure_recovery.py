import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.runtime.failure_recovery import (
    FailureRecoveryPolicy,
    RecoveryAction,
    classify_failure,
    decide_recovery,
    parse_benchmark_results,
    parse_cache_events,
    parse_prefetch_events,
    parse_request_results,
    parse_scheduler_hints,
    summarize_failure_recovery,
    write_failure_events_jsonl,
    write_fallback_hints_jsonl,
    write_recovery_decisions_csv,
)
from scripts.reporting.analyze_failure_recovery import collect_failure_events, write_manifest, write_report


class FailureRecoveryTests(unittest.TestCase):
    def test_classifies_common_failure_text(self) -> None:
        self.assertEqual(classify_failure("CUDA out of memory"), "memory_oom")
        self.assertEqual(classify_failure("Endpoint connection refused"), "endpoint_failure")
        self.assertEqual(classify_failure("HTTP 500"), "http_error")
        self.assertEqual(classify_failure("profile missing"), "profile_missing")

    def test_benchmark_and_request_failures_map_to_recovery_actions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            request_results = tmp / "request_results.jsonl"
            write_rows(
                benchmark,
                ["case", "backend", "model", "request_count", "success_count", "status", "errors"],
                [
                    {
                        "case": "oom",
                        "backend": "vllm",
                        "model": "qwen",
                        "request_count": 2,
                        "success_count": 0,
                        "status": "error",
                        "errors": "CUDA out of memory",
                    }
                ],
            )
            request_results.write_text(
                json.dumps(
                    {
                        "request_id": "req-1",
                        "case": "endpoint",
                        "status": "error",
                        "error": "Endpoint connection refused",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events = [*parse_benchmark_results(benchmark), *parse_request_results(request_results)]
            decisions = decide_recovery(events, FailureRecoveryPolicy())
            actions = {decision.failure_type: decision.action for decision in decisions}

            self.assertEqual(actions["memory_oom"], RecoveryAction.REDUCE_WORKLOAD)
            self.assertEqual(actions["endpoint_failure"], RecoveryAction.FALLBACK_BASELINE)

    def test_prefetch_cache_and_scheduler_failures_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            prefetch = tmp / "prefetch_events.jsonl"
            cache = tmp / "cache_events.jsonl"
            hints = tmp / "scheduler_hints.jsonl"
            prefetch.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_type": "prefetch_failed",
                                "status": "failed",
                                "request_id": "pref-1",
                                "chunk_id": "chunk-a",
                                "message": "timeout",
                            }
                        ),
                        json.dumps(
                            {
                                "event_type": "prefetch_waste",
                                "status": "observed",
                                "request_id": "pref-2",
                                "chunk_id": "chunk-b",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cache.write_text(
                json.dumps(
                    {
                        "event_type": "cache_load",
                        "status": "partial_or_failed",
                        "request_id": "req-cache",
                        "chunk_id": "chunk-c",
                        "raw_line": "retrieve fail",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hints.write_text(
                json.dumps(
                    {
                        "request_id": "req-hint",
                        "action": "drop",
                        "reason": "drop requested",
                        "metadata": {"chunk_id": "chunk-d"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events = [*parse_prefetch_events(prefetch), *parse_cache_events(cache), *parse_scheduler_hints(hints)]
            decisions = decide_recovery(events)
            action_by_failure = {decision.failure_type: decision.action for decision in decisions}

            self.assertEqual(action_by_failure["prefetch_failed"], RecoveryAction.DISABLE_PREFETCH)
            self.assertEqual(action_by_failure["prefetch_waste"], RecoveryAction.DISABLE_PREFETCH)
            self.assertEqual(action_by_failure["cache_load_failed"], RecoveryAction.RECOMPUTE)
            self.assertEqual(action_by_failure["scheduler_drop"], RecoveryAction.SKIP_OBJECT)

    def test_missing_artifact_and_outputs_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            missing = tmp / "missing.jsonl"
            events = parse_prefetch_events(missing)
            decisions = decide_recovery(events)
            events_path = tmp / "failure_events.jsonl"
            decisions_path = tmp / "recovery_decisions.csv"
            hints_path = tmp / "fallback_hints.jsonl"
            report_path = tmp / "failure_recovery_report.md"
            manifest_path = tmp / "failure_recovery_manifest.json"
            args = Namespace(
                benchmark_results=[],
                request_results=[],
                prefetch_events=[str(missing)],
                cache_events=[],
                scheduler_hints=[],
            )

            write_failure_events_jsonl(events_path, events)
            write_recovery_decisions_csv(decisions_path, decisions)
            write_fallback_hints_jsonl(hints_path, decisions)
            write_report(report_path, args, events, decisions, events_path, decisions_path, hints_path)
            write_manifest(manifest_path, args, events, decisions, events_path, decisions_path, hints_path, report_path)

            failure_records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(failure_records[0]["failure_type"], "artifact_missing")
            with decisions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["action"], "collect_evidence")
            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(hints[0]["action"], "collect_evidence")
            self.assertIn("# Failure Recovery And Degradation Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["failure_count"], 1)

    def test_collect_failure_events_uses_all_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            write_rows(
                benchmark,
                ["case", "request_count", "success_count", "status", "errors"],
                [{"case": "a", "request_count": 1, "success_count": 0, "status": "error", "errors": "timeout"}],
            )
            args = Namespace(
                benchmark_results=[str(benchmark)],
                request_results=[],
                prefetch_events=[],
                cache_events=[],
                scheduler_hints=[],
            )

            events = collect_failure_events(args)
            decisions = decide_recovery(events)
            summary = summarize_failure_recovery(events, decisions)

            self.assertEqual(summary["failure_type_counts"], {"timeout": 1})
            self.assertEqual(summary["recovery_action_counts"], {"fallback_baseline": 1})


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
