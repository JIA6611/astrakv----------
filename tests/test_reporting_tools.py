import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astrakv.runtime.cache_events import (
    parse_benchmark_results,
    parse_log_line,
    parse_request_results,
    summarize_events,
    write_events_jsonl,
)
from scripts.reporting.analyze_stress_results import StressInput, looks_oom, summarize_run
from scripts.reporting.compare_real_runs import RunInput, comparison_claim, compare_runs, validate_run_pair
from scripts.benchmark.run_selective_prefetch_real import (
    build_prefetch_benchmark_rows,
    infer_prefetch_hit,
    json_safe_metadata,
)
from astrakv.runtime.endpoint_prefetch import EndpointResult
from astrakv.prefetch.async_engine import PrefetchStatus
from scripts.reporting.build_observation_feasibility_report import main as build_observation_main


class ReportingToolTests(unittest.TestCase):
    def test_cache_event_parser_extracts_hit_and_load(self) -> None:
        events = parse_log_line(
            "2026-06-08 10:00:00 request_id=req-1 LMCache hit tokens: 128, need to load: 64 local_cpu",
            source="server.log",
            line_number=7,
        )

        self.assertIsInstance(events, list)
        records = [event.to_record() for event in events]  # type: ignore[union-attr]
        self.assertEqual([record["event_type"] for record in records], ["cache_hit", "cache_load"])
        self.assertEqual(records[0]["request_id"], "req-1")
        self.assertEqual(records[0]["tier"], "cpu")
        self.assertEqual(records[0]["metadata"]["hit_tokens"], 128)
        self.assertEqual(records[1]["status"], "planned")
        self.assertEqual(records[1]["metadata"]["tokens"], 64)

    def test_cache_event_file_parsers_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            requests_path = tmp / "request_results.jsonl"
            benchmark_path = tmp / "benchmark_results.csv"
            events_path = tmp / "cache_events.jsonl"

            requests_path.write_text(
                json.dumps(
                    {
                        "request_id": "req-1",
                        "status": "ok",
                        "case": "bs1_ctx512_out64",
                        "latency_ms": 123.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case",
                        "backend",
                        "request_count",
                        "success_count",
                        "latency_p95_ms",
                        "gpu_memory_peak_mb",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case": "bs1_ctx512_out64",
                        "backend": "vllm",
                        "request_count": 1,
                        "success_count": 1,
                        "latency_p95_ms": 123.0,
                        "gpu_memory_peak_mb": 1000,
                    }
                )

            events = parse_request_results(requests_path) + parse_benchmark_results(benchmark_path)
            summary = summarize_events(events)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["event_type_counts"]["request_result"], 1)
            self.assertEqual(summary["event_type_counts"]["benchmark_case_metrics"], 1)

            write_events_jsonl(events, events_path)
            written = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(written), 2)
            self.assertEqual(written[0]["schema"], "astra-cache-event-v1")

    def test_compare_real_runs_aligns_rows_and_calculates_memory_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline = tmp / "baseline.csv"
            variant = tmp / "variant.csv"
            fieldnames = [
                "case",
                "backend",
                "model",
                "batch_size",
                "context_length",
                "output_tokens",
                "request_count",
                "success_count",
                "ttft_ms",
                "tpot_ms",
                "latency_ms",
                "latency_p95_ms",
                "throughput_tokens_s",
                "process_rss_peak_mb",
                "cpu_memory_peak_mb",
                "gpu_memory_peak_mb",
                "gpu_util_peak_pct",
                "disk_read_delta_mb",
                "disk_write_delta_mb",
                "errors",
            ]
            write_rows(
                baseline,
                fieldnames,
                [
                    {
                        "case": "bs1_ctx512_out64",
                        "backend": "vllm",
                        "model": "m",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_ms": 1000,
                        "latency_p95_ms": 1100,
                        "throughput_tokens_s": 50,
                        "process_rss_peak_mb": 500,
                        "cpu_memory_peak_mb": 500,
                        "gpu_memory_peak_mb": 1000,
                        "gpu_util_peak_pct": 80,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                        "errors": "",
                    }
                ],
            )
            write_rows(
                variant,
                fieldnames,
                [
                    {
                        "case": "bs1_ctx512_out64",
                        "backend": "lmcache_cpu",
                        "model": "m",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 90,
                        "tpot_ms": 11,
                        "latency_ms": 950,
                        "latency_p95_ms": 1000,
                        "throughput_tokens_s": 52,
                        "process_rss_peak_mb": 800,
                        "cpu_memory_peak_mb": 800,
                        "gpu_memory_peak_mb": 700,
                        "gpu_util_peak_pct": 75,
                        "disk_read_delta_mb": 10,
                        "disk_write_delta_mb": 20,
                        "errors": "",
                    }
                ],
            )

            rows = compare_runs([RunInput("vllm", baseline), RunInput("cpu", variant)])
            variant_row = next(row for row in rows if row["run"] == "cpu")
            self.assertTrue(variant_row["baseline_matched"])
            self.assertEqual(variant_row["ttft_ms_delta"], -10.0)
            self.assertEqual(variant_row["process_rss_peak_mb"], 800.0)
            self.assertEqual(variant_row["gpu_memory_reduction_pct_vs_baseline"], 30.0)
            self.assertEqual(variant_row["success_rate"], 1.0)

    def test_pair_validation_requires_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = validate_run_pair([
                RunInput("baseline", tmp / "baseline.csv"),
                RunInput("variant", tmp / "variant.csv"),
            ])

            self.assertFalse(result.eligible)
            self.assertIn("missing_manifest:baseline", result.errors)

    def test_comparison_claim_is_paired_by_default_or_explicitly_non_paired(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            runs = [RunInput("baseline", tmp / "baseline.csv"), RunInput("variant", tmp / "variant.csv")]
            claim, validation = comparison_claim(runs, unpaired=False)
            self.assertEqual(claim, "paired_claim_blocked")
            self.assertIsNotNone(validation)
            self.assertFalse(validation.eligible)
            self.assertEqual(comparison_claim(runs, unpaired=True), ("non_paired_no_claims", None))

    def test_stress_summary_reports_capacity_and_oom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            csv_path = tmp / "benchmark_results.csv"
            fieldnames = [
                "case",
                "batch_size",
                "context_length",
                "output_tokens",
                "request_count",
                "success_count",
                "latency_p95_ms",
                "tpot_p95_ms",
                "process_rss_peak_mb",
                "gpu_memory_peak_mb",
                "cpu_memory_peak_mb",
                "disk_read_delta_mb",
                "disk_write_delta_mb",
                "status",
                "errors",
            ]
            write_rows(
                csv_path,
                fieldnames,
                [
                    {
                        "case": "bs1_ctx2048_out64",
                        "batch_size": 1,
                        "context_length": 2048,
                        "output_tokens": 64,
                        "request_count": 3,
                        "success_count": 3,
                        "latency_p95_ms": 900,
                        "tpot_p95_ms": 20,
                        "process_rss_peak_mb": 1000,
                        "gpu_memory_peak_mb": 10000,
                        "cpu_memory_peak_mb": 1000,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                        "status": "ok",
                        "errors": "",
                    },
                    {
                        "case": "bs4_ctx8192_out64",
                        "batch_size": 4,
                        "context_length": 8192,
                        "output_tokens": 64,
                        "request_count": 3,
                        "success_count": 0,
                        "latency_p95_ms": "",
                        "tpot_p95_ms": "",
                        "process_rss_peak_mb": 1200,
                        "gpu_memory_peak_mb": 16000,
                        "cpu_memory_peak_mb": 1200,
                        "disk_read_delta_mb": 128,
                        "disk_write_delta_mb": 256,
                        "status": "error",
                        "errors": "CUDA out of memory during allocation",
                    },
                ],
            )

            summary = summarize_run(StressInput("stress", csv_path))
            self.assertEqual(summary["total_cases"], 2)
            self.assertEqual(summary["success_requests"], 3)
            self.assertEqual(summary["oom_case_count"], 1)
            self.assertEqual(summary["max_success_context"], 2048)
            self.assertEqual(summary["process_rss_peak_mb"], 1200.0)
            self.assertEqual(summary["gpu_memory_peak_mb"], 16000.0)
            self.assertTrue(looks_oom("CUDA out of memory"))
            self.assertFalse(looks_oom("ordinary memory sample text"))

    def test_endpoint_prefetch_hit_heuristic_and_json_safe_metadata(self) -> None:
        no_prefetch = EndpointResult(
            request_id="cold",
            status="ok",
            latency_ms=1000.0,
            ttft_ms=200.0,
            output_tokens_observed=4,
            throughput_tokens_s=4.0,
        )
        demand = EndpointResult(
            request_id="warm",
            status="ok",
            latency_ms=800.0,
            ttft_ms=150.0,
            output_tokens_observed=4,
            throughput_tokens_s=5.0,
        )

        hit, evidence = infer_prefetch_hit(
            no_prefetch=no_prefetch,
            demand=demand,
            prefetch_status=PrefetchStatus.COMPLETED,
            improvement_threshold_pct=5.0,
        )
        self.assertTrue(hit)
        self.assertEqual(evidence, "latency_improvement_heuristic")

        metadata = json_safe_metadata({"case": "ctx128", "endpoint_request": object(), "priority": 1})
        self.assertEqual(metadata, {"case": "ctx128", "priority": 1})

    def test_prefetch_rows_emit_benchmark_like_schema(self) -> None:
        args = type("Args", (), {"model": "test-model", "output_tokens": 16})()
        rows = [
            {
                "case": "ctx1024_rep0",
                "context_length": 1024,
                "no_prefetch_status": "ok",
                "prefetch_demand_status": "ok",
                "no_prefetch_ttft_ms": 200,
                "prefetch_demand_ttft_ms": 150,
                "no_prefetch_latency_ms": 1000,
                "prefetch_demand_latency_ms": 800,
                "no_prefetch_output_tokens": 16,
                "prefetch_demand_output_tokens": 16,
                "no_prefetch_error": "",
                "prefetch_demand_error": "",
            }
        ]

        benchmark_rows = build_prefetch_benchmark_rows(rows, args)
        self.assertEqual([row["backend"] for row in benchmark_rows], ["astrakv_no_prefetch", "astrakv_prefetch_demand"])
        self.assertEqual(benchmark_rows[0]["request_count"], 1)
        self.assertEqual(benchmark_rows[1]["success_count"], 1)
        self.assertEqual(benchmark_rows[1]["ttft_ms"], 150.0)
        for field in (
            "case",
            "backend",
            "model",
            "request_count",
            "success_count",
            "ttft_ms",
            "latency_ms",
            "latency_p95_ms",
            "throughput_tokens_s",
            "process_rss_peak_mb",
            "cpu_memory_peak_mb",
            "gpu_memory_peak_mb",
            "disk_read_delta_mb",
            "disk_write_delta_mb",
        ):
            self.assertIn(field, benchmark_rows[0])

    def test_observation_report_exposes_load_target_state_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            candidate_report = tmp / "predictor_candidate_report.jsonl"
            request_results = tmp / "request_results.jsonl"
            runtime_events_raw = tmp / "runtime_events_raw.jsonl"
            runtime_command_receipts = tmp / "runtime_command_receipts.jsonl"
            runtime_structured_events = tmp / "runtime_structured_events.jsonl"
            backend_binding_events = tmp / "backend_binding_events.jsonl"
            online_profile_checkpoint = tmp / "online_profile_checkpoint.json"
            output_dir = tmp / "report"

            candidate_report.write_text(
                json.dumps({
                    "schema": "astrakv-predictor-candidate-report-v1",
                    "request_id": "req-1",
                    "candidate_object_id": "prefix-a",
                    "object_level": "prefix",
                    "predicted_class": "exact-next",
                    "lead_distance_requests": 1,
                    "estimated_reusable_tokens": 128,
                    "estimated_kv_bytes": 256,
                    "confidence": 0.95,
                    "reason": "exact_next_locality",
                }) + "\n",
                encoding="utf-8",
            )
            write_jsonl(request_results, [
                {"request_id": "req-1", "cache_key": "prefix-a", "request_started_s": 1.0, "request_ended_s": 1.2, "arrival_index": 1},
                {"request_id": "req-2", "cache_key": "prefix-a", "request_started_s": 2.2, "request_ended_s": 3.0, "arrival_index": 2},
            ])
            write_jsonl(runtime_events_raw, [
                {"object_key": "prefix-a", "timestamp_ns": 1_500_000_000, "tier_after": "ssd"},
            ])
            write_jsonl(runtime_command_receipts, [])
            write_jsonl(runtime_structured_events, [])
            write_jsonl(backend_binding_events, [
                {
                    "record_type": "binding",
                    "request_id": "req-1",
                    "object_key": "prefix-a",
                    "backend_object_id": "backend-a",
                    "binding_id": "binding-a",
                    "binding_generation": 1,
                    "execution_spec": {
                        "actions": {
                            "prefetch": {"status": "ready"},
                            "load": {"status": "ready", "load_target_id": "load-target-1"},
                        }
                    },
                }
            ])
            online_profile_checkpoint.write_text(json.dumps({
                "schema": "astrakv-online-profile-v1",
                "run_id": "run-a",
                "event_count": 0,
                "dispatch_count": 0,
                "last_event_id": None,
                "objects": {
                    "backend-a": {
                        "last_load_target_state": "consumed",
                        "last_load_target_consumed_at_ns": 123,
                        "prefetch_waste_count": 0,
                        "current_tier": "ssd",
                        "active_reference_count": 0,
                        "pending_operations": {},
                        "action_reservation": None,
                    }
                },
                "controller_state": {"breaker": {"state": "closed"}},
            }), encoding="utf-8")

            with patch(
                "scripts.reporting.build_observation_feasibility_report.parse_args",
                return_value=type(
                    "Args",
                    (),
                    {
                        "candidate_report": str(candidate_report),
                        "request_results": str(request_results),
                        "runtime_events_raw": str(runtime_events_raw),
                        "runtime_command_receipts": str(runtime_command_receipts),
                        "runtime_structured_events": str(runtime_structured_events),
                        "backend_binding_events": str(backend_binding_events),
                        "online_profile_checkpoint": str(online_profile_checkpoint),
                        "output_dir": str(output_dir),
                        "minimum_lead_time_ms": 250.0,
                    },
                )(),
            ):
                self.assertEqual(build_observation_main(), 0)

            rows = [json.loads(line) for line in (output_dir / "observation_feasibility_report.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(rows[0]["feasibility_status"], "prefetchable_now")
            self.assertTrue(rows[0]["load_target_stale"])
            self.assertEqual(rows[0]["load_target_state"], "consumed")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
