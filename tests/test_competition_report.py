import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.reporting.build_competition_report import (
    ArtifactInput,
    summarize_artifacts,
    write_inventory,
    write_manifest,
    write_report,
)


class CompetitionReportTests(unittest.TestCase):
    def test_summarize_benchmark_quality_and_scheduler_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            quality = tmp / "quality_results.csv"
            schedule = tmp / "object_schedule_decisions.csv"

            write_rows(
                benchmark,
                [
                    "case",
                    "request_count",
                    "success_count",
                    "ttft_ms",
                    "tpot_ms",
                    "latency_p95_ms",
                    "process_rss_peak_mb",
                    "gpu_util_peak_pct",
                    "gpu_memory_peak_mb",
                    "cpu_memory_peak_mb",
                    "disk_read_delta_mb",
                    "disk_write_delta_mb",
                ],
                [
                    {
                        "case": "a",
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_p95_ms": 500,
                        "process_rss_peak_mb": 650,
                        "gpu_util_peak_pct": 55,
                        "gpu_memory_peak_mb": 1000,
                        "cpu_memory_peak_mb": 600,
                        "disk_read_delta_mb": 1,
                        "disk_write_delta_mb": 2,
                    },
                    {
                        "case": "b",
                        "request_count": 2,
                        "success_count": 1,
                        "ttft_ms": 200,
                        "tpot_ms": 20,
                        "latency_p95_ms": 800,
                        "process_rss_peak_mb": 750,
                        "gpu_util_peak_pct": 65,
                        "gpu_memory_peak_mb": 1200,
                        "cpu_memory_peak_mb": 700,
                        "disk_read_delta_mb": 3,
                        "disk_write_delta_mb": 4,
                    },
                ],
            )
            write_rows(
                quality,
                ["sample_id", "exact_match", "normalized_match", "token_divergence_rate", "ppl_delta"],
                [
                    {
                        "sample_id": "a",
                        "exact_match": 1,
                        "normalized_match": 1,
                        "token_divergence_rate": 0.0,
                        "ppl_delta": 0.1,
                    },
                    {
                        "sample_id": "b",
                        "exact_match": 0,
                        "normalized_match": 1,
                        "token_divergence_rate": 0.2,
                        "ppl_delta": 0.3,
                    },
                ],
            )
            write_rows(
                schedule,
                ["chunk_id", "action", "gpu_bytes_after"],
                [
                    {"chunk_id": "hot", "action": "prefetch", "gpu_bytes_after": 100},
                    {"chunk_id": "warm", "action": "offload", "gpu_bytes_after": 100},
                ],
            )

            artifacts = [
                ArtifactInput("vllm", "benchmark", benchmark),
                ArtifactInput("quality", "quality", quality),
                ArtifactInput("schedule", "object_schedule", schedule),
            ]
            summaries = summarize_artifacts(artifacts)

            self.assertEqual(summaries[f"vllm:benchmark:{benchmark}"]["success_rate"], 0.75)
            self.assertEqual(summaries[f"vllm:benchmark:{benchmark}"]["mean_ttft_ms"], 150.0)
            self.assertEqual(summaries[f"vllm:benchmark:{benchmark}"]["max_gpu_memory_peak_mb"], 1200.0)
            self.assertEqual(summaries[f"vllm:benchmark:{benchmark}"]["max_process_rss_peak_mb"], 750.0)
            self.assertEqual(summaries[f"vllm:benchmark:{benchmark}"]["max_gpu_util_peak_pct"], 65.0)
            self.assertEqual(summaries[f"quality:quality:{quality}"]["exact_match_rate"], 0.5)
            self.assertEqual(
                summaries[f"schedule:object_schedule:{schedule}"]["action_counts"],
                {"offload": 1, "prefetch": 1},
            )

    def test_report_manifest_and_inventory_mark_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            missing = tmp / "missing.csv"
            write_rows(
                benchmark,
                ["case", "request_count", "success_count", "ttft_ms"],
                [{"case": "a", "request_count": 1, "success_count": 1, "ttft_ms": 10}],
            )
            artifacts = [
                ArtifactInput("vllm", "benchmark", benchmark),
                ArtifactInput("missing_quality", "quality", missing),
            ]
            summaries = summarize_artifacts(artifacts)
            args = SimpleNamespace(title="AstraKV-W Test Report", command=["python test"])
            inventory = tmp / "artifact_inventory.csv"
            manifest = tmp / "manifest.json"
            report = tmp / "competition_report.md"

            write_inventory(inventory, artifacts, summaries)
            write_manifest(manifest, args, artifacts, summaries)
            write_report(report, args, artifacts, summaries, inventory, manifest)

            with inventory.open("r", encoding="utf-8", newline="") as handle:
                inventory_rows = list(csv.DictReader(handle))
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            report_text = report.read_text(encoding="utf-8")

            self.assertEqual(inventory_rows[1]["status"], "missing")
            self.assertEqual(manifest_payload["title"], "AstraKV-W Test Report")
            self.assertIn("## Benchmark Metrics", report_text)
            self.assertIn("RSS MB", report_text)
            self.assertIn("GPU util %", report_text)
            self.assertNotIn("GPU MB | CPU MB", report_text)
            self.assertIn("missing_quality", report_text)
            self.assertIn("## Limitations", report_text)

    def test_summarize_new_evidence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            vm_summary = tmp / "vm_summary.json"
            prefetch = tmp / "prefetch_results.csv"
            cache_events = tmp / "cache_events.jsonl"
            server_log = tmp / "server.log"
            stress = tmp / "stress_summary.csv"

            vm_summary.write_text(
                json.dumps(
                    {
                        "summary": {"prefetch_coverage_rate": 0.75},
                        "latency": {"warm_over_cold_speedup": 2.5},
                    }
                ),
                encoding="utf-8",
            )
            write_rows(
                prefetch,
                [
                    "case",
                    "prefetch_submitted",
                    "prefetch_completed",
                    "prefetch_failed",
                    "prefetch_hit",
                    "prefetch_waste",
                    "ttft_delta_pct",
                    "latency_delta_pct",
                ],
                [
                    {
                        "case": "ctx1024",
                        "prefetch_submitted": 2,
                        "prefetch_completed": 2,
                        "prefetch_failed": 0,
                        "prefetch_hit": 1,
                        "prefetch_waste": 1,
                        "ttft_delta_pct": 10,
                        "latency_delta_pct": 5,
                    }
                ],
            )
            cache_events.write_text(
                "\n".join(
                    [
                        json.dumps({"event_type": "cache_hit", "status": "ok"}),
                        json.dumps({"event_type": "cache_miss", "status": "observed"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            server_log.write_text(
                "Model loading took 13.25 GiB memory\n"
                "Available KV cache memory: 7.5 GiB\n"
                "GPU KV cache size: 123,456 tokens\n",
                encoding="utf-8",
            )
            write_rows(
                stress,
                ["run", "total_cases", "failed_case_count", "oom_case_count", "max_success_context"],
                [{"run": "extreme", "total_cases": 4, "failed_case_count": 1, "oom_case_count": 1, "max_success_context": 8192}],
            )

            artifacts = [
                ArtifactInput("vm", "vm_evidence", vm_summary),
                ArtifactInput("prefetch", "prefetch", prefetch),
                ArtifactInput("cache", "cache_events", cache_events),
                ArtifactInput("server", "server_log", server_log),
                ArtifactInput("stress", "stress", stress),
            ]
            summaries = summarize_artifacts(artifacts)
            self.assertEqual(summaries[f"prefetch:prefetch:{prefetch}"]["prefetch_hit_rate"], 0.5)
            self.assertEqual(summaries[f"cache:cache_events:{cache_events}"]["event_counts"]["cache_hit"], 1)
            self.assertEqual(summaries[f"server:server_log:{server_log}"]["model_loading_memory_gib"], 13.25)
            self.assertEqual(summaries[f"stress:stress:{stress}"]["oom_case_count"], 1)

            report = tmp / "competition_report.md"
            write_report(
                report,
                SimpleNamespace(title="AstraKV-W Test Report", command=[]),
                artifacts,
                summaries,
                tmp / "inventory.csv",
                tmp / "manifest.json",
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("## VM Evidence", text)
            self.assertIn("## Prefetch Evidence", text)
            self.assertIn("## Cache Event Evidence", text)
            self.assertIn("## Server Startup Memory Evidence", text)
            self.assertIn("## Stress Boundary", text)

    def test_report_keeps_runtime_and_vm_agreements_separate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            runtime = tmp / "runtime.json"
            vm = tmp / "vm.json"
            runtime.write_text(json.dumps({
                "comparison_scope": "runtime",
                "summary": {"ground_truth_status": "insufficient_ground_truth", "reason": "no structured events", "metrics": {}},
            }), encoding="utf-8")
            vm.write_text(json.dumps({
                "comparison_scope": "vm_poc",
                "summary": {"ground_truth_status": "valid", "reason": "mmap acknowledgements", "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "object_coverage": 1.0}},
            }), encoding="utf-8")
            artifacts = [
                ArtifactInput("runtime", "eviction_agreement", runtime),
                ArtifactInput("vm", "eviction_agreement", vm),
            ]
            summaries = summarize_artifacts(artifacts)
            report = tmp / "report.md"
            write_report(report, SimpleNamespace(title="Test", command=[]), artifacts, summaries, tmp / "i.csv", tmp / "m.json")
            text = report.read_text(encoding="utf-8")
            self.assertIn("## 真实 Runtime 一致性", text)
            self.assertIn("no structured events", text)
            self.assertIn("## VM PoC 逻辑对象一致性", text)
            self.assertIn("mmap acknowledgements", text)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
