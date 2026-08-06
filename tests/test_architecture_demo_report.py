import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.build_architecture_demo_report import (
    DemoPaths,
    build_demo_summary,
    collect_demo_artifacts,
    summarize_artifacts,
    write_report,
)


class ArchitectureDemoReportTests(unittest.TestCase):
    def test_demo_report_summarizes_prefetch_policy_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            main = tmp / "main"
            boundary_pass = tmp / "pass"
            boundary_fail = tmp / "fail"
            summary = tmp / "summary.md"
            create_demo_fixture(main, boundary_pass, boundary_fail, summary)

            paths = DemoPaths(
                main_evidence=main,
                boundary_pass=boundary_pass,
                boundary_fail=boundary_fail,
                summary=summary,
                command_log=None,
                include_vm_smoke=None,
                include_live_smoke=None,
            )
            artifacts = collect_demo_artifacts(paths)
            summaries = summarize_artifacts(artifacts)
            demo = build_demo_summary(paths, artifacts, summaries, ["demo command"])

            self.assertEqual(demo["prefetch_comparison"]["status"], "ok")
            self.assertAlmostEqual(demo["prefetch_comparison"]["comparisons"][0]["ttft_change_pct"], -50.0)
            self.assertEqual(demo["policy_ablation"]["status"], "ok")
            self.assertAlmostEqual(demo["policy_ablation"]["ttft_delta_pct_vs_baseline"], -25.0)
            self.assertEqual(demo["policy_ablation"]["chunks_scored"], 3.0)
            self.assertEqual(demo["boundary_pass"]["lmcache_disk_write_mb"], 19840.0)
            self.assertEqual(demo["boundary_fail"]["needed_kv_cache_gib"], 1.75)
            self.assertEqual(demo["boundary_fail"]["available_kv_cache_gib"], 1.67)
            self.assertEqual(demo["cache_events"]["boundary_disk_cache_events"]["rows"], 2)

            report_path = tmp / "demo_report.md"
            write_report(
                report_path,
                demo,
                artifacts,
                summaries,
                tmp / "artifact_inventory.csv",
                tmp / "demo_summary.json",
                tmp / "selected_artifacts",
            )
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("赛题要求对照", text)
            self.assertIn("-50.00%", text)
            self.assertIn("19840", text)
            self.assertIn("case-level GPU framebuffer memory", text)

    def test_missing_evidence_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            paths = DemoPaths(
                main_evidence=tmp / "missing_main",
                boundary_pass=tmp / "missing_pass",
                boundary_fail=tmp / "missing_fail",
                summary=tmp / "missing_summary.md",
                command_log=None,
                include_vm_smoke=None,
                include_live_smoke=None,
            )
            artifacts = collect_demo_artifacts(paths)
            summaries = summarize_artifacts(artifacts)
            demo = build_demo_summary(paths, artifacts, summaries, [])

            self.assertEqual(demo["prefetch_comparison"]["status"], "missing")
            self.assertEqual(demo["policy_ablation"]["status"], "missing")
            self.assertEqual(demo["boundary_pass"]["status"], "missing")
            self.assertEqual(demo["boundary_fail"]["status"], "missing")
            self.assertTrue(any(row["status"] == "missing" for row in demo["artifact_status"]))


def create_demo_fixture(main: Path, boundary_pass: Path, boundary_fail: Path, summary: Path) -> None:
    e2e = main / "01_e2e"
    write_rows(
        e2e / "step5_prefetch" / "prefetch_benchmark_results.csv",
        [
            "case",
            "backend",
            "context_length",
            "request_count",
            "success_count",
            "ttft_ms",
            "latency_ms",
        ],
        [
            {
                "case": "ctx1024_no_prefetch",
                "backend": "astrakv_no_prefetch",
                "context_length": 1024,
                "request_count": 1,
                "success_count": 1,
                "ttft_ms": 200,
                "latency_ms": 1000,
            },
            {
                "case": "ctx1024_prefetch_demand",
                "backend": "astrakv_prefetch_demand",
                "context_length": 1024,
                "request_count": 1,
                "success_count": 1,
                "ttft_ms": 100,
                "latency_ms": 900,
            },
        ],
    )
    write_rows(
        e2e / "step5_prefetch" / "prefetch_results.csv",
        ["case", "prefetch_submitted", "prefetch_completed", "prefetch_failed", "prefetch_hit", "prefetch_waste"],
        [{"case": "ctx1024", "prefetch_submitted": 1, "prefetch_completed": 1, "prefetch_failed": 0, "prefetch_hit": 1, "prefetch_waste": 0}],
    )
    write_rows(
        e2e / "step7_policy_ablation" / "policy_ablation_results.csv",
        [
            "policy",
            "benchmark_cases",
            "prefetch_cases",
            "chunks_scored",
            "success_rate",
            "ttft_ms_mean",
            "latency_ms_mean",
            "prefetch_hit_rate",
            "ttft_delta_pct_vs_baseline",
            "latency_delta_pct_vs_baseline",
        ],
        [
            {
                "policy": "no_prefetch",
                "benchmark_cases": 2,
                "prefetch_cases": 0,
                "chunks_scored": 0,
                "success_rate": 1,
                "ttft_ms_mean": 200,
                "latency_ms_mean": 1000,
            },
            {
                "policy": "astrakv_combined",
                "benchmark_cases": 1,
                "prefetch_cases": 1,
                "chunks_scored": 3,
                "success_rate": 1,
                "ttft_ms_mean": 150,
                "latency_ms_mean": 1100,
                "prefetch_hit_rate": 1,
                "ttft_delta_pct_vs_baseline": -25,
                "latency_delta_pct_vs_baseline": 10,
            },
        ],
    )
    write_rows(
        e2e / "step7_chunk_scores" / "chunk_scores.csv",
        ["chunk_id", "action", "score"],
        [{"chunk_id": "a", "action": "offload", "score": 0.1}],
    )
    write_jsonl(
        e2e / "cache_events" / "step4_lmcache_disk" / "cache_events.jsonl",
        [{"event_type": "cache_store", "status": "observed"}],
    )
    write_jsonl(
        e2e / "cache_events" / "step5_prefetch" / "cache_events.jsonl",
        [{"event_type": "cache_hit", "status": "observed"}],
    )
    (main / "07_final_report").mkdir(parents=True, exist_ok=True)
    (main / "07_final_report" / "competition_report.md").write_text("# report\n", encoding="utf-8")
    write_json(main / "04_os_vm" / "dgx_spark_vm" / "dgx_spark_vm_evidence_summary.json", {"summary": {"prefetch_requests": 1}})
    write_json(main / "04_os_vm" / "mmap_kv_cache" / "mmap_kv_demo_summary.json", {"summary": {"total_blocks": 8}})
    write_rows(main / "05_quality" / "lmcache_disk_vs_vllm" / "quality_results.csv", ["sample_id", "exact_match"], [{"sample_id": "a", "exact_match": 1}])

    write_rows(
        boundary_pass / "02_boundary_32k" / "stress_analysis" / "stress_summary.csv",
        [
            "run",
            "total_requests",
            "success_requests",
            "success_rate",
            "max_success_context",
            "max_success_batch",
            "process_rss_peak_mb",
            "disk_write_delta_mb",
            "worst_latency_p95_ms",
        ],
        [
            {"run": "vllm", "total_requests": 10, "success_requests": 10, "success_rate": 1, "max_success_context": 32768, "max_success_batch": 16, "process_rss_peak_mb": 3000, "disk_write_delta_mb": 10, "worst_latency_p95_ms": 1000},
            {"run": "lmcache_disk", "total_requests": 10, "success_requests": 10, "success_rate": 1, "max_success_context": 32768, "max_success_batch": 16, "process_rss_peak_mb": 7000, "disk_write_delta_mb": 19840, "worst_latency_p95_ms": 900},
        ],
    )
    write_jsonl(
        boundary_pass / "03_cache_events" / "lmcache_disk_boundary" / "cache_events.jsonl",
        [{"event_type": "cache_store", "status": "observed"}, {"event_type": "cache_hit", "status": "observed"}],
    )
    (boundary_pass / "02_boundary_32k").mkdir(parents=True, exist_ok=True)
    (boundary_pass / "02_boundary_32k" / "vllm_server.log").write_text("Available KV cache memory: 2.86 GiB\n", encoding="utf-8")
    (boundary_pass / "02_boundary_32k" / "lmcache_disk_server.log").write_text("Available KV cache memory: 2.88 GiB\n", encoding="utf-8")

    fail_log = (
        "Available KV cache memory: 1.67 GiB\n"
        "ValueError: To serve at least one request with the model's max seq len (32768), "
        "(1.75 GiB KV cache is needed, which is larger than the available KV cache memory (1.67 GiB). "
        "Based on the available memory, the estimated maximum model length is 31328.\n"
    )
    (boundary_fail / "02_boundary_32k").mkdir(parents=True, exist_ok=True)
    (boundary_fail / "02_boundary_32k" / "vllm_server.log").write_text(fail_log, encoding="utf-8")
    summary.write_text("# summary\n", encoding="utf-8")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
