import csv
import tempfile
import unittest
from pathlib import Path

from scripts.policy.analyze_policy_ablation import (
    PolicyInput,
    summarize_policies,
    write_csv,
    write_report,
)


class PolicyAblationTests(unittest.TestCase):
    def test_summarize_policies_aggregates_benchmark_prefetch_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline = tmp / "baseline" / "benchmark_results.csv"
            combined_benchmark = tmp / "combined" / "benchmark_results.csv"
            prefetch = tmp / "combined" / "prefetch_results.csv"
            scores = tmp / "combined" / "chunk_scores.csv"
            baseline.parent.mkdir()
            combined_benchmark.parent.mkdir()

            write_rows(
                baseline,
                [
                    "case",
                    "request_count",
                    "success_count",
                    "ttft_ms",
                    "tpot_ms",
                    "latency_ms",
                    "latency_p95_ms",
                    "throughput_tokens_s",
                    "process_rss_peak_mb",
                    "gpu_memory_peak_mb",
                    "cpu_memory_peak_mb",
                    "disk_read_delta_mb",
                    "disk_write_delta_mb",
                ],
                [
                    {
                        "case": "ctx512",
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_ms": 900,
                        "latency_p95_ms": 950,
                        "throughput_tokens_s": 40,
                        "process_rss_peak_mb": 500,
                        "gpu_memory_peak_mb": 1000,
                        "cpu_memory_peak_mb": 500,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                    }
                ],
            )
            write_rows(
                combined_benchmark,
                baseline.read_text(encoding="utf-8").splitlines()[0].split(","),
                [
                    {
                        "case": "ctx512",
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 80,
                        "tpot_ms": 11,
                        "latency_ms": 850,
                        "latency_p95_ms": 900,
                        "throughput_tokens_s": 44,
                        "process_rss_peak_mb": 800,
                        "gpu_memory_peak_mb": 700,
                        "cpu_memory_peak_mb": 800,
                        "disk_read_delta_mb": 12,
                        "disk_write_delta_mb": 16,
                    }
                ],
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
                ],
                [
                    {
                        "case": "ctx512",
                        "prefetch_submitted": 2,
                        "prefetch_completed": 2,
                        "prefetch_failed": 0,
                        "prefetch_hit": 1,
                        "prefetch_waste": 1,
                    }
                ],
            )
            write_rows(
                scores,
                ["chunk_id", "action", "score"],
                [
                    {"chunk_id": "a", "action": "prefetch", "score": 0.8},
                    {"chunk_id": "b", "action": "offload", "score": 0.3},
                    {"chunk_id": "c", "action": "drop", "score": 0.1},
                ],
            )

            rows = summarize_policies(
                [
                    PolicyInput("no_prefetch", kind="no_prefetch", benchmark_path=baseline),
                    PolicyInput(
                        "astrakv_combined",
                        kind="astrakv_combined",
                        benchmark_path=combined_benchmark,
                        prefetch_path=prefetch,
                        chunk_scores_path=scores,
                    ),
                ]
            )

            combined = next(row for row in rows if row["policy"] == "astrakv_combined")
            self.assertEqual(combined["benchmark_cases"], 1)
            self.assertEqual(combined["success_rate"], 1.0)
            self.assertEqual(combined["prefetch_hit_rate"], 0.5)
            self.assertEqual(combined["prefetch_waste_rate"], 0.5)
            self.assertEqual(combined["chunk_action_prefetch"], 1)
            self.assertEqual(combined["chunk_action_offload"], 1)
            self.assertEqual(combined["chunk_action_drop"], 1)
            self.assertEqual(combined["process_rss_peak_mb"], 800.0)
            self.assertEqual(combined["process_rss_delta_pct_vs_baseline"], 60.0)
            self.assertEqual(combined["gpu_memory_reduction_pct_vs_baseline"], 30.0)
            self.assertEqual(combined["throughput_delta_pct_vs_baseline"], 10.0)

    def test_report_marks_missing_metric_groups(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            rows = summarize_policies([PolicyInput("reuse_aware", kind="reuse_aware")])
            self.assertEqual(
                rows[0]["missing_metric_groups"],
                "missing_benchmark;missing_prefetch;missing_chunk_scores",
            )

            csv_path = tmp / "policy_ablation_results.csv"
            report_path = tmp / "policy_ablation_report.md"
            write_csv(csv_path, rows)
            write_report(report_path, [PolicyInput("reuse_aware", kind="reuse_aware")], rows)

            self.assertTrue(csv_path.exists())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# Policy Ablation Report", report)
            self.assertIn("RSS MB", report)
            self.assertIn("missing_benchmark;missing_prefetch;missing_chunk_scores", report)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
