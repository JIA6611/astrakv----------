import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.multi_model import (
    compare_runs,
    parse_run_spec,
    summarize_runs,
    write_csv,
    write_manifest,
)
from scripts.reporting.analyze_multi_model_evaluation import write_report


class MultiModelEvaluationTests(unittest.TestCase):
    def test_parse_run_spec(self) -> None:
        item = parse_run_spec("qwen:dense:vllm=results/qwen", order=3)

        self.assertEqual(item.model_id, "qwen")
        self.assertEqual(item.model_family, "dense")
        self.assertEqual(item.backend, "vllm")
        self.assertEqual(item.csv_path, Path("results/qwen") / "benchmark_results.csv")
        self.assertEqual(item.order, 3)

    def test_summary_and_comparison_compute_per_model_baseline_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline = tmp / "qwen_vllm" / "benchmark_results.csv"
            variant = tmp / "qwen_lmcache" / "benchmark_results.csv"
            moe = tmp / "moe_vllm" / "benchmark_results.csv"

            write_benchmark(
                baseline,
                [
                    {
                        "case": "bs1_ctx512_out64",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_ms": 800,
                        "latency_p95_ms": 900,
                        "throughput_tokens_s": 40,
                        "gpu_memory_peak_mb": 1000,
                        "cpu_memory_peak_mb": 500,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                        "status": "ok",
                    }
                ],
            )
            write_benchmark(
                variant,
                [
                    {
                        "case": "bs1_ctx512_out64",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 2,
                        "success_count": 2,
                        "ttft_ms": 110,
                        "tpot_ms": 11,
                        "latency_ms": 850,
                        "latency_p95_ms": 950,
                        "throughput_tokens_s": 38,
                        "gpu_memory_peak_mb": 700,
                        "cpu_memory_peak_mb": 800,
                        "disk_read_delta_mb": 12,
                        "disk_write_delta_mb": 16,
                        "status": "ok",
                    }
                ],
            )
            write_benchmark(
                moe,
                [
                    {
                        "case": "bs1_ctx512_out64",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 1,
                        "success_count": 1,
                        "ttft_ms": 200,
                        "tpot_ms": 20,
                        "latency_ms": 1000,
                        "latency_p95_ms": 1200,
                        "throughput_tokens_s": 25,
                        "gpu_memory_peak_mb": 2000,
                        "cpu_memory_peak_mb": 900,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                        "status": "ok",
                    }
                ],
            )

            inputs = [
                parse_run_spec(f"qwen:dense:vllm={baseline.parent}", order=0),
                parse_run_spec(f"qwen:dense:lmcache_cpu={variant.parent}", order=1),
                parse_run_spec(f"mixtral:moe:vllm={moe.parent}", order=2),
            ]
            summary = summarize_runs(inputs)
            comparison = compare_runs(inputs)

            qwen_variant_summary = next(row for row in summary if row["backend"] == "lmcache_cpu")
            qwen_variant_comparison = next(row for row in comparison if row["model_id"] == "qwen" and row["backend"] == "lmcache_cpu")
            moe_comparison = next(row for row in comparison if row["model_id"] == "mixtral")

            self.assertEqual(qwen_variant_summary["success_rate"], 1.0)
            self.assertEqual(qwen_variant_summary["max_context_length"], 512)
            self.assertEqual(qwen_variant_comparison["gpu_memory_reduction_pct_vs_baseline"], 30.0)
            self.assertIn("less_gpu", qwen_variant_comparison["help_hurt_summary"])
            self.assertEqual(moe_comparison["baseline_backend"], "vllm")
            self.assertEqual(moe_comparison["help_hurt_summary"], "neutral_or_baseline")

    def test_outputs_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "run" / "benchmark_results.csv"
            write_benchmark(
                benchmark,
                [
                    {
                        "case": "a",
                        "batch_size": 1,
                        "context_length": 512,
                        "output_tokens": 64,
                        "request_count": 1,
                        "success_count": 1,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_ms": 800,
                        "latency_p95_ms": 900,
                        "throughput_tokens_s": 40,
                        "gpu_memory_peak_mb": 1000,
                        "cpu_memory_peak_mb": 500,
                        "disk_read_delta_mb": 0,
                        "disk_write_delta_mb": 0,
                        "status": "ok",
                    }
                ],
            )
            inputs = [parse_run_spec(f"qwen:dense:vllm={benchmark.parent}", order=0)]
            summary_rows = summarize_runs(inputs)
            comparison_rows = compare_runs(inputs)
            summary_path = tmp / "multi_model_summary.csv"
            comparison_path = tmp / "multi_model_comparison.csv"
            report_path = tmp / "multi_model_report.md"
            manifest_path = tmp / "multi_model_manifest.json"

            write_csv(summary_path, summary_rows)
            write_csv(comparison_path, comparison_rows)
            write_report(report_path, inputs, summary_rows, comparison_rows, summary_path, comparison_path)
            write_manifest(manifest_path, inputs, summary_rows, comparison_rows)

            self.assertTrue(summary_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertIn("# Multi-Model Evaluation Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_ids"], ["qwen"])
            self.assertEqual(manifest["summary_count"], 1)


def write_benchmark(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
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
        "gpu_memory_peak_mb",
        "cpu_memory_peak_mb",
        "disk_read_delta_mb",
        "disk_write_delta_mb",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
