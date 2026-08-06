import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.runtime.memory_pressure import (
    MemoryPressureAction,
    MemoryPressureConfig,
    MemoryPressureController,
    MemoryPressureLevel,
    observations_from_benchmark_csv,
    observations_from_sample_csv,
    observations_from_trace_jsonl,
    summarize_decisions,
    write_decisions_csv,
    write_pressure_hints_jsonl,
)
from scripts.reporting.analyze_memory_pressure import write_manifest, write_report


class MemoryPressureControllerTests(unittest.TestCase):
    def test_benchmark_oom_maps_to_critical_reduce_workload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            with benchmark.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case",
                        "context_length",
                        "batch_size",
                        "request_count",
                        "success_count",
                        "status",
                        "errors",
                        "gpu_memory_peak_mb",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case": "ctx8192_bs4",
                        "context_length": 8192,
                        "batch_size": 4,
                        "request_count": 5,
                        "success_count": 0,
                        "status": "failed",
                        "errors": "CUDA out of memory",
                        "gpu_memory_peak_mb": 76000,
                    }
                )

            observations = observations_from_benchmark_csv(benchmark, run_id="vllm")
            decision = MemoryPressureController(
                MemoryPressureConfig(gpu_capacity_mb=80000)
            ).assess(observations[0])

            self.assertEqual(decision.level, MemoryPressureLevel.CRITICAL)
            self.assertEqual(decision.score, 1.0)
            self.assertEqual(decision.primary_action, MemoryPressureAction.REDUCE_BATCH_OR_CONTEXT)
            self.assertIn(MemoryPressureAction.OFFLOAD_MORE, decision.actions)

    def test_sample_csv_uses_capacity_ratio_and_disk_delta(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            samples = tmp / "case_a_samples.csv"
            with samples.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "gpu_used_mb", "cpu_rss_mb", "gpu_util_pct", "disk_read_mb", "disk_write_mb"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "0",
                        "gpu_used_mb": "1000",
                        "cpu_rss_mb": "100",
                        "gpu_util_pct": "10",
                        "disk_read_mb": "10",
                        "disk_write_mb": "5",
                    }
                )
                writer.writerow(
                    {
                        "timestamp": "1",
                        "gpu_used_mb": "9000",
                        "cpu_rss_mb": "200",
                        "gpu_util_pct": "90",
                        "disk_read_mb": "20",
                        "disk_write_mb": "8",
                    }
                )

            observation = observations_from_sample_csv(samples, run_id="sample-run")[0]
            decision = MemoryPressureController(
                MemoryPressureConfig(gpu_capacity_mb=10000, disk_high_mb=100)
            ).assess(observation)

            self.assertEqual(observation.case, "case_a")
            self.assertEqual(observation.sample_count, 2)
            self.assertEqual(observation.disk_read_delta_mb, 10.0)
            self.assertEqual(observation.disk_write_delta_mb, 3.0)
            self.assertEqual(decision.level, MemoryPressureLevel.HIGH)
            self.assertIn(MemoryPressureAction.DROP_LOW_REUSE, decision.actions)

    def test_trace_memory_events_are_grouped_by_case(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            trace = tmp / "trace_events.jsonl"
            records = [
                {
                    "event_type": "memory_sample",
                    "category": "memory",
                    "case": "case-t",
                    "metadata": {"gpu_used_mb": "10", "cpu_rss_mb": "20", "disk_read_mb": "1"},
                },
                {
                    "event_type": "memory_sample",
                    "category": "memory",
                    "case": "case-t",
                    "metadata": {"gpu_used_mb": "30", "cpu_rss_mb": "40", "disk_read_mb": "9"},
                },
            ]
            trace.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            observation = observations_from_trace_jsonl(trace)[0]

            self.assertEqual(observation.case, "case-t")
            self.assertEqual(observation.gpu_memory_peak_mb, 30.0)
            self.assertEqual(observation.cpu_memory_peak_mb, 40.0)
            self.assertEqual(observation.disk_read_delta_mb, 8.0)
            self.assertEqual(observation.sample_count, 2)

    def test_missing_artifact_collects_evidence(self) -> None:
        missing = Path("does_not_exist.csv")

        observation = observations_from_benchmark_csv(missing)[0]
        decision = MemoryPressureController().assess(observation)

        self.assertTrue(observation.missing)
        self.assertEqual(decision.level, MemoryPressureLevel.UNKNOWN)
        self.assertEqual(decision.primary_action, MemoryPressureAction.COLLECT_EVIDENCE)

    def test_outputs_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            with benchmark.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["case", "request_count", "success_count", "gpu_memory_peak_mb"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case": "case-out",
                        "request_count": 10,
                        "success_count": 10,
                        "gpu_memory_peak_mb": 1000,
                    }
                )
            observations = observations_from_benchmark_csv(benchmark)
            decisions = MemoryPressureController(MemoryPressureConfig(gpu_capacity_mb=10000)).assess_many(observations)
            decisions_path = tmp / "memory_pressure_decisions.csv"
            hints_path = tmp / "memory_pressure_hints.jsonl"
            report_path = tmp / "memory_pressure_report.md"
            manifest_path = tmp / "memory_pressure_manifest.json"

            write_decisions_csv(decisions_path, decisions)
            write_pressure_hints_jsonl(hints_path, decisions)
            args = Namespace(
                benchmark_results=[str(benchmark)],
                samples=[],
                trace_events=[],
                run_id="",
                gpu_capacity_mb=10000,
                cpu_capacity_mb=0,
                gpu_medium_ratio=0.70,
                gpu_high_ratio=0.85,
                gpu_critical_ratio=0.95,
                cpu_medium_ratio=0.70,
                cpu_high_ratio=0.85,
                cpu_critical_ratio=0.95,
                disk_medium_mb=512.0,
                disk_high_mb=4096.0,
                disk_critical_mb=16384.0,
                error_medium_rate=0.01,
                error_high_rate=0.05,
                error_critical_rate=0.20,
            )
            write_report(report_path, args, decisions, decisions_path, hints_path)
            write_manifest(manifest_path, args, decisions, decisions_path, hints_path, report_path)

            with decisions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(rows[0]["level"], "low")
            self.assertEqual(hints[0]["metadata"]["object_type"], "memory_pressure")
            self.assertEqual(manifest["summary"], summarize_decisions(decisions))
            self.assertIn("# Memory Pressure Controller Report", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
