import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.reporting.build_demo_dashboard import (
    build_dashboard_data,
    load_dashboard_artifacts,
    write_dashboard_data,
    write_dashboard_html,
    write_manifest,
)
from scripts.reporting.build_competition_report import summarize_artifacts


class DemoDashboardTests(unittest.TestCase):
    def test_dashboard_data_summarizes_benchmark_and_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            benchmark = tmp / "benchmark_results.csv"
            missing = tmp / "missing_quality.csv"
            write_rows(
                benchmark,
                [
                    "case",
                    "request_count",
                    "success_count",
                    "ttft_ms",
                    "tpot_ms",
                    "latency_p95_ms",
                    "gpu_memory_peak_mb",
                    "cpu_memory_peak_mb",
                ],
                [
                    {
                        "case": "a",
                        "request_count": 2,
                        "success_count": 1,
                        "ttft_ms": 100,
                        "tpot_ms": 10,
                        "latency_p95_ms": 500,
                        "gpu_memory_peak_mb": 1000,
                        "cpu_memory_peak_mb": 600,
                    }
                ],
            )
            args = SimpleNamespace(
                title="AstraKV-W Dashboard Test",
                command=["python run"],
                artifact=[f"missing_quality:quality={missing}"],
                benchmark=[f"vllm={benchmark}"],
                quality="",
                hidden_state="",
                vm_demo="",
                moe_trace="",
                moe_loading="",
                moe_prediction="",
                competition_report="",
            )

            artifacts = load_dashboard_artifacts(args)
            summaries = summarize_artifacts(artifacts)
            data = build_dashboard_data(args, artifacts, summaries)

            self.assertEqual(data["title"], "AstraKV-W Dashboard Test")
            self.assertEqual(len(data["benchmark_cards"]), 1)
            self.assertEqual(data["benchmark_cards"][0]["success_rate"], 0.5)
            self.assertEqual(len(data["missing"]), 1)
            self.assertEqual(data["missing"][0]["label"], "missing_quality")

    def test_dashboard_outputs_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            quality = tmp / "quality_results.csv"
            write_rows(
                quality,
                ["sample_id", "exact_match", "normalized_match", "token_divergence_rate"],
                [{"sample_id": "s1", "exact_match": 1, "normalized_match": 1, "token_divergence_rate": 0.0}],
            )
            args = SimpleNamespace(
                title="AstraKV-W Dashboard",
                command=[],
                artifact=[],
                benchmark=[],
                quality=str(quality),
                hidden_state="",
                vm_demo="",
                moe_trace="",
                moe_loading="",
                moe_prediction="",
                competition_report="",
            )
            artifacts = load_dashboard_artifacts(args)
            summaries = summarize_artifacts(artifacts)
            data = build_dashboard_data(args, artifacts, summaries)
            html_path = tmp / "dashboard.html"
            data_path = tmp / "dashboard_data.json"
            manifest_path = tmp / "dashboard_manifest.json"

            write_dashboard_data(data_path, data)
            write_dashboard_html(html_path, data)
            write_manifest(manifest_path, args, data, html_path, data_path)

            payload = json.loads(data_path.read_text(encoding="utf-8"))
            html_text = html_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema"], "astra-demo-dashboard-v1")
            self.assertIn("<!doctype html>", html_text)
            self.assertIn("Quality Evidence", html_text)
            self.assertEqual(manifest["schema"], "astra-demo-dashboard-manifest-v1")
            self.assertEqual(manifest["artifact_count"], 1)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
