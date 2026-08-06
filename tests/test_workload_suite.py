import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.workload_suite import (
    WORKLOAD_SCHEMA_VERSION,
    build_competition_workload_suite,
    summarize_workload_cases,
)
from scripts.benchmark.generate_workload_suite import write_manifest, write_report, write_suite_jsonl


class WorkloadSuiteTests(unittest.TestCase):
    def test_suite_contains_required_workload_types(self) -> None:
        cases = build_competition_workload_suite(
            long_context_tokens=1024,
            memory_pressure_tokens=2048,
            repeated_prefix_tokens=512,
        )
        workload_types = {case.workload_type for case in cases}

        self.assertIn("short_chat", workload_types)
        self.assertIn("long_context_qa", workload_types)
        self.assertIn("prefix_reuse", workload_types)
        self.assertIn("rag_repeated_prefix", workload_types)
        self.assertIn("memory_pressure", workload_types)
        self.assertGreaterEqual(len([case for case in cases if case.repeat_group]), 3)

    def test_records_are_json_serializable_and_summary_counts_types(self) -> None:
        cases = build_competition_workload_suite(
            long_context_tokens=1024,
            memory_pressure_tokens=2048,
            repeated_prefix_tokens=512,
        )
        records = [case.to_record() for case in cases]
        summary = summarize_workload_cases(cases)

        self.assertEqual(records[0]["schema"], WORKLOAD_SCHEMA_VERSION)
        self.assertEqual(summary["case_count"], len(cases))
        self.assertEqual(summary["type_counts"]["prefix_reuse"], 2)
        self.assertEqual(summary["max_context_length"], 2048)
        json.dumps(records)

    def test_writer_outputs_suite_manifest_and_report(self) -> None:
        cases = build_competition_workload_suite(
            long_context_tokens=512,
            memory_pressure_tokens=1024,
            repeated_prefix_tokens=256,
        )
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            suite_path = tmp / "suite.jsonl"
            manifest_path = tmp / "manifest.json"
            report_path = tmp / "report.md"
            args = type(
                "Args",
                (),
                {
                    "long_context_tokens": 512,
                    "memory_pressure_tokens": 1024,
                    "repeated_prefix_tokens": 256,
                },
            )()

            write_suite_jsonl(suite_path, cases)
            write_manifest(manifest_path, args, cases, suite_path)
            write_report(report_path, args, cases, suite_path, manifest_path)

            rows = [json.loads(line) for line in suite_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")

            self.assertEqual(len(rows), len(cases))
            self.assertEqual(manifest["schema"], WORKLOAD_SCHEMA_VERSION)
            self.assertIn("# Competition Workload Suite Report", report)
            self.assertIn("memory_pressure", report)


if __name__ == "__main__":
    unittest.main()
