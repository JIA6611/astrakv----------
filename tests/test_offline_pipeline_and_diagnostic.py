import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from astrakv.runtime.profile_db import ProfileDB
from scripts.benchmark import diagnose_runtime
from scripts.policy import run_offline_eviction_pipeline as pipeline


class OfflinePipelineAndDiagnosticTests(unittest.TestCase):
    def test_perf_permission_rejection_is_not_available(self) -> None:
        record = diagnose_runtime.tool_artifact_record(
            "perf",
            Path("/tmp/perf.txt"),
            SimpleNamespace(
                returncode=255,
                stderr="Access to performance monitoring and observability operations is limited.",
            ),
        )

        self.assertEqual(record["collection_status"], "not_available")
        self.assertIn("Access to performance", record["stderr_summary"])

    def test_three_workload_pipeline_writes_gate_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            workload = root / "workload.jsonl"
            workload.write_text(json.dumps({"request_id": "r", "prompt": "p", "prefix_id": "prefix", "arrival_index": 0, "reuse_ratio": 0.0, "reuse_bucket": "none"}) + "\n", encoding="utf-8")
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({"request_id": "r", "event_type": "cache_load", "bytes": 4096, "metadata": {"legacy_unlinked": False}}) + "\n", encoding="utf-8")
            profile = root / "profile.json"
            ProfileDB().save(profile)
            entries = [{"workload_id": name, "workload_manifest": str(workload), "trace": str(trace), "profile_db": str(profile), "gpu_capacity_bytes": 4096, "cpu_capacity_bytes": 4096, "ssd_capacity_bytes": 4096, "default_object_bytes": 4096, "profile_source": "separate_profiling_run"} for name in ("a", "b", "c")]
            config = root / "set.json"
            config.write_text(json.dumps({"schema": "astrakv-offline-workload-set-v1", "workloads": entries}), encoding="utf-8")
            out = root / "out"
            old = sys.argv
            try:
                sys.argv = ["pipeline", "--workload-set", str(config), "--run-id", "run", "--output-dir", str(out)]
                self.assertEqual(pipeline.main(), 0)
            finally:
                sys.argv = old
            self.assertEqual(json.loads((out / "offline_safety_gate.json").read_text(encoding="utf-8"))["status"], "accepted")
            self.assertEqual(json.loads((out / "offline_pipeline_status.json").read_text(encoding="utf-8"))["successful_manifest_count"], 3)

    def test_diagnostic_runner_writes_platform_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            out = Path(raw_tmp) / "diagnostic"
            old = sys.argv
            try:
                sys.argv = ["diagnose_runtime", "--duration-seconds", "0.1", "--output-dir", str(out)]
                self.assertEqual(diagnose_runtime.main(), 0)
            finally:
                sys.argv = old
            manifest = json.loads((out / "diagnostic_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("perf", manifest["capabilities"])
            self.assertIn("perf_event_paranoid", manifest["capabilities"]["perf"])
            self.assertTrue((out / "diagnostic_samples.csv").exists())


if __name__ == "__main__":
    unittest.main()
