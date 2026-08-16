import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.reporting import aggregate_evict_ablation


class AggregateEvictAblationTests(unittest.TestCase):
    def test_arm_aggregation_ignores_nested_warmup_state_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            arm_root = Path(raw_tmp) / "arm-evict-b"
            cell = arm_root / "rep-1" / "qasper"
            state_dir = cell / "baseline-state"
            run_dir = cell / "baseline"
            (state_dir / "warmup-state").mkdir(parents=True)
            run_dir.mkdir(parents=True)
            (state_dir / "runtime_command_receipts.jsonl").write_text(
                json.dumps({
                    "action": "evict",
                    "status": "completed",
                    "object_key": "prefix-a",
                    "timestamp_ns": 1,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "request_results.jsonl").write_text(
                json.dumps({"status": "ok", "ttft_ms": 12.5}) + "\n",
                encoding="utf-8",
            )
            output = arm_root / "arm_metrics.json"

            with patch.object(sys, "argv", [
                "aggregate_evict_ablation.py",
                "--arm-root", str(arm_root),
                "--output", str(output),
            ]):
                aggregate_evict_ablation.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["merged"]["run_count"], 1)
            self.assertEqual(payload["merged"]["evict_completed"], 1)
            self.assertEqual(payload["merged"]["ttft_ms"]["count"], 1)
            self.assertEqual(len(payload["runs"]), 1)
            self.assertEqual(Path(payload["runs"][0]["state_dir"]), state_dir)

    def test_e11_runner_requires_arm_aggregation_to_succeed(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "entrypoints" / "run_evict_b_vs_lru_suite.sh"
        ).read_text(encoding="utf-8")
        aggregate_block = runner.split(
            '"$PYTHON" scripts/reporting/aggregate_evict_ablation.py',
            1,
        )[1].split("done", 1)[0]
        self.assertNotIn("|| true", aggregate_block)
        self.assertNotIn("2>/dev/null", aggregate_block)


if __name__ == "__main__":
    unittest.main()
