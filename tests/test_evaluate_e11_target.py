from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.evaluate_e11_target import crash_check, evaluate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluateE11TargetTests(unittest.TestCase):
    def make_run(self, root: Path, arm: str, ttfts: list[float]) -> Path:
        run = root / arm / "rep-1" / "qasper" / "baseline"
        run.mkdir(parents=True)
        rows = [
            {
                "request_id": f"runtime-{arm}-{index}",
                "sample_id": f"sample-{index}",
                "status": "ok",
                "ttft_ms": value,
            }
            for index, value in enumerate(ttfts)
        ]
        write_jsonl(run / "request_results.jsonl", rows)
        write_jsonl(run / "workload_source.jsonl", [{"sample_id": f"sample-{i}"} for i in range(len(rows))])
        (root / arm / "arm_metrics.json").write_text("{}\n", encoding="utf-8")
        return run

    def test_role_isolation_deduplicates_exported_and_state_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            evict_run = self.make_run(root, "arm-evict-b", [50.0 + i for i in range(20)])
            self.make_run(root, "arm-lru", [100.0 + i for i in range(20)])
            receipt = {
                "receipt_id": "receipt-1",
                "command_id": "command-1",
                "action": "evict",
                "status": "completed",
                "object_key": "object-1",
                "tier_before": "cpu",
                "tier_after": "absent",
                "timestamp_ns": 100,
            }
            write_jsonl(evict_run / "runtime_command_receipts.jsonl", [receipt])
            # This is the same state artifact copied by the ablation wrapper.
            write_jsonl(
                evict_run.parent / "baseline-state" / "runtime_command_receipts.jsonl",
                [receipt],
            )
            write_jsonl(evict_run / "runtime_events_raw.jsonl", [])

            result = evaluate(root, "baseline")

            self.assertTrue(result["pairing"]["eligible"])
            self.assertEqual(result["evict"]["arm-evict-b"]["evict_completed"], 1)
            self.assertEqual(result["ttft"]["cell_count"], 1)
            self.assertLess(result["ttft"]["p95_delta_ci_upper_percent"], 0.0)
            self.assertEqual(result["conclusion"]["tier"], "① 真机优于 LRU（可声称）")

    def test_request_key_mismatch_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            evict_run = self.make_run(root, "arm-evict-b", [50.0, 60.0])
            lru_run = self.make_run(root, "arm-lru", [100.0, 110.0])
            rows = [
                {"sample_id": "different", "status": "ok", "ttft_ms": 100.0},
                {"sample_id": "sample-1", "status": "ok", "ttft_ms": 110.0},
            ]
            write_jsonl(lru_run / "request_results.jsonl", rows)
            write_jsonl(
                evict_run / "runtime_command_receipts.jsonl",
                [{"receipt_id": "r1", "action": "evict", "status": "completed", "object_key": "o1"}],
            )

            result = evaluate(root, "baseline")

            self.assertFalse(result["pairing"]["eligible"])
            self.assertIn("request_pair_mismatch:rep-1/qasper", result["pairing"]["errors"])
            self.assertEqual(result["conclusion"]["tier"], "③ inconclusive")

    def test_shutdown_induced_engine_dead_is_not_a_runtime_crash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            run = self.make_run(root, "arm-evict-b", [10.0])
            log = run.parent / "baseline-server.log"
            log.write_text(
                "[shutdown] EngineCore: trigger received signal=SIGTERM\n"
                "[shutdown] EngineCore: start mode=abort timeout=0s\n"
                "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue\n",
                encoding="utf-8",
            )
            self.assertEqual(crash_check(root, "baseline"), [])

            log.write_text(
                "request processing failed\n"
                "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue\n",
                encoding="utf-8",
            )
            self.assertEqual(len(crash_check(root, "baseline")), 1)


if __name__ == "__main__":
    unittest.main()
