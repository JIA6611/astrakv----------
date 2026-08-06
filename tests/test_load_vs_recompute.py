import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from astrakv.scheduler.decision import (
    LoadRecomputeAction,
    LoadRecomputeConfig,
    LoadRecomputePlanner,
    PartialPlanStats,
    partial_plan_stats_from_records,
)
from scripts.policy.decide_load_vs_recompute import load_partial_plan, write_decisions_csv, write_hints_jsonl


class LoadVsRecomputeTests(unittest.TestCase):
    def test_cache_friendly_chunk_prefers_load(self) -> None:
        profile = ChunkProfile(
            chunk_id="hot",
            workload_id="w",
            case="req-1",
            request_count=10,
            reuse_count=8,
            cache_hits=6,
            cache_misses=1,
            bytes_loaded=1000,
            load_latency_ms_total=20.0,
            load_latency_count=1,
        )
        planner = LoadRecomputePlanner(LoadRecomputeConfig(memory_pressure=0.1, default_tokens=100))

        decision = planner.decide_profile(profile)

        self.assertEqual(decision.action, LoadRecomputeAction.LOAD)
        self.assertIn("load:", decision.reason)
        self.assertGreater(decision.priority, 0)
        self.assertEqual(decision.to_hint().action, "load")

    def test_expensive_io_under_pressure_prefers_recompute(self) -> None:
        profile = ChunkProfile(
            chunk_id="warm",
            workload_id="w",
            request_count=10,
            reuse_count=4,
            cache_hits=1,
            cache_misses=4,
            bytes_loaded=1000,
            load_latency_ms_total=200.0,
            load_latency_count=1,
        )
        planner = LoadRecomputePlanner(
            LoadRecomputeConfig(
                memory_pressure=0.8,
                deadline_ms=300.0,
                recompute_latency_per_token_ms=0.05,
                recompute_overhead_ms=5.0,
                default_tokens=100,
            )
        )

        decision = planner.decide_profile(profile)

        self.assertEqual(decision.action, LoadRecomputeAction.RECOMPUTE)
        self.assertIn("cheaper than IO", decision.reason)

    def test_both_paths_over_deadline_defer(self) -> None:
        profile = ChunkProfile(
            chunk_id="slow",
            workload_id="w",
            request_count=10,
            reuse_count=5,
            bytes_loaded=1000,
            load_latency_ms_total=300.0,
            load_latency_count=1,
        )
        planner = LoadRecomputePlanner(
            LoadRecomputeConfig(
                memory_pressure=0.2,
                deadline_ms=50.0,
                recompute_latency_per_token_ms=1.0,
                recompute_overhead_ms=10.0,
                default_tokens=100,
            )
        )

        decision = planner.decide_profile(profile)

        self.assertEqual(decision.action, LoadRecomputeAction.DEFER)
        self.assertIn("exceed deadline", decision.reason)

    def test_low_reuse_high_pressure_drops(self) -> None:
        profile = ChunkProfile(chunk_id="cold", workload_id="w", request_count=100, reuse_count=0)
        planner = LoadRecomputePlanner(LoadRecomputeConfig(memory_pressure=0.9))

        decision = planner.decide_profile(profile)

        self.assertEqual(decision.action, LoadRecomputeAction.DROP)
        self.assertIn("low reuse", decision.reason)

    def test_partial_plan_skip_drops_and_partial_load_reduces_io(self) -> None:
        skipped = ChunkProfile(chunk_id="skipped", workload_id="w", request_count=10, reuse_count=5)
        partial = ChunkProfile(
            chunk_id="partial",
            workload_id="w",
            request_count=10,
            reuse_count=5,
            bytes_loaded=1000,
            load_latency_ms_total=100.0,
            load_latency_count=1,
        )
        planner = LoadRecomputePlanner(LoadRecomputeConfig(memory_pressure=0.3, deadline_ms=80.0))

        drop_decision = planner.decide_profile(
            skipped,
            PartialPlanStats(chunk_id="skipped", action="skip", skipped_bytes=1000, skipped_tokens=100),
        )
        load_decision = planner.decide_profile(
            partial,
            PartialPlanStats(
                chunk_id="partial",
                action="load_partial",
                loaded_bytes=500,
                skipped_bytes=500,
                loaded_tokens=50,
                skipped_tokens=50,
            ),
        )

        self.assertEqual(drop_decision.action, LoadRecomputeAction.DROP)
        self.assertEqual(load_decision.action, LoadRecomputeAction.LOAD)
        self.assertEqual(load_decision.estimated_load_ms, 50.0)
        self.assertEqual(load_decision.estimated_skipped_bytes, 500)

    def test_decide_db_orders_and_outputs_are_writable(self) -> None:
        db = ProfileDB()
        db.chunks["w:hot"] = ChunkProfile(
            chunk_id="hot",
            workload_id="w",
            request_count=10,
            reuse_count=8,
            cache_hits=5,
            cache_misses=1,
            bytes_loaded=1000,
            load_latency_ms_total=10.0,
            load_latency_count=1,
        )
        db.chunks["w:cold"] = ChunkProfile(chunk_id="cold", workload_id="w", request_count=10, reuse_count=0)
        decisions = LoadRecomputePlanner(LoadRecomputeConfig(memory_pressure=0.9)).decide_db(db)

        self.assertEqual(decisions[0].chunk_id, "hot")

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            decisions_path = tmp / "decisions.csv"
            hints_path = tmp / "hints.jsonl"
            write_decisions_csv(decisions_path, decisions)
            write_hints_jsonl(hints_path, decisions)

            with decisions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn(rows[0]["action"], {item.value for item in LoadRecomputeAction})

            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(hints), 2)
            self.assertIn("chunk_id", hints[0]["metadata"])

    def test_partial_plan_jsonl_loader(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "partial_kv_plan.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "chunk_id": "chunk",
                        "loaded_tokens": 5,
                        "skipped_tokens": 5,
                        "loaded_bytes": 50,
                        "skipped_bytes": 50,
                        "action": "load_partial",
                        "reason": "partial",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stats = load_partial_plan(path)
            direct = partial_plan_stats_from_records([json.loads(path.read_text(encoding="utf-8"))])

            self.assertEqual(stats["chunk"].loaded_bytes, 50)
            self.assertEqual(direct["chunk"].byte_saving_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
