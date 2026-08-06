import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from astrakv.scheduler.decision import LoadRecomputeAction, LoadRecomputeDecision
from astrakv.scheduler.object_scheduler import (
    ObjectScheduleAction,
    ObjectScheduleCandidate,
    ObjectSchedulerConfig,
    UnifiedObjectScheduler,
    candidates_from_profile_db,
    load_decision_from_record,
)
from scripts.policy.run_unified_object_scheduler import write_decisions_csv, write_hints_jsonl


class UnifiedObjectSchedulerTests(unittest.TestCase):
    def test_budget_keeps_high_priority_and_offloads_when_full(self) -> None:
        candidates = [
            ObjectScheduleCandidate(
                chunk_id="hot",
                workload_id="w",
                size_bytes=100,
                chunk_score=0.8,
                chunk_action="prefetch",
                load_action="load",
                reuse_frequency=0.9,
                cache_hit_rate=0.8,
                prefetch_hit_rate=0.8,
            ),
            ObjectScheduleCandidate(
                chunk_id="warm",
                workload_id="w",
                size_bytes=100,
                chunk_score=0.4,
                chunk_action="keep",
                load_action="load",
                reuse_frequency=0.3,
            ),
        ]
        scheduler = UnifiedObjectScheduler(ObjectSchedulerConfig(gpu_budget_bytes=100, default_object_bytes=100))

        decisions = scheduler.schedule(candidates)
        by_chunk = {decision.chunk_id: decision for decision in decisions}

        self.assertEqual(by_chunk["hot"].action, ObjectScheduleAction.PREFETCH)
        self.assertEqual(by_chunk["hot"].gpu_bytes_after, 100)
        self.assertEqual(by_chunk["warm"].action, ObjectScheduleAction.OFFLOAD)
        self.assertEqual(by_chunk["warm"].gpu_bytes_after, 100)

    def test_load_recompute_drop_and_defer_override_gpu_budget(self) -> None:
        candidates = [
            ObjectScheduleCandidate(chunk_id="r", workload_id="w", load_action="recompute", size_bytes=100),
            ObjectScheduleCandidate(chunk_id="d", workload_id="w", load_action="drop", size_bytes=100),
            ObjectScheduleCandidate(chunk_id="f", workload_id="w", load_action="defer", size_bytes=100),
        ]
        decisions = UnifiedObjectScheduler(ObjectSchedulerConfig(gpu_budget_bytes=300)).schedule(candidates)
        by_chunk = {decision.chunk_id: decision for decision in decisions}

        self.assertEqual(by_chunk["r"].action, ObjectScheduleAction.RECOMPUTE)
        self.assertEqual(by_chunk["d"].action, ObjectScheduleAction.DROP)
        self.assertEqual(by_chunk["f"].action, ObjectScheduleAction.DEFER)
        self.assertEqual(sum(item.gpu_bytes_after for item in decisions), 0)

    def test_low_priority_candidate_can_drop(self) -> None:
        decision = UnifiedObjectScheduler(
            ObjectSchedulerConfig(gpu_budget_bytes=0, drop_threshold=0.2)
        ).schedule(
            [
                ObjectScheduleCandidate(
                    chunk_id="cold",
                    workload_id="w",
                    size_bytes=100,
                    chunk_score=0.01,
                    reuse_frequency=0.0,
                )
            ]
        )[0]

        self.assertEqual(decision.action, ObjectScheduleAction.DROP)
        self.assertIn("drop threshold", decision.reason)

    def test_candidates_from_profile_db_merges_scores_and_load_decisions(self) -> None:
        db = ProfileDB()
        db.chunks["w:hot"] = ChunkProfile(
            chunk_id="hot",
            workload_id="w",
            case="case-hot",
            request_count=10,
            reuse_count=8,
            cache_hits=5,
            cache_misses=1,
            prefetch_hits=3,
            bytes_loaded=2048,
            load_latency_ms_total=30.0,
            load_latency_count=1,
            tier_counts={"cpu": 2, "gpu": 1},
        )
        load_decision = LoadRecomputeDecision(
            chunk_id="hot",
            workload_id="w",
            action=LoadRecomputeAction.LOAD,
            estimated_load_ms=15.0,
            estimated_recompute_ms=40.0,
            priority=77,
        )

        candidates = candidates_from_profile_db(
            db,
            chunk_scores={"hot": {"score": "0.75", "action": "prefetch"}},
            load_decisions={"hot": load_decision},
            default_size_bytes=1024,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.chunk_id, "hot")
        self.assertEqual(candidate.size_bytes, 2048)
        self.assertEqual(candidate.current_tier, "cpu")
        self.assertEqual(candidate.chunk_score, 0.75)
        self.assertEqual(candidate.chunk_action, "prefetch")
        self.assertEqual(candidate.load_action, "load")
        self.assertEqual(candidate.load_priority, 77)
        self.assertEqual(candidate.load_ms, 15.0)

    def test_load_decision_from_record_and_outputs_are_writable(self) -> None:
        decision = load_decision_from_record(
            {
                "chunk_id": "chunk",
                "workload_id": "w",
                "action": "load",
                "estimated_load_ms": "10.5",
                "estimated_recompute_ms": "20.0",
                "priority": "42",
            }
        )
        self.assertEqual(decision.action, LoadRecomputeAction.LOAD)
        self.assertEqual(decision.priority, 42)

        schedule_decision = UnifiedObjectScheduler(ObjectSchedulerConfig(gpu_budget_bytes=100)).schedule(
            [
                ObjectScheduleCandidate(
                    chunk_id="chunk",
                    workload_id="w",
                    chunk_score=0.8,
                    chunk_action="keep",
                    load_action="load",
                    size_bytes=50,
                )
            ]
        )[0]

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            decisions_path = tmp / "decisions.csv"
            hints_path = tmp / "hints.jsonl"
            write_decisions_csv(decisions_path, [schedule_decision])
            write_hints_jsonl(hints_path, [schedule_decision])

            with decisions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["chunk_id"], "chunk")
            self.assertEqual(rows[0]["action"], "keep")

            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(hints[0]["action"], "keep")
            self.assertEqual(hints[0]["metadata"]["chunk_id"], "chunk")


if __name__ == "__main__":
    unittest.main()
