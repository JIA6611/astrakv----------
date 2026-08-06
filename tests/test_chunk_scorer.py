import csv
import tempfile
import unittest
from pathlib import Path

from astrakv.prefetch.scorer import ChunkAction, ChunkScorer, ChunkScorerConfig
from astrakv.runtime.profile_db import ChunkProfile, ProfileDB
from scripts.policy.score_chunks import write_csv


class ChunkScorerTests(unittest.TestCase):
    def test_high_reuse_expensive_chunk_is_prefetched(self) -> None:
        profile = ChunkProfile(
            chunk_id="hot",
            workload_id="w",
            request_count=10,
            reuse_count=8,
            cache_hits=6,
            cache_misses=2,
            cache_loads=3,
            prefetch_hits=4,
            prefetch_waste=0,
            load_latency_ms_total=300.0,
            load_latency_count=3,
            bytes_loaded=1024,
            tier_counts={"cpu": 3, "gpu": 4},
        )
        scorer = ChunkScorer(
            ChunkScorerConfig(
                prefetch_threshold=0.45,
                memory_pressure=0.0,
                load_latency_reference_ms=100.0,
            )
        )

        score = scorer.score_profile(profile)

        self.assertEqual(score.action, ChunkAction.PREFETCH)
        self.assertGreater(score.score, 0.45)
        self.assertIn("high reuse", score.reason)
        self.assertIn("expensive load", score.reason)

    def test_low_reuse_wasteful_chunk_is_dropped(self) -> None:
        profile = ChunkProfile(
            chunk_id="cold",
            workload_id="w",
            request_count=10,
            reuse_count=0,
            cache_hits=0,
            cache_misses=5,
            prefetch_hits=0,
            prefetch_waste=4,
            bytes_loaded=64 * 1024 * 1024,
            tier_counts={"gpu": 1},
        )
        scorer = ChunkScorer(
            ChunkScorerConfig(
                memory_pressure=0.8,
                prefetch_threshold=0.45,
                keep_threshold=0.25,
                offload_threshold=0.12,
            )
        )

        score = scorer.score_profile(profile)

        self.assertEqual(score.action, ChunkAction.DROP)
        self.assertLess(score.score, 0.12)
        self.assertIn("low reuse", score.reason)
        self.assertIn("prefetch waste observed", score.reason)

    def test_memory_pressure_can_offload_moderate_reuse_chunk(self) -> None:
        profile = ChunkProfile(
            chunk_id="warm",
            workload_id="w",
            request_count=10,
            reuse_count=2,
            cache_hits=1,
            cache_misses=3,
            cache_loads=1,
            load_latency_ms_total=30.0,
            load_latency_count=1,
            tier_counts={"gpu": 2},
        )
        scorer = ChunkScorer(
            ChunkScorerConfig(
                memory_pressure=0.9,
                prefetch_threshold=0.9,
                keep_threshold=0.8,
                offload_threshold=0.2,
            )
        )

        score = scorer.score_profile(profile)

        self.assertEqual(score.action, ChunkAction.OFFLOAD)
        self.assertIn("memory pressure", score.reason)

    def test_score_db_orders_by_score_and_csv_is_writable(self) -> None:
        db = ProfileDB()
        hot = ChunkProfile(
            chunk_id="hot",
            workload_id="w",
            request_count=10,
            reuse_count=9,
            cache_hits=5,
            cache_misses=1,
            prefetch_hits=2,
            load_latency_ms_total=100.0,
            load_latency_count=1,
        )
        cold = ChunkProfile(chunk_id="cold", workload_id="w", request_count=10)
        db.chunks["w:hot"] = hot
        db.chunks["w:cold"] = cold

        scores = ChunkScorer(ChunkScorerConfig(prefetch_threshold=0.4)).score_db(db)
        self.assertEqual(scores[0].chunk_id, "hot")

        with tempfile.TemporaryDirectory() as raw_tmp:
            csv_path = Path(raw_tmp) / "chunk_scores.csv"
            write_csv(csv_path, scores)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["chunk_id"], "hot")
            self.assertIn(rows[0]["action"], {item.value for item in ChunkAction})


if __name__ == "__main__":
    unittest.main()
