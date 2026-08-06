import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.profile_db import (
    LayerSensitivityRecord,
    ProfileDB,
    QualityGuardRecord,
    load_profile_from_trace_jsonl,
)
from astrakv.runtime.trace_schema import TraceEvent, write_trace_jsonl


class ProfileDBTests(unittest.TestCase):
    def test_profile_db_aggregates_chunk_and_workload_features(self) -> None:
        events = [
            TraceEvent(
                event_type="cache_hit",
                category="kv",
                source="trace",
                status="ok",
                request_id="req-1",
                case="case-a",
                backend="lmcache_cpu",
                model="model-a",
                chunk_id="chunk-a",
                tier="cpu",
            ),
            TraceEvent(
                event_type="cache_miss",
                category="kv",
                source="trace",
                status="ok",
                request_id="req-2",
                case="case-a",
                backend="lmcache_cpu",
                model="model-a",
                chunk_id="chunk-a",
                tier="cpu",
            ),
            TraceEvent(
                event_type="cache_load",
                category="kv",
                source="trace",
                status="completed",
                request_id="req-2",
                case="case-a",
                backend="lmcache_cpu",
                model="model-a",
                chunk_id="chunk-a",
                tier="cpu",
                bytes=2048,
                latency_ms=7.5,
            ),
            TraceEvent(
                event_type="prefetch_hit",
                category="prefetch",
                source="trace",
                status="observed",
                request_id="req-3",
                case="case-a",
                backend="lmcache_cpu",
                model="model-a",
                chunk_id="chunk-a",
                tier="gpu",
            ),
            TraceEvent(
                event_type="memory_sample",
                category="memory",
                source="sample.csv",
                status="observed",
                case="case-a",
                backend="lmcache_cpu",
                model="model-a",
                metadata={
                    "cpu_rss_mb": "1234",
                    "gpu_used_mb": "4321",
                    "gpu_util_pct": "88",
                    "disk_read_mb": "10",
                    "disk_write_mb": "20",
                },
            ),
        ]

        db = ProfileDB.from_trace_events(events, workload_id="workload-a")
        workload = db.workloads["workload-a"]
        chunk = db.get_chunk("chunk-a", workload_id="workload-a")

        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertEqual(workload.event_count, 5)
        self.assertEqual(workload.gpu_used_peak_mb, 4321.0)
        self.assertEqual(workload.cpu_rss_peak_mb, 1234.0)
        self.assertEqual(chunk.cache_hits, 1)
        self.assertEqual(chunk.cache_misses, 1)
        self.assertEqual(chunk.cache_loads, 1)
        self.assertEqual(chunk.bytes_loaded, 2048)
        self.assertEqual(chunk.avg_load_latency_ms, 7.5)
        self.assertEqual(chunk.prefetch_hits, 1)
        self.assertGreater(chunk.reuse_frequency, 0.0)
        self.assertEqual(chunk.cache_hit_rate, 0.5)

    def test_profile_db_persists_and_reloads(self) -> None:
        events = [
            TraceEvent(
                event_type="cache_hit",
                category="kv",
                source="trace",
                status="ok",
                request_id="req-1",
                case="case-a",
                chunk_id="chunk-a",
                tier="gpu",
            )
        ]
        db = ProfileDB.from_trace_events(events, workload_id="persisted")

        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "profile_db.json"
            db.save(path)
            loaded = ProfileDB.load(path)

            self.assertIn("persisted", loaded.workloads)
            self.assertIsNotNone(loaded.get_chunk("chunk-a", workload_id="persisted"))
            self.assertEqual(loaded.get_chunk("chunk-a", workload_id="persisted").cache_hits, 1)  # type: ignore[union-attr]

    def test_profile_db_loads_from_trace_jsonl(self) -> None:
        events = [
            TraceEvent(
                event_type="prefetch_waste",
                category="prefetch",
                source="trace",
                status="observed",
                request_id="req-1",
                case="case-a",
                chunk_id="chunk-a",
                tier="gpu",
            ),
            TraceEvent(
                event_type="prefetch_hit",
                category="prefetch",
                source="trace",
                status="observed",
                request_id="req-2",
                case="case-a",
                chunk_id="chunk-a",
                tier="gpu",
            ),
        ]

        with tempfile.TemporaryDirectory() as raw_tmp:
            trace_path = Path(raw_tmp) / "trace_events.jsonl"
            write_trace_jsonl(events, trace_path)
            db = load_profile_from_trace_jsonl(trace_path, workload_id="jsonl")
            chunk = db.get_chunk("chunk-a", workload_id="jsonl")

            self.assertIsNotNone(chunk)
            assert chunk is not None
            self.assertEqual(chunk.prefetch_hits, 1)
            self.assertEqual(chunk.prefetch_waste, 1)
            self.assertEqual(chunk.prefetch_hit_rate, 0.5)

    def test_profile_record_is_json_serializable(self) -> None:
        db = ProfileDB.from_trace_events(
            [
                TraceEvent(
                    event_type="memory_sample",
                    category="memory",
                    source="sample",
                    status="observed",
                    case="case-a",
                    metadata={"cpu_rss_mb": "1"},
                )
            ],
            workload_id="serializable",
        )
        encoded = json.dumps(db.to_record())
        self.assertIn("astra-profile-db-v1", encoded)

    def test_profile_db_persists_layer_sensitivity_and_quality_guards(self) -> None:
        db = ProfileDB()
        db.put_layer_sensitivity(
            LayerSensitivityRecord(
                workload_id="w",
                layer_id=4,
                sensitivity_score=0.92,
                partial_load_allowed=False,
                recompute_allowed=True,
                prefetch_priority_boost=0.25,
            )
        )
        db.put_quality_guard(
            QualityGuardRecord(
                workload_id="w",
                chunk_id="chunk-a",
                layer_id=4,
                quality_tier="strict",
                max_ppl_delta=0.05,
                min_cka=0.98,
                partial_load_allowed=False,
                recompute_allowed=False,
            )
        )

        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "profile_db.json"
            db.save(path)
            loaded = ProfileDB.load(path)

            sensitivity = loaded.get_layer_sensitivity(4, workload_id="w")
            guard = loaded.get_quality_guard(workload_id="w", chunk_id="chunk-a", layer_id=4)
            self.assertIsNotNone(sensitivity)
            self.assertIsNotNone(guard)
            assert sensitivity is not None and guard is not None
            self.assertFalse(sensitivity.partial_load_allowed)
            self.assertEqual(guard.quality_tier, "strict")
            self.assertFalse(guard.recompute_allowed)


if __name__ == "__main__":
    unittest.main()
