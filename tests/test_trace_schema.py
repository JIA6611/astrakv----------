import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.runtime.trace_schema import (
    TRACE_SCHEMA_VERSION,
    load_jsonl,
    load_memory_samples,
    summarize_trace_events,
    trace_from_cache_record,
    trace_from_prefetch_record,
    validate_trace_record,
    write_trace_jsonl,
)
from scripts.policy.build_trace_store import build_events, expand_sample_paths


class TraceSchemaTests(unittest.TestCase):
    def test_cache_record_converts_to_unified_trace(self) -> None:
        trace = trace_from_cache_record(
            {
                "schema": "astra-cache-event-v1",
                "event_type": "cache_hit",
                "source": "server.log",
                "status": "ok",
                "request_id": "req-1",
                "tier": "cpu",
                "start_time": "2026-06-08 10:00:00",
                "metadata": {
                    "case": "bs1_ctx512_out64",
                    "backend": "lmcache_cpu",
                    "hit_tokens": 128,
                },
            }
        )

        record = trace.to_record()
        self.assertEqual(record["schema"], TRACE_SCHEMA_VERSION)
        self.assertEqual(record["event_type"], "cache_hit")
        self.assertEqual(record["category"], "kv")
        self.assertEqual(record["request_id"], "req-1")
        self.assertEqual(record["case"], "bs1_ctx512_out64")
        self.assertEqual(record["backend"], "lmcache_cpu")
        self.assertEqual(record["tier"], "cpu")
        self.assertEqual(validate_trace_record(record), [])

    def test_prefetch_record_converts_to_unified_trace(self) -> None:
        trace = trace_from_prefetch_record(
            {
                "schema": "astra-prefetch-event-v1",
                "event_type": "prefetch_completed",
                "case": "ctx1024_rep0",
                "mode": "astrakv_prefetch",
                "request_id": "pref-1",
                "chunk_id": "ctx1024_rep0",
                "target_tier": "gpu",
                "status": "completed",
                "metadata": {
                    "latency_ms": 123.0,
                    "metadata": {"backend": "lmcache_cpu"},
                },
            }
        )

        record = trace.to_record()
        self.assertEqual(record["category"], "prefetch")
        self.assertEqual(record["case"], "ctx1024_rep0")
        self.assertEqual(record["backend"], "lmcache_cpu")
        self.assertEqual(record["latency_ms"], 123.0)
        self.assertEqual(validate_trace_record(record), [])

    def test_memory_samples_convert_and_infer_case(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            sample_path = Path(raw_tmp) / "bs1_ctx512_out64_samples.csv"
            with sample_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp_s",
                        "cpu_rss_mb",
                        "gpu_used_mb",
                        "gpu_util_pct",
                        "disk_read_mb",
                        "disk_write_mb",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp_s": "1.0",
                        "cpu_rss_mb": "100",
                        "gpu_used_mb": "200",
                        "gpu_util_pct": "75",
                        "disk_read_mb": "10",
                        "disk_write_mb": "20",
                    }
                )

            events = load_memory_samples(sample_path)
            self.assertEqual(len(events), 1)
            record = events[0].to_record()
            self.assertEqual(record["event_type"], "memory_sample")
            self.assertEqual(record["category"], "memory")
            self.assertEqual(record["case"], "bs1_ctx512_out64")
            self.assertEqual(record["metadata"]["gpu_util_pct"], "75")
            self.assertEqual(validate_trace_record(record), [])

    def test_missing_inputs_become_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            missing_jsonl = tmp / "missing.jsonl"
            missing_samples = tmp / "missing_samples.csv"

            cache_trace = trace_from_cache_record(load_jsonl(missing_jsonl)[0])
            sample_trace = load_memory_samples(missing_samples)[0]

            self.assertEqual(cache_trace.category, "error")
            self.assertEqual(cache_trace.status, "missing")
            self.assertEqual(sample_trace.category, "error")
            self.assertEqual(sample_trace.status, "missing")
            self.assertEqual(validate_trace_record(cache_trace.to_record()), [])
            self.assertEqual(validate_trace_record(sample_trace.to_record()), [])

    def test_trace_store_builds_from_all_p0_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cache_path = tmp / "cache_events.jsonl"
            prefetch_path = tmp / "prefetch_events.jsonl"
            samples_dir = tmp / "samples"
            samples_dir.mkdir()
            sample_path = samples_dir / "bs1_ctx512_out64_samples.csv"

            cache_path.write_text(
                json.dumps(
                    {
                        "schema": "astra-cache-event-v1",
                        "event_type": "cache_miss",
                        "source": "server.log",
                        "status": "ok",
                        "request_id": "req-1",
                        "tier": "cpu",
                        "metadata": {"case": "bs1_ctx512_out64"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prefetch_path.write_text(
                json.dumps(
                    {
                        "schema": "astra-prefetch-event-v1",
                        "event_type": "prefetch_hit",
                        "case": "ctx512_rep0",
                        "request_id": "pref-1",
                        "chunk_id": "ctx512_rep0",
                        "target_tier": "gpu",
                        "status": "observed",
                        "metadata": {"hit_evidence": "latency_improvement_heuristic"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with sample_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp_s",
                        "cpu_rss_mb",
                        "gpu_used_mb",
                        "gpu_util_pct",
                        "disk_read_mb",
                        "disk_write_mb",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp_s": "1.0",
                        "cpu_rss_mb": "100",
                        "gpu_used_mb": "200",
                        "gpu_util_pct": "75",
                        "disk_read_mb": "10",
                        "disk_write_mb": "20",
                    }
                )

            args = Namespace(
                cache_events=[str(cache_path)],
                prefetch_events=[str(prefetch_path)],
                samples=[str(samples_dir)],
            )
            events = build_events(args)
            self.assertEqual(len(events), 3)
            summary = summarize_trace_events(events)
            self.assertEqual(summary["category_counts"]["kv"], 1)
            self.assertEqual(summary["category_counts"]["prefetch"], 1)
            self.assertEqual(summary["category_counts"]["memory"], 1)
            self.assertEqual(summary["prefetch_hit_rate"], 1.0)
            self.assertEqual(expand_sample_paths(samples_dir), [sample_path])

            trace_path = tmp / "trace_events.jsonl"
            write_trace_jsonl(events, trace_path)
            written = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(written), 3)
            self.assertTrue(all(item["schema"] == TRACE_SCHEMA_VERSION for item in written))


if __name__ == "__main__":
    unittest.main()
