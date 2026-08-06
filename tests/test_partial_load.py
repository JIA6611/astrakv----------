import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.kv_cache.partial_load import (
    PartialKVLoadPlanner,
    PartialKVLoadRequest,
    PartialKVLoadTarget,
    PartialLoadAction,
    TokenSpan,
    build_partial_load_targets,
    chunk_meta_from_record,
)
from scripts.policy.plan_partial_kv_load import load_chunks, write_decisions_csv, write_summary_csv


class PartialKVLoadTests(unittest.TestCase):
    def test_planner_creates_full_partial_and_skip_decisions(self) -> None:
        chunks = [
            KVChunkMeta(
                request_id="req-1",
                layer_id=0,
                start_token=0,
                end_token=100,
                chunk_id="layer0",
                tier=MemoryTier.SSD,
                size_bytes=1000,
            ),
            KVChunkMeta(
                request_id="req-1",
                layer_id=1,
                start_token=0,
                end_token=100,
                chunk_id="layer1",
                tier=MemoryTier.CPU,
                size_bytes=2000,
            ),
            KVChunkMeta(
                request_id="req-1",
                layer_id=2,
                start_token=0,
                end_token=100,
                chunk_id="layer2",
                tier=MemoryTier.CPU,
                size_bytes=3000,
            ),
        ]
        request = PartialKVLoadRequest(
            request_id="req-1",
            target_layers=(0, 1),
            token_spans=(TokenSpan(0, 100), TokenSpan(40, 80)),
            target_tier=MemoryTier.GPU,
            plan_id="plan-1",
        )

        decisions = PartialKVLoadPlanner().plan(chunks, request)

        by_chunk = {decision.chunk_id: decision for decision in decisions}
        self.assertEqual(by_chunk["layer0"].action, PartialLoadAction.LOAD_FULL)
        self.assertEqual(by_chunk["layer0"].loaded_bytes, 1000)
        self.assertEqual(by_chunk["layer1"].action, PartialLoadAction.LOAD_FULL)
        self.assertEqual(by_chunk["layer2"].action, PartialLoadAction.SKIP)
        self.assertEqual(by_chunk["layer2"].skipped_bytes, 3000)

    def test_partial_span_estimates_loaded_and_skipped_bytes(self) -> None:
        chunk = KVChunkMeta(
            request_id="req-1",
            layer_id=0,
            start_token=0,
            end_token=100,
            chunk_id="chunk",
            tier=MemoryTier.SSD,
            size_bytes=1000,
        )
        request = PartialKVLoadRequest(
            request_id="req-1",
            token_spans=(TokenSpan(25, 75),),
            plan_id="plan-2",
        )

        decision = PartialKVLoadPlanner().plan([chunk], request)[0]

        self.assertEqual(decision.action, PartialLoadAction.LOAD_PARTIAL)
        self.assertEqual(decision.loaded_tokens, 50)
        self.assertEqual(decision.skipped_tokens, 50)
        self.assertEqual(decision.loaded_bytes, 500)
        self.assertEqual(decision.skipped_bytes, 500)
        self.assertEqual(decision.reason, "partial_token_span_selected")

    def test_summary_reports_byte_saving_rate(self) -> None:
        chunk = KVChunkMeta(
            request_id="req-1",
            layer_id=0,
            start_token=0,
            end_token=100,
            chunk_id="chunk",
            size_bytes=1000,
        )
        request = PartialKVLoadRequest(request_id="req-1", token_spans=(TokenSpan(0, 25),), plan_id="plan-3")
        planner = PartialKVLoadPlanner()
        decisions = planner.plan([chunk], request)
        summary = planner.summarize(decisions, plan_id=request.plan_id, request_id=request.request_id)

        self.assertEqual(summary.loaded_bytes, 250)
        self.assertEqual(summary.skipped_bytes, 750)
        self.assertAlmostEqual(summary.byte_saving_rate, 0.75)

    def test_build_partial_load_targets_keeps_prefix_aligned_contiguous_ranges(self) -> None:
        chunk = KVChunkMeta(
            request_id="req-1",
            layer_id=3,
            start_token=0,
            end_token=100,
            chunk_id="chunk",
            tier=MemoryTier.SSD,
            size_bytes=1000,
        )
        request = PartialKVLoadRequest(
            request_id="req-1",
            token_spans=(TokenSpan(0, 40),),
            plan_id="plan-5",
        )

        decisions = PartialKVLoadPlanner().plan([chunk], request)
        targets = build_partial_load_targets(decisions, object_key="prefix-a")

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertIsInstance(target, PartialKVLoadTarget)
        self.assertEqual(target.chunk_id, "chunk")
        self.assertEqual(target.object_key, "prefix-a")
        self.assertTrue(target.prefix_aligned)
        self.assertTrue(target.contiguous)
        self.assertEqual(target.token_span.start_token, 0)
        self.assertEqual(target.token_span.end_token, 40)

    def test_chunk_records_load_from_snapshot_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            snapshot = tmp / "snapshot.json"
            csv_path = tmp / "chunks.csv"
            snapshot.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "request_id": "req-1",
                                "layer_id": 0,
                                "start_token": 0,
                                "end_token": 16,
                                "chunk_id": "json-chunk",
                                "tier": "disk",
                                "size_bytes": 160,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["request_id", "layer_id", "start_token", "end_token", "chunk_id", "size_bytes"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "request_id": "req-2",
                        "layer_id": 1,
                        "start_token": 16,
                        "end_token": 32,
                        "chunk_id": "csv-chunk",
                        "size_bytes": 320,
                    }
                )

            json_chunks = load_chunks(snapshot)
            csv_chunks = load_chunks(csv_path)

            self.assertEqual(json_chunks[0].chunk_id, "json-chunk")
            self.assertEqual(json_chunks[0].tier, MemoryTier.SSD)
            self.assertEqual(csv_chunks[0].chunk_id, "csv-chunk")
            self.assertEqual(csv_chunks[0].size_bytes, 320)

    def test_decision_and_summary_csv_are_writable(self) -> None:
        chunk = chunk_meta_from_record(
            {
                "request_id": "req-1",
                "layer_id": 0,
                "start_token": 0,
                "end_token": 10,
                "chunk_id": "chunk",
                "size_bytes": 100,
            }
        )
        request = PartialKVLoadRequest(request_id="req-1", token_spans=(TokenSpan(0, 5),), plan_id="plan-4")
        planner = PartialKVLoadPlanner()
        decisions = planner.plan([chunk], request)
        summary = planner.summarize(decisions, plan_id=request.plan_id, request_id=request.request_id)

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            decisions_path = tmp / "decisions.csv"
            summary_path = tmp / "summary.csv"
            write_decisions_csv(decisions_path, decisions)
            write_summary_csv(summary_path, summary.to_record())

            with decisions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["action"], "load_partial")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["loaded_bytes"], "50")


if __name__ == "__main__":
    unittest.main()
