import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.moe_events import (
    MOE_EVENT_SCHEMA_VERSION,
    events_from_record,
    parse_events_jsonl,
    parse_router_log_line,
    summarize_events,
    write_events_jsonl,
    write_expert_summary_csv,
)
from scripts.research.extract_moe_expert_events import write_manifest, write_report


class MoEExpertEventTests(unittest.TestCase):
    def test_router_log_line_expands_topk_experts(self) -> None:
        events = parse_router_log_line(
            "2026-06-08 12:00:00 MoE router request_id=req-1 layer=3 token=17 "
            "top_k=2 experts=[4,9] scores=[0.8,0.2] tier=gpu latency_ms=1.5",
            source="router.log",
            line_number=7,
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "expert_route")
        self.assertEqual(events[0].request_id, "req-1")
        self.assertEqual(events[0].layer_id, 3)
        self.assertEqual(events[0].token_index, 17)
        self.assertEqual(events[0].expert_id, "4")
        self.assertEqual(events[0].expert_rank, 0)
        self.assertEqual(events[0].score, 0.8)
        self.assertEqual(events[0].top_k, 2)
        self.assertEqual(events[0].tier, "gpu")
        self.assertEqual(events[0].latency_ms, 1.5)
        self.assertEqual(events[1].expert_id, "9")

    def test_jsonl_records_preserve_zero_layer_and_token(self) -> None:
        events = events_from_record(
            {
                "schema": "external-router-v1",
                "event_type": "expert_route",
                "request_id": "req-zero",
                "layer_id": 0,
                "token_index": 0,
                "experts": [1, 2],
                "scores": [0.6, 0.4],
                "tier": "cpu",
                "metadata": {"model": "mixtral"},
            },
            source="events.jsonl",
            line_number=1,
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].layer_id, 0)
        self.assertEqual(events[0].token_index, 0)
        self.assertEqual(events[0].metadata["source_schema"], "external-router-v1")
        self.assertEqual(events[0].metadata["model"], "mixtral")

    def test_summary_reports_hot_experts_and_hit_rate(self) -> None:
        route_events = parse_router_log_line(
            "MoE router request_id=req-1 layer=2 token=5 experts=[4,7] scores=[0.7,0.3]",
            source="router.log",
            line_number=1,
        )
        hit_event = events_from_record(
            {"event_type": "expert_hit", "request_id": "req-1", "layer_id": 2, "token_index": 5, "expert_id": 4},
            source="events.jsonl",
        )[0]
        miss_event = events_from_record(
            {"event_type": "expert_miss", "request_id": "req-2", "layer_id": 2, "token_index": 8, "expert_id": 9},
            source="events.jsonl",
        )[0]

        summary = summarize_events([*route_events, hit_event, miss_event])
        self.assertEqual(summary["total_events"], 4)
        self.assertEqual(summary["routed_token_count"], 1)
        self.assertEqual(summary["unique_expert_ids"], 3)
        self.assertEqual(summary["expert_hit_rate"], 0.5)
        self.assertEqual(summary["expert_rows"][0]["expert_id"], "4")
        self.assertEqual(summary["expert_rows"][0]["activation_count"], 2)

    def test_jsonl_csv_report_and_manifest_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            input_path = tmp / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "event_type": "expert_route",
                        "request_id": "req-1",
                        "layer_id": 1,
                        "token_index": 2,
                        "experts": [3, 5],
                        "scores": [0.9, 0.1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events = parse_events_jsonl(input_path)
            events_path = tmp / "moe_expert_events.jsonl"
            csv_path = tmp / "moe_expert_summary.csv"
            report_path = tmp / "moe_expert_report.md"
            manifest_path = tmp / "moe_expert_manifest.json"

            write_events_jsonl(events, events_path)
            write_expert_summary_csv(events, csv_path)
            write_report(report_path, events, {"router_logs": [], "events_jsonl": [str(input_path)]}, events_path, csv_path)
            write_manifest(manifest_path, events, {"router_logs": [], "events_jsonl": [str(input_path)]}, events_path, csv_path, report_path)

            records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["schema"], MOE_EVENT_SCHEMA_VERSION)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("# MoE Expert Activation Trace Summary", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["total_events"], 2)


if __name__ == "__main__":
    unittest.main()
