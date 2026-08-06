import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.research.run_moe_route_trace import (
    infer_top_k,
    router_records_from_logits,
    summarize_records,
    write_manifest,
    write_report,
)


class MoERouteTraceTests(unittest.TestCase):
    def test_router_logits_export_topk_expert_records(self) -> None:
        logits = np.array(
            [
                [[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]],
            ]
        )

        records = router_records_from_logits(
            [logits],
            input_ids=[11, 22],
            request_id="req-0",
            model_name="local-moe",
            top_k=2,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event_type"], "expert_route")
        self.assertEqual(records[0]["layer_id"], 0)
        self.assertEqual(records[0]["token_index"], 0)
        self.assertEqual(records[0]["experts"], [1, 2])
        self.assertEqual(records[1]["experts"], [0, 2])
        self.assertEqual(records[0]["metadata"]["source"], "hf_forward_router_logits")

    def test_summary_counts_requests_layers_and_experts(self) -> None:
        records = [
            {"request_id": "a", "layer_id": 0, "experts": [1, 2]},
            {"request_id": "a", "layer_id": 1, "experts": [2]},
            {"request_id": "b", "layer_id": 0, "experts": [3]},
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["unique_requests"], 2)
        self.assertEqual(summary["unique_layers"], 2)
        self.assertEqual(summary["unique_experts"], 3)
        self.assertEqual(summary["unique_layer_experts"], 4)

    def test_infer_topk_uses_common_config_fields(self) -> None:
        cfg = Namespace(num_experts_per_tok=4)

        self.assertEqual(infer_top_k(cfg), 4)
        self.assertEqual(infer_top_k(Namespace()), 2)

    def test_report_and_manifest_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            events_path = tmp / "moe_route_events.jsonl"
            summary_path = tmp / "moe_route_summary.json"
            report_path = tmp / "moe_route_report.md"
            manifest_path = tmp / "moe_route_manifest.json"
            summary = {
                "total_records": 2,
                "unique_requests": 1,
                "unique_layers": 1,
                "unique_experts": 2,
                "unique_layer_experts": 2,
            }
            args = Namespace(
                model="models/local-moe",
                local_files_only=True,
                device="cpu",
                dtype="float32",
                max_input_tokens=16,
            )

            write_report(
                report_path,
                args,
                ["hello"],
                [{"request_id": "req-0", "prompt_chars": 5, "input_tokens": 1, "route_events": 2}],
                summary,
                events_path,
            )
            write_manifest(
                manifest_path,
                args,
                [{"request_id": "req-0", "prompt_chars": 5, "input_tokens": 1, "route_events": 2}],
                summary,
                events_path,
                summary_path,
                report_path,
            )

            self.assertIn("# MoE Route Trace Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "astra-moe-route-trace-manifest-v1")
            self.assertTrue(manifest["inputs"]["local_files_only"])


if __name__ == "__main__":
    unittest.main()
