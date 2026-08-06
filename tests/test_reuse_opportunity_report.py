from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.reporting.build_reuse_opportunity_report import build_reuse_opportunity_report


def observation(
    arrival_index: int, block_hashes: list[str], reused_tokens: int, *, dataset_id: str = "qasper"
) -> dict[str, object]:
    return {
        "evidence_class": "modeled_dataset_metadata",
        "workflow_id": f"wf-{arrival_index}",
        "parent_request_id": f"req-{arrival_index}",
        "subtask_index": 0,
        "arrival_index": arrival_index,
        "dataset_id": dataset_id,
        "workload_id": "qasper-random",
        "adapter": "single_request",
        "token_count": 8,
        "block_size_tokens": 4,
        "block_hashes": block_hashes,
        "historical_reused_tokens": reused_tokens,
        "historical_reuse_count": reused_tokens // 4,
        "kv_bytes_per_token": 16,
        "potential_kv_bytes": reused_tokens * 16,
    }


class ReuseOpportunityReportTests(unittest.TestCase):
    def test_marks_pilot_incomplete_without_two_named_independent_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "qasper.jsonl"
            raw.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        observation(0, ["a", "b"], 0),
                        observation(1, ["a", "c"], 4),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_reuse_opportunity_report(
                {
                    "sampling": {
                        "seed": 20260717,
                        "smoke_request_count": 10,
                        "pilot_request_count": 50,
                        "selection_method": "published_arrival_order_prefix",
                    },
                    "required_dataset_slots": [
                        {"slot": "qasper", "dataset_id": "qasper", "source_id": "task1-qasper", "raw_observation": str(raw)},
                        {"slot": "independent_dataset_2", "dataset_id": None, "status": "unavailable", "reason": "not provided"},
                        {"slot": "independent_dataset_3", "dataset_id": None, "status": "unavailable", "reason": "not provided"},
                    ]
                }
            )

        self.assertEqual(report["decision"], "observation_incomplete")
        self.assertEqual(report["raw_workloads"][0]["request_count"], 2)
        self.assertEqual(report["raw_workloads"][0]["reusable_token_ratio"], 0.25)
        self.assertEqual(report["raw_workloads"][0]["unique_prefix_block_count"], 3)
        self.assertEqual(report["raw_workloads"][0]["duplicated_prefix_block_ratio"], 0.25)
        self.assertEqual(report["composed_stress_workloads"], [])
        self.assertEqual(report["sampling"]["seed"], 20260717)
        self.assertNotIn("aggregate", report)
        self.assertNotIn("observed_cache_hit", json.dumps(report))

    def test_keeps_declared_composed_stress_separate_from_raw_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            composed = root / "composed.jsonl"
            raw.write_text(json.dumps(observation(0, ["a", "b"], 0)) + "\n", encoding="utf-8")
            composed.write_text(
                json.dumps(observation(0, ["x", "x"], 4, dataset_id="qasper-stress-v1")) + "\n",
                encoding="utf-8",
            )
            report = build_reuse_opportunity_report(
                {
                    "required_dataset_slots": [
                        {"slot": "qasper", "dataset_id": "qasper", "source_id": "task1-qasper", "raw_observation": str(raw)}
                    ],
                    "composed_stress": [
                        {"dataset_id": "qasper-stress-v1", "source_id": "qasper", "observation": str(composed)}
                    ],
                }
            )

        self.assertEqual(report["decision"], "raw_workload_selected_with_composed_stress")
        self.assertEqual(len(report["raw_workloads"]), 1)
        self.assertEqual(len(report["composed_stress_workloads"]), 1)
        self.assertNotEqual(report["raw_workloads"][0]["input_sha256"], report["composed_stress_workloads"][0]["input_sha256"])


if __name__ == "__main__":
    unittest.main()
