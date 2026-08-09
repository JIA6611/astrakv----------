from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.aggregate_kv_core_repeats import aggregate_acceptance_paths


class AggregateKVCoreRepeatsTests(unittest.TestCase):
    def _acceptance(self, root: Path, index: int, *, eligible: bool = True) -> Path:
        baseline = root / f"baseline-{index}"
        variant = root / f"variant-{index}"
        baseline.mkdir()
        variant.mkdir()
        rows = [
            {"sample_id": "a", "ttft_ms": 100.0},
            {"sample_id": "b", "ttft_ms": 200.0},
        ]
        (baseline / "request_results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
        )
        right = [dict(row, ttft_ms=row["ttft_ms"] * (0.98 if index == 0 else 1.01)) for row in rows]
        (variant / "request_results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in right) + "\n", encoding="utf-8",
        )
        record = {
            "schema": "astrakv-kv-core-acceptance-v2",
            "phase": "E2",
            "eligible": eligible,
            "errors": [] if eligible else ["failed"],
            "paired_manifest": {
                "artifact_hashes": {"baseline": {"workload": "workload"}, "variant": {"workload": "workload"}},
                "runs": {"baseline": {"path": str(baseline)}, "variant": {"path": str(variant)}},
            },
            "throughput_tokens_s": {"baseline": 10.0, "variant": 10.0},
            "request_accounting_count": 2,
            "uma_measurement": {"baseline": "process_rss_only", "variant": "process_rss_only"},
        }
        path = root / f"acceptance-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_aggregates_repeat_clusters_and_refuses_uma_claim_without_cgroup(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            result = aggregate_acceptance_paths([self._acceptance(root, 0), self._acceptance(root, 1)])
        self.assertTrue(result["eligible"])
        self.assertEqual(result["analysis_unit"], "independent_service_startup_repeat")
        self.assertEqual(result["correctness"]["request_accounting_count_total"], 4)
        self.assertEqual(result["uma_physical_memory_evidence"], "not_available_for_claim")
        self.assertEqual(result["performance_conclusion"], "inconclusive_no_performance_claim")

    def test_rejects_ineligible_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            result = aggregate_acceptance_paths([self._acceptance(root, 0), self._acceptance(root, 1, eligible=False)])
        self.assertFalse(result["eligible"])
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
