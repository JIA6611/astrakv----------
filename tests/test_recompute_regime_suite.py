"""Unit tests for the recompute-only arm variant and the regime report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark.materialize_recompute_only_workload import materialize as make_recompute_variant
from scripts.reporting.build_load_recompute_regime_report import build


def _canonical_rows() -> list[dict]:
    rows = []
    for index, bucket in enumerate(("none", "high", "low")):
        rows.append({
            "schema": "astra-runtime-workload-v1",
            "request_id": f"workload-{index:06d}",
            "prompt": "shared context question",
            "prefix_id": "group-1",
            "prefix_hash": "sha256:abc",
            "cache_key": "sha256:abc",
            "arrival_index": index,
            "reuse_ratio": 0.0 if bucket == "none" else 0.9,
            "reuse_bucket": bucket,
            "context_length": 1024,
            "expected_output_tokens": 16,
            "batch_size": 1,
            "sleep_before_s": 0.0,
            "prefetch_lead_s": 0.0,
            "case": "workload",
            "metadata": {"generation_seed": 0, "visit": index},
        })
    return rows


class RecomputeOnlyVariantTests(unittest.TestCase):
    def test_only_revisits_get_force_recompute_and_identity_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            source = Path(raw_tmp) / "repeated_long_prefix.jsonl"
            source.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in _canonical_rows()),
                encoding="utf-8",
            )
            output = Path(raw_tmp) / "variant"
            manifest = make_recompute_variant(source, output)
            self.assertEqual(manifest["force_recompute_rows"], 2)
            self.assertEqual(manifest["seed_rows_unchanged"], True)
            rows = [json.loads(line) for line in (output / "repeated_long_prefix.jsonl").read_text().splitlines()]
            source_rows = _canonical_rows()
            self.assertEqual(len(rows), len(source_rows))
            for original, row in zip(source_rows, rows):
                self.assertEqual(row["request_id"], original["request_id"])
                self.assertEqual(row["case"], original["case"])
                self.assertEqual(row["arrival_index"], original["arrival_index"])
            self.assertNotIn("kv_core_decision_probe", rows[0]["metadata"])
            for row in rows[1:]:
                self.assertEqual(row["metadata"]["kv_core_decision_probe"], {"force_recompute": True})


def _write_run_dir(root: Path, arm: str, *, ttfts: list[float], uma_gb: float, loaded_tokens: int, blocks: int = 100) -> None:
    run = root
    run.mkdir(parents=True, exist_ok=True)
    requests = [
        {
            "request_id": f"sample-{index:03d}",
            "sample_id": f"sample-{index:03d}",
            "workload_case": f"{arm}_case",
            "status": "ok",
            "ttft_ms": ttft,
            "output_tokens_observed": 8,
        }
        for index, ttft in enumerate(ttfts)
    ]
    (run / "request_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests), encoding="utf-8",
    )
    (run / "uma_resource_samples.jsonl").write_text(
        json.dumps({
            "cgroup_memory_current_bytes": int(uma_gb * 1024**3),
            "cgroup_memory_status": "valid",
            "process_rss_bytes": int(uma_gb * 0.8 * 1024**3),
        }) + "\n", encoding="utf-8",
    )
    (run / "kv_core_run_metadata.json").write_text(
        json.dumps({"vllm_kv_block_budget": blocks}) + "\n", encoding="utf-8",
    )
    (run / "kv_core_request_accounting.jsonl").write_text(
        json.dumps({
            "native_request_id": "native-sample",
            "terminal": True,
            "allocated_external_tokens": loaded_tokens,
            "actual_loaded_tokens": loaded_tokens,
        }) + "\n", encoding="utf-8",
    )


class RegimeReportTests(unittest.TestCase):
    def test_verdicts_ttft_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            _write_run_dir(root / "repeated_long_prefix/recompute_only", "recompute_only", ttfts=[200.0] * 8, uma_gb=15.0, loaded_tokens=0)
            _write_run_dir(root / "repeated_long_prefix/full", "full", ttfts=[150.0] * 8, uma_gb=20.0, loaded_tokens=4096)
            _write_run_dir(root / "repeated_long_prefix/partial", "partial", ttfts=[160.0] * 8, uma_gb=18.0, loaded_tokens=2048)
            _write_run_dir(root / "constrained_kv_churn/off", "off", ttfts=[180.0] * 8, uma_gb=10.0, loaded_tokens=0)
            _write_run_dir(root / "constrained_kv_churn/recompute_only", "recompute_only", ttfts=[190.0] * 8, uma_gb=10.1, loaded_tokens=0)
            _write_run_dir(root / "constrained_kv_churn/full", "full", ttfts=[170.0] * 8, uma_gb=20.0, loaded_tokens=4096)
            _write_run_dir(root / "constrained_kv_churn/partial", "partial", ttfts=[175.0] * 8, uma_gb=16.0, loaded_tokens=2048)
            cells = [
                {"workload": w, "arm": a, "phase": p, "baseline_dir": str(root / w / a), "variant_dir": str(root / w / a)}
                for w, a, p in (
                    ("repeated_long_prefix", "recompute_only", "E2R"),
                    ("repeated_long_prefix", "full", "E3"),
                    ("repeated_long_prefix", "partial", "E4"),
                    ("constrained_kv_churn", "off", "E0"),
                    ("constrained_kv_churn", "recompute_only", "E2R"),
                    ("constrained_kv_churn", "full", "E3"),
                    ("constrained_kv_churn", "partial", "E4"),
                )
            ]
            summary = build(cells)
            verdicts = summary["record"]["verdicts"]
            self.assertTrue(verdicts["repeated_long_prefix"]["label"].startswith("load wins TTFT"))
            self.assertTrue(verdicts["constrained_kv_churn"]["label"].startswith("recompute-only wins memory"))
            self.assertIn("## Verdicts", summary["markdown"])


if __name__ == "__main__":
    unittest.main()
