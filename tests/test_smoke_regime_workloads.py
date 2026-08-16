"""Unit tests for the smoke regime workload materializer."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark.materialize_smoke_regime_workloads import materialize


def _args(output_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        seed=0,
        context_tokens=6000,
        groups=2,
        revisits=3,
        churn_groups=4,
        churn_revisits=2,
        random_rows=6,
        output_tokens=16,
        prefetch_lead_s=0.25,
    )


class SmokeRegimeWorkloadsTests(unittest.TestCase):
    def test_smoke_workloads_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            out = Path(raw_tmp) / "smoke"
            manifest = materialize(_args(str(out)))
            self.assertEqual(
                sorted(manifest["workloads"]),
                ["constrained_kv_churn", "queued_concurrency", "random_no_reuse", "repeated_long_prefix"],
            )
            self.assertEqual(manifest["workloads"]["repeated_long_prefix"]["row_count"], 2 * 3)
            self.assertEqual(manifest["workloads"]["constrained_kv_churn"]["row_count"], 4 * 2)
            self.assertEqual(manifest["workloads"]["queued_concurrency"]["row_count"], 2 * 3)
            self.assertEqual(manifest["workloads"]["random_no_reuse"]["row_count"], 6)

            rows = [
                json.loads(line)
                for line in (out / "repeated_long_prefix.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 6)
            seeds = [row for row in rows if row["reuse_bucket"] == "none"]
            revisits = [row for row in rows if row["reuse_bucket"] != "none"]
            self.assertEqual(len(seeds), 2)
            self.assertEqual(len(revisits), 4)
            for row in rows:
                self.assertEqual(row["case"], "repeated_long_prefix")
                self.assertEqual(row["metadata"]["workload_type"], "repeated_long_prefix")
                self.assertTrue(row["metadata"]["smoke"])
                self.assertTrue(row["context_length"] >= 5120, "context must retain a tokenizer margin above the partial cap")
            for row in revisits:
                self.assertGreater(row["prefetch_lead_s"], 0.0)
                self.assertGreater(row["sleep_before_s"], 0.0)
            for row in seeds:
                self.assertEqual(row["prefetch_lead_s"], 0.0)

    def test_smoke_workloads_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            first = Path(raw_tmp) / "a"
            second = Path(raw_tmp) / "b"
            materialize(_args(str(first)))
            materialize(_args(str(second)))
            for name in ("repeated_long_prefix", "constrained_kv_churn"):
                a = (first / f"{name}.jsonl").read_bytes()
                b = (second / f"{name}.jsonl").read_bytes()
                self.assertEqual(a, b)

    def test_context_below_partial_cap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            args = _args(str(Path(raw_tmp) / "smoke"))
            args.context_tokens = 1024
            with self.assertRaises(SystemExit):
                materialize(args)


if __name__ == "__main__":
    unittest.main()
