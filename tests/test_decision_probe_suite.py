"""Unit tests for the decision-probe workload materializer and validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark.materialize_decision_probe_workload import (
    SCENARIOS,
    WORKLOAD_NAME,
    materialize,
)
from scripts.reporting.validate_decision_probe import EXPECTED_ACTION, validate


def _source_row() -> dict:
    return {
        "schema": "astra-runtime-workload-v1",
        "request_id": "qasper-grouped-000000",
        "prompt": "shared context question",
        "prefix_id": "group-1",
        "prefix_hash": "sha256:abc",
        "cache_key": "sha256:abc",
        "arrival_index": 0,
        "reuse_ratio": 1.0,
        "reuse_bucket": "high",
        "context_length": 1024,
        "expected_output_tokens": 128,
        "batch_size": 1,
        "sleep_before_s": 0.0,
        "prefetch_lead_s": 0.0,
        "case": "source",
        "metadata": {"exact_prefix": True, "messages": [{"role": "user", "content": "shared context question"}]},
    }


def _manifest() -> dict:
    return {
        "schema": "astrakv-decision-probe-workload-v1",
        "workload": WORKLOAD_NAME,
        "reps_per_scenario": 3,
        "scenarios": {
            name: {
                "expected_decision": spec["expected_decision"],
                "expected_reason": spec["expected_reason"],
                "probe": spec["probe"],
            }
            for name, spec in SCENARIOS.items()
        },
    }


class DecisionProbeMaterializerTests(unittest.TestCase):
    def test_materializes_seed_plus_all_scenarios_with_locked_labels(self) -> None:
        from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl

        with tempfile.TemporaryDirectory() as raw_tmp:
            source_path = Path(raw_tmp) / "source.jsonl"
            source_path.write_text(json.dumps(_source_row()) + "\n", encoding="utf-8")
            source = load_runtime_workload_jsonl(source_path)[0]
            rows = materialize(source, output_tokens=8)
            self.assertEqual(rows[0].case, "decision_probe_seed")
            self.assertNotIn("kv_core_decision_probe", rows[0].metadata)
            self.assertEqual(len(rows), 1 + len(SCENARIOS) * 3)
            request_ids = {row.request_id for row in rows}
            self.assertEqual(len(request_ids), len(rows))
            probe_rows = [row for row in rows if row.case.startswith("decision_probe_S")]
            self.assertEqual(len(probe_rows), len(SCENARIOS) * 3)
            for row in probe_rows:
                scenario = row.case.rsplit("_", 1)[-1]
                spec = SCENARIOS[scenario]
                self.assertEqual(row.metadata["expected_decision"], spec["expected_decision"])
                self.assertEqual(row.metadata["expected_reason"], spec["expected_reason"])
                self.assertEqual(row.metadata["kv_core_decision_probe"], spec["probe"])
                self.assertEqual(row.cache_key, source.cache_key)


class DecisionProbeValidatorTests(unittest.TestCase):
    def _run_dir(self, tmp: str, *, corrupt_reason: str | None = None) -> Path:
        run = Path(tmp) / "run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "decision_probe_workload.manifest.json").write_text(
            json.dumps(_manifest(), sort_keys=True) + "\n", encoding="utf-8",
        )
        requests: list[dict] = []
        decisions: list[dict] = []
        associations: list[dict] = []
        accounting: list[dict] = []
        rep = 0
        for scenario, spec in SCENARIOS.items():
            for _ in range(3):
                rep += 1
                request_id = f"decision-probe-{scenario}-{rep:02d}"
                native_id = f"native-{request_id}"
                requests.append({
                    "request_id": request_id,
                    "workload_case": f"decision_probe_{scenario}",
                    "status": "ok",
                    "ttft_ms": 100.0,
                })
                reason = spec["expected_reason"] if corrupt_reason is None or scenario != "S1" else corrupt_reason
                decisions.append({
                    "request_id": request_id,
                    "action": EXPECTED_ACTION[spec["expected_decision"]],
                    "reason": reason,
                    "load_cost_ms": 1.0 if spec["expected_decision"] == "load" else 900.0,
                    "recompute_cost_ms": 800.0 if spec["expected_decision"] == "load" else 100.0,
                    "test_only": scenario == "S5",
                })
                associations.append({
                    "request_id": request_id,
                    "runtime_request_id": native_id,
                    "status": "associated",
                })
                if spec["expected_decision"] == "load":
                    accounting.append({
                        "native_request_id": native_id,
                        "terminal": True,
                        "allocated_external_tokens": 2048,
                        "actual_loaded_tokens": 2048,
                        "recompute_confirmed": False,
                        "recomputed_tokens": 0,
                    })
                else:
                    accounting.append({
                        "native_request_id": native_id,
                        "terminal": True,
                        "allocated_external_tokens": 0,
                        "actual_loaded_tokens": 0,
                        "recompute_confirmed": True,
                        "recomputed_tokens": 4096,
                    })
        (run / "request_results.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests), encoding="utf-8",
        )
        (run / "kv_core_policy_decisions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8",
        )
        (run / "request_context_associations.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in associations), encoding="utf-8",
        )
        (run / "kv_core_request_accounting.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in accounting), encoding="utf-8",
        )
        (run / "kv_core_run_metadata.json").write_text(
            json.dumps({"equivalence_test_enabled": True}) + "\n", encoding="utf-8",
        )
        return run

    def test_validator_accepts_locked_labels_and_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            run = self._run_dir(raw_tmp)
            errors: list[str] = []
            record = validate(run, run / "decision_probe_workload.manifest.json", errors)
            self.assertTrue(record["eligible"], record["errors"])
            self.assertEqual(record["probe_rows"], {"total": len(SCENARIOS) * 3, "passed": len(SCENARIOS) * 3})

    def test_validator_rejects_reason_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            run = self._run_dir(raw_tmp, corrupt_reason="recompute_cheaper")
            errors: list[str] = []
            record = validate(run, run / "decision_probe_workload.manifest.json", errors)
            self.assertFalse(record["eligible"])
            self.assertTrue(any("reason_mismatch:S1:" in error for error in record["errors"]))


if __name__ == "__main__":
    unittest.main()
