from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow
from scripts.benchmark.materialize_e3_prefetch_performance_workload import materialize
from scripts.reporting.validate_e3_prefetch_performance import validate


def source() -> RuntimeWorkloadRow:
    return RuntimeWorkloadRow(
        request_id="qasper-source", prompt="exact qasper prompt", prefix_id="prefix",
        prefix_hash="prefix-hash", cache_key="cache-key", arrival_index=0,
        reuse_ratio=1.0, reuse_bucket="high", context_length=4096,
        expected_output_tokens=32,
        metadata={"exact_prefix": True, "messages": [{"role": "user", "content": "exact qasper prompt"}]},
    )


class E3PrefetchPerformanceTests(unittest.TestCase):
    def test_runner_uses_the_supported_kv_core_manifest_scope(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "entrypoints" / "run_e3_prefetch_performance_suite.sh"
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("--claim-scope kv_core", source)
        self.assertIn("ASTRAKV_KV_CORE_PREFILL_ONLINE_CALIBRATION=true", source)
        self.assertNotIn("--claim-scope exploratory_prefetch_only", source)
        self.assertNotIn("--connector-version lmcache-vllm-v1-0.4.7 \\\n+    #", source)

    def test_materialized_workload_has_one_seed_and_identical_prefetch_targets(self) -> None:
        rows = materialize(source(), revisits=16, prefetch_lead_s=0.25, output_tokens=16)
        self.assertEqual(len(rows), 17)
        self.assertEqual(rows[0].metadata["e3_prefetch_role"], "seed")
        self.assertEqual(rows[0].prefetch_lead_s, 0.0)
        self.assertEqual({row.prompt for row in rows}, {"exact qasper prompt"})
        self.assertEqual({row.cache_key for row in rows}, {"cache-key"})
        self.assertEqual({row.metadata["generation_seed"] for row in rows}, {0})
        self.assertEqual({row.prefetch_lead_s for row in rows[1:]}, {0.25})
        self.assertTrue(all("-revisit-" in row.request_id for row in rows[1:]))

    def test_validator_accepts_only_prefetch_switch_and_consumed_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline, variant = root / "baseline", root / "variant"
            for path, prefetch in ((baseline, False), (variant, True)):
                path.mkdir()
                (path / "e3_prefetch_control.json").write_text(json.dumps({
                    "mode": "active", "topology": "gpu_cpu_ssd", "local_cpu_enabled": True,
                    "local_disk_enabled": True, "admission_enabled": True,
                    "prefill_online_calibration_enabled": True,
                    "invalidate_disk_backed_cpu_on_prefetch_lead": True,
                    "cpu_prefetch_enabled": prefetch, "model": "Qwen3-8B", "dtype": "bfloat16",
                    "output_tokens": 16, "prefetch_lead_s": 0.25,
                }), encoding="utf-8")
                requests = [
                    {"request_id": "e3-prefetch-x-revisit-01", "status": "ok", "ttft_ms": 100.0 if not prefetch else 80.0,
                     "prompt_hash": "p", "context_length": 4096, "generation_seed": 0,
                     "prefetch_lead_s": 0.25, "runtime_association_status": "linked"},
                ]
                (path / "request_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
                (path / "workload_source.jsonl").write_text('{"same": true}\n', encoding="utf-8")
                accounting = {
                    "logical_request_id": "e3-prefetch-x-revisit-01", "lookup_hit_tokens": 4096,
                    "allocated_external_tokens": 4096, "actual_loaded_tokens": 4096,
                    "recomputed_tokens": 0,
                }
                (path / "kv_core_request_accounting.jsonl").write_text(json.dumps(accounting) + "\n", encoding="utf-8")
                observation = {
                    "source": "scheduler_compute_progress", "accepted": True,
                    "prefill_tokens": 256, "sample_ms_per_token": 0.1,
                    "observed_prefill_ms_per_token": 0.1,
                }
                (path / "kv_core_cost_observations.jsonl").write_text(
                    json.dumps(observation) + "\n", encoding="utf-8",
                )
            ticket = {
                "prefetch_id": "ticket", "source_tier": "ssd", "target_tier": "cpu", "status": "consumed",
                "completed_bytes": 1, "target_request_id": "e3-prefetch-x-revisit-01",
                "consumer_request_id": "e3-prefetch-x-revisit-01", "physical_object_id": "obj",
                "binding_generation": 1, "prefix_hash": "p", "native_key": "key", "compatibility_identity": "compat",
            }
            (variant / "kv_core_prefetch_tickets.jsonl").write_text(json.dumps(ticket) + "\n", encoding="utf-8")
            result = validate(baseline, variant)
            (baseline / "kv_core_cost_observations.jsonl").unlink()
            missing_cost = validate(baseline, variant)
        self.assertTrue(result["measurement_valid"])
        self.assertFalse(result["correctness_accepted"])
        self.assertFalse(result["eligible_for_e4"])
        self.assertAlmostEqual(result["ttft_ms"]["p95_delta_percent"], -20.0)
        self.assertFalse(missing_cost["measurement_valid"])
        self.assertIn("baseline_online_prefill_cost_missing", missing_cost["errors"])


if __name__ == "__main__":
    unittest.main()
