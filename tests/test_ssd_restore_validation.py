from __future__ import annotations

import unittest
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.run_ssd_restore_validation import (
    build_reset_url,
    evaluate_ssd_restore_evidence,
    metric_delta,
    parse_prometheus_metrics,
)


class SSDRestoreValidationTests(unittest.TestCase):
    def test_restore_probe_configs_bind_metrics_to_loopback(self):
        probe = yaml.safe_load((ROOT / "configs" / "lmcache_disk_restore_probe.yaml").read_text(encoding="utf-8"))
        odirect = yaml.safe_load((ROOT / "configs" / "lmcache_disk_restore_odirect_probe.yaml").read_text(encoding="utf-8"))

        self.assertFalse(probe["local_cpu"])
        self.assertEqual(probe["max_local_cpu_size"], 2.0)
        self.assertTrue(probe["internal_api_server_enabled"])
        self.assertEqual(probe["internal_api_server_host"], "127.0.0.1")
        self.assertEqual(probe["internal_api_server_include_index_list"], [1])
        self.assertFalse(probe["extra_config"]["use_odirect"])
        self.assertTrue(odirect["extra_config"]["use_odirect"])

    def test_reset_keeps_external_connector_state(self):
        self.assertEqual(
            build_reset_url("http://127.0.0.1:8002"),
            "http://127.0.0.1:8002/reset_prefix_cache?reset_external=false",
        )

    def test_metric_delta_uses_only_numeric_prometheus_values(self):
        before = {"lmcache:num_retrieve_requests": 2.0, "ignored": "bad"}
        after = {"lmcache:num_retrieve_requests": 5.0, "new": 3.0}

        self.assertEqual(
            metric_delta(before, after),
            {"lmcache:num_retrieve_requests": 3.0, "new": 3.0},
        )

    def test_prometheus_parser_keeps_worker_counter_name_without_labels(self):
        metrics = parse_prometheus_metrics(
            "lmcache:num_retrieve_requests_total{role=\"scheduler\"} 9\n"
            "lmcache:num_retrieve_requests_total{role=\"worker\"} 2\n"
            "lmcache:num_retrieve_requests_created{role=\"worker\"} 3\n"
        )

        self.assertEqual(metrics, {"lmcache:num_retrieve_requests_total": 2.0})

    def test_restore_gate_rejects_missing_disk_read(self):
        evidence = evaluate_ssd_restore_evidence(
            vllm_cached_tokens=0,
            lmcache_loaded_tokens=256,
            retrieve_requests_delta=1.0,
            need_to_load_tokens=256,
            disk_read_observed=False,
            request_status="ok",
        )

        self.assertEqual(evidence["status"], "insufficient_ssd_restore_evidence")
        self.assertIn("disk_read_not_observed", evidence["missing"])

    def test_restore_gate_accepts_complete_correlated_evidence(self):
        evidence = evaluate_ssd_restore_evidence(
            vllm_cached_tokens=0,
            lmcache_loaded_tokens=256,
            retrieve_requests_delta=1.0,
            need_to_load_tokens=256,
            disk_read_observed=True,
            request_status="ok",
        )

        self.assertEqual(evidence["status"], "ssd_restore_evidence_complete")
        self.assertEqual(evidence["missing"], [])


if __name__ == "__main__":
    unittest.main()
