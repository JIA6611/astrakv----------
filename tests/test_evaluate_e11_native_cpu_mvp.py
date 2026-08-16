from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.evaluate_e11_native_cpu_mvp import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluateE11NativeCPUMVPTest(unittest.TestCase):
    def _build_arm(self, root: Path, arm: str, ttft: list[float], *, ssd_policy="LRUCachePolicy") -> None:
        run_dir = root / arm / "rep-1" / "qasper" / "baseline"
        run_dir.mkdir(parents=True)
        policy = "astrakv" if arm == "arm-evict-b" else "lru"
        effective = "astrakv_native_cpu" if arm == "arm-evict-b" else "lmcache_lru"
        workload = [{"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"} for index in range(len(ttft))]
        _write_jsonl(run_dir / "workload_source.jsonl", workload)
        _write_jsonl(run_dir / "request_results.jsonl", [
            {
                "sample_id": f"sample-{index}",
                "request_id": f"request-{index}",
                "status": "ok",
                "ttft_ms": value,
                "cache_key": "hot-a" if index < 3 else ("hot-b" if index < 6 else f"cold-{index}"),
                "reuse_ratio": 0.666667 if index < 6 else 0.0,
            }
            for index, value in enumerate(ttft)
        ])
        _write_jsonl(run_dir / "native_policy_installation.jsonl", [{
            "record_type": "native_policy_installation",
            "status": "installed",
            "cpu_requested_policy": policy,
            "cpu_effective_policy": effective,
            "cpu_same_native_capacity_path": True,
            "cpu_delegate_policy_class": "LRUCachePolicy",
            "ssd_policy_unchanged": True,
            "ssd_effective_policy_class": ssd_policy,
        }])
        _write_jsonl(run_dir / "native_cache_policy_evictions.jsonl", [
            {
                "record_type": "native_cache_policy_eviction",
                "selection_id": f"{arm}-selection",
                "status": "selected",
                "effective_policy": effective,
                "backend_key_identity": "victim-astra" if arm == "arm-evict-b" else "victim-lru",
                "timestamp_ns": 100,
                "signals": {"logical_object_key": "prefix-cold"},
            },
            {
                "record_type": "native_cache_policy_eviction",
                "selection_id": f"{arm}-selection",
                "status": "completed",
                "effective_policy": effective,
                "backend_key_identity": "victim-astra" if arm == "arm-evict-b" else "victim-lru",
                "timestamp_ns": 200,
                "signals": {"logical_object_key": "prefix-cold"},
                "terminal_condition": "local_cpu_backend.remove:force=false",
            },
        ])
        _write_jsonl(run_dir / "runtime_events_raw.jsonl", [])

    def test_valid_native_pair_reports_claimable_improvement(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._build_arm(root, "arm-lru", [100.0, 120.0, 110.0, 130.0, 125.0, 115.0, 105.0])
            self._build_arm(root, "arm-evict-b", [80.0, 90.0, 85.0, 100.0, 95.0, 90.0, 85.0])

            result = evaluate(root, "baseline")

            self.assertTrue(result["correctness_valid"])
            self.assertTrue(result["directional_improvement"])
            self.assertFalse(result["claimable_improvement"])
            self.assertFalse(result["formal_repeat_ready"])
            self.assertEqual(result["native_evictions"]["arm-lru"]["completed"], 1)
            self.assertEqual(result["native_evictions"]["arm-evict-b"]["completed"], 1)

    def test_changed_ssd_policy_invalidates_experiment(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._build_arm(root, "arm-lru", [100.0, 120.0, 110.0, 130.0, 125.0, 115.0, 105.0])
            self._build_arm(root, "arm-evict-b", [90.0, 110.0, 100.0, 120.0, 115.0, 105.0, 95.0], ssd_policy="FIFOCachePolicy")

            result = evaluate(root, "baseline")

            self.assertFalse(result["installations"]["arm-evict-b"]["valid"])
            self.assertFalse(result["correctness_valid"])
            self.assertEqual(result["conclusion"]["tier"], "invalid_or_inconclusive")

    def test_ttft_improvement_is_not_claimable_when_bad_eviction_worsens(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._build_arm(root, "arm-lru", [100.0, 120.0, 110.0, 130.0, 125.0, 115.0, 105.0])
            self._build_arm(root, "arm-evict-b", [80.0, 90.0, 85.0, 100.0, 95.0, 90.0, 85.0])
            run = root / "arm-evict-b" / "rep-1" / "qasper" / "baseline"
            _write_jsonl(run / "runtime_events_raw.jsonl", [{
                "action": "cache_hit",
                "status": "completed",
                "object_key": "prefix-cold",
                "request_id": "later-request",
                "timestamp_ns": 300,
            }])

            result = evaluate(root, "baseline")

            self.assertTrue(result["correctness_valid"])
            self.assertFalse(result["quality_valid"])
            self.assertFalse(result["directional_improvement"])
            self.assertFalse(result["claimable_improvement"])
            self.assertEqual(result["conclusion"]["tier"], "valid_no_improvement")

    def test_same_request_reaccess_is_not_a_bad_eviction(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._build_arm(root, "arm-lru", [100.0] * 7)
            self._build_arm(root, "arm-evict-b", [90.0] * 7)
            run = root / "arm-evict-b" / "rep-1" / "qasper" / "baseline"
            evictions = [
                {
                    **row,
                    "signals": {
                        "logical_object_key": "prefix-cold",
                        "request_id": "same-request",
                    },
                }
                for row in (
                    json.loads(line)
                    for line in (run / "native_cache_policy_evictions.jsonl").read_text().splitlines()
                )
            ]
            _write_jsonl(run / "native_cache_policy_evictions.jsonl", evictions)
            _write_jsonl(run / "runtime_events_raw.jsonl", [{
                "action": "cache_load",
                "status": "completed",
                "object_key": "prefix-cold",
                "request_id": "same-request",
                "timestamp_ns": 300,
            }])

            result = evaluate(root, "baseline")

            self.assertEqual(result["bad_eviction"]["arm-evict-b"]["reaccessed_within_window"], 0)
            self.assertTrue(result["quality_valid"])


if __name__ == "__main__":
    unittest.main()
