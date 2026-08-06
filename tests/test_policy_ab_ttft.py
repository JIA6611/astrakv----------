import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.policy_ab_ttft import (
    build_suite_report,
    build_workload_bundle,
    write_workload_bundle,
)


class PolicyAbTtftTests(unittest.TestCase):
    def test_workload_bundle_generates_cycle_shaped_rows_with_idle_revisits(self) -> None:
        bundle = build_workload_bundle(
            anchor_count=2,
            churn_variants=3,
            prompt_tokens=64,
            warmup_cycles=1,
            sample_cycles=2,
            idle_seconds=2.5,
        )

        self.assertEqual(bundle.manifest["row_count"], 12)
        self.assertEqual(bundle.manifest["phase_counts"]["anchor_revisit"], 3)
        revisit_rows = [
            row for row in bundle.rows
            if str((row.get("metadata") or {}).get("phase") or "") == "anchor_revisit"
        ]
        self.assertEqual(len(revisit_rows), 3)
        self.assertTrue(all(float(row.get("sleep_before_s") or 0.0) == 2.5 for row in revisit_rows))
        self.assertTrue(all(int(row["expected_output_tokens"]) == 1 for row in bundle.rows))

    def test_suite_report_filters_to_evicted_and_prefetched_revisits(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            bundle = build_workload_bundle(
                anchor_count=2,
                churn_variants=3,
                prompt_tokens=48,
                warmup_cycles=1,
                sample_cycles=3,
                idle_seconds=0.0,
            )
            write_workload_bundle(root / "workload", bundle)
            revisit_rows = [
                row for row in bundle.rows
                if str((row.get("metadata") or {}).get("phase") or "") == "anchor_revisit"
                and str((row.get("metadata") or {}).get("cycle_kind") or "") == "sample"
            ]

            baseline_requests = []
            variant_requests = []
            baseline_events = []
            variant_events = []
            for index, row in enumerate(revisit_rows, start=1):
                request_started_s = 1000.0 + index
                request_base = {
                    "schema": "astrakv-benchmark-request-v2",
                    "request_id": row["request_id"],
                    "status": "ok",
                    "cache_key": row["cache_key"],
                    "prefix_id": row["prefix_id"],
                    "request_started_s": request_started_s,
                    "ttft_ms": 220.0 + index * 10.0,
                    "latency_ms": 500.0 + index * 10.0,
                    "case": row["case"],
                }
                baseline_requests.append(request_base)
                variant_requests.append({**request_base, "ttft_ms": 140.0 + index * 5.0})
                event_base = {
                    "schema": "astrakv-backend-hook-v2",
                    "record_type": "event",
                    "run_id": "run-a",
                    "request_id": f"seed-{index}",
                    "object_key": row["cache_key"],
                    "object_level": "prefix",
                    "backend_object_id": f"backend-{index}",
                    "status": "completed",
                    "binding_generation": 1,
                }
                baseline_events.append(
                    {
                        **event_base,
                        "event_id": f"baseline-offload-{index}",
                        "action": "offload",
                        "timestamp_ns": int((request_started_s * 1000.0 - 20.0) * 1_000_000.0),
                        "tier_before": "gpu",
                        "tier_after": "ssd",
                    }
                )
                variant_events.extend(
                    [
                        {
                            **event_base,
                            "event_id": f"variant-prefetch-{index}",
                            "action": "prefetch",
                            "timestamp_ns": int((request_started_s * 1000.0 - 30.0) * 1_000_000.0),
                            "tier_before": "ssd",
                            "tier_after": "cpu",
                        },
                        {
                            **event_base,
                            "event_id": f"variant-offload-{index}",
                            "action": "offload",
                            "timestamp_ns": int((request_started_s * 1000.0 - 20.0) * 1_000_000.0),
                            "tier_before": "gpu",
                            "tier_after": "ssd",
                        },
                    ]
                )

            write_jsonl(root / "baseline" / "request_results.jsonl", baseline_requests)
            write_jsonl(root / "baseline" / "runtime_events_raw.jsonl", baseline_events)
            write_jsonl(root / "baseline" / "runtime_structured_events.jsonl", [])
            write_jsonl(root / "baseline" / "astrakv_runtime_commands.jsonl", [])
            write_jsonl(root / "baseline" / "runtime_command_receipts.jsonl", [])
            write_jsonl(root / "variant" / "request_results.jsonl", variant_requests)
            write_jsonl(root / "variant" / "runtime_events_raw.jsonl", variant_events)
            write_jsonl(root / "variant" / "runtime_structured_events.jsonl", [])
            write_jsonl(root / "variant" / "astrakv_runtime_commands.jsonl", [])
            write_jsonl(root / "variant" / "runtime_command_receipts.jsonl", [])

            report = build_suite_report(root)

            self.assertEqual(report["schema"], "astrakv-policy-ab-ttft-report-v1")
            self.assertEqual(report["roles"]["baseline"]["valid_revisit_sample_count"], 3)
            self.assertEqual(report["roles"]["variant"]["valid_revisit_sample_count"], 3)
            self.assertEqual(report["comparison"]["verdict"], "variant_better")
            self.assertGreater(report["comparison"]["ttft_delta_ms"]["mean"], 0.0)
            self.assertTrue(any(item["source"] == "cache" for item in report["event_timeline"]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
