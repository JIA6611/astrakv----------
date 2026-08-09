import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.benchmarks.runtime_workload import WorkloadContractError, load_runtime_workload_jsonl
from astrakv.runtime.profile_db import ProfileDB
from scripts.policy.build_trace_store import build_events
from astrakv.scheduler.object_scheduler import candidates_from_profile_db


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "request_id": "req-1",
        "prompt": "shared prefix prompt",
        "prefix_id": "prefix-a",
        "prefix_hash": "hash-a",
        "cache_key": "cache-a",
        "arrival_index": 1,
        "reuse_ratio": 0.75,
        "reuse_bucket": "high",
        "case": "case-a",
    }
    row.update(overrides)
    return row


class RuntimeWorkloadContractTests(unittest.TestCase):
    def test_contract_validates_and_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "workload.jsonl"
            path.write_text(
                json.dumps(valid_row(request_id="req-2", arrival_index=2)) + "\n"
                + json.dumps(valid_row(request_id="req-1", arrival_index=1)) + "\n",
                encoding="utf-8",
            )
            rows = load_runtime_workload_jsonl(path)
            self.assertEqual([row.request_id for row in rows], ["req-1", "req-2"])
            self.assertEqual(rows[0].request_metadata("run-a")["run_id"], "run-a")

    def test_contract_accepts_non_negative_sleep_before_revisit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "workload.jsonl"
            path.write_text(json.dumps(valid_row(sleep_before_s=1.5)) + "\n", encoding="utf-8")
            rows = load_runtime_workload_jsonl(path)
            self.assertEqual(rows[0].sleep_before_s, 1.5)

            path.write_text(json.dumps(valid_row(sleep_before_s=-0.1)) + "\n", encoding="utf-8")
            with self.assertRaises(WorkloadContractError):
                load_runtime_workload_jsonl(path)

    def test_contract_keeps_prefetch_lead_distinct_from_arrival_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "workload.jsonl"
            path.write_text(
                json.dumps(valid_row(sleep_before_s=1.5, prefetch_lead_s=0.05)) + "\n",
                encoding="utf-8",
            )
            row = load_runtime_workload_jsonl(path)[0]
            self.assertEqual(row.sleep_before_s, 1.5)
            self.assertEqual(row.prefetch_lead_s, 0.05)
            self.assertEqual(row.to_record()["prefetch_lead_s"], 0.05)

            path.write_text(json.dumps(valid_row(prefetch_lead_s=-0.1)) + "\n", encoding="utf-8")
            with self.assertRaises(WorkloadContractError):
                load_runtime_workload_jsonl(path)

    def test_contract_rejects_missing_duplicates_and_invalid_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "workload.jsonl"
            path.write_text(json.dumps(valid_row(reuse_ratio=1.1)) + "\n", encoding="utf-8")
            with self.assertRaises(WorkloadContractError):
                load_runtime_workload_jsonl(path)

            path.write_text(
                json.dumps(valid_row()) + "\n" + json.dumps(valid_row(request_id="req-2")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(WorkloadContractError):
                load_runtime_workload_jsonl(path)

    def test_identity_flows_from_workload_to_trace_profile_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            workload = tmp / "workload.jsonl"
            cache_events = tmp / "cache_events.jsonl"
            workload.write_text(json.dumps(valid_row()) + "\n", encoding="utf-8")
            cache_events.write_text(json.dumps({
                "event_type": "cache_hit", "source": "server.log", "status": "ok",
                "request_id": "req-1", "metadata": {},
            }) + "\n", encoding="utf-8")
            events = build_events(Namespace(
                cache_events=[str(cache_events)], prefetch_events=[], samples=[],
                workload_manifest=str(workload), run_id="run-a",
            ))
            self.assertEqual(events[0].metadata["prefix_id"], "prefix-a")
            self.assertFalse(events[0].metadata["legacy_unlinked"])
            db = ProfileDB.from_trace_events(events, workload_id="workload-a")
            profile = db.get_chunk("cache-a", workload_id="workload-a")
            assert profile is not None
            self.assertEqual(profile.run_id, "run-a")
            self.assertEqual(profile.prefix_id, "prefix-a")
            candidate = candidates_from_profile_db(db)[0]
            self.assertEqual(candidate.metadata["request_id"], "req-1")
            self.assertEqual(candidate.metadata["arrival_index"], 1)


if __name__ == "__main__":
    unittest.main()
