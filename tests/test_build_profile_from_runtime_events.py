"""Unit tests for the KV-Core runtime events -> astra-trace-v1 converter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.policy.build_profile_from_runtime_events import build_events  # noqa: E402


PREFIX_A = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PREFIX_B = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Args:
    run_id = "train-qasper-test"
    workload_manifest = ""
    associations = ""
    binding_events = ""
    native_callbacks = ""
    prefetch_receipts = ""


class BuildProfileFromRuntimeEventsTests(unittest.TestCase):
    def test_joins_runtime_signals_onto_canonical_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            canonical = [
                {"request_id": "req-0", "cache_key": PREFIX_A, "context_length": 100, "arrival_index": 0},
                {"request_id": "req-1", "cache_key": PREFIX_A, "context_length": 100, "arrival_index": 1},
                {"request_id": "req-2", "cache_key": PREFIX_B, "context_length": 90, "arrival_index": 2},
            ]
            associations = [
                {"request_id": "req-0", "runtime_request_id": "chatcmpl-req-0"},
                {"request_id": "req-1", "runtime_request_id": "chatcmpl-req-1"},
                {"request_id": "req-2", "runtime_request_id": "chatcmpl-req-2"},
            ]
            native = [
                {"callback": "scheduler_exact_lookup", "request_id": "chatcmpl-req-0",
                 "locally_cached_tokens": 0, "lookup_hit_tokens": 0},
                {"callback": "scheduler_exact_lookup", "request_id": "chatcmpl-req-1",
                 "locally_cached_tokens": 4096, "lookup_hit_tokens": 4096},
                {"callback": "scheduler_compute_progress", "request_id": "chatcmpl-req-1"},
            ]
            bindings = [
                {"action": "cache_store", "status": "submitted", "request_id": "req-0",
                 "tier_after": "cpu", "backend_object_id": "bo-0",
                 "metadata": {"size_bytes": 37748736}},
                {"action": "cache_store", "status": "completed", "request_id": "req-0",
                 "tier_after": "cpu", "backend_object_id": "bo-0"},
            ]
            receipts = [
                {"action": "prefetch", "status": "completed", "request_id": "req-0",
                 "metadata": {"object_key": PREFIX_A, "prefetched": 1, "failure_reason": None},
                 "bytes": 641728512},
            ]
            manifest = tmp / "canonical.jsonl"
            assoc_path = tmp / "assoc.jsonl"
            native_path = tmp / "native.jsonl"
            binding_path = tmp / "bindings.jsonl"
            receipt_path = tmp / "receipts.jsonl"
            _write_jsonl(manifest, canonical)
            _write_jsonl(assoc_path, associations)
            _write_jsonl(native_path, native)
            _write_jsonl(binding_path, bindings)
            _write_jsonl(receipt_path, receipts)

            args = _Args()
            args.workload_manifest = str(manifest)
            args.associations = str(assoc_path)
            args.binding_events = str(binding_path)
            args.native_callbacks = str(native_path)
            args.prefetch_receipts = str(receipt_path)
            events = build_events(args)

        records = [event.to_record() for event in events]
        by_type: dict[str, list[dict]] = {}
        for record in records:
            by_type.setdefault(record["event_type"], []).append(record)
        self.assertEqual(len(by_type.get("cache_lookup", [])), 2)
        self.assertEqual(len(by_type.get("cache_hit", [])), 1)
        self.assertEqual(len(by_type.get("cache_miss", [])), 1)
        self.assertEqual(len(by_type.get("cache_store", [])), 2)
        self.assertEqual(len(by_type.get("prefetch", [])), 1)
        hit = by_type["cache_hit"][0]
        self.assertEqual(hit["chunk_id"], PREFIX_A)
        self.assertEqual(hit["request_id"], "req-1")
        store = by_type["cache_store"][0]
        self.assertEqual(store["chunk_id"], PREFIX_A)
        self.assertEqual(store["bytes"], 37748736)
        prefetch = by_type["prefetch"][0]
        self.assertEqual(prefetch["chunk_id"], PREFIX_A)

    def test_converter_profile_chain_has_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            canonical = [
                {"request_id": "req-0", "cache_key": PREFIX_A},
                {"request_id": "req-1", "cache_key": PREFIX_A},
            ]
            associations = [
                {"request_id": "req-0", "runtime_request_id": "chatcmpl-req-0"},
                {"request_id": "req-1", "runtime_request_id": "chatcmpl-req-1"},
            ]
            native = [
                {"callback": "scheduler_exact_lookup", "request_id": "chatcmpl-req-0", "locally_cached_tokens": 0},
                {"callback": "scheduler_exact_lookup", "request_id": "chatcmpl-req-1", "locally_cached_tokens": 4096},
            ]
            bindings = [
                {"action": "cache_store", "status": "completed", "request_id": "req-0", "tier_after": "cpu",
                 "metadata": {"size_bytes": 37748736}},
            ]
            manifest = tmp / "canonical.jsonl"
            assoc_path = tmp / "assoc.jsonl"
            native_path = tmp / "native.jsonl"
            binding_path = tmp / "bindings.jsonl"
            _write_jsonl(manifest, canonical)
            _write_jsonl(assoc_path, associations)
            _write_jsonl(native_path, native)
            _write_jsonl(binding_path, bindings)
            trace_path = tmp / "trace.jsonl"
            profile_dir = tmp / "profile"

            args = _Args()
            args.workload_manifest = str(manifest)
            args.associations = str(assoc_path)
            args.binding_events = str(binding_path)
            args.native_callbacks = str(native_path)
            events = build_events(args)
            trace_path.write_text(
                "".join(json.dumps(e.to_record(), sort_keys=True) + "\n" for e in events),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/policy/build_profile_db.py"),
                    "--trace-events", str(trace_path),
                    "--workload-id", "train-qasper",
                    "--output-dir", str(profile_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            db = json.loads((profile_dir / "profile_db.json").read_text(encoding="utf-8"))
            chunks = {c["chunk_id"]: c for c in db["chunks"]}
            self.assertIn(PREFIX_A, chunks)
            self.assertGreater(chunks[PREFIX_A]["cache_hits"], 0)
            self.assertGreater(chunks[PREFIX_A]["request_count"], 0)


if __name__ == "__main__":
    unittest.main()
