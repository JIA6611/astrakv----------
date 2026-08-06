import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.normalize_runtime_eviction_events import load_verified_structured_events


class StructuredEventNormalizationTests(unittest.TestCase):
    def test_only_verified_same_run_events_become_runtime_structured(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({
                "run_id": "run", "request_id": "req", "object_key": "prefix", "object_level": "prefix",
                "action": "cache_evict", "status": "completed", "timestamp_ns": 10,
            }) + "\n", encoding="utf-8")
            verification = root / "verification.json"
            verification.write_text(json.dumps({"status": "verified", "run_id": "run", "events_sha256": hashlib.sha256(events.read_bytes()).hexdigest()}), encoding="utf-8")
            args = type("Args", (), {"structured_events": str(events), "structured_hook_verification": str(verification), "run_id": "run"})()
            output = load_verified_structured_events(args, {"req": {"arrival_index": 2}})
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0].provenance, "runtime_structured")
            self.assertEqual(output[0].actual_action, "evict")

    def test_unverified_events_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            events = root / "events.jsonl"
            events.write_text("{}\n", encoding="utf-8")
            verification = root / "verification.json"
            verification.write_text(json.dumps({"status": "rejected", "run_id": "run"}), encoding="utf-8")
            args = type("Args", (), {"structured_events": str(events), "structured_hook_verification": str(verification), "run_id": "run"})()
            with self.assertRaises(SystemExit):
                load_verified_structured_events(args, {})


if __name__ == "__main__":
    unittest.main()
