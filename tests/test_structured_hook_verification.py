import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "scripts" / "benchmark" / "verify_structured_eviction_hook.py"
SPEC = importlib.util.spec_from_file_location("structured_hook_verifier", MODULE)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class StructuredHookVerificationTests(unittest.TestCase):
    def test_successful_object_event_is_valid(self) -> None:
        self.assertEqual(VERIFIER.event_errors({
            "run_id": "r", "request_id": "q", "object_key": "p", "object_level": "prefix",
            "action": "cache_evict", "status": "completed", "timestamp_ns": 1,
        }, "r"), [])

    def test_log_like_event_without_object_or_timestamp_is_rejected(self) -> None:
        errors = VERIFIER.event_errors({"run_id": "r", "status": "completed", "action": "cache_evict"}, "r")
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
