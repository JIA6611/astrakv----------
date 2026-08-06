import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.scheduler.hints import SchedulerHint


class SchedulerHintIndexTests(unittest.TestCase):
    def test_best_hint_prefers_high_priority_match(self) -> None:
        index = SchedulerHintIndex.from_hints(
            [
                SchedulerHint(
                    request_id="req-1",
                    action="offload",
                    reason="cold",
                    priority=10,
                    metadata={"chunk_id": "chunk-a"},
                ),
                SchedulerHint(
                    request_id="req-1",
                    action="prefetch",
                    reason="prefix hot",
                    priority=50,
                    metadata={"chunk_id": "chunk-a"},
                ),
            ]
        )

        hint = index.best_hint_for_object(request_id="req-1", backend_object_id="chunk-a", object_key="prefix-a")

        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertEqual(hint.action, "prefetch")
        self.assertEqual(hint.priority, 50)

    def test_jsonl_loader_reads_scheduler_hint_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "hints.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "request_id": "req-1",
                        "action": "load",
                        "reason": "reuse",
                        "priority": 42,
                        "metadata": {"chunk_id": "chunk-a"},
                    }
                ) + "\n",
                encoding="utf-8",
            )
            index = SchedulerHintIndex.from_jsonl(path)

            hint = index.best_hint_for_object(request_id="req-1", backend_object_id="chunk-a", object_key="prefix-a")
            self.assertIsNotNone(hint)
            assert hint is not None
            self.assertEqual(hint.action, "load")


if __name__ == "__main__":
    unittest.main()
