import unittest

from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.scheduler.hints import SchedulerHint


class SchedulerHintIndexTests(unittest.TestCase):
    def test_best_hint_prefers_prefix_and_object_keys_before_request_fallback(self) -> None:
        index = SchedulerHintIndex.from_hints([
            SchedulerHint(
                request_id="req-a",
                action="defer",
                reason="request fallback",
                priority=1,
                metadata={},
            ),
            SchedulerHint(
                request_id="",
                action="prefetch",
                reason="prefix hit",
                priority=5,
                metadata={"prefix_id": "prefix-1", "object_key": "prefix-1"},
            ),
        ])

        hint = index.best_hint_for_object(
            request_id="req-a",
            backend_object_id="",
            object_key="prefix-1",
            prefix_id="prefix-1",
            prefix_hash="",
        )

        self.assertIsNotNone(hint)
        self.assertEqual(hint.action, "prefetch")
        self.assertEqual(hint.reason, "prefix hit")

    def test_best_hint_matches_backend_object_id(self) -> None:
        index = SchedulerHintIndex.from_hints([
            SchedulerHint(
                request_id="",
                action="prefetch",
                reason="object hit",
                priority=4,
                metadata={"backend_object_id": "block-9"},
            )
        ])

        hint = index.best_hint_for_object(
            request_id="req-b",
            backend_object_id="block-9",
            object_key="prefix-2",
            prefix_id="",
            prefix_hash="",
        )

        self.assertIsNotNone(hint)
        self.assertEqual(hint.reason, "object hit")


if __name__ == "__main__":
    unittest.main()
