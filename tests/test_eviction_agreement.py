import unittest

from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision, RuntimeEvictionEvent
from astrakv.runtime.eviction_agreement import compare_eviction


def decision(
    *,
    object_key: str = "prefix-a",
    request_id: str = "req-1",
    run_id: str = "run-a",
    action: str = "offload",
    index: int | None = 1,
    bytes: int | None = 100,
) -> OfflineEvictionDecision:
    return OfflineEvictionDecision(
        run_id=run_id,
        decision_id=f"decision-{object_key}",
        request_id=request_id,
        object_key=object_key,
        object_level=ObjectLevel.PREFIX,
        predicted_action=action,
        target_tier="disk",
        decision_index=index,
        bytes=bytes,
        decision_time_ns=100,
    )


def event(
    *,
    object_key: str = "prefix-a",
    request_id: str = "req-2",
    run_id: str = "run-a",
    action: str = "evict",
    index: int | None = 2,
    provenance: str = "runtime_structured",
    event_id: str = "event-a",
    bytes: int | None = 100,
) -> RuntimeEvictionEvent:
    return RuntimeEvictionEvent(
        run_id=run_id,
        runtime_event_id=event_id,
        request_id=request_id,
        object_key=object_key,
        object_level=ObjectLevel.PREFIX,
        actual_action=action,
        tier_after="disk",
        arrival_index=index,
        timestamp_ns=200,
        bytes=bytes,
        status="completed",
        provenance=provenance,
    )


class EvictionAgreementTests(unittest.TestCase):
    def test_matching_prefix_yields_tp_and_metrics(self) -> None:
        summary = compare_eviction([decision()], [event()], prediction_window_requests=10)

        self.assertEqual(summary.ground_truth_status, "valid")
        self.assertEqual(summary.metrics["tp"], 1)
        self.assertEqual(summary.metrics["fp"], 0)
        self.assertEqual(summary.metrics["fn"], 0)
        self.assertEqual(summary.metrics["precision"], 1.0)
        self.assertEqual(summary.metrics["recall"], 1.0)
        self.assertEqual(summary.metrics["f1"], 1.0)
        self.assertEqual(summary.metrics["tier_transition_accuracy"], 1.0)
        self.assertEqual(summary.metrics["byte_weighted_agreement"], 1.0)
        self.assertEqual(summary.metrics["lead_time_ns_mean"], 100.0)
        # Both actions are eviction-class actions, but the exact spelling differs.
        self.assertEqual(summary.metrics["exact_action_agreement"], 0.0)

    def test_unmatched_prediction_and_actual_yield_fp_and_fn(self) -> None:
        summary = compare_eviction(
            [decision(object_key="prefix-pred")],
            [event(object_key="prefix-actual")],
        )

        self.assertEqual(summary.metrics["tp"], 0)
        self.assertEqual(summary.metrics["fp"], 1)
        self.assertEqual(summary.metrics["fn"], 1)
        self.assertEqual(summary.metrics["precision"], 0.0)
        self.assertEqual(summary.metrics["recall"], 0.0)

    def test_cross_run_and_window_boundary_do_not_match(self) -> None:
        cross_run = compare_eviction([decision()], [event(run_id="another-run")])
        self.assertEqual(cross_run.metrics["tp"], 0)
        self.assertEqual(cross_run.metrics["fp"], 1)
        self.assertEqual(cross_run.metrics["fn"], 1)

        out_of_window = compare_eviction([decision(index=1)], [event(index=12)], prediction_window_requests=10)
        self.assertEqual(out_of_window.metrics["tp"], 0)
        self.assertEqual(out_of_window.metrics["fp"], 1)
        self.assertEqual(out_of_window.metrics["fn"], 1)

    def test_duplicate_runtime_acknowledgement_is_deduplicated(self) -> None:
        duplicate = event(event_id="copied-event")
        summary = compare_eviction([decision()], [event(), duplicate])

        self.assertEqual(summary.metrics["runtime_event_count"], 1)
        self.assertEqual(summary.metrics["tp"], 1)
        self.assertEqual(summary.metrics["fn"], 0)

    def test_log_heuristic_never_creates_real_runtime_accuracy(self) -> None:
        summary = compare_eviction([decision()], [event(provenance="log_heuristic")])

        self.assertEqual(summary.ground_truth_status, "insufficient_ground_truth")
        self.assertEqual(summary.metrics["runtime_event_count"], 0)


if __name__ == "__main__":
    unittest.main()
