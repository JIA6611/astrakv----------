import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.moe.expert_predictor import (
    ExpertPredictorConfig,
    RouterAwareExpertPredictor,
    load_expert_load_plan,
    observations_from_events,
    summarize_predictions,
    write_predictions_csv,
    write_prefetch_hints_jsonl,
)
from astrakv.runtime.moe_events import events_from_record, write_events_jsonl
from scripts.research.predict_moe_experts import write_manifest, write_report


class MoEExpertPredictorTests(unittest.TestCase):
    def test_observations_group_route_events_by_token(self) -> None:
        events = events_from_record(
            {
                "event_type": "expert_route",
                "request_id": "req-1",
                "layer_id": 0,
                "token_index": 0,
                "experts": [2, 1],
                "scores": [0.4, 0.6],
            },
            source="moe.jsonl",
        )

        observations = observations_from_events(events)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].request_id, "req-1")
        self.assertEqual(observations[0].layer_id, 0)
        self.assertEqual(observations[0].token_index, 0)
        self.assertEqual(observations[0].expert_ids, ("2", "1"))
        self.assertEqual(observations[0].scores, (0.4, 0.6))

    def test_previous_token_predicts_next_token_experts(self) -> None:
        events = []
        for token_index in (0, 1):
            events.extend(
                events_from_record(
                    {
                        "event_type": "expert_route",
                        "request_id": "req-1",
                        "layer_id": 1,
                        "token_index": token_index,
                        "experts": [1, 3],
                        "scores": [0.7, 0.3],
                    },
                    source="moe.jsonl",
                )
            )
        observations = observations_from_events(events)
        predictor = RouterAwareExpertPredictor(
            ExpertPredictorConfig(
                top_k=2,
                previous_token_weight=1.0,
                hot_expert_weight=0.0,
                load_plan_weight=0.0,
                gpu_resident_bonus=0.0,
            )
        )

        predictions = predictor.predict(observations)
        first = next(item for item in predictions if item.source_token_index == 0)
        summary = summarize_predictions(predictions)

        self.assertEqual(set(first.predicted_experts), {"1", "3"})
        self.assertEqual(set(first.actual_experts), {"1", "3"})
        self.assertEqual(first.coverage, 1.0)
        self.assertEqual(summary["evaluated_prediction_count"], 1)
        self.assertEqual(summary["expert_prefetch_hit_rate"], 1.0)
        self.assertEqual(summary["expert_prefetch_waste_rate"], 0.0)

    def test_load_plan_gpu_residency_boosts_prediction_score(self) -> None:
        events = events_from_record(
            {
                "event_type": "expert_route",
                "request_id": "req-boost",
                "layer_id": 2,
                "token_index": 0,
                "experts": [4],
            },
            source="moe.jsonl",
        )
        observations = observations_from_events(events)
        load_plan = {
            "layer2:expert4": {
                "layer_id": "2",
                "expert_id": "4",
                "priority": "0.8",
                "target_tier": "gpu",
            }
        }
        predictor = RouterAwareExpertPredictor(
            ExpertPredictorConfig(
                top_k=1,
                previous_token_weight=0.5,
                hot_expert_weight=0.0,
                load_plan_weight=0.25,
                gpu_resident_bonus=0.1,
            )
        )

        prediction = predictor.predict(observations, load_plan=load_plan)[0]

        self.assertEqual(prediction.predicted_experts, ("4",))
        self.assertAlmostEqual(prediction.predicted_scores[0], 0.8)
        self.assertFalse(prediction.metadata["actual_available"])

    def test_history_window_predictor_uses_recent_non_previous_experts(self) -> None:
        events = []
        for token_index, experts in [(0, [7]), (1, [2]), (2, [7])]:
            events.extend(
                events_from_record(
                    {
                        "event_type": "expert_route",
                        "request_id": "req-window",
                        "layer_id": 0,
                        "token_index": token_index,
                        "experts": experts,
                    },
                    source="moe.jsonl",
                )
            )
        observations = observations_from_events(events)
        predictor = RouterAwareExpertPredictor(
            ExpertPredictorConfig(
                top_k=1,
                predictor_name="history_window",
                history_window=2,
                previous_token_weight=0.0,
                history_window_weight=1.0,
                transition_weight=0.0,
                hot_expert_weight=0.0,
                load_plan_weight=0.0,
                gpu_resident_bonus=0.0,
            )
        )

        predictions = predictor.predict(observations)
        prediction = next(item for item in predictions if item.source_token_index == 1)

        self.assertEqual(prediction.predictor_name, "history_window")
        self.assertEqual(prediction.window_size, 2)
        self.assertEqual(prediction.predicted_experts, ("7",))
        self.assertEqual(prediction.hit_experts, ("7",))
        self.assertEqual(prediction.coverage, 1.0)

    def test_plan_loader_preserves_zero_layer_and_expert(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            plan_path = tmp / "plan.csv"
            with plan_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["layer_id", "expert_id", "priority", "target_tier"])
                writer.writeheader()
                writer.writerow({"layer_id": 0, "expert_id": 0, "priority": 0.9, "target_tier": "gpu"})

            plan = load_expert_load_plan(plan_path)

            self.assertIn("layer0:expert0", plan)
            self.assertEqual(plan["layer0:expert0"]["target_tier"], "gpu")

    def test_outputs_are_writable(self) -> None:
        events = []
        for token_index, experts in [(0, [1, 2]), (1, [1, 2])]:
            events.extend(
                events_from_record(
                    {
                        "event_type": "expert_route",
                        "request_id": "req-out",
                        "layer_id": 1,
                        "token_index": token_index,
                        "experts": experts,
                    },
                    source="moe.jsonl",
                )
            )
        observations = observations_from_events(events)
        predictions = RouterAwareExpertPredictor(ExpertPredictorConfig(top_k=2)).predict(observations)

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            events_path = tmp / "moe_events.jsonl"
            predictions_path = tmp / "predictions.csv"
            hints_path = tmp / "hints.jsonl"
            report_path = tmp / "report.md"
            manifest_path = tmp / "manifest.json"
            write_events_jsonl(events, events_path)
            write_predictions_csv(predictions_path, predictions)
            write_prefetch_hints_jsonl(hints_path, predictions)
            args = Namespace(
                moe_events=str(events_path),
                expert_load_plan="",
                top_k=2,
                predictor_name="next_token",
                history_window=1,
                previous_token_weight=0.55,
                history_window_weight=0.20,
                transition_weight=0.0,
                hot_expert_weight=0.30,
                load_plan_weight=0.15,
                gpu_resident_bonus=0.10,
                min_score=0.0,
            )
            write_report(report_path, args, observations, predictions, predictions_path, hints_path)
            write_manifest(manifest_path, args, observations, predictions, predictions_path, hints_path, report_path)

            with predictions_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(hints[0]["action"], "expert_prefetch")
            self.assertEqual(hints[0]["metadata"]["predictor_name"], "next_token")
            self.assertIn("# MoE Expert Predictor Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["predictor_name"], "next_token")
            self.assertEqual(manifest["summary"]["expert_prefetch_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
