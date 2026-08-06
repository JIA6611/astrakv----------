"""Router-aware MoE expert prediction helpers.

The predictor consumes already-recorded MoE route events and produces passive
expert prefetch hints for the next token. It does not call a model router,
prefetch weights, or modify a serving runtime.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from astrakv.moe.expert_loader import expert_key
from astrakv.runtime.moe_events import MoEExpertEvent
from astrakv.scheduler.hints import SchedulerHint


@dataclass(frozen=True, slots=True)
class ExpertRouteObservation:
    request_id: str
    layer_id: int | None
    token_index: int
    expert_ids: tuple[str, ...]
    scores: tuple[float, ...] = field(default_factory=tuple)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int | None, int]:
        return (self.request_id, self.layer_id, self.token_index)


@dataclass(frozen=True, slots=True)
class ExpertPredictorConfig:
    top_k: int = 2
    predictor_name: str = "next_token"
    history_window: int = 1
    previous_token_weight: float = 0.55
    history_window_weight: float = 0.20
    transition_weight: float = 0.0
    hot_expert_weight: float = 0.30
    load_plan_weight: float = 0.15
    gpu_resident_bonus: float = 0.10
    min_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ExpertPrediction:
    request_id: str
    layer_id: int | None
    source_token_index: int
    target_token_index: int
    predicted_experts: tuple[str, ...]
    predicted_scores: tuple[float, ...]
    predictor_name: str = "next_token"
    window_size: int = 1
    transition_score: float = 0.0
    actual_experts: tuple[str, ...] = field(default_factory=tuple)
    hit_experts: tuple[str, ...] = field(default_factory=tuple)
    wasted_experts: tuple[str, ...] = field(default_factory=tuple)
    missed_experts: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return len(self.hit_experts) / max(1, len(self.predicted_experts))

    @property
    def waste_rate(self) -> float:
        return len(self.wasted_experts) / max(1, len(self.predicted_experts))

    @property
    def coverage(self) -> float:
        return len(self.hit_experts) / max(1, len(self.actual_experts))

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_id": "" if self.layer_id is None else self.layer_id,
            "source_token_index": self.source_token_index,
            "target_token_index": self.target_token_index,
            "predictor_name": self.predictor_name,
            "window_size": self.window_size,
            "transition_score": round(self.transition_score, 6),
            "predicted_experts": ",".join(self.predicted_experts),
            "predicted_scores": ",".join(f"{score:.6f}" for score in self.predicted_scores),
            "actual_experts": ",".join(self.actual_experts),
            "hit_experts": ",".join(self.hit_experts),
            "wasted_experts": ",".join(self.wasted_experts),
            "missed_experts": ",".join(self.missed_experts),
            "hit_rate": round(self.hit_rate, 6),
            "waste_rate": round(self.waste_rate, 6),
            "coverage": round(self.coverage, 6),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_hints(self) -> list[SchedulerHint]:
        hints: list[SchedulerHint] = []
        for expert_id, score in zip(self.predicted_experts, self.predicted_scores):
            hints.append(
                SchedulerHint(
                    request_id=self.request_id,
                    action="expert_prefetch",
                    reason=self.reason,
                    priority=int(round(score * 100)),
                    metadata={
                        "object_type": "moe_expert",
                        "layer_id": self.layer_id,
                        "expert_id": expert_id,
                        "predictor_name": self.predictor_name,
                        "window_size": self.window_size,
                        "transition_score": self.transition_score,
                        "source_token_index": self.source_token_index,
                        "target_token_index": self.target_token_index,
                        "prediction_score": score,
                        "actual_hit": expert_id in self.hit_experts,
                        **dict(self.metadata),
                    },
                )
            )
        return hints


class RouterAwareExpertPredictor:
    def __init__(self, config: ExpertPredictorConfig | None = None) -> None:
        self.config = config or ExpertPredictorConfig()

    def predict(
        self,
        observations: Iterable[ExpertRouteObservation],
        *,
        load_plan: dict[str, dict[str, Any]] | None = None,
    ) -> list[ExpertPrediction]:
        observation_list = sorted(observations, key=lambda item: (item.request_id, item.layer_id or -1, item.token_index))
        by_key = {observation.key: observation for observation in observation_list}
        by_stream = observations_by_stream(observation_list)
        hot_by_layer = layer_hotness(observation_list)
        transition_model = build_transition_model(by_stream)
        load_plan_by_key = load_plan or {}

        predictions: list[ExpertPrediction] = []
        for observation in observation_list:
            target_token_index = observation.token_index + 1
            actual = by_key.get((observation.request_id, observation.layer_id, target_token_index))
            stream = by_stream.get((observation.request_id, observation.layer_id), [])
            history = history_for_observation(
                observation,
                stream,
                self._effective_history_window(),
            )
            transition_scores = transition_scores_for_observation(observation, transition_model)
            candidate_scores = self._score_candidates(
                observation,
                history,
                hot_by_layer,
                load_plan_by_key,
                transition_scores,
            )
            selected = [
                (expert_id, score)
                for expert_id, score in sorted(candidate_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
                if score >= self.config.min_score
            ][: self.config.top_k]
            if not selected:
                continue
            predicted_experts = tuple(expert_id for expert_id, _ in selected)
            predicted_scores = tuple(score for _, score in selected)
            actual_experts = actual.expert_ids if actual is not None else tuple()
            hit_experts = tuple(expert_id for expert_id in predicted_experts if expert_id in actual_experts)
            wasted_experts = (
                tuple(expert_id for expert_id in predicted_experts if expert_id not in actual_experts)
                if actual is not None
                else tuple()
            )
            missed_experts = tuple(expert_id for expert_id in actual_experts if expert_id not in predicted_experts)
            selected_transition_score = average(
                transition_scores.get(expert_id, 0.0)
                for expert_id in predicted_experts
            )
            predictions.append(
                ExpertPrediction(
                    request_id=observation.request_id,
                    layer_id=observation.layer_id,
                    source_token_index=observation.token_index,
                    target_token_index=target_token_index,
                    predicted_experts=predicted_experts,
                    predicted_scores=predicted_scores,
                    predictor_name=self.config.predictor_name,
                    window_size=self._effective_history_window(),
                    transition_score=selected_transition_score,
                    actual_experts=actual_experts,
                    hit_experts=hit_experts,
                    wasted_experts=wasted_experts,
                    missed_experts=missed_experts,
                    reason=self._reason(),
                    metadata={
                        "source": observation.source,
                        "actual_available": actual is not None,
                        "history_tokens": [item.token_index for item in history],
                    },
                )
            )
        return predictions

    def _score_candidates(
        self,
        observation: ExpertRouteObservation,
        history: list[ExpertRouteObservation],
        hot_by_layer: dict[int | None, dict[str, float]],
        load_plan: dict[str, dict[str, Any]],
        transition_scores: dict[str, float],
    ) -> dict[str, float]:
        cfg = self.config
        scores: dict[str, float] = {}
        current_key = observation.key
        for item in history:
            distance = max(0, observation.token_index - item.token_index)
            recency = 1.0 / (distance + 1)
            weight = cfg.previous_token_weight if item.key == current_key else cfg.history_window_weight * recency
            for expert_id in item.expert_ids:
                scores[expert_id] = scores.get(expert_id, 0.0) + weight

        if not history:
            for expert_id in observation.expert_ids:
                scores[expert_id] = scores.get(expert_id, 0.0) + cfg.previous_token_weight

        for expert_id, transition_score in transition_scores.items():
            scores[expert_id] = scores.get(expert_id, 0.0) + cfg.transition_weight * transition_score

        for expert_id, share in hot_by_layer.get(observation.layer_id, {}).items():
            scores[expert_id] = scores.get(expert_id, 0.0) + cfg.hot_expert_weight * share

        for key in (expert_key(observation.layer_id, expert_id) for expert_id in set(scores)):
            entry = load_plan.get(key)
            if not entry:
                continue
            expert_id = str(entry.get("expert_id", ""))
            priority = as_float(entry.get("priority"))
            scores[expert_id] = scores.get(expert_id, 0.0) + cfg.load_plan_weight * priority
            if str(entry.get("target_tier", "")) == "gpu":
                scores[expert_id] += cfg.gpu_resident_bonus
        return {expert_id: clamp(score) for expert_id, score in scores.items()}

    def _effective_history_window(self) -> int:
        if self.config.predictor_name == "next_token":
            return 1
        return max(1, int(self.config.history_window))

    def _reason(self) -> str:
        if self.config.predictor_name == "history_window":
            return "history-window experts plus transition statistics, layer hotness, and load-plan residency"
        if self.config.predictor_name == "profile_guided":
            return "profile-guided experts from route history, layer hotness, transition statistics, and load-plan residency"
        return "previous-token experts plus layer hotness and load-plan residency"


def observations_from_events(events: Iterable[MoEExpertEvent]) -> list[ExpertRouteObservation]:
    grouped: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    for event in events:
        if event.event_type not in {"expert_route", "expert_selected", "expert_prefetch"}:
            continue
        if event.token_index is None or not event.expert_id:
            continue
        key = (event.request_id, event.layer_id, event.token_index)
        item = grouped.setdefault(
            key,
            {
                "request_id": event.request_id,
                "layer_id": event.layer_id,
                "token_index": event.token_index,
                "experts": [],
                "source": event.source,
            },
        )
        item["experts"].append(
            (
                event.expert_rank if event.expert_rank is not None else len(item["experts"]),
                event.expert_id,
                event.score if event.score is not None else 0.0,
            )
        )

    observations: list[ExpertRouteObservation] = []
    for item in grouped.values():
        ranked = sorted(item["experts"], key=lambda value: value[0])
        observations.append(
            ExpertRouteObservation(
                request_id=item["request_id"],
                layer_id=item["layer_id"],
                token_index=item["token_index"],
                expert_ids=tuple(expert_id for _, expert_id, _ in ranked),
                scores=tuple(score for _, _, score in ranked),
                source=item["source"],
            )
        )
    return observations


def load_expert_load_plan(path: str | Path) -> dict[str, dict[str, Any]]:
    plan_path = Path(path)
    if not plan_path.exists():
        return {}
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        layer_id = _optional_int(row.get("layer_id"))
        expert_id = str(row.get("expert_id", ""))
        if not expert_id:
            continue
        output[expert_key(layer_id, expert_id)] = row
    return output


def write_predictions_csv(path: str | Path, predictions: Iterable[ExpertPrediction]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [prediction.to_record() for prediction in predictions]
    fieldnames = [
        "request_id",
        "layer_id",
        "source_token_index",
        "target_token_index",
        "predictor_name",
        "window_size",
        "transition_score",
        "predicted_experts",
        "predicted_scores",
        "actual_experts",
        "hit_experts",
        "wasted_experts",
        "missed_experts",
        "hit_rate",
        "waste_rate",
        "coverage",
        "reason",
        "metadata",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_prefetch_hints_jsonl(path: str | Path, predictions: Iterable[ExpertPrediction]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            for hint in prediction.to_hints():
                handle.write(
                    json.dumps(
                        {
                            "schema": "astra-scheduler-hint-v1",
                            "request_id": hint.request_id,
                            "action": hint.action,
                            "reason": hint.reason,
                            "priority": hint.priority,
                            "metadata": hint.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def summarize_predictions(predictions: Iterable[ExpertPrediction]) -> dict[str, Any]:
    prediction_list = list(predictions)
    predicted_total = 0
    evaluated_predicted_total = 0
    actual_total = 0
    hit_total = 0
    waste_total = 0
    missed_total = 0
    evaluated = 0
    for prediction in prediction_list:
        predicted_total += len(prediction.predicted_experts)
        if prediction.metadata.get("actual_available"):
            evaluated += 1
            evaluated_predicted_total += len(prediction.predicted_experts)
            actual_total += len(prediction.actual_experts)
            hit_total += len(prediction.hit_experts)
            waste_total += len(prediction.wasted_experts)
            missed_total += len(prediction.missed_experts)
    return {
        "prediction_count": len(prediction_list),
        "evaluated_prediction_count": evaluated,
        "predicted_expert_total": predicted_total,
        "evaluated_predicted_expert_total": evaluated_predicted_total,
        "actual_expert_total": actual_total,
        "hit_expert_total": hit_total,
        "wasted_expert_total": waste_total,
        "missed_expert_total": missed_total,
        "expert_prefetch_hit_rate": hit_total / max(1, evaluated_predicted_total),
        "expert_prefetch_waste_rate": waste_total / max(1, evaluated_predicted_total),
        "expert_coverage_rate": hit_total / max(1, actual_total),
    }


def layer_hotness(observations: Iterable[ExpertRouteObservation]) -> dict[int | None, dict[str, float]]:
    counts: dict[int | None, dict[str, int]] = {}
    for observation in observations:
        layer_counts = counts.setdefault(observation.layer_id, {})
        for expert_id in observation.expert_ids:
            layer_counts[expert_id] = layer_counts.get(expert_id, 0) + 1
    output: dict[int | None, dict[str, float]] = {}
    for layer_id, layer_counts in counts.items():
        total = sum(layer_counts.values())
        output[layer_id] = {
            expert_id: count / max(1, total)
            for expert_id, count in layer_counts.items()
        }
    return output


def observations_by_stream(
    observations: Iterable[ExpertRouteObservation],
) -> dict[tuple[str, int | None], list[ExpertRouteObservation]]:
    streams: dict[tuple[str, int | None], list[ExpertRouteObservation]] = {}
    for observation in observations:
        streams.setdefault((observation.request_id, observation.layer_id), []).append(observation)
    for key, stream in streams.items():
        streams[key] = sorted(stream, key=lambda item: item.token_index)
    return streams


def history_for_observation(
    observation: ExpertRouteObservation,
    stream: list[ExpertRouteObservation],
    window_size: int,
) -> list[ExpertRouteObservation]:
    lower_bound = observation.token_index - max(1, window_size) + 1
    return [
        item
        for item in stream
        if lower_bound <= item.token_index <= observation.token_index
    ]


def build_transition_model(
    streams: dict[tuple[str, int | None], list[ExpertRouteObservation]],
) -> dict[tuple[int | None, str], dict[str, float]]:
    counts: dict[tuple[int | None, str], dict[str, int]] = {}
    for (_request_id, layer_id), stream in streams.items():
        for current, nxt in zip(stream, stream[1:]):
            if nxt.token_index != current.token_index + 1:
                continue
            for source_expert in current.expert_ids:
                next_counts = counts.setdefault((layer_id, source_expert), {})
                for target_expert in nxt.expert_ids:
                    next_counts[target_expert] = next_counts.get(target_expert, 0) + 1

    model: dict[tuple[int | None, str], dict[str, float]] = {}
    for key, next_counts in counts.items():
        total = sum(next_counts.values())
        model[key] = {
            expert_id: count / max(1, total)
            for expert_id, count in next_counts.items()
        }
    return model


def transition_scores_for_observation(
    observation: ExpertRouteObservation,
    model: dict[tuple[int | None, str], dict[str, float]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not observation.expert_ids:
        return scores
    for source_expert in observation.expert_ids:
        for target_expert, probability in model.get((observation.layer_id, source_expert), {}).items():
            scores[target_expert] = scores.get(target_expert, 0.0) + probability / len(observation.expert_ids)
    return scores


def average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def as_float(value: Any) -> float:
    if value in (None, "", "None", "nan", "n/a"):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
