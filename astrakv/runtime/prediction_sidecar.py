"""Versioned sidecar artifacts for advisory runtime prediction input."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.eviction import ObjectLevel


PREDICTOR_CANDIDATE_REPORT_SCHEMA = "astrakv-predictor-candidate-report-v1"
SIDECAR_PREDICTION_SCHEMA = "astrakv-sidecar-prediction-v1"
FORBIDDEN_BENCHMARK_AWARE_FIELDS = frozenset(
    {"reuse_group", "phase", "reference_request_id", "task", "workload_type"}
)


def _required_text(record: dict[str, Any], field_name: str) -> str:
    value = str(record.get(field_name) or "")
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _required_float(record: dict[str, Any], field_name: str) -> float:
    try:
        return float(record[field_name])
    except KeyError as exc:
        raise ValueError(f"{field_name} is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _required_int(record: dict[str, Any], field_name: str) -> int:
    try:
        return int(record[field_name])
    except KeyError as exc:
        raise ValueError(f"{field_name} is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _validate_forbidden_fields(record: dict[str, Any]) -> None:
    present = sorted(field for field in FORBIDDEN_BENCHMARK_AWARE_FIELDS if field in record)
    if present:
        raise ValueError(
            "prediction artifacts must not expose benchmark-aware fields: "
            + ", ".join(present)
        )


@dataclass(frozen=True, slots=True)
class PredictorCandidateRecord:
    request_id: str
    candidate_object_id: str
    object_level: ObjectLevel
    predicted_class: str
    lead_distance_requests: int
    estimated_reusable_tokens: int
    estimated_kv_bytes: int
    confidence: float
    reason: str
    schema: str = PREDICTOR_CANDIDATE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREDICTOR_CANDIDATE_REPORT_SCHEMA:
            raise ValueError("unsupported predictor candidate schema")
        if self.predicted_class not in {
            "exact-next",
            "fixed-revisit",
            "structural-partial",
        }:
            raise ValueError("predicted_class is invalid")
        if self.lead_distance_requests < 0:
            raise ValueError("lead_distance_requests must be non-negative")
        if self.estimated_reusable_tokens < 0 or self.estimated_kv_bytes < 0:
            raise ValueError("estimated reuse values must be non-negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PredictorCandidateRecord":
        _validate_forbidden_fields(record)
        return cls(
            request_id=_required_text(record, "request_id"),
            candidate_object_id=_required_text(record, "candidate_object_id"),
            object_level=ObjectLevel(_required_text(record, "object_level")),
            predicted_class=_required_text(record, "predicted_class"),
            lead_distance_requests=_required_int(record, "lead_distance_requests"),
            estimated_reusable_tokens=_required_int(record, "estimated_reusable_tokens"),
            estimated_kv_bytes=_required_int(record, "estimated_kv_bytes"),
            confidence=_required_float(record, "confidence"),
            reason=_required_text(record, "reason"),
            schema=str(record.get("schema") or PREDICTOR_CANDIDATE_REPORT_SCHEMA),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "candidate_object_id": self.candidate_object_id,
            "object_level": self.object_level.value,
            "predicted_class": self.predicted_class,
            "lead_distance_requests": self.lead_distance_requests,
            "estimated_reusable_tokens": self.estimated_reusable_tokens,
            "estimated_kv_bytes": self.estimated_kv_bytes,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SidecarPrediction:
    run_id: str
    request_id: str
    candidate_object_id: str
    object_level: ObjectLevel
    score: float
    recommended_lead_time_ms: float
    confidence: float
    reason: str
    evidence_source: str
    predicted_class: str
    expires_at_ns: int
    schema: str = SIDECAR_PREDICTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIDECAR_PREDICTION_SCHEMA:
            raise ValueError("unsupported sidecar prediction schema")
        if self.predicted_class not in {
            "exact-next",
            "fixed-revisit",
            "structural-partial",
        }:
            raise ValueError("predicted_class is invalid")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.recommended_lead_time_ms < 0:
            raise ValueError("recommended_lead_time_ms must be non-negative")
        if self.expires_at_ns < 0:
            raise ValueError("expires_at_ns must be non-negative")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SidecarPrediction":
        _validate_forbidden_fields(record)
        return cls(
            schema=str(record.get("schema") or SIDECAR_PREDICTION_SCHEMA),
            run_id=_required_text(record, "run_id"),
            request_id=_required_text(record, "request_id"),
            candidate_object_id=_required_text(record, "candidate_object_id"),
            object_level=ObjectLevel(_required_text(record, "object_level")),
            score=_required_float(record, "score"),
            recommended_lead_time_ms=_required_float(record, "recommended_lead_time_ms"),
            confidence=_required_float(record, "confidence"),
            reason=_required_text(record, "reason"),
            evidence_source=_required_text(record, "evidence_source"),
            predicted_class=_required_text(record, "predicted_class"),
            expires_at_ns=_required_int(record, "expires_at_ns"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "candidate_object_id": self.candidate_object_id,
            "object_level": self.object_level.value,
            "score": self.score,
            "recommended_lead_time_ms": self.recommended_lead_time_ms,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence_source": self.evidence_source,
            "predicted_class": self.predicted_class,
            "expires_at_ns": self.expires_at_ns,
        }

    def is_expired(self, *, now_ns: int | None = None) -> bool:
        current = time.time_ns() if now_ns is None else int(now_ns)
        return self.expires_at_ns > 0 and current >= self.expires_at_ns


class PredictionSidecarIndex:
    """Index advisory predictions by request_id without widening runtime scope."""

    def __init__(self, predictions: Iterable[SidecarPrediction], *, run_id: str | None = None) -> None:
        self.run_id = run_id or ""
        self._predictions: dict[str, list[SidecarPrediction]] = {}
        for item in predictions:
            if self.run_id and item.run_id != self.run_id:
                continue
            self._predictions.setdefault(item.request_id, []).append(item)
        for request_id, rows in self._predictions.items():
            self._predictions[request_id] = sorted(
                rows,
                key=lambda entry: (
                    -entry.score,
                    -entry.confidence,
                    entry.candidate_object_id,
                    request_id,
                ),
            )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        run_id: str | None = None,
    ) -> "PredictionSidecarIndex":
        rows: list[SidecarPrediction] = []
        input_path = Path(path)
        if not input_path.is_file():
            raise ValueError(f"sidecar prediction JSONL not found: {input_path}")
        for line_number, raw_line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"sidecar prediction line {line_number} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"sidecar prediction line {line_number} must be an object")
            rows.append(SidecarPrediction.from_record(record))
        return cls(rows, run_id=run_id)

    def advisory_for(
        self,
        *,
        request_id: str,
        candidate_object_id: str,
        object_level: ObjectLevel,
        now_ns: int | None = None,
    ) -> SidecarPrediction | None:
        for entry in self._predictions.get(request_id, ()):
            if (
                entry.candidate_object_id == candidate_object_id
                and entry.object_level is object_level
                and not entry.is_expired(now_ns=now_ns)
            ):
                return entry
        return None

    def predictions_for_request(
        self,
        request_id: str,
        *,
        now_ns: int | None = None,
    ) -> tuple[SidecarPrediction, ...]:
        return tuple(
            entry
            for entry in self._predictions.get(request_id, ())
            if not entry.is_expired(now_ns=now_ns)
        )
