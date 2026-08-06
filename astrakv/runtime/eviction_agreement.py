"""Compare offline eviction decisions with normalized runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from astrakv.runtime.eviction import OfflineEvictionDecision, RuntimeEvictionEvent


@dataclass(frozen=True, slots=True)
class AgreementRow:
    run_id: str
    object_key: str
    object_level: str
    predicted_action: str
    actual_action: str
    classification: str
    tier_match: bool | None
    decision_time_ns: int | None
    runtime_time_ns: int | None
    lead_time_ns: int | None
    decision_bytes: int | None
    runtime_bytes: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "object_key": self.object_key,
            "object_level": self.object_level,
            "predicted_action": self.predicted_action,
            "actual_action": self.actual_action,
            "classification": self.classification,
            "tier_match": "" if self.tier_match is None else int(self.tier_match),
            "decision_time_ns": self.decision_time_ns,
            "runtime_time_ns": self.runtime_time_ns,
            "lead_time_ns": self.lead_time_ns,
            "decision_bytes": self.decision_bytes,
            "runtime_bytes": self.runtime_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    ground_truth_status: str
    reason: str
    rows: tuple[AgreementRow, ...]
    metrics: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "ground_truth_status": self.ground_truth_status,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "row_count": len(self.rows),
        }


def compare_eviction(
    decisions: Iterable[OfflineEvictionDecision],
    events: Iterable[RuntimeEvictionEvent],
    *,
    request_indexes: dict[str, int] | None = None,
    prediction_window_requests: int = 10,
    comparison_scope: str = "runtime",
) -> AgreementSummary:
    decision_list = [
        item
        for item in decisions
        if (
            item.predicts_eviction
            and item.run_id
            and item.request_id
            and item.object_key
            and not item.metadata.get("legacy_unlinked", False)
        )
    ]
    all_events = list(events)
    allowed_provenance = _allowed_provenance(comparison_scope)
    ground_truth = _deduplicate_events(
        item
        for item in all_events
        if (
            item.is_ground_truth
            and item.provenance in allowed_provenance
            and item.is_eviction
            and item.run_id
            and item.request_id
            and item.object_key
        )
    )
    if not decision_list:
        return AgreementSummary(
            "insufficient_ground_truth",
            "no offline eviction/offload decisions with run_id, request_id, and object keys",
            (),
            _empty_metrics(),
        )
    if not ground_truth:
        return AgreementSummary(
            "insufficient_ground_truth",
            f"no {comparison_scope} eviction events with comparable run_id, request_id, and object keys",
            (),
            {**_empty_metrics(), "offline_decision_count": len(decision_list), "runtime_event_count": 0},
        )

    indexes = request_indexes or {}
    event_by_key: dict[tuple[str, str, str], list[RuntimeEvictionEvent]] = {}
    for event in ground_truth:
        event_by_key.setdefault((event.run_id, event.object_level.value, event.object_key), []).append(event)
    for values in event_by_key.values():
        values.sort(key=lambda item: (item.arrival_index if item.arrival_index is not None else 10**18, item.timestamp_ns or 10**18))

    rows: list[AgreementRow] = []
    matched_event_ids: set[str] = set()
    for decision in decision_list:
        key = (decision.run_id, decision.object_level.value, decision.object_key)
        candidates = event_by_key.get(key, [])
        event = next((item for item in candidates if item.runtime_event_id not in matched_event_ids and _in_window(decision, item, indexes, prediction_window_requests)), None)
        if event is None:
            rows.append(_row_for_missing_event(decision))
            continue
        matched_event_ids.add(event.runtime_event_id)
        tier_match = _tier_match(decision.target_tier, event.tier_after)
        lead = None if decision.decision_time_ns is None or event.timestamp_ns is None else event.timestamp_ns - decision.decision_time_ns
        rows.append(AgreementRow(
            run_id=decision.run_id, object_key=decision.object_key, object_level=decision.object_level.value,
            predicted_action=decision.predicted_action, actual_action=event.actual_action, classification="tp",
            tier_match=tier_match, decision_time_ns=decision.decision_time_ns, runtime_time_ns=event.timestamp_ns,
            lead_time_ns=lead, decision_bytes=decision.bytes, runtime_bytes=event.bytes,
            metadata={"runtime_event_id": event.runtime_event_id, "provenance": event.provenance},
        ))
    for event in ground_truth:
        if event.runtime_event_id not in matched_event_ids:
            rows.append(AgreementRow(
                run_id=event.run_id, object_key=event.object_key, object_level=event.object_level.value,
                predicted_action="", actual_action=event.actual_action, classification="fn", tier_match=None,
                decision_time_ns=None, runtime_time_ns=event.timestamp_ns, lead_time_ns=None,
                decision_bytes=None, runtime_bytes=event.bytes,
                metadata={"runtime_event_id": event.runtime_event_id, "provenance": event.provenance},
            ))
    reason = (
        "structured runtime eviction events were comparable"
        if comparison_scope == "runtime"
        else "standalone mmap VM-PoC execution acknowledgements were comparable"
    )
    return AgreementSummary("valid", reason, tuple(rows), _metrics(rows, decision_list, ground_truth))


def _in_window(
    decision: OfflineEvictionDecision,
    event: RuntimeEvictionEvent,
    indexes: dict[str, int],
    width: int,
) -> bool:
    decision_index = decision.decision_index if decision.decision_index is not None else indexes.get(decision.request_id)
    event_index = event.arrival_index if event.arrival_index is not None else indexes.get(event.request_id)
    if decision_index is None or event_index is None:
        return True
    return decision_index < event_index <= decision_index + max(1, width)


def _row_for_missing_event(decision: OfflineEvictionDecision) -> AgreementRow:
    return AgreementRow(
        run_id=decision.run_id, object_key=decision.object_key, object_level=decision.object_level.value,
        predicted_action=decision.predicted_action, actual_action="", classification="fp", tier_match=None,
        decision_time_ns=decision.decision_time_ns, runtime_time_ns=None, lead_time_ns=None,
        decision_bytes=decision.bytes, runtime_bytes=None, metadata=dict(decision.metadata),
    )


def _tier_match(expected: str, actual: str) -> bool | None:
    if expected in {"", "unknown"} or actual in {"", "unknown"}:
        return None
    aliases = {"ssd": "disk", "disk": "disk"}
    return aliases.get(expected, expected) == aliases.get(actual, actual)


def _metrics(rows: list[AgreementRow], decisions: list[OfflineEvictionDecision], events: list[RuntimeEvictionEvent]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.classification == "tp")
    fp = sum(1 for row in rows if row.classification == "fp")
    fn = sum(1 for row in rows if row.classification == "fn")
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    tier_rows = [row for row in rows if row.tier_match is not None]
    matched_rows = [row for row in rows if row.classification == "tp"]
    bytes_total = 0
    bytes_matched = 0
    lead_times = [row.lead_time_ns for row in rows if row.lead_time_ns is not None]
    for row in rows:
        weight = max(row.decision_bytes or 0, row.runtime_bytes or 0, 1)
        bytes_total += weight
        if row.classification == "tp":
            bytes_matched += weight
    return {
        "offline_decision_count": len(decisions),
        "runtime_event_count": len(events),
        "comparable_row_count": len(rows),
        "object_coverage": tp / max(1, len(decisions)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_action_agreement": sum(
            1 for row in matched_rows if row.predicted_action == row.actual_action
        ) / max(1, len(matched_rows)),
        "tier_transition_accuracy": sum(1 for row in tier_rows if row.tier_match) / max(1, len(tier_rows)),
        "byte_weighted_agreement": bytes_matched / max(1, bytes_total),
        "lead_time_ns_mean": sum(lead_times) / len(lead_times) if lead_times else None,
    }


def _empty_metrics() -> dict[str, Any]:
    return {"offline_decision_count": 0, "runtime_event_count": 0, "comparable_row_count": 0}


def _deduplicate_events(events: Iterable[RuntimeEvictionEvent]) -> list[RuntimeEvictionEvent]:
    """Keep one record for repeated copies of the same runtime acknowledgement."""

    result: list[RuntimeEvictionEvent] = []
    seen: set[tuple[object, ...]] = set()
    for event in events:
        identity = (
            event.run_id,
            event.request_id,
            event.object_level.value,
            event.object_key,
            event.actual_action,
            event.arrival_index,
            event.timestamp_ns,
            event.status,
            event.provenance,
        )
        if identity not in seen:
            seen.add(identity)
            result.append(event)
    return result


def _allowed_provenance(comparison_scope: str) -> set[str]:
    if comparison_scope == "runtime":
        return {"runtime_structured"}
    if comparison_scope == "vm_poc":
        return {"vm_poc_execution"}
    raise ValueError("comparison_scope must be 'runtime' or 'vm_poc'")
