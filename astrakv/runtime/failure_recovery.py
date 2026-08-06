"""Failure recovery and fallback planning helpers.

The module reads archived runtime artifacts and emits passive recovery hints.
It does not retry requests, move memory, or modify serving-runtime behavior.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from astrakv.scheduler.hints import SchedulerHint


FAILURE_SCHEMA_VERSION = "astra-failure-event-v1"
RECOVERY_SCHEMA_VERSION = "astra-recovery-decision-v1"


class RecoveryAction(str, Enum):
    FALLBACK_BASELINE = "fallback_baseline"
    DISABLE_PREFETCH = "disable_prefetch"
    USE_CPU_TIER = "use_cpu_tier"
    USE_DISK_TIER = "use_disk_tier"
    RECOMPUTE = "recompute"
    REDUCE_WORKLOAD = "reduce_workload"
    SKIP_OBJECT = "skip_object"
    COLLECT_EVIDENCE = "collect_evidence"
    ALERT = "alert"


@dataclass(frozen=True, slots=True)
class FailureEvent:
    failure_type: str
    component: str
    source: str
    source_type: str
    severity: str = "warning"
    event_id: str = field(default_factory=lambda: uuid4().hex)
    request_id: str = ""
    object_id: str = ""
    case: str = ""
    status: str = ""
    error: str = ""
    raw_event_type: str = ""
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "failure_type": self.failure_type,
            "component": self.component,
            "severity": self.severity,
            "source": self.source,
            "source_type": self.source_type,
            "request_id": self.request_id,
            "object_id": self.object_id,
            "case": self.case,
            "status": self.status,
            "error": self.error,
            "raw_event_type": self.raw_event_type,
            "line_number": self.line_number,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    event_id: str
    failure_type: str
    component: str
    action: RecoveryAction
    priority: int
    reason: str
    target_mode: str = "safe_baseline"
    request_id: str = ""
    object_id: str = ""
    case: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_SCHEMA_VERSION,
            "event_id": self.event_id,
            "failure_type": self.failure_type,
            "component": self.component,
            "action": self.action.value,
            "priority": self.priority,
            "target_mode": self.target_mode,
            "request_id": self.request_id,
            "object_id": self.object_id,
            "case": self.case,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_hint(self) -> SchedulerHint:
        return SchedulerHint(
            request_id=self.request_id or self.case or self.object_id or self.event_id,
            action=self.action.value,
            reason=self.reason,
            priority=self.priority,
            metadata={
                "schema": RECOVERY_SCHEMA_VERSION,
                "event_id": self.event_id,
                "failure_type": self.failure_type,
                "component": self.component,
                "target_mode": self.target_mode,
                "object_id": self.object_id,
                "case": self.case,
                **dict(self.metadata),
            },
        )


class FailureRecoveryPolicy:
    def decide(self, event: FailureEvent) -> RecoveryDecision:
        failure_type = event.failure_type
        if failure_type in {"memory_oom", "allocation_failed"}:
            return self._decision(
                event,
                RecoveryAction.REDUCE_WORKLOAD,
                100,
                "memory allocation failed; reduce batch/context and fall back to safe baseline",
            )
        if failure_type in {"endpoint_failure", "timeout", "http_error"}:
            return self._decision(
                event,
                RecoveryAction.FALLBACK_BASELINE,
                95,
                "endpoint path failed; use baseline request path and record the cause",
            )
        if failure_type in {"prefetch_failed", "prefetch_waste"}:
            return self._decision(
                event,
                RecoveryAction.DISABLE_PREFETCH,
                85,
                "prefetch failed or wasted work; disable prefetch for this object/workload",
                target_mode="no_prefetch",
            )
        if failure_type in {"cache_load_failed", "cache_missing", "tier_load_failed"}:
            return self._decision(
                event,
                RecoveryAction.RECOMPUTE,
                80,
                "cache/tier load failed; recompute or use baseline runtime path",
            )
        if failure_type in {"profile_missing", "artifact_missing"}:
            return self._decision(
                event,
                RecoveryAction.COLLECT_EVIDENCE,
                45,
                "required profile or artifact is missing; continue safely and collect evidence",
                target_mode="observability_only",
            )
        if failure_type == "scheduler_drop":
            return self._decision(
                event,
                RecoveryAction.SKIP_OBJECT,
                55,
                "scheduler already requested object drop; skip the object safely",
                target_mode="advisory_skip",
            )
        return self._decision(
            event,
            RecoveryAction.ALERT,
            50,
            "unclassified failure; keep baseline path available and alert operator",
        )

    def _decision(
        self,
        event: FailureEvent,
        action: RecoveryAction,
        priority: int,
        reason: str,
        *,
        target_mode: str = "safe_baseline",
    ) -> RecoveryDecision:
        return RecoveryDecision(
            event_id=event.event_id,
            failure_type=event.failure_type,
            component=event.component,
            action=action,
            priority=priority,
            target_mode=target_mode,
            request_id=event.request_id,
            object_id=event.object_id,
            case=event.case,
            reason=reason,
            metadata={
                "severity": event.severity,
                "source": event.source,
                "source_type": event.source_type,
                "status": event.status,
                "raw_event_type": event.raw_event_type,
                "error": event.error,
                **dict(event.metadata),
            },
        )


def parse_benchmark_results(path: str | Path) -> list[FailureEvent]:
    csv_path = Path(path)
    if not csv_path.exists():
        return [missing_event(csv_path, "benchmark")]
    events: list[FailureEvent] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            status = str(row.get("status", ""))
            request_count = as_int(row.get("request_count"))
            success_count = as_int(row.get("success_count"))
            error = str(row.get("errors", ""))
            if status.lower() == "ok" and success_count >= request_count:
                continue
            events.append(
                FailureEvent(
                    failure_type=classify_failure(error or status),
                    component="benchmark",
                    source=str(csv_path),
                    source_type="benchmark",
                    severity=severity_for_text(error or status),
                    case=str(row.get("case", "")),
                    status=status,
                    error=error,
                    raw_event_type="benchmark_case",
                    line_number=line_number,
                    metadata={
                        "request_count": request_count,
                        "success_count": success_count,
                        "backend": row.get("backend", ""),
                        "model": row.get("model", ""),
                    },
                )
            )
    return events


def parse_request_results(path: str | Path) -> list[FailureEvent]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [missing_event(jsonl_path, "request_results")]
    events: list[FailureEvent] = []
    for line_number, record in iter_jsonl(jsonl_path):
        status = str(record.get("status", ""))
        error = str(record.get("error", ""))
        if status.lower() == "ok" and not error:
            continue
        events.append(
            FailureEvent(
                failure_type=classify_failure(error or status),
                component="endpoint",
                source=str(jsonl_path),
                source_type="request_results",
                severity=severity_for_text(error or status),
                request_id=str(record.get("request_id", "")),
                case=str(record.get("case", "")),
                status=status,
                error=error,
                raw_event_type="request_result",
                line_number=line_number,
                metadata={"backend": record.get("backend", ""), "model": record.get("model", "")},
            )
        )
    return events


def parse_prefetch_events(path: str | Path) -> list[FailureEvent]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [missing_event(jsonl_path, "prefetch")]
    events: list[FailureEvent] = []
    for line_number, record in iter_jsonl(jsonl_path):
        event_type = str(record.get("event_type", ""))
        status = str(record.get("status", ""))
        message = str(record.get("message", ""))
        if event_type not in {"prefetch_failed", "prefetch_waste"} and status.lower() not in {"failed", "error"}:
            continue
        failure_type = "prefetch_waste" if event_type == "prefetch_waste" else "prefetch_failed"
        events.append(
            FailureEvent(
                failure_type=failure_type,
                component="prefetch",
                source=str(jsonl_path),
                source_type="prefetch_events",
                severity="warning" if failure_type == "prefetch_waste" else "error",
                request_id=str(record.get("request_id", "")),
                object_id=str(record.get("chunk_id", "")),
                case=str(record.get("case", "")),
                status=status,
                error=message,
                raw_event_type=event_type,
                line_number=line_number,
                metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            )
        )
    return events


def parse_cache_events(path: str | Path) -> list[FailureEvent]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [missing_event(jsonl_path, "cache")]
    events: list[FailureEvent] = []
    for line_number, record in iter_jsonl(jsonl_path):
        event_type = str(record.get("event_type", ""))
        status = str(record.get("status", ""))
        error = str(record.get("error", ""))
        lowered = f"{event_type} {status} {error}".lower()
        if not any(token in lowered for token in ("missing", "failed", "fail", "partial_or_failed")):
            continue
        events.append(
            FailureEvent(
                failure_type=classify_cache_failure(event_type, status, error),
                component="cache",
                source=str(jsonl_path),
                source_type="cache_events",
                severity=severity_for_text(lowered),
                request_id=str(record.get("request_id", "")),
                object_id=str(record.get("chunk_id") or record.get("cache_key") or ""),
                status=status,
                error=error or str(record.get("raw_line", "")),
                raw_event_type=event_type,
                line_number=line_number,
                metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            )
        )
    return events


def parse_scheduler_hints(path: str | Path) -> list[FailureEvent]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [missing_event(jsonl_path, "scheduler_hints")]
    events: list[FailureEvent] = []
    for line_number, record in iter_jsonl(jsonl_path):
        action = str(record.get("action", ""))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if action not in {"drop", "defer"} and str(metadata.get("status", "")).lower() not in {"failed", "error"}:
            continue
        failure_type = "scheduler_drop" if action == "drop" else classify_failure(str(metadata.get("error", action)))
        events.append(
            FailureEvent(
                failure_type=failure_type,
                component="scheduler",
                source=str(jsonl_path),
                source_type="scheduler_hints",
                severity="info" if action in {"drop", "defer"} else "warning",
                request_id=str(record.get("request_id", "")),
                object_id=str(metadata.get("chunk_id") or metadata.get("expert_id") or ""),
                status=str(metadata.get("status", action)),
                error=str(metadata.get("error", "")),
                raw_event_type=action,
                line_number=line_number,
                metadata=metadata,
            )
        )
    return events


def decide_recovery(events: Iterable[FailureEvent], policy: FailureRecoveryPolicy | None = None) -> list[RecoveryDecision]:
    active_policy = policy or FailureRecoveryPolicy()
    return [active_policy.decide(event) for event in events]


def summarize_failure_recovery(events: Iterable[FailureEvent], decisions: Iterable[RecoveryDecision]) -> dict[str, Any]:
    event_list = list(events)
    decision_list = list(decisions)
    return {
        "failure_count": len(event_list),
        "decision_count": len(decision_list),
        "failure_type_counts": count_by(event_list, "failure_type"),
        "component_counts": count_by(event_list, "component"),
        "severity_counts": count_by(event_list, "severity"),
        "recovery_action_counts": count_decisions(decision_list),
        "critical_or_error_count": sum(1 for event in event_list if event.severity in {"critical", "error"}),
    }


def write_failure_events_jsonl(path: str | Path, events: Iterable[FailureEvent]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def write_recovery_decisions_csv(path: str | Path, decisions: Iterable[RecoveryDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [decision.to_record() for decision in decisions]
    fieldnames = [
        "schema",
        "event_id",
        "failure_type",
        "component",
        "action",
        "priority",
        "target_mode",
        "request_id",
        "object_id",
        "case",
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


def write_fallback_hints_jsonl(path: str | Path, decisions: Iterable[RecoveryDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            hint = decision.to_hint()
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


def classify_failure(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("out of memory", "oom", "cuda out of memory", "cannot allocate memory")):
        return "memory_oom"
    if any(token in lowered for token in ("allocation failed", "memory allocation failed")):
        return "allocation_failed"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "http" in lowered:
        return "http_error"
    if any(token in lowered for token in ("connection", "refused", "unreachable", "endpoint")):
        return "endpoint_failure"
    if "profile" in lowered and "missing" in lowered:
        return "profile_missing"
    if "missing" in lowered or "not found" in lowered:
        return "artifact_missing"
    if "prefetch" in lowered and ("fail" in lowered or "waste" in lowered):
        return "prefetch_failed"
    if "cache" in lowered and ("fail" in lowered or "retrieve" in lowered or "load" in lowered):
        return "cache_load_failed"
    return "runtime_failure"


def classify_cache_failure(event_type: str, status: str, error: str) -> str:
    lowered = f"{event_type} {status} {error}".lower()
    if "missing" in lowered:
        return "cache_missing"
    if "load" in lowered or "retrieve" in lowered:
        return "cache_load_failed"
    if "tier" in lowered:
        return "tier_load_failed"
    return classify_failure(lowered)


def severity_for_text(text: str) -> str:
    failure_type = classify_failure(text)
    if failure_type in {"memory_oom", "allocation_failed"}:
        return "critical"
    if failure_type in {"endpoint_failure", "timeout", "http_error", "cache_load_failed"}:
        return "error"
    if failure_type in {"artifact_missing", "profile_missing"}:
        return "warning"
    return "warning"


def missing_event(path: Path, source_type: str) -> FailureEvent:
    return FailureEvent(
        failure_type="artifact_missing",
        component=source_type,
        source=str(path),
        source_type=source_type,
        severity="warning",
        status="missing",
        error=f"artifact not found: {path}",
        raw_event_type="artifact_missing",
        metadata={"path": str(path)},
    )


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.lstrip("\ufeff")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "event_type": "parse_error",
                    "status": "failed",
                    "error": str(exc),
                    "raw_line": line.strip(),
                }
                continue
            if isinstance(record, dict):
                yield line_number, record


def count_by(events: list[FailureEvent], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        value = str(getattr(event, field_name))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def count_decisions(decisions: list[RecoveryDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        value = decision.action.value
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
