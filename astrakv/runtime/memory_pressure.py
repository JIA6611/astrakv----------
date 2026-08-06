"""Memory pressure controller helpers for archived AstraKV-W artifacts.

The controller converts benchmark rows, continuous memory samples, and unified
trace memory events into passive pressure decisions. It does not modify a
running vLLM/LMCache backend or move KV objects by itself.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from astrakv.scheduler.hints import SchedulerHint


OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "oom",
    "cannot allocate memory",
    "memory allocation failed",
    "allocation failed",
    "cuda error 2",
)


class MemoryPressureLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryPressureAction(str, Enum):
    OBSERVE = "observe"
    COLLECT_EVIDENCE = "collect_evidence"
    REDUCE_PREFETCH_BUDGET = "reduce_prefetch_budget"
    OFFLOAD_MORE = "offload_more"
    DROP_LOW_REUSE = "drop_low_reuse"
    REDUCE_BATCH_OR_CONTEXT = "reduce_batch_or_context"


@dataclass(frozen=True, slots=True)
class MemoryPressureConfig:
    gpu_capacity_mb: float = 0.0
    cpu_capacity_mb: float = 0.0
    gpu_medium_ratio: float = 0.70
    gpu_high_ratio: float = 0.85
    gpu_critical_ratio: float = 0.95
    cpu_medium_ratio: float = 0.70
    cpu_high_ratio: float = 0.85
    cpu_critical_ratio: float = 0.95
    disk_medium_mb: float = 512.0
    disk_high_mb: float = 4096.0
    disk_critical_mb: float = 16384.0
    error_medium_rate: float = 0.01
    error_high_rate: float = 0.05
    error_critical_rate: float = 0.20


@dataclass(frozen=True, slots=True)
class PressureSignal:
    name: str
    level: MemoryPressureLevel
    value: float | str
    threshold: float | str
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MemoryPressureObservation:
    source: str
    run_id: str = ""
    case: str = ""
    context_length: int = 0
    batch_size: int = 0
    total_requests: int = 0
    success_requests: int = 0
    error_rate: float | None = None
    oom_detected: bool = False
    gpu_memory_peak_mb: float | None = None
    cpu_memory_peak_mb: float | None = None
    gpu_util_peak_pct: float | None = None
    disk_read_delta_mb: float | None = None
    disk_write_delta_mb: float | None = None
    sample_count: int = 0
    missing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return self.case or self.run_id or self.source

    @property
    def has_pressure_metrics(self) -> bool:
        return any(
            value is not None
            for value in (
                self.error_rate,
                self.gpu_memory_peak_mb,
                self.cpu_memory_peak_mb,
                self.disk_read_delta_mb,
                self.disk_write_delta_mb,
            )
        ) or self.oom_detected


@dataclass(frozen=True, slots=True)
class MemoryPressureDecision:
    observation: MemoryPressureObservation
    level: MemoryPressureLevel
    score: float
    actions: tuple[MemoryPressureAction, ...]
    reason: str
    signals: tuple[PressureSignal, ...] = field(default_factory=tuple)

    @property
    def primary_action(self) -> MemoryPressureAction:
        return self.actions[0] if self.actions else MemoryPressureAction.OBSERVE

    def to_record(self) -> dict[str, Any]:
        observation = self.observation
        return {
            "source": observation.source,
            "run_id": observation.run_id,
            "case": observation.case,
            "context_length": observation.context_length,
            "batch_size": observation.batch_size,
            "level": self.level.value,
            "memory_pressure": self.score,
            "primary_action": self.primary_action.value,
            "actions": ";".join(action.value for action in self.actions),
            "reason": self.reason,
            "total_requests": observation.total_requests,
            "success_requests": observation.success_requests,
            "error_rate": "" if observation.error_rate is None else observation.error_rate,
            "oom_detected": observation.oom_detected,
            "gpu_memory_peak_mb": "" if observation.gpu_memory_peak_mb is None else observation.gpu_memory_peak_mb,
            "cpu_memory_peak_mb": "" if observation.cpu_memory_peak_mb is None else observation.cpu_memory_peak_mb,
            "gpu_util_peak_pct": "" if observation.gpu_util_peak_pct is None else observation.gpu_util_peak_pct,
            "disk_read_delta_mb": "" if observation.disk_read_delta_mb is None else observation.disk_read_delta_mb,
            "disk_write_delta_mb": "" if observation.disk_write_delta_mb is None else observation.disk_write_delta_mb,
            "sample_count": observation.sample_count,
            "missing": observation.missing,
            "signals": json.dumps([signal.to_record() for signal in self.signals], ensure_ascii=False),
        }

    def to_hints(self) -> list[SchedulerHint]:
        hints: list[SchedulerHint] = []
        for action in self.actions:
            hints.append(
                SchedulerHint(
                    request_id=self.observation.request_id,
                    action=action.value,
                    reason=self.reason,
                    priority=int(round(self.score * 100)),
                    metadata={
                        "object_type": "memory_pressure",
                        "level": self.level.value,
                        "memory_pressure": self.score,
                        "run_id": self.observation.run_id,
                        "case": self.observation.case,
                        "context_length": self.observation.context_length,
                        "batch_size": self.observation.batch_size,
                        "recommended_memory_pressure_arg": self.score,
                        "signals": [signal.to_record() for signal in self.signals],
                    },
                )
            )
        return hints


class MemoryPressureController:
    def __init__(self, config: MemoryPressureConfig | None = None) -> None:
        self.config = config or MemoryPressureConfig()

    def assess(self, observation: MemoryPressureObservation) -> MemoryPressureDecision:
        if observation.missing:
            signal = PressureSignal(
                name="missing_artifact",
                level=MemoryPressureLevel.UNKNOWN,
                value="missing",
                threshold="present artifact",
                reason="input artifact is missing",
            )
            return self._decision(
                observation,
                MemoryPressureLevel.UNKNOWN,
                (MemoryPressureAction.COLLECT_EVIDENCE,),
                "missing artifact; collect pressure evidence before making runtime claims",
                (signal,),
            )

        signals = tuple(self._signals(observation))
        if not observation.has_pressure_metrics:
            signal = PressureSignal(
                name="missing_metrics",
                level=MemoryPressureLevel.UNKNOWN,
                value="none",
                threshold="memory/error metric",
                reason="no GPU, CPU, disk, OOM, or error-rate metrics were available",
            )
            return self._decision(
                observation,
                MemoryPressureLevel.UNKNOWN,
                (MemoryPressureAction.COLLECT_EVIDENCE,),
                "no pressure metrics available; collect benchmark samples or trace memory events",
                (signal,),
            )

        level = max((signal.level for signal in signals), key=level_rank, default=MemoryPressureLevel.LOW)
        actions = actions_for_level(level, oom_detected=observation.oom_detected)
        reason = explain_decision(level, signals)
        return self._decision(observation, level, actions, reason, signals)

    def assess_many(self, observations: Iterable[MemoryPressureObservation]) -> list[MemoryPressureDecision]:
        return [self.assess(observation) for observation in observations]

    def _signals(self, observation: MemoryPressureObservation) -> list[PressureSignal]:
        cfg = self.config
        signals: list[PressureSignal] = []
        if observation.oom_detected:
            signals.append(
                PressureSignal(
                    name="oom",
                    level=MemoryPressureLevel.CRITICAL,
                    value=1.0,
                    threshold="0 OOM events",
                    reason="OOM-like failure text was observed",
                )
            )
        if observation.error_rate is not None:
            signals.append(
                signal_from_thresholds(
                    "error_rate",
                    observation.error_rate,
                    cfg.error_medium_rate,
                    cfg.error_high_rate,
                    cfg.error_critical_rate,
                    "request failure rate",
                )
            )
        if cfg.gpu_capacity_mb > 0 and observation.gpu_memory_peak_mb is not None:
            ratio = observation.gpu_memory_peak_mb / max(1.0, cfg.gpu_capacity_mb)
            signals.append(
                signal_from_thresholds(
                    "gpu_memory_ratio",
                    ratio,
                    cfg.gpu_medium_ratio,
                    cfg.gpu_high_ratio,
                    cfg.gpu_critical_ratio,
                    "GPU memory peak over configured capacity",
                )
            )
        if cfg.cpu_capacity_mb > 0 and observation.cpu_memory_peak_mb is not None:
            ratio = observation.cpu_memory_peak_mb / max(1.0, cfg.cpu_capacity_mb)
            signals.append(
                signal_from_thresholds(
                    "cpu_memory_ratio",
                    ratio,
                    cfg.cpu_medium_ratio,
                    cfg.cpu_high_ratio,
                    cfg.cpu_critical_ratio,
                    "CPU RSS peak over configured capacity",
                )
            )
        disk_total = sum_optional(observation.disk_read_delta_mb, observation.disk_write_delta_mb)
        if disk_total is not None:
            signals.append(
                signal_from_thresholds(
                    "disk_traffic_mb",
                    disk_total,
                    cfg.disk_medium_mb,
                    cfg.disk_high_mb,
                    cfg.disk_critical_mb,
                    "SSD read/write traffic for the case",
                )
            )
        if not signals:
            signals.append(
                PressureSignal(
                    name="available_metrics_below_threshold",
                    level=MemoryPressureLevel.LOW,
                    value=0.0,
                    threshold="configured medium thresholds",
                    reason="available pressure metrics did not cross configured thresholds",
                )
            )
        return signals

    def _decision(
        self,
        observation: MemoryPressureObservation,
        level: MemoryPressureLevel,
        actions: tuple[MemoryPressureAction, ...],
        reason: str,
        signals: tuple[PressureSignal, ...],
    ) -> MemoryPressureDecision:
        return MemoryPressureDecision(
            observation=observation,
            level=level,
            score=score_for_level(level),
            actions=actions,
            reason=reason,
            signals=signals,
        )


def signal_from_thresholds(
    name: str,
    value: float,
    medium: float,
    high: float,
    critical: float,
    reason: str,
) -> PressureSignal:
    if value >= critical:
        level = MemoryPressureLevel.CRITICAL
        threshold = critical
    elif value >= high:
        level = MemoryPressureLevel.HIGH
        threshold = high
    elif value >= medium:
        level = MemoryPressureLevel.MEDIUM
        threshold = medium
    else:
        level = MemoryPressureLevel.LOW
        threshold = medium
    return PressureSignal(name=name, level=level, value=value, threshold=threshold, reason=reason)


def actions_for_level(
    level: MemoryPressureLevel,
    *,
    oom_detected: bool = False,
) -> tuple[MemoryPressureAction, ...]:
    if level == MemoryPressureLevel.UNKNOWN:
        return (MemoryPressureAction.COLLECT_EVIDENCE,)
    if level == MemoryPressureLevel.LOW:
        return (MemoryPressureAction.OBSERVE,)
    if level == MemoryPressureLevel.MEDIUM:
        return (MemoryPressureAction.REDUCE_PREFETCH_BUDGET,)
    if level == MemoryPressureLevel.HIGH:
        return (
            MemoryPressureAction.OFFLOAD_MORE,
            MemoryPressureAction.REDUCE_PREFETCH_BUDGET,
            MemoryPressureAction.DROP_LOW_REUSE,
        )
    if oom_detected:
        return (
            MemoryPressureAction.REDUCE_BATCH_OR_CONTEXT,
            MemoryPressureAction.OFFLOAD_MORE,
            MemoryPressureAction.DROP_LOW_REUSE,
            MemoryPressureAction.REDUCE_PREFETCH_BUDGET,
        )
    return (
        MemoryPressureAction.OFFLOAD_MORE,
        MemoryPressureAction.DROP_LOW_REUSE,
        MemoryPressureAction.REDUCE_PREFETCH_BUDGET,
    )


def explain_decision(level: MemoryPressureLevel, signals: Iterable[PressureSignal]) -> str:
    strongest = sorted(signals, key=lambda signal: level_rank(signal.level), reverse=True)
    if not strongest:
        return f"{level.value}: no pressure signals"
    signal = strongest[0]
    return f"{level.value}: {signal.name}={signal.value} crossed {signal.threshold}; {signal.reason}"


def score_for_level(level: MemoryPressureLevel) -> float:
    return {
        MemoryPressureLevel.UNKNOWN: 0.0,
        MemoryPressureLevel.LOW: 0.15,
        MemoryPressureLevel.MEDIUM: 0.50,
        MemoryPressureLevel.HIGH: 0.75,
        MemoryPressureLevel.CRITICAL: 1.0,
    }[level]


def level_rank(level: MemoryPressureLevel) -> int:
    return {
        MemoryPressureLevel.UNKNOWN: -1,
        MemoryPressureLevel.LOW: 0,
        MemoryPressureLevel.MEDIUM: 1,
        MemoryPressureLevel.HIGH: 2,
        MemoryPressureLevel.CRITICAL: 3,
    }[level]


def observations_from_benchmark_csv(path: str | Path, run_id: str = "") -> list[MemoryPressureObservation]:
    csv_path = resolve_benchmark_path(path)
    if not csv_path.exists():
        return [missing_observation(csv_path, run_id=run_id)]
    rows = load_csv_rows(csv_path)
    observations: list[MemoryPressureObservation] = []
    for row in rows:
        total = as_int(first_present(row, "request_count", "total_requests"))
        success = as_int(first_present(row, "success_count", "success_requests"))
        explicit_error_rate = as_float(row.get("error_rate"))
        error_rate = explicit_error_rate if explicit_error_rate is not None else infer_error_rate(total, success, row.get("status"))
        case = str(first_present(row, "case", "case_id", "request_id") or csv_path.stem)
        observations.append(
            MemoryPressureObservation(
                source=str(csv_path),
                run_id=run_id or csv_path.parent.name or csv_path.stem,
                case=case,
                context_length=as_int(row.get("context_length")),
                batch_size=as_int(row.get("batch_size")),
                total_requests=total,
                success_requests=success,
                error_rate=error_rate,
                oom_detected=looks_oom(first_present(row, "errors", "error", "message", "status")),
                gpu_memory_peak_mb=as_float(first_present(row, "gpu_memory_peak_mb", "gpu_used_peak_mb")),
                cpu_memory_peak_mb=as_float(first_present(row, "cpu_memory_peak_mb", "cpu_rss_peak_mb")),
                gpu_util_peak_pct=as_float(first_present(row, "gpu_util_peak_pct", "gpu_utilization_peak_pct")),
                disk_read_delta_mb=as_float(first_present(row, "disk_read_delta_mb", "disk_read_mb")),
                disk_write_delta_mb=as_float(first_present(row, "disk_write_delta_mb", "disk_write_mb")),
                sample_count=as_int(row.get("sample_count")),
                metadata={"row": row},
            )
        )
    return observations


def observations_from_sample_csv(path: str | Path, run_id: str = "") -> list[MemoryPressureObservation]:
    sample_path = Path(path)
    if sample_path.is_dir():
        observations: list[MemoryPressureObservation] = []
        for child in sorted(sample_path.glob("*_samples.csv")):
            observations.extend(observations_from_sample_csv(child, run_id=run_id or sample_path.name))
        return observations
    if not sample_path.exists():
        return [missing_observation(sample_path, run_id=run_id)]
    rows = load_csv_rows(sample_path)
    case = infer_case_from_sample_path(sample_path)
    gpu_values = floats_from_rows(rows, "gpu_used_mb")
    cpu_values = floats_from_rows(rows, "cpu_rss_mb")
    gpu_util_values = floats_from_rows(rows, "gpu_util_pct")
    disk_read_values = floats_from_rows(rows, "disk_read_mb")
    disk_write_values = floats_from_rows(rows, "disk_write_mb")
    return [
        MemoryPressureObservation(
            source=str(sample_path),
            run_id=run_id or sample_path.parent.name,
            case=case,
            gpu_memory_peak_mb=max(gpu_values) if gpu_values else None,
            cpu_memory_peak_mb=max(cpu_values) if cpu_values else None,
            gpu_util_peak_pct=max(gpu_util_values) if gpu_util_values else None,
            disk_read_delta_mb=delta(disk_read_values),
            disk_write_delta_mb=delta(disk_write_values),
            sample_count=len(rows),
            metadata={"sample_file": str(sample_path)},
        )
    ]


def observations_from_trace_jsonl(path: str | Path, run_id: str = "") -> list[MemoryPressureObservation]:
    trace_path = Path(path)
    if not trace_path.exists():
        return [missing_observation(trace_path, run_id=run_id)]
    grouped: dict[str, dict[str, Any]] = {}
    for record in load_jsonl_records(trace_path):
        if str(record.get("category", "")) != "memory" and str(record.get("event_type", "")) != "memory_sample":
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        case = str(record.get("case") or record.get("request_id") or trace_path.stem)
        item = grouped.setdefault(
            case,
            {
                "source": str(trace_path),
                "run_id": run_id or trace_path.parent.name,
                "case": case,
                "gpu": [],
                "cpu": [],
                "gpu_util": [],
                "disk_read": [],
                "disk_write": [],
                "count": 0,
            },
        )
        item["count"] += 1
        append_float(item["gpu"], first_present(metadata, "gpu_used_mb", "gpu_memory_peak_mb"))
        append_float(item["cpu"], first_present(metadata, "cpu_rss_mb", "cpu_memory_peak_mb"))
        append_float(item["gpu_util"], metadata.get("gpu_util_pct"))
        append_float(item["disk_read"], first_present(metadata, "disk_read_mb", "disk_read_delta_mb"))
        append_float(item["disk_write"], first_present(metadata, "disk_write_mb", "disk_write_delta_mb"))
    observations: list[MemoryPressureObservation] = []
    for item in grouped.values():
        observations.append(
            MemoryPressureObservation(
                source=item["source"],
                run_id=item["run_id"],
                case=item["case"],
                gpu_memory_peak_mb=max(item["gpu"]) if item["gpu"] else None,
                cpu_memory_peak_mb=max(item["cpu"]) if item["cpu"] else None,
                gpu_util_peak_pct=max(item["gpu_util"]) if item["gpu_util"] else None,
                disk_read_delta_mb=delta(item["disk_read"]),
                disk_write_delta_mb=delta(item["disk_write"]),
                sample_count=item["count"],
            )
        )
    if not observations:
        return [
            MemoryPressureObservation(
                source=str(trace_path),
                run_id=run_id or trace_path.parent.name,
                case=trace_path.stem,
                sample_count=0,
                metadata={"note": "trace contained no memory events"},
            )
        ]
    return observations


def missing_observation(path: Path, run_id: str = "") -> MemoryPressureObservation:
    return MemoryPressureObservation(source=str(path), run_id=run_id, case=path.stem, missing=True)


def write_decisions_csv(path: str | Path, decisions: Iterable[MemoryPressureDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [decision.to_record() for decision in decisions]
    fieldnames = [
        "source",
        "run_id",
        "case",
        "context_length",
        "batch_size",
        "level",
        "memory_pressure",
        "primary_action",
        "actions",
        "reason",
        "total_requests",
        "success_requests",
        "error_rate",
        "oom_detected",
        "gpu_memory_peak_mb",
        "cpu_memory_peak_mb",
        "gpu_util_peak_pct",
        "disk_read_delta_mb",
        "disk_write_delta_mb",
        "sample_count",
        "missing",
        "signals",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pressure_hints_jsonl(path: str | Path, decisions: Iterable[MemoryPressureDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            for hint in decision.to_hints():
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


def summarize_decisions(decisions: Iterable[MemoryPressureDecision]) -> dict[str, Any]:
    decision_list = list(decisions)
    level_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    max_score = 0.0
    for decision in decision_list:
        level_counts[decision.level.value] = level_counts.get(decision.level.value, 0) + 1
        max_score = max(max_score, decision.score)
        for action in decision.actions:
            action_counts[action.value] = action_counts.get(action.value, 0) + 1
    return {
        "decision_count": len(decision_list),
        "max_memory_pressure": max_score,
        "level_counts": dict(sorted(level_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }


def resolve_benchmark_path(path: str | Path) -> Path:
    item = Path(path)
    return item / "benchmark_results.csv" if item.is_dir() else item


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def looks_oom(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def infer_error_rate(total: int, success: int, status: Any) -> float | None:
    if total > 0:
        return max(0, total - success) / max(1, total)
    if str(status or "").lower() not in {"", "ok", "success", "completed"}:
        return 1.0
    return None


def infer_case_from_sample_path(path: Path) -> str:
    name = path.stem
    return name[: -len("_samples")] if name.endswith("_samples") else name


def floats_from_rows(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        append_float(values, row.get(field))
    return values


def append_float(values: list[float], value: Any) -> None:
    parsed = as_float(value)
    if parsed is not None:
        values.append(parsed)


def delta(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def sum_optional(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", "None", "nan", "n/a"):
            return value
    return None


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    parsed = as_float(value)
    return int(parsed) if parsed is not None else 0
