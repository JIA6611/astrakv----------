"""Unified trace schema for AstraKV-W runtime artifacts.

The trace layer normalizes existing P0 artifacts into one schema. It keeps raw
cache events, prefetch events, and metric samples intact while producing a
common JSONL stream for profile-guided policies and reports.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


TRACE_SCHEMA_VERSION = "astra-trace-v1"

REQUIRED_FIELDS = (
    "schema",
    "event_id",
    "event_type",
    "category",
    "source",
    "status",
    "metadata",
)

KNOWN_CATEGORIES = {
    "benchmark",
    "error",
    "kv",
    "memory",
    "placement",
    "prefetch",
    "request",
    "runtime",
    "unknown",
}


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event_type: str
    category: str
    source: str
    status: str = "observed"
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = ""
    run_id: str = ""
    request_id: str = ""
    case: str = ""
    attribution_mode: str = "unattributed"
    backend: str = ""
    model: str = ""
    chunk_id: str = ""
    cache_key: str = ""
    tier: str = "unknown"
    bytes: int | None = None
    latency_ms: float | None = None
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category,
            "source": self.source,
            "status": self.status,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "case": self.case,
            "attribution_mode": self.attribution_mode,
            "backend": self.backend,
            "model": self.model,
            "chunk_id": self.chunk_id,
            "cache_key": self.cache_key,
            "tier": self.tier,
            "bytes": self.bytes,
            "latency_ms": self.latency_ms,
            "line_number": self.line_number,
            "metadata": dict(self.metadata),
        }


def trace_from_cache_record(record: dict[str, Any]) -> TraceEvent:
    metadata = _dict(record.get("metadata"))
    event_type = str(record.get("event_type", "cache_event"))
    return TraceEvent(
        event_type=event_type,
        category=category_for_event_type(event_type),
        source=str(record.get("source", "")),
        status=str(record.get("status", "observed")),
        timestamp=str(record.get("start_time") or record.get("end_time") or ""),
        run_id=str(record.get("run_id") or metadata.get("run_id") or ""),
        request_id=str(record.get("request_id", "")),
        case=str(metadata.get("case", "")),
        backend=str(metadata.get("backend", "")),
        model=str(metadata.get("model", "")),
        chunk_id=str(record.get("chunk_id", "")),
        cache_key=str(record.get("cache_key", "")),
        tier=str(record.get("tier", "unknown") or "unknown"),
        bytes=_optional_int(record.get("bytes")),
        latency_ms=_optional_float(record.get("latency_ms")),
        line_number=_optional_int(record.get("line_number")),
        metadata={
            "source_schema": record.get("schema", ""),
            "raw_event_type": event_type,
            **metadata,
        },
    )


def trace_from_prefetch_record(record: dict[str, Any]) -> TraceEvent:
    metadata = _dict(record.get("metadata"))
    event_type = str(record.get("event_type", "prefetch_event"))
    endpoint = _dict(metadata.get("metadata")) if "metadata" in metadata else {}
    return TraceEvent(
        event_type=event_type,
        category=category_for_event_type(event_type),
        source=str(record.get("source", "prefetch_events.jsonl")),
        status=str(record.get("status", "observed")),
        run_id=str(record.get("run_id") or metadata.get("run_id", "") or endpoint.get("run_id", "")),
        request_id=str(record.get("request_id", "")),
        case=str(record.get("case", "")),
        backend=str(metadata.get("backend", "") or endpoint.get("backend", "")),
        model=str(metadata.get("model", "") or endpoint.get("model", "")),
        chunk_id=str(record.get("chunk_id", "")),
        tier=str(record.get("target_tier", "unknown") or "unknown"),
        latency_ms=_optional_float(metadata.get("latency_ms")),
        metadata={
            "source_schema": record.get("schema", ""),
            "mode": record.get("mode", ""),
            "message": record.get("message", ""),
            **metadata,
        },
    )


def trace_from_memory_sample(row: dict[str, Any], *, source: str, case: str = "") -> TraceEvent:
    attribution_mode = str(row.get("attribution_mode") or "case_boundary")
    request_id = str(row.get("request_id") or "") if attribution_mode == "exclusive_request" else ""
    sample_case = str(row.get("case") or case)
    identity_keys = {
        "timestamp_s", "run_id", "case", "request_id", "request_started_s", "request_ended_s",
        "sample_path", "active_request_ids", "shared_request_ids", "shared_boundary_ids", "attribution_mode",
        "cpu_rss_mb", "gpu_used_mb", "gpu_util_pct", "disk_read_mb", "disk_write_mb",
    }
    return TraceEvent(
        event_type="memory_sample",
        category="memory",
        source=source,
        status="observed",
        timestamp=str(row.get("timestamp_s", "")),
        run_id=str(row.get("run_id") or ""),
        request_id=request_id,
        case=sample_case,
        attribution_mode=attribution_mode,
        tier="host_gpu_disk",
        metadata={
            "run_id": row.get("run_id", ""),
            "attribution_mode": attribution_mode,
            "active_request_ids": row.get("active_request_ids", ""),
            "shared_request_ids": row.get("shared_request_ids", ""),
            "shared_boundary_ids": row.get("shared_boundary_ids", ""),
            "request_started_s": row.get("request_started_s", ""),
            "request_ended_s": row.get("request_ended_s", ""),
            "sample_path": row.get("sample_path", ""),
            "cpu_rss_mb": row.get("cpu_rss_mb", ""),
            "gpu_used_mb": row.get("gpu_used_mb", ""),
            "gpu_util_pct": row.get("gpu_util_pct", ""),
            "disk_read_mb": row.get("disk_read_mb", ""),
            "disk_write_mb": row.get("disk_write_mb", ""),
            **{f"diagnostic_{key}": value for key, value in row.items() if key not in identity_keys},
        },
    )


def category_for_event_type(event_type: str) -> str:
    lowered = event_type.lower()
    if lowered.startswith("cache_") or lowered in {"register_kv"}:
        return "kv"
    if "prefetch" in lowered:
        return "prefetch"
    if lowered.startswith("request_") or lowered == "demand_completed":
        return "request"
    if "benchmark" in lowered:
        return "benchmark"
    if "memory" in lowered or "sample" in lowered:
        return "memory"
    if "placement" in lowered or "offload" in lowered:
        return "placement"
    if "error" in lowered or "missing" in lowered or "failed" in lowered:
        return "error"
    if lowered in {"config_load", "connector_init"}:
        return "runtime"
    return "unknown"


def validate_trace_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            errors.append(f"missing field: {field_name}")
    if record.get("schema") != TRACE_SCHEMA_VERSION:
        errors.append(f"invalid schema: {record.get('schema')}")
    if not record.get("event_type"):
        errors.append("event_type is empty")
    if not record.get("source"):
        errors.append("source is empty")
    category = record.get("category")
    if category not in KNOWN_CATEGORIES:
        errors.append(f"unknown category: {category}")
    if not isinstance(record.get("metadata"), dict):
        errors.append("metadata must be an object")
    return errors


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [
            {
                "schema": "missing-input",
                "event_type": "trace_input_missing",
                "source": str(jsonl_path),
                "status": "missing",
                "metadata": {"path": str(jsonl_path)},
            }
        ]
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "schema": "invalid-json",
                        "event_type": "trace_parse_error",
                        "source": str(jsonl_path),
                        "status": "failed",
                        "line_number": line_number,
                        "metadata": {"error": str(exc), "raw_line": line.strip()},
                    }
                )
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def load_memory_samples(path: str | Path, case: str = "") -> list[TraceEvent]:
    sample_path = Path(path)
    if not sample_path.exists():
        return [
            TraceEvent(
                event_type="memory_samples_missing",
                category="error",
                source=str(sample_path),
                status="missing",
                case=case,
                metadata={"path": str(sample_path)},
            )
        ]
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            trace_from_memory_sample(row, source=str(sample_path), case=case or infer_case_from_sample_path(sample_path))
            for row in csv.DictReader(handle)
        ]


def infer_case_from_sample_path(path: Path) -> str:
    name = path.name
    suffix = "_samples.csv"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def write_trace_jsonl(events: Iterable[TraceEvent], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def summarize_trace_events(events: Iterable[TraceEvent]) -> dict[str, Any]:
    total = 0
    categories: dict[str, int] = {}
    event_types: dict[str, int] = {}
    statuses: dict[str, int] = {}
    tiers: dict[str, int] = {}
    request_ids: set[str] = set()
    kv_hits = 0
    kv_misses = 0
    prefetch_hits = 0
    prefetch_waste = 0
    for event in events:
        total += 1
        categories[event.category] = categories.get(event.category, 0) + 1
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        statuses[event.status] = statuses.get(event.status, 0) + 1
        tiers[event.tier] = tiers.get(event.tier, 0) + 1
        if event.request_id:
            request_ids.add(event.request_id)
        if event.event_type == "cache_hit":
            kv_hits += 1
        elif event.event_type == "cache_miss":
            kv_misses += 1
        elif event.event_type == "prefetch_hit":
            prefetch_hits += 1
        elif event.event_type == "prefetch_waste":
            prefetch_waste += 1
    return {
        "total_events": total,
        "category_counts": dict(sorted(categories.items())),
        "event_type_counts": dict(sorted(event_types.items())),
        "status_counts": dict(sorted(statuses.items())),
        "tier_counts": dict(sorted(tiers.items())),
        "unique_request_ids": len(request_ids),
        "kv_hit_rate": kv_hits / max(1, kv_hits + kv_misses),
        "prefetch_hit_rate": prefetch_hits / max(1, prefetch_hits + prefetch_waste),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
