"""Read-only cache event extraction helpers.

The helpers in this module parse already-generated runtime artifacts such as
server logs, request JSONL files, and benchmark CSV files. They do not import
or modify vLLM, LMCache, or any third-party runtime.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


EVENT_SCHEMA_VERSION = "astra-cache-event-v1"


@dataclass(frozen=True, slots=True)
class CacheEvent:
    event_type: str
    source: str
    status: str = "observed"
    request_id: str = ""
    cache_key: str = ""
    chunk_id: str = ""
    tier: str = "unknown"
    bytes: int | None = None
    start_time: str = ""
    end_time: str = ""
    latency_ms: float | None = None
    line_number: int | None = None
    raw_line: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA_VERSION,
            "event_type": self.event_type,
            "source": self.source,
            "status": self.status,
            "request_id": self.request_id,
            "cache_key": self.cache_key,
            "chunk_id": self.chunk_id,
            "tier": self.tier,
            "bytes": self.bytes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "line_number": self.line_number,
            "raw_line": self.raw_line,
            "metadata": dict(self.metadata),
        }


def parse_server_log(path: str | Path, source_name: str | None = None) -> list[CacheEvent]:
    log_path = Path(path)
    source = source_name or str(log_path)
    events: list[CacheEvent] = []
    if not log_path.exists():
        return [
            CacheEvent(
                event_type="log_missing",
                source=source,
                status="missing",
                metadata={"path": str(log_path)},
            )
        ]

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            event = parse_log_line(raw, source=source, line_number=line_number)
            if event is not None:
                events.extend(event if isinstance(event, list) else [event])
    return events


def parse_log_line(
    line: str,
    *,
    source: str,
    line_number: int,
) -> CacheEvent | list[CacheEvent] | None:
    line = line.lstrip("\ufeff").removeprefix("锘?")
    lowered = line.lower()
    if not _looks_cache_related(lowered):
        return None

    timestamp = _extract_timestamp(line)
    request_id = _extract_request_id(line)
    tier = _infer_tier(lowered)

    config_match = re.search(r"Loading LMCache config file\s+(?P<config>\S+)", line)
    if config_match:
        return CacheEvent(
            event_type="config_load",
            source=source,
            status="ok",
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
            metadata={"config_file": config_match.group("config")},
        )

    if "lmcache initialized" in lowered or "creating lmcacheengine" in lowered:
        return CacheEvent(
            event_type="connector_init",
            source=source,
            status="ok",
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
            metadata=_keyword_metadata(lowered),
        )

    if "registering kv caches" in lowered:
        return CacheEvent(
            event_type="register_kv",
            source=source,
            status="ok",
            request_id=request_id,
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
        )

    hit_match = re.search(
        r"LMCache hit tokens:\s*(?P<hit>None|\d+)(?:,\s*need to load:\s*(?P<load>\d+))?",
        line,
    )
    if hit_match:
        hit_raw = hit_match.group("hit")
        hit_tokens = None if hit_raw == "None" else int(hit_raw)
        load_tokens = _optional_int(hit_match.group("load"))
        event_type = "cache_miss" if not hit_tokens else "cache_hit"
        events = [
            CacheEvent(
                event_type=event_type,
                source=source,
                status="ok",
                request_id=request_id,
                tier=tier,
                start_time=timestamp,
                end_time=timestamp,
                line_number=line_number,
                raw_line=line,
                metadata={
                    "hit_tokens": hit_tokens,
                    "need_to_load_tokens": load_tokens,
                },
            )
        ]
        if load_tokens and load_tokens > 0:
            events.append(
                CacheEvent(
                    event_type="cache_load",
                    source=source,
                    status="planned",
                    request_id=request_id,
                    tier=tier,
                    start_time=timestamp,
                    end_time=timestamp,
                    line_number=line_number,
                    raw_line=line,
                    metadata={"tokens": load_tokens},
                )
            )
        return events

    retrieved_match = re.search(r"Retrieved\s+(?P<tokens>\d+)\s+tokens", line)
    if retrieved_match:
        return CacheEvent(
            event_type="cache_load",
            source=source,
            status="completed",
            request_id=request_id,
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
            metadata={"tokens": int(retrieved_match.group("tokens"))},
        )

    if "retrieved tokens is less" in lowered or "retrieve fail" in lowered:
        return CacheEvent(
            event_type="cache_load",
            source=source,
            status="partial_or_failed",
            request_id=request_id,
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
        )

    if "lookup" in lowered and ("cache" in lowered or "lmcache" in lowered):
        return CacheEvent(
            event_type="cache_lookup",
            source=source,
            status="observed",
            request_id=request_id,
            cache_key=_extract_lookup_id(line),
            tier=tier,
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
        )

    if ("store" in lowered or "stored" in lowered or "save" in lowered) and (
        "cache" in lowered or "lmcache" in lowered
    ):
        return CacheEvent(
            event_type="cache_store",
            source=source,
            status="observed",
            request_id=request_id,
            tier=tier,
            bytes=_extract_bytes(line),
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
        )

    if "offload" in lowered or "evict" in lowered:
        return CacheEvent(
            event_type="cache_offload",
            source=source,
            status="observed",
            request_id=request_id,
            tier=tier,
            bytes=_extract_bytes(line),
            start_time=timestamp,
            end_time=timestamp,
            line_number=line_number,
            raw_line=line,
        )

    return CacheEvent(
        event_type="cache_log",
        source=source,
        status="observed",
        request_id=request_id,
        tier=tier,
        start_time=timestamp,
        end_time=timestamp,
        line_number=line_number,
        raw_line=line,
        metadata=_keyword_metadata(lowered),
    )


def parse_request_results(path: str | Path) -> list[CacheEvent]:
    request_path = Path(path)
    if not request_path.exists():
        return [
            CacheEvent(
                event_type="request_results_missing",
                source=str(request_path),
                status="missing",
                metadata={"path": str(request_path)},
            )
        ]

    events: list[CacheEvent] = []
    with request_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                events.append(
                    CacheEvent(
                        event_type="request_result_parse_error",
                        source=str(request_path),
                        status="failed",
                        line_number=line_number,
                        raw_line=line.strip(),
                        metadata={"error": str(exc)},
                    )
                )
                continue
            events.append(
                CacheEvent(
                    event_type="request_result",
                    source=str(request_path),
                    status=str(item.get("status", "unknown")),
                    request_id=str(item.get("request_id", "")),
                    latency_ms=_optional_float(item.get("latency_ms")),
                    line_number=line_number,
                    metadata={
                        "case": item.get("case", ""),
                        "backend": item.get("backend", ""),
                        "model": item.get("model", ""),
                        "batch_size": item.get("batch_size", ""),
                        "context_length": item.get("context_length", ""),
                        "output_tokens_observed": item.get("output_tokens_observed", ""),
                        "ttft_ms": item.get("ttft_ms", ""),
                        "tpot_ms": item.get("tpot_ms", ""),
                        "gpu_probe": item.get("gpu_probe", ""),
                        "error": item.get("error", ""),
                    },
                )
            )
    return events


def parse_benchmark_results(path: str | Path) -> list[CacheEvent]:
    csv_path = Path(path)
    if not csv_path.exists():
        return [
            CacheEvent(
                event_type="benchmark_results_missing",
                source=str(csv_path),
                status="missing",
                metadata={"path": str(csv_path)},
            )
        ]

    events: list[CacheEvent] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            events.append(
                CacheEvent(
                    event_type="benchmark_case_metrics",
                    source=str(csv_path),
                    status=str(row.get("status", "unknown")),
                    line_number=row_number,
                    metadata={
                        "case": row.get("case", ""),
                        "backend": row.get("backend", ""),
                        "request_count": row.get("request_count", ""),
                        "success_count": row.get("success_count", ""),
                        "ttft_ms": row.get("ttft_ms", ""),
                        "tpot_ms": row.get("tpot_ms", ""),
                        "latency_p95_ms": row.get("latency_p95_ms", ""),
                        "throughput_tokens_s": row.get("throughput_tokens_s", ""),
                        "cpu_memory_peak_mb": row.get("cpu_memory_peak_mb", ""),
                        "gpu_memory_peak_mb": row.get("gpu_memory_peak_mb", ""),
                        "gpu_util_peak_pct": row.get("gpu_util_peak_pct", ""),
                        "disk_read_delta_mb": row.get("disk_read_delta_mb", ""),
                        "disk_write_delta_mb": row.get("disk_write_delta_mb", ""),
                        "sample_count": row.get("sample_count", ""),
                        "errors": row.get("errors", ""),
                    },
                )
            )
    return events


def write_events_jsonl(events: Iterable[CacheEvent], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def summarize_events(events: Iterable[CacheEvent]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    tiers: dict[str, int] = {}
    total = 0
    for event in events:
        total += 1
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
        statuses[event.status] = statuses.get(event.status, 0) + 1
        tiers[event.tier] = tiers.get(event.tier, 0) + 1
    return {
        "total_events": total,
        "event_type_counts": dict(sorted(counts.items())),
        "status_counts": dict(sorted(statuses.items())),
        "tier_counts": dict(sorted(tiers.items())),
    }


def _looks_cache_related(lowered_line: str) -> bool:
    keywords = (
        "lmcache",
        "kv cache",
        "kvcache",
        "kv_transfer",
        "kv transfer",
        "cache hit",
        "hit tokens",
        "retrieved",
        "retrieve",
        "lookup",
        "store",
        "offload",
        "evict",
        "local_cpu",
        "local_disk",
    )
    return any(keyword in lowered_line for keyword in keywords)


def _infer_tier(lowered_line: str) -> str:
    if "local_disk" in lowered_line or "disk" in lowered_line or "ssd" in lowered_line:
        return "disk"
    if "local_cpu" in lowered_line or "cpu" in lowered_line:
        return "cpu"
    if "gpu" in lowered_line or "cuda" in lowered_line:
        return "gpu"
    return "unknown"


def _extract_timestamp(line: str) -> str:
    match = re.match(
        r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)",
        line,
    )
    return match.group("ts") if match else ""


def _extract_request_id(line: str) -> str:
    patterns = (
        r"(?:request_id|request id|req_id|req id|lookup_id|lookup id)[=: ]+['\"]?(?P<id>[A-Za-z0-9_.:-]+)",
        r"request\s+(?P<id>[A-Za-z0-9_.:-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group("id").rstrip(",'\")")
    return ""


def _extract_lookup_id(line: str) -> str:
    match = re.search(r"(?:lookup_id|lookup id)[=: ]+['\"]?(?P<id>[A-Za-z0-9_.:-]+)", line, re.IGNORECASE)
    return match.group("id").rstrip(",'\")") if match else ""


def _extract_bytes(line: str) -> int | None:
    match = re.search(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|KiB|MB|MiB|GB|GiB|bytes)", line)
    if not match:
        return None
    value = float(match.group("num"))
    unit = match.group("unit").lower()
    factors = {
        "b": 1,
        "bytes": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000 * 1000,
        "mib": 1024 * 1024,
        "gb": 1000 * 1000 * 1000,
        "gib": 1024 * 1024 * 1024,
    }
    return int(value * factors.get(unit, 1))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _keyword_metadata(lowered_line: str) -> dict[str, bool]:
    return {
        "mentions_lmcache": "lmcache" in lowered_line,
        "mentions_connector": "connector" in lowered_line,
        "mentions_lookup": "lookup" in lowered_line,
        "mentions_retrieve": "retrieve" in lowered_line or "retrieved" in lowered_line,
        "mentions_store": "store" in lowered_line or "stored" in lowered_line,
        "mentions_offload": "offload" in lowered_line,
    }
