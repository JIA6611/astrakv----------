"""Read-only MoE expert event extraction helpers.

The helpers in this module normalize already-generated router logs or JSONL
records from MoE serving runs. They do not import, monkeypatch, or modify
vLLM, LMCache, Hugging Face, or model-specific runtime code.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MOE_EVENT_SCHEMA_VERSION = "astra-moe-event-v1"

ROUTE_EVENT_TYPES = {"expert_route", "expert_selected", "expert_prefetch"}
HIT_EVENT_TYPES = {"expert_hit", "expert_load_hit"}
MISS_EVENT_TYPES = {"expert_miss", "expert_load_miss"}


@dataclass(frozen=True, slots=True)
class MoEExpertEvent:
    event_type: str
    source: str
    status: str = "observed"
    request_id: str = ""
    token_index: int | None = None
    layer_id: int | None = None
    expert_id: str = ""
    expert_rank: int | None = None
    score: float | None = None
    top_k: int | None = None
    tier: str = "unknown"
    bytes: int | None = None
    latency_ms: float | None = None
    timestamp: str = ""
    line_number: int | None = None
    raw_line: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": MOE_EVENT_SCHEMA_VERSION,
            "event_type": self.event_type,
            "source": self.source,
            "status": self.status,
            "request_id": self.request_id,
            "token_index": self.token_index,
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "expert_rank": self.expert_rank,
            "score": self.score,
            "top_k": self.top_k,
            "tier": self.tier,
            "bytes": self.bytes,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "line_number": self.line_number,
            "raw_line": self.raw_line,
            "metadata": dict(self.metadata),
        }


def parse_router_log(path: str | Path, source_name: str | None = None) -> list[MoEExpertEvent]:
    log_path = Path(path)
    source = source_name or str(log_path)
    if not log_path.exists():
        return [
            MoEExpertEvent(
                event_type="router_log_missing",
                source=source,
                status="missing",
                metadata={"path": str(log_path)},
            )
        ]

    events: list[MoEExpertEvent] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            parsed = parse_router_log_line(raw, source=source, line_number=line_number)
            if parsed:
                events.extend(parsed)
    return events


def parse_router_log_line(line: str, *, source: str, line_number: int) -> list[MoEExpertEvent]:
    line = line.lstrip("\ufeff")
    lowered = line.lower()
    if not _looks_moe_related(lowered):
        return []

    event_type = _infer_event_type(lowered)
    request_id = _extract_text_field(line, ("request_id", "req_id", "request id", "req id"))
    token_index = _extract_int_field(line, ("token_index", "token", "tok"))
    layer_id = _extract_int_field(line, ("layer_id", "layer"))
    top_k = _extract_int_field(line, ("top_k", "topk"))
    tier = _infer_tier(lowered)
    timestamp = _extract_timestamp(line)
    latency_ms = _extract_float_field(line, ("latency_ms", "latency"))
    bytes_value = _extract_bytes(line)
    experts = _extract_experts(line)
    scores = _extract_scores(line)

    if not experts:
        single_expert = _extract_text_field(line, ("expert_id", "expert", "expert id"))
        if single_expert:
            experts = [single_expert]

    if not experts:
        return [
            MoEExpertEvent(
                event_type="expert_log",
                source=source,
                status="observed",
                request_id=request_id,
                token_index=token_index,
                layer_id=layer_id,
                tier=tier,
                bytes=bytes_value,
                latency_ms=latency_ms,
                timestamp=timestamp,
                line_number=line_number,
                raw_line=line,
                metadata=_keyword_metadata(lowered),
            )
        ]

    inferred_top_k = top_k if top_k is not None else len(experts)
    events: list[MoEExpertEvent] = []
    for rank, expert_id in enumerate(experts):
        score = scores[rank] if rank < len(scores) else None
        events.append(
            MoEExpertEvent(
                event_type=event_type,
                source=source,
                status="observed",
                request_id=request_id,
                token_index=token_index,
                layer_id=layer_id,
                expert_id=str(expert_id),
                expert_rank=rank,
                score=score,
                top_k=inferred_top_k,
                tier=tier,
                bytes=bytes_value,
                latency_ms=latency_ms,
                timestamp=timestamp,
                line_number=line_number,
                raw_line=line,
                metadata=_keyword_metadata(lowered),
            )
        )
    return events


def parse_events_jsonl(path: str | Path) -> list[MoEExpertEvent]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return [
            MoEExpertEvent(
                event_type="moe_events_missing",
                source=str(jsonl_path),
                status="missing",
                metadata={"path": str(jsonl_path)},
            )
        ]

    events: list[MoEExpertEvent] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                events.append(
                    MoEExpertEvent(
                        event_type="moe_event_parse_error",
                        source=str(jsonl_path),
                        status="failed",
                        line_number=line_number,
                        raw_line=line.strip(),
                        metadata={"error": str(exc)},
                    )
                )
                continue
            if not isinstance(record, dict):
                continue
            events.extend(events_from_record(record, source=str(jsonl_path), line_number=line_number))
    return events


def events_from_record(record: dict[str, Any], *, source: str, line_number: int | None = None) -> list[MoEExpertEvent]:
    experts = _record_experts(record)
    scores = _record_scores(record)
    event_type = str(record.get("event_type") or record.get("type") or "expert_route")
    metadata = _dict(record.get("metadata"))
    top_k = _optional_int(record.get("top_k") or record.get("topk")) or (len(experts) if experts else None)

    if not experts:
        return [
            MoEExpertEvent(
                event_type=event_type,
                source=str(record.get("source") or source),
                status=str(record.get("status", "observed")),
                request_id=str(record.get("request_id") or record.get("req_id") or ""),
                token_index=_optional_int(_first_present(record, "token_index", "token")),
                layer_id=_optional_int(_first_present(record, "layer_id", "layer")),
                tier=str(record.get("tier", "unknown") or "unknown"),
                bytes=_optional_int(record.get("bytes")),
                latency_ms=_optional_float(record.get("latency_ms")),
                timestamp=str(record.get("timestamp", "")),
                line_number=line_number,
                raw_line=str(record.get("raw_line", "")),
                metadata={**metadata, "source_schema": record.get("schema", "")},
            )
        ]

    events: list[MoEExpertEvent] = []
    for rank, expert_id in enumerate(experts):
        rank_from_record = _optional_int(record.get("expert_rank") or record.get("rank"))
        expert_rank = rank_from_record if len(experts) == 1 and rank_from_record is not None else rank
        events.append(
            MoEExpertEvent(
                event_type=event_type,
                source=str(record.get("source") or source),
                status=str(record.get("status", "observed")),
                request_id=str(record.get("request_id") or record.get("req_id") or ""),
                token_index=_optional_int(_first_present(record, "token_index", "token")),
                layer_id=_optional_int(_first_present(record, "layer_id", "layer")),
                expert_id=str(expert_id),
                expert_rank=expert_rank,
                score=scores[rank] if rank < len(scores) else _optional_float(record.get("score")),
                top_k=top_k,
                tier=str(record.get("tier", "unknown") or "unknown"),
                bytes=_optional_int(record.get("bytes")),
                latency_ms=_optional_float(record.get("latency_ms")),
                timestamp=str(record.get("timestamp", "")),
                line_number=line_number,
                raw_line=str(record.get("raw_line", "")),
                metadata={**metadata, "source_schema": record.get("schema", "")},
            )
        )
    return events


def write_events_jsonl(events: Iterable[MoEExpertEvent], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def write_expert_summary_csv(events: Iterable[MoEExpertEvent], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_events(events)
    rows = summary["expert_rows"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_id",
                "expert_id",
                "activation_count",
                "token_count",
                "avg_score",
                "bytes",
                "latency_ms",
                "hotness_rank",
                "hotness_share",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_events(events: Iterable[MoEExpertEvent]) -> dict[str, Any]:
    event_list = list(events)
    event_type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    request_ids: set[str] = set()
    routed_tokens: set[tuple[str, int | None, int | None]] = set()
    expert_stats: dict[tuple[int | None, str], dict[str, Any]] = {}
    route_count = 0
    hit_count = 0
    miss_count = 0
    prefetch_count = 0
    offload_count = 0
    total_expert_bytes = 0
    total_latency_ms = 0.0
    latency_count = 0

    for event in event_list:
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        status_counts[event.status] = status_counts.get(event.status, 0) + 1
        tier_counts[event.tier] = tier_counts.get(event.tier, 0) + 1
        if event.request_id:
            request_ids.add(event.request_id)
        if event.latency_ms is not None:
            total_latency_ms += event.latency_ms
            latency_count += 1
        if event.bytes:
            total_expert_bytes += event.bytes

        if event.event_type in ROUTE_EVENT_TYPES:
            route_count += 1
            if event.request_id or event.layer_id is not None or event.token_index is not None:
                routed_tokens.add((event.request_id, event.layer_id, event.token_index))
        if event.event_type in HIT_EVENT_TYPES:
            hit_count += 1
        if event.event_type in MISS_EVENT_TYPES:
            miss_count += 1
        if "prefetch" in event.event_type:
            prefetch_count += 1
        if "offload" in event.event_type or "evict" in event.event_type:
            offload_count += 1

        if not event.expert_id:
            continue
        key = (event.layer_id, event.expert_id)
        stats = expert_stats.setdefault(
            key,
            {
                "layer_id": event.layer_id,
                "expert_id": event.expert_id,
                "activation_count": 0,
                "tokens": set(),
                "score_sum": 0.0,
                "score_count": 0,
                "bytes": 0,
                "latency_ms": 0.0,
            },
        )
        stats["activation_count"] += 1
        if event.request_id or event.token_index is not None:
            stats["tokens"].add((event.request_id, event.token_index))
        if event.score is not None:
            stats["score_sum"] += event.score
            stats["score_count"] += 1
        if event.bytes:
            stats["bytes"] += event.bytes
        if event.latency_ms is not None:
            stats["latency_ms"] += event.latency_ms

    total_activations = sum(int(stats["activation_count"]) for stats in expert_stats.values())
    expert_rows: list[dict[str, Any]] = []
    for (_, _), stats in expert_stats.items():
        score_count = int(stats["score_count"])
        activation_count = int(stats["activation_count"])
        expert_rows.append(
            {
                "layer_id": "" if stats["layer_id"] is None else stats["layer_id"],
                "expert_id": stats["expert_id"],
                "activation_count": activation_count,
                "token_count": len(stats["tokens"]),
                "avg_score": _round_or_empty(stats["score_sum"] / score_count if score_count else None),
                "bytes": int(stats["bytes"]),
                "latency_ms": _round_or_empty(stats["latency_ms"]),
                "hotness_share": _round_or_empty(activation_count / total_activations if total_activations else None),
            }
        )
    expert_rows.sort(key=lambda row: (-int(row["activation_count"]), str(row["layer_id"]), str(row["expert_id"])))
    for index, row in enumerate(expert_rows, start=1):
        row["hotness_rank"] = index

    return {
        "total_events": len(event_list),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "unique_request_ids": len(request_ids),
        "routed_token_count": len(routed_tokens),
        "unique_expert_ids": len({row["expert_id"] for row in expert_rows}),
        "unique_layer_experts": len(expert_rows),
        "route_event_count": route_count,
        "expert_hit_count": hit_count,
        "expert_miss_count": miss_count,
        "expert_hit_rate": hit_count / max(1, hit_count + miss_count),
        "expert_prefetch_count": prefetch_count,
        "expert_offload_count": offload_count,
        "expert_memory_bytes": total_expert_bytes,
        "avg_latency_ms": total_latency_ms / latency_count if latency_count else None,
        "expert_rows": expert_rows,
    }


def _looks_moe_related(lowered_line: str) -> bool:
    keywords = (
        "moe",
        "expert",
        "router",
        "routing",
        "routed",
        "top_k",
        "topk",
        "gate",
        "gating",
    )
    return any(keyword in lowered_line for keyword in keywords)


def _infer_event_type(lowered_line: str) -> str:
    if "hit" in lowered_line and "miss" not in lowered_line:
        return "expert_hit"
    if "miss" in lowered_line:
        return "expert_miss"
    if "prefetch" in lowered_line:
        return "expert_prefetch"
    if "load" in lowered_line:
        return "expert_load"
    if "offload" in lowered_line or "evict" in lowered_line:
        return "expert_offload"
    return "expert_route"


def _extract_experts(line: str) -> list[str]:
    patterns = (
        r"(?:experts|expert_ids|expert ids|selected_experts|selected experts)\s*[=:]\s*\[(?P<values>[^\]]*)\]",
        r"(?:experts|expert_ids|expert ids|selected_experts|selected experts)\s*[=:]\s*(?P<values>[0-9A-Za-z_.:-]+(?:\s*,\s*[0-9A-Za-z_.:-]+)*)",
    )
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return _split_values(match.group("values"))
    return []


def _extract_scores(line: str) -> list[float]:
    patterns = (
        r"(?:scores|probs|probabilities|weights)\s*[=:]\s*\[(?P<values>[^\]]*)\]",
        r"(?:scores|probs|probabilities|weights)\s*[=:]\s*(?P<values>[-+0-9.eE]+(?:\s*,\s*[-+0-9.eE]+)*)",
    )
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return [_optional_float(value) for value in _split_values(match.group("values")) if _optional_float(value) is not None]
    return []


def _record_experts(record: dict[str, Any]) -> list[str]:
    for key in ("experts", "expert_ids", "selected_experts"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if value not in (None, ""):
            return _split_values(str(value))
    expert_id = _first_present(record, "expert_id", "expert")
    return [str(expert_id)] if expert_id not in (None, "") else []


def _record_scores(record: dict[str, Any]) -> list[float]:
    for key in ("scores", "probs", "probabilities", "weights"):
        value = record.get(key)
        if isinstance(value, list):
            return [float(item) for item in value if _optional_float(item) is not None]
        if value not in (None, ""):
            return [_optional_float(item) for item in _split_values(str(value)) if _optional_float(item) is not None]
    score = _optional_float(record.get("score"))
    return [score] if score is not None else []


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_text_field(line: str, names: tuple[str, ...]) -> str:
    for name in names:
        pattern_name = re.escape(name).replace(r"\ ", r"\s+")
        pattern = pattern_name + r"\s*[=:]\s*['\"]?(?P<value>[A-Za-z0-9_.:-]+)"
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group("value").rstrip(",'\")")
    return ""


def _extract_int_field(line: str, names: tuple[str, ...]) -> int | None:
    value = _extract_text_field(line, names)
    return _optional_int(value)


def _extract_float_field(line: str, names: tuple[str, ...]) -> float | None:
    value = _extract_text_field(line, names)
    return _optional_float(value)


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


def _split_values(value: str) -> list[str]:
    cleaned = value.strip().strip("[]()")
    if not cleaned:
        return []
    return [item.strip().strip("'\"") for item in re.split(r"[, ]+", cleaned) if item.strip()]


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


def _round_or_empty(value: float | None) -> float | str:
    return "" if value is None else round(float(value), 6)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _keyword_metadata(lowered_line: str) -> dict[str, bool]:
    return {
        "mentions_moe": "moe" in lowered_line,
        "mentions_router": "router" in lowered_line or "routing" in lowered_line,
        "mentions_gate": "gate" in lowered_line or "gating" in lowered_line,
        "mentions_prefetch": "prefetch" in lowered_line,
        "mentions_load": "load" in lowered_line,
        "mentions_offload": "offload" in lowered_line or "evict" in lowered_line,
    }
