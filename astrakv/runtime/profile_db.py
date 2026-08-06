"""Profile-guided runtime database for AstraKV-W.

ProfileDB aggregates unified trace events into reusable workload and chunk
statistics. It is intentionally lightweight JSON so profiles can be moved
between benchmark runs and consumed by policy/scoring code without a service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.trace_schema import TRACE_SCHEMA_VERSION, TraceEvent, load_jsonl


PROFILE_DB_VERSION = "astra-profile-db-v1"


@dataclass(slots=True)
class LayerSensitivityRecord:
    workload_id: str
    layer_id: int
    sensitivity_score: float = 0.0
    partial_load_allowed: bool = True
    recompute_allowed: bool = True
    prefetch_priority_boost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "layer_id": self.layer_id,
            "sensitivity_score": self.sensitivity_score,
            "partial_load_allowed": self.partial_load_allowed,
            "recompute_allowed": self.recompute_allowed,
            "prefetch_priority_boost": self.prefetch_priority_boost,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "LayerSensitivityRecord":
        return cls(
            workload_id=str(record.get("workload_id", "")),
            layer_id=int(record.get("layer_id", 0)),
            sensitivity_score=as_float(record.get("sensitivity_score")) or 0.0,
            partial_load_allowed=bool(record.get("partial_load_allowed", True)),
            recompute_allowed=bool(record.get("recompute_allowed", True)),
            prefetch_priority_boost=as_float(record.get("prefetch_priority_boost")) or 0.0,
            metadata=dict(record.get("metadata") or {}),
        )


@dataclass(slots=True)
class QualityGuardRecord:
    workload_id: str
    chunk_id: str = ""
    layer_id: int | None = None
    quality_tier: str = "unknown"
    max_ppl_delta: float | None = None
    min_cka: float | None = None
    partial_load_allowed: bool = True
    recompute_allowed: bool = True
    prefetch_priority_boost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "chunk_id": self.chunk_id,
            "layer_id": self.layer_id,
            "quality_tier": self.quality_tier,
            "max_ppl_delta": self.max_ppl_delta,
            "min_cka": self.min_cka,
            "partial_load_allowed": self.partial_load_allowed,
            "recompute_allowed": self.recompute_allowed,
            "prefetch_priority_boost": self.prefetch_priority_boost,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "QualityGuardRecord":
        return cls(
            workload_id=str(record.get("workload_id", "")),
            chunk_id=str(record.get("chunk_id", "")),
            layer_id=as_int(record.get("layer_id")),
            quality_tier=str(record.get("quality_tier", "unknown")),
            max_ppl_delta=as_float(record.get("max_ppl_delta")),
            min_cka=as_float(record.get("min_cka")),
            partial_load_allowed=bool(record.get("partial_load_allowed", True)),
            recompute_allowed=bool(record.get("recompute_allowed", True)),
            prefetch_priority_boost=as_float(record.get("prefetch_priority_boost")) or 0.0,
            metadata=dict(record.get("metadata") or {}),
        )


@dataclass(slots=True)
class ChunkProfile:
    chunk_id: str
    workload_id: str
    case: str = ""
    cache_key: str = ""
    request_id: str = ""
    prefix_id: str = ""
    run_id: str = ""
    arrival_index: int | None = None
    reuse_ratio: float | None = None
    reuse_bucket: str = ""
    legacy_unlinked: bool = True
    request_count: int = 0
    reuse_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_loads: int = 0
    cache_stores: int = 0
    offloads: int = 0
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_hits: int = 0
    prefetch_waste: int = 0
    bytes_loaded: int = 0
    load_latency_ms_total: float = 0.0
    load_latency_count: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)

    @property
    def reuse_frequency(self) -> float:
        return self.reuse_count / max(1, self.request_count)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(1, self.cache_hits + self.cache_misses)

    @property
    def prefetch_hit_rate(self) -> float:
        return self.prefetch_hits / max(1, self.prefetch_hits + self.prefetch_waste)

    @property
    def avg_load_latency_ms(self) -> float:
        return self.load_latency_ms_total / max(1, self.load_latency_count)

    def observe(self, event: TraceEvent) -> None:
        self.case = self.case or event.case
        self.cache_key = self.cache_key or event.cache_key
        metadata = event.metadata
        self.request_id = self.request_id or event.request_id
        self.prefix_id = self.prefix_id or str(metadata.get("prefix_id") or "")
        self.run_id = self.run_id or str(metadata.get("run_id") or "")
        self.arrival_index = self.arrival_index if self.arrival_index is not None else as_int(metadata.get("arrival_index"))
        self.reuse_ratio = self.reuse_ratio if self.reuse_ratio is not None else as_float(metadata.get("reuse_ratio"))
        self.reuse_bucket = self.reuse_bucket or str(metadata.get("reuse_bucket") or "")
        self.legacy_unlinked = self.legacy_unlinked and bool(metadata.get("legacy_unlinked", True))
        self.status_counts[event.status] = self.status_counts.get(event.status, 0) + 1
        self.tier_counts[event.tier] = self.tier_counts.get(event.tier, 0) + 1
        if event.request_id:
            self.request_count += 1
        if event.event_type in {"cache_hit", "cache_load", "cache_store", "prefetch_hit"}:
            self.reuse_count += 1
        if event.event_type == "cache_hit":
            self.cache_hits += 1
        elif event.event_type == "cache_miss":
            self.cache_misses += 1
        elif event.event_type == "cache_load":
            self.cache_loads += 1
            if event.bytes:
                self.bytes_loaded += event.bytes
            if event.latency_ms is not None:
                self.load_latency_ms_total += event.latency_ms
                self.load_latency_count += 1
        elif event.event_type == "cache_store":
            self.cache_stores += 1
        elif event.event_type == "cache_offload":
            self.offloads += 1
        elif event.event_type == "prefetch_submitted":
            self.prefetch_submitted += 1
        elif event.event_type == "prefetch_completed":
            self.prefetch_completed += 1
        elif event.event_type == "prefetch_hit":
            self.prefetch_hits += 1
        elif event.event_type == "prefetch_waste":
            self.prefetch_waste += 1

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "workload_id": self.workload_id,
            "case": self.case,
            "cache_key": self.cache_key,
            "request_id": self.request_id,
            "prefix_id": self.prefix_id,
            "run_id": self.run_id,
            "arrival_index": self.arrival_index,
            "reuse_ratio": self.reuse_ratio,
            "reuse_bucket": self.reuse_bucket,
            "legacy_unlinked": self.legacy_unlinked,
            "request_count": self.request_count,
            "reuse_count": self.reuse_count,
            "reuse_frequency": self.reuse_frequency,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_loads": self.cache_loads,
            "cache_stores": self.cache_stores,
            "offloads": self.offloads,
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_completed": self.prefetch_completed,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_waste": self.prefetch_waste,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "bytes_loaded": self.bytes_loaded,
            "avg_load_latency_ms": self.avg_load_latency_ms,
            "load_latency_count": self.load_latency_count,
            "tier_counts": dict(sorted(self.tier_counts.items())),
            "status_counts": dict(sorted(self.status_counts.items())),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ChunkProfile":
        profile = cls(
            chunk_id=str(record.get("chunk_id", "")),
            workload_id=str(record.get("workload_id", "")),
            case=str(record.get("case", "")),
            cache_key=str(record.get("cache_key", "")),
            request_id=str(record.get("request_id", "")),
            prefix_id=str(record.get("prefix_id", "")),
            run_id=str(record.get("run_id", "")),
            arrival_index=as_int(record.get("arrival_index")),
            reuse_ratio=as_float(record.get("reuse_ratio")),
            reuse_bucket=str(record.get("reuse_bucket", "")),
            legacy_unlinked=bool(record.get("legacy_unlinked", True)),
            request_count=int(record.get("request_count", 0)),
            reuse_count=int(record.get("reuse_count", 0)),
            cache_hits=int(record.get("cache_hits", 0)),
            cache_misses=int(record.get("cache_misses", 0)),
            cache_loads=int(record.get("cache_loads", 0)),
            cache_stores=int(record.get("cache_stores", 0)),
            offloads=int(record.get("offloads", 0)),
            prefetch_submitted=int(record.get("prefetch_submitted", 0)),
            prefetch_completed=int(record.get("prefetch_completed", 0)),
            prefetch_hits=int(record.get("prefetch_hits", 0)),
            prefetch_waste=int(record.get("prefetch_waste", 0)),
            bytes_loaded=int(record.get("bytes_loaded", 0)),
            load_latency_count=int(record.get("load_latency_count", 0)),
            tier_counts=_int_dict(record.get("tier_counts")),
            status_counts=_int_dict(record.get("status_counts")),
        )
        profile.load_latency_ms_total = float(record.get("avg_load_latency_ms", 0.0)) * max(
            1, profile.load_latency_count
        )
        return profile


@dataclass(slots=True)
class WorkloadProfile:
    workload_id: str
    event_count: int = 0
    request_count: int = 0
    cases: set[str] = field(default_factory=set)
    backends: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    category_counts: dict[str, int] = field(default_factory=dict)
    tier_counts: dict[str, int] = field(default_factory=dict)
    memory_sample_count: int = 0
    cpu_rss_peak_mb: float | None = None
    gpu_used_peak_mb: float | None = None
    gpu_util_peak_pct: float | None = None
    disk_read_peak_mb: float | None = None
    disk_write_peak_mb: float | None = None

    def observe(self, event: TraceEvent) -> None:
        self.event_count += 1
        if event.request_id:
            self.request_count += 1
        if event.case:
            self.cases.add(event.case)
        if event.backend:
            self.backends.add(event.backend)
        if event.model:
            self.models.add(event.model)
        self.category_counts[event.category] = self.category_counts.get(event.category, 0) + 1
        self.tier_counts[event.tier] = self.tier_counts.get(event.tier, 0) + 1
        if event.category == "memory":
            self.memory_sample_count += 1
            self.cpu_rss_peak_mb = max_optional(self.cpu_rss_peak_mb, as_float(event.metadata.get("cpu_rss_mb")))
            self.gpu_used_peak_mb = max_optional(self.gpu_used_peak_mb, as_float(event.metadata.get("gpu_used_mb")))
            self.gpu_util_peak_pct = max_optional(self.gpu_util_peak_pct, as_float(event.metadata.get("gpu_util_pct")))
            self.disk_read_peak_mb = max_optional(self.disk_read_peak_mb, as_float(event.metadata.get("disk_read_mb")))
            self.disk_write_peak_mb = max_optional(self.disk_write_peak_mb, as_float(event.metadata.get("disk_write_mb")))

    def to_record(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "event_count": self.event_count,
            "request_count": self.request_count,
            "cases": sorted(self.cases),
            "backends": sorted(self.backends),
            "models": sorted(self.models),
            "category_counts": dict(sorted(self.category_counts.items())),
            "tier_counts": dict(sorted(self.tier_counts.items())),
            "memory_sample_count": self.memory_sample_count,
            "cpu_rss_peak_mb": self.cpu_rss_peak_mb,
            "gpu_used_peak_mb": self.gpu_used_peak_mb,
            "gpu_util_peak_pct": self.gpu_util_peak_pct,
            "disk_read_peak_mb": self.disk_read_peak_mb,
            "disk_write_peak_mb": self.disk_write_peak_mb,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "WorkloadProfile":
        return cls(
            workload_id=str(record.get("workload_id", "")),
            event_count=int(record.get("event_count", 0)),
            request_count=int(record.get("request_count", 0)),
            cases=set(str(item) for item in record.get("cases", [])),
            backends=set(str(item) for item in record.get("backends", [])),
            models=set(str(item) for item in record.get("models", [])),
            category_counts=_int_dict(record.get("category_counts")),
            tier_counts=_int_dict(record.get("tier_counts")),
            memory_sample_count=int(record.get("memory_sample_count", 0)),
            cpu_rss_peak_mb=as_float(record.get("cpu_rss_peak_mb")),
            gpu_used_peak_mb=as_float(record.get("gpu_used_peak_mb")),
            gpu_util_peak_pct=as_float(record.get("gpu_util_peak_pct")),
            disk_read_peak_mb=as_float(record.get("disk_read_peak_mb")),
            disk_write_peak_mb=as_float(record.get("disk_write_peak_mb")),
        )


class ProfileDB:
    def __init__(self, version: str = PROFILE_DB_VERSION) -> None:
        self.version = version
        self.workloads: dict[str, WorkloadProfile] = {}
        self.chunks: dict[str, ChunkProfile] = {}
        self.layer_sensitivity: dict[str, LayerSensitivityRecord] = {}
        self.quality_guards: dict[str, QualityGuardRecord] = {}

    @classmethod
    def from_trace_events(cls, events: Iterable[TraceEvent], workload_id: str = "default") -> "ProfileDB":
        db = cls()
        for event in events:
            db.observe(event, workload_id=workload_id)
        return db

    @classmethod
    def from_trace_records(cls, records: Iterable[dict[str, Any]], workload_id: str = "default") -> "ProfileDB":
        return cls.from_trace_events((trace_event_from_record(record) for record in records), workload_id=workload_id)

    def observe(self, event: TraceEvent, workload_id: str = "default") -> None:
        workload_key = workload_id or infer_workload_id(event)
        workload = self.workloads.setdefault(workload_key, WorkloadProfile(workload_id=workload_key))
        workload.observe(event)

        chunk_key = infer_chunk_key(event)
        if not chunk_key:
            return
        profile_key = f"{workload_key}:{chunk_key}"
        chunk = self.chunks.setdefault(
            profile_key,
            ChunkProfile(
                chunk_id=chunk_key,
                workload_id=workload_key,
                case=event.case,
                cache_key=event.cache_key,
                request_id=event.request_id,
                prefix_id=str(event.metadata.get("prefix_id") or ""),
                run_id=str(event.metadata.get("run_id") or ""),
                arrival_index=as_int(event.metadata.get("arrival_index")),
                reuse_ratio=as_float(event.metadata.get("reuse_ratio")),
                reuse_bucket=str(event.metadata.get("reuse_bucket") or ""),
                legacy_unlinked=bool(event.metadata.get("legacy_unlinked", True)),
            ),
        )
        chunk.observe(event)

    def get_chunk(self, chunk_id: str, workload_id: str = "default") -> ChunkProfile | None:
        return self.chunks.get(f"{workload_id}:{chunk_id}")

    def put_layer_sensitivity(self, record: LayerSensitivityRecord) -> None:
        self.layer_sensitivity[f"{record.workload_id}:{record.layer_id}"] = record

    def get_layer_sensitivity(self, layer_id: int, workload_id: str = "default") -> LayerSensitivityRecord | None:
        return self.layer_sensitivity.get(f"{workload_id}:{layer_id}")

    def put_quality_guard(self, record: QualityGuardRecord) -> None:
        self.quality_guards[_quality_guard_key(record.workload_id, record.chunk_id, record.layer_id)] = record

    def get_quality_guard(
        self,
        *,
        workload_id: str = "default",
        chunk_id: str = "",
        layer_id: int | None = None,
    ) -> QualityGuardRecord | None:
        direct = self.quality_guards.get(_quality_guard_key(workload_id, chunk_id, layer_id))
        if direct is not None:
            return direct
        if chunk_id:
            chunkless = self.quality_guards.get(_quality_guard_key(workload_id, "", layer_id))
            if chunkless is not None:
                return chunkless
        if layer_id is not None:
            return self.quality_guards.get(_quality_guard_key(workload_id, "", layer_id))
        return None

    def top_chunks(self, limit: int = 10) -> list[ChunkProfile]:
        return sorted(
            self.chunks.values(),
            key=lambda item: (
                item.reuse_frequency,
                item.cache_hits + item.prefetch_hits,
                item.cache_loads,
            ),
            reverse=True,
        )[: max(0, limit)]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_DB_VERSION,
            "source_trace_schema": TRACE_SCHEMA_VERSION,
            "workloads": [
                profile.to_record()
                for profile in sorted(self.workloads.values(), key=lambda item: item.workload_id)
            ],
            "chunks": [
                profile.to_record()
                for profile in sorted(self.chunks.values(), key=lambda item: (item.workload_id, item.chunk_id))
            ],
            "layer_sensitivity": [
                record.to_record()
                for record in sorted(self.layer_sensitivity.values(), key=lambda item: (item.workload_id, item.layer_id))
            ],
            "quality_guards": [
                record.to_record()
                for record in sorted(
                    self.quality_guards.values(),
                    key=lambda item: (item.workload_id, item.chunk_id, -1 if item.layer_id is None else item.layer_id),
                )
            ],
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_record(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProfileDB":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        db = cls(version=str(record.get("schema", PROFILE_DB_VERSION)))
        for item in record.get("workloads", []):
            workload = WorkloadProfile.from_record(item)
            db.workloads[workload.workload_id] = workload
        for item in record.get("chunks", []):
            chunk = ChunkProfile.from_record(item)
            db.chunks[f"{chunk.workload_id}:{chunk.chunk_id}"] = chunk
        for item in record.get("layer_sensitivity", []):
            db.put_layer_sensitivity(LayerSensitivityRecord.from_record(item))
        for item in record.get("quality_guards", []):
            db.put_quality_guard(QualityGuardRecord.from_record(item))
        return db

    def merge_profile_guards(
        self,
        *,
        layer_sensitivity: Iterable[LayerSensitivityRecord] = (),
        quality_guards: Iterable[QualityGuardRecord] = (),
    ) -> None:
        for record in layer_sensitivity:
            self.put_layer_sensitivity(record)
        for record in quality_guards:
            self.put_quality_guard(record)


def load_profile_from_trace_jsonl(path: str | Path, workload_id: str = "default") -> ProfileDB:
    return ProfileDB.from_trace_records(load_jsonl(path), workload_id=workload_id)


def trace_event_from_record(record: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        event_type=str(record.get("event_type", "")),
        category=str(record.get("category", "unknown")),
        source=str(record.get("source", "")),
        status=str(record.get("status", "observed")),
        event_id=str(record.get("event_id", "")),
        timestamp=str(record.get("timestamp", "")),
        request_id=str(record.get("request_id", "")),
        case=str(record.get("case", "")),
        backend=str(record.get("backend", "")),
        model=str(record.get("model", "")),
        chunk_id=str(record.get("chunk_id", "")),
        cache_key=str(record.get("cache_key", "")),
        tier=str(record.get("tier", "unknown") or "unknown"),
        bytes=as_int(record.get("bytes")),
        latency_ms=as_float(record.get("latency_ms")),
        line_number=as_int(record.get("line_number")),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    )


def infer_workload_id(event: TraceEvent) -> str:
    if event.backend and event.model:
        return f"{event.backend}:{event.model}"
    return event.backend or event.model or "default"


def infer_chunk_key(event: TraceEvent) -> str:
    if event.chunk_id:
        return event.chunk_id
    if event.cache_key:
        return event.cache_key
    if event.metadata.get("prefix_id"):
        return str(event.metadata["prefix_id"])
    if event.case and event.category in {"kv", "prefetch", "memory"}:
        return event.case
    return ""


def _quality_guard_key(workload_id: str, chunk_id: str, layer_id: int | None) -> str:
    normalized_layer = "*" if layer_id is None else str(layer_id)
    return f"{workload_id}:{chunk_id}:{normalized_layer}"


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def max_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, int] = {}
    for key, item in value.items():
        try:
            output[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return output


__all__ = [
    "PROFILE_DB_VERSION",
    "ChunkProfile",
    "WorkloadProfile",
    "LayerSensitivityRecord",
    "QualityGuardRecord",
    "ProfileDB",
    "load_profile_from_trace_jsonl",
    "trace_event_from_record",
]
