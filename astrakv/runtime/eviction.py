"""Contracts and adapters for offline-policy/runtime-eviction validation.

The real vLLM/LMCache path is observational by default.  It may normalize
runtime artifacts, but it never treats a scheduler hint as an executed action.
The mmap adapter is a separate, executable OS-VM proof-of-concept path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from astrakv.runtime.cache_events import CacheEvent, parse_server_log
if TYPE_CHECKING:
    from astrakv.runtime.backend_hook import BackendActionReceipt
    from astrakv.vm.dgx_spark_adapter import DgxSparkKVAdapter


class ObjectLevel(str, Enum):
    PREFIX = "prefix"
    CACHE_KEY = "cache_key"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    object_levels: tuple[ObjectLevel, ...] = (ObjectLevel.PREFIX,)
    can_collect_events: bool = True
    can_execute_eviction: bool = False
    can_acknowledge_execution: bool = False
    can_report_tier_transition: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "object_levels": [item.value for item in self.object_levels],
            "can_collect_events": self.can_collect_events,
            "can_execute_eviction": self.can_execute_eviction,
            "can_acknowledge_execution": self.can_acknowledge_execution,
            "can_report_tier_transition": self.can_report_tier_transition,
        }


@dataclass(frozen=True, slots=True)
class OfflineEvictionDecision:
    run_id: str
    decision_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    predicted_action: str
    target_tier: str = "unknown"
    bytes: int | None = None
    decision_time_ns: int | None = None
    decision_index: int | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicts_eviction(self) -> bool:
        return self.predicted_action in {"offload", "drop", "evict"}

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "predicted_action": self.predicted_action,
            "target_tier": self.target_tier,
            "bytes": self.bytes,
            "decision_time_ns": self.decision_time_ns,
            "decision_index": self.decision_index,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeEvictionEvent:
    run_id: str
    runtime_event_id: str
    request_id: str
    object_key: str
    object_level: ObjectLevel
    actual_action: str
    tier_before: str = "unknown"
    tier_after: str = "unknown"
    bytes: int | None = None
    timestamp_ns: int | None = None
    arrival_index: int | None = None
    status: str = "observed"
    provenance: str = "runtime_structured"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_eviction(self) -> bool:
        return self.actual_action in {"offload", "evict", "drop"}

    @property
    def is_ground_truth(self) -> bool:
        return self.provenance in {"runtime_structured", "vm_poc_execution"} and self.status in {
            "completed",
            "ok",
            "executed",
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runtime_event_id": self.runtime_event_id,
            "request_id": self.request_id,
            "object_key": self.object_key,
            "object_level": self.object_level.value,
            "actual_action": self.actual_action,
            "tier_before": self.tier_before,
            "tier_after": self.tier_after,
            "bytes": self.bytes,
            "timestamp_ns": self.timestamp_ns,
            "arrival_index": self.arrival_index,
            "status": self.status,
            "provenance": self.provenance,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeActionResult:
    status: str
    message: str
    event: RuntimeEvictionEvent | None = None
    receipt: BackendActionReceipt | None = None


def offline_decision_from_record(record: dict[str, Any], *, run_id: str, ordinal: int = 0) -> OfflineEvictionDecision:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else _metadata_from_text(record.get("metadata"))
    if str(record.get("legacy_unlinked") or "").lower() in {"1", "true", "yes"}:
        metadata = {"legacy_unlinked": True, **metadata}
    explicit_key = str(record.get("object_key") or "")
    explicit_level = str(record.get("object_level") or "")
    prefix_id = str(metadata.get("prefix_id") or record.get("prefix_id") or "")
    cache_key = str(record.get("cache_key") or metadata.get("cache_key") or "")
    case = str(record.get("case") or metadata.get("case") or "")
    chunk_id = str(record.get("chunk_id") or "")
    if explicit_key and explicit_level in {item.value for item in ObjectLevel}:
        key, level = explicit_key, ObjectLevel(explicit_level)
    elif prefix_id:
        key, level = prefix_id, ObjectLevel.PREFIX
    elif cache_key:
        key, level = cache_key, ObjectLevel.CACHE_KEY
    else:
        # Historical scheduler records use case/chunk IDs as logical objects;
        # they must not be advertised as physical runtime blocks.
        key, level = case or chunk_id, ObjectLevel.PREFIX
        metadata = {"legacy_unlinked": True, "legacy_logical_object": True, **metadata}
    return OfflineEvictionDecision(
        run_id=run_id,
        decision_id=str(record.get("decision_id") or f"offline-{ordinal}"),
        request_id=str(record.get("request_id") or metadata.get("request_id") or case or chunk_id),
        object_key=key,
        object_level=level,
        predicted_action=str(record.get("action") or ""),
        target_tier=str(record.get("target_tier") or metadata.get("target_tier") or "unknown"),
        bytes=_as_int(record.get("size_bytes") or record.get("bytes")),
        decision_time_ns=_as_int(record.get("decision_time_ns")),
        decision_index=_as_int(record.get("arrival_index") or metadata.get("arrival_index")),
        reason=str(record.get("reason") or ""),
        metadata=metadata,
    )


class VllmLmCacheArtifactAdapter:
    """Read-only adapter for vLLM/LMCache logs and exported cache events."""

    name = "vllm_lmcache_artifact"

    def __init__(self, *, run_id: str, request_objects: dict[str, dict[str, Any]] | None = None) -> None:
        self.run_id = run_id
        self.request_objects = request_objects or {}
        self.last_skipped_evidence: list[dict[str, Any]] = []

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(object_levels=(ObjectLevel.PREFIX, ObjectLevel.CACHE_KEY))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "mode": "observational", **self.capabilities().to_record()}

    def collect_runtime_events(self, events: Iterable[CacheEvent]) -> list[RuntimeEvictionEvent]:
        self.last_skipped_evidence = []
        normalized: list[RuntimeEvictionEvent] = []
        for index, raw in enumerate(events):
            event = self._normalize(raw, index)
            if event is not None:
                normalized.append(event)
            elif raw.event_type in {"cache_offload", "cache_evict"}:
                # The source artifact remains untouched.  Record why it cannot
                # participate in the standardized comparison denominator.
                self.last_skipped_evidence.append({
                    "event_type": raw.event_type,
                    "source": raw.source,
                    "line_number": raw.line_number,
                    "request_id": raw.request_id,
                    "reason": "missing_stable_request_or_object_association",
                })
        return normalized

    def collect_from_paths(self, *, server_logs: Iterable[str | Path] = (), cache_events: Iterable[str | Path] = ()) -> list[RuntimeEvictionEvent]:
        raw: list[CacheEvent] = []
        for path in server_logs:
            raw.extend(parse_server_log(path))
        for path in cache_events:
            raw.extend(_load_cache_events(path))
        return self.collect_runtime_events(raw)

    def apply_hint(self, _decision: OfflineEvictionDecision) -> RuntimeActionResult:
        return RuntimeActionResult(
            status="unsupported",
            message="vLLM/LMCache artifact adapter is observational; no third-party runtime action was invoked",
        )

    def _normalize(self, raw: CacheEvent, ordinal: int) -> RuntimeEvictionEvent | None:
        action = {"cache_offload": "offload", "cache_evict": "evict"}.get(raw.event_type)
        if action is None:
            return None
        request_meta = self.request_objects.get(raw.request_id, {})
        cache_key = raw.cache_key or str(request_meta.get("cache_key") or "")
        prefix_id = str(request_meta.get("prefix_id") or request_meta.get("case") or "")
        if raw.chunk_id:
            # Log-derived chunk IDs are treated as cache keys unless a structured
            # adapter explicitly promotes them to block-level evidence.
            object_key, level = raw.chunk_id, ObjectLevel.CACHE_KEY
        elif cache_key:
            object_key, level = cache_key, ObjectLevel.CACHE_KEY
        elif prefix_id:
            object_key, level = prefix_id, ObjectLevel.PREFIX
        else:
            return None
        return RuntimeEvictionEvent(
            run_id=self.run_id,
            runtime_event_id=f"artifact-{ordinal}-{raw.line_number or 0}",
            request_id=raw.request_id,
            object_key=object_key,
            object_level=level,
            actual_action=action,
            tier_before="unknown",
            tier_after=raw.tier,
            bytes=raw.bytes,
            timestamp_ns=_timestamp_to_ns(raw.end_time or raw.start_time),
            arrival_index=_as_int(request_meta.get("arrival_index")),
            status=raw.status,
            provenance="log_heuristic",
            metadata={"source": raw.source, "line_number": raw.line_number, "raw_event_type": raw.event_type},
        )


class MMapEvictionAdapter:
    """Executable adapter for the standalone mmap/DGX virtual-memory PoC."""

    name = "mmap_eviction_poc"

    def __init__(
        self,
        adapter: DgxSparkKVAdapter,
        *,
        run_id: str,
        object_bindings: dict[tuple[ObjectLevel, str], str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.run_id = run_id
        # A binding states that a ProfileDB logical object is represented by a
        # particular mmap PoC chunk.  It is deliberately explicit: no vLLM
        # tensor/block identity may be inferred from this standalone mapping.
        self.object_bindings = object_bindings or {}

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            object_levels=(ObjectLevel.PREFIX, ObjectLevel.CACHE_KEY, ObjectLevel.BLOCK),
            can_execute_eviction=True,
            can_acknowledge_execution=True,
            can_report_tier_transition=True,
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "mode": "vm_poc_execution", **self.capabilities().to_record()}

    def apply_hint(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        if decision.predicted_action == "prefetch":
            return self.prefetch_object(decision)
        if not decision.predicts_eviction:
            return RuntimeActionResult("unsupported", f"action {decision.predicted_action} is not an eviction action")
        mmap_chunk_id = self.object_bindings.get((decision.object_level, decision.object_key), decision.object_key)
        try:
            action = self.adapter.evict_chunk(mmap_chunk_id)
        except KeyError as exc:
            event = RuntimeEvictionEvent(
                run_id=self.run_id,
                runtime_event_id=f"vm-{decision.decision_id}",
                request_id=decision.request_id,
                object_key=decision.object_key,
                object_level=decision.object_level,
                actual_action="evict",
                timestamp_ns=time.time_ns(),
                arrival_index=decision.decision_index,
                status="failed",
                provenance="vm_poc_execution",
                metadata={
                    "mmap_chunk_id": mmap_chunk_id,
                    "execution_scope": "vm_poc_execution",
                    "error": str(exc),
                },
            )
            return RuntimeActionResult("failed", f"unknown mmap chunk: {mmap_chunk_id}", event)
        event = RuntimeEvictionEvent(
            run_id=self.run_id,
            runtime_event_id=f"vm-{decision.decision_id}",
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
            actual_action="evict",
            tier_before="os_page_cache",
            tier_after=self.adapter.config.target_tier.value,
            bytes=None,
            timestamp_ns=time.time_ns(),
            arrival_index=decision.decision_index,
            status="executed" if action.ok else "failed",
            provenance="vm_poc_execution",
            metadata={
                "resident_ratio": action.resident_ratio,
                "latency_us": action.latency_us,
                "block_ids": list(action.block_ids),
                "mmap_chunk_id": mmap_chunk_id,
                "execution_scope": "vm_poc_execution",
            },
        )
        return RuntimeActionResult(event.status, "mmap eviction executed" if action.ok else "mmap eviction failed", event)

    def prefetch_object(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        """Request Linux page-cache prefetch for one explicitly bound PoC object."""

        return self._execute_non_evict_action(decision, action_name="prefetch", adapter_method="prefetch_chunk")

    def read_object(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        """Read one PoC object and record the on-demand page-cache acknowledgement."""

        return self._execute_non_evict_action(decision, action_name="read", adapter_method="read_chunk")

    def _execute_non_evict_action(
        self,
        decision: OfflineEvictionDecision,
        *,
        action_name: str,
        adapter_method: str,
    ) -> RuntimeActionResult:
        mmap_chunk_id = self.object_bindings.get((decision.object_level, decision.object_key), decision.object_key)
        try:
            raw_result = getattr(self.adapter, adapter_method)(mmap_chunk_id)
            action = raw_result[1] if adapter_method == "read_chunk" else raw_result
        except KeyError as exc:
            event = RuntimeEvictionEvent(
                run_id=self.run_id,
                runtime_event_id=f"vm-{action_name}-{decision.decision_id}",
                request_id=decision.request_id,
                object_key=decision.object_key,
                object_level=decision.object_level,
                actual_action=action_name,
                timestamp_ns=time.time_ns(),
                arrival_index=decision.decision_index,
                status="failed",
                provenance="vm_poc_execution",
                metadata={"mmap_chunk_id": mmap_chunk_id, "execution_scope": "vm_poc_execution", "error": str(exc)},
            )
            return RuntimeActionResult("failed", f"unknown mmap chunk: {mmap_chunk_id}", event)
        event = RuntimeEvictionEvent(
            run_id=self.run_id,
            runtime_event_id=f"vm-{action_name}-{decision.decision_id}",
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
            actual_action=action_name,
            tier_before=self.adapter.config.target_tier.value if action_name == "prefetch" else "os_page_cache",
            tier_after="os_page_cache",
            timestamp_ns=time.time_ns(),
            arrival_index=decision.decision_index,
            status="executed" if action.ok else "failed",
            provenance="vm_poc_execution",
            metadata={
                "resident_ratio": action.resident_ratio,
                "latency_us": action.latency_us,
                "block_ids": list(action.block_ids),
                "mmap_chunk_id": mmap_chunk_id,
                "execution_scope": "vm_poc_execution",
            },
        )
        return RuntimeActionResult(event.status, f"mmap {action_name} executed" if action.ok else f"mmap {action_name} failed", event)


def write_runtime_events_jsonl(path: str | Path, events: Iterable[RuntimeEvictionEvent]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def load_runtime_events_jsonl(path: str | Path) -> list[RuntimeEvictionEvent]:
    result: list[RuntimeEvictionEvent] = []
    input_path = Path(path)
    if not input_path.exists():
        return result
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
            result.append(runtime_event_from_record(item))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return result


def runtime_event_from_record(record: dict[str, Any]) -> RuntimeEvictionEvent:
    return RuntimeEvictionEvent(
        run_id=str(record.get("run_id") or ""),
        runtime_event_id=str(record.get("runtime_event_id") or ""),
        request_id=str(record.get("request_id") or ""),
        object_key=str(record.get("object_key") or ""),
        object_level=ObjectLevel(str(record.get("object_level") or ObjectLevel.PREFIX.value)),
        actual_action=str(record.get("actual_action") or ""),
        tier_before=str(record.get("tier_before") or "unknown"),
        tier_after=str(record.get("tier_after") or "unknown"),
        bytes=_as_int(record.get("bytes")),
        timestamp_ns=_as_int(record.get("timestamp_ns")),
        arrival_index=_as_int(record.get("arrival_index")),
        status=str(record.get("status") or "observed"),
        provenance=str(record.get("provenance") or "runtime_structured"),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    )


def _load_cache_events(path: str | Path) -> list[CacheEvent]:
    result: list[CacheEvent] = []
    input_path = Path(path)
    if not input_path.exists():
        return result
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        result.append(CacheEvent(
            event_type=str(item.get("event_type") or ""), source=str(item.get("source") or str(input_path)),
            status=str(item.get("status") or "observed"), request_id=str(item.get("request_id") or ""),
            cache_key=str(item.get("cache_key") or ""), chunk_id=str(item.get("chunk_id") or ""),
            tier=str(item.get("tier") or "unknown"), bytes=_as_int(item.get("bytes")),
            start_time=str(item.get("start_time") or ""), end_time=str(item.get("end_time") or ""),
            latency_ms=_as_float(item.get("latency_ms")), line_number=_as_int(item.get("line_number")),
            raw_line=str(item.get("raw_line") or ""), metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        ))
    return result


def _metadata_from_text(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _as_int(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_to_ns(value: str) -> int | None:
    try:
        return int(float(value) * 1_000_000_000)
    except (TypeError, ValueError):
        return None
