"""Partial KV load planning primitives.

This module describes partial load intent for KV chunks. It does not allocate
tensors, read cache payloads, or call vLLM/LMCache internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4

from .metadata import KVChunkMeta, MemoryTier


PARTIAL_LOAD_SCHEMA_VERSION = "astra-partial-kv-load-v1"


class PartialLoadAction(str, Enum):
    LOAD_FULL = "load_full"
    LOAD_PARTIAL = "load_partial"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class TokenSpan:
    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token < self.start_token:
            raise ValueError("end_token must be greater than or equal to start_token")

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def overlaps(self, other: "TokenSpan") -> bool:
        return self.start_token < other.end_token and other.start_token < self.end_token

    def intersection(self, other: "TokenSpan") -> "TokenSpan | None":
        if not self.overlaps(other):
            return None
        return TokenSpan(max(self.start_token, other.start_token), min(self.end_token, other.end_token))

    def to_record(self) -> dict[str, int]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class PartialKVLoadRequest:
    request_id: str
    target_layers: tuple[int, ...] = field(default_factory=tuple)
    token_spans: tuple[TokenSpan, ...] = field(default_factory=tuple)
    target_tier: MemoryTier = MemoryTier.GPU
    plan_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target_tier, MemoryTier):
            object.__setattr__(self, "target_tier", MemoryTier(str(self.target_tier)))
        if any(layer < 0 for layer in self.target_layers):
            raise ValueError("target_layers must be non-negative")

    def layer_selected(self, layer_id: int) -> bool:
        return not self.target_layers or layer_id in set(self.target_layers)

    def selected_span_for(self, chunk: KVChunkMeta) -> TokenSpan | None:
        selected = self.selected_spans_for(chunk)
        if not selected:
            return None
        return TokenSpan(selected[0].start_token, selected[-1].end_token)

    def selected_spans_for(self, chunk: KVChunkMeta) -> tuple[TokenSpan, ...]:
        chunk_span = TokenSpan(chunk.start_token, chunk.end_token)
        if not self.token_spans:
            return (chunk_span,)
        intersections = [span.intersection(chunk_span) for span in self.token_spans]
        selected = [span for span in intersections if span is not None and span.token_count > 0]
        if not selected:
            return ()
        return merge_overlapping_token_spans(selected)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PARTIAL_LOAD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "target_layers": list(self.target_layers),
            "token_spans": [span.to_record() for span in self.token_spans],
            "target_tier": self.target_tier.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PartialKVLoadDecision:
    plan_id: str
    chunk_id: str
    request_id: str
    layer_id: int
    chunk_start_token: int
    chunk_end_token: int
    action: PartialLoadAction
    selected_start_token: int | None = None
    selected_end_token: int | None = None
    selected_spans: tuple[TokenSpan, ...] = field(default_factory=tuple)
    loaded_tokens: int = 0
    skipped_tokens: int = 0
    loaded_bytes: int = 0
    skipped_bytes: int = 0
    target_tier: MemoryTier = MemoryTier.GPU
    source_tier: MemoryTier = MemoryTier.UNKNOWN
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_tokens(self) -> int:
        return self.chunk_end_token - self.chunk_start_token

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PARTIAL_LOAD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "chunk_id": self.chunk_id,
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "chunk_start_token": self.chunk_start_token,
            "chunk_end_token": self.chunk_end_token,
            "chunk_tokens": self.chunk_tokens,
            "action": self.action.value,
            "selected_start_token": self.selected_start_token,
            "selected_end_token": self.selected_end_token,
            "selected_spans": [span.to_record() for span in self.selected_spans],
            "loaded_tokens": self.loaded_tokens,
            "skipped_tokens": self.skipped_tokens,
            "loaded_bytes": self.loaded_bytes,
            "skipped_bytes": self.skipped_bytes,
            "target_tier": self.target_tier.value,
            "source_tier": self.source_tier.value,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PartialKVLoadTarget:
    """Stable online handoff target for contiguous prefix-aligned partial loads."""

    plan_id: str
    request_id: str
    chunk_id: str
    object_key: str = ""
    object_level: str = "prefix"
    layer_id: int | None = None
    token_span: TokenSpan | None = None
    selected_tokens: int = 0
    total_tokens: int = 0
    target_tier: MemoryTier = MemoryTier.GPU
    source_tier: MemoryTier = MemoryTier.UNKNOWN
    allow_partial: bool = True
    prefix_aligned: bool = True
    contiguous: bool = True
    requires_recompute_fallback: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_end_token(self) -> int | None:
        return None if self.token_span is None else self.token_span.end_token

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PARTIAL_LOAD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "chunk_id": self.chunk_id,
            "object_key": self.object_key,
            "object_level": self.object_level,
            "layer_id": self.layer_id,
            "token_span": None if self.token_span is None else self.token_span.to_record(),
            "selected_tokens": self.selected_tokens,
            "total_tokens": self.total_tokens,
            "target_tier": self.target_tier.value,
            "source_tier": self.source_tier.value,
            "allow_partial": self.allow_partial,
            "prefix_aligned": self.prefix_aligned,
            "contiguous": self.contiguous,
            "requires_recompute_fallback": self.requires_recompute_fallback,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_decision(
        cls,
        decision: "PartialKVLoadDecision",
        *,
        object_key: str = "",
        object_level: str = "prefix",
        requires_recompute_fallback: bool = True,
    ) -> "PartialKVLoadTarget":
        span = None
        if decision.selected_start_token is not None and decision.selected_end_token is not None:
            span = TokenSpan(decision.selected_start_token, decision.selected_end_token)
        return cls(
            plan_id=decision.plan_id,
            request_id=decision.request_id,
            chunk_id=decision.chunk_id,
            object_key=object_key,
            object_level=object_level,
            layer_id=decision.layer_id,
            token_span=span,
            selected_tokens=decision.loaded_tokens,
            total_tokens=decision.chunk_tokens,
            target_tier=decision.target_tier,
            source_tier=decision.source_tier,
            allow_partial=decision.action is PartialLoadAction.LOAD_PARTIAL,
            prefix_aligned=(span.start_token == 0) if span is not None else False,
            contiguous=len(decision.selected_spans) <= 1,
            requires_recompute_fallback=requires_recompute_fallback,
            metadata=dict(decision.metadata),
        )


@dataclass(frozen=True, slots=True)
class PartialKVLoadSummary:
    plan_id: str
    request_id: str
    total_chunks: int
    load_full: int
    load_partial: int
    skip: int
    loaded_tokens: int
    skipped_tokens: int
    loaded_bytes: int
    skipped_bytes: int

    @property
    def byte_saving_rate(self) -> float:
        total = self.loaded_bytes + self.skipped_bytes
        return self.skipped_bytes / max(1, total)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PARTIAL_LOAD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "total_chunks": self.total_chunks,
            "load_full": self.load_full,
            "load_partial": self.load_partial,
            "skip": self.skip,
            "loaded_tokens": self.loaded_tokens,
            "skipped_tokens": self.skipped_tokens,
            "loaded_bytes": self.loaded_bytes,
            "skipped_bytes": self.skipped_bytes,
            "byte_saving_rate": self.byte_saving_rate,
        }


class PartialKVLoadPlanner:
    """Creates partial load decisions from chunk metadata."""

    def plan(
        self,
        chunks: Iterable[KVChunkMeta],
        request: PartialKVLoadRequest,
    ) -> list[PartialKVLoadDecision]:
        decisions = [self._decide(chunk, request) for chunk in chunks if chunk.request_id == request.request_id]
        return sorted(decisions, key=lambda item: (item.layer_id, item.chunk_start_token, item.chunk_id))

    def summarize(
        self,
        decisions: Iterable[PartialKVLoadDecision],
        *,
        plan_id: str,
        request_id: str,
    ) -> PartialKVLoadSummary:
        decision_list = list(decisions)
        action_counts: dict[PartialLoadAction, int] = {action: 0 for action in PartialLoadAction}
        loaded_tokens = 0
        skipped_tokens = 0
        loaded_bytes = 0
        skipped_bytes = 0
        for decision in decision_list:
            action_counts[decision.action] += 1
            loaded_tokens += decision.loaded_tokens
            skipped_tokens += decision.skipped_tokens
            loaded_bytes += decision.loaded_bytes
            skipped_bytes += decision.skipped_bytes
        return PartialKVLoadSummary(
            plan_id=plan_id,
            request_id=request_id,
            total_chunks=len(decision_list),
            load_full=action_counts[PartialLoadAction.LOAD_FULL],
            load_partial=action_counts[PartialLoadAction.LOAD_PARTIAL],
            skip=action_counts[PartialLoadAction.SKIP],
            loaded_tokens=loaded_tokens,
            skipped_tokens=skipped_tokens,
            loaded_bytes=loaded_bytes,
            skipped_bytes=skipped_bytes,
        )

    def _decide(self, chunk: KVChunkMeta, request: PartialKVLoadRequest) -> PartialKVLoadDecision:
        chunk_tokens = chunk.token_count
        source_bytes = int(chunk.size_bytes or 0)
        if not request.layer_selected(chunk.layer_id):
            return self._skip_decision(chunk, request, source_bytes, "layer_not_selected")

        selected_spans = request.selected_spans_for(chunk)
        if not selected_spans:
            return self._skip_decision(chunk, request, source_bytes, "token_span_not_selected")

        loaded_tokens = sum(span.token_count for span in selected_spans)
        skipped_tokens = max(0, chunk_tokens - loaded_tokens)
        loaded_bytes = estimate_bytes(source_bytes, loaded_tokens, chunk_tokens)
        skipped_bytes = max(0, source_bytes - loaded_bytes)
        if loaded_tokens >= chunk_tokens:
            action = PartialLoadAction.LOAD_FULL
            reason = "full_chunk_selected"
        else:
            action = PartialLoadAction.LOAD_PARTIAL
            reason = "partial_token_span_selected"
        selected_start = selected_spans[0].start_token
        selected_end = selected_spans[-1].end_token
        return PartialKVLoadDecision(
            plan_id=request.plan_id,
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            layer_id=chunk.layer_id,
            chunk_start_token=chunk.start_token,
            chunk_end_token=chunk.end_token,
            action=action,
            selected_start_token=selected_start,
            selected_end_token=selected_end,
            selected_spans=selected_spans,
            loaded_tokens=loaded_tokens,
            skipped_tokens=skipped_tokens,
            loaded_bytes=loaded_bytes,
            skipped_bytes=skipped_bytes,
            target_tier=request.target_tier,
            source_tier=chunk.tier,
            reason=reason,
            metadata={
                "cache_key": chunk.cache_key,
                "block_ids": list(chunk.block_ids),
                "dtype": chunk.dtype,
                "device": chunk.device,
                **dict(chunk.metadata),
            },
        )

    def _skip_decision(
        self,
        chunk: KVChunkMeta,
        request: PartialKVLoadRequest,
        source_bytes: int,
        reason: str,
    ) -> PartialKVLoadDecision:
        return PartialKVLoadDecision(
            plan_id=request.plan_id,
            chunk_id=chunk.chunk_id,
            request_id=chunk.request_id,
            layer_id=chunk.layer_id,
            chunk_start_token=chunk.start_token,
            chunk_end_token=chunk.end_token,
            action=PartialLoadAction.SKIP,
            loaded_tokens=0,
            skipped_tokens=chunk.token_count,
            loaded_bytes=0,
            skipped_bytes=source_bytes,
            target_tier=request.target_tier,
            source_tier=chunk.tier,
            reason=reason,
            metadata={
                "cache_key": chunk.cache_key,
                "block_ids": list(chunk.block_ids),
                "dtype": chunk.dtype,
                "device": chunk.device,
                **dict(chunk.metadata),
            },
        )


def merge_token_spans(spans: Iterable[TokenSpan]) -> TokenSpan:
    span_list = sorted(spans, key=lambda item: (item.start_token, item.end_token))
    if not span_list:
        raise ValueError("cannot merge empty token spans")
    return TokenSpan(span_list[0].start_token, max(span.end_token for span in span_list))


def merge_overlapping_token_spans(spans: Iterable[TokenSpan]) -> tuple[TokenSpan, ...]:
    span_list = sorted(spans, key=lambda item: (item.start_token, item.end_token))
    if not span_list:
        return ()
    merged: list[TokenSpan] = [span_list[0]]
    for span in span_list[1:]:
        current = merged[-1]
        if span.start_token <= current.end_token:
            merged[-1] = TokenSpan(current.start_token, max(current.end_token, span.end_token))
        else:
            merged.append(span)
    return tuple(merged)


def build_partial_load_targets(
    decisions: Iterable[PartialKVLoadDecision],
    *,
    object_key: str = "",
    object_level: str = "prefix",
    require_prefix_aligned: bool = True,
    require_contiguous: bool = True,
    requires_recompute_fallback: bool = True,
) -> list[PartialKVLoadTarget]:
    targets: list[PartialKVLoadTarget] = []
    for decision in decisions:
        if decision.action is PartialLoadAction.SKIP:
            continue
        target = PartialKVLoadTarget.from_decision(
            decision,
            object_key=object_key,
            object_level=object_level,
            requires_recompute_fallback=requires_recompute_fallback,
        )
        if require_prefix_aligned and not target.prefix_aligned:
            continue
        if require_contiguous and not target.contiguous:
            continue
        targets.append(target)
    return targets


def estimate_bytes(source_bytes: int, selected_tokens: int, total_tokens: int) -> int:
    if source_bytes <= 0 or total_tokens <= 0 or selected_tokens <= 0:
        return 0
    return min(source_bytes, max(1, round(source_bytes * selected_tokens / total_tokens)))


def chunk_meta_from_record(record: dict[str, Any]) -> KVChunkMeta:
    return KVChunkMeta(
        request_id=str(record.get("request_id", "")),
        layer_id=int(record.get("layer_id", 0)),
        start_token=int(record.get("start_token", record.get("chunk_start_token", 0))),
        end_token=int(record.get("end_token", record.get("chunk_end_token", 0))),
        block_ids=tuple(int(item) for item in record.get("block_ids", []) if str(item) != ""),
        chunk_id=str(record.get("chunk_id") or uuid4().hex),
        tier=memory_tier_from(record.get("tier", record.get("source_tier", "unknown"))),
        dtype=none_if_empty(record.get("dtype")),
        device=none_if_empty(record.get("device")),
        size_bytes=optional_int(record.get("size_bytes")),
        cache_key=none_if_empty(record.get("cache_key")),
        adapter_name=none_if_empty(record.get("adapter_name")),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    )


def memory_tier_from(value: Any) -> MemoryTier:
    normalized = str(value or "unknown").strip().lower()
    aliases = {"disk": "ssd", "local_disk": "ssd", "local_cpu": "cpu", "cuda": "gpu"}
    normalized = aliases.get(normalized, normalized)
    try:
        return MemoryTier(normalized)
    except ValueError:
        return MemoryTier.UNKNOWN


def none_if_empty(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def optional_int(value: Any) -> int | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
