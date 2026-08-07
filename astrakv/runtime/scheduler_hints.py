"""Runtime-facing loader for offline scheduler hints.

This keeps the offline policy chain and the online controller on one stable
interchange format: `SchedulerHint` records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from astrakv.scheduler.hints import SchedulerHint


@dataclass(slots=True)
class SchedulerHintIndex:
    by_request: dict[str, list[SchedulerHint]] = field(default_factory=dict)
    by_backend_object: dict[str, list[SchedulerHint]] = field(default_factory=dict)
    by_object_key: dict[str, list[SchedulerHint]] = field(default_factory=dict)
    by_cache_key: dict[str, list[SchedulerHint]] = field(default_factory=dict)
    by_prefix_id: dict[str, list[SchedulerHint]] = field(default_factory=dict)
    by_prefix_hash: dict[str, list[SchedulerHint]] = field(default_factory=dict)

    @classmethod
    def from_hints(cls, hints: Iterable[SchedulerHint]) -> "SchedulerHintIndex":
        index = cls()
        for hint in hints:
            metadata = dict(hint.metadata)
            if hint.request_id:
                index.by_request.setdefault(hint.request_id, []).append(hint)
            backend_object_id = str(metadata.get("chunk_id") or metadata.get("backend_object_id") or "")
            if backend_object_id:
                index.by_backend_object.setdefault(backend_object_id, []).append(hint)
            object_key = str(metadata.get("object_key") or "")
            if object_key:
                index.by_object_key.setdefault(object_key, []).append(hint)
            cache_key = str(metadata.get("cache_key") or "")
            if cache_key:
                index.by_cache_key.setdefault(cache_key, []).append(hint)
            prefix_id = str(metadata.get("prefix_id") or "")
            if prefix_id:
                index.by_prefix_id.setdefault(prefix_id, []).append(hint)
            prefix_hash = str(metadata.get("prefix_hash") or "")
            if prefix_hash:
                index.by_prefix_hash.setdefault(prefix_hash, []).append(hint)
        for mapping in (
            index.by_request,
            index.by_backend_object,
            index.by_object_key,
            index.by_cache_key,
            index.by_prefix_id,
            index.by_prefix_hash,
        ):
            for key, rows in mapping.items():
                mapping[key] = sorted(
                    rows,
                    key=lambda item: (-int(item.priority), str(item.action), key, str(item.reason)),
                )
        return index

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SchedulerHintIndex":
        rows: list[SchedulerHint] = []
        input_path = Path(path)
        if not input_path.is_file():
            raise ValueError(f"scheduler hint JSONL not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"scheduler hint line {line_number} is invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"scheduler hint line {line_number} must be an object")
                rows.append(SchedulerHint(
                    request_id=str(record.get("request_id") or ""),
                    action=str(record.get("action") or ""),
                    reason=str(record.get("reason") or ""),
                    priority=int(record.get("priority") or 0),
                    metadata=dict(record.get("metadata") or {}),
                ))
        return cls.from_hints(rows)

    def hints_for_request(self, request_id: str) -> tuple[SchedulerHint, ...]:
        return tuple(self.by_request.get(request_id, ()))

    def best_hint_for_object(
        self,
        *,
        request_id: str,
        backend_object_id: str,
        object_key: str,
        prefix_id: str = "",
        prefix_hash: str = "",
    ) -> SchedulerHint | None:
        seen: set[int] = set()
        candidates = (
            self.by_backend_object.get(backend_object_id, ()),
            self.by_object_key.get(object_key, ()),
            self.by_cache_key.get(object_key, ()),
            self.by_prefix_id.get(prefix_id, ()) if prefix_id else (),
            self.by_prefix_id.get(object_key, ()) if object_key else (),
            self.by_prefix_hash.get(prefix_hash, ()) if prefix_hash else (),
            self.by_request.get(request_id, ()),
        )
        for bucket in candidates:
            for hint in bucket:
                marker = id(hint)
                if marker in seen:
                    continue
                seen.add(marker)
                metadata = hint.metadata
                if str(metadata.get("chunk_id") or metadata.get("backend_object_id") or "") == backend_object_id:
                    return hint
                if str(metadata.get("object_key") or "") == object_key:
                    return hint
                if str(metadata.get("cache_key") or "") == object_key:
                    return hint
                if str(metadata.get("prefix_id") or "") in {object_key, prefix_id}:
                    return hint
                if prefix_hash and str(metadata.get("prefix_hash") or "") == prefix_hash:
                    return hint
                if hint.request_id and hint.request_id == request_id:
                    return hint
        return None
