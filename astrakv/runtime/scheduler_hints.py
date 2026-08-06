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

    @classmethod
    def from_hints(cls, hints: Iterable[SchedulerHint]) -> "SchedulerHintIndex":
        index = cls()
        for hint in hints:
            index.by_request.setdefault(hint.request_id, []).append(hint)
        for request_id, rows in index.by_request.items():
            index.by_request[request_id] = sorted(
                rows,
                key=lambda item: (-int(item.priority), str(item.action), request_id),
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
    ) -> SchedulerHint | None:
        for hint in self.by_request.get(request_id, ()):  # already sorted by priority desc
            metadata = hint.metadata
            if str(metadata.get("chunk_id") or "") == backend_object_id:
                return hint
            if str(metadata.get("object_key") or "") == object_key:
                return hint
            if str(metadata.get("cache_key") or "") == object_key:
                return hint
            if str(metadata.get("prefix_id") or "") == object_key:
                return hint
        return None

