"""Async prefetch engine skeleton.

The engine provides lifecycle bookkeeping around prefetch requests. It uses an
adapter callback for real work, so no runtime-specific logic is embedded here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from astrakv.kv_cache.metadata import MemoryTier


class PrefetchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class PrefetchRequest:
    chunk_id: str
    target_tier: MemoryTier = MemoryTier.GPU
    priority: int = 0
    request_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrefetchResult:
    request_id: str
    chunk_id: str
    status: PrefetchStatus
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


PrefetchAdapter = Callable[[PrefetchRequest], Awaitable[PrefetchResult] | PrefetchResult]


class AsyncPrefetchEngine:
    """Tracks asynchronous prefetch work through an adapter callback."""

    def __init__(self, adapter: PrefetchAdapter | None = None) -> None:
        self.adapter = adapter or self._noop_adapter
        self._requests: dict[str, PrefetchRequest] = {}
        self._statuses: dict[str, PrefetchStatus] = {}
        self._results: dict[str, PrefetchResult] = {}

    async def submit(self, request: PrefetchRequest) -> PrefetchResult:
        self._requests[request.request_id] = request
        self._statuses[request.request_id] = PrefetchStatus.RUNNING
        try:
            result = self.adapter(request)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            result = PrefetchResult(
                request_id=request.request_id,
                chunk_id=request.chunk_id,
                status=PrefetchStatus.FAILED,
                message=str(exc),
            )
        self._statuses[request.request_id] = result.status
        self._results[request.request_id] = result
        return result

    def submit_nowait(self, request: PrefetchRequest) -> asyncio.Task[PrefetchResult]:
        return asyncio.create_task(self.submit(request))

    def status(self, request_id: str) -> PrefetchStatus | None:
        return self._statuses.get(request_id)

    def result(self, request_id: str) -> PrefetchResult | None:
        return self._results.get(request_id)

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for request_id, request in sorted(self._requests.items()):
            result = self._results.get(request_id)
            records.append(
                {
                    "request_id": request_id,
                    "chunk_id": request.chunk_id,
                    "target_tier": request.target_tier.value,
                    "priority": request.priority,
                    "status": self._statuses.get(request_id, PrefetchStatus.PENDING).value,
                    "message": result.message if result is not None else "",
                    "metadata": dict(request.metadata),
                }
            )
        return records

    @staticmethod
    def _noop_adapter(request: PrefetchRequest) -> PrefetchResult:
        return PrefetchResult(
            request_id=request.request_id,
            chunk_id=request.chunk_id,
            status=PrefetchStatus.COMPLETED,
            message="noop prefetch adapter",
        )
