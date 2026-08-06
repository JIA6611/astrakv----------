"""Endpoint-level prefetch adapter for real OpenAI-compatible backends.

This module deliberately stays outside vLLM/LMCache internals. It issues real
HTTP requests to an already-running OpenAI-compatible endpoint. When that
endpoint is backed by vLLM + LMCache, warmup requests can populate or retrieve
prefix KV through the normal runtime path, and server logs can be parsed later
as cache-event evidence.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from astrakv.kv_cache.metadata import MemoryTier
from astrakv.prefetch.async_engine import PrefetchRequest, PrefetchResult, PrefetchStatus


@dataclass(frozen=True, slots=True)
class EndpointRequest:
    request_id: str
    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EndpointResult:
    request_id: str
    status: str
    latency_ms: float
    ttft_ms: float | None
    output_tokens_observed: int
    throughput_tokens_s: float
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "output_tokens_observed": self.output_tokens_observed,
            "throughput_tokens_s": self.throughput_tokens_s,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class OpenAIEndpointClient:
    """Tiny streaming client for vLLM's OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout_seconds: float = 600.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def chat_completion(self, request: EndpointRequest) -> EndpointResult:
        started = time.perf_counter()
        ttft_ms: float | None = None
        observed_tokens = 0
        usage_tokens: int | None = None
        status = "ok"
        error = ""

        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            for event in self._stream_chat_completion(payload):
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    usage_tokens = int(usage["completion_tokens"])

                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if content:
                        observed_tokens += 1
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - endpoint diagnostics need the concrete message.
            status = "error"
            error = classify_endpoint_error(exc)

        ended = time.perf_counter()
        latency_ms = (ended - started) * 1000.0
        output_tokens = usage_tokens if usage_tokens is not None else observed_tokens
        throughput = output_tokens / max(ended - started, 1e-9)
        return EndpointResult(
            request_id=request.request_id,
            status=status,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            output_tokens_observed=output_tokens,
            throughput_tokens_s=throughput,
            error=error,
            metadata=dict(request.metadata),
        )

    def _stream_chat_completion(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


class EndpointPrefetchAdapter:
    """AsyncPrefetchEngine adapter that turns prefetch hints into real requests."""

    def __init__(self, client: OpenAIEndpointClient) -> None:
        self.client = client

    async def __call__(self, request: PrefetchRequest) -> PrefetchResult:
        endpoint_request = request.metadata.get("endpoint_request")
        if not isinstance(endpoint_request, EndpointRequest):
            return PrefetchResult(
                request_id=request.request_id,
                chunk_id=request.chunk_id,
                status=PrefetchStatus.FAILED,
                message="missing endpoint_request metadata",
                metadata={"target_tier": request.target_tier.value},
            )

        result = await asyncio.to_thread(self.client.chat_completion, endpoint_request)
        status = PrefetchStatus.COMPLETED if result.ok else PrefetchStatus.FAILED
        return PrefetchResult(
            request_id=request.request_id,
            chunk_id=request.chunk_id,
            status=status,
            message="endpoint warmup completed" if result.ok else result.error,
            metadata={
                "target_tier": request.target_tier.value,
                "endpoint_result": result.to_record(),
            },
        )


def make_prefetch_request(
    *,
    chunk_id: str,
    endpoint_request: EndpointRequest,
    target_tier: MemoryTier = MemoryTier.GPU,
    priority: int = 0,
    metadata: dict[str, Any] | None = None,
) -> PrefetchRequest:
    payload = dict(metadata or {})
    payload["endpoint_request"] = endpoint_request
    return PrefetchRequest(
        chunk_id=chunk_id,
        target_tier=target_tier,
        priority=priority,
        metadata=payload,
    )


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[:-3].rstrip("/")
    return cleaned


def classify_endpoint_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        return f"Endpoint HTTP {exc.code}: {detail}"
    if isinstance(exc, urllib.error.URLError):
        return f"Endpoint connection error: {exc.reason}"
    return f"Endpoint runtime error: {type(exc).__name__}: {exc}"
