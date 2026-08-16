"""Request-ahead MoE prefill and routed-expert evidence collection.

The client sends one non-streaming request to the same vLLM endpoint before
the measured request begins.  This is a real MoE prefill that can populate the
configured KV backend.  It records router choices returned by vLLM, but it
does not claim selective expert-weight paging or migration.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from astrakv.runtime.artifact_contract import MOE_RUNTIME_ARTIFACT_NAMES
from astrakv.runtime.moe_events import MOE_EVENT_SCHEMA_VERSION, MoEExpertEvent


MOE_PREPARE_RECEIPT_SCHEMA = "astrakv-moe-prepare-receipt-v1"
MOE_ROUTE_SUMMARY_SCHEMA = "astrakv-moe-route-summary-v1"
MOE_RAW_MANIFEST_SCHEMA = "astrakv-moe-routed-experts-manifest-v1"


@dataclass(frozen=True, slots=True)
class MoePrepareConfig:
    enabled: bool = False
    mode: str = "full_prefix_prefill"
    max_tokens: int = 1
    capture_window_tokens: int = 256
    timeout_seconds: float = 600.0
    require_exact_token_ids: bool = True
    fail_open: bool = True
    expected_layers: int | None = 40
    expected_top_k: int | None = 8
    max_expert_id: int | None = 255
    model_type: str = "qwen3_5_moe"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MoePrepareConfig":
        raw = dict(value or {})
        config = cls(
            enabled=_as_bool(raw.get("enabled"), False),
            mode=str(raw.get("mode") or "full_prefix_prefill"),
            max_tokens=int(raw.get("max_tokens") or 1),
            capture_window_tokens=int(raw.get("capture_window_tokens") or 256),
            timeout_seconds=float(raw.get("timeout_seconds") or 600.0),
            require_exact_token_ids=_as_bool(raw.get("require_exact_token_ids"), True),
            fail_open=_as_bool(raw.get("fail_open"), True),
            expected_layers=_optional_int(raw.get("expected_layers"), default=40),
            expected_top_k=_optional_int(raw.get("expected_top_k"), default=8),
            max_expert_id=_optional_int(raw.get("max_expert_id"), default=255),
            model_type=str(raw.get("model_type") or "qwen3_5_moe"),
        )
        if config.mode != "full_prefix_prefill":
            raise ValueError(f"unsupported MoE prepare mode: {config.mode}")
        if config.max_tokens != 1:
            raise ValueError("MoE prepare max_tokens must be exactly 1")
        if config.capture_window_tokens <= 0:
            raise ValueError("MoE capture_window_tokens must be positive")
        if config.timeout_seconds <= 0:
            raise ValueError("MoE prepare timeout_seconds must be positive")
        return config

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "capture_window_tokens": self.capture_window_tokens,
            "timeout_seconds": self.timeout_seconds,
            "require_exact_token_ids": self.require_exact_token_ids,
            "fail_open": self.fail_open,
            "expected_layers": self.expected_layers,
            "expected_top_k": self.expected_top_k,
            "max_expert_id": self.max_expert_id,
            "model_type": self.model_type,
        }


@dataclass(frozen=True, slots=True)
class MoePrepareResult:
    status: str = "disabled"
    latency_ms: float = 0.0
    route_event_count: int = 0
    unique_experts: int = 0
    model_type: str = ""
    error: str = ""
    started_s: float = 0.0
    ended_s: float = 0.0
    captured_tokens: int = 0
    layers: int = 0
    top_k: int = 0
    raw_routing_path: str = ""
    raw_routing_sha256: str = ""


def decode_routed_experts(
    encoded: str,
    *,
    expected_layers: int | None = None,
    expected_top_k: int | None = None,
    max_expert_id: int | None = None,
) -> tuple[np.ndarray, bytes]:
    """Decode and validate vLLM's base64-encoded NumPy routed-expert array."""

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("response is missing routed_experts")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("routed_experts is not valid base64") from exc
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ValueError("routed_experts is not a valid NumPy array") from exc
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        raise ValueError("routed_experts must have shape [tokens, layers, top_k]")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError("routed_experts dimensions must be non-empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("routed_experts must use an integer dtype")
    if expected_layers is not None and array.shape[1] != expected_layers:
        raise ValueError(
            f"routed_experts layer count {array.shape[1]} does not match {expected_layers}"
        )
    if expected_top_k is not None and array.shape[2] != expected_top_k:
        raise ValueError(
            f"routed_experts top_k {array.shape[2]} does not match {expected_top_k}"
        )
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0:
        raise ValueError("routed_experts contains a negative expert id")
    if max_expert_id is not None and maximum > max_expert_id:
        raise ValueError(
            f"routed_experts expert id {maximum} exceeds {max_expert_id}"
        )
    return array, raw


def routed_experts_to_events(
    array: np.ndarray,
    *,
    request_id: str,
    token_start: int,
    source: str = "vllm_return_routed_experts",
    model: str = "",
) -> Iterable[MoEExpertEvent]:
    """Yield one normalized route event per token/layer/expert rank."""

    top_k = int(array.shape[2])
    for token_offset in range(int(array.shape[0])):
        for layer_id in range(int(array.shape[1])):
            for expert_rank in range(top_k):
                yield MoEExpertEvent(
                    event_type="expert_route",
                    source=source,
                    status="observed",
                    request_id=request_id,
                    token_index=token_start + token_offset,
                    layer_id=layer_id,
                    expert_id=str(int(array[token_offset, layer_id, expert_rank])),
                    expert_rank=expert_rank,
                    top_k=top_k,
                    tier="gpu",
                    metadata={"model": model, "evidence": "vllm_routed_experts"},
                )


class MoePrepareClient:
    """Run real request-ahead prefill and persist auditable MoE evidence."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        output_dir: str | Path,
        config: MoePrepareConfig,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.config = config
        self.raw_dir = self.output_dir / "moe_routed_experts"
        self._lock = threading.Lock()
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._route_event_count = 0
        self._unique_experts: set[int] = set()
        self._layer_expert_counts: dict[int, dict[int, int]] = {}
        self._raw_entries: list[dict[str, Any]] = []

    def prepare(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        request_id: str,
        exact_token_ids: tuple[int, ...],
        context_published: bool,
        runtime_association_status: str,
        prefix_id: str = "",
        prefix_hash: str = "",
        cache_key: str = "",
    ) -> MoePrepareResult:
        if not self.config.enabled:
            return MoePrepareResult()
        if not context_published:
            return self._record_skip(
                status="skipped_context_unpublished",
                request_id=request_id,
                model=model,
                runtime_association_status=runtime_association_status,
                prefix_id=prefix_id,
                prefix_hash=prefix_hash,
                cache_key=cache_key,
            )
        if self.config.require_exact_token_ids and not exact_token_ids:
            return self._record_skip(
                status="skipped_missing_exact_tokens",
                request_id=request_id,
                model=model,
                runtime_association_status=runtime_association_status,
                prefix_id=prefix_id,
                prefix_hash=prefix_hash,
                cache_key=cache_key,
            )

        prompt_tokens = len(exact_token_ids)
        token_start = max(0, prompt_tokens - self.config.capture_window_tokens)
        prepare_request_id = f"{request_id}:moe-prepare"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "user": prepare_request_id,
            "routed_experts_prompt_start": token_start,
        }
        started_s = time.time()
        started = time.perf_counter()
        try:
            response = self._post_json(
                f"{self.base_url}/v1/chat/completions",
                payload,
                request_id=prepare_request_id,
            )
            choice = _first_choice(response)
            array, raw = decode_routed_experts(
                choice.get("routed_experts", ""),
                expected_layers=self.config.expected_layers,
                expected_top_k=self.config.expected_top_k,
                max_expert_id=self.config.max_expert_id,
            )
            ended_s = time.time()
            latency_ms = (time.perf_counter() - started) * 1000.0
            result = self._record_success(
                array=array,
                raw=raw,
                request_id=request_id,
                prepare_request_id=prepare_request_id,
                model=model,
                token_start=token_start,
                prompt_tokens=prompt_tokens,
                runtime_association_status=runtime_association_status,
                prefix_id=prefix_id,
                prefix_hash=prefix_hash,
                cache_key=cache_key,
                started_s=started_s,
                ended_s=ended_s,
                latency_ms=latency_ms,
            )
            return result
        except Exception as exc:
            ended_s = time.time()
            latency_ms = (time.perf_counter() - started) * 1000.0
            error = f"{type(exc).__name__}: {exc}"
            result = self._record_failure(
                request_id=request_id,
                prepare_request_id=prepare_request_id,
                model=model,
                prompt_tokens=prompt_tokens,
                token_start=token_start,
                runtime_association_status=runtime_association_status,
                prefix_id=prefix_id,
                prefix_hash=prefix_hash,
                cache_key=cache_key,
                started_s=started_s,
                ended_s=ended_s,
                latency_ms=latency_ms,
                error=error,
            )
            if not self.config.fail_open:
                raise RuntimeError(error) from exc
            return result

    def _post_json(
        self, url: str, payload: dict[str, Any], *, request_id: str,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Request-Id": request_id,
            },
        )
        with urllib.request.urlopen(  # noqa: S310 - endpoint is explicit benchmark input.
            request, timeout=self.config.timeout_seconds,
        ) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("MoE prepare endpoint returned a non-object response")
        return decoded

    def _record_success(
        self,
        *,
        array: np.ndarray,
        raw: bytes,
        request_id: str,
        prepare_request_id: str,
        model: str,
        token_start: int,
        prompt_tokens: int,
        runtime_association_status: str,
        prefix_id: str,
        prefix_hash: str,
        cache_key: str,
        started_s: float,
        ended_s: float,
        latency_ms: float,
    ) -> MoePrepareResult:
        safe_id = _safe_filename(request_id)
        raw_path = self.raw_dir / f"{safe_id}.npy"
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        events = list(
            routed_experts_to_events(
                array,
                request_id=request_id,
                token_start=token_start,
                model=model,
            )
        )
        unique_experts = {int(value) for value in np.unique(array).tolist()}
        result = MoePrepareResult(
            status="prepared",
            latency_ms=latency_ms,
            route_event_count=len(events),
            unique_experts=len(unique_experts),
            model_type=self.config.model_type,
            started_s=started_s,
            ended_s=ended_s,
            captured_tokens=int(array.shape[0]),
            layers=int(array.shape[1]),
            top_k=int(array.shape[2]),
            raw_routing_path=str(raw_path.relative_to(self.output_dir)),
            raw_routing_sha256=raw_sha256,
        )
        receipt = self._receipt(
            result,
            request_id=request_id,
            prepare_request_id=prepare_request_id,
            model=model,
            prompt_tokens=prompt_tokens,
            token_start=token_start,
            runtime_association_status=runtime_association_status,
            prefix_id=prefix_id,
            prefix_hash=prefix_hash,
            cache_key=cache_key,
        )
        with self._lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            self._append_jsonl(self._artifact("moe_route_events"), (event.to_record() for event in events))
            self._append_jsonl(self._artifact("moe_prepare_receipts"), (receipt,))
            self._request_count += 1
            self._success_count += 1
            self._route_event_count += len(events)
            self._unique_experts.update(unique_experts)
            for token_layers in array:
                for layer_id, experts in enumerate(token_layers):
                    counts = self._layer_expert_counts.setdefault(layer_id, {})
                    for expert in experts:
                        expert_id = int(expert)
                        counts[expert_id] = counts.get(expert_id, 0) + 1
            self._raw_entries.append({
                "request_id": request_id,
                "path": str(raw_path.relative_to(self.output_dir)),
                "sha256": raw_sha256,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": len(raw),
            })
            self._write_summaries()
        return result

    def _record_failure(self, **values: Any) -> MoePrepareResult:
        result = MoePrepareResult(
            status="prepare_failed",
            latency_ms=float(values["latency_ms"]),
            model_type=self.config.model_type,
            error=str(values["error"]),
            started_s=float(values["started_s"]),
            ended_s=float(values["ended_s"]),
        )
        receipt = self._receipt(result, **{
            key: values[key]
            for key in (
                "request_id", "prepare_request_id", "model", "prompt_tokens",
                "token_start", "runtime_association_status", "prefix_id",
                "prefix_hash", "cache_key",
            )
        })
        with self._lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._append_jsonl(self._artifact("moe_prepare_receipts"), (receipt,))
            self._request_count += 1
            self._failure_count += 1
            self._write_summaries()
        return result

    def _record_skip(
        self,
        *,
        status: str,
        request_id: str,
        model: str,
        runtime_association_status: str,
        prefix_id: str,
        prefix_hash: str,
        cache_key: str,
    ) -> MoePrepareResult:
        now = time.time()
        result = MoePrepareResult(
            status=status,
            model_type=self.config.model_type,
            started_s=now,
            ended_s=now,
        )
        receipt = self._receipt(
            result,
            request_id=request_id,
            prepare_request_id=f"{request_id}:moe-prepare",
            model=model,
            prompt_tokens=0,
            token_start=0,
            runtime_association_status=runtime_association_status,
            prefix_id=prefix_id,
            prefix_hash=prefix_hash,
            cache_key=cache_key,
        )
        with self._lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._append_jsonl(self._artifact("moe_prepare_receipts"), (receipt,))
            self._request_count += 1
            self._failure_count += 1
            self._write_summaries()
        return result

    def _receipt(
        self,
        result: MoePrepareResult,
        *,
        request_id: str,
        prepare_request_id: str,
        model: str,
        prompt_tokens: int,
        token_start: int,
        runtime_association_status: str,
        prefix_id: str,
        prefix_hash: str,
        cache_key: str,
    ) -> dict[str, Any]:
        return {
            "schema": MOE_PREPARE_RECEIPT_SCHEMA,
            "request_id": request_id,
            "prepare_request_id": prepare_request_id,
            "model": model,
            "model_type": self.config.model_type,
            "mode": self.config.mode,
            "status": result.status,
            "error": result.error,
            "context_published": True,
            "runtime_association_status": runtime_association_status,
            "prefix_id": prefix_id,
            "prefix_hash": prefix_hash,
            "cache_key": cache_key,
            "prompt_tokens": prompt_tokens,
            "route_token_start": token_start,
            "captured_tokens": result.captured_tokens,
            "layers": result.layers,
            "top_k": result.top_k,
            "route_event_count": result.route_event_count,
            "unique_experts": result.unique_experts,
            "started_s": result.started_s,
            "ended_s": result.ended_s,
            "latency_ms": round(result.latency_ms, 6),
            "raw_routing_path": result.raw_routing_path,
            "raw_routing_sha256": result.raw_routing_sha256,
            "claim_boundary": (
                "real MoE prefill and routed-expert evidence; no selective expert-weight paging"
            ),
        }

    def _artifact(self, role: str) -> Path:
        return self.output_dir / MOE_RUNTIME_ARTIFACT_NAMES[role]

    @staticmethod
    def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_summaries(self) -> None:
        summary = {
            "schema": MOE_ROUTE_SUMMARY_SCHEMA,
            "model_type": self.config.model_type,
            "prepare_request_count": self._request_count,
            "successful_prepare_count": self._success_count,
            "failed_or_skipped_prepare_count": self._failure_count,
            "route_event_count": self._route_event_count,
            "unique_expert_count": len(self._unique_experts),
            "unique_experts": sorted(self._unique_experts),
            "layer_expert_counts": [
                {
                    "layer_id": layer_id,
                    "expert_counts": {
                        str(expert_id): count
                        for expert_id, count in sorted(counts.items())
                    },
                }
                for layer_id, counts in sorted(self._layer_expert_counts.items())
            ],
        }
        self._artifact("moe_route_summary").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raw_manifest = {
            "schema": MOE_RAW_MANIFEST_SCHEMA,
            "entries": list(self._raw_entries),
        }
        self._artifact("moe_routed_experts_manifest").write_text(
            json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _first_choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("MoE prepare response is missing choices[0]")
    return choices[0]


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("._")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:80] or 'request'}-{digest}"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _optional_int(value: Any, *, default: int | None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)
