"""Canonical workload contract for offline/runtime eviction validation.

This module deliberately validates externally supplied JSONL rather than
generating a replacement dataset.  The same rows are consumed by endpoint
benchmarking, trace/profile construction, policy scheduling, and the mmap
proof-of-concept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


RUNTIME_WORKLOAD_SCHEMA_VERSION = "astra-runtime-workload-v1"
REUSE_BUCKETS = frozenset({"none", "low", "medium", "high"})
REQUIRED_FIELDS = (
    "request_id",
    "prompt",
    "prefix_id",
    "arrival_index",
    "reuse_ratio",
    "reuse_bucket",
)


@dataclass(frozen=True, slots=True)
class RuntimeWorkloadRow:
    request_id: str
    prompt: str
    prefix_id: str
    arrival_index: int
    reuse_ratio: float
    reuse_bucket: str
    prefix_hash: str = ""
    cache_key: str = ""
    context_length: int | None = None
    expected_output_tokens: int | None = None
    batch_size: int | None = None
    sleep_before_s: float | None = None
    prefetch_lead_s: float | None = None
    case: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_WORKLOAD_SCHEMA_VERSION,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prefix_id": self.prefix_id,
            "prefix_hash": self.prefix_hash,
            "cache_key": self.cache_key,
            "arrival_index": self.arrival_index,
            "reuse_ratio": self.reuse_ratio,
            "reuse_bucket": self.reuse_bucket,
            "context_length": self.context_length,
            "expected_output_tokens": self.expected_output_tokens,
            "batch_size": self.batch_size,
            "sleep_before_s": self.sleep_before_s,
            "prefetch_lead_s": self.prefetch_lead_s,
            "case": self.case,
            "metadata": dict(self.metadata),
        }

    def request_metadata(self, run_id: str) -> dict[str, Any]:
        return {**self.to_record(), "run_id": run_id}


class WorkloadContractError(ValueError):
    """Raised when an external workload cannot safely drive the closed loop."""


def load_runtime_workload_jsonl(path: str | Path) -> list[RuntimeWorkloadRow]:
    input_path = Path(path)
    if not input_path.exists():
        raise WorkloadContractError(f"workload JSONL not found: {input_path}")
    rows: list[RuntimeWorkloadRow] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkloadContractError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise WorkloadContractError(f"line {line_number}: row must be a JSON object")
        rows.append(runtime_workload_row_from_record(raw, line_number=line_number))
    validate_runtime_workload(rows)
    return sorted(rows, key=lambda item: item.arrival_index)


def runtime_workload_row_from_record(record: dict[str, Any], *, line_number: int | None = None) -> RuntimeWorkloadRow:
    prefix = f"line {line_number}: " if line_number is not None else ""
    missing = [name for name in REQUIRED_FIELDS if record.get(name) in (None, "")]
    if missing:
        raise WorkloadContractError(f"{prefix}missing required field(s): {', '.join(missing)}")
    request_id = str(record["request_id"])
    prompt = record["prompt"]
    prefix_id = str(record["prefix_id"])
    if not isinstance(prompt, str):
        raise WorkloadContractError(f"{prefix}prompt must be a string")
    arrival_index = _int_value(record["arrival_index"], "arrival_index", prefix)
    reuse_ratio = _float_value(record["reuse_ratio"], "reuse_ratio", prefix)
    if not 0.0 <= reuse_ratio <= 1.0:
        raise WorkloadContractError(f"{prefix}reuse_ratio must be in [0, 1]")
    reuse_bucket = str(record["reuse_bucket"])
    if reuse_bucket not in REUSE_BUCKETS:
        raise WorkloadContractError(f"{prefix}reuse_bucket must be one of: {', '.join(sorted(REUSE_BUCKETS))}")
    context_length = _optional_int(record.get("context_length"), "context_length", prefix)
    expected_output_tokens = _optional_int(record.get("expected_output_tokens"), "expected_output_tokens", prefix)
    batch_size = _optional_int(record.get("batch_size"), "batch_size", prefix)
    sleep_before_s = _optional_float(record.get("sleep_before_s"), "sleep_before_s", prefix)
    prefetch_lead_s = _optional_float(record.get("prefetch_lead_s"), "prefetch_lead_s", prefix)
    for name, value, minimum in (
        ("arrival_index", arrival_index, 0),
        ("context_length", context_length, 0),
        ("expected_output_tokens", expected_output_tokens, 1),
        ("batch_size", batch_size, 1),
    ):
        if value is not None and value < minimum:
            raise WorkloadContractError(f"{prefix}{name} must be >= {minimum}")
    if sleep_before_s is not None and sleep_before_s < 0.0:
        raise WorkloadContractError(f"{prefix}sleep_before_s must be >= 0")
    if prefetch_lead_s is not None and prefetch_lead_s < 0.0:
        raise WorkloadContractError(f"{prefix}prefetch_lead_s must be >= 0")
    return RuntimeWorkloadRow(
        request_id=request_id,
        prompt=prompt,
        prefix_id=prefix_id,
        arrival_index=arrival_index,
        reuse_ratio=reuse_ratio,
        reuse_bucket=reuse_bucket,
        prefix_hash=str(record.get("prefix_hash") or ""),
        cache_key=str(record.get("cache_key") or ""),
        context_length=context_length,
        expected_output_tokens=expected_output_tokens,
        batch_size=batch_size,
        sleep_before_s=sleep_before_s,
        prefetch_lead_s=prefetch_lead_s,
        case=str(record.get("case") or ""),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    )


def validate_runtime_workload(rows: Iterable[RuntimeWorkloadRow]) -> None:
    request_ids: set[str] = set()
    arrival_indexes: set[int] = set()
    count = 0
    for row in rows:
        count += 1
        if row.request_id in request_ids:
            raise WorkloadContractError(f"duplicate request_id: {row.request_id}")
        if row.arrival_index in arrival_indexes:
            raise WorkloadContractError(f"duplicate arrival_index: {row.arrival_index}")
        request_ids.add(row.request_id)
        arrival_indexes.add(row.arrival_index)
    if count == 0:
        raise WorkloadContractError("workload JSONL contains no rows")


def workload_request_mapping(rows: Iterable[RuntimeWorkloadRow]) -> dict[str, dict[str, Any]]:
    return {row.request_id: row.to_record() for row in rows}


def _int_value(value: Any, name: str, prefix: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadContractError(f"{prefix}{name} must be an integer") from exc


def _float_value(value: Any, name: str, prefix: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadContractError(f"{prefix}{name} must be a number") from exc


def _optional_int(value: Any, name: str, prefix: str) -> int | None:
    if value in (None, ""):
        return None
    return _int_value(value, name, prefix)


def _optional_float(value: Any, name: str, prefix: str) -> float | None:
    if value in (None, ""):
        return None
    return _float_value(value, name, prefix)
