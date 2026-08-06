"""Hidden-state drift and CKA evaluation helpers.

The helpers operate on archived JSONL artifacts. They do not run a model,
install hooks, import torch, or allocate GPU tensors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class HiddenStateRecord:
    sample_id: str
    layer_id: int | None
    token_index: int | None
    values: tuple[tuple[float, ...], ...]
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        layer = "*" if self.layer_id is None else str(self.layer_id)
        token = "*" if self.token_index is None else str(self.token_index)
        return f"{self.sample_id}|layer={layer}|token={token}"

    @property
    def shape(self) -> tuple[int, int]:
        if not self.values:
            return (0, 0)
        return (len(self.values), len(self.values[0]))


@dataclass(frozen=True, slots=True)
class HiddenStateComparison:
    sample_id: str
    layer_id: int | None
    token_index: int | None
    status: str
    baseline_shape: tuple[int, int]
    variant_shape: tuple[int, int]
    element_count: int
    cka: float | None = None
    cosine_similarity: float | None = None
    mse: float | None = None
    l2_drift: float | None = None
    max_abs_diff: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "layer_id": "" if self.layer_id is None else self.layer_id,
            "token_index": "" if self.token_index is None else self.token_index,
            "status": self.status,
            "baseline_shape": format_shape(self.baseline_shape),
            "variant_shape": format_shape(self.variant_shape),
            "element_count": self.element_count,
            "cka": "" if self.cka is None else self.cka,
            "cosine_similarity": "" if self.cosine_similarity is None else self.cosine_similarity,
            "mse": "" if self.mse is None else self.mse,
            "l2_drift": "" if self.l2_drift is None else self.l2_drift,
            "max_abs_diff": "" if self.max_abs_diff is None else self.max_abs_diff,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HiddenStateSummary:
    comparison_count: int
    ok_count: int
    status_counts: dict[str, int]
    mean_cka: float | None
    mean_cosine_similarity: float | None
    mean_mse: float | None
    mean_l2_drift: float | None
    max_l2_drift: float | None
    max_abs_diff: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "comparison_count": self.comparison_count,
            "ok_count": self.ok_count,
            "status_counts": dict(self.status_counts),
            "mean_cka": "" if self.mean_cka is None else self.mean_cka,
            "mean_cosine_similarity": ""
            if self.mean_cosine_similarity is None
            else self.mean_cosine_similarity,
            "mean_mse": "" if self.mean_mse is None else self.mean_mse,
            "mean_l2_drift": "" if self.mean_l2_drift is None else self.mean_l2_drift,
            "max_l2_drift": "" if self.max_l2_drift is None else self.max_l2_drift,
            "max_abs_diff": "" if self.max_abs_diff is None else self.max_abs_diff,
        }


def compare_record_sets(
    baseline_records: Iterable[HiddenStateRecord],
    variant_records: Iterable[HiddenStateRecord],
) -> list[HiddenStateComparison]:
    baseline_by_key = {record.key: record for record in baseline_records}
    variant_by_key = {record.key: record for record in variant_records}
    ordered_keys = list(baseline_by_key)
    ordered_keys.extend(key for key in variant_by_key if key not in baseline_by_key)
    return [
        compare_records(baseline_by_key.get(key), variant_by_key.get(key), key=key)
        for key in ordered_keys
    ]


def compare_records(
    baseline: HiddenStateRecord | None,
    variant: HiddenStateRecord | None,
    *,
    key: str = "",
) -> HiddenStateComparison:
    if baseline is None:
        sample_id, layer_id, token_index = parse_key(key)
        return HiddenStateComparison(
            sample_id=sample_id,
            layer_id=layer_id,
            token_index=token_index,
            status="missing_baseline",
            baseline_shape=(0, 0),
            variant_shape=variant.shape if variant is not None else (0, 0),
            element_count=0,
            reason="baseline hidden state is missing",
        )
    if variant is None:
        return HiddenStateComparison(
            sample_id=baseline.sample_id,
            layer_id=baseline.layer_id,
            token_index=baseline.token_index,
            status="missing_variant",
            baseline_shape=baseline.shape,
            variant_shape=(0, 0),
            element_count=0,
            reason="variant hidden state is missing",
        )
    if baseline.shape != variant.shape:
        return HiddenStateComparison(
            sample_id=baseline.sample_id,
            layer_id=baseline.layer_id,
            token_index=baseline.token_index,
            status="shape_mismatch",
            baseline_shape=baseline.shape,
            variant_shape=variant.shape,
            element_count=0,
            reason="baseline and variant hidden-state shapes differ",
        )
    element_count = baseline.shape[0] * baseline.shape[1]
    if element_count == 0:
        return HiddenStateComparison(
            sample_id=baseline.sample_id,
            layer_id=baseline.layer_id,
            token_index=baseline.token_index,
            status="empty",
            baseline_shape=baseline.shape,
            variant_shape=variant.shape,
            element_count=0,
            reason="hidden state is empty",
        )

    flat_left = flatten(baseline.values)
    flat_right = flatten(variant.values)
    diffs = [left - right for left, right in zip(flat_left, flat_right)]
    mse_value = sum(diff * diff for diff in diffs) / max(1, len(diffs))
    return HiddenStateComparison(
        sample_id=baseline.sample_id,
        layer_id=baseline.layer_id,
        token_index=baseline.token_index,
        status="ok",
        baseline_shape=baseline.shape,
        variant_shape=variant.shape,
        element_count=element_count,
        cka=linear_cka(baseline.values, variant.values),
        cosine_similarity=cosine_similarity(flat_left, flat_right),
        mse=mse_value,
        l2_drift=math.sqrt(sum(diff * diff for diff in diffs)),
        max_abs_diff=max((abs(diff) for diff in diffs), default=0.0),
        reason="hidden states compared successfully",
        metadata={
            "baseline_source": baseline.source,
            "variant_source": variant.source,
        },
    )


def summarize_comparisons(comparisons: Iterable[HiddenStateComparison]) -> HiddenStateSummary:
    items = list(comparisons)
    ok_items = [item for item in items if item.status == "ok"]
    statuses: dict[str, int] = {}
    for item in items:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    return HiddenStateSummary(
        comparison_count=len(items),
        ok_count=len(ok_items),
        status_counts=dict(sorted(statuses.items())),
        mean_cka=mean_optional([item.cka for item in ok_items]),
        mean_cosine_similarity=mean_optional([item.cosine_similarity for item in ok_items]),
        mean_mse=mean_optional([item.mse for item in ok_items]),
        mean_l2_drift=mean_optional([item.l2_drift for item in ok_items]),
        max_l2_drift=max_optional([item.l2_drift for item in ok_items]),
        max_abs_diff=max_optional([item.max_abs_diff for item in ok_items]),
    )


def load_hidden_state_jsonl(path: str | Path) -> list[HiddenStateRecord]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(str(jsonl_path))
    records: list[HiddenStateRecord] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.lstrip("\ufeff")
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(hidden_state_from_record(record, source=str(jsonl_path), line_number=line_number))
    return records


def hidden_state_from_record(
    record: dict[str, Any],
    *,
    source: str = "",
    line_number: int | None = None,
) -> HiddenStateRecord:
    sample_id = str(
        first_present(record, "sample_id", "prompt_id", "request_id", "case", "id")
        or f"row{line_number or 0}"
    )
    raw_values = first_present(record, "hidden_state", "hidden_states", "activation", "activations", "values")
    values = normalize_matrix(raw_values)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return HiddenStateRecord(
        sample_id=sample_id,
        layer_id=optional_int(first_present(record, "layer_id", "layer")),
        token_index=optional_int(first_present(record, "token_index", "token")),
        values=values,
        source=source,
        metadata={**metadata, "line_number": line_number},
    )


def normalize_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    if value is None:
        return tuple()
    if isinstance(value, list) and value and all(is_number(item) for item in value):
        return (tuple(float(item) for item in value),)
    if isinstance(value, list):
        rows: list[tuple[float, ...]] = []
        width: int | None = None
        for row in value:
            if not isinstance(row, list):
                continue
            numbers = tuple(float(item) for item in row if is_number(item))
            if width is None:
                width = len(numbers)
            if len(numbers) == width:
                rows.append(numbers)
        return tuple(rows)
    return tuple()


def linear_cka(left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]) -> float | None:
    if shape(left) != shape(right) or not left:
        return None
    left_centered = center_columns(left)
    right_centered = center_columns(right)
    xty = matmul_transpose_left(left_centered, right_centered)
    xtx = matmul_transpose_left(left_centered, left_centered)
    yty = matmul_transpose_left(right_centered, right_centered)
    numerator = frobenius_norm_squared(xty)
    denominator = math.sqrt(frobenius_norm_squared(xtx) * frobenius_norm_squared(yty))
    if denominator <= 0.0:
        return None
    return numerator / denominator


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(lval * rval for lval, rval in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return None
    return dot / denominator


def center_columns(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    if not matrix:
        return []
    rows, cols = shape(matrix)
    means = [sum(matrix[row][col] for row in range(rows)) / rows for col in range(cols)]
    return [[matrix[row][col] - means[col] for col in range(cols)] for row in range(rows)]


def matmul_transpose_left(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if not left or not right:
        return []
    rows = len(left)
    left_cols = len(left[0])
    right_cols = len(right[0])
    return [
        [
            sum(left[row][left_col] * right[row][right_col] for row in range(rows))
            for right_col in range(right_cols)
        ]
        for left_col in range(left_cols)
    ]


def frobenius_norm_squared(matrix: list[list[float]]) -> float:
    return sum(value * value for row in matrix for value in row)


def flatten(matrix: tuple[tuple[float, ...], ...]) -> list[float]:
    return [value for row in matrix for value in row]


def shape(matrix: tuple[tuple[float, ...], ...]) -> tuple[int, int]:
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def parse_key(key: str) -> tuple[str, int | None, int | None]:
    sample_id = key
    layer_id: int | None = None
    token_index: int | None = None
    parts = key.split("|")
    if parts:
        sample_id = parts[0]
    for part in parts[1:]:
        if part.startswith("layer="):
            layer_id = optional_int(part.removeprefix("layer="))
        elif part.startswith("token="):
            token_index = optional_int(part.removeprefix("token="))
    return sample_id, layer_id, token_index


def format_shape(value: tuple[int, int]) -> str:
    return f"{value[0]}x{value[1]}"


def first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def optional_int(value: Any) -> int | None:
    if value in (None, "", "*", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mean_optional(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return sum(real) / len(real)


def max_optional(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return max(real)
