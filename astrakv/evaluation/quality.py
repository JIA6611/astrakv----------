"""Quality and output-consistency evaluation helpers.

The helpers are dependency-light so they can run locally from archived JSONL
outputs. PPL is reported when records already contain loss/logprob evidence;
this module does not run a language model by itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class QualityComparison:
    sample_id: str
    baseline_text: str
    variant_text: str
    exact_match: bool
    normalized_match: bool
    baseline_tokens: int
    variant_tokens: int
    token_edit_distance: int
    token_divergence_rate: float
    char_edit_distance: int
    char_divergence_rate: float
    baseline_ppl: float | None = None
    variant_ppl: float | None = None
    ppl_delta: float | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "exact_match": int(self.exact_match),
            "normalized_match": int(self.normalized_match),
            "baseline_tokens": self.baseline_tokens,
            "variant_tokens": self.variant_tokens,
            "token_edit_distance": self.token_edit_distance,
            "token_divergence_rate": self.token_divergence_rate,
            "char_edit_distance": self.char_edit_distance,
            "char_divergence_rate": self.char_divergence_rate,
            "baseline_ppl": "" if self.baseline_ppl is None else self.baseline_ppl,
            "variant_ppl": "" if self.variant_ppl is None else self.variant_ppl,
            "ppl_delta": "" if self.ppl_delta is None else self.ppl_delta,
            "baseline_text": self.baseline_text,
            "variant_text": self.variant_text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class QualitySummary:
    sample_count: int
    ok_count: int
    exact_match_rate: float
    normalized_match_rate: float
    mean_token_divergence_rate: float
    mean_char_divergence_rate: float
    mean_baseline_ppl: float | None
    mean_variant_ppl: float | None
    mean_ppl_delta: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "ok_count": self.ok_count,
            "exact_match_rate": self.exact_match_rate,
            "normalized_match_rate": self.normalized_match_rate,
            "mean_token_divergence_rate": self.mean_token_divergence_rate,
            "mean_char_divergence_rate": self.mean_char_divergence_rate,
            "mean_baseline_ppl": "" if self.mean_baseline_ppl is None else self.mean_baseline_ppl,
            "mean_variant_ppl": "" if self.mean_variant_ppl is None else self.mean_variant_ppl,
            "mean_ppl_delta": "" if self.mean_ppl_delta is None else self.mean_ppl_delta,
        }


def compare_output_records(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    *,
    sample_id: str | None = None,
) -> QualityComparison:
    resolved_id = sample_id or infer_sample_id(baseline) or infer_sample_id(variant)
    baseline_text = extract_text(baseline)
    variant_text = extract_text(variant)
    baseline_norm = normalize_text(baseline_text)
    variant_norm = normalize_text(variant_text)
    baseline_tokens = tokenize(baseline_text)
    variant_tokens = tokenize(variant_text)
    token_distance = levenshtein_distance(baseline_tokens, variant_tokens)
    char_distance = levenshtein_distance(list(baseline_norm), list(variant_norm))
    baseline_ppl = extract_ppl(baseline)
    variant_ppl = extract_ppl(variant)
    return QualityComparison(
        sample_id=resolved_id,
        baseline_text=baseline_text,
        variant_text=variant_text,
        exact_match=baseline_text == variant_text,
        normalized_match=baseline_norm == variant_norm,
        baseline_tokens=len(baseline_tokens),
        variant_tokens=len(variant_tokens),
        token_edit_distance=token_distance,
        token_divergence_rate=distance_rate(token_distance, len(baseline_tokens), len(variant_tokens)),
        char_edit_distance=char_distance,
        char_divergence_rate=distance_rate(char_distance, len(baseline_norm), len(variant_norm)),
        baseline_ppl=baseline_ppl,
        variant_ppl=variant_ppl,
        ppl_delta=None if baseline_ppl is None or variant_ppl is None else variant_ppl - baseline_ppl,
        status="ok" if baseline_text or variant_text else "empty_outputs",
        metadata={
            "baseline_status": baseline.get("status", ""),
            "variant_status": variant.get("status", ""),
            "baseline_error": baseline.get("error", ""),
            "variant_error": variant.get("error", ""),
        },
    )


def summarize_quality(comparisons: Iterable[QualityComparison]) -> QualitySummary:
    items = list(comparisons)
    ok_items = [item for item in items if item.status == "ok"]
    return QualitySummary(
        sample_count=len(items),
        ok_count=len(ok_items),
        exact_match_rate=mean([1.0 if item.exact_match else 0.0 for item in ok_items]),
        normalized_match_rate=mean([1.0 if item.normalized_match else 0.0 for item in ok_items]),
        mean_token_divergence_rate=mean([item.token_divergence_rate for item in ok_items]),
        mean_char_divergence_rate=mean([item.char_divergence_rate for item in ok_items]),
        mean_baseline_ppl=mean_optional([item.baseline_ppl for item in ok_items]),
        mean_variant_ppl=mean_optional([item.variant_ppl for item in ok_items]),
        mean_ppl_delta=mean_optional([item.ppl_delta for item in ok_items]),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def token_divergence_rate(baseline: str, variant: str) -> float:
    baseline_tokens = tokenize(baseline)
    variant_tokens = tokenize(variant)
    distance = levenshtein_distance(baseline_tokens, variant_tokens)
    return distance_rate(distance, len(baseline_tokens), len(variant_tokens))


def levenshtein_distance(left: list[Any], right: list[Any]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            replace_cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + replace_cost,
                )
            )
        previous = current
    return previous[-1]


def distance_rate(distance: int, baseline_length: int, variant_length: int) -> float:
    return distance / max(1, baseline_length, variant_length)


def extract_text(record: dict[str, Any]) -> str:
    for key in (
        "output",
        "output_text",
        "text",
        "response",
        "content",
        "completion",
        "generated_text",
    ):
        value = record.get(key)
        if isinstance(value, str):
            return value
    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
    return ""


def infer_sample_id(record: dict[str, Any]) -> str:
    for key in ("sample_id", "prompt_id", "case", "request_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("sample_id", "prompt_id", "case", "request_id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def extract_ppl(record: dict[str, Any]) -> float | None:
    direct = as_float(record.get("ppl", record.get("perplexity")))
    if direct is not None:
        return direct
    if "loss" in record and "negative_log_likelihood" not in record and "nll" not in record:
        loss = as_float(record.get("loss"))
        return None if loss is None else math.exp(loss)
    nll = as_float(
        record.get(
            "negative_log_likelihood",
            record.get("nll", record.get("loss")),
        )
    )
    token_count = as_float(record.get("token_count", record.get("tokens", record.get("output_tokens"))))
    if nll is None or token_count in (None, 0):
        return None
    return math.exp(nll / token_count)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_optional(values: list[float | None]) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return sum(real) / len(real)


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
