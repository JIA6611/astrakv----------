"""Evaluation helpers for AstraKV-W."""

from .hidden_state import (
    HiddenStateComparison,
    HiddenStateRecord,
    HiddenStateSummary,
    compare_record_sets as compare_hidden_state_record_sets,
    linear_cka,
    summarize_comparisons as summarize_hidden_state_comparisons,
)
from .quality import (
    QualityComparison,
    QualitySummary,
    compare_output_records,
    normalize_text,
    summarize_quality,
    token_divergence_rate,
)

__all__ = [
    "HiddenStateComparison",
    "HiddenStateRecord",
    "HiddenStateSummary",
    "compare_hidden_state_record_sets",
    "linear_cka",
    "QualityComparison",
    "QualitySummary",
    "compare_output_records",
    "normalize_text",
    "summarize_hidden_state_comparisons",
    "summarize_quality",
    "token_divergence_rate",
]
