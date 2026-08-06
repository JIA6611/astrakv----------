"""Build a unified request-level reuse analysis table.

This report intentionally keeps workload-only evidence and trace-backed
observations in one table without forcing a false row-level join between
unrelated sources. The resulting artifacts help classify workloads into:

- exact-next
- fixed-revisit
- structural-partial
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DEFAULT_GROUPED_ROOT = Path(r"E:\下载\results_full\workload_prompts")
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_OUTPUT_DIR = Path("results/reuse_pattern_analysis")
DEFAULT_KV_BYTES_PER_TOKEN = 1


@dataclass(frozen=True, slots=True)
class SourceSummary:
    source_id: str
    source_kind: str
    source_name: str
    request_count: int
    repeated_request_count: int
    repeated_group_count: int
    exact_prefix_request_count: int
    structural_reuse_request_count: int
    exact_next_request_count: int
    fixed_revisit_request_count: int
    structural_partial_request_count: int
    dominant_class: str
    consecutive_reuse_ratio: float
    nearest_distance_p50: float | None
    nearest_distance_p95: float | None
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "request_count": self.request_count,
            "repeated_request_count": self.repeated_request_count,
            "repeated_group_count": self.repeated_group_count,
            "exact_prefix_request_count": self.exact_prefix_request_count,
            "structural_reuse_request_count": self.structural_reuse_request_count,
            "exact_next_request_count": self.exact_next_request_count,
            "fixed_revisit_request_count": self.fixed_revisit_request_count,
            "structural_partial_request_count": self.structural_partial_request_count,
            "dominant_class": self.dominant_class,
            "consecutive_reuse_ratio": self.consecutive_reuse_ratio,
            "nearest_distance_p50": self.nearest_distance_p50,
            "nearest_distance_p95": self.nearest_distance_p95,
            "notes": self.notes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-root", default=str(DEFAULT_GROUPED_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--kv-bytes-per-token", type=int, default=DEFAULT_KV_BYTES_PER_TOKEN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grouped_root = Path(args.grouped_root)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    kv_bytes_per_token = int(args.kv_bytes_per_token)
    if kv_bytes_per_token <= 0:
        raise ValueError("kv-bytes-per-token must be positive")

    request_rows: list[dict[str, Any]] = []
    summaries: list[SourceSummary] = []

    grouped_rows, grouped_summaries = analyze_grouped_workloads(
        grouped_root=grouped_root, kv_bytes_per_token=kv_bytes_per_token
    )
    request_rows.extend(grouped_rows)
    summaries.extend(grouped_summaries)

    trace_rows, trace_summaries = analyze_request_results(
        results_root=results_root, kv_bytes_per_token=kv_bytes_per_token
    )
    request_rows.extend(trace_rows)
    summaries.extend(trace_summaries)

    request_rows.sort(
        key=lambda item: (
            str(item.get("source_kind") or ""),
            str(item.get("source_name") or ""),
            _sortable_number(item.get("sequence_index")),
            str(item.get("row_id") or ""),
        )
    )
    summaries.sort(key=lambda item: (item.source_kind, item.source_name, item.source_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    request_jsonl = output_dir / "unified_reuse_analysis.jsonl"
    request_csv = output_dir / "unified_reuse_analysis.csv"
    summary_csv = output_dir / "unified_reuse_source_summary.csv"
    summary_md = output_dir / "unified_reuse_source_summary.md"

    write_jsonl(request_jsonl, request_rows)
    write_csv(request_csv, request_rows)
    write_csv(summary_csv, [item.to_record() for item in summaries])
    write_summary_markdown(summary_md, summaries, request_rows, kv_bytes_per_token)

    print(f"Unified reuse JSONL written to {request_jsonl}")
    print(f"Unified reuse CSV written to {request_csv}")
    print(f"Source summary CSV written to {summary_csv}")
    print(f"Source summary Markdown written to {summary_md}")
    return 0


def analyze_grouped_workloads(
    *, grouped_root: Path, kv_bytes_per_token: int
) -> tuple[list[dict[str, Any]], list[SourceSummary]]:
    rows: list[dict[str, Any]] = []
    summaries: list[SourceSummary] = []
    if not grouped_root.is_dir():
        return rows, summaries

    for grouped_path in sorted(grouped_root.glob("*/grouped_prompts.jsonl")):
        dataset = grouped_path.parent.name
        raw_rows = load_jsonl_records(grouped_path)
        ordered = sorted(
            [item for item in raw_rows if isinstance(item, dict)],
            key=lambda item: _sortable_number(item.get("order")),
        )
        if not ordered:
            continue

        source_rows, summary = build_grouped_source_rows(
            grouped_path=grouped_path,
            dataset=dataset,
            rows=ordered,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        rows.extend(source_rows)
        summaries.append(summary)
    return rows, summaries


def build_grouped_source_rows(
    *,
    grouped_path: Path,
    dataset: str,
    rows: list[dict[str, Any]],
    kv_bytes_per_token: int,
) -> tuple[list[dict[str, Any]], SourceSummary]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("reuse_group") or "")].append(row)

    group_positions: dict[str, list[int]] = {}
    group_exactness: dict[str, bool] = {}
    for group_id, group_rows in by_group.items():
        ordered = sorted(group_rows, key=lambda item: _sortable_number(item.get("order")))
        positions = [_int_or_none(item.get("order")) for item in ordered]
        group_positions[group_id] = [value for value in positions if value is not None]
        context_hashes = {str(item.get("context_hash") or "") for item in ordered if str(item.get("context_hash") or "")}
        shared_flags = {bool(item.get("shared_context", False)) for item in ordered}
        group_exactness[group_id] = len(ordered) > 1 and bool(context_hashes) and len(context_hashes) == 1 and shared_flags == {True}

    consecutive_hits = 0
    repeated_rows = 0
    nearest_distances: list[int] = []
    source_rows: list[dict[str, Any]] = []

    source_pattern_class = classify_grouped_source(by_group, group_exactness)

    for sequence_index, row in enumerate(sorted(rows, key=lambda item: _sortable_number(item.get("order")))):
        group_id = str(row.get("reuse_group") or "")
        group_rows = by_group.get(group_id, [])
        group_size = len(group_rows)
        order = _int_or_none(row.get("order"))
        prev_distance, next_distance, nearest_distance = neighbor_distances(group_positions.get(group_id, []), order)
        exact_prefix = group_exactness.get(group_id, False)
        structural_reuse = bool(row.get("shared_context", False)) and not exact_prefix and group_size > 1
        reusable_tokens = estimate_grouped_reusable_tokens(row, group_size)
        estimated_kv_bytes = reusable_tokens * kv_bytes_per_token

        if nearest_distance is not None:
            repeated_rows += 1
            nearest_distances.append(nearest_distance)
            if nearest_distance == 1:
                consecutive_hits += 1

        reuse_class = classify_request_pattern(
            source_pattern_class=source_pattern_class,
            exact_prefix=exact_prefix,
            structural_reuse=structural_reuse,
            group_size=group_size,
            nearest_distance=nearest_distance,
        )
        source_rows.append(
            {
                "row_id": f"grouped:{dataset}:{sequence_index:06d}",
                "source_kind": "grouped_prompt",
                "source_name": dataset,
                "source_id": str(grouped_path),
                "analysis_scope": "workload_only",
                "join_status": "workload_only",
                "sequence_index": sequence_index,
                "request_id": str(row.get("request_id") or ""),
                "run_id": "",
                "dataset": str(row.get("dataset") or ""),
                "task": str(row.get("task") or dataset),
                "workload_type": str(row.get("workload_type") or "grouped"),
                "scenario": "longbench_grouped",
                "request_path": str(grouped_path),
                "runtime_events_path": "",
                "prefix_id": group_id,
                "cache_key": group_id,
                "workflow_id": str(row.get("workflow_id") or ""),
                "reuse_group": group_id,
                "runtime_object_key": group_id,
                "runtime_object_level": "prefix",
                "group_size": group_size,
                "arrival_index": order,
                "sequence_order": order,
                "prev_same_group_distance": prev_distance,
                "next_same_group_distance": next_distance,
                "adjacent_distance": nearest_distance,
                "estimated_reusable_tokens": reusable_tokens,
                "estimated_kv_bytes": estimated_kv_bytes,
                "kv_bytes_per_token_assumption": kv_bytes_per_token,
                "exact_prefix": exact_prefix,
                "structural_reuse": structural_reuse,
                "reuse_class": reuse_class,
                "source_pattern_class": source_pattern_class,
                "trace_event_count": 0,
                "cache_load_available_count": 0,
                "cache_store_completed_count": 0,
                "max_observed_cached_tokens": None,
                "ttft_ms": None,
                "latency_ms": None,
                "throughput_tokens_s": None,
                "trace_evidence": False,
                "notes": "grouped prompt workload only",
            }
        )

    consecutive_ratio = (consecutive_hits / repeated_rows) if repeated_rows else 0.0
    nearest_p50, nearest_p95 = percentile_pair(nearest_distances)
    summary = SourceSummary(
        source_id=str(grouped_path),
        source_kind="grouped_prompt",
        source_name=dataset,
        request_count=len(source_rows),
        repeated_request_count=repeated_rows,
        repeated_group_count=sum(1 for items in by_group.values() if len(items) > 1),
        exact_prefix_request_count=sum(1 for row in source_rows if row["exact_prefix"]),
        structural_reuse_request_count=sum(1 for row in source_rows if row["structural_reuse"]),
        exact_next_request_count=sum(1 for row in source_rows if row["reuse_class"] == "exact-next"),
        fixed_revisit_request_count=sum(1 for row in source_rows if row["reuse_class"] == "fixed-revisit"),
        structural_partial_request_count=sum(1 for row in source_rows if row["reuse_class"] == "structural-partial"),
        dominant_class=source_pattern_class,
        consecutive_reuse_ratio=consecutive_ratio,
        nearest_distance_p50=nearest_p50,
        nearest_distance_p95=nearest_p95,
        notes=f"grouped workload file: {grouped_path}",
    )
    return source_rows, summary


def classify_grouped_source(
    by_group: dict[str, list[dict[str, Any]]], group_exactness: dict[str, bool]
) -> str:
    repeated_distances: list[int] = []
    consecutive_hits = 0
    repeated_rows = 0
    exact_group_count = 0
    repeated_group_count = 0

    for group_id, group_rows in by_group.items():
        if len(group_rows) <= 1:
            continue
        repeated_group_count += 1
        if group_exactness.get(group_id, False):
            exact_group_count += 1
        positions = sorted(_int_or_none(item.get("order")) for item in group_rows)
        normalized = [value for value in positions if value is not None]
        for index, arrival_index in enumerate(normalized):
            prior = normalized[index - 1] if index > 0 else None
            after = normalized[index + 1] if index + 1 < len(normalized) else None
            candidates = [
                value for value in (
                    None if prior is None else arrival_index - prior,
                    None if after is None else after - arrival_index,
                )
                if value is not None
            ]
            if not candidates:
                continue
            repeated_rows += 1
            nearest = min(candidates)
            repeated_distances.append(nearest)
            if nearest == 1:
                consecutive_hits += 1

    if not repeated_group_count:
        return "structural-partial"
    consecutive_ratio = consecutive_hits / repeated_rows if repeated_rows else 0.0
    nearest_p50 = median(repeated_distances) if repeated_distances else None
    exact_group_ratio = exact_group_count / repeated_group_count
    if exact_group_ratio >= 0.7 and consecutive_ratio >= 0.2 and nearest_p50 == 1:
        return "exact-next"
    return "structural-partial"


def estimate_grouped_reusable_tokens(row: dict[str, Any], group_size: int) -> int:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    context_tokens = _int_or_none(metadata.get("context_token_estimate"))
    if context_tokens is None:
        context_tokens = _int_or_none(metadata.get("estimated_context_tokens"))
    if context_tokens is None:
        context_tokens = _int_or_none(metadata.get("estimated_prompt_tokens"))
    if group_size <= 1 or context_tokens is None:
        return 0
    return max(0, context_tokens)


def analyze_request_results(
    *, results_root: Path, kv_bytes_per_token: int
) -> tuple[list[dict[str, Any]], list[SourceSummary]]:
    request_paths = sorted(results_root.glob("**/request_results.jsonl"))
    run_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_paths_by_run: dict[str, set[str]] = defaultdict(set)

    for request_path in request_paths:
        raw_rows = load_jsonl_records(request_path)
        if not raw_rows:
            continue
        events_path = discover_runtime_events_path(request_path)
        event_rows = load_jsonl_records(events_path) if events_path is not None else []

        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            normalized = dict(row)
            normalized["_request_path"] = str(request_path)
            normalized["_row_index"] = row_index
            normalized["_runtime_events_path"] = str(events_path) if events_path is not None else ""
            run_rows[run_id].append(normalized)
        for event in event_rows:
            if not isinstance(event, dict):
                continue
            run_id = str(event.get("run_id") or "")
            if not run_id:
                continue
            run_events[run_id].append(event)
            if events_path is not None:
                event_paths_by_run[run_id].add(str(events_path))

    request_rows: list[dict[str, Any]] = []
    summaries: list[SourceSummary] = []
    for run_id in sorted(run_rows):
        source_rows, summary = build_trace_source_rows(
            run_id=run_id,
            request_rows=run_rows[run_id],
            event_rows=run_events.get(run_id, []),
            event_paths=sorted(event_paths_by_run.get(run_id, set())),
            kv_bytes_per_token=kv_bytes_per_token,
        )
        request_rows.extend(source_rows)
        summaries.append(summary)
    return request_rows, summaries


def build_trace_source_rows(
    *,
    run_id: str,
    request_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    event_paths: list[str],
    kv_bytes_per_token: int,
) -> tuple[list[dict[str, Any]], SourceSummary]:
    ordered = sorted(
        request_rows,
        key=lambda item: (
            _sortable_number(item.get("request_started_s")),
            _sortable_number(item.get("arrival_index")),
            _sortable_number(item.get("_row_index")),
            str(item.get("_request_path") or ""),
        ),
    )
    scenario = dominant_string([str(item.get("workload_type") or item.get("dataset") or "") for item in ordered])
    source_pattern_class = classify_trace_source_pattern(scenario)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        group_id = trace_group_id(item)
        by_group[group_id].append(item)

    group_positions: dict[str, list[int]] = defaultdict(list)
    group_exactness: dict[str, bool] = {}
    for sequence_index, item in enumerate(ordered):
        item["_sequence_index"] = sequence_index
        group_positions[trace_group_id(item)].append(sequence_index)
    for group_id, group_items in by_group.items():
        prompt_hashes = {
            str(item.get("prompt_hash") or "")
            for item in group_items
            if str(item.get("prompt_hash") or "")
        }
        prefix_hashes = {
            str(item.get("prefix_hash") or "")
            for item in group_items
            if str(item.get("prefix_hash") or "")
        }
        reuse_ratios = {_float_or_none(item.get("reuse_ratio")) for item in group_items}
        exact_by_prompt = len(group_items) > 1 and bool(prompt_hashes) and len(prompt_hashes) == 1
        exact_by_prefix = len(group_items) > 1 and bool(prefix_hashes) and len(prefix_hashes) == 1
        exact_by_ratio = 1.0 in reuse_ratios
        group_exactness[group_id] = exact_by_prompt or exact_by_prefix or exact_by_ratio

    event_index = build_event_index(event_rows)
    repeated_rows = 0
    consecutive_hits = 0
    nearest_distances: list[int] = []
    source_rows: list[dict[str, Any]] = []

    for item in ordered:
        sequence_index = int(item["_sequence_index"])
        group_id = trace_group_id(item)
        group_size = len(by_group[group_id])
        prev_distance, next_distance, nearest_distance = neighbor_distances(
            group_positions[group_id], sequence_index
        )
        exact_prefix = trace_exact_prefix(item, group_exactness.get(group_id, False), group_size)
        structural_reuse = trace_structural_reuse(item)
        reusable_tokens = estimate_trace_reusable_tokens(item, exact_prefix, structural_reuse, group_size)
        estimated_kv_bytes = reusable_tokens * kv_bytes_per_token

        if nearest_distance is not None:
            repeated_rows += 1
            nearest_distances.append(nearest_distance)
            if nearest_distance == 1:
                consecutive_hits += 1

        row_events = events_for_request(event_index, item)
        cache_load_available_count = 0
        cache_store_completed_count = 0
        max_observed_cached_tokens: int | None = None
        for event in row_events:
            action = str(event.get("action") or "")
            status = str(event.get("status") or "")
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            cached_tokens = _int_or_none(metadata.get("lmcache_cached_tokens"))
            if cached_tokens is not None:
                max_observed_cached_tokens = (
                    cached_tokens
                    if max_observed_cached_tokens is None
                    else max(max_observed_cached_tokens, cached_tokens)
                )
            if action == "cache_load" and status in {"available", "completed"}:
                cache_load_available_count += 1
            if action == "cache_store" and status == "completed":
                cache_store_completed_count += 1

        reuse_class = classify_request_pattern(
            source_pattern_class=source_pattern_class,
            exact_prefix=exact_prefix,
            structural_reuse=structural_reuse,
            group_size=group_size,
            nearest_distance=nearest_distance,
        )
        source_rows.append(
            {
                "row_id": f"trace:{run_id}:{sequence_index:06d}",
                "source_kind": "request_result",
                "source_name": scenario or run_id,
                "source_id": run_id,
                "analysis_scope": "trace_backed",
                "join_status": "exact_workload_trace_match",
                "sequence_index": sequence_index,
                "request_id": str(item.get("request_id") or ""),
                "run_id": run_id,
                "dataset": str(item.get("dataset") or ""),
                "task": str(item.get("task") or ""),
                "workload_type": str(item.get("workload_type") or ""),
                "scenario": scenario,
                "request_path": str(item.get("_request_path") or ""),
                "runtime_events_path": str(item.get("_runtime_events_path") or ""),
                "prefix_id": str(item.get("prefix_id") or ""),
                "cache_key": str(item.get("cache_key") or ""),
                "workflow_id": str(item.get("workflow_id") or ""),
                "reuse_group": group_id,
                "runtime_object_key": runtime_object_key_for_row(item),
                "runtime_object_level": runtime_object_level_for_row(item),
                "group_size": group_size,
                "arrival_index": _int_or_none(item.get("arrival_index")),
                "sequence_order": sequence_index,
                "prev_same_group_distance": prev_distance,
                "next_same_group_distance": next_distance,
                "adjacent_distance": nearest_distance,
                "estimated_reusable_tokens": reusable_tokens,
                "estimated_kv_bytes": estimated_kv_bytes,
                "kv_bytes_per_token_assumption": kv_bytes_per_token,
                "exact_prefix": exact_prefix,
                "structural_reuse": structural_reuse,
                "reuse_class": reuse_class,
                "source_pattern_class": source_pattern_class,
                "trace_event_count": len(row_events),
                "cache_load_available_count": cache_load_available_count,
                "cache_store_completed_count": cache_store_completed_count,
                "max_observed_cached_tokens": max_observed_cached_tokens,
                "ttft_ms": _float_or_none(item.get("ttft_ms")),
                "latency_ms": _float_or_none(item.get("latency_ms")),
                "throughput_tokens_s": _float_or_none(item.get("throughput_tokens_s")),
                "trace_evidence": bool(row_events),
                "notes": "request_results joined with runtime_events_raw when available",
            }
        )

    consecutive_ratio = (consecutive_hits / repeated_rows) if repeated_rows else 0.0
    nearest_p50, nearest_p95 = percentile_pair(nearest_distances)
    summary = SourceSummary(
        source_id=run_id,
        source_kind="request_result",
        source_name=scenario or run_id,
        request_count=len(source_rows),
        repeated_request_count=repeated_rows,
        repeated_group_count=sum(1 for items in by_group.values() if len(items) > 1),
        exact_prefix_request_count=sum(1 for row in source_rows if row["exact_prefix"]),
        structural_reuse_request_count=sum(1 for row in source_rows if row["structural_reuse"]),
        exact_next_request_count=sum(1 for row in source_rows if row["reuse_class"] == "exact-next"),
        fixed_revisit_request_count=sum(1 for row in source_rows if row["reuse_class"] == "fixed-revisit"),
        structural_partial_request_count=sum(1 for row in source_rows if row["reuse_class"] == "structural-partial"),
        dominant_class=source_pattern_class,
        consecutive_reuse_ratio=consecutive_ratio,
        nearest_distance_p50=nearest_p50,
        nearest_distance_p95=nearest_p95,
        notes=f"runtime event files: {', '.join(event_paths) if event_paths else 'none'}",
    )
    return source_rows, summary


def classify_trace_source_pattern(scenario: str) -> str:
    lowered = scenario.lower()
    if lowered == "policy_ab_ttft":
        return "fixed-revisit"
    if lowered == "text_kv_consistency":
        return "structural-partial"
    return "structural-partial"


def trace_group_id(row: dict[str, Any]) -> str:
    candidates = (
        row.get("reuse_group"),
        row.get("prefix_id"),
        row.get("cache_key"),
        row.get("prefix_hash"),
        row.get("request_id"),
    )
    for candidate in candidates:
        value = str(candidate or "")
        if value:
            return value
    return ""


def runtime_object_key_for_row(row: dict[str, Any]) -> str:
    candidates = (
        row.get("runtime_object_key"),
        row.get("cache_key"),
        row.get("prefix_id"),
        row.get("workflow_id"),
        row.get("reuse_group"),
        row.get("request_id"),
    )
    for candidate in candidates:
        value = str(candidate or "")
        if value:
            return value
    return ""


def runtime_object_level_for_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("runtime_object_level") or row.get("object_level") or "")
    if explicit in {"prefix", "cache_key", "block"}:
        return explicit
    if str(row.get("cache_key") or "") and runtime_object_key_for_row(row) == str(row.get("cache_key") or ""):
        return "cache_key"
    return "prefix"


def trace_exact_prefix(row: dict[str, Any], group_exact: bool, group_size: int) -> bool:
    reuse_ratio = _float_or_none(row.get("reuse_ratio"))
    if reuse_ratio is not None and reuse_ratio >= 0.999:
        return True
    if group_exact and group_size > 1:
        return True
    return False


def trace_structural_reuse(row: dict[str, Any]) -> bool:
    reuse_ratio = _float_or_none(row.get("reuse_ratio"))
    if reuse_ratio is not None and 0.0 < reuse_ratio < 0.999:
        return True
    target_ratio = _float_or_none(row.get("target_prefix_ratio"))
    if target_ratio is not None and 0.0 < target_ratio < 0.999:
        return True
    return False


def estimate_trace_reusable_tokens(
    row: dict[str, Any], exact_prefix: bool, structural_reuse: bool, group_size: int
) -> int:
    explicit = _int_or_none(row.get("estimated_reusable_tokens"))
    if explicit is not None:
        return max(0, explicit)

    context_length = _int_or_none(row.get("context_length"))
    estimated_kv_tokens = _int_or_none(row.get("estimated_kv_tokens"))
    if context_length is None and estimated_kv_tokens is not None:
        context_length = estimated_kv_tokens

    reuse_ratio = _float_or_none(row.get("reuse_ratio"))
    if context_length is not None and reuse_ratio is not None and reuse_ratio > 0.0:
        return max(0, round(context_length * reuse_ratio))
    if exact_prefix and context_length is not None and group_size > 1:
        return max(0, context_length)
    if structural_reuse and context_length is not None and reuse_ratio is not None:
        return max(0, round(context_length * reuse_ratio))
    return 0


def build_event_index(event_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in event_rows:
        request_id = str(event.get("request_id") or "")
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        runtime_reqmeta_id = str(metadata.get("runtime_reqmeta_id") or "")
        if runtime_reqmeta_id:
            indexed[(request_id, runtime_reqmeta_id)].append(event)
        indexed[(request_id, "")].append(event)
    return indexed


def events_for_request(
    event_index: dict[tuple[str, str], list[dict[str, Any]]], request_row: dict[str, Any]
) -> list[dict[str, Any]]:
    request_id = str(request_row.get("request_id") or "")
    endpoint_response_id = str(request_row.get("endpoint_response_id") or "")
    if endpoint_response_id:
        keyed = event_index.get((request_id, endpoint_response_id), [])
        if keyed:
            return keyed
    return event_index.get((request_id, ""), [])


def discover_runtime_events_path(request_path: Path) -> Path | None:
    same_dir = request_path.with_name("runtime_events_raw.jsonl")
    if same_dir.is_file():
        return same_dir
    state_dir = request_path.parent.parent / "state" / "runtime_events_raw.jsonl"
    if state_dir.is_file():
        return state_dir
    return None


def classify_request_pattern(
    *,
    source_pattern_class: str,
    exact_prefix: bool,
    structural_reuse: bool,
    group_size: int,
    nearest_distance: int | None,
) -> str:
    if source_pattern_class == "fixed-revisit" and group_size > 1:
        return "fixed-revisit"
    if exact_prefix and nearest_distance == 1:
        return "exact-next"
    if structural_reuse:
        return "structural-partial"
    if exact_prefix and group_size > 1:
        return "structural-partial"
    return ""


def neighbor_distances(
    positions: list[int], current_position: int | None
) -> tuple[int | None, int | None, int | None]:
    if current_position is None:
        return None, None, None
    if current_position not in positions:
        return None, None, None
    index = positions.index(current_position)
    prev_distance = None
    next_distance = None
    if index > 0:
        prev_distance = current_position - positions[index - 1]
    if index + 1 < len(positions):
        next_distance = positions[index + 1] - current_position
    candidates = [value for value in (prev_distance, next_distance) if value is not None]
    nearest = min(candidates) if candidates else None
    return prev_distance, next_distance, nearest


def percentile_pair(values: list[int]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    p50 = float(median(ordered))
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    p95 = float(ordered[p95_index])
    return p50, p95


def dominant_string(values: Iterable[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        normalized = str(value or "")
        if normalized:
            counts[normalized] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def load_jsonl_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames = sorted({key for row in materialized for key in row.keys()})
    if not fieldnames:
        fieldnames = ["empty"]
        materialized = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def write_summary_markdown(
    path: Path,
    summaries: list[SourceSummary],
    request_rows: list[dict[str, Any]],
    kv_bytes_per_token: int,
) -> None:
    lines = [
        "# Unified Reuse Source Summary",
        "",
        f"- KV bytes per token assumption: `{kv_bytes_per_token}`",
        f"- Total request rows: `{len(request_rows)}`",
        f"- Total sources: `{len(summaries)}`",
        "",
        "| Source | Kind | Requests | Repeated rows | Dominant class | Exact-next | Fixed-revisit | Structural-partial | Consecutive reuse ratio |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item.source_name} | {item.source_kind} | {item.request_count} | "
            f"{item.repeated_request_count} | {item.dominant_class} | "
            f"{item.exact_next_request_count} | {item.fixed_revisit_request_count} | "
            f"{item.structural_partial_request_count} | {item.consecutive_reuse_ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `grouped_prompt` rows are workload-only evidence from `grouped_prompts.jsonl`.",
            "- `request_result` rows are trace-backed evidence joined with `runtime_events_raw.jsonl` when present.",
            "- `exact-next` means exact prefix plus nearest same-group distance equals 1.",
            "- `fixed-revisit` means the benchmark family is explicitly organized around scheduled revisits.",
            "- `structural-partial` covers partial-prefix reuse and exact matches that are not adjacent enough to count as next-request locality.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sortable_number(value: Any) -> float:
    number = _float_or_none(value)
    if number is None:
        return float("inf")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
