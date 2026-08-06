"""Build a unified AstraKV-W trace store from P0 artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.trace_schema import (  # noqa: E402
    TraceEvent,
    load_jsonl,
    load_memory_samples,
    summarize_trace_events,
    trace_from_cache_record,
    trace_from_prefetch_record,
    validate_trace_record,
    write_trace_jsonl,
)
from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl, workload_request_mapping  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = build_events(args)
    trace_path = output_dir / args.trace_name
    summary_path = output_dir / args.summary_name
    write_trace_jsonl(events, trace_path)
    write_summary(summary_path, events, args, trace_path)
    print(f"Trace events written to {trace_path}")
    print(f"Trace summary written to {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-events",
        action="append",
        default=[],
        help="P0-5 cache_events.jsonl path. Can be provided multiple times.",
    )
    parser.add_argument("--workload-manifest", default="", help="Canonical runtime workload JSONL for request identity enrichment.")
    parser.add_argument("--run-id", default="", help="Run identifier attached to linked trace events.")
    parser.add_argument(
        "--prefetch-events",
        action="append",
        default=[],
        help="P0-8 prefetch_events.jsonl path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--samples",
        action="append",
        default=[],
        help="Continuous samples CSV path or a directory containing *_samples.csv. Can be repeated.",
    )
    parser.add_argument("--diagnostic-samples", action="append", default=[], help="Diagnostic CSV produced by diagnose_runtime.py. Can be repeated.")
    parser.add_argument("--output-dir", default="results/trace_store")
    parser.add_argument("--trace-name", default="trace_events.jsonl")
    parser.add_argument("--summary-name", default="trace_summary.md")
    return parser.parse_args()


def build_events(args: argparse.Namespace) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in args.cache_events:
        for record in load_jsonl(path):
            events.append(trace_from_cache_record(record))
    for path in args.prefetch_events:
        for record in load_jsonl(path):
            events.append(trace_from_prefetch_record(record))
    for raw_path in args.samples:
        for sample_path in expand_sample_paths(raw_path):
            events.extend(load_memory_samples(sample_path))
    for raw_path in getattr(args, "diagnostic_samples", []):
        events.extend(load_memory_samples(raw_path))
    workload_manifest = str(getattr(args, "workload_manifest", "") or "")
    run_id = str(getattr(args, "run_id", "") or "")
    if not workload_manifest:
        return [_mark_legacy_event(event, run_id) for event in events]
    request_mapping = workload_request_mapping(load_runtime_workload_jsonl(workload_manifest))
    return [_enrich_event(event, request_mapping, run_id) for event in events]


def _enrich_event(event: TraceEvent, request_mapping: dict[str, dict[str, Any]], run_id: str) -> TraceEvent:
    row = request_mapping.get(event.request_id)
    if row is None:
        return _mark_legacy_event(event, run_id)
    metadata = {
        **event.metadata,
        "run_id": run_id,
        "prefix_id": row["prefix_id"],
        "prefix_hash": row.get("prefix_hash", ""),
        "cache_key": row.get("cache_key", ""),
        "arrival_index": row["arrival_index"],
        "reuse_ratio": row["reuse_ratio"],
        "reuse_bucket": row["reuse_bucket"],
        "legacy_unlinked": False,
    }
    return replace(
        event,
        case=event.case or str(row.get("case") or ""),
        cache_key=event.cache_key or str(row.get("cache_key") or ""),
        metadata=metadata,
    )


def _mark_legacy_event(event: TraceEvent, run_id: str) -> TraceEvent:
    return replace(event, metadata={**event.metadata, "run_id": run_id, "legacy_unlinked": True})


def expand_sample_paths(path: str | Path) -> list[Path]:
    sample_path = Path(path)
    if sample_path.is_dir():
        return sorted(sample_path.glob("*_samples.csv"))
    return [sample_path]


def write_summary(
    path: Path,
    events: list[TraceEvent],
    args: argparse.Namespace,
    trace_path: Path,
) -> None:
    summary = summarize_trace_events(events)
    validation_errors = collect_validation_errors(events)
    lines = [
        "# Unified Trace Store Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Cache event files: `{', '.join(args.cache_events) if args.cache_events else 'none'}`",
        f"- Prefetch event files: `{', '.join(args.prefetch_events) if args.prefetch_events else 'none'}`",
        f"- Sample inputs: `{', '.join(args.samples) if args.samples else 'none'}`",
        f"- Diagnostic sample inputs: `{', '.join(getattr(args, 'diagnostic_samples', [])) if getattr(args, 'diagnostic_samples', []) else 'none'}`",
        f"- Workload manifest: `{args.workload_manifest or 'none'}`",
        f"- Run id: `{args.run_id or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Trace JSONL: `{trace_path}`",
        "",
        "## Summary",
        "",
        f"- Total events: `{summary['total_events']}`",
        f"- Unique request ids: `{summary['unique_request_ids']}`",
        f"- KV hit rate: `{summary['kv_hit_rate']:.4f}`",
        f"- Prefetch hit rate: `{summary['prefetch_hit_rate']:.4f}`",
        "",
        "### Categories",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    append_count_table(lines, summary["category_counts"])
    lines.extend(["", "### Event Types", "", "| event type | count |", "| --- | ---: |"])
    append_count_table(lines, summary["event_type_counts"])
    lines.extend(["", "### Statuses", "", "| status | count |", "| --- | ---: |"])
    append_count_table(lines, summary["status_counts"])
    lines.extend(["", "### Tiers", "", "| tier | count |", "| --- | ---: |"])
    append_count_table(lines, summary["tier_counts"])
    lines.extend(["", "## Validation", ""])
    if validation_errors:
        lines.append("| event | error |")
        lines.append("| --- | --- |")
        for item in validation_errors[:50]:
            lines.append(f"| `{item['event_id']}` | {item['error']} |")
        if len(validation_errors) > 50:
            lines.append(f"| ... | {len(validation_errors) - 50} more validation error(s) omitted |")
    else:
        lines.append("- All trace records passed schema validation.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This trace store normalizes P0 artifacts; it does not replace original cache, prefetch, or sample files.",
            "- Use this output as the input for ProfileDB, chunk scoring, and competition report figures.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def append_count_table(lines: list[str], counts: dict[str, int]) -> None:
    if not counts:
        lines.append("| none | 0 |")
        return
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")


def collect_validation_errors(events: list[TraceEvent]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for event in events:
        record = event.to_record()
        for error in validate_trace_record(record):
            errors.append({"event_id": record.get("event_id", ""), "error": error})
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
