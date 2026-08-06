"""Extract read-only cache events from real benchmark artifacts.

This tool parses server logs, request JSONL, and benchmark CSV files that were
already produced by vLLM/LMCache benchmark runs. It does not import or modify
vLLM, LMCache, or any third-party runtime code.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.cache_events import (  # noqa: E402
    CacheEvent,
    parse_benchmark_results,
    parse_request_results,
    parse_server_log,
    summarize_events,
    write_events_jsonl,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events: list[CacheEvent] = []
    inputs: dict[str, Any] = {
        "server_logs": [str(item) for item in args.server_log],
        "request_results": str(args.request_results) if args.request_results else "",
        "benchmark_results": str(args.benchmark_results) if args.benchmark_results else "",
    }

    for log_path in args.server_log:
        events.extend(parse_server_log(log_path))

    if args.request_results:
        events.extend(parse_request_results(args.request_results))

    if args.benchmark_results:
        events.extend(parse_benchmark_results(args.benchmark_results))

    jsonl_path = output_dir / args.events_name
    summary_path = output_dir / args.summary_name
    write_events_jsonl(events, jsonl_path)
    write_summary(summary_path, events, inputs, jsonl_path)
    print(f"Cache events written to {jsonl_path}")
    print(f"Cache event summary written to {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-log",
        action="append",
        default=[],
        help="Path to a vLLM/LMCache server log. Can be provided multiple times.",
    )
    parser.add_argument("--request-results", help="Path to request_results.jsonl.")
    parser.add_argument("--benchmark-results", help="Path to benchmark_results.csv.")
    parser.add_argument("--output-dir", default="results/cache_events")
    parser.add_argument("--events-name", default="cache_events.jsonl")
    parser.add_argument("--summary-name", default="cache_event_summary.md")
    return parser.parse_args()


def write_summary(
    path: Path,
    events: list[CacheEvent],
    inputs: dict[str, Any],
    jsonl_path: Path,
) -> None:
    summary = summarize_events(events)
    examples = events[:10]
    lines = [
        "# Cache Event Extraction Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Server logs: `{', '.join(inputs['server_logs']) if inputs['server_logs'] else 'none'}`",
        f"- Request results: `{inputs['request_results'] or 'none'}`",
        f"- Benchmark results: `{inputs['benchmark_results'] or 'none'}`",
        "",
        "## Outputs",
        "",
        f"- Events JSONL: `{jsonl_path}`",
        "",
        "## Summary",
        "",
        f"- Total events: `{summary['total_events']}`",
        "",
        "### Event Types",
        "",
        "| event type | count |",
        "| --- | ---: |",
    ]
    for event_type, count in summary["event_type_counts"].items():
        lines.append(f"| {event_type} | {count} |")

    lines.extend(["", "### Statuses", "", "| status | count |", "| --- | ---: |"])
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "### Tiers", "", "| tier | count |", "| --- | ---: |"])
    for tier, count in summary["tier_counts"].items():
        lines.append(f"| {tier} | {count} |")

    lines.extend(
        [
            "",
            "## First Events",
            "",
            "| type | status | source | request | tier | line |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for event in examples:
        line_number = "" if event.line_number is None else str(event.line_number)
        lines.append(
            f"| {event.event_type} | {event.status} | {event.source} | "
            f"{event.request_id or 'n/a'} | {event.tier} | {line_number} |"
        )

    lines.extend(
        [
            "",
            "## Validation Notes",
            "",
            "- `request_result` and `benchmark_case_metrics` events provide benchmark context even when LMCache logs are sparse.",
            "- `cache_hit`, `cache_miss`, `cache_load`, `cache_store`, and `cache_offload` depend on installed LMCache/vLLM log wording.",
            "- This extractor is read-only and does not prove that an optimization occurred by itself. Use it together with server logs and benchmark reports.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
