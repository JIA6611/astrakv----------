"""Extract read-only MoE expert activation events from router artifacts.

This tool parses MoE/router logs and JSONL event exports that were already
produced by a serving run. It does not import or modify model/runtime internals.
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

from astrakv.runtime.moe_events import (  # noqa: E402
    MoEExpertEvent,
    parse_events_jsonl,
    parse_router_log,
    summarize_events,
    write_events_jsonl,
    write_expert_summary_csv,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events: list[MoEExpertEvent] = []
    inputs: dict[str, Any] = {
        "router_logs": [str(item) for item in args.router_log],
        "events_jsonl": [str(item) for item in args.events_jsonl],
    }

    for log_path in args.router_log:
        events.extend(parse_router_log(log_path))
    for jsonl_path in args.events_jsonl:
        events.extend(parse_events_jsonl(jsonl_path))

    events_path = output_dir / args.events_name
    summary_csv_path = output_dir / args.summary_csv_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_events_jsonl(events, events_path)
    write_expert_summary_csv(events, summary_csv_path)
    write_report(report_path, events, inputs, events_path, summary_csv_path)
    write_manifest(manifest_path, events, inputs, events_path, summary_csv_path, report_path)

    print(f"MoE expert events written to {events_path}")
    print(f"MoE expert summary CSV written to {summary_csv_path}")
    print(f"MoE expert report written to {report_path}")
    print(f"MoE expert manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-log",
        action="append",
        default=[],
        help="Path to a MoE/router log. Can be provided multiple times.",
    )
    parser.add_argument(
        "--events-jsonl",
        action="append",
        default=[],
        help="Path to JSONL MoE expert events. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", default="results/moe_expert_events")
    parser.add_argument("--events-name", default="moe_expert_events.jsonl")
    parser.add_argument("--summary-csv-name", default="moe_expert_summary.csv")
    parser.add_argument("--report-name", default="moe_expert_report.md")
    parser.add_argument("--manifest-name", default="moe_expert_manifest.json")
    return parser.parse_args()


def write_report(
    path: Path,
    events: list[MoEExpertEvent],
    inputs: dict[str, Any],
    events_path: Path,
    summary_csv_path: Path,
) -> None:
    summary = summarize_events(events)
    expert_rows = summary["expert_rows"][:20]
    event_examples = events[:10]
    avg_latency = summary["avg_latency_ms"]
    avg_latency_text = "n/a" if avg_latency is None else f"{avg_latency:.3f}"

    lines = [
        "# MoE Expert Activation Trace Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Router logs: `{', '.join(inputs['router_logs']) if inputs['router_logs'] else 'none'}`",
        f"- Events JSONL: `{', '.join(inputs['events_jsonl']) if inputs['events_jsonl'] else 'none'}`",
        "",
        "## Outputs",
        "",
        f"- MoE events JSONL: `{events_path}`",
        f"- Expert summary CSV: `{summary_csv_path}`",
        "",
        "## Summary",
        "",
        f"- Total events: `{summary['total_events']}`",
        f"- Unique requests: `{summary['unique_request_ids']}`",
        f"- Routed tokens: `{summary['routed_token_count']}`",
        f"- Unique expert ids: `{summary['unique_expert_ids']}`",
        f"- Unique layer-expert pairs: `{summary['unique_layer_experts']}`",
        f"- Route events: `{summary['route_event_count']}`",
        f"- Expert hit rate: `{summary['expert_hit_rate']:.6f}`",
        f"- Expert prefetch events: `{summary['expert_prefetch_count']}`",
        f"- Expert offload events: `{summary['expert_offload_count']}`",
        f"- Expert memory bytes observed: `{summary['expert_memory_bytes']}`",
        f"- Average event latency ms: `{avg_latency_text}`",
        "",
        "### Event Types",
        "",
        "| event type | count |",
        "| --- | ---: |",
    ]
    for event_type, count in summary["event_type_counts"].items():
        lines.append(f"| {event_type} | {count} |")

    lines.extend(["", "### Tiers", "", "| tier | count |", "| --- | ---: |"])
    for tier, count in summary["tier_counts"].items():
        lines.append(f"| {tier} | {count} |")

    lines.extend(
        [
            "",
            "## Hot Experts",
            "",
            "| rank | layer | expert | activations | tokens | avg score | bytes | latency ms | share |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in expert_rows:
        lines.append(
            f"| {row['hotness_rank']} | {row['layer_id']} | {row['expert_id']} | "
            f"{row['activation_count']} | {row['token_count']} | {row['avg_score']} | "
            f"{row['bytes']} | {row['latency_ms']} | {row['hotness_share']} |"
        )

    lines.extend(
        [
            "",
            "## First Events",
            "",
            "| type | status | request | layer | token | expert | rank | score | tier | line |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for event in event_examples:
        line_number = "" if event.line_number is None else str(event.line_number)
        lines.append(
            f"| {event.event_type} | {event.status} | {event.request_id or 'n/a'} | "
            f"{_empty(event.layer_id)} | {_empty(event.token_index)} | {event.expert_id or 'n/a'} | "
            f"{_empty(event.expert_rank)} | {_empty(event.score)} | {event.tier} | {line_number} |"
        )

    lines.extend(
        [
            "",
            "## Validation Notes",
            "",
            "- This extractor proves that MoE routing evidence can be normalized and summarized.",
            "- Real MoE claims require logs or JSONL records from an actual MoE model run.",
            "- Expert memory bytes and hit/miss rates appear only when the upstream runtime or adapter exports them.",
            "- This tool is read-only and does not implement selective expert loading by itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    events: list[MoEExpertEvent],
    inputs: dict[str, Any],
    events_path: Path,
    summary_csv_path: Path,
    report_path: Path,
) -> None:
    summary = summarize_events(events)
    manifest = {
        "schema": "astra-moe-extraction-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,
        "outputs": {
            "events_jsonl": str(events_path),
            "summary_csv": str(summary_csv_path),
            "report": str(report_path),
        },
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "expert_rows"
        },
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _empty(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
