"""Analyze failure artifacts and emit passive fallback hints.

This script reads archived benchmark, request, prefetch, cache, and scheduler
artifacts. It does not retry requests or modify a runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.failure_recovery import (  # noqa: E402
    FailureEvent,
    RecoveryDecision,
    decide_recovery,
    parse_benchmark_results,
    parse_cache_events,
    parse_prefetch_events,
    parse_request_results,
    parse_scheduler_hints,
    summarize_failure_recovery,
    write_failure_events_jsonl,
    write_fallback_hints_jsonl,
    write_recovery_decisions_csv,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = collect_failure_events(args)
    decisions = decide_recovery(events)

    events_path = output_dir / args.events_name
    decisions_path = output_dir / args.decisions_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_failure_events_jsonl(events_path, events)
    write_recovery_decisions_csv(decisions_path, decisions)
    write_fallback_hints_jsonl(hints_path, decisions)
    write_report(report_path, args, events, decisions, events_path, decisions_path, hints_path)
    write_manifest(manifest_path, args, events, decisions, events_path, decisions_path, hints_path, report_path)

    print(f"Failure events written to {events_path}")
    print(f"Recovery decisions written to {decisions_path}")
    print(f"Fallback hints written to {hints_path}")
    print(f"Failure recovery report written to {report_path}")
    print(f"Failure recovery manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results", action="append", default=[], help="benchmark_results.csv path.")
    parser.add_argument("--request-results", action="append", default=[], help="request_results.jsonl path.")
    parser.add_argument("--prefetch-events", action="append", default=[], help="prefetch_events.jsonl path.")
    parser.add_argument("--cache-events", action="append", default=[], help="cache_events.jsonl path.")
    parser.add_argument("--scheduler-hints", action="append", default=[], help="scheduler/object/fallback hints JSONL path.")
    parser.add_argument("--output-dir", default="results/failure_recovery")
    parser.add_argument("--events-name", default="failure_events.jsonl")
    parser.add_argument("--decisions-name", default="recovery_decisions.csv")
    parser.add_argument("--hints-name", default="fallback_hints.jsonl")
    parser.add_argument("--report-name", default="failure_recovery_report.md")
    parser.add_argument("--manifest-name", default="failure_recovery_manifest.json")
    return parser.parse_args()


def collect_failure_events(args: argparse.Namespace) -> list[FailureEvent]:
    events: list[FailureEvent] = []
    for path in args.benchmark_results:
        events.extend(parse_benchmark_results(path))
    for path in args.request_results:
        events.extend(parse_request_results(path))
    for path in args.prefetch_events:
        events.extend(parse_prefetch_events(path))
    for path in args.cache_events:
        events.extend(parse_cache_events(path))
    for path in args.scheduler_hints:
        events.extend(parse_scheduler_hints(path))
    return events


def write_report(
    path: Path,
    args: argparse.Namespace,
    events: list[FailureEvent],
    decisions: list[RecoveryDecision],
    events_path: Path,
    decisions_path: Path,
    hints_path: Path,
) -> None:
    summary = summarize_failure_recovery(events, decisions)
    lines = [
        "# Failure Recovery And Degradation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
    ]
    for label, paths in (
        ("Benchmark results", args.benchmark_results),
        ("Request results", args.request_results),
        ("Prefetch events", args.prefetch_events),
        ("Cache events", args.cache_events),
        ("Scheduler hints", args.scheduler_hints),
    ):
        lines.append(f"- {label}: `{', '.join(paths) if paths else 'none'}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Failure events: `{summary['failure_count']}`",
            f"- Recovery decisions: `{summary['decision_count']}`",
            f"- Critical/error failures: `{summary['critical_or_error_count']}`",
            "",
            "### Failure Types",
            "",
            "| failure type | count |",
            "| --- | ---: |",
        ]
    )
    for failure_type, count in summary["failure_type_counts"].items():
        lines.append(f"| {failure_type} | {count} |")

    lines.extend(["", "### Recovery Actions", "", "| action | count |", "| --- | ---: |"])
    for action, count in summary["recovery_action_counts"].items():
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| component | failure | action | priority | request | object | case | reason |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for decision in decisions[:80]:
        lines.append(
            f"| {decision.component} | {decision.failure_type} | {decision.action.value} | "
            f"{decision.priority} | {decision.request_id or 'n/a'} | {decision.object_id or 'n/a'} | "
            f"{decision.case or 'n/a'} | {decision.reason} |"
        )
    if len(decisions) > 80:
        lines.append(f"| ... |  |  |  |  |  |  | {len(decisions) - 80} more decision(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Recovery decisions are passive hints for adapters and operators.",
            "- `fallback_baseline` keeps the request on the stable baseline path.",
            "- `disable_prefetch` avoids repeating failed or wasteful prefetch work.",
            "- `recompute` is the safe fallback when cache/tier loading fails.",
            "- `reduce_workload` is used for OOM or allocation failures.",
            "- Real recovery claims require runtime logs proving that fallback hints were consumed.",
            "",
            "## Artifacts",
            "",
            f"- `{events_path}`",
            f"- `{decisions_path}`",
            f"- `{hints_path}`",
            "- `failure_recovery_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    events: list[FailureEvent],
    decisions: list[RecoveryDecision],
    events_path: Path,
    decisions_path: Path,
    hints_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-failure-recovery-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "benchmark_results": args.benchmark_results,
            "request_results": args.request_results,
            "prefetch_events": args.prefetch_events,
            "cache_events": args.cache_events,
            "scheduler_hints": args.scheduler_hints,
        },
        "outputs": {
            "failure_events": str(events_path),
            "recovery_decisions": str(decisions_path),
            "fallback_hints": str(hints_path),
            "report": str(report_path),
        },
        "summary": summarize_failure_recovery(events, decisions),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
