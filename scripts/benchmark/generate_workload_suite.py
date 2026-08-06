"""Generate the AstraKV-W competition workload suite."""

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

from astrakv.benchmarks.workload_suite import (  # noqa: E402
    WORKLOAD_SCHEMA_VERSION,
    WorkloadCase,
    build_competition_workload_suite,
    summarize_workload_cases,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_competition_workload_suite(
        long_context_tokens=args.long_context_tokens,
        memory_pressure_tokens=args.memory_pressure_tokens,
        repeated_prefix_tokens=args.repeated_prefix_tokens,
    )

    suite_path = output_dir / args.suite_name
    manifest_path = output_dir / args.manifest_name
    report_path = output_dir / args.report_name
    write_suite_jsonl(suite_path, cases)
    write_manifest(manifest_path, args, cases, suite_path)
    write_report(report_path, args, cases, suite_path, manifest_path)
    print(f"Workload suite written to {suite_path}")
    print(f"Workload manifest written to {manifest_path}")
    print(f"Workload report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmarks/prompts")
    parser.add_argument("--suite-name", default="competition_workload_suite.jsonl")
    parser.add_argument("--manifest-name", default="competition_workload_manifest.json")
    parser.add_argument("--report-name", default="competition_workload_report.md")
    parser.add_argument("--long-context-tokens", type=int, default=4096)
    parser.add_argument("--memory-pressure-tokens", type=int, default=8192)
    parser.add_argument("--repeated-prefix-tokens", type=int, default=2048)
    return parser.parse_args()


def write_suite_jsonl(path: Path, cases: list[WorkloadCase]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_record(), ensure_ascii=False) + "\n")


def write_manifest(path: Path, args: argparse.Namespace, cases: list[WorkloadCase], suite_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema": WORKLOAD_SCHEMA_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "suite_path": str(suite_path),
        "parameters": {
            "long_context_tokens": args.long_context_tokens,
            "memory_pressure_tokens": args.memory_pressure_tokens,
            "repeated_prefix_tokens": args.repeated_prefix_tokens,
        },
        "summary": summarize_workload_cases(cases),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    cases: list[WorkloadCase],
    suite_path: Path,
    manifest_path: Path,
) -> None:
    summary = summarize_workload_cases(cases)
    lines = [
        "# Competition Workload Suite Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Parameters",
        "",
        f"- Long context tokens: `{args.long_context_tokens}`",
        f"- Memory pressure tokens: `{args.memory_pressure_tokens}`",
        f"- Repeated prefix tokens: `{args.repeated_prefix_tokens}`",
        "",
        "## Summary",
        "",
        f"- Schema: `{WORKLOAD_SCHEMA_VERSION}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Max context length: `{summary['max_context_length']}`",
        f"- Total expected output tokens: `{summary['total_expected_output_tokens']}`",
        "",
        "### Workload Types",
        "",
        "| workload type | count |",
        "| --- | ---: |",
    ]
    for workload_type, count in summary["type_counts"].items():
        lines.append(f"| {workload_type} | {count} |")
    lines.extend(
        [
            "",
            "### Cases",
            "",
            "| sample | type | context | output | repeat group | tags |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case.sample_id} | {case.workload_type} | {case.context_length} | "
            f"{case.expected_output_tokens} | {case.repeat_group} | {', '.join(case.tags)} |"
        )

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "Quality evaluation endpoint mode:",
            "",
            "```bash",
            "python scripts/research/evaluate_quality.py \\",
            f"  --prompts {suite_path} \\",
            "  --baseline-base-url http://127.0.0.1:8000 \\",
            "  --variant-base-url http://127.0.0.1:8001 \\",
            "  --temperature 0.0 \\",
            "  --top-p 1.0 \\",
            "  --output-dir results/p1_8_quality_suite",
            "```",
            "",
            "## Artifacts",
            "",
            f"- `{suite_path}`",
            f"- `{manifest_path}`",
            "- `competition_workload_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
