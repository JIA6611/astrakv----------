"""Generate load-vs-recompute decisions from ProfileDB and partial KV plans."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.profile_db import ProfileDB  # noqa: E402
from astrakv.scheduler.decision import (  # noqa: E402
    LoadRecomputeConfig,
    LoadRecomputeDecision,
    LoadRecomputePlanner,
    partial_plan_stats_from_records,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = ProfileDB.load(args.profile_db)
    partial_plans = load_partial_plan(args.partial_plan) if args.partial_plan else {}
    planner = LoadRecomputePlanner(config_from_args(args))
    decisions = planner.decide_db(db, partial_plans=partial_plans)

    decisions_path = output_dir / args.decisions_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    write_decisions_csv(decisions_path, decisions)
    write_hints_jsonl(hints_path, decisions)
    write_report(report_path, args, decisions, decisions_path, hints_path, partial_plans)
    print(f"Load-vs-recompute decisions written to {decisions_path}")
    print(f"Scheduler hints written to {hints_path}")
    print(f"Load-vs-recompute report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-db", required=True, help="ProfileDB JSON path from scripts/policy/build_profile_db.py.")
    parser.add_argument("--partial-plan", default="", help="Optional P1-5 partial_kv_plan.jsonl.")
    parser.add_argument("--output-dir", default="results/load_vs_recompute")
    parser.add_argument("--decisions-name", default="load_recompute_decisions.csv")
    parser.add_argument("--hints-name", default="load_recompute_hints.jsonl")
    parser.add_argument("--report-name", default="load_recompute_report.md")
    parser.add_argument("--memory-pressure", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=120.0)
    parser.add_argument("--load-latency-fallback-ms", type=float, default=80.0)
    parser.add_argument("--io-bandwidth-bytes-per-ms", type=float, default=64 * 1024 * 1024 / 1000)
    parser.add_argument("--recompute-latency-per-token-ms", type=float, default=0.08)
    parser.add_argument("--recompute-overhead-ms", type=float, default=8.0)
    parser.add_argument("--recompute-penalty", type=float, default=1.10)
    parser.add_argument("--default-tokens", type=int, default=1024)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LoadRecomputeConfig:
    return LoadRecomputeConfig(
        memory_pressure=args.memory_pressure,
        deadline_ms=args.deadline_ms,
        load_latency_fallback_ms=args.load_latency_fallback_ms,
        io_bandwidth_bytes_per_ms=args.io_bandwidth_bytes_per_ms,
        recompute_latency_per_token_ms=args.recompute_latency_per_token_ms,
        recompute_overhead_ms=args.recompute_overhead_ms,
        recompute_penalty=args.recompute_penalty,
        default_tokens=args.default_tokens,
    )


def load_partial_plan(path: str | Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    plan_path = Path(path)
    if not plan_path.exists():
        raise SystemExit(f"Partial plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {plan_path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return partial_plan_stats_from_records(records)


def write_decisions_csv(path: Path, decisions: list[LoadRecomputeDecision]) -> None:
    rows = [decision.to_record() for decision in decisions]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0].keys() if key != "metadata"] + ["metadata"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_hints_jsonl(path: Path, decisions: list[LoadRecomputeDecision]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision.to_hint()), ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    args: argparse.Namespace,
    decisions: list[LoadRecomputeDecision],
    decisions_path: Path,
    hints_path: Path,
    partial_plans: dict[str, Any],
) -> None:
    action_counts: dict[str, int] = {}
    for decision in decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
    loaded_bytes = sum(decision.estimated_loaded_bytes for decision in decisions)
    skipped_bytes = sum(decision.estimated_skipped_bytes for decision in decisions)
    byte_saving_rate = skipped_bytes / max(1, loaded_bytes + skipped_bytes)

    lines = [
        "# Load-vs-Recompute Decision Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- ProfileDB: `{args.profile_db}`",
        f"- Partial plan: `{args.partial_plan or 'none'}`",
        f"- Memory pressure: `{args.memory_pressure}`",
        f"- Deadline ms: `{args.deadline_ms}`",
        f"- Recompute latency per token ms: `{args.recompute_latency_per_token_ms}`",
        f"- Recompute overhead ms: `{args.recompute_overhead_ms}`",
        "",
        "## Summary",
        "",
        f"- Decisions: `{len(decisions)}`",
        f"- Partial plan chunks matched by id: `{len(partial_plans)}`",
        f"- Estimated loaded bytes: `{loaded_bytes}`",
        f"- Estimated skipped bytes: `{skipped_bytes}`",
        f"- Estimated byte saving rate: `{byte_saving_rate:.4f}`",
        "",
        "| action | count |",
        "| --- | ---: |",
    ]
    if action_counts:
        for action, count in sorted(action_counts.items()):
            lines.append(f"| {action} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Top Decisions",
            "",
            "| chunk | action | priority | load ms | recompute ms | loaded bytes | skipped bytes | reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for decision in decisions[:50]:
        lines.append(
            f"| {decision.chunk_id} | {decision.action.value} | {decision.priority} | "
            f"{decision.estimated_load_ms:.4f} | {decision.estimated_recompute_ms:.4f} | "
            f"{decision.estimated_loaded_bytes} | {decision.estimated_skipped_bytes} | {decision.reason} |"
        )
    if len(decisions) > 50:
        lines.append(f"| ... |  |  |  |  |  |  | {len(decisions) - 50} more decision(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `load` means estimated IO is cheaper, cache-friendly, or a partial plan keeps IO inside the deadline.",
            "- `recompute` means estimated compute is cheaper than IO under the configured memory pressure.",
            "- `defer` means neither load nor recompute is predicted to satisfy the deadline comfortably.",
            "- `drop` means profile evidence is weak under memory pressure, or P1-5 explicitly skipped the chunk.",
            "- These decisions are passive hints. Backend adapters must still perform safe execution and logging.",
            "",
            "## Artifacts",
            "",
            f"- `{decisions_path}`",
            f"- `{hints_path}`",
            "- `load_recompute_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
