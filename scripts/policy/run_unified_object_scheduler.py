"""Run the unified object scheduler MVP from existing policy artifacts."""

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
from astrakv.scheduler.object_scheduler import (  # noqa: E402
    ObjectSchedulerConfig,
    ObjectScheduleDecision,
    UnifiedObjectScheduler,
    candidates_from_profile_db,
    load_decision_from_record,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = ProfileDB.load(args.profile_db)
    chunk_scores = load_chunk_scores(args.chunk_scores) if args.chunk_scores else {}
    load_decisions = load_load_decisions(args.load_recompute_decisions) if args.load_recompute_decisions else {}
    candidates = candidates_from_profile_db(
        db,
        chunk_scores=chunk_scores,
        load_decisions=load_decisions,
        default_size_bytes=args.default_object_bytes,
    )
    scheduler = UnifiedObjectScheduler(config_from_args(args))
    decisions = scheduler.schedule(candidates)

    decisions_path = output_dir / args.decisions_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    write_decisions_csv(decisions_path, decisions)
    write_hints_jsonl(hints_path, decisions)
    write_report(report_path, args, decisions, decisions_path, hints_path)
    print(f"Object schedule decisions written to {decisions_path}")
    print(f"Object scheduler hints written to {hints_path}")
    print(f"Object schedule report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-db", required=True, help="ProfileDB JSON path.")
    parser.add_argument("--chunk-scores", default="", help="Optional P1-3 chunk_scores.csv path.")
    parser.add_argument(
        "--load-recompute-decisions",
        default="",
        help="Optional P1-6 load_recompute_decisions.csv path.",
    )
    parser.add_argument("--output-dir", default="results/unified_object_scheduler")
    parser.add_argument("--decisions-name", default="object_schedule_decisions.csv")
    parser.add_argument("--hints-name", default="object_scheduler_hints.jsonl")
    parser.add_argument("--report-name", default="object_schedule_report.md")
    parser.add_argument("--gpu-budget-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--default-object-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--memory-pressure", type=float, default=0.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ObjectSchedulerConfig:
    return ObjectSchedulerConfig(
        gpu_budget_bytes=args.gpu_budget_bytes,
        default_object_bytes=args.default_object_bytes,
        memory_pressure=args.memory_pressure,
    )


def load_chunk_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    return {str(row.get("chunk_id", "")): row for row in rows if row.get("chunk_id")}


def load_load_decisions(path: str | Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in read_csv(path):
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id:
            continue
        output[chunk_id] = load_decision_from_record(row)
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_decisions_csv(path: Path, decisions: list[ObjectScheduleDecision]) -> None:
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


def write_hints_jsonl(path: Path, decisions: list[ObjectScheduleDecision]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision.to_hint()), ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    args: argparse.Namespace,
    decisions: list[ObjectScheduleDecision],
    decisions_path: Path,
    hints_path: Path,
) -> None:
    action_counts: dict[str, int] = {}
    gpu_resident_bytes = 0
    for decision in decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
        if decision.action.value in {"prefetch", "keep"}:
            gpu_resident_bytes += decision.size_bytes
    budget = max(1, args.gpu_budget_bytes)
    lines = [
        "# Unified Object Scheduler Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- ProfileDB: `{args.profile_db}`",
        f"- Chunk scores: `{args.chunk_scores or 'none'}`",
        f"- Load/recompute decisions: `{args.load_recompute_decisions or 'none'}`",
        f"- GPU budget bytes: `{args.gpu_budget_bytes}`",
        f"- Default object bytes: `{args.default_object_bytes}`",
        f"- Memory pressure: `{args.memory_pressure}`",
        "",
        "## Summary",
        "",
        f"- Decisions: `{len(decisions)}`",
        f"- Scheduled GPU bytes: `{gpu_resident_bytes}`",
        f"- GPU budget utilization: `{gpu_resident_bytes / budget:.4f}`",
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
            "| chunk | action | priority | size bytes | GPU bytes after | source chunk action | source load action | reason |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for decision in decisions[:50]:
        lines.append(
            f"| {decision.chunk_id} | {decision.action.value} | {decision.priority:.4f} | "
            f"{decision.size_bytes} | {decision.gpu_bytes_after} | "
            f"{decision.source_chunk_action} | {decision.source_load_action} | {decision.reason} |"
        )
    if len(decisions) > 50:
        lines.append(f"| ... |  |  |  |  |  |  | {len(decisions) - 50} more decision(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `prefetch` and `keep` consume the configured GPU byte budget.",
            "- `offload` keeps the object in a lower tier when GPU budget is exhausted.",
            "- `recompute`, `defer`, and `drop` follow P1-6 decisions when present.",
            "- These are passive object-scheduling hints; backend adapters must perform real movement and logging.",
            "",
            "## Artifacts",
            "",
            f"- `{decisions_path}`",
            f"- `{hints_path}`",
            "- `object_schedule_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
