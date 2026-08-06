"""Build a reusable AstraKV-W ProfileDB from unified trace events."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.profile_db import ProfileDB, load_profile_from_trace_jsonl  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = load_profile_from_trace_jsonl(args.trace_events, workload_id=args.workload_id)
    db_path = output_dir / args.db_name
    report_path = output_dir / args.report_name
    db.save(db_path)
    write_report(report_path, db, args, db_path)
    print(f"ProfileDB written to {db_path}")
    print(f"Profile report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-events", required=True, help="Unified astra-trace-v1 JSONL file.")
    parser.add_argument("--workload-id", default="default", help="Reusable workload/profile id.")
    parser.add_argument("--output-dir", default="results/profile_db")
    parser.add_argument("--db-name", default="profile_db.json")
    parser.add_argument("--report-name", default="profile_report.md")
    return parser.parse_args()


def write_report(path: Path, db: ProfileDB, args: argparse.Namespace, db_path: Path) -> None:
    record = db.to_record()
    workloads = record["workloads"]
    chunks = db.top_chunks(limit=20)
    lines = [
        "# ProfileDB Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Trace events: `{args.trace_events}`",
        f"- Workload id: `{args.workload_id}`",
        "",
        "## Outputs",
        "",
        f"- ProfileDB JSON: `{db_path}`",
        "",
        "## Workloads",
        "",
        "| workload | events | requests | cases | backends | GPU MB peak | CPU MB peak | memory samples |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for workload in workloads:
        workload_row = dict(workload)
        workload_row["case_count"] = len(workload.get("cases", []))
        workload_row["backend_list"] = ", ".join(workload.get("backends", [])) or "n/a"
        lines.append(
            "| {workload_id} | {event_count} | {request_count} | {case_count} | {backend_list} | "
            "{gpu_used_peak_mb} | {cpu_rss_peak_mb} | {memory_sample_count} |".format(
                **workload_row,
            )
        )

    lines.extend(
        [
            "",
            "## Top Chunk Profiles",
            "",
            "| chunk | case | reuse freq | cache hit | loads | avg load ms | prefetch hit | bytes loaded | tiers |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for chunk in chunks:
        row = chunk.to_record()
        lines.append(
            "| {chunk_id} | {case} | {reuse_frequency:.4f} | {cache_hit_rate:.4f} | "
            "{cache_loads} | {avg_load_latency_ms:.4f} | {prefetch_hit_rate:.4f} | "
            "{bytes_loaded} | {tiers} |".format(
                **row,
                tiers=", ".join(f"{key}:{value}" for key, value in row["tier_counts"].items()) or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `reuse freq` estimates how often a chunk-like object reappears in request/cache/prefetch events.",
            "- `avg load ms` uses explicit load latency when the trace contains it; otherwise it remains `0.0`.",
            "- Tier counts summarize observed GPU/CPU/SSD/unknown placement hints from trace events.",
            "- This ProfileDB is reusable input for chunk scoring and policy ablations.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
