"""Score ProfileDB chunks for prefetch, keep, offload, or drop decisions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.prefetch.scorer import ChunkScore, ChunkScorer, ChunkScorerConfig  # noqa: E402
from astrakv.runtime.profile_db import ProfileDB  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = ProfileDB.load(args.profile_db)
    scorer = ChunkScorer(config_from_args(args))
    scores = scorer.score_db(db)
    csv_path = output_dir / args.csv_name
    report_path = output_dir / args.report_name
    write_csv(csv_path, scores)
    write_report(report_path, args, scores, csv_path)
    print(f"Chunk scores written to {csv_path}")
    print(f"Chunk score report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-db", required=True, help="ProfileDB JSON path from scripts/policy/build_profile_db.py.")
    parser.add_argument("--output-dir", default="results/chunk_scores")
    parser.add_argument("--csv-name", default="chunk_scores.csv")
    parser.add_argument("--report-name", default="chunk_score_report.md")
    parser.add_argument("--memory-pressure", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--load-latency-reference-ms", type=float, default=100.0)
    parser.add_argument("--size-reference-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--prefetch-threshold", type=float, default=0.62)
    parser.add_argument("--keep-threshold", type=float, default=0.38)
    parser.add_argument("--offload-threshold", type=float, default=0.18)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ChunkScorerConfig:
    return ChunkScorerConfig(
        memory_pressure=args.memory_pressure,
        deadline_ms=args.deadline_ms,
        load_latency_reference_ms=args.load_latency_reference_ms,
        size_reference_bytes=args.size_reference_bytes,
        prefetch_threshold=args.prefetch_threshold,
        keep_threshold=args.keep_threshold,
        offload_threshold=args.offload_threshold,
    )


def write_csv(path: Path, scores: list[ChunkScore]) -> None:
    rows = [score.to_record() for score in scores]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    scores: list[ChunkScore],
    csv_path: Path,
) -> None:
    action_counts: dict[str, int] = {}
    for score in scores:
        action_counts[score.action.value] = action_counts.get(score.action.value, 0) + 1

    lines = [
        "# Chunk Score Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- ProfileDB: `{args.profile_db}`",
        f"- Memory pressure: `{args.memory_pressure}`",
        f"- Deadline ms: `{args.deadline_ms}`",
        f"- Load latency reference ms: `{args.load_latency_reference_ms}`",
        f"- Size reference bytes: `{args.size_reference_bytes}`",
        "",
        "## Outputs",
        "",
        f"- Chunk scores CSV: `{csv_path}`",
        "",
        "## Action Counts",
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
            "## Top Scores",
            "",
            "| chunk | action | score | reuse | load ms | prefetch hit | reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for score in scores[:20]:
        record = score.to_record()
        lines.append(
            "| {chunk_id} | {action} | {score:.4f} | {reuse_frequency:.4f} | "
            "{avg_load_latency_ms:.4f} | {prefetch_hit_rate:.4f} | {reason} |".format(**record)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `prefetch` means the profile indicates high reuse, deadline pressure, or expensive load with acceptable waste.",
            "- `keep` means the chunk is useful enough to retain when budget allows.",
            "- `offload` means preserve the object in a lower tier under memory pressure.",
            "- `drop` means weak reuse/profile evidence or high waste makes the chunk a low priority.",
            "- Scores are advisory policy hints. Runtime adapters remain responsible for safe execution.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
