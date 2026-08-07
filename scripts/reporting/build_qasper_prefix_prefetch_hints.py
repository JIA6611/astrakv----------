"""Build prefix-prefetch scheduler hints from canonical Qasper workloads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl


def main() -> int:
    args = parse_args()
    rows = load_runtime_workload_jsonl(args.workload_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.prefix_id].append(row)

    hints_path = output_dir / "prefix_prefetch_hints.jsonl"
    report_path = output_dir / "prefix_prefetch_hint_report.json"
    hints: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []

    for prefix_id, prefix_rows in sorted(grouped.items()):
        ordered = sorted(prefix_rows, key=lambda item: item.arrival_index)
        if len(ordered) < 2:
            continue
        reuse_distances = [
            ordered[index].arrival_index - ordered[index - 1].arrival_index
            for index in range(1, len(ordered))
        ]
        avg_distance = sum(reuse_distances) / max(1, len(reuse_distances))
        anchor = ordered[0]
        priority = max(1, min(100, int(round(anchor.reuse_ratio * 100))))
        metadata = {
            "prefix_id": prefix_id,
            "prefix_hash": anchor.prefix_hash or prefix_id,
            "cache_key": anchor.cache_key or prefix_id,
            "object_key": prefix_id,
            "expected_revisit_distance": avg_distance,
            "request_count": len(ordered),
            "source_workload": str(args.workload_jsonl),
            "workload_family": args.workload_family,
        }
        hints.append({
            "request_id": "",
            "action": "prefetch",
            "reason": "qasper prefix reuse profile nominated this group for runtime prefix prefetch",
            "priority": priority,
            "metadata": metadata,
        })
        report_rows.append({
            "prefix_id": prefix_id,
            "request_count": len(ordered),
            "avg_revisit_distance": avg_distance,
            "priority": priority,
            "reuse_ratio": anchor.reuse_ratio,
        })

    with hints_path.open("w", encoding="utf-8") as handle:
        for row in hints:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report_path.write_text(
        json.dumps(
            {
                "schema": "astrakv-qasper-prefix-prefetch-hints-v1",
                "workload_jsonl": str(args.workload_jsonl),
                "hint_count": len(hints),
                "rows": report_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(hints_path)
    print(report_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload-family", default="qasper_grouped")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
