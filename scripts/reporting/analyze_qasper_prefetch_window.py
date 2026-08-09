"""Analyze inter-arrival feasibility and optionally emit a paced Qasper workload."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow, load_runtime_workload_jsonl


def main() -> int:
    args = parse_args()
    rows = load_runtime_workload_jsonl(args.workload_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[RuntimeWorkloadRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prefix_id].append(row)

    analysis_rows: list[dict[str, object]] = []
    paced_rows: list[dict[str, object]] = []
    prefetch_leads: dict[str, float] = {}
    required_window_ms = max(0.0, float(args.required_window_ms))
    borderline_ratio = max(0.0, float(args.borderline_ratio))

    for row in sorted(rows, key=lambda item: item.arrival_index):
        paced_rows.append(row.to_record())

    for prefix_id, prefix_rows in sorted(grouped.items()):
        ordered = sorted(prefix_rows, key=lambda item: item.arrival_index)
        for index in range(1, len(ordered)):
            current = ordered[index]
            previous = ordered[index - 1]
            gap = current.arrival_index - previous.arrival_index
            window_ms = gap * float(args.arrival_gap_ms)
            if window_ms >= required_window_ms:
                feasibility = "window_sufficient"
            elif window_ms >= required_window_ms * borderline_ratio:
                feasibility = "window_borderline"
            else:
                feasibility = "window_insufficient"
                prefetch_leads[current.request_id] = max(
                    prefetch_leads.get(current.request_id, 0.0),
                    required_window_ms / 1000.0,
                )
            analysis_rows.append({
                "prefix_id": prefix_id,
                "request_id": current.request_id,
                "previous_request_id": previous.request_id,
                "previous_arrival_index": previous.arrival_index,
                "arrival_index": current.arrival_index,
                "arrival_gap": gap,
                "estimated_inter_arrival_window_ms": window_ms,
                "required_window_ms": required_window_ms,
                "window_feasibility": feasibility,
            })

    for row in paced_rows:
        request_id = str(row.get("request_id") or "")
        if request_id in prefetch_leads:
            # This is published intent -> HTTP dispatch time, not a delay
            # before ingress. It therefore represents a real opportunity for
            # SSD -> LocalCPUBackend work to overlap before the target runs.
            row["prefetch_lead_s"] = max(
                float(row.get("prefetch_lead_s") or 0.0), prefetch_leads[request_id],
            )

    analysis_path = output_dir / "prefetch_window_analysis.jsonl"
    summary_path = output_dir / "prefetch_window_summary.json"
    paced_path = output_dir / "qasper_grouped_prefetch_friendly_workload.jsonl"

    with analysis_path.open("w", encoding="utf-8") as handle:
        for row in analysis_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with paced_path.open("w", encoding="utf-8") as handle:
        for row in paced_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for row in analysis_rows:
        counts[str(row["window_feasibility"])] += 1
    summary = {
        "schema": "astrakv-qasper-prefetch-window-v1",
        "workload_jsonl": str(args.workload_jsonl),
        "required_window_ms": required_window_ms,
        "arrival_gap_ms": float(args.arrival_gap_ms),
        "counts": dict(sorted(counts.items())),
        "analysis_rows": len(analysis_rows),
        "prefetch_lead_request_count": len(prefetch_leads),
        "paced_workload_jsonl": str(paced_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(analysis_path)
    print(summary_path)
    print(paced_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-window-ms", type=float, default=50.0)
    parser.add_argument("--borderline-ratio", type=float, default=0.75)
    parser.add_argument("--arrival-gap-ms", type=float, default=25.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
