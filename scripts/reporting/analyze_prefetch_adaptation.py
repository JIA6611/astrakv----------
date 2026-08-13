"""Temporal adaptation analysis for Prefetch-A/B evidence.

Answers "cold start vs after-N observations": split one ablation role's
requests into arrival-order windows and report, per window, TTFT P50/P95
together with Prefetch-A decisions/tickets and Prefetch-B receipts whose
timestamps fall inside the window's request time span.  A growing prefetch
hit/consumed count (or shrinking waste) across windows is the online
learner adaptation signal (RuntimePrefixIndex EMA).

Expected inputs under ``--role-dir`` and ``--state-dir``:
  request_results.jsonl           (arrival_index, request_started_s, ttft_ms)
  kv_core_policy_decisions.jsonl  (prefetch_ssd_to_cpu, prefetch_id, timestamp_ns)
  kv_core_prefetch_tickets.jsonl  (prefetch_id, status)
  runtime_command_receipts.jsonl  (action=prefetch, status, timestamp_ns)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-prefetch-adaptation-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, int(round(len(sorted_values) * percentile / 100.0 - 0.5))))
    return sorted_values[index]


def _window_bounds(requests: list[dict[str, Any]], windows: int) -> list[tuple[float, float, int, int]]:
    """Return (start_s, end_s, first_arrival, last_arrival) per window."""
    if not requests:
        return []
    chunk = math.ceil(len(requests) / max(1, windows))
    bounds: list[tuple[float, float, int, int]] = []
    for start in range(0, len(requests), chunk):
        window_requests = requests[start:start + chunk]
        starts = [float(row.get("request_started_s") or 0.0) for row in window_requests]
        ends = [float(row.get("request_ended_s") or s) for row, s in zip(window_requests, starts)]
        bounds.append((
            min(starts),
            max(ends),
            int(window_requests[0].get("arrival_index") or 0),
            int(window_requests[-1].get("arrival_index") or 0),
        ))
    return bounds


def analyze_role(
    role_dir: Path,
    state_dir: Path,
    *,
    windows: int,
) -> dict[str, Any]:
    requests = _read_jsonl(role_dir / "request_results.jsonl")
    requests.sort(key=lambda row: int(row.get("arrival_index") or 0))
    decisions = _read_jsonl(state_dir / "kv_core_policy_decisions.jsonl")
    tickets = _read_jsonl(state_dir / "kv_core_prefetch_tickets.jsonl")
    receipts = _read_jsonl(state_dir / "runtime_command_receipts.jsonl")
    if not receipts:
        receipts = _read_jsonl(role_dir / "runtime_command_receipts.jsonl")

    ticket_time: dict[str, float] = {}
    for decision in decisions:
        prefetch_id = str(decision.get("prefetch_id") or "")
        if prefetch_id:
            ticket_time[prefetch_id] = float(decision.get("timestamp_ns") or 0) / 1e9

    bounds = _window_bounds(requests, windows)
    rows: list[dict[str, Any]] = []
    for index, (start_s, end_s, first_arrival, last_arrival) in enumerate(bounds):
        window_requests = [
            row for row in requests
            if first_arrival <= int(row.get("arrival_index") or 0) <= last_arrival
        ]
        ttfts = sorted(
            float(row["ttft_ms"])
            for row in window_requests
            if str(row.get("status") or "") == "ok" and row.get("ttft_ms") not in (None, "")
        )
        a_decisions = [
            row for row in decisions
            if str(row.get("action") or "") == "prefetch_ssd_to_cpu"
            and start_s <= float(row.get("timestamp_ns") or 0) / 1e9 <= end_s
        ]
        a_ids = {str(row.get("prefetch_id") or "") for row in a_decisions}
        a_tickets = [row for row in tickets if str(row.get("prefetch_id") or "") in a_ids]
        b_receipts = [
            row for row in receipts
            if str(row.get("action") or "") == "prefetch"
            and start_s <= float(row.get("timestamp_ns") or 0) / 1e9 <= end_s
        ]
        rows.append({
            "window": index,
            "arrival_range": [first_arrival, last_arrival],
            "request_count": len(window_requests),
            "ttft_p50_ms": _percentile(ttfts, 50),
            "ttft_p95_ms": _percentile(ttfts, 95),
            "prefetch_a": {
                "decision_count": len(a_decisions),
                "ticket_statuses": {
                    status: sum(1 for row in a_tickets if str(row.get("status") or "") == status)
                    for status in sorted({str(row.get("status") or "") for row in a_tickets})
                },
            },
            "prefetch_b": {
                "receipt_count": len(b_receipts),
                "completed_with_bytes": sum(
                    1 for row in b_receipts
                    if str(row.get("status") or "") == "completed"
                    and int(row.get("metadata", {}).get("prefetched") or 0) > 0
                ),
            },
        })

    return {
        "schema": SCHEMA,
        "role_dir": str(role_dir),
        "state_dir": str(state_dir),
        "window_count": len(rows),
        "windows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-dir", required=True)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    role_dir = Path(args.role_dir)
    state_dir = Path(args.state_dir) if args.state_dir else role_dir / "state"
    if not role_dir.is_dir():
        raise SystemExit(f"role dir is not a directory: {role_dir}")
    summary = analyze_role(role_dir, state_dir, windows=max(1, args.windows))
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
