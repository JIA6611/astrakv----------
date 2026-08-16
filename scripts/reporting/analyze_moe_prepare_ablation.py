#!/usr/bin/env python3
"""Validate and summarize a paired MoE request-ahead prefill demonstration."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _by_request(_read_jsonl(baseline_dir / "request_results.jsonl"))
    prepared = _by_request(_read_jsonl(prepared_dir / "request_results.jsonl"))
    receipts = _by_request(_read_jsonl(prepared_dir / "moe_prepare_receipts.jsonl"))
    route_events = _read_jsonl(prepared_dir / "moe_route_events.jsonl")
    cache_events = _read_jsonl(prepared_dir / "runtime_events_raw.jsonl")
    paired_ids = sorted(set(baseline) & set(prepared))

    failures: list[str] = []
    if not paired_ids:
        failures.append("no paired request IDs were found")
    if set(baseline) != set(prepared):
        failures.append("baseline and prepared request ID sets differ")
    for request_id in paired_ids:
        left, right = baseline[request_id], prepared[request_id]
        if left.get("status") != "ok" or right.get("status") != "ok":
            failures.append(f"request {request_id} did not complete successfully in both arms")
        if right.get("moe_prepare_status") != "prepared":
            failures.append(f"request {request_id} has MoE status {right.get('moe_prepare_status')}")
        if int(right.get("moe_route_event_count") or 0) <= 0:
            failures.append(f"request {request_id} has no routed-expert events")
        if list(left.get("output_token_ids") or []) != list(right.get("output_token_ids") or []):
            failures.append(f"request {request_id} generated different token IDs")
        receipt = receipts.get(request_id)
        if receipt is None:
            failures.append(f"request {request_id} has no prepare receipt")
        elif float(receipt.get("ended_s") or 0.0) > float(right.get("request_started_s") or 0.0):
            failures.append(f"request {request_id} prepare ended after measured request start")
        elif int(receipt.get("layers") or 0) != 40 or int(receipt.get("top_k") or 0) != 8:
            failures.append(f"request {request_id} routed-expert shape is not [tokens,40,8]")

    invalid_experts = [
        row for row in route_events
        if _int_or_none(row.get("expert_id")) is None
        or not 0 <= int(row["expert_id"]) <= 255
    ]
    if invalid_experts:
        failures.append(f"{len(invalid_experts)} route events contain expert IDs outside 0..255")
    cache_hit_count = sum(1 for row in cache_events if _is_structured_cache_hit(row))
    if cache_hit_count <= 0:
        failures.append("prepared arm has no structured LMCache hit evidence")

    context_rows = _context_summary(baseline, prepared, paired_ids)
    report = {
        "schema": "astrakv-moe-prepare-ablation-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_dir": str(baseline_dir),
        "prepared_dir": str(prepared_dir),
        "paired_request_count": len(paired_ids),
        "prepared_request_count": sum(
            1 for row in prepared.values() if row.get("moe_prepare_status") == "prepared"
        ),
        "route_event_count": len(route_events),
        "unique_expert_count": len({str(row.get("expert_id")) for row in route_events}),
        "structured_cache_hit_event_count": cache_hit_count,
        "output_token_ids_equal": all(
            list(baseline[key].get("output_token_ids") or [])
            == list(prepared[key].get("output_token_ids") or [])
            for key in paired_ids
        ),
        "contexts": context_rows,
        "status": "accepted" if not failures else "failed",
        "failures": failures,
        "claim_boundary": (
            "real request-ahead MoE prefill and KV reuse; no selective expert-weight paging"
        ),
    }
    (output_dir / "moe_prepare_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "moe_prepare_ablation.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 1


def _context_summary(
    baseline: dict[str, dict[str, Any]],
    prepared: dict[str, dict[str, Any]],
    paired_ids: Iterable[str],
) -> list[dict[str, Any]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for request_id in paired_ids:
        groups[int(prepared[request_id].get("context_length") or 0)].append(request_id)
    rows: list[dict[str, Any]] = []
    for context_length, request_ids in sorted(groups.items()):
        baseline_ttft = [float(baseline[key]["ttft_ms"]) for key in request_ids]
        prepared_ttft = [float(prepared[key]["ttft_ms"]) for key in request_ids]
        baseline_median = statistics.median(baseline_ttft)
        prepared_median = statistics.median(prepared_ttft)
        rows.append({
            "context_length": context_length,
            "request_count": len(request_ids),
            "baseline_median_ttft_ms": baseline_median,
            "prepared_median_ttft_ms": prepared_median,
            "median_ttft_delta_ms": prepared_median - baseline_median,
            "median_ttft_improvement_ratio": (
                (baseline_median - prepared_median) / baseline_median
                if baseline_median > 0 else None
            ),
        })
    return rows


def _is_structured_cache_hit(record: dict[str, Any]) -> bool:
    action = str(record.get("action") or record.get("event_type") or "").lower()
    status = str(record.get("status") or "").lower()
    if "cache_hit" in action or action in {"hit", "retrieve_hit", "lookup_hit"}:
        return status not in {"error", "failed", "miss"}
    for mapping in (record, record.get("metadata")):
        if not isinstance(mapping, dict):
            continue
        for key in ("cache_hit_blocks", "hit_blocks", "retrieved_blocks", "num_hit_tokens"):
            value = _int_or_none(mapping.get(key))
            if value is not None and value > 0:
                return True
    return False


def _by_request(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("request_id")): row
        for row in rows
        if row.get("request_id") not in (None, "")
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Request-Ahead Prefill Ablation",
        "",
        f"- Status: `{report['status']}`",
        f"- Paired requests: `{report['paired_request_count']}`",
        f"- Prepared requests: `{report['prepared_request_count']}`",
        f"- Route events: `{report['route_event_count']}`",
        f"- Unique experts: `{report['unique_expert_count']}`",
        f"- Structured cache-hit events: `{report['structured_cache_hit_event_count']}`",
        f"- Output token IDs equal: `{report['output_token_ids_equal']}`",
        "",
        "| context tokens | requests | baseline median TTFT ms | prepared median TTFT ms | delta ms | improvement |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["contexts"]:
        ratio = row["median_ttft_improvement_ratio"]
        ratio_text = "n/a" if ratio is None else f"{100.0 * ratio:.2f}%"
        lines.append(
            f"| {row['context_length']} | {row['request_count']} | "
            f"{row['baseline_median_ttft_ms']:.3f} | {row['prepared_median_ttft_ms']:.3f} | "
            f"{row['median_ttft_delta_ms']:.3f} | {ratio_text} |"
        )
    lines.extend(["", "## Acceptance", ""])
    if report["failures"]:
        lines.extend(f"- FAILED: {item}" for item in report["failures"])
    else:
        lines.append("- All route, timing, output, and cache-evidence checks passed.")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
