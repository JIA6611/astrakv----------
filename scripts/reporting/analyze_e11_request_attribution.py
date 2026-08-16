"""Build request-level attribution for an E11 native CPU policy A/B.

The report separates observations that happened before the two victim
sequences diverged from requests that could have been affected by the policy.
It also summarizes native-load receipts and selector hot-path instrumentation
when those artifacts are available.  Missing evidence is reported explicitly;
it is never converted into a zero-valued measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scripts.reporting.evaluate_e11_native_cpu_mvp import _native_rows
from scripts.reporting.evaluate_e11_target import (
    ARMS,
    artifact_path,
    as_float,
    as_int,
    as_str,
    load_jsonl,
    request_index,
    selected_run_dirs,
)


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 4)


def _selected_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        row for row in _native_rows(run_dir)
        if as_str(row.get("status")) == "selected"
    ]


def _first_divergence(
    astra_rows: list[dict[str, Any]],
    lru_rows: list[dict[str, Any]],
) -> int | None:
    for index, (astra, lru) in enumerate(zip(astra_rows, lru_rows)):
        if as_str(astra.get("backend_key_identity")) != as_str(lru.get("backend_key_identity")):
            return index
    if len(astra_rows) != len(lru_rows):
        return min(len(astra_rows), len(lru_rows))
    return None


def _request_time_ns(row: dict[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return int(float(value) * 1_000_000_000)
    except (TypeError, ValueError):
        return None


def _phase_for_request(
    astra: dict[str, Any],
    lru: dict[str, Any],
    astra_divergence_ns: int | None,
    lru_divergence_ns: int | None,
) -> str:
    if astra_divergence_ns is None or lru_divergence_ns is None:
        return "no_divergence_evidence"
    astra_start = _request_time_ns(astra, "request_started_s")
    astra_end = _request_time_ns(astra, "request_ended_s")
    lru_start = _request_time_ns(lru, "request_started_s")
    lru_end = _request_time_ns(lru, "request_ended_s")
    if None in {astra_start, astra_end, lru_start, lru_end}:
        return "missing_request_timestamps"
    if astra_end <= astra_divergence_ns and lru_end <= lru_divergence_ns:
        return "pre_divergence"
    if astra_start >= astra_divergence_ns and lru_start >= lru_divergence_ns:
        return "post_divergence"
    return "spans_or_mixed_divergence"


def _native_load_by_request(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = artifact_path(run_dir, "kv_core_native_receipts.jsonl")
    if path is None:
        return {}, {
            "artifact_present": False,
            "completed_receipts": None,
            "matched_request_ids": None,
            "bytes_loaded": None,
            "load_latency_ms": None,
        }
    rows = [
        row for row in load_jsonl(path)
        if as_str(row.get("status")) == "completed"
    ]
    by_request: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = as_str(
            row.get("logical_request_id")
            or row.get("request_id")
            or row.get("target_request_id")
        )
        if not request_id:
            continue
        item = by_request.setdefault(request_id, {
            "receipt_count": 0,
            "bytes_loaded": 0,
            "load_latency_ms": 0.0,
        })
        item["receipt_count"] += 1
        item["bytes_loaded"] += as_int(row.get("bytes_loaded"))
        item["load_latency_ms"] += as_int(row.get("load_latency_ns")) / 1_000_000.0
    for item in by_request.values():
        item["load_latency_ms"] = round(item["load_latency_ms"], 4)
    return by_request, {
        "artifact_present": True,
        "completed_receipts": len(rows),
        "matched_request_ids": len(by_request),
        "bytes_loaded": sum(as_int(row.get("bytes_loaded")) for row in rows),
        "load_latency_ms": round(
            sum(as_int(row.get("load_latency_ns")) for row in rows) / 1_000_000.0,
            4,
        ),
    }


def _selector_overhead(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        as_int(row.get("selection_duration_ns")) / 1_000_000.0
        for row in rows if row.get("selection_duration_ns") is not None
    ]
    scoring = [
        as_int(row.get("policy_scoring_duration_ns")) / 1_000_000.0
        for row in rows if row.get("policy_scoring_duration_ns") is not None
    ]
    return {
        "instrumented": bool(durations),
        "selection_count": len(rows),
        "instrumented_count": len(durations),
        "selection_mean_ms": round(statistics.mean(durations), 4) if durations else None,
        "selection_p95_ms": _percentile(durations, 0.95),
        "selection_total_ms": round(sum(durations), 4) if durations else None,
        "policy_scoring_total_ms": round(sum(scoring), 4) if scoring else None,
        "candidate_scan_total": (
            sum(as_int(row.get("candidate_scan_count")) for row in rows)
            if durations else None
        ),
        "fallback_candidate_total": (
            sum(as_int(row.get("fallback_candidate_count")) for row in rows)
            if durations else None
        ),
    }


def _future_reaccesses(run_dir: Path, selected: list[dict[str, Any]]) -> dict[str, Any]:
    events = [
        row for row in load_jsonl(artifact_path(run_dir, "runtime_events_raw.jsonl"))
        if as_str(row.get("action")) in {"cache_hit", "cache_load"}
        and as_str(row.get("status")) in {"completed", "available", "ok", "executed"}
    ]
    result: dict[str, Any] = {}
    for row in selected:
        signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
        logical_key = as_str(signals.get("logical_object_key"))
        if not logical_key or logical_key in result:
            continue
        selected_ns = as_int(row.get("timestamp_ns"))
        future = [
            event for event in events
            if as_str(event.get("object_key")) == logical_key
            and as_int(event.get("timestamp_ns")) > selected_ns
            and (
                not as_str(signals.get("request_id"))
                or as_str(event.get("request_id") or event.get("logical_request_id"))
                != as_str(signals.get("request_id"))
            )
        ]
        first = min(future, key=lambda event: as_int(event.get("timestamp_ns")), default=None)
        result[logical_key] = {
            "future_reaccessed": first is not None,
            "first_reaccess_delay_ms": (
                round((as_int(first.get("timestamp_ns")) - selected_ns) / 1_000_000.0, 4)
                if first is not None else None
            ),
            "first_reaccess_request_id": (
                as_str(first.get("request_id") or first.get("logical_request_id"))
                if first is not None else ""
            ),
            "same_request_reaccess_ignored": sum(
                as_str(event.get("object_key")) == logical_key
                and as_int(event.get("timestamp_ns")) > selected_ns
                and bool(as_str(signals.get("request_id")))
                and as_str(event.get("request_id") or event.get("logical_request_id"))
                == as_str(signals.get("request_id"))
                for event in events
            ),
        }
    return result


def _summarize_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [as_float(row.get("delta_ms")) for row in rows]
    return {
        "count": len(rows),
        "lru_mean_ms": round(statistics.mean(as_float(row["lru_ttft_ms"]) for row in rows), 4)
        if rows else None,
        "astrakv_mean_ms": round(statistics.mean(as_float(row["astrakv_ttft_ms"]) for row in rows), 4)
        if rows else None,
        "delta_mean_ms": round(statistics.mean(deltas), 4) if rows else None,
        "delta_total_ms": round(sum(deltas), 4) if rows else None,
    }


def analyze(root: Path, role: str = "baseline") -> dict[str, Any]:
    dirs = {arm: selected_run_dirs(root / arm, role) for arm in ARMS}
    cells: dict[str, Any] = {}
    evidence_gaps: set[str] = set()
    for cell in sorted(set(dirs["arm-evict-b"]) | set(dirs["arm-lru"])):
        astra_dir = dirs["arm-evict-b"].get(cell)
        lru_dir = dirs["arm-lru"].get(cell)
        if astra_dir is None or lru_dir is None:
            evidence_gaps.add(f"missing_arm:{cell}")
            continue
        astra_requests, _ = request_index(astra_dir)
        lru_requests, _ = request_index(lru_dir)
        astra_selected = _selected_rows(astra_dir)
        lru_selected = _selected_rows(lru_dir)
        divergence_index = _first_divergence(astra_selected, lru_selected)
        astra_divergence_ns = (
            as_int(astra_selected[divergence_index].get("timestamp_ns"))
            if divergence_index is not None and divergence_index < len(astra_selected) else None
        )
        lru_divergence_ns = (
            as_int(lru_selected[divergence_index].get("timestamp_ns"))
            if divergence_index is not None and divergence_index < len(lru_selected) else None
        )
        astra_load, astra_load_summary = _native_load_by_request(astra_dir)
        lru_load, lru_load_summary = _native_load_by_request(lru_dir)
        if not astra_load_summary["artifact_present"] or not lru_load_summary["artifact_present"]:
            evidence_gaps.add(f"native_load_receipts_missing:{cell}")
        astra_overhead = _selector_overhead(astra_selected)
        lru_overhead = _selector_overhead(lru_selected)
        if not astra_overhead["instrumented"] or not lru_overhead["instrumented"]:
            evidence_gaps.add(f"selector_timing_missing:{cell}")

        request_rows: list[dict[str, Any]] = []
        for key in sorted(set(astra_requests) & set(lru_requests), key=lambda item: (
            as_int(astra_requests[item].get("arrival_index")), item
        )):
            astra = astra_requests[key]
            lru = lru_requests[key]
            astra_ttft = as_float(astra.get("ttft_ms"))
            lru_ttft = as_float(lru.get("ttft_ms"))
            request_id = as_str(astra.get("request_id"))
            phase = _phase_for_request(
                astra, lru, astra_divergence_ns, lru_divergence_ns
            )
            if phase == "missing_request_timestamps":
                evidence_gaps.add(f"request_timestamps_missing:{cell}")
            request_rows.append({
                "pair_key": key,
                "request_id": request_id,
                "arrival_index": as_int(astra.get("arrival_index")),
                "reuse_bucket": as_str(astra.get("reuse_bucket")),
                "reuse_ratio": as_float(astra.get("reuse_ratio")),
                "phase": phase,
                "lru_ttft_ms": round(lru_ttft, 4),
                "astrakv_ttft_ms": round(astra_ttft, 4),
                "delta_ms": round(astra_ttft - lru_ttft, 4),
                "delta_percent": round((astra_ttft / lru_ttft - 1.0) * 100.0, 4)
                if lru_ttft > 0 else None,
                "native_load": {
                    "lru": lru_load.get(as_str(lru.get("request_id"))),
                    "astrakv": astra_load.get(request_id),
                },
            })
        positive_total = sum(max(0.0, as_float(row["delta_ms"])) for row in request_rows)
        for row in request_rows:
            row["positive_regression_share"] = round(
                max(0.0, as_float(row["delta_ms"])) / positive_total,
                6,
            ) if positive_total else 0.0

        by_phase = {
            phase: _summarize_phase([row for row in request_rows if row["phase"] == phase])
            for phase in sorted({as_str(row["phase"]) for row in request_rows})
        }
        cells[cell] = {
            "first_divergence_index_zero_based": divergence_index,
            "first_divergence_ordinal": divergence_index + 1 if divergence_index is not None else None,
            "divergence_timestamp_ns": {
                "arm-evict-b": astra_divergence_ns,
                "arm-lru": lru_divergence_ns,
            },
            "requests": request_rows,
            "phase_summary": by_phase,
            "native_load": {
                "arm-evict-b": astra_load_summary,
                "arm-lru": lru_load_summary,
            },
            "selector_overhead": {
                "arm-evict-b": astra_overhead,
                "arm-lru": lru_overhead,
            },
            "victim_future_reaccess": {
                "arm-evict-b": _future_reaccesses(astra_dir, astra_selected),
                "arm-lru": _future_reaccesses(lru_dir, lru_selected),
            },
        }

    all_requests = [row for cell in cells.values() for row in cell["requests"]]
    top_regressions = sorted(all_requests, key=lambda row: as_float(row["delta_ms"]), reverse=True)[:5]
    return {
        "schema": "astrakv-e11-request-attribution-v2",
        "root": str(root),
        "role": role,
        "cells": cells,
        "top_ttft_regressions": top_regressions,
        "evidence_gaps": sorted(evidence_gaps),
        "interpretation": {
            "performance_causality_established": False,
            "reason": (
                "This report localizes timing and I/O evidence. A single fixed-order repeat cannot "
                "separate victim-policy effects from server cold-state or arm-order effects."
            ),
            "recommended_next_step": (
                "Run at most one reverse-order repeat only if selector timing and native-load evidence "
                "are required for the final claim; otherwise retain valid_no_improvement."
            ),
        },
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# E11 request-level 归因报告",
        "",
        f"- 结果目录：`{result['root']}`",
        f"- 证据缺口：`{result['evidence_gaps'] or '无'}`",
        "- 因果边界：单次固定顺序 A/B 只能定位差异，不能排除 server 冷态与 arm 顺序混杂。",
        "",
    ]
    for cell_name, cell in result["cells"].items():
        lines += [
            f"## {cell_name}",
            "",
            f"- victim 首次分叉：第 {cell['first_divergence_ordinal']} 次选择（1-based）",
            f"- native load：`{cell['native_load']}`",
            f"- selector overhead：`{cell['selector_overhead']}`",
            "",
            "| idx | request | reuse | phase | LRU TTFT | AstraKV-W TTFT | delta | 正向劣化占比 |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in cell["requests"]:
            lines.append(
                f"| {row['arrival_index']} | {row['request_id']} | {row['reuse_bucket']} | "
                f"{row['phase']} | {row['lru_ttft_ms']:.2f} ms | "
                f"{row['astrakv_ttft_ms']:.2f} ms | {row['delta_ms']:+.2f} ms | "
                f"{row['positive_regression_share'] * 100:.2f}% |"
            )
        lines += [
            "",
            f"- 分阶段汇总：`{cell['phase_summary']}`",
            f"- victim 后续重访问：`{cell['victim_future_reaccess']}`",
            "",
        ]
    lines += [
        "## 结论边界",
        "",
        f"- `{result['interpretation']['reason']}`",
        f"- 建议：`{result['interpretation']['recommended_next_step']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline",), default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.role)
    output = args.output or args.root / "e11_request_attribution.md"
    json_output = args.json_output or args.root / "e11_request_attribution.json"
    output.write_text(render_report(result), encoding="utf-8")
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(output),
        "json": str(json_output),
        "evidence_gaps": result["evidence_gaps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
