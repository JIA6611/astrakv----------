#!/usr/bin/env python3
"""Aggregate independent validated KV-Core repetitions without pseudo-replication.

Each acceptance file is one independent service startup and a paired
E0/E2, E2/E3, E3/E4, or E5C/E5 comparison.  Request rows within a repeat share the same
server and cache-state realization, so this tool resamples repeat clusters
first and paired requests only within each selected cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "astrakv-kv-core-repeat-aggregate-v1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def paired_ttft_rows(baseline: list[dict[str, Any]], variant: list[dict[str, Any]]) -> list[tuple[float, float]]:
    baseline_by_id = {str(row.get("sample_id") or ""): row for row in baseline}
    variant_by_id = {str(row.get("sample_id") or ""): row for row in variant}
    pairs: list[tuple[float, float]] = []
    for sample_id in sorted(set(baseline_by_id) & set(variant_by_id)):
        if not sample_id:
            continue
        try:
            left = float(baseline_by_id[sample_id]["ttft_ms"])
            right = float(variant_by_id[sample_id]["ttft_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if left > 0 and right >= 0:
            pairs.append((left, right))
    return pairs


def p95_delta_percent(pairs: list[tuple[float, float]]) -> float | None:
    baseline = percentile((left for left, _ in pairs), 0.95)
    variant = percentile((right for _, right in pairs), 0.95)
    if baseline is None or variant is None or baseline <= 0:
        return None
    return (variant - baseline) / baseline * 100.0


def hierarchical_ttft_bootstrap(
    repeats: list[list[tuple[float, float]]], *, samples: int = 10_000, seed: int = 0,
) -> tuple[float | None, tuple[float | None, float | None]]:
    """Bootstrap p95 deltas with service-startup repeats as the cluster unit."""
    per_repeat = [p95_delta_percent(pairs) for pairs in repeats]
    if not repeats or any(value is None for value in per_repeat):
        return None, (None, None)
    point = statistics.mean(value for value in per_repeat if value is not None)
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sample_deltas: list[float] = []
        for _ in repeats:
            cluster = repeats[generator.randrange(len(repeats))]
            within = [cluster[generator.randrange(len(cluster))] for _ in cluster]
            delta = p95_delta_percent(within)
            if delta is not None:
                sample_deltas.append(delta)
        if len(sample_deltas) == len(repeats):
            values.append(statistics.mean(sample_deltas))
    return point, (percentile(values, 0.025), percentile(values, 0.975))


def _throughput_delta_percent(record: dict[str, Any]) -> float | None:
    values = record.get("throughput_tokens_s")
    if not isinstance(values, dict):
        return None
    try:
        baseline = float(values["baseline"])
        variant = float(values["variant"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if baseline <= 0 else (variant - baseline) / baseline * 100.0


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def aggregate_acceptance_paths(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two independent repeat acceptance files are required")
    records = [read_json(path) for path in paths]
    errors: list[str] = []
    phase = ""
    workload_hash = ""
    pairs: list[list[tuple[float, float]]] = []
    per_repeat: list[dict[str, Any]] = []
    for path, record in zip(paths, records, strict=True):
        repeat_errors: list[str] = []
        if record.get("schema") != "astrakv-kv-core-acceptance-v2":
            repeat_errors.append("acceptance_schema_mismatch")
        if record.get("eligible") is not True or record.get("errors"):
            repeat_errors.append("repeat_not_eligible")
        record_phase = str(record.get("phase") or "")
        if not phase:
            phase = record_phase
        elif record_phase != phase:
            repeat_errors.append("phase_mismatch")
        paired = record.get("paired_manifest") if isinstance(record.get("paired_manifest"), dict) else {}
        artifact_hashes = paired.get("artifact_hashes") if isinstance(paired.get("artifact_hashes"), dict) else {}
        baseline_hashes = artifact_hashes.get("baseline") if isinstance(artifact_hashes.get("baseline"), dict) else {}
        variant_hashes = artifact_hashes.get("variant") if isinstance(artifact_hashes.get("variant"), dict) else {}
        current_workload_hash = str(baseline_hashes.get("workload") or "")
        if not current_workload_hash or current_workload_hash != str(variant_hashes.get("workload") or ""):
            repeat_errors.append("paired_workload_hash_mismatch")
        if not workload_hash:
            workload_hash = current_workload_hash
        elif current_workload_hash != workload_hash:
            repeat_errors.append("repeat_workload_hash_mismatch")
        runs = paired.get("runs") if isinstance(paired.get("runs"), dict) else {}
        baseline_path = Path(str((runs.get("baseline") or {}).get("path") or ""))
        variant_path = Path(str((runs.get("variant") or {}).get("path") or ""))
        current_pairs = paired_ttft_rows(
            read_jsonl(baseline_path / "request_results.jsonl"),
            read_jsonl(variant_path / "request_results.jsonl"),
        )
        if not current_pairs:
            repeat_errors.append("paired_ttft_rows_missing")
        delta = p95_delta_percent(current_pairs)
        if delta is None:
            repeat_errors.append("repeat_ttft_p95_missing")
        if repeat_errors:
            errors.extend(f"{path}:{error}" for error in repeat_errors)
        pairs.append(current_pairs)
        per_repeat.append({
            "acceptance_path": str(path),
            "request_pair_count": len(current_pairs),
            "ttft_p95_delta_percent": delta,
            "throughput_delta_percent": _throughput_delta_percent(record),
            "request_accounting_count": record.get("request_accounting_count"),
            "uma_measurement": record.get("uma_measurement"),
        })
    point, interval = hierarchical_ttft_bootstrap(pairs) if not errors else (None, (None, None))
    ttft_values = [float(item["ttft_p95_delta_percent"]) for item in per_repeat if item["ttft_p95_delta_percent"] is not None]
    throughput_values = [float(item["throughput_delta_percent"]) for item in per_repeat if item["throughput_delta_percent"] is not None]
    directions = {"improvement" if value < 0 else "regression" if value > 0 else "neutral" for value in ttft_values}
    stable_improvement = bool(
        point is not None and interval[1] is not None and interval[1] < 0.0 and directions == {"improvement"}
    )
    cgroup_valid = all(
        isinstance(item["uma_measurement"], dict)
        and item["uma_measurement"].get("baseline") == "cgroup_valid"
        and item["uma_measurement"].get("variant") == "cgroup_valid"
        for item in per_repeat
    )
    return {
        "schema": SCHEMA,
        "phase": phase,
        "eligible": not errors,
        "errors": errors,
        "repeat_count": len(paths),
        "workload_sha256": workload_hash,
        "analysis_unit": "independent_service_startup_repeat",
        "resampling": {
            "method": "hierarchical_bootstrap_repeat_then_paired_request",
            "seed": 0,
            "samples": 10_000,
        },
        "ttft_p95_delta_percent": {
            **_stats(ttft_values),
            "hierarchical_bootstrap_ci_percent": list(interval),
            "stable_improvement": stable_improvement,
            "direction_set": sorted(directions),
        },
        "throughput_delta_percent": _stats(throughput_values),
        "correctness": {
            "all_repeats_eligible": not errors,
            "request_accounting_count_total": sum(
                int(item["request_accounting_count"] or 0) for item in per_repeat
            ),
        },
        "uma_physical_memory_evidence": "cgroup_valid" if cgroup_valid else "not_available_for_claim",
        "performance_conclusion": (
            "stable_improvement" if stable_improvement else "inconclusive_no_performance_claim"
        ),
        "repeats": per_repeat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", action="append", required=True, help="One acceptance.json per independent repeat.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = aggregate_acceptance_paths([Path(value) for value in args.acceptance])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
