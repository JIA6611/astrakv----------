#!/usr/bin/env python3
"""Build an independent fixed-SLO KV block capacity claim from real runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-kv-core-capacity-sweep-v1"


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
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    return values[max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))]


def throughput(rows: list[dict[str, Any]]) -> float | None:
    starts: list[float] = []
    ends: list[float] = []
    tokens = 0
    for row in rows:
        try:
            start = float(row["request_started_s"])
            end = float(row["request_ended_s"])
            observed = int(row["output_tokens_observed"])
        except (KeyError, TypeError, ValueError):
            return None
        if end <= start or observed < 0:
            return None
        starts.append(start)
        ends.append(end)
        tokens += observed
    makespan = max(ends) - min(starts) if starts else 0.0
    return tokens / makespan if makespan > 0 else None


def quality_fingerprint(rows: list[dict[str, Any]]) -> dict[str, tuple[Any, Any, Any]]:
    result: dict[str, tuple[Any, Any, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            return {}
        result[sample_id] = (
            row.get("output_token_ids"),
            row.get("finish_reason"),
            row.get("deterministic_logprob"),
        )
    return result


def inspect_run(path: Path, reference_quality: dict[str, tuple[Any, Any, Any]], args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(path / "experiment_manifest.json")
    runtime = read_json(path / "kv_core_run_metadata.json")
    rows = read_jsonl(path / "request_results.jsonl")
    try:
        budget = int(runtime["vllm_kv_block_budget"])
    except (KeyError, TypeError, ValueError):
        budget = 0
    ttft = percentile([
        float(row["ttft_ms"]) for row in rows
        if isinstance(row.get("ttft_ms"), (int, float))
    ], 0.95)
    observed_throughput = throughput(rows)
    quality = quality_fingerprint(rows)
    reasons: list[str] = []
    if not rows or any(row.get("status") != "ok" for row in rows):
        reasons.append("request_failure")
    if budget <= 0:
        reasons.append("kv_block_budget_missing")
    if ttft is None or ttft > args.slo_ttft_p95_ms:
        reasons.append("ttft_slo_failed")
    if observed_throughput is None or observed_throughput < args.slo_throughput_tokens_s:
        reasons.append("throughput_slo_failed")
    if not quality or quality != reference_quality:
        reasons.append("quality_mismatch")
    return {
        "path": str(path),
        "accepted": not reasons,
        "reasons": reasons,
        "vllm_kv_block_budget": budget,
        "ttft_p95_ms": ttft,
        "throughput_tokens_s": observed_throughput,
        "manifest": manifest,
        "request_order": [str(row.get("sample_id") or "") for row in rows],
        "output_length": [int(row.get("output_tokens_target") or 0) for row in rows],
    }


def control_fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
    manifest = record.get("manifest") or {}
    return (
        manifest.get("model"), manifest.get("model_revision"),
        manifest.get("tokenizer_revision"), manifest.get("dtype"),
        manifest.get("quantization"), manifest.get("workload_sha256"),
        manifest.get("random_seed"), manifest.get("cache_state"),
        manifest.get("matrix_sha256"),
        json.dumps(manifest.get("software") or {}, sort_keys=True),
        tuple(record.get("request_order") or ()),
        tuple(record.get("output_length") or ()),
    )


def path_budget(path: Path) -> int:
    try:
        return max(0, int(read_json(path / "kv_core_run_metadata.json").get("vllm_kv_block_budget") or 0))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--variant-run", action="append", required=True)
    parser.add_argument("--phase", required=True, choices=("E2", "E4"))
    parser.add_argument("--slo-ttft-p95-ms", type=float, required=True)
    parser.add_argument("--slo-throughput-tokens-s", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.slo_ttft_p95_ms <= 0 or args.slo_throughput_tokens_s <= 0:
        raise ValueError("capacity SLO thresholds must be positive")
    baseline_paths = [Path(value) for value in args.baseline_run]
    variant_paths = [Path(value) for value in args.variant_run]
    baseline_paths.sort(key=path_budget, reverse=True)
    variant_paths.sort(key=path_budget, reverse=True)
    reference_rows = read_jsonl(baseline_paths[0] / "request_results.jsonl")
    reference_quality = quality_fingerprint(reference_rows)
    baseline = [inspect_run(path, reference_quality, args) for path in baseline_paths]
    variant = [inspect_run(path, reference_quality, args) for path in variant_paths]
    errors: list[str] = []
    all_records = baseline + variant
    controls = {control_fingerprint(record) for record in all_records}
    if len(controls) != 1:
        errors.append("capacity_sweep_control_mismatch")
    if len({record["vllm_kv_block_budget"] for record in baseline}) < 2:
        errors.append("baseline_capacity_sweep_too_small")
    if len({record["vllm_kv_block_budget"] for record in variant}) < 2:
        errors.append("variant_capacity_sweep_too_small")
    accepted_baseline = [record["vllm_kv_block_budget"] for record in baseline if record["accepted"]]
    accepted_variant = [record["vllm_kv_block_budget"] for record in variant if record["accepted"]]
    baseline_budget = min(accepted_baseline) if accepted_baseline else None
    variant_budget = min(accepted_variant) if accepted_variant else None
    if baseline_budget is None or variant_budget is None:
        errors.append("accepted_capacity_endpoint_missing")
    elif variant_budget > baseline_budget * 0.90:
        errors.append("kv_capacity_saving_not_proven")
    manifest = (baseline[0].get("manifest") or {}) if baseline else {}
    first = baseline[0] if baseline else {}
    record = {
        "schema": SCHEMA,
        "phase": args.phase,
        "eligible": not errors,
        "errors": errors,
        "accepted_vllm_kv_block_budget": {
            "baseline": baseline_budget,
            "variant": variant_budget,
        },
        "controls": {
            "model": manifest.get("model"),
            "dtype": manifest.get("dtype"),
            "workload_sha256": manifest.get("workload_sha256"),
            "request_order": first.get("request_order"),
            "seed": manifest.get("random_seed"),
            "sampling": manifest.get("matrix_sha256"),
            "output_length": first.get("output_length"),
            "cache_state": manifest.get("cache_state"),
            "slo": {
                "ttft_p95_ms": args.slo_ttft_p95_ms,
                "throughput_tokens_s": args.slo_throughput_tokens_s,
            },
        },
        "runs": {"baseline": baseline, "variant": variant},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
