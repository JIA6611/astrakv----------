#!/usr/bin/env python3
"""Validate a single controlled AstraKV-W KV-Core comparison.

This is intentionally a gate, not a charting utility.  It refuses to call a
run eligible when request identity, native receipts, output equivalence, UMA
evidence, or the declared control variable is missing.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.paired_run import PairedRunInput, validate_paired_runs


SCHEMA = "astrakv-kv-core-acceptance-v1"
PHASES = {"E1", "E2", "E3", "E4"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def percentile(values: Iterable[float], q: float) -> float | None:
    sorted_values = sorted(values)
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def paired_ttft_bootstrap(baseline: list[dict[str, Any]], variant: list[dict[str, Any]], *, samples: int = 2000) -> tuple[float | None, tuple[float | None, float | None]]:
    base = {str(row.get("sample_id")): row for row in baseline}
    variant_by_id = {str(row.get("sample_id")): row for row in variant}
    ids = sorted(set(base) & set(variant_by_id))
    pairs: list[tuple[float, float]] = []
    for sample_id in ids:
        try:
            left = float(base[sample_id]["ttft_ms"])
            right = float(variant_by_id[sample_id]["ttft_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if left > 0 and right >= 0:
            pairs.append((left, right))
    if not pairs:
        return None, (None, None)
    def delta(items: list[tuple[float, float]]) -> float:
        left = percentile((pair[0] for pair in items), 0.95)
        right = percentile((pair[1] for pair in items), 0.95)
        assert left is not None and right is not None
        return (right - left) / left * 100.0
    point = delta(pairs)
    generator = random.Random(0)
    sampled = [delta([pairs[generator.randrange(len(pairs))] for _ in pairs]) for _ in range(samples)]
    return point, (percentile(sampled, 0.025), percentile(sampled, 0.975))


def validate_quality(baseline: list[dict[str, Any]], variant: list[dict[str, Any]], errors: list[str]) -> None:
    left = {str(row.get("sample_id")): row for row in baseline}
    right = {str(row.get("sample_id")): row for row in variant}
    if not left or set(left) != set(right):
        errors.append("quality_sample_coverage_mismatch")
        return
    for sample_id in sorted(left):
        a, b = left[sample_id], right[sample_id]
        for field in ("output_token_ids", "finish_reason", "deterministic_logprob"):
            if field not in a or field not in b:
                errors.append(f"quality_field_missing:{field}")
                continue
            if a[field] != b[field]:
                errors.append(f"quality_mismatch:{field}:{sample_id}")
        if not isinstance(a.get("output_token_ids"), list) or not a["output_token_ids"]:
            errors.append(f"quality_token_evidence_missing:{sample_id}")
        if not str(a.get("finish_reason") or ""):
            errors.append(f"quality_finish_reason_missing:{sample_id}")
        if not isinstance(a.get("deterministic_logprob"), (int, float)):
            errors.append(f"quality_logprob_missing:{sample_id}")


def validate_receipts(rows: list[dict[str, Any]], expected_request_ids: set[str], errors: list[str]) -> None:
    seen: set[str] = set()
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in seen:
            errors.append("receipt_request_identity_mismatch")
            continue
        seen.add(request_id)
        try:
            lookup = int(row["lookup_hit_tokens"])
            allocated = int(row["allocated_external_tokens"])
            loaded = int(row["actual_loaded_tokens"])
            requested = int(row["requested_prefix_tokens"])
            recomputed = int(row["recomputed_tokens"])
            generation = int(row["binding_generation"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_native_receipt:{request_id}")
            continue
        if not (lookup >= allocated >= loaded >= 0) or recomputed != max(0, requested - loaded) or generation <= 0:
            errors.append(f"receipt_invariant_failed:{request_id}")
    if seen != expected_request_ids:
        errors.append("receipt_request_coverage_mismatch")


def uma_peak(rows: list[dict[str, Any]]) -> int | None:
    values: list[int] = []
    for row in rows:
        for field in ("cgroup_memory_current_bytes", "process_rss_bytes"):
            try:
                values.append(int(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
    return max(values) if values else None


def aggregate_throughput(rows: list[dict[str, Any]]) -> float | None:
    total_tokens = 0.0
    total_seconds = 0.0
    for row in rows:
        try:
            tokens = float(row["output_tokens_observed"])
            latency_ms = float(row["latency_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if tokens >= 0 and latency_ms > 0:
            total_tokens += tokens
            total_seconds += latency_ms / 1000.0
    return total_tokens / total_seconds if total_seconds else None


def kv_block_budget(run: Path) -> int | None:
    try:
        payload = json.loads((run / "kv_core_run_metadata.json").read_text(encoding="utf-8"))
        value = int(payload["vllm_kv_block_budget"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def load_run_artifact(run: Path, filename: str) -> list[dict[str, Any]]:
    return read_jsonl(run / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_path, variant_path = Path(args.baseline), Path(args.variant)
    paired = validate_paired_runs(PairedRunInput("baseline", baseline_path), PairedRunInput("variant", variant_path))
    errors = list(paired.errors)
    baseline_requests = load_run_artifact(baseline_path, "request_results.jsonl")
    variant_requests = load_run_artifact(variant_path, "request_results.jsonl")
    validate_quality(baseline_requests, variant_requests, errors)
    request_ids = {str(row.get("request_id") or "") for row in variant_requests}
    if "" in request_ids:
        errors.append("request_identity_missing")
    if args.phase != "E1":
        receipts = load_run_artifact(variant_path, "kv_core_native_receipts.jsonl")
        validate_receipts(receipts, request_ids, errors)
    baseline_uma = load_run_artifact(baseline_path, "uma_resource_samples.jsonl")
    variant_uma = load_run_artifact(variant_path, "uma_resource_samples.jsonl")
    base_peak, variant_peak = uma_peak(baseline_uma), uma_peak(variant_uma)
    if base_peak is None or variant_peak is None:
        errors.append("uma_evidence_missing")
    elif variant_peak > base_peak * 1.02:
        errors.append("uma_peak_regression")
    point, interval = paired_ttft_bootstrap(baseline_requests, variant_requests)
    if point is None:
        errors.append("ttft_evidence_missing")
    if args.phase == "E3" and (point is None or point > -5.0 or interval[1] is None or interval[1] >= 0.0):
        errors.append("e3_ttft_acceptance_failed")
    baseline_throughput = aggregate_throughput(baseline_requests)
    variant_throughput = aggregate_throughput(variant_requests)
    if baseline_throughput is None or variant_throughput is None:
        errors.append("throughput_evidence_missing")
    elif variant_throughput < baseline_throughput * 0.98:
        errors.append("throughput_regression")
    no_reuse_baseline = [row for row in baseline_requests if str(row.get("workload_type")) == "random_no_reuse"]
    no_reuse_variant = [row for row in variant_requests if str(row.get("workload_type")) == "random_no_reuse"]
    if no_reuse_baseline or no_reuse_variant:
        no_reuse_point, _ = paired_ttft_bootstrap(no_reuse_baseline, no_reuse_variant)
        if no_reuse_point is None or no_reuse_point > 2.0:
            errors.append("no_reuse_ttft_regression")
    baseline_budget = kv_block_budget(baseline_path)
    variant_budget = kv_block_budget(variant_path)
    if args.phase in {"E2", "E4"}:
        if baseline_budget is None or variant_budget is None:
            errors.append("kv_block_budget_evidence_missing")
        elif variant_budget > baseline_budget * 0.90:
            errors.append("kv_capacity_saving_not_proven")
    if args.phase == "E4":
        # Upper-bound partial load must retain receipt accounting, enforced above.
        partial_rows = [row for row in load_run_artifact(variant_path, "kv_core_native_receipts.jsonl") if int(row.get("allocated_external_tokens") or 0) < int(row.get("lookup_hit_tokens") or 0)]
        if not partial_rows:
            errors.append("e4_partial_prefix_evidence_missing")
    record = {
        "schema": SCHEMA,
        "phase": args.phase,
        "eligible": not errors,
        "errors": list(dict.fromkeys(errors)),
        "paired_manifest": paired.record,
        "ttft_p95_delta_percent": point,
        "ttft_p95_bootstrap_ci_percent": list(interval),
        "uma_peak_bytes": {"baseline": base_peak, "variant": variant_peak},
        "throughput_tokens_s": {"baseline": baseline_throughput, "variant": variant_throughput},
        "vllm_kv_block_budget": {"baseline": baseline_budget, "variant": variant_budget},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
