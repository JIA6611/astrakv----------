#!/usr/bin/env python3
"""Validate E3-P evidence without promoting it to a correctness claim.

E3-P compares two fresh services with the same ``gpu_cpu_ssd`` topology and
the same request-owned native LMCache load path.  The sole intended runtime
difference is whether a target request gets an authenticated, cancellable
SSD-to-LocalCPUBackend promotion during its exact pre-dispatch lead window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "astrakv-e3-prefetch-performance-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = validate(Path(args.baseline), Path(args.variant))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["measurement_valid"] else 2


def validate(baseline: Path, variant: Path) -> dict[str, Any]:
    errors: list[str] = []
    baseline_control = read_json(baseline / "e3_prefetch_control.json")
    variant_control = read_json(variant / "e3_prefetch_control.json")
    validate_controls(baseline_control, variant_control, errors)
    baseline_workload = baseline / "workload_source.jsonl"
    variant_workload = variant / "workload_source.jsonl"
    if not baseline_workload.is_file() or not variant_workload.is_file():
        errors.append("workload_source_missing")
    elif sha256_file(baseline_workload) != sha256_file(variant_workload):
        errors.append("workload_sha256_mismatch")

    baseline_requests = read_jsonl(baseline / "request_results.jsonl")
    variant_requests = read_jsonl(variant / "request_results.jsonl")
    revisits = validate_request_pairs(baseline_requests, variant_requests, errors)
    validate_associations(baseline_requests, "baseline", errors)
    validate_associations(variant_requests, "variant", errors)

    baseline_tickets = read_jsonl(baseline / "kv_core_prefetch_tickets.jsonl")
    variant_tickets = read_jsonl(variant / "kv_core_prefetch_tickets.jsonl")
    validate_ticket_evidence(baseline_tickets, variant_tickets, revisits, errors)
    validate_native_accounting(
        read_jsonl(baseline / "kv_core_request_accounting.jsonl"),
        revisits, "baseline", errors,
    )
    validate_native_accounting(
        read_jsonl(variant / "kv_core_request_accounting.jsonl"),
        revisits, "variant", errors,
    )
    validate_cost_observations(
        read_jsonl(baseline / "kv_core_cost_observations.jsonl"), "baseline", errors,
    )
    validate_cost_observations(
        read_jsonl(variant / "kv_core_cost_observations.jsonl"), "variant", errors,
    )
    ttft = paired_ttft(baseline_requests, variant_requests, revisits)
    if ttft["count"] == 0:
        errors.append("revisit_ttft_missing")
    return {
        "schema": SCHEMA,
        "status": "exploratory_performance_only",
        "measurement_valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "correctness_accepted": False,
        "eligible_for_e4": False,
        "eligible_for_capacity_claim": False,
        "comparison": "gpu_cpu_ssd_active_admission_prefetch_off_vs_ssd_to_cpu_prefetch_on",
        "revisit_count": ttft["count"],
        "ttft_ms": ttft,
        "controls": {
            "baseline": baseline_control,
            "variant": variant_control,
        },
        "required_next_gate": "teacher_forced_profile_and_strict_e3_correctness",
    }


def validate_controls(baseline: dict[str, Any], variant: dict[str, Any], errors: list[str]) -> None:
    required = {
        "mode": "active",
        "topology": "gpu_cpu_ssd",
        "local_cpu_enabled": True,
        "local_disk_enabled": True,
        "admission_enabled": True,
        "prefill_online_calibration_enabled": True,
        "invalidate_disk_backed_cpu_on_prefetch_lead": True,
    }
    for label, control, expected_prefetch in (
        ("baseline", baseline, False),
        ("variant", variant, True),
    ):
        for field, expected in required.items():
            if control.get(field) != expected:
                errors.append(f"{label}_control_{field}_mismatch")
        if control.get("cpu_prefetch_enabled") is not expected_prefetch:
            errors.append(f"{label}_prefetch_switch_mismatch")
    shared = set(required) | {"model", "dtype", "output_tokens", "prefetch_lead_s"}
    for field in shared:
        if baseline.get(field) != variant.get(field):
            errors.append(f"control_variable_mismatch:{field}")


def validate_request_pairs(
    baseline: list[dict[str, Any]], variant: list[dict[str, Any]], errors: list[str],
) -> tuple[str, ...]:
    by_id = {str(row.get("request_id") or ""): row for row in baseline}
    variant_by_id = {str(row.get("request_id") or ""): row for row in variant}
    revisit_ids = tuple(sorted(
        request_id for request_id in by_id
        if "-revisit-" in request_id
    ))
    if not revisit_ids or set(revisit_ids) != {
        request_id for request_id in variant_by_id if "-revisit-" in request_id
    }:
        errors.append("revisit_request_identity_mismatch")
        return revisit_ids
    for request_id in revisit_ids:
        left, right = by_id[request_id], variant_by_id[request_id]
        for field in ("prompt_hash", "context_length", "generation_seed", "prefetch_lead_s"):
            if left.get(field) != right.get(field):
                errors.append(f"revisit_control_mismatch:{field}:{request_id}")
        for label, row in (("baseline", left), ("variant", right)):
            if row.get("status") != "ok":
                errors.append(f"{label}_request_failed:{request_id}")
            if not isinstance(row.get("ttft_ms"), (int, float)) or float(row["ttft_ms"]) <= 0.0:
                errors.append(f"{label}_ttft_missing:{request_id}")
    return revisit_ids


def validate_associations(rows: list[dict[str, Any]], label: str, errors: list[str]) -> None:
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id and "-revisit-" in request_id and row.get("runtime_association_status") != "linked":
            errors.append(f"{label}_runtime_association_missing:{request_id}")


def validate_ticket_evidence(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    revisit_ids: Iterable[str],
    errors: list[str],
) -> None:
    if any(str(row.get("source_tier")) == "ssd" and str(row.get("target_tier")) == "cpu" for row in baseline):
        errors.append("baseline_emitted_prefetch_ticket")
    consumed = [
        row for row in variant
        if str(row.get("source_tier")) == "ssd"
        and str(row.get("target_tier")) == "cpu"
        and str(row.get("status")) == "consumed"
        and int(row.get("completed_bytes") or 0) > 0
    ]
    if not consumed:
        errors.append("variant_prefetch_consumption_missing")
        return
    consumed_by_target = {
        str(row.get("target_request_id") or ""): row
        for row in consumed
    }
    for request_id in revisit_ids:
        if request_id not in consumed_by_target:
            errors.append(f"variant_prefetch_not_consumed:{request_id}")
    for row in consumed:
        if row.get("consumer_request_id") != row.get("target_request_id"):
            errors.append("prefetch_consumer_request_mismatch")
        if not all(str(row.get(field) or "") for field in (
            "physical_object_id", "binding_generation", "prefix_hash", "native_key", "compatibility_identity",
        )):
            errors.append("prefetch_identity_missing")


def validate_native_accounting(
    rows: list[dict[str, Any]], revisit_ids: Iterable[str], label: str, errors: list[str],
) -> None:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("logical_request_id") or row.get("request_id") or "")
        if request_id:
            latest[request_id] = row
    for request_id in revisit_ids:
        row = latest.get(request_id)
        if row is None:
            errors.append(f"{label}_native_accounting_missing:{request_id}")
            continue
        try:
            lookup = int(row["lookup_hit_tokens"])
            allocated = int(row["allocated_external_tokens"])
            loaded = int(row["actual_loaded_tokens"])
            recomputed = int(row["recomputed_tokens"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}_native_accounting_invalid:{request_id}")
            continue
        if not (lookup >= allocated >= loaded >= 0) or recomputed < 0:
            errors.append(f"{label}_native_accounting_invariant_failed:{request_id}")
        if loaded <= 0:
            errors.append(f"{label}_native_load_missing:{request_id}")


def validate_cost_observations(
    rows: list[dict[str, Any]], label: str, errors: list[str],
) -> None:
    accepted: list[dict[str, Any]] = []
    for row in rows:
        try:
            valid = (
                row.get("accepted") is True
                and str(row.get("source") or "") == "scheduler_compute_progress"
                and int(row.get("prefill_tokens") or 0) > 0
                and float(row.get("sample_ms_per_token") or 0.0) > 0.0
                and float(row.get("observed_prefill_ms_per_token") or 0.0) > 0.0
            )
        except (TypeError, ValueError):
            valid = False
        if valid:
            accepted.append(row)
    if not accepted:
        errors.append(f"{label}_online_prefill_cost_missing")


def paired_ttft(
    baseline: list[dict[str, Any]], variant: list[dict[str, Any]], revisit_ids: Iterable[str],
) -> dict[str, Any]:
    by_id = {str(row.get("request_id") or ""): row for row in baseline}
    variant_by_id = {str(row.get("request_id") or ""): row for row in variant}
    pairs = [
        (float(by_id[request_id]["ttft_ms"]), float(variant_by_id[request_id]["ttft_ms"]))
        for request_id in revisit_ids
        if request_id in by_id and request_id in variant_by_id
        and isinstance(by_id[request_id].get("ttft_ms"), (int, float))
        and isinstance(variant_by_id[request_id].get("ttft_ms"), (int, float))
        and float(by_id[request_id]["ttft_ms"]) > 0.0
    ]
    if not pairs:
        return {"count": 0, "baseline_p95": None, "variant_p95": None, "p95_delta_percent": None, "bootstrap_ci95_percent": [None, None]}
    baseline_values, variant_values = zip(*pairs)
    baseline_p95, variant_p95 = percentile(baseline_values, 95.0), percentile(variant_values, 95.0)
    point = 100.0 * (variant_p95 - baseline_p95) / baseline_p95
    generator = random.Random(0)
    bootstrap = []
    for _ in range(1000):
        sampled = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        base = percentile([pair[0] for pair in sampled], 95.0)
        var = percentile([pair[1] for pair in sampled], 95.0)
        bootstrap.append(100.0 * (var - base) / base)
    bootstrap.sort()
    return {
        "count": len(pairs),
        "baseline_p95": baseline_p95,
        "variant_p95": variant_p95,
        "p95_delta_percent": point,
        "bootstrap_ci95_percent": [bootstrap[24], bootstrap[974]],
    }


def percentile(values: Iterable[float], percent: float) -> float:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        raise ValueError("percentile requires values")
    index = (len(data) - 1) * percent / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return data[lower]
    return data[lower] + (data[upper] - data[lower]) * (index - lower)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            row for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for row in [json.loads(line)] if isinstance(row, dict)
        ]
    except (OSError, json.JSONDecodeError):
        return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
