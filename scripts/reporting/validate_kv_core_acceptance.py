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
from astrakv.runtime.third_party_patch import PATCH_ID, REQUIRED_CALLBACKS
from astrakv.runtime.kv_core_connector import native_key_prefix_ok


SCHEMA = "astrakv-kv-core-acceptance-v2"
PHASES = {"E1", "E2", "E3", "E3C", "E4", "E5", "E5C"}
PREFETCH_BENEFIT_WORKLOADS = {"repeated_long_prefix", "queued_concurrency"}
PARTIAL_LOAD_WORKLOADS = {"repeated_long_prefix", "constrained_kv_churn"}


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


def _association_index(
    rows: list[dict[str, Any]], expected_request_ids: set[str], errors: list[str],
    *, expected_run_id: str = "",
) -> dict[str, dict[str, str]]:
    """Build native->logical bindings from runtime-owned association receipts."""
    index: dict[str, dict[str, str]] = {}
    logical_ids: set[str] = set()
    for row in rows:
        if str(row.get("status") or "") != "associated":
            errors.append("association_receipt_not_associated")
            continue
        if expected_run_id and str(row.get("run_id") or "") != expected_run_id:
            errors.append("association_receipt_run_id_mismatch")
            continue
        native = str(row.get("runtime_request_id") or "")
        logical = str(row.get("request_id") or "")
        event_id = str(row.get("runtime_event_id") or "")
        nonce = str(row.get("request_nonce") or "")
        try:
            expires_at_ns = int(row.get("expires_at_ns") or 0)
        except (TypeError, ValueError):
            expires_at_ns = 0
        if (
            not native or not logical or not event_id or not nonce
            or not str(row.get("session_id") or "")
            or expires_at_ns <= 0
            or not str(row.get("mac") or "")
        ):
            errors.append("association_receipt_identity_missing")
            continue
        current = {
            "logical_request_id": logical,
            "runtime_event_id": event_id,
            "association_receipt_reference": event_id,
            "request_nonce": nonce,
        }
        prior = index.get(native)
        if prior is not None and prior != current:
            errors.append(f"association_receipt_conflict:{native}")
            continue
        index[native] = current
        logical_ids.add(logical)
    if expected_request_ids and not expected_request_ids.issubset(logical_ids):
        errors.append("association_request_coverage_mismatch")
    return index


def validate_variant_request_associations(
    request_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    errors: list[str],
    *,
    expected_run_id: str = "",
) -> dict[str, dict[str, str]]:
    """Require benchmark rows to match runtime-emitted ReqMeta receipts.

    The benchmark client verifies the HMAC before writing ``linked``. This
    validator deliberately does not infer identities from ``chatcmpl-*``
    strings: it only accepts the exact native ID/event ID emitted by the
    runtime-owned association artifact.
    """
    expected_by_logical: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        logical = str(row.get("request_id") or "")
        if not logical:
            errors.append("request_identity_missing")
            continue
        if logical in expected_by_logical:
            errors.append(f"request_result_identity_duplicate:{logical}")
            continue
        expected_by_logical[logical] = row
    expected_request_ids = set(expected_by_logical)
    if not association_rows:
        errors.append("request_association_artifact_missing")
        association_index: dict[str, dict[str, str]] = {}
    else:
        association_index = _association_index(
            association_rows,
            expected_request_ids,
            errors,
            expected_run_id=expected_run_id,
        )
        associated_logical = {
            binding["logical_request_id"] for binding in association_index.values()
        }
        if associated_logical != expected_request_ids:
            errors.append("association_request_exact_coverage_mismatch")
    for logical, row in expected_by_logical.items():
        if str(row.get("runtime_association_status") or "") != "linked":
            errors.append(f"request_result_association_not_linked:{logical}")
            continue
        native = str(row.get("runtime_request_id") or "")
        event_id = str(row.get("runtime_event_id") or "")
        if not native or not event_id:
            errors.append(f"request_result_association_identity_missing:{logical}")
            continue
        binding = association_index.get(native)
        if binding is None:
            errors.append(f"request_result_association_missing:{logical}")
            continue
        if binding["logical_request_id"] != logical:
            errors.append(f"request_result_logical_identity_mismatch:{logical}")
        if binding["runtime_event_id"] != event_id:
            errors.append(f"request_result_association_event_mismatch:{logical}")
    return association_index


def _normalize_native_identity(
    row: dict[str, Any], association_index: dict[str, dict[str, str]], errors: list[str],
    *, artifact: str,
) -> dict[str, Any] | None:
    """Attach audited native/logical identity without guessing from text."""
    if not association_index:
        return dict(row)
    native = str(row.get("native_request_id") or row.get("request_id") or "")
    binding = association_index.get(native)
    if binding is None:
        errors.append(f"{artifact}_association_missing:{native}")
        return None
    explicit_logical = str(row.get("logical_request_id") or "")
    if explicit_logical and explicit_logical != binding["logical_request_id"]:
        errors.append(f"{artifact}_logical_identity_mismatch:{native}")
        return None
    explicit_reference = str(row.get("association_receipt_reference") or "")
    if explicit_reference and explicit_reference != binding["association_receipt_reference"]:
        errors.append(f"{artifact}_association_reference_mismatch:{native}")
        return None
    normalized = dict(row)
    normalized.update(binding)
    normalized["native_request_id"] = native
    normalized["request_id"] = binding["logical_request_id"]
    return normalized


def validate_request_accounting(
    rows: list[dict[str, Any]], expected_request_ids: set[str], errors: list[str],
    *, association_index: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Validate the last native observation for every request.

    These records are not synthetic load receipts: a zero-allocation terminal
    record proves scheduler-declined recompute, while a positive allocation is
    terminal only after the native connector reports completion.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = _normalize_native_identity(row, association_index or {}, errors, artifact="accounting")
        if normalized is None:
            continue
        row = normalized
        request_id = str(row.get("request_id") or "")
        if not request_id:
            errors.append("accounting_request_identity_missing")
            continue
        latest[request_id] = row
    if set(latest) != expected_request_ids:
        errors.append("accounting_request_coverage_mismatch")
    for request_id, row in latest.items():
        try:
            lookup = int(row["lookup_hit_tokens"])
            allocated = int(row["allocated_external_tokens"])
            loaded = int(row["actual_loaded_tokens"])
            requested = int(row["requested_prefix_tokens"])
            local = int(row["locally_cached_tokens"])
            missing = int(row["missing_tokens"])
            unallocated = int(row["unallocated_recompute_tokens"])
            shortfall = int(row["load_shortfall_tokens"])
            recomputed = int(row["recomputed_tokens"])
            generation = int(row["binding_generation"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_request_accounting:{request_id}")
            continue
        identity = tuple(str(row.get(field) or "") for field in (
            "physical_object_id", "native_key", "compatibility_identity", "prefix_hash",
        ))
        expected_missing = max(0, requested - local - loaded)
        expected_unallocated = max(0, requested - local - allocated)
        expected_shortfall = allocated - loaded
        if (
            not (lookup >= allocated >= loaded >= 0)
            or missing != expected_missing
            or unallocated != expected_unallocated
            or shortfall != expected_shortfall
            or missing != unallocated + shortfall
            or generation <= 0
            or not all(identity)
        ):
            errors.append(f"accounting_invariant_failed:{request_id}")
        confirmed = row.get("recompute_confirmed") is True
        if confirmed and (shortfall != 0 or recomputed != unallocated):
            errors.append(f"accounting_recompute_closure_failed:{request_id}")
        if not confirmed and recomputed != 0:
            errors.append(f"accounting_unconfirmed_recompute:{request_id}")
        finish_status = str(row.get("finish_status") or "").upper()
        cancelled_or_failed = "ABORT" in finish_status or "ERROR" in finish_status
        if not cancelled_or_failed and unallocated > 0 and not confirmed:
            errors.append(f"accounting_recompute_not_confirmed:{request_id}")
        if shortfall > 0:
            errors.append(f"accounting_native_load_shortfall:{request_id}")
        if row.get("terminal") is not True:
            errors.append(f"accounting_not_terminal:{request_id}")
        reason = str(row.get("terminal_reason") or "")
        if allocated == 0 and reason not in {"scheduler_declined_recompute", "native_recompute_evidence_missing"}:
            errors.append(f"accounting_invalid_recompute_reason:{request_id}")
        if allocated > 0 and reason not in {
            "native_load_completed", "native_partial_prefix_load_recompute",
            "native_load_shortfall_unsafe",
        }:
            errors.append(f"accounting_missing_native_completion:{request_id}")
    return list(latest.values())


def validate_receipts(
    rows: list[dict[str, Any]], accounting: list[dict[str, Any]], errors: list[str],
    *, association_index: dict[str, dict[str, str]] | None = None,
) -> None:
    expected_loaded = {
        str(row.get("request_id")) for row in accounting
        if int(row.get("allocated_external_tokens") or 0) > 0
    }
    accounting_by_request = {str(row.get("request_id")): row for row in accounting}
    seen: set[str] = set()
    for row in rows:
        normalized = _normalize_native_identity(row, association_index or {}, errors, artifact="receipt")
        if normalized is None:
            continue
        row = normalized
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
            local = int(row["locally_cached_tokens"])
            missing = int(row["missing_tokens"])
            unallocated = int(row["unallocated_recompute_tokens"])
            shortfall = int(row["load_shortfall_tokens"])
            generation = int(row["binding_generation"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_native_receipt:{request_id}")
            continue
        if (
            not (lookup >= allocated >= loaded >= 0)
            or missing != max(0, requested - local - loaded)
            or unallocated != max(0, requested - local - allocated)
            or shortfall != allocated - loaded
            or missing != unallocated + shortfall
            or generation <= 0
        ):
            errors.append(f"receipt_invariant_failed:{request_id}")
        if shortfall > 0:
            errors.append(f"receipt_native_load_shortfall:{request_id}")
        accounting_row = accounting_by_request.get(request_id)
        if accounting_row is None:
            errors.append(f"receipt_without_accounting:{request_id}")
            continue
        if row.get("binding_generation") != accounting_row.get("binding_generation"):
            errors.append(f"receipt_identity_mismatch:binding_generation:{request_id}")
        if row.get("allocated_external_tokens") != accounting_row.get("allocated_external_tokens"):
            errors.append(f"receipt_identity_mismatch:allocated_external_tokens:{request_id}")
        expected_keys = str(accounting_row.get("native_key") or "")
        observed_keys = str(row.get("native_key") or "")
        keys_equal = expected_keys == observed_keys
        keys_partial = native_key_prefix_ok(expected_keys, observed_keys)
        if not (keys_equal or keys_partial):
            errors.append(f"receipt_identity_mismatch:native_key:{request_id}")
        elif not keys_equal:
            # Block-aligned shorter prefix of the same object after churn
            # evicted tail chunks; derived identity fields legitimately differ
            # and the evicted tail is reconciled through missing_tokens.
            pass
        else:
            for field in ("physical_object_id", "compatibility_identity", "prefix_hash"):
                if row.get(field) != accounting_row.get(field):
                    errors.append(f"receipt_identity_mismatch:{field}:{request_id}")
    if not expected_loaded.issubset(seen):
        errors.append("receipt_request_coverage_mismatch")


def resource_peaks(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in ("cgroup_memory_current_bytes", "process_rss_bytes", "lmcache_cpu_occupancy_bytes", "lmcache_ssd_occupancy_bytes"):
        values: list[int] = []
        for row in rows:
            try:
                values.append(int(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
        result[field] = max(values) if values else None
    return result


def uma_measurement_status(rows: list[dict[str, Any]], peaks: dict[str, int | None]) -> str:
    """Classify physical-memory evidence without upgrading missing data.

    RSS is useful for diagnosis but is not a GB10 UMA total-memory proof.  A
    cgroup measurement is accepted only when the runtime emitted at least one
    positive value from a resolver that explicitly marked it valid.
    """
    cgroup_rows = [
        row for row in rows
        if row.get("cgroup_memory_status") == "valid"
        and isinstance(row.get("cgroup_memory_current_bytes"), int)
        and int(row["cgroup_memory_current_bytes"]) > 0
    ]
    if cgroup_rows and (peaks.get("cgroup_memory_current_bytes") or 0) > 0:
        return "cgroup_valid"
    if (peaks.get("process_rss_bytes") or 0) > 0:
        return "process_rss_only"
    return "unavailable"


def aggregate_throughput(rows: list[dict[str, Any]]) -> float | None:
    total_tokens = 0.0
    starts: list[float] = []
    ends: list[float] = []
    for row in rows:
        try:
            tokens = float(row["output_tokens_observed"])
            started = float(row["request_started_s"])
            ended = float(row["request_ended_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if tokens >= 0 and ended > started > 0:
            total_tokens += tokens
            starts.append(started)
            ends.append(ended)
    makespan = max(ends) - min(starts) if starts and ends else 0.0
    return total_tokens / makespan if makespan > 0 else None


def kv_block_budget(run: Path) -> int | None:
    try:
        payload = json.loads((run / "kv_core_run_metadata.json").read_text(encoding="utf-8"))
        value = int(payload["vllm_kv_block_budget"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def runtime_metadata(run: Path) -> dict[str, Any]:
    try:
        value = json.loads((run / "kv_core_run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def experiment_workload_id(run: Path) -> str:
    try:
        payload = json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("workload_id") or "") if isinstance(payload, dict) else ""


def validate_capacity_sweep(path: Path, phase: str) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ["capacity_sweep_manifest_unreadable"], {}
    if payload.get("schema") != "astrakv-kv-core-capacity-sweep-v1":
        errors.append("capacity_sweep_schema_mismatch")
    if payload.get("phase") != phase:
        errors.append("capacity_sweep_phase_mismatch")
    if payload.get("eligible") is not True:
        errors.append("capacity_sweep_not_eligible")
    try:
        baseline = int(payload["accepted_vllm_kv_block_budget"]["baseline"])
        variant = int(payload["accepted_vllm_kv_block_budget"]["variant"])
    except (KeyError, TypeError, ValueError):
        errors.append("capacity_sweep_block_budget_missing")
    else:
        if baseline <= 0 or variant <= 0 or variant > baseline * 0.90:
            errors.append("kv_capacity_saving_not_proven")
    controls = payload.get("controls")
    required_controls = {
        "model", "dtype", "workload_sha256", "request_order", "seed",
        "sampling", "output_length", "cache_state", "slo",
    }
    if not isinstance(controls, dict) or not required_controls.issubset(controls):
        errors.append("capacity_sweep_controls_incomplete")
    return not errors, errors, payload


def validate_current_callback_smoke(
    run: Path,
    errors: list[str],
    *,
    require_native_load: bool,
) -> None:
    try:
        payload = json.loads((run / "callback-smoke.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("callback_smoke_missing_for_current_run")
        return
    if payload.get("patch_id") != PATCH_ID or tuple(payload.get("callbacks") or ()) != REQUIRED_CALLBACKS:
        errors.append("callback_smoke_patch_identity_mismatch")
    required_for_request = {
        "scheduler_exact_lookup", "scheduler_external_admission",
        "connector_metadata", "scheduler_compute_progress", "request_finished",
    }
    if require_native_load:
        required_for_request.update({"native_load_start", "native_load_completion"})
    if not required_for_request.issubset(set(payload.get("observed_callbacks") or ())):
        errors.append("callback_smoke_incomplete_for_current_run")


def validate_prefetch(rows: list[dict[str, Any]], errors: list[str]) -> None:
    consumed = 0
    completed_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("source_tier")) != "ssd" or str(row.get("target_tier")) != "cpu":
            continue
        prefetch_id = str(row.get("prefetch_id") or "")
        if not prefetch_id:
            errors.append("e3_prefetch_identity_missing")
            continue
        completed_by_id[prefetch_id] = row
    for prefetch_id, row in completed_by_id.items():
        if row.get("status") != "consumed" or int(row.get("completed_bytes") or 0) <= 0:
            continue
        target = str(row.get("target_request_id") or "")
        consumer = str(row.get("consumer_request_id") or "")
        identity = tuple(str(row.get(field) or "") for field in (
            "physical_object_id", "binding_generation", "prefix_hash",
            "native_key", "compatibility_identity",
        ))
        if not target or consumer != target or not all(identity):
            errors.append(f"e3_prefetch_consumer_identity_failed:{prefetch_id}")
            continue
        consumed += 1
    if not consumed:
        errors.append("e3_prefetch_consumption_evidence_missing")


def validate_prefetch_disabled(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """Fail closed if the E3C A/A control emitted any prefetch ticket."""
    for label, rows in (("baseline", baseline_rows), ("variant", variant_rows)):
        for row in rows:
            if str(row.get("source_tier")) == "ssd" and str(row.get("target_tier")) == "cpu":
                errors.append(f"e3c_prefetch_ticket_emitted:{label}")
                break


def validate_external_reaping(run: Path, errors: list[str], *, expect_reaps: bool) -> dict[str, Any]:
    """Validate E5/E5C cold external-copy reaping evidence (fail-closed)."""
    rows = load_run_artifact(run, "kv_core_external_reaps.jsonl")
    reaps = [row for row in rows if str(row.get("status")) in {"demoted", "invalidated"}]
    if expect_reaps:
        if not reaps:
            errors.append("e5_external_reap_evidence_missing")
            return {"reap_count": 0, "freed_bytes": 0}
        freed_bytes = sum(max(0, int(row.get("freed_bytes") or 0)) for row in reaps)
        if freed_bytes <= 0:
            errors.append("e5_reap_freed_bytes_missing")
        return {"reap_count": len(reaps), "freed_bytes": freed_bytes}
    leaks = [row for row in rows if str(row.get("status")) in {"demoted", "invalidated", "failed"}]
    if leaks:
        errors.append("e5c_reap_control_leak")
    return {"reap_count": len(rows), "freed_bytes": 0}


def load_run_artifact(run: Path, filename: str) -> list[dict[str, Any]]:
    return read_jsonl(run / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--capacity-sweep-manifest",
        help="Optional independent fixed-SLO KV block capacity sweep for E2/E4.",
    )
    args = parser.parse_args()
    baseline_path, variant_path = Path(args.baseline), Path(args.variant)
    paired = validate_paired_runs(PairedRunInput("baseline", baseline_path), PairedRunInput("variant", variant_path))
    errors = list(paired.errors)
    baseline_requests = load_run_artifact(baseline_path, "request_results.jsonl")
    variant_requests = load_run_artifact(variant_path, "request_results.jsonl")
    workload_types = {
        str(row.get("workload_type") or row.get("workload_id") or "")
        for row in variant_requests
    }
    workload_types.discard("")
    manifest_workload = experiment_workload_id(variant_path)
    if manifest_workload:
        workload_types.add(manifest_workload)
    prefetch_benefit_eligible = bool(workload_types & PREFETCH_BENEFIT_WORKLOADS)
    partial_benefit_eligible = bool(workload_types & PARTIAL_LOAD_WORKLOADS)
    validate_quality(baseline_requests, variant_requests, errors)
    request_ids = {str(row.get("request_id") or "") for row in variant_requests}
    if "" in request_ids:
        errors.append("request_identity_missing")
    try:
        variant_manifest_payload = json.loads(
            (variant_path / "experiment_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        variant_manifest_payload = {}
    if not isinstance(variant_manifest_payload, dict):
        variant_manifest_payload = {}
    association_rows = load_run_artifact(variant_path, "request_context_associations.jsonl")
    association_index = validate_variant_request_associations(
        variant_requests,
        association_rows,
        errors,
        expected_run_id=str(variant_manifest_payload.get("run_id") or ""),
    )
    accounting: list[dict[str, Any]] = []
    if args.phase == "E1":
        # E1 is deliberately run with the repeated exact-prefix smoke. It is
        # not eligible unless the current vendor patch emitted every callback,
        # including the request-owned native load pair.
        validate_current_callback_smoke(
            variant_path, errors, require_native_load=True,
        )
    else:
        accounting = load_run_artifact(variant_path, "kv_core_request_accounting.jsonl")
        accounting = validate_request_accounting(
            accounting, request_ids, errors, association_index=association_index,
        )
        validate_current_callback_smoke(
            variant_path,
            errors,
            require_native_load=any(
                int(row.get("allocated_external_tokens") or 0) > 0
                for row in accounting
            ),
        )
        receipts = load_run_artifact(variant_path, "kv_core_native_receipts.jsonl")
        validate_receipts(receipts, accounting, errors, association_index=association_index)
    baseline_uma = load_run_artifact(baseline_path, "uma_resource_samples.jsonl")
    variant_uma = load_run_artifact(variant_path, "uma_resource_samples.jsonl")
    baseline_resources, variant_resources = resource_peaks(baseline_uma), resource_peaks(variant_uma)
    base_peak = baseline_resources["cgroup_memory_current_bytes"]
    variant_peak = variant_resources["cgroup_memory_current_bytes"]
    baseline_uma_status = uma_measurement_status(baseline_uma, baseline_resources)
    variant_uma_status = uma_measurement_status(variant_uma, variant_resources)
    cgroup_evidence_valid = (
        baseline_uma_status == "cgroup_valid" and variant_uma_status == "cgroup_valid"
    )
    if args.phase != "E1" and not cgroup_evidence_valid:
        errors.append("uma_evidence_missing")
    elif args.phase not in {"E1", "E3C", "E5C"} and variant_peak is not None and base_peak is not None and variant_peak > base_peak * 1.02:
        errors.append("uma_peak_regression")
    point, interval = paired_ttft_bootstrap(baseline_requests, variant_requests)
    if point is None and args.phase not in {"E3C", "E5C"}:
        errors.append("ttft_evidence_missing")
    if args.phase == "E3" and prefetch_benefit_eligible and (
        point is None or point > -5.0 or interval[1] is None or interval[1] >= 0.0
    ):
        errors.append("e3_ttft_acceptance_failed")
    baseline_throughput = aggregate_throughput(baseline_requests)
    variant_throughput = aggregate_throughput(variant_requests)
    if (baseline_throughput is None or variant_throughput is None) and args.phase not in {"E3C", "E5C"}:
        errors.append("throughput_evidence_missing")
    elif args.phase not in {"E3C", "E5C"} and variant_throughput < baseline_throughput * 0.98:
        errors.append("throughput_regression")
    no_reuse_baseline = [row for row in baseline_requests if str(row.get("workload_type")) == "random_no_reuse"]
    no_reuse_variant = [row for row in variant_requests if str(row.get("workload_type")) == "random_no_reuse"]
    if args.phase not in {"E3C", "E5C"} and (no_reuse_baseline or no_reuse_variant):
        no_reuse_point, _ = paired_ttft_bootstrap(no_reuse_baseline, no_reuse_variant)
        if no_reuse_point is None or no_reuse_point > 2.0:
            errors.append("no_reuse_ttft_regression")
    baseline_budget = kv_block_budget(baseline_path)
    variant_budget = kv_block_budget(variant_path)
    capacity_record: dict[str, Any] = {}
    capacity_status = "not_evaluated"
    if args.capacity_sweep_manifest:
        capacity_ok, capacity_errors, capacity_record = validate_capacity_sweep(
            Path(args.capacity_sweep_manifest), args.phase,
        )
        errors.extend(capacity_errors)
        capacity_status = "passed" if capacity_ok else "failed"
    if args.phase in {"E3", "E3C", "E4", "E5", "E5C"}:
        for label, path in (("baseline", baseline_path), ("variant", variant_path)):
            metadata = runtime_metadata(path)
            if metadata.get("topology") != "gpu_cpu_ssd" or metadata.get("lmcache_local_cpu_enabled") is not True:
                errors.append(f"e3_local_cpu_topology_not_proven:{label}")
            if metadata.get("disk_backed_cpu_invalidation_on_prefetch_lead") is not True:
                errors.append(f"e3_disk_backed_cpu_invalidation_not_proven:{label}")
        if args.phase in {"E3", "E4", "E5"} and prefetch_benefit_eligible:
            validate_prefetch(load_run_artifact(variant_path, "kv_core_prefetch_tickets.jsonl"), errors)
        if args.phase == "E3C":
            validate_prefetch_disabled(
                load_run_artifact(baseline_path, "kv_core_prefetch_tickets.jsonl"),
                load_run_artifact(variant_path, "kv_core_prefetch_tickets.jsonl"),
                errors,
            )
    if args.phase in {"E4", "E5", "E5C"} and partial_benefit_eligible:
        # Upper-bound partial load must retain receipt accounting, enforced above.
        partial_rows = [row for row in accounting if int(row.get("allocated_external_tokens") or 0) < int(row.get("lookup_hit_tokens") or 0)]
        if not partial_rows:
            errors.append("e4_partial_prefix_evidence_missing")
    reap_evidence = {
        "baseline": {"reap_count": 0, "freed_bytes": 0},
        "variant": {"reap_count": 0, "freed_bytes": 0},
    }
    if args.phase in {"E5", "E5C"}:
        reap_evidence["baseline"] = validate_external_reaping(baseline_path, errors, expect_reaps=False)
        if args.phase == "E5":
            reap_evidence["variant"] = validate_external_reaping(variant_path, errors, expect_reaps=True)
            base_cpu = baseline_resources.get("lmcache_cpu_occupancy_bytes")
            variant_cpu = variant_resources.get("lmcache_cpu_occupancy_bytes")
            if base_cpu is not None and variant_cpu is not None and variant_cpu > base_cpu * 1.02:
                errors.append("e5_external_occupancy_regression")
        else:
            reap_evidence["variant"] = validate_external_reaping(variant_path, errors, expect_reaps=False)
    record = {
        "schema": SCHEMA,
        "phase": args.phase,
        "eligible": not errors,
        "errors": list(dict.fromkeys(errors)),
        "paired_manifest": paired.record,
        "ttft_p95_delta_percent": point,
        "ttft_p95_bootstrap_ci_percent": list(interval),
        "uma_peak_bytes": {"baseline": base_peak, "variant": variant_peak},
        "resource_peaks_bytes": {"baseline": baseline_resources, "variant": variant_resources},
        "throughput_tokens_s": {"baseline": baseline_throughput, "variant": variant_throughput},
        "vllm_kv_block_budget": {"baseline": baseline_budget, "variant": variant_budget},
        "capacity_claim": {"status": capacity_status, "manifest": capacity_record},
        "prefetch_benefit_eligible": prefetch_benefit_eligible and args.phase != "E3C",
        "partial_load_benefit_eligible": partial_benefit_eligible and args.phase != "E3C",
        "external_reap_evidence": reap_evidence,
        "control_mode": (
            "cpu_tier_aa_no_prefetch"
            if args.phase == "E3C"
            else ("cold_reap_aa" if args.phase == "E5C" else ("cold_reap_active" if args.phase == "E5" else ""))
        ),
        "request_accounting_count": len(accounting),
        "uma_measurement": {
            "baseline": baseline_uma_status,
            "variant": variant_uma_status,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
