#!/usr/bin/env python3
"""Validate one variant-only load/recompute regime cell.

The regime matrix compares independent variant cells, so the paired
baseline/variant acceptance gate cannot validate an individual cell.  This
gate validates the archived run and native accounting without making a
performance claim: request success and quality evidence, manifest hashes,
runtime-owned identity associations, callback/receipt closure, UMA evidence,
and the declared arm's load/partial/recompute semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.experiment_manifest import file_sha256  # noqa: E402
from scripts.reporting.validate_kv_core_acceptance import (  # noqa: E402
    load_run_artifact,
    resource_peaks,
    runtime_metadata,
    uma_measurement_status,
    validate_current_callback_smoke,
    validate_prefetch,
    validate_receipts,
    validate_request_accounting,
    validate_variant_request_associations,
)


SCHEMA = "astrakv-load-recompute-regime-cell-v1"
PHASE_BY_ARM = {"off": "E0", "full": "E3", "partial": "E4", "recompute_only": "E2R"}
REQUIRED_MANIFEST_ARTIFACTS = (
    "workload", "matrix", "environment", "control_environment",
    "benchmark", "requests", "quality",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(PHASE_BY_ARM))
    parser.add_argument("--phase", required=True, choices=sorted(set(PHASE_BY_ARM.values())))
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors: list[str] = []
    record = validate(Path(args.run_dir), args.arm, args.phase, args.workload, errors)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["eligible"] else 2


def validate(
    run: Path,
    arm: str,
    phase: str,
    workload: str,
    errors: list[str],
) -> dict[str, Any]:
    manifest_path = run / "experiment_manifest.json"
    if PHASE_BY_ARM.get(arm) != phase:
        errors.append(f"arm_phase_mismatch:{arm}:{phase}")
    manifest = _load_json(manifest_path)
    if manifest is None:
        errors.append("experiment_manifest_missing")
        manifest = {}
    _validate_manifest(run, manifest, workload, errors)

    requests = load_run_artifact(run, "request_results.jsonl")
    request_ids = {str(row.get("request_id") or "") for row in requests}
    if not requests:
        errors.append("request_results_missing")
    if "" in request_ids or len(request_ids) != len(requests):
        errors.append("request_identity_invalid")
    for row in requests:
        request_id = str(row.get("request_id") or "")
        if str(row.get("status") or "") != "ok":
            errors.append(f"request_not_ok:{request_id}")
        if not isinstance(row.get("output_token_ids"), list) or not row["output_token_ids"]:
            errors.append(f"quality_token_evidence_missing:{request_id}")
        if not str(row.get("finish_reason") or ""):
            errors.append(f"quality_finish_reason_missing:{request_id}")
        if not isinstance(row.get("deterministic_logprob"), (int, float)):
            errors.append(f"quality_logprob_missing:{request_id}")

    uma_rows = load_run_artifact(run, "uma_resource_samples.jsonl")
    peaks = resource_peaks(uma_rows)
    uma_status = uma_measurement_status(uma_rows, peaks)
    if uma_status != "cgroup_valid":
        errors.append("uma_evidence_missing")

    accounting: list[dict[str, Any]] = []
    if arm != "off":
        associations = load_run_artifact(run, "request_context_associations.jsonl")
        association_index = validate_variant_request_associations(
            requests,
            associations,
            errors,
            expected_run_id=str(manifest.get("run_id") or ""),
        )
        accounting = validate_request_accounting(
            load_run_artifact(run, "kv_core_request_accounting.jsonl"),
            request_ids,
            errors,
            association_index=association_index,
        )
        require_load = arm in {"full", "partial"}
        validate_current_callback_smoke(run, errors, require_native_load=require_load)
        validate_receipts(
            load_run_artifact(run, "kv_core_native_receipts.jsonl"),
            accounting,
            errors,
            association_index=association_index,
        )
        _validate_runtime_mode(run, arm, errors)
        validate_arm_semantics(
            arm,
            accounting,
            load_run_artifact(run, "kv_core_policy_decisions.jsonl"),
            load_run_artifact(run, "kv_core_prefetch_tickets.jsonl"),
            errors,
        )

    return {
        "schema": SCHEMA,
        "phase": phase,
        "arm": arm,
        "workload": workload,
        "eligible": not errors,
        "errors": list(dict.fromkeys(errors)),
        "request_count": len(requests),
        "ok_count": sum(str(row.get("status") or "") == "ok" for row in requests),
        "request_accounting_count": len(accounting),
        "loaded_tokens": sum(
            max(0, int(row.get("actual_loaded_tokens") or 0)) for row in accounting
        ),
        "uma_measurement": uma_status,
        "uma_peak_bytes": peaks.get("cgroup_memory_current_bytes"),
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else "",
    }


def validate_arm_semantics(
    arm: str,
    accounting: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    prefetch_tickets: list[dict[str, Any]],
    errors: list[str],
) -> None:
    hit_rows = [row for row in accounting if int(row.get("lookup_hit_tokens") or 0) > 0]
    loaded_rows = [row for row in hit_rows if int(row.get("actual_loaded_tokens") or 0) > 0]
    if not hit_rows:
        errors.append("native_lookup_hit_evidence_missing")
        return
    if arm == "full":
        if not loaded_rows:
            errors.append("full_load_evidence_missing")
        if any(
            int(row.get("allocated_external_tokens") or 0) != int(row.get("lookup_hit_tokens") or 0)
            or int(row.get("actual_loaded_tokens") or 0) != int(row.get("allocated_external_tokens") or 0)
            for row in loaded_rows
        ):
            errors.append("full_load_accounting_mismatch")
        validate_prefetch(prefetch_tickets, errors)
    elif arm == "partial":
        partial_rows = [
            row for row in hit_rows
            if 0 < int(row.get("allocated_external_tokens") or 0) < int(row.get("lookup_hit_tokens") or 0)
            and int(row.get("actual_loaded_tokens") or 0) == int(row.get("allocated_external_tokens") or 0)
        ]
        if not partial_rows:
            errors.append("e4_partial_prefix_evidence_missing")
        validate_prefetch(prefetch_tickets, errors)
    elif arm == "recompute_only":
        invalid = [
            row for row in accounting
            if int(row.get("allocated_external_tokens") or 0) != 0
            or int(row.get("actual_loaded_tokens") or 0) != 0
            or row.get("recompute_confirmed") is not True
            or int(row.get("recomputed_tokens") or 0) <= 0
        ]
        if invalid:
            errors.append("recompute_only_accounting_mismatch")
        forced = [
            row for row in decisions
            if str(row.get("reason") or "") == "equivalence_probe_force_recompute"
            and row.get("test_only") is True
        ]
        if len(forced) < len(hit_rows):
            errors.append("recompute_only_force_decision_coverage_missing")


def _validate_runtime_mode(run: Path, arm: str, errors: list[str]) -> None:
    metadata = runtime_metadata(run)
    if arm in {"full", "partial"}:
        if metadata.get("topology") != "gpu_cpu_ssd" or metadata.get("lmcache_local_cpu_enabled") is not True:
            errors.append("load_arm_cpu_topology_not_proven")
        if metadata.get("disk_backed_cpu_invalidation_on_prefetch_lead") is not True:
            errors.append("load_arm_disk_invalidation_not_proven")
    elif arm == "recompute_only":
        if metadata.get("topology") != "gpu_ssd" or metadata.get("lmcache_local_cpu_enabled") is not False:
            errors.append("recompute_only_topology_mismatch")
        if metadata.get("equivalence_test_enabled") is not True:
            errors.append("recompute_only_test_gate_not_proven")


def _validate_manifest(
    run: Path,
    manifest: dict[str, Any],
    workload: str,
    errors: list[str],
) -> None:
    if manifest.get("schema") != "astrakv-experiment-manifest-v2":
        errors.append("experiment_manifest_schema_mismatch")
    if manifest.get("pair_role") != "variant" or manifest.get("claim_scope") != "kv_core":
        errors.append("experiment_manifest_role_or_scope_mismatch")
    if str(manifest.get("workload_id") or "") != workload:
        errors.append("experiment_manifest_workload_mismatch")
    for field in (
        "run_id", "pair_id", "model", "model_revision", "tokenizer_revision",
        "dtype", "quantization",
    ):
        if not str(manifest.get(field) or ""):
            errors.append(f"experiment_manifest_identity_missing:{field}")
    paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    hashes = manifest.get("artifact_hashes") if isinstance(manifest.get("artifact_hashes"), dict) else {}
    for role in REQUIRED_MANIFEST_ARTIFACTS:
        value = str(paths.get(role) or "")
        expected_hash = str(hashes.get(role) or "")
        path = Path(value)
        if not path.is_absolute():
            path = run / path
        if not value or not path.is_file():
            errors.append(f"manifest_artifact_missing:{role}")
        elif not expected_hash or file_sha256(path) != expected_hash:
            errors.append(f"manifest_artifact_hash_mismatch:{role}")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
