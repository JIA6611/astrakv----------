#!/usr/bin/env python3
"""Validate same-engine native-load versus native-recompute KV equivalence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.validate_kv_core_acceptance import load_run_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run = Path(args.run_dir)
    errors: list[str] = []
    requests = load_run_artifact(run, "request_results.jsonl")
    by_case = {str(row.get("workload_case") or ""): row for row in requests}
    loaded, recompute = by_case.get("kv_equivalence_loaded"), by_case.get("kv_equivalence_recompute")
    if loaded is None or recompute is None:
        errors.append("equivalence_request_cases_missing")
    else:
        _validate_output_equivalence(loaded, recompute, args.logprob_tolerance, errors)
        _validate_context_equivalence(run, loaded, recompute, errors)
        _validate_accounting(run, loaded, recompute, errors)
    decisions = load_run_artifact(run, "kv_core_policy_decisions.jsonl")
    if not any(row.get("reason") == "equivalence_probe_force_recompute" and row.get("test_only") is True for row in decisions):
        errors.append("equivalence_force_recompute_decision_missing")
    try:
        runtime_metadata = json.loads((run / "kv_core_run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        runtime_metadata = {}
    if runtime_metadata.get("equivalence_test_enabled") is not True:
        errors.append("equivalence_test_mode_not_proven")
    record = {
        "schema": "astrakv-kv-equivalence-v1",
        "phase": "P3a_same_engine_generation_equivalence",
        "eligible": not errors,
        "errors": list(dict.fromkeys(errors)),
        "quality_evidence_scope": "same_engine_deterministic_generation",
        "teacher_forced_loss_proven": False,
        "loaded_request_id": "" if loaded is None else loaded.get("request_id"),
        "recompute_request_id": "" if recompute is None else recompute.get("request_id"),
        "logprob_tolerance": args.logprob_tolerance,
        "next_required_gate": "teacher_forced_profile_and_strict_e3_correctness",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["eligible"] else 2


def _validate_output_equivalence(loaded: dict[str, Any], recompute: dict[str, Any], tolerance: float, errors: list[str]) -> None:
    for field in ("status", "prompt_hash", "context_length", "generation_seed", "output_token_ids", "finish_reason"):
        if loaded.get(field) != recompute.get(field):
            errors.append(f"equivalence_mismatch:{field}")
    for row in (loaded, recompute):
        if row.get("status") != "ok" or not isinstance(row.get("output_token_ids"), list) or not row["output_token_ids"]:
            errors.append("equivalence_output_evidence_missing")
    try:
        delta = abs(float(loaded["deterministic_logprob"]) - float(recompute["deterministic_logprob"]))
    except (KeyError, TypeError, ValueError):
        errors.append("equivalence_logprob_missing")
    else:
        if not math.isfinite(delta) or delta > tolerance:
            errors.append("equivalence_logprob_mismatch")


def _validate_context_equivalence(run: Path, loaded: dict[str, Any], recompute: dict[str, Any], errors: list[str]) -> None:
    contexts = {str(row.get("request_id") or ""): row for row in load_run_artifact(run, "request_context.jsonl")}
    try:
        a, b = contexts[str(loaded["request_id"])], contexts[str(recompute["request_id"])]
    except KeyError:
        errors.append("equivalence_exact_context_missing")
        return
    for field in ("exact_token_ids", "prefix_id", "cache_key", "prefix_hash", "tokenizer_revision", "chat_template_revision"):
        if a.get("metadata", {}).get(field) != b.get("metadata", {}).get(field):
            errors.append(f"equivalence_context_mismatch:{field}")


def _validate_accounting(run: Path, loaded: dict[str, Any], recompute: dict[str, Any], errors: list[str]) -> None:
    associations = {
        str(row.get("request_id") or ""): str(row.get("runtime_request_id") or "")
        for row in load_run_artifact(run, "request_context_associations.jsonl")
        if row.get("status") == "associated"
    }
    accounting = {
        str(row.get("native_request_id") or row.get("request_id") or ""): row
        for row in load_run_artifact(run, "kv_core_request_accounting.jsonl")
        if row.get("terminal") is True
    }
    try:
        load_row = accounting[associations[str(loaded["request_id"])]]
        recompute_row = accounting[associations[str(recompute["request_id"])]]
    except KeyError:
        errors.append("equivalence_native_accounting_missing")
        return
    if int(load_row.get("actual_loaded_tokens") or 0) <= 0 or int(load_row.get("allocated_external_tokens") or 0) <= 0:
        errors.append("equivalence_native_load_not_proven")
    if any(int(recompute_row.get(field) or 0) != 0 for field in ("allocated_external_tokens", "actual_loaded_tokens")):
        errors.append("equivalence_recompute_was_not_scheduler_declined")
    if recompute_row.get("recompute_confirmed") is not True or int(recompute_row.get("recomputed_tokens") or 0) <= 0:
        errors.append("equivalence_recompute_not_proven")
    for field in ("physical_object_id", "binding_generation", "native_key", "compatibility_identity", "prefix_hash"):
        if load_row.get(field) != recompute_row.get(field):
            errors.append(f"equivalence_kv_identity_mismatch:{field}")


if __name__ == "__main__":
    raise SystemExit(main())
