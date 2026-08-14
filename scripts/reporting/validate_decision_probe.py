#!/usr/bin/env python3
"""Validate the load-vs-recompute decision-correctness probe run.

For every probe row the runtime must emit a decision whose reason matches the
locked scenario label, and the native accounting must prove that a ``load``
decision really loaded external KV while a ``recompute`` decision really was
scheduler-declined with a confirmed native recompute.  Cost-model consistency
between the recorded ``load_cost_ms``/``recompute_cost_ms`` and the recorded
reason is reported as informational evidence, not as a gate.
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

from scripts.reporting.validate_kv_core_acceptance import load_run_artifact  # noqa: E402


SCHEMA = "astrakv-decision-probe-v1"
EXPECTED_ACTION = {"load": "admit_external_prefix", "recompute": "recompute"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workload-manifest", default="", help="decision_probe_workload.manifest.json; defaults to run-dir copy.")
    args = parser.parse_args()
    run = Path(args.run_dir)
    manifest_path = Path(args.workload_manifest) if args.workload_manifest else run / "decision_probe_workload.manifest.json"
    errors: list[str] = []
    summary = validate(run, manifest_path, errors)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["eligible"] else 2


def validate(run: Path, manifest_path: Path, errors: list[str]) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest is None or manifest.get("schema") != "astrakv-decision-probe-workload-v1":
        errors.append("decision_probe_manifest_missing")
        return _record(manifest, errors, {}, 0, 0, 0)
    scenarios: dict[str, Any] = dict(manifest.get("scenarios") or {})
    requests = load_run_artifact(run, "request_results.jsonl")
    decisions = {
        str(row.get("request_id") or ""): row
        for row in load_run_artifact(run, "kv_core_policy_decisions.jsonl")
    }
    accounting = _accounting_index(run)
    runtime_metadata = _load_json(run / "kv_core_run_metadata.json") or {}
    if runtime_metadata.get("equivalence_test_enabled") is not True:
        errors.append("equivalence_test_mode_not_proven")

    per_scenario: dict[str, Any] = {}
    total_rows = 0
    passed_rows = 0
    cost_checks = {"consistent": 0, "checked": 0, "skipped": 0}
    for name, spec in sorted(scenarios.items()):
        expected_action = EXPECTED_ACTION[str(spec.get("expected_decision") or "")]
        expected_reason = str(spec.get("expected_reason") or "")
        rows = [
            row for row in requests
            if str(row.get("workload_case") or "") == f"decision_probe_{name}"
        ]
        cell: dict[str, Any] = {"request_count": len(rows), "ok": 0, "failures": []}
        for row in rows:
            total_rows += 1
            request_id = str(row.get("request_id") or "")
            label = f"{name}:{request_id.rsplit('-', 1)[-1]}"
            decision = decisions.get(request_id)
            if decision is None:
                cell["failures"].append(f"decision_missing:{label}")
                continue
            local_errors: list[str] = []
            actual_reason = str(decision.get("reason") or "")
            actual_action = str(decision.get("action") or "")
            if actual_reason != expected_reason:
                local_errors.append(f"reason_mismatch:{label}:expected={expected_reason},actual={actual_reason}")
            if actual_action != expected_action:
                local_errors.append(f"action_mismatch:{label}:expected={expected_action},actual={actual_action}")
            if name == "S5" and decision.get("test_only") is not True:
                local_errors.append(f"force_recompute_not_test_only:{label}")
            _check_cost_consistency(decision, cost_checks)
            accounting_row = accounting.get(request_id)
            if accounting_row is None:
                local_errors.append(f"accounting_missing:{label}")
            elif str(spec.get("expected_decision")) == "load":
                if int(accounting_row.get("allocated_external_tokens") or 0) <= 0:
                    local_errors.append(f"accounting_load_not_proven:{label}")
                if int(accounting_row.get("actual_loaded_tokens") or 0) <= 0:
                    local_errors.append(f"accounting_load_tokens_missing:{label}")
            else:
                if int(accounting_row.get("allocated_external_tokens") or 0) != 0:
                    local_errors.append(f"accounting_recompute_not_declined:{label}")
                if int(accounting_row.get("actual_loaded_tokens") or 0) != 0:
                    local_errors.append(f"accounting_recompute_loaded:{label}")
                if accounting_row.get("recompute_confirmed") is not True:
                    local_errors.append(f"accounting_recompute_not_confirmed:{label}")
                if int(accounting_row.get("recomputed_tokens") or 0) <= 0:
                    local_errors.append(f"accounting_recompute_tokens_missing:{label}")
            if local_errors:
                cell["failures"].extend(local_errors)
                errors.extend(local_errors)
            else:
                cell["ok"] += 1
                passed_rows += 1
        per_scenario[name] = cell

    return _record(
        manifest, errors, per_scenario,
        total_rows=total_rows, passed_rows=passed_rows, cost_checks=cost_checks,
    )


def _accounting_index(run: Path) -> dict[str, Any]:
    associations = {
        str(row.get("request_id") or ""): str(row.get("runtime_request_id") or "")
        for row in load_run_artifact(run, "request_context_associations.jsonl")
        if str(row.get("status") or "") == "associated"
    }
    accounting = {
        str(row.get("native_request_id") or row.get("request_id") or ""): row
        for row in load_run_artifact(run, "kv_core_request_accounting.jsonl")
        if row.get("terminal") is True
    }
    index: dict[str, Any] = {}
    for logical_id, runtime_id in associations.items():
        row = accounting.get(runtime_id)
        if row is not None:
            index[logical_id] = row
    return index


def _check_cost_consistency(decision: dict[str, Any], counters: dict[str, int]) -> None:
    reason = str(decision.get("reason") or "")
    try:
        load_ms = float(decision.get("load_cost_ms") or 0.0)
        recompute_ms = float(decision.get("recompute_cost_ms") or 0.0)
    except (TypeError, ValueError):
        counters["skipped"] += 1
        return
    if reason == "native_load_cheaper":
        counters["checked"] += 1
        if load_ms < recompute_ms:
            counters["consistent"] += 1
    elif reason == "recompute_cheaper":
        counters["checked"] += 1
        if load_ms >= recompute_ms:
            counters["consistent"] += 1
    else:
        counters["skipped"] += 1


def _record(
    manifest: dict[str, Any] | None,
    errors: list[str],
    per_scenario: dict[str, Any],
    *,
    total_rows: int,
    passed_rows: int,
    cost_checks: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "phase": "load_vs_recompute_decision_probe",
        "eligible": not errors,
        "errors": list(dict.fromkeys(errors)),
        "scenario_count": len(per_scenario),
        "probe_rows": {"total": total_rows, "passed": passed_rows},
        "per_scenario": per_scenario,
        "offline_cost_model_consistency": cost_checks,
        "workload_manifest": manifest,
        "next_required_gate": "load_recompute_regime_matrix",
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
