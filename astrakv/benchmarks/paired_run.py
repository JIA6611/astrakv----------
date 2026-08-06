"""Strict evidence gate for paired baseline/variant benchmark claims."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from astrakv.benchmarks.experiment_manifest import file_sha256
from astrakv.runtime.backend_capabilities import (
    INSTALLATION_EVIDENCE_SCHEMA,
    InstallationEvidence,
    SUPPORTED_CONNECTOR_NAME,
    SUPPORTED_LMCACHE_VERSION,
    SUPPORTED_VLLM_VERSION,
)


PAIRED_RUN_SCHEMA = "astrakv-paired-run-manifest-v2"
MANIFEST_SCHEMA = "astrakv-experiment-manifest-v2"
BACKEND_HOOK_SCHEMA = "astrakv-backend-hook-v2"
VALID_CLAIM_SCOPES = {"benchmark", "online_control"}
SOURCE_ARTIFACTS = ("workload", "matrix", "environment")
COMMON_ARTIFACTS = SOURCE_ARTIFACTS + ("benchmark", "requests", "quality")
ONLINE_CONTROL_ARTIFACTS = (
    "backend_capabilities",
    "backend_binding_events",
    "runtime_events_raw",
    "astrakv_runtime_commands",
    "runtime_command_receipts",
    "runtime_structured_events",
    "online_profile_checkpoint",
    "trace",
)
SUCCESSFUL_RECEIPT_STATUSES = {"completed", "ok", "executed"}


@dataclass(frozen=True, slots=True)
class PairedRunInput:
    label: str
    run_path: Path

    @property
    def manifest_path(self) -> Path:
        return self.run_path / "experiment_manifest.json"


@dataclass(frozen=True, slots=True)
class PairedRunValidation:
    eligible: bool
    errors: tuple[str, ...]
    record: dict[str, Any]


def validate_paired_runs(baseline: PairedRunInput, variant: PairedRunInput) -> PairedRunValidation:
    """Prove that two archived runs support a controlled comparison claim."""
    errors: list[str] = []
    manifests = {"baseline": _load_manifest(baseline, errors), "variant": _load_manifest(variant, errors)}
    if any(value is None for value in manifests.values()):
        return _result(baseline, variant, manifests, errors, {}, {})
    baseline_manifest = manifests["baseline"]
    variant_manifest = manifests["variant"]
    assert baseline_manifest is not None and variant_manifest is not None
    _validate_manifest_identity("baseline", baseline_manifest, errors)
    _validate_manifest_identity("variant", variant_manifest, errors)
    _require_pair_identity(baseline_manifest, variant_manifest, errors)
    claim_scope = _require_claim_scope(baseline_manifest, variant_manifest, errors)
    _require_equal_identities(baseline_manifest, variant_manifest, errors)
    artifacts = {
        "baseline": _validate_artifacts(baseline, baseline_manifest, errors),
        "variant": _validate_artifacts(variant, variant_manifest, errors),
    }
    coverage = _validate_coverage(artifacts, errors)
    if claim_scope == "online_control":
        for label, run, manifest in (("baseline", baseline, baseline_manifest), ("variant", variant, variant_manifest)):
            _validate_online_control_evidence(label, run, manifest, artifacts[label], errors)
    return _result(baseline, variant, manifests, errors, artifacts, coverage, claim_scope=claim_scope)


def write_paired_run_manifest(path: str | Path, result: PairedRunValidation) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.record, indent=2, ensure_ascii=False), encoding="utf-8")


def _result(baseline: PairedRunInput, variant: PairedRunInput, manifests: dict[str, dict[str, Any] | None], errors: list[str], artifacts: dict[str, dict[str, Path]], coverage: dict[str, Any], *, claim_scope: str = "unvalidated") -> PairedRunValidation:
    unique_errors = tuple(_unique(errors))
    record = {
        "schema": PAIRED_RUN_SCHEMA,
        "eligible": not unique_errors,
        "errors": list(unique_errors),
        "pair_id": next((str(item.get("pair_id") or "") for item in manifests.values() if item), ""),
        "claim_scope": claim_scope,
        "runs": {label: {"label": run.label, "path": str(run.run_path), "manifest": str(run.manifest_path)} for label, run in (("baseline", baseline), ("variant", variant))},
        "manifest_hashes": {label: file_sha256(run.manifest_path) for label, run in (("baseline", baseline), ("variant", variant))},
        "artifact_hashes": {label: {role: file_sha256(path) for role, path in entries.items()} for label, entries in artifacts.items()},
        "coverage": coverage,
    }
    return PairedRunValidation(not unique_errors, unique_errors, record)


def _load_manifest(run: PairedRunInput, errors: list[str]) -> dict[str, Any] | None:
    if not run.manifest_path.is_file():
        errors.append(f"missing_manifest:{run.label}")
        return None
    try:
        value = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid_manifest:{run.label}")
        return None
    if not isinstance(value, dict):
        errors.append(f"invalid_manifest:{run.label}")
        return None
    return value


def _validate_manifest_identity(label: str, manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"invalid_manifest_schema:{label}")
    for field in ("run_id", "pair_id", "pair_role", "workload_sha256", "matrix_sha256", "environment_sha256"):
        if not _identity(manifest.get(field)):
            errors.append(f"empty_identity:{label}:{field}")
    for field in ("model", "model_revision", "tokenizer_revision", "dtype", "quantization"):
        if not _identity(manifest.get(field)):
            errors.append(f"empty_model_identity:{label}:{field}")


def _identity(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "unknown", "none", "n/a"} else text


def _require_pair_identity(baseline: dict[str, Any], variant: dict[str, Any], errors: list[str]) -> None:
    if _identity(baseline.get("pair_id")) != _identity(variant.get("pair_id")):
        errors.append("pair_id_mismatch")
    if baseline.get("pair_role") != "baseline":
        errors.append("invalid_pair_role:baseline")
    if variant.get("pair_role") != "variant":
        errors.append("invalid_pair_role:variant")


def _require_claim_scope(baseline: dict[str, Any], variant: dict[str, Any], errors: list[str]) -> str:
    baseline_scope = str(baseline.get("claim_scope") or "")
    variant_scope = str(variant.get("claim_scope") or "")
    if baseline_scope not in VALID_CLAIM_SCOPES:
        errors.append("invalid_claim_scope:baseline")
    if variant_scope not in VALID_CLAIM_SCOPES:
        errors.append("invalid_claim_scope:variant")
    if baseline_scope != variant_scope:
        errors.append("claim_scope_mismatch")
        return "unvalidated"
    return baseline_scope if baseline_scope in VALID_CLAIM_SCOPES else "unvalidated"


def _require_equal_identities(baseline: dict[str, Any], variant: dict[str, Any], errors: list[str]) -> None:
    for field in ("workload_sha256", "matrix_sha256", "environment_sha256"):
        if _identity(baseline.get(field)) != _identity(variant.get(field)):
            errors.append(f"{field}_mismatch")
    if tuple(_identity(baseline.get(field)) for field in ("model", "model_revision", "tokenizer_revision", "dtype", "quantization")) != tuple(_identity(variant.get(field)) for field in ("model", "model_revision", "tokenizer_revision", "dtype", "quantization")):
        errors.append("model_identity_mismatch")


def _validate_artifacts(run: PairedRunInput, manifest: dict[str, Any], errors: list[str]) -> dict[str, Path]:
    paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    hashes = manifest.get("artifact_hashes") if isinstance(manifest.get("artifact_hashes"), dict) else {}
    required = COMMON_ARTIFACTS + (ONLINE_CONTROL_ARTIFACTS if manifest.get("claim_scope") == "online_control" else ())
    result: dict[str, Path] = {}
    for role in required:
        target = _artifact_path(run.run_path, paths.get(role))
        if target is None or not target.is_file():
            errors.append(f"missing_artifact:{run.label}:{role}")
            continue
        actual = file_sha256(target)
        if not actual or hashes.get(role) != actual:
            errors.append(f"artifact_hash_mismatch:{run.label}:{role}")
            continue
        result[role] = target
    for role, field in (("workload", "workload_sha256"), ("matrix", "matrix_sha256"), ("environment", "environment_sha256")):
        if role in result and manifest.get(field) != file_sha256(result[role]):
            errors.append(f"source_hash_mismatch:{run.label}:{role}")
    return result


def _artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    target = Path(value)
    return target if target.is_absolute() else root / target


def _validate_coverage(artifacts: dict[str, dict[str, Path]], errors: list[str]) -> dict[str, Any]:
    coverage: dict[str, Any] = {"cases": [], "sample_ids": []}
    benchmark_cases = {label: _csv_values(path["benchmark"], "case", label, "benchmark", errors) if "benchmark" in path else set() for label, path in artifacts.items()}
    if all(benchmark_cases.values()):
        if benchmark_cases["baseline"] != benchmark_cases["variant"]:
            errors.append("case_coverage_mismatch")
        coverage["cases"] = sorted(benchmark_cases["baseline"])
    request_rows = {label: _read_jsonl(path["requests"], label, "requests", errors) if "requests" in path else [] for label, path in artifacts.items()}
    request_samples: dict[str, set[str]] = {}
    for label, rows in request_rows.items():
        cases = _unique_values(rows, "case", label, "requests", errors)
        request_samples[label] = _unique_values(rows, "sample_id", label, "requests", errors)
        _unique_values(rows, "request_id", label, "requests", errors)
        if benchmark_cases[label] and cases != benchmark_cases[label]:
            errors.append(f"request_case_coverage_mismatch:{label}")
    if not request_samples["baseline"] or request_samples["baseline"] != request_samples["variant"]:
        errors.append("sample_provenance_mismatch")
    coverage["sample_ids"] = sorted(request_samples["baseline"])
    for label, paths in artifacts.items():
        if "quality" not in paths:
            continue
        quality_samples = _csv_values(paths["quality"], "sample_id", label, "quality", errors)
        if not quality_samples or quality_samples != request_samples[label]:
            errors.append("sample_provenance_mismatch")
    return coverage


def _validate_online_control_evidence(label: str, run: PairedRunInput, manifest: dict[str, Any], artifacts: dict[str, Path], errors: list[str]) -> None:
    if any(role not in artifacts for role in ONLINE_CONTROL_ARTIFACTS + ("requests",)):
        return
    request_rows = _read_jsonl(artifacts["requests"], label, "requests", errors)
    request_ids = _unique_values(request_rows, "request_id", label, "requests", errors)
    bindings = _read_jsonl(artifacts["backend_binding_events"], label, "backend_binding_events", errors)
    events = _read_jsonl(artifacts["runtime_events_raw"], label, "runtime_events_raw", errors)
    commands = _read_jsonl(artifacts["astrakv_runtime_commands"], label, "astrakv_runtime_commands", errors)
    receipts = _read_jsonl(artifacts["runtime_command_receipts"], label, "runtime_command_receipts", errors)
    traces = _read_jsonl(artifacts["trace"], label, "trace", errors)
    run_id = str(manifest.get("run_id"))
    binding_index: dict[str, dict[str, Any]] = {}
    for row in bindings:
        if not _hook_row(row, "binding", label, run_id, errors):
            continue
        key = _binding_key(row, label, "backend_binding_events", errors)
        if key is None:
            continue
        if key in binding_index:
            errors.append(f"duplicate_binding_id:{label}")
        binding_index[key] = row
        if not _row_request(row, request_ids) or not _object_identity(row):
            errors.append(f"binding_identity_mismatch:{label}")
    for row in events:
        if _hook_row(row, "event", label, run_id, errors):
            _validate_bound_row(row, binding_index, request_ids, label, "event", errors)
    command_index: dict[str, dict[str, Any]] = {}
    for row in commands:
        if not _hook_row(row, "command", label, run_id, errors):
            continue
        command_id = _required_text(row, "command_id")
        if not command_id:
            errors.append(f"invalid_command_row:{label}")
            continue
        if command_id in command_index:
            errors.append(f"duplicate_command_id:{label}")
        command_index[command_id] = row
        _validate_bound_row(row, binding_index, request_ids, label, "command", errors)
        if not _required_text(row, "action"):
            errors.append(f"invalid_command_action:{label}")
    receipt_by_command: dict[str, list[dict[str, Any]]] = {}
    receipt_ids: set[str] = set()
    for row in receipts:
        if not _hook_row(row, "receipt", label, run_id, errors):
            continue
        receipt_id = _required_text(row, "receipt_id")
        if not receipt_id or receipt_id in receipt_ids:
            errors.append(f"duplicate_receipt_id:{label}")
        receipt_ids.add(receipt_id)
        command_id = _required_text(row, "command_id")
        receipt_by_command.setdefault(command_id, []).append(row)
    for command_id, command in command_index.items():
        matches = receipt_by_command.get(command_id, [])
        if len(matches) != 1 or matches[0].get("status") not in SUCCESSFUL_RECEIPT_STATUSES:
            errors.append(f"terminal_receipt_count_mismatch:{label}:{command_id}")
            continue
        receipt = matches[0]
        for field in ("run_id", "binding_id", "backend_object_id", "action"):
            if receipt.get(field) != command.get(field):
                errors.append(f"receipt_command_mismatch:{label}")
                break
    if set(receipt_by_command) != set(command_index):
        errors.append(f"receipt_command_mismatch:{label}")
    trace_request_ids: set[str] = set()
    for row in traces:
        if not isinstance(row, dict) or not _required_text(row, "schema") or row.get("run_id") != run_id:
            errors.append(f"invalid_trace_row:{label}")
            continue
        request_id = _required_text(row, "request_id")
        if not request_id or request_id in trace_request_ids:
            errors.append(f"trace_request_coverage_mismatch:{label}")
        trace_request_ids.add(request_id)
    if trace_request_ids != request_ids:
        errors.append(f"trace_request_coverage_mismatch:{label}")
    try:
        preflight = json.loads(artifacts["backend_capabilities"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid_backend_capabilities:{label}")
        return
    if not _valid_execution_preflight(preflight, run_id):
        errors.append(f"backend_capabilities_incompatible:{label}")


def _valid_execution_preflight(preflight: Any, run_id: str) -> bool:
    if not isinstance(preflight, dict) or preflight.get("schema") != "astrakv-backend-capabilities-v1":
        return False
    if preflight.get("compatible") is not True or preflight.get("execution_eligible") is not True or preflight.get("run_id") != run_id:
        return False
    if preflight.get("backend_versions") != {"vllm": SUPPORTED_VLLM_VERSION, "lmcache": SUPPORTED_LMCACHE_VERSION}:
        return False
    if preflight.get("connector") != {"name": SUPPORTED_CONNECTOR_NAME, "version": SUPPORTED_LMCACHE_VERSION}:
        return False
    if "drop" not in (preflight.get("allowed_actions") or ()) or "prefix" not in (preflight.get("object_levels") or ()):
        return False
    flags = preflight.get("capability_flags")
    evidence = preflight.get("installation_evidence")
    if not (
        isinstance(flags, dict) and bool(flags) and all(value is True for value in flags.values())
        and isinstance(evidence, dict) and evidence.get("schema") == INSTALLATION_EVIDENCE_SCHEMA
        and all(_required_text(evidence, field) for field in ("source", "method", "session_id", "probe_digest"))
    ):
        return False
    return InstallationEvidence(
        source=str(evidence["source"]), method=str(evidence["method"]), session_id=str(evidence["session_id"]),
        probe_digest=str(evidence["probe_digest"]), schema=str(evidence["schema"]),
    ).is_valid(
        vllm_version=SUPPORTED_VLLM_VERSION, lmcache_version=SUPPORTED_LMCACHE_VERSION,
        connector_name=SUPPORTED_CONNECTOR_NAME, connector_version=SUPPORTED_LMCACHE_VERSION,
        endpoint_identity=preflight.get("endpoint_identity"),
    )


def _hook_row(row: dict[str, Any], record_type: str, label: str, run_id: str, errors: list[str]) -> bool:
    if row.get("schema") != BACKEND_HOOK_SCHEMA or row.get("record_type") != record_type or row.get("run_id") != run_id:
        errors.append(f"invalid_{record_type}_row:{label}")
        return False
    return True


def _validate_bound_row(row: dict[str, Any], bindings: dict[str, dict[str, Any]], request_ids: set[str], label: str, role: str, errors: list[str]) -> None:
    key = _binding_key(row, label, role, errors)
    binding = bindings.get(key) if key else None
    if binding is None or not _row_request(row, request_ids) or not _object_identity(row):
        errors.append(f"binding_evidence_mismatch:{label}")
        return
    for field in ("run_id", "object_key", "object_level", "backend_object_id"):
        if row.get(field) != binding.get(field):
            errors.append(f"binding_evidence_mismatch:{label}")
            break


def _binding_key(row: dict[str, Any], label: str, role: str, errors: list[str]) -> str | None:
    binding_id = _required_text(row, "binding_id")
    try:
        generation = int(row.get("binding_generation"))
    except (TypeError, ValueError):
        generation = 0
    if not binding_id or generation <= 0:
        errors.append(f"invalid_binding_generation:{label}:{role}")
        return None
    return binding_id


def _object_identity(row: dict[str, Any]) -> bool:
    return all(_required_text(row, field) for field in ("object_key", "object_level", "backend_object_id"))


def _row_request(row: dict[str, Any], request_ids: set[str]) -> bool:
    return bool(_required_text(row, "request_id") and row.get("request_id") in request_ids)


def _required_text(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or "").strip()


def _read_jsonl(path: Path, label: str, role: str, errors: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            errors.append(f"malformed_jsonl:{label}:{role}")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"malformed_jsonl:{label}:{role}")
            continue
        if not isinstance(row, dict):
            errors.append(f"nonobject_jsonl:{label}:{role}")
            continue
        result.append(row)
    return result


def _csv_values(path: Path, field: str, label: str, role: str, errors: list[str]) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or field not in reader.fieldnames:
            errors.append(f"invalid_csv_schema:{label}:{role}")
            return set()
        values = [str(row.get(field) or "").strip() for row in reader]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        errors.append(f"invalid_csv_coverage:{label}:{role}")
    return {value for value in values if value}


def _unique_values(rows: list[dict[str, Any]], field: str, label: str, role: str, errors: list[str]) -> set[str]:
    values = [_required_text(row, field) for row in rows]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        errors.append(f"duplicate_or_missing_{field}:{label}:{role}")
    return {value for value in values if value}


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
