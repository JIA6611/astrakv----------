"""Export raw runtime-control state into the paired-evidence artifact contract."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

from astrakv.runtime.artifact_contract import (
    FINAL_RUNTIME_ARTIFACT_NAMES,
    auxiliary_artifact_path,
    canonical_artifact_path,
    find_runtime_artifact,
)


ONLINE_ARTIFACT_ROLES = tuple(FINAL_RUNTIME_ARTIFACT_NAMES) + ("trace",)


def export_online_control_artifacts(state_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write the final-contract online artifact snapshot.

    Legacy runtime state names remain accepted as inputs, but every exported
    result uses the canonical final names.
    """
    state = Path(state_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {role: find_runtime_artifact(state, role) for role in FINAL_RUNTIME_ARTIFACT_NAMES}
    targets = {role: canonical_artifact_path(output, role) for role in FINAL_RUNTIME_ARTIFACT_NAMES}

    _write_jsonl(
        targets["backend_binding_events"],
        _latest_bindings(sources["backend_binding_events"]),
    )
    _write_jsonl(
        targets["runtime_events_raw"],
        _validated_event_rows(sources["runtime_events_raw"]),
    )
    _copy_or_empty(sources["astrakv_runtime_commands"], targets["astrakv_runtime_commands"])
    _copy_or_empty(sources["runtime_command_receipts"], targets["runtime_command_receipts"])
    _copy_or_empty(sources["runtime_structured_events"], targets["runtime_structured_events"])
    _copy_or_empty(sources["native_policy_installation"], targets["native_policy_installation"])
    _copy_or_empty(sources["native_cache_policy_evictions"], targets["native_cache_policy_evictions"])
    _copy_or_empty(sources["kv_core_native_callbacks"], targets["kv_core_native_callbacks"])
    _copy_or_empty(sources["kv_core_native_receipts"], targets["kv_core_native_receipts"])

    source_capabilities = sources["backend_capabilities"]
    if source_capabilities is None:
        raise FileNotFoundError(f"runtime capabilities are missing under: {state}")
    shutil.copyfile(source_capabilities, targets["backend_capabilities"])

    source_profile = sources["online_profile_checkpoint"]
    if source_profile is None:
        _write_empty_profile_checkpoint(targets["online_profile_checkpoint"], _infer_run_id(sources))
    else:
        shutil.copyfile(source_profile, targets["online_profile_checkpoint"])

    trace_target = auxiliary_artifact_path(output, "trace")
    _write_jsonl(trace_target, _trace_rows(output / "request_results.jsonl"))
    targets["trace"] = trace_target
    return targets


def refresh_online_control_manifest(state_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Refresh only runtime-derived artifact paths and hashes in an existing manifest."""
    output = Path(output_dir)
    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be a JSON object")
    exported = export_online_control_artifacts(state_dir, output)
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    artifact_hashes = dict(manifest.get("artifact_hashes") or {})
    for role, path in exported.items():
        artifact_paths[role] = path.name
        artifact_hashes[role] = _sha256(path)
    manifest["artifact_paths"] = artifact_paths
    manifest["artifact_hashes"] = artifact_hashes
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _latest_bindings(path: Path | None) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("record_type") not in {None, "binding"}:
            continue
        binding_id = str(row.get("binding_id") or "")
        if binding_id:
            latest[binding_id] = row
    return list(latest.values())


def _trace_rows(request_results: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _read_jsonl(request_results):
        request_id = str(row.get("request_id") or "")
        run_id = str(row.get("run_id") or "")
        if request_id and run_id and request_id not in seen:
            rows.append({"schema": "astrakv-trace-event-v1", "run_id": run_id, "request_id": request_id})
            seen.add(request_id)
    return rows


def _validated_event_rows(path: Path | None) -> list[dict[str, Any]]:
    return [
        row for row in _read_jsonl(path)
        if row.get("schema") == "astrakv-backend-hook-v2"
        and row.get("record_type") == "event"
    ]


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _copy_or_empty(source: Path | None, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copyfile(source, target)
    else:
        target.write_text("", encoding="utf-8")


def _infer_run_id(sources: dict[str, Path | None]) -> str:
    for role in ("backend_capabilities", "backend_binding_events", "runtime_events_raw"):
        for row in _read_jsonl(sources.get(role)):
            run_id = str(row.get("run_id") or "")
            if run_id:
                return run_id
        source = sources.get(role)
        if source is not None and source.suffix == ".json":
            try:
                run_id = str(json.loads(source.read_text(encoding="utf-8")).get("run_id") or "")
            except (OSError, json.JSONDecodeError):
                run_id = ""
            if run_id:
                return run_id
    return "unknown"


def _write_empty_profile_checkpoint(path: Path, run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "astrakv-online-profile-v1",
                "run_id": run_id,
                "event_count": 0,
                "last_event_id": None,
                "objects": {},
                "events": [],
                "event_fingerprints": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
