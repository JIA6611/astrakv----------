"""Fail-closed verifier for the version-locked KV-Core connector patch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


PATCH_SCHEMA = "astrakv-kv-core-connector-patch-v1"
PATCH_ID = "astrakv-kv-core-vllm-0.23.0-lmcache-0.4.7"
SUPPORTED_VLLM = "0.23.0"
SUPPORTED_LMCACHE = "0.4.7"
REQUIRED_CALLBACKS = (
    "scheduler_exact_lookup",
    "scheduler_external_admission",
    "connector_metadata",
    "native_load_completion",
)


@dataclass(frozen=True, slots=True)
class PatchVerification:
    compatible: bool
    reasons: tuple[str, ...]
    record: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_connector_patch(
    deployment_manifest_path: Path | str,
    *,
    distribution_version: Callable[[str], str] = metadata.version,
    callback_smoke: Callable[[], bool] | None = None,
) -> PatchVerification:
    """Verify exact deployed source inputs before enabling active KV-Core mode.

    The repository intentionally does not contain vendor source hashes.  They
    must be generated from the installed/deployed source tree, so an absent or
    placeholder manifest fails closed instead of creating false provenance.
    """
    reasons: list[str] = []
    manifest_path = Path(deployment_manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
        reasons.append("deployment_manifest_unreadable")
    if payload.get("schema") != PATCH_SCHEMA or payload.get("patch_id") != PATCH_ID:
        reasons.append("patch_identity_mismatch")
    for package, expected in (("vllm", SUPPORTED_VLLM), ("lmcache", SUPPORTED_LMCACHE)):
        try:
            actual = distribution_version(package)
        except metadata.PackageNotFoundError:
            actual = ""
        if actual != expected:
            reasons.append(f"version_mismatch:{package}:{actual or 'missing'}")
    hooks = payload.get("callbacks")
    if not isinstance(hooks, list) or tuple(hooks) != REQUIRED_CALLBACKS:
        reasons.append("callback_contract_mismatch")
    sources = payload.get("source_files")
    if not isinstance(sources, list) or not sources:
        reasons.append("source_hashes_missing")
    else:
        for row in sources:
            if not isinstance(row, dict):
                reasons.append("source_hashes_invalid")
                continue
            path = Path(str(row.get("path") or ""))
            expected = str(row.get("sha256") or "")
            if len(expected) != 64 or not path.is_file():
                reasons.append("source_hashes_invalid")
            elif sha256_file(path) != expected:
                reasons.append(f"source_hash_mismatch:{path.name}")
    marker = payload.get("patch_marker")
    marker_path = Path(str(marker.get("path") or "")) if isinstance(marker, dict) else None
    if (
        not isinstance(marker, dict)
        or marker.get("id") != PATCH_ID
        or marker_path is None
        or not marker_path.is_file()
    ):
        reasons.append("patch_marker_missing")
    else:
        try:
            if PATCH_ID not in marker_path.read_text(encoding="utf-8"):
                reasons.append("patch_marker_invalid")
        except OSError:
            reasons.append("patch_marker_invalid")
    if callback_smoke is None:
        reasons.append("callback_smoke_missing")
    else:
        try:
            if callback_smoke() is not True:
                reasons.append("callback_smoke_failed")
        except Exception:
            reasons.append("callback_smoke_failed")
    record = {
        "schema": PATCH_SCHEMA,
        "patch_id": PATCH_ID,
        "compatible": not reasons,
        "reasons": reasons,
        "required_callbacks": list(REQUIRED_CALLBACKS),
        "deployment_manifest": str(manifest_path),
    }
    return PatchVerification(not reasons, tuple(reasons), record)


__all__ = ["PATCH_ID", "PATCH_SCHEMA", "PatchVerification", "REQUIRED_CALLBACKS", "verify_connector_patch"]
