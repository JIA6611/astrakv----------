"""Canonical artifact names for audited online runtime runs."""

from __future__ import annotations

from pathlib import Path


FINAL_RUNTIME_ARTIFACT_NAMES = {
    "backend_capabilities": "backend_capabilities.json",
    "backend_binding_events": "backend_binding_events.jsonl",
    "runtime_events_raw": "runtime_events_raw.jsonl",
    "astrakv_runtime_commands": "astrakv_runtime_commands.jsonl",
    "runtime_command_receipts": "runtime_command_receipts.jsonl",
    "runtime_structured_events": "runtime_structured_events.jsonl",
    "online_profile_checkpoint": "online_profile_checkpoint.json",
}

AUXILIARY_RUNTIME_ARTIFACT_NAMES = {
    "trace": "trace_events.jsonl",
}

LEGACY_RUNTIME_ARTIFACT_NAMES = {
    "backend_capabilities": ("preflight.json", "online_preflight.json"),
    "backend_binding_events": ("bindings.jsonl", "online_bindings.jsonl"),
    "runtime_events_raw": ("events.jsonl", "online_events.jsonl"),
    "astrakv_runtime_commands": ("commands.jsonl", "online_commands.jsonl"),
    "runtime_command_receipts": ("receipts.jsonl", "online_receipts.jsonl"),
    "runtime_structured_events": (),
    "online_profile_checkpoint": (),
}


def canonical_artifact_path(root: str | Path, role: str) -> Path:
    """Return the final-contract path for one runtime artifact role."""

    try:
        filename = FINAL_RUNTIME_ARTIFACT_NAMES[role]
    except KeyError as exc:
        raise ValueError(f"unknown runtime artifact role: {role}") from exc
    return Path(root) / filename


def auxiliary_artifact_path(root: str | Path, role: str) -> Path:
    """Return the path for a derived, non-ground-truth runtime artifact."""

    try:
        filename = AUXILIARY_RUNTIME_ARTIFACT_NAMES[role]
    except KeyError as exc:
        raise ValueError(f"unknown auxiliary runtime artifact role: {role}") from exc
    return Path(root) / filename


def find_runtime_artifact(root: str | Path, role: str) -> Path | None:
    """Prefer the final name, then accept known legacy state names on input."""

    directory = Path(root)
    canonical = canonical_artifact_path(directory, role)
    if canonical.is_file():
        return canonical
    for filename in LEGACY_RUNTIME_ARTIFACT_NAMES.get(role, ()):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None
