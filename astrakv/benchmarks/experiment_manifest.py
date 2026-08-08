"""Shared reproducibility manifest for benchmark, simulator, and DGX evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_SENSITIVE_COMMAND_OPTION = re.compile(
    r"(?P<prefix>--(?:request-context-secret-hex|api-key)(?:=|\s+))(?:'[^']*'|\"[^\"]*\"|\S+)"
)


def redact_command_text(command: str) -> str:
    """Remove known credential values from exported command provenance."""
    return _SENSITIVE_COMMAND_OPTION.sub(lambda match: f"{match.group('prefix')}[REDACTED]", command)


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    run_id: str
    workload_id: str = "unknown"
    workload_path: str = ""
    workload_sha256: str = ""
    model: str = "unknown"
    model_revision: str = "unknown"
    tokenizer_revision: str = "unknown"
    dtype: str = "unknown"
    quantization: str = "unknown"
    random_seed: str = "unknown"
    cache_state: str = "unknown"
    capacities: dict[str, int | str] = field(default_factory=dict)
    command: str = "unknown"
    connector_version: str = "unknown"
    input_hashes: dict[str, str] = field(default_factory=dict)
    # v1 consumers ignore these additive paired-evidence fields.
    pair_id: str = ""
    pair_role: str = ""
    matrix_sha256: str = ""
    environment_sha256: str = ""
    # The full environment is run-specific audit evidence.  This fingerprint
    # contains only variables that must remain fixed across a paired control.
    control_environment_sha256: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    claim_scope: str = "benchmark"
    software: dict[str, str] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "astrakv-experiment-manifest-v2",
            "run_id": self.run_id or "unknown",
            "workload_id": self.workload_id or "unknown",
            "workload_path": self.workload_path,
            "workload_sha256": self.workload_sha256,
            "model": self.model or "unknown",
            "model_revision": self.model_revision or "unknown",
            "tokenizer_revision": self.tokenizer_revision or "unknown",
            "dtype": self.dtype or "unknown",
            "quantization": self.quantization or "unknown",
            "random_seed": self.random_seed or "unknown",
            "cache_state": self.cache_state or "unknown",
            "capacities": dict(self.capacities),
            "command": redact_command_text(self.command or "unknown"),
            "connector_version": self.connector_version or "unknown",
            "input_hashes": dict(self.input_hashes),
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
            "matrix_sha256": self.matrix_sha256,
            "environment_sha256": self.environment_sha256,
            "control_environment_sha256": self.control_environment_sha256,
            "artifact_paths": dict(self.artifact_paths),
            "artifact_hashes": dict(self.artifact_hashes),
            "claim_scope": self.claim_scope or "benchmark",
            "software": {**software_versions(), **dict(self.software)},
            "gpu": {**gpu_environment(), **dict(self.gpu)},
            "host": {"platform": platform.platform(), "python": sys.version, "machine": platform.machine()},
        }


def file_sha256(path: str | Path | None) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(path): digest for path in paths if (digest := file_sha256(path))}


def software_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("torch", "vllm", "lmcache"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    result["cuda_runtime"] = os.environ.get("CUDA_VERSION", "unknown")
    return result


def gpu_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"), "driver_version": "unknown", "cuda_version": "unknown", "inventory": []}
    try:
        output = subprocess.check_output([executable, "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL, timeout=10)
        inventory = []
        driver = "unknown"
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                driver = parts[3]
                inventory.append({"index": parts[0], "name": parts[1], "uuid": parts[2], "driver_version": parts[3]})
        cuda = subprocess.check_output([executable], text=True, stderr=subprocess.DEVNULL, timeout=10)
        cuda_line = next((line for line in cuda.splitlines() if "CUDA Version:" in line), "")
        cuda_version = cuda_line.split("CUDA Version:")[-1].strip().split()[0] if cuda_line else "unknown"
        return {"available": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"), "driver_version": driver, "cuda_version": cuda_version, "inventory": inventory}
    except Exception as exc:
        return {"available": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"), "driver_version": "unknown", "cuda_version": "unknown", "inventory": [], "error": f"{type(exc).__name__}: {exc}"}


def write_experiment_manifest(path: str | Path, manifest: ExperimentManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = manifest.to_record()
    hashes = dict(record["artifact_hashes"])
    for role, raw_path in record["artifact_paths"].items():
        candidate = Path(raw_path)
        digest = file_sha256(candidate if candidate.is_absolute() else target.parent / candidate)
        if digest:
            hashes.setdefault(role, digest)
    record["artifact_hashes"] = hashes
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
