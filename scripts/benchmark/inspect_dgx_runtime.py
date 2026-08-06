"""Record DGX runtime provenance before attempting structured eviction validation.

The inspector is read-only.  It never enables a connector, mutates a third
party package, or claims that a discoverable package provides object-level
eviction events.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.experiment_manifest import ExperimentManifest, input_hashes  # noqa: E402


def main() -> int:
    args = parse_args()
    payload = inspect_runtime(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DGX runtime environment written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/runtime_environment.json")
    parser.add_argument("--model", default=os.environ.get("ASTRAKV_MODEL", ""))
    parser.add_argument("--launch-command", default="")
    parser.add_argument("--lmcache-config", default=os.environ.get("LMCACHE_CONFIG_FILE", ""))
    parser.add_argument("--disk-path", default=".")
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--workload-id", default="unknown")
    parser.add_argument("--workload-manifest", default="")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--tokenizer-revision", default="unknown")
    parser.add_argument("--dtype", default="unknown")
    parser.add_argument("--quantization", default="unknown")
    parser.add_argument("--random-seed", default="unknown")
    parser.add_argument("--cache-state", choices=("cold", "warm", "unknown"), default="unknown")
    parser.add_argument("--connector-version", default="unknown")
    parser.add_argument("--structured-hook-verification", default="", help="Output JSON from verify_structured_eviction_hook.py")
    return parser.parse_args()


def inspect_runtime(args: argparse.Namespace) -> dict[str, Any]:
    package_names = ("vllm", "lmcache", "torch")
    packages = {name: package_record(name) for name in package_names}
    connector_candidates = (
        "lmcache.integration.vllm",
        "lmcache.integration.vllm.vllm_connector",
        "lmcache.vllm",
    )
    discovered = [name for name in connector_candidates if module_available(name)]
    requested_hook = os.environ.get("ASTRAKV_STRUCTURED_EVENT_HOOK", "")
    hook_status = "unavailable"
    verification = structured_hook_verification(str(getattr(args, "structured_hook_verification", "") or ""))
    if verification.get("status") == "verified":
        hook_status = "verified"
    elif requested_hook:
        hook_status = "manual_verification_required"
    elif discovered:
        hook_status = "manual_verification_required"
    capacities = {"disk_free_bytes": disk_record(args.disk_path).get("free_bytes", "unknown")}
    experiment = ExperimentManifest(
        run_id=str(getattr(args, "run_id", "unknown") or "unknown"), workload_id=str(getattr(args, "workload_id", "unknown") or "unknown"), workload_path=str(getattr(args, "workload_manifest", "") or ""),
        model=args.model or "unknown", model_revision=str(getattr(args, "model_revision", "unknown") or "unknown"),
        tokenizer_revision=str(getattr(args, "tokenizer_revision", "unknown") or "unknown"), dtype=str(getattr(args, "dtype", "unknown") or "unknown"), quantization=str(getattr(args, "quantization", "unknown") or "unknown"),
        random_seed=str(getattr(args, "random_seed", "unknown") or "unknown"), cache_state=str(getattr(args, "cache_state", "unknown") or "unknown"), capacities=capacities,
        command=args.launch_command or "unknown", connector_version=str(getattr(args, "connector_version", "unknown") or "unknown"),
        input_hashes=input_hashes((str(getattr(args, "workload_manifest", "") or ""), args.lmcache_config)),
    ).to_record()
    return {
        "schema": "astra-dgx-runtime-environment-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
        },
        "packages": packages,
        "cuda": nvidia_smi(),
        "model": args.model,
        "launch_command": args.launch_command,
        "lmcache_config": file_record(args.lmcache_config),
        "disk": disk_record(args.disk_path),
        "experiment_manifest": experiment,
        "connector": {
            "expected_name": "LMCacheConnectorV1",
            "discoverable_modules": discovered,
            "requested_structured_event_hook": requested_hook,
            "structured_event_hook_status": hook_status,
            "structured_hook_verification": verification,
            "claim_boundary": "Package/module discovery is not proof of a stable object-level eviction hook.",
        },
    }


def structured_hook_verification(path: str) -> dict[str, Any]:
    if not path:
        return {"status": "unavailable", "path": ""}
    target = Path(path)
    if not target.exists():
        return {"status": "unavailable", "path": str(target), "reason": "verification_file_missing"}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "path": str(target), "reason": f"invalid_verification_file: {exc}"}
    return {"status": str(payload.get("status") or "unavailable"), "path": str(target), "valid_event_count": payload.get("valid_event_count", 0)}


def package_record(name: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    spec = importlib.util.find_spec(name)
    return {
        "installed": spec is not None,
        "version": version,
        "origin": spec.origin if spec is not None else "",
    }


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "output": ""}
    try:
        output = subprocess.check_output(
            [executable, "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return {"available": True, "output": output.strip()}
    except Exception as exc:  # noqa: BLE001 - preflight should preserve diagnostics.
        return {"available": True, "output": "", "error": f"{type(exc).__name__}: {exc}"}


def file_record(path: str) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    target = Path(path)
    return {"path": str(target), "exists": target.exists()}


def disk_record(path: str) -> dict[str, Any]:
    target = Path(path)
    try:
        usage = shutil.disk_usage(target)
        return {"path": str(target), "total_bytes": usage.total, "free_bytes": usage.free}
    except OSError as exc:
        return {"path": str(target), "error": str(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
