"""Verify externally exported public runtime eviction events without mutating it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED = ("run_id", "request_id", "object_key", "object_level", "status")
VALID_LEVELS = {"prefix", "cache_key", "block"}
VALID_ACTIONS = {"cache_offload", "cache_evict", "offload", "evict"}
SUCCESS = {"completed", "ok", "executed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="JSONL directly exported by a public backend hook")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="results/structured_hook_verification.json")
    parser.add_argument("--backend-version", default="unknown")
    parser.add_argument("--connector-version", default="unknown")
    args = parser.parse_args()
    valid, invalid = validate(Path(args.events), args.run_id)
    result = {
        "schema": "astrakv-structured-hook-verification-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events": str(Path(args.events)),
        "events_sha256": sha256_file(args.events),
        "run_id": args.run_id,
        "backend_version": args.backend_version,
        "connector_version": args.connector_version,
        "status": "verified" if valid and not invalid else "rejected",
        "valid_event_count": len(valid),
        "invalid_event_count": len(invalid),
        "invalid_examples": invalid[:20],
        "claim_boundary": "Verification checks event shape only; it does not add a vLLM/LMCache action API.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Structured hook verification {result['status']}: {output}")
    return 0 if result["status"] == "verified" else 2


def validate(path: Path, run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": line_number, "reason": f"invalid_json: {exc}"})
            continue
        errors = event_errors(row, run_id)
        if errors:
            invalid.append({"line": line_number, "reason": "; ".join(errors)})
        else:
            valid.append(row)
    if not valid and not invalid:
        invalid.append({"line": 0, "reason": "no events supplied"})
    return valid, invalid


def event_errors(row: Any, run_id: str) -> list[str]:
    if not isinstance(row, dict):
        return ["row is not an object"]
    errors = [f"missing {key}" for key in REQUIRED if row.get(key) in (None, "")]
    if str(row.get("run_id") or "") != run_id:
        errors.append("run_id does not match requested run")
    action = str(row.get("action") or row.get("event_type") or "")
    if action not in VALID_ACTIONS:
        errors.append("action is not cache_offload/cache_evict")
    if str(row.get("object_level") or "") not in VALID_LEVELS:
        errors.append("object_level is invalid")
    if str(row.get("status") or "") not in SUCCESS:
        errors.append("status is not successful")
    if row.get("timestamp") in (None, "") and row.get("timestamp_ns") in (None, ""):
        errors.append("missing timestamp")
    return errors


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
