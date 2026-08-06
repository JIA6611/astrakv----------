"""Normalize read-only vLLM/LMCache artifacts into runtime eviction events.

The output is intentionally observational.  Parsed server-log offload/evict
messages have ``provenance=log_heuristic`` and cannot be used to claim real
runtime eviction agreement.  A future version-locked adapter may produce
``runtime_structured`` events only after a public backend hook is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.eviction import (  # noqa: E402
    ObjectLevel,
    RuntimeEvictionEvent,
    VllmLmCacheArtifactAdapter,
    write_runtime_events_jsonl,
)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request_objects = load_request_objects(args.request_results, args.workload_manifest)
    adapter = VllmLmCacheArtifactAdapter(run_id=args.run_id, request_objects=request_objects)
    events = adapter.collect_from_paths(server_logs=args.server_log, cache_events=args.cache_events)
    structured_events = load_verified_structured_events(args, request_objects)
    events.extend(structured_events)
    write_runtime_events_jsonl(output, events)
    write_manifest(
        output.with_suffix(".manifest.json"),
        args,
        adapter.describe(),
        request_objects,
        events,
        adapter.last_skipped_evidence,
        structured_events,
    )
    print(f"Normalized runtime eviction events written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-log", action="append", default=[], help="Raw vLLM/LMCache server log; repeatable.")
    parser.add_argument("--cache-events", action="append", default=[], help="Existing cache_events.jsonl; repeatable.")
    parser.add_argument("--request-results", default="", help="Benchmark request_results.jsonl with prefix metadata.")
    parser.add_argument("--workload-manifest", default="", help="Workload JSONL fallback for request metadata.")
    parser.add_argument("--structured-events", default="", help="JSONL exported by a verified public runtime hook.")
    parser.add_argument("--structured-hook-verification", default="", help="Verified JSON from verify_structured_eviction_hook.py.")
    parser.add_argument("--output", default="results/runtime_eviction_events.jsonl")
    return parser.parse_args()


def load_verified_structured_events(args: argparse.Namespace, request_objects: dict[str, dict[str, Any]]) -> list[RuntimeEvictionEvent]:
    if not args.structured_events:
        return []
    if not args.structured_hook_verification:
        raise SystemExit("--structured-events requires --structured-hook-verification")
    try:
        verification = json.loads(Path(args.structured_hook_verification).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid structured hook verification: {exc}") from exc
    if verification.get("status") != "verified" or verification.get("run_id") != args.run_id:
        raise SystemExit("structured events are not eligible: hook verification must be verified for the same run_id")
    if verification.get("events_sha256") != sha256_file(args.structured_events):
        raise SystemExit("structured events are not eligible: verification hash does not match the supplied event file")
    events: list[RuntimeEvictionEvent] = []
    for ordinal, line in enumerate(Path(args.structured_events).read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            level = ObjectLevel(str(row["object_level"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        request_id = str(row.get("request_id") or "")
        object_key = str(row.get("object_key") or "")
        if not request_id or not object_key or str(row.get("run_id") or "") != args.run_id:
            continue
        action = {"cache_offload": "offload", "cache_evict": "evict"}.get(str(row.get("action") or row.get("event_type") or ""), str(row.get("action") or ""))
        timestamp = row.get("timestamp_ns", row.get("timestamp"))
        try:
            timestamp_ns = int(timestamp)
        except (TypeError, ValueError):
            timestamp_ns = None
        events.append(RuntimeEvictionEvent(
            run_id=args.run_id, runtime_event_id=str(row.get("runtime_event_id") or f"structured-{ordinal}"),
            request_id=request_id, object_key=object_key, object_level=level, actual_action=action,
            tier_before=str(row.get("tier_before") or "unknown"), tier_after=str(row.get("tier_after") or row.get("tier") or "unknown"),
            bytes=as_int(row.get("bytes")), timestamp_ns=timestamp_ns,
            arrival_index=as_int((request_objects.get(request_id) or {}).get("arrival_index")),
            status=str(row.get("status") or "observed"), provenance="runtime_structured",
            metadata={"source": args.structured_events, "hook_verification": args.structured_hook_verification},
        ))
    return events


def load_request_objects(*paths: str) -> dict[str, dict[str, Any]]:
    """Load stable request-to-logical-object metadata, first source wins only for blanks."""

    result: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            request_id = str(row.get("request_id") or "")
            if not request_id:
                continue
            current = result.setdefault(request_id, {})
            for key in ("prefix_id", "prefix_hash", "cache_key", "case", "arrival_index", "reuse_ratio", "reuse_bucket"):
                value = row.get(key)
                if value not in (None, "") and current.get(key) in (None, ""):
                    current[key] = value
    return result


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    adapter: dict[str, Any],
    request_objects: dict[str, dict[str, Any]],
    events: list[Any],
    skipped_evidence: list[dict[str, Any]],
    structured_events: list[Any],
) -> None:
    provenance_counts = Counter(str(event.provenance) for event in events)
    path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "adapter": adapter,
                "inputs": {
                    "server_logs": args.server_log,
                    "cache_events": args.cache_events,
                    "request_results": args.request_results,
                    "workload_manifest": args.workload_manifest,
                },
                "request_object_mapping_count": len(request_objects),
                "normalized_event_count": len(events),
                "unmatched_eviction_evidence_count": len(skipped_evidence),
                "unmatched_eviction_evidence": skipped_evidence,
                "provenance_counts": dict(provenance_counts),
                "structured_event_count": len(structured_events),
                "ground_truth_status": "valid" if structured_events else "insufficient_ground_truth",
                "ground_truth_reason": "verified public structured hook events were normalized" if structured_events else "artifact adapter emits log_heuristic observations only",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def as_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
