"""Build an astra-trace-v1 profile source from KV-Core runtime artifacts.

The benchmark's own ``trace_events.jsonl`` is only a per-request stub, and the
server-log cache-event extractor loses chunk identity.  This converter instead
joins the KV-Core runtime artifacts that DO carry identity:

  - ``kv_core_native_callbacks.jsonl``   scheduler_exact_lookup hit signals
                                        (per runtime request id)
  - ``request_context_associations.jsonl``  logical <-> runtime request ids
  - canonical workload JSONL              logical request id -> prefix sha256
  - ``backend_binding_events.jsonl``     cache_store/release events (logical id)
  - ``runtime_command_receipts.jsonl``   prefetch receipts (metadata.object_key)

The emitted stream is ``astra-trace-v1`` with ``chunk_id`` = the canonical
prefix object id, which is what the controller's offline ProfileDB lookup
uses (see ``OnlinePolicyController._offline_profile_for``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.trace_schema import TRACE_SCHEMA_VERSION, TraceEvent  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _event(
    *,
    event_type: str,
    category: str,
    source: str,
    status: str,
    request_id: str = "",
    chunk_id: str = "",
    cache_key: str = "",
    tier: str = "unknown",
    bytes: int | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str = "",
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        category=category,
        source=source,
        status=status,
        event_id=uuid4().hex,
        run_id=run_id,
        request_id=request_id,
        chunk_id=chunk_id,
        cache_key=cache_key or chunk_id,
        tier=tier,
        bytes=bytes,
        latency_ms=latency_ms,
        metadata=dict(metadata or {}),
    )


def build_events(args: argparse.Namespace) -> list[TraceEvent]:
    run_id = args.run_id
    canonical = {
        str(row.get("request_id") or ""): str(
            row.get("cache_key") or row.get("prefix_id") or row.get("prefix_hash") or ""
        )
        for row in _load_jsonl(Path(args.workload_manifest))
    }
    logical_to_runtime: dict[str, str] = {}
    runtime_to_logical: dict[str, str] = {}
    for row in _load_jsonl(Path(args.associations)):
        logical = str(row.get("request_id") or "")
        runtime = str(row.get("runtime_request_id") or "")
        if logical and runtime:
            logical_to_runtime[logical] = runtime
            runtime_to_logical[runtime] = logical

    events: list[TraceEvent] = []

    # Per-request benchmark events from the canonical manifest.
    for row in _load_jsonl(Path(args.workload_manifest)):
        request_id = str(row.get("request_id") or "")
        chunk_id = canonical.get(request_id, "")
        events.append(_event(
            event_type="request_result",
            category="request",
            source="canonical_workload",
            status="ok",
            request_id=request_id,
            chunk_id=chunk_id,
            run_id=run_id,
            metadata={
                "context_length": row.get("context_length"),
                "reuse_ratio": row.get("reuse_ratio"),
                "arrival_index": row.get("arrival_index"),
            },
        ))

    # Cache store / release events from the binding registry (logical id).
    for row in _load_jsonl(Path(args.binding_events)):
        action = str(row.get("action") or "")
        status = str(row.get("status") or "")
        request_id = str(row.get("request_id") or "")
        chunk_id = canonical.get(request_id, "")
        if action == "cache_store" and status in {"submitted", "completed"}:
            metadata = dict(row.get("metadata") or {})
            size_bytes = row.get("bytes")
            if size_bytes is None:
                try:
                    size_bytes = int(metadata.get("size_bytes") or 0) or None
                except (TypeError, ValueError):
                    size_bytes = None
            events.append(_event(
                event_type="cache_store",
                category="kv",
                source="binding_registry",
                status="completed" if status == "completed" else "observed",
                request_id=request_id,
                chunk_id=chunk_id,
                tier=str(row.get("tier_after") or "cpu"),
                bytes=size_bytes,
                run_id=run_id,
                metadata={
                    "backend_object_id": str(row.get("backend_object_id") or ""),
                    "verified_backend_hook": True,
                },
            ))

    # Exact-lookup hit/miss signals from the scheduler callbacks (runtime id).
    for row in _load_jsonl(Path(args.native_callbacks)):
        if str(row.get("callback") or "") != "scheduler_exact_lookup":
            continue
        runtime_request_id = str(row.get("request_id") or "")
        logical = runtime_to_logical.get(runtime_request_id, "")
        chunk_id = canonical.get(logical, "")
        locally_cached = max(0, int(row.get("locally_cached_tokens") or 0))
        lookup_hit = max(0, int(row.get("lookup_hit_tokens") or 0))
        # ``lookup_hit_tokens`` is the LMCache-external hit AFTER subtracting
        # vLLM's locally computed tokens (available_hit = hit - local).  With
        # vLLM prefix caching OFF, revisits have locally_cached=0 but
        # lookup_hit>0 (LMCache supplied the prefix); with it ON, the opposite
        # happens.  Either signal means LMCache provided the prefix, so the
        # profile must count the union as a cache hit.
        lmcache_hit = locally_cached + lookup_hit
        events.append(_event(
            event_type="cache_lookup",
            category="kv",
            source="scheduler_exact_lookup",
            status="observed",
            request_id=logical or runtime_request_id,
            chunk_id=chunk_id,
            tier="cpu" if locally_cached > 0 else "ssd",
            run_id=run_id,
            metadata={
                "locally_cached_tokens": locally_cached,
                "lookup_hit_tokens": lookup_hit,
                "lmcache_hit_tokens": lmcache_hit,
                "physical_object_id": str(row.get("physical_object_id") or ""),
            },
        ))
        if chunk_id:
            events.append(_event(
                event_type="cache_hit" if lmcache_hit > 0 else "cache_miss",
                category="kv",
                source="scheduler_exact_lookup",
                status="observed",
                request_id=logical or runtime_request_id,
                chunk_id=chunk_id,
                tier="cpu" if lmcache_hit > 0 else "ssd",
                run_id=run_id,
                metadata={
                    "locally_cached_tokens": locally_cached,
                    "lookup_hit_tokens": lookup_hit,
                    "lmcache_hit_tokens": lmcache_hit,
                },
            ))

    # Prefetch receipts (metadata.object_key carries the canonical id).
    for row in _load_jsonl(Path(args.prefetch_receipts)):
        if str(row.get("action") or "") != "prefetch":
            continue
        metadata = dict(row.get("metadata") or {})
        chunk_id = str(metadata.get("object_key") or "")
        status = str(row.get("status") or "")
        events.append(_event(
            event_type="prefetch",
            category="prefetch",
            source="runtime_command_receipts",
            status=status,
            request_id=str(row.get("request_id") or ""),
            chunk_id=chunk_id,
            tier="cpu",
            bytes=row.get("bytes") if isinstance(row.get("bytes"), int) else None,
            run_id=run_id,
            metadata={
                "prefetched": int(metadata.get("prefetched") or 0),
                "failure_reason": str(metadata.get("failure_reason") or ""),
                "backend_object_id": str(row.get("backend_object_id") or ""),
            },
        ))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-callbacks", required=True)
    parser.add_argument("--associations", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--binding-events", required=True)
    parser.add_argument("--prefetch-receipts", required=True)
    parser.add_argument("--run-id", default="train-trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    events = build_events(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")
    print(f"Trace events written to {output} ({len(events)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
