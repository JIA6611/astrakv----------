#!/usr/bin/env python3
"""Build the fixed single-prefix workload for exploratory E3-P runs.

The workload is intentionally narrow: one request populates the exact native
LMCache prefix, then identical target requests revisit it.  A target's
``prefetch_lead_s`` is consumed after its authenticated request context has
been published and before its HTTP request is sent.  This gives the active
variant a real SSD-to-LocalCPUBackend overlap window without issuing a warmup
HTTP request or pre-writing vLLM GPU KV.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import (
    RuntimeWorkloadRow,
    load_runtime_workload_jsonl,
)


WORKLOAD_NAME = "e3_prefetch_single_exact_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-request-id", default="")
    parser.add_argument("--revisits", type=int, default=16)
    parser.add_argument("--prefetch-lead-s", type=float, default=0.25)
    parser.add_argument("--output-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.revisits < 1:
        raise SystemExit("--revisits must be positive")
    if args.prefetch_lead_s <= 0.0:
        raise SystemExit("--prefetch-lead-s must be positive")
    if args.output_tokens < 1:
        raise SystemExit("--output-tokens must be positive")
    source_path = Path(args.source_workload)
    source = select_source(source_path, args.source_request_id)
    rows = materialize(
        source,
        revisits=args.revisits,
        prefetch_lead_s=args.prefetch_lead_s,
        output_tokens=args.output_tokens,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = output_dir / f"{WORKLOAD_NAME}.jsonl"
    workload.write_text(
        "".join(json.dumps(row.to_record(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "astrakv-e3-prefetch-workload-v1",
        "workload": WORKLOAD_NAME,
        "source_workload": str(source_path),
        "source_workload_sha256": sha256_file(source_path),
        "source_request_id": source.request_id,
        "source_prompt_sha256": hashlib.sha256(source.prompt.encode()).hexdigest(),
        "workload_sha256": sha256_file(workload),
        "seed_requests": 1,
        "identical_revisits": args.revisits,
        "prefetch_lead_s": args.prefetch_lead_s,
        "output_tokens": args.output_tokens,
        "performance_scope": "exploratory_prefetch_only",
    }
    (output_dir / "e3_prefetch_workload.manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def select_source(path: Path, requested_id: str) -> RuntimeWorkloadRow:
    raw_metadata: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                metadata = item.get("metadata")
                raw_metadata[str(item.get("request_id") or "")] = (
                    metadata if isinstance(metadata, dict) else {}
                )
    candidates = [
        row for row in load_runtime_workload_jsonl(path)
        if raw_metadata.get(row.request_id, {}).get("exact_prefix") is True
        and isinstance(raw_metadata.get(row.request_id, {}).get("messages"), list)
        and row.context_length and row.context_length > 0
        and row.cache_key and row.prefix_hash
    ]
    if requested_id:
        candidates = [row for row in candidates if row.request_id == requested_id]
    if not candidates:
        raise ValueError("no eligible exact-prefix source row with messages and cache identity")
    return min(candidates, key=lambda row: row.arrival_index)


def materialize(
    source: RuntimeWorkloadRow,
    *,
    revisits: int,
    prefetch_lead_s: float,
    output_tokens: int,
) -> list[RuntimeWorkloadRow]:
    messages = source.metadata.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("source metadata.messages is required")
    if revisits < 1 or prefetch_lead_s <= 0.0 or output_tokens < 1:
        raise ValueError("invalid E3-P workload parameters")
    digest = hashlib.sha256(source.request_id.encode()).hexdigest()[:12]
    rows: list[RuntimeWorkloadRow] = []
    for index in range(revisits + 1):
        role = "seed" if index == 0 else "revisit"
        request_id = (
            f"e3-prefetch-{digest}-seed"
            if index == 0
            else f"e3-prefetch-{digest}-revisit-{index:02d}"
        )
        metadata = copy.deepcopy(source.metadata)
        metadata.update({
            "workload_type": WORKLOAD_NAME,
            "scenario": "e3_prefetch_off_vs_ssd_to_cpu",
            "e3_prefetch_role": role,
            "sample_id": request_id,
            "source_request_id": source.request_id,
            "exact_prefix": True,
            "generation_seed": 0,
        })
        rows.append(RuntimeWorkloadRow(
            request_id=request_id,
            prompt=source.prompt,
            prefix_id=source.prefix_id,
            prefix_hash=source.prefix_hash,
            cache_key=source.cache_key,
            arrival_index=index,
            reuse_ratio=0.0 if index == 0 else 1.0,
            reuse_bucket="none" if index == 0 else "high",
            context_length=source.context_length,
            expected_output_tokens=output_tokens,
            batch_size=1,
            prefetch_lead_s=0.0 if index == 0 else prefetch_lead_s,
            case=f"e3_prefetch_{role}",
            metadata=metadata,
        ))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
