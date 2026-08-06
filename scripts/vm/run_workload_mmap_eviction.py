"""Execute ProfileDB scheduler eviction decisions through the mmap VM PoC.

This is a standalone OS virtual-memory experiment.  It uses the same workload
manifest identifiers as endpoint observation, but it does not access or claim
to evict vLLM/LMCache KV tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.eviction import (  # noqa: E402
    MMapEvictionAdapter,
    ObjectLevel,
    offline_decision_from_record,
    write_runtime_events_jsonl,
)
from astrakv.runtime.offline_safety import GatedRuntimeAdapter, load_gate  # noqa: E402
from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl  # noqa: E402


def main() -> int:
    args = parse_args()
    gate = load_gate(args.offline_safety_gate)
    if not gate.result.allowed:
        raise SystemExit("offline safety gate rejected VM-PoC action: " + "; ".join(gate.result.reasons))
    if not sys.platform.startswith("linux"):
        raise SystemExit("mmap VM-PoC execution requires Linux madvise/mincore; use the observational path on this host")

    # Keep Linux-only imports out of P0 import paths.
    import numpy as np

    from astrakv.kv_cache.metadata import KVChunkMeta
    from astrakv.vm.dgx_spark_adapter import DgxSparkKVAdapter, DgxSparkKVAdapterConfig

    decisions = [
        offline_decision_from_record(row, run_id=args.run_id, ordinal=index)
        for index, row in enumerate(read_csv(args.offline_decisions), start=1)
    ]
    decisions = [item for item in decisions if item.predicts_eviction]
    workload_objects = load_workload_objects(args.workload_manifest)
    bindings: dict[tuple[ObjectLevel, str], str] = {}
    for decision in decisions:
        key = (decision.object_level, decision.object_key)
        if key in workload_objects:
            bindings.setdefault(key, f"poc-{decision.object_level.value}-{len(bindings):06d}")

    blocks_per_object = max(1, math.ceil(args.object_bytes / args.block_size_bytes))
    required_blocks = len(bindings) * blocks_per_object
    if required_blocks > args.total_blocks:
        raise SystemExit(f"need {required_blocks} mmap blocks for the bound logical objects, but --total-blocks is {args.total_blocks}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = DgxSparkKVAdapterConfig(
        backing_file=args.backing_file,
        total_blocks=args.total_blocks,
        block_size_bytes=args.block_size_bytes,
        keep_backing_file=args.keep_backing_file,
    )
    events = []
    with DgxSparkKVAdapter(config) as kv_adapter:
        for (level, object_key), chunk_id in bindings.items():
            record = kv_adapter.register_chunk(KVChunkMeta(
                request_id=object_key,
                layer_id=0,
                start_token=0,
                end_token=1,
                chunk_id=chunk_id,
                size_bytes=args.object_bytes,
                cache_key=object_key if level == ObjectLevel.CACHE_KEY else None,
                metadata={"logical_object_key": object_key, "logical_object_level": level.value},
            ))
            item_count = math.ceil(record.size_bytes / 2)
            kv_adapter.write_chunk(record.chunk.chunk_id, np.zeros(item_count, dtype=np.float16))

        adapter = GatedRuntimeAdapter(
            MMapEvictionAdapter(kv_adapter, run_id=args.run_id, object_bindings=bindings), gate
        )
        for decision in decisions:
            result = adapter.apply_hint(decision)
            if result.event is not None:
                events.append(result.event)

    write_runtime_events_jsonl(output, events)
    write_manifest(output.with_suffix(".manifest.json"), args, decisions, bindings, events)
    print(f"mmap VM-PoC eviction events written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-decisions", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backing-file", required=True)
    parser.add_argument("--offline-safety-gate", required=True, help="Accepted offline_safety_gate.json required before VM actions.")
    parser.add_argument("--output", default="results/mmap_eviction_events.jsonl")
    parser.add_argument("--total-blocks", type=int, default=1024)
    parser.add_argument("--block-size-bytes", type=int, default=4096)
    parser.add_argument("--object-bytes", type=int, default=4096)
    parser.add_argument("--keep-backing-file", action="store_true")
    return parser.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_workload_objects(path: str) -> set[tuple[ObjectLevel, str]]:
    objects: set[tuple[ObjectLevel, str]] = set()
    for row in load_runtime_workload_jsonl(path):
        prefix_id = row.prefix_id
        cache_key = row.cache_key
        if prefix_id:
            objects.add((ObjectLevel.PREFIX, prefix_id))
        if cache_key:
            objects.add((ObjectLevel.CACHE_KEY, cache_key))
    return objects


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    decisions: list[Any],
    bindings: dict[tuple[ObjectLevel, str], str],
    events: list[Any],
) -> None:
    path.write_text(json.dumps({
        "run_id": args.run_id,
        "execution_scope": "vm_poc_execution",
        "claim_boundary": "standalone mmap VM PoC; not vLLM/LMCache KV eviction",
        "offline_decision_count": len(decisions),
        "bound_logical_object_count": len(bindings),
        "execution_event_count": len(events),
        "successful_execution_count": sum(1 for event in events if event.status == "executed"),
        "failed_execution_count": sum(1 for event in events if event.status == "failed"),
        "workload_manifest": args.workload_manifest,
        "offline_decisions": args.offline_decisions,
        "offline_safety_gate": args.offline_safety_gate,
        "bindings": [
            {"object_level": level.value, "object_key": key, "mmap_chunk_id": chunk_id}
            for (level, key), chunk_id in sorted(bindings.items(), key=lambda item: (item[0][0].value, item[0][1]))
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
