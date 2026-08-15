"""Materialize grouped workload prompts into the canonical runtime JSONL contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-prompts-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--interleave",
        action="store_true",
        help=(
            "Round-robin across reuse groups so the same object's visits are "
            "spaced apart. This creates SSD-resident-but-CPU-evicted revisit "
            "windows that Prefetch-A/B can actually act on; without it, all "
            "visits of an object are consecutive and neither strategy fires."
        ),
    )
    parser.add_argument(
        "--interleave-pattern",
        choices=("round-robin", "fire-consume"),
        default="round-robin",
        help=(
            "round-robin: uniform spacing (default).  fire-consume: emit all "
            "first visits, then per object a far revisit (spaced enough that "
            "the CPU hot pool evicts it, so Prefetch-B fires) immediately "
            "followed by a near revisit (so the prefetched CPU copy survives "
            "and is consumed).  fire-consume is required for B to show TTFT "
            "benefit: with a uniform cycle the prefetched copy is always "
            "evicted before the next visit."
        ),
    )
    parser.add_argument(
        "--prefetch-lead-s",
        type=float,
        default=0.0,
        help=(
            "Stamp prefetch_lead_s on every workload row.  With the KV-Core "
            "invalidate-on-lead flag this frees the request's own CPU copy at "
            "ingress so Prefetch-A (arrival fill) can repopulate it from SSD "
            "during the lead window without needing free capacity in a full "
            "CPU pool."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.grouped_prompts_jsonl)
    rows = load_jsonl(input_path)
    ordered = sorted(rows, key=lambda row: int(row.get("order") or 0))
    if args.limit > 0:
        ordered = ordered[: args.limit]
    if args.interleave:
        ordered = (
            fire_consume_groups(ordered)
            if args.interleave_pattern == "fire-consume"
            else interleave_groups(ordered)
        )
    dataset = args.dataset or input_path.parent.name
    task = args.task or dataset
    group_sizes = reuse_group_sizes(ordered)
    exact_prefix_groups = exact_prefix_groups_for(ordered)

    workload_rows: list[RuntimeWorkloadRow] = []
    distance_rows: list[dict[str, Any]] = []
    previous_by_group: dict[str, int] = {}
    for index, row in enumerate(ordered):
        reuse_group = str(row.get("reuse_group") or "")
        request_id = str(row.get("request_id") or f"{dataset}-grouped-{index:06d}")
        group_size = group_sizes.get(reuse_group, 1)
        # In interleave mode the row's original ``order`` no longer reflects
        # the execution sequence; the new enumeration index is authoritative.
        arrival_index = index if args.interleave else int(row.get("order") or index)
        reuse_ratio = 0.0 if group_size <= 1 else (group_size - 1) / group_size
        reuse_bucket = "none" if group_size == 1 else ("medium" if group_size == 2 else "high")
        metadata = {
            "dataset": str(row.get("dataset") or dataset),
            "task": str(row.get("task") or task),
            "workload_type": str(row.get("workload_type") or "grouped"),
            "reuse_group_size": group_size,
            "shared_context": bool(row.get("shared_context", False)),
            "exact_prefix": reuse_group in exact_prefix_groups,
            "context_hash": str(row.get("context_hash") or ""),
            "question_hash": str(row.get("question_hash") or ""),
            "ground_truth": str(row.get("ground_truth") or ""),
            "messages": row.get("messages") if isinstance(row.get("messages"), list) else None,
            "source_prompt_path": str(input_path),
        }
        workload_rows.append(
            RuntimeWorkloadRow(
                request_id=request_id,
                prompt=str(row.get("prompt") or ""),
                prefix_id=reuse_group or request_id,
                prefix_hash=reuse_group or request_id,
                cache_key=reuse_group or request_id,
                arrival_index=arrival_index,
                reuse_ratio=reuse_ratio,
                reuse_bucket=reuse_bucket,
                context_length=estimate_prompt_tokens(str(row.get("prompt") or "")),
                expected_output_tokens=int(row.get("max_tokens") or 128),
                batch_size=1,
                case=f"{dataset}_grouped_exact_next",
                prefetch_lead_s=max(0.0, args.prefetch_lead_s),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )
        prior = previous_by_group.get(reuse_group)
        distance_rows.append(
            {
                "request_id": request_id,
                "candidate_object_id": reuse_group or request_id,
                "arrival_index": arrival_index,
                "previous_same_object_arrival_index": "" if prior is None else prior,
                "reuse_distance": "" if prior is None else arrival_index - prior,
                "group_size": group_size,
                "exact_prefix": reuse_group in exact_prefix_groups,
            }
        )
        previous_by_group[reuse_group] = arrival_index

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = output_dir / f"{dataset}_grouped_exact_next_canonical_workload.jsonl"
    summary_path = output_dir / f"{dataset}_grouped_exact_next_summary.json"
    distance_path = output_dir / f"{dataset}_grouped_exact_next_reuse_distance.csv"

    with workload_path.open("w", encoding="utf-8") as handle:
        for row in workload_rows:
            handle.write(json.dumps(row.to_record(), ensure_ascii=False) + "\n")

    with distance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distance_rows[0]) if distance_rows else ["request_id"])
        writer.writeheader()
        writer.writerows(distance_rows)

    summary_path.write_text(
        json.dumps(
            {
                "schema": "astrakv-grouped-exact-next-workload-v1",
                "dataset": dataset,
                "task": task,
                "request_count": len(workload_rows),
                "reuse_group_count": len(group_sizes),
                "shared_group_count": sum(1 for size in group_sizes.values() if size > 1),
                "exact_prefix_group_count": len(exact_prefix_groups),
                "input_sha256": sha256_file(input_path),
                "canonical_workload": str(workload_path),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Canonical grouped workload written to {workload_path}")
    print(f"Reuse distance report written to {distance_path}")
    print(f"Grouped workload summary written to {summary_path}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        if not all(record.get(field) not in (None, "") for field in ("prompt", "reuse_group")):
            raise ValueError(f"{path}:{line_number} is missing prompt or reuse_group")
        rows.append(record)
    if not rows:
        raise ValueError(f"grouped prompt file is empty: {path}")
    return rows


def interleave_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin rows across reuse groups, preserving intra-group order.

    The grouped source orders every visit of an object consecutively.  A/B
    prefetch need the object to be fully evicted from the CPU hot cache before
    its next visit; interleaving different groups in between creates that
    eviction window while keeping the exact same prompt/object set.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("reuse_group") or "")].append(row)
    # Deterministic group order: first appearance in the original workload.
    group_order: list[str] = []
    for row in rows:
        group = str(row.get("reuse_group") or "")
        if group not in group_order:
            group_order.append(group)
    result: list[dict[str, Any]] = []
    cursor: dict[str, int] = {}
    pending = True
    while pending:
        pending = False
        for group in group_order:
            bucket = buckets[group]
            position = cursor.get(group, 0)
            if position < len(bucket):
                result.append(bucket[position])
                cursor[group] = position + 1
                pending = True
    return result


def fire_consume_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fire-and-consume revisit schedule for Prefetch-B.

    Every reuse group with >=3 rows contributes three visits: the first visit,
    a far revisit and an immediately following near revisit.  The far visit is
    separated from the first visit by N-1 other objects (their combined KV
    exceeds the CPU hot pool, so the object is evicted and Prefetch-B fires at
    the far visit's LMCache load); the near visit follows the far visit with
    zero intervening requests, so the prefetched CPU copy survives and is
    consumed.  Sequence: [all first visits] then [far, near] per group.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("reuse_group") or "")].append(row)
    eligible = [bucket for bucket in buckets.values() if len(bucket) >= 3]
    result: list[dict[str, Any]] = []
    for bucket in eligible:
        result.append(bucket[0])
    for bucket in eligible:
        result.append(bucket[1])
        result.append(bucket[2])
    return result


def reuse_group_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        sizes[str(row.get("reuse_group") or "")] += 1
    return dict(sizes)


def exact_prefix_groups_for(rows: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reuse_group") or "")].append(row)
    result: set[str] = set()
    for reuse_group, items in grouped.items():
        if len(items) <= 1:
            continue
        context_hashes = {
            str(item.get("context_hash") or "")
            for item in items
            if str(item.get("context_hash") or "")
        }
        shared = {bool(item.get("shared_context", False)) for item in items}
        if context_hashes and len(context_hashes) == 1 and shared == {True}:
            result.add(reuse_group)
    return result


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt.split()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
