"""Create causal, prefix-level offline decisions for one task-one workload."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl
from astrakv.runtime.profile_db import ProfileDB
from astrakv.runtime.trace_schema import TraceEvent, write_trace_jsonl
from astrakv.scheduler.object_scheduler import ObjectSchedulerConfig, UnifiedObjectScheduler, candidates_from_profile_db


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    workload = load_runtime_workload_jsonl(args.workload_manifest)
    db = ProfileDB()
    events: list[TraceEvent] = []
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_actions: dict[str, str] = {}
    for row in workload:
        candidates = candidates_from_profile_db(db, default_size_bytes=args.default_object_bytes)
        scheduler = UnifiedObjectScheduler(ObjectSchedulerConfig(
            gpu_budget_bytes=args.gpu_capacity_bytes, default_object_bytes=args.default_object_bytes,
        ))
        scheduled = [item.to_record() for item in scheduler.schedule(candidates)]
        for item in scheduled:
            object_key = str(item.get("object_key") or "")
            action = str(item.get("action") or "defer")
            if not object_key or last_actions.get(object_key) == action:
                continue
            last_actions[object_key] = action
            decisions.append(decision_record(args.run_id, item, row, action, int(item.get("size_bytes") or args.default_object_bytes)))
        size = logical_size(row, args.kv_bytes_per_token)
        if row.prefix_id not in last_actions:
            action = "prefetch" if row.prefix_id in seen else "defer"
            last_actions[row.prefix_id] = action
            decisions.append(decision_record(args.run_id, {"object_key": row.prefix_id, "object_level": "prefix"}, row, action, size))
        event_type = "cache_hit" if row.prefix_id in seen else "cache_load"
        event = TraceEvent(
            event_type=event_type, category="kv", source="task1_dataset_logical_access",
            status="modeled", request_id=row.request_id, case=row.case, chunk_id=row.prefix_id,
            tier="gpu", bytes=size, metadata={
                "run_id": args.run_id, "prefix_id": row.prefix_id, "prefix_hash": row.prefix_hash,
                "arrival_index": row.arrival_index, "reuse_ratio": row.reuse_ratio,
                "reuse_bucket": row.reuse_bucket, "legacy_unlinked": False,
                "provenance": "modeled_dataset_metadata", "profile_mode": "causal_same_workload_history",
                "reuse_group_size": row.metadata.get("reuse_group_size"),
            },
        )
        events.append(event)
        db.observe(event, workload_id=args.workload_id)
        seen.add(row.prefix_id)
    trace_path = output / "causal_logical_trace.jsonl"
    db_path = output / "causal_profile_db.json"
    decisions_path = output / "causal_object_schedule_decisions.csv"
    hints_path = output / "causal_object_scheduler_hints.jsonl"
    write_trace_jsonl(events, trace_path)
    db.save(db_path)
    write_csv(decisions_path, decisions)
    with hints_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            if item["action"] == "prefetch":
                handle.write(json.dumps({key: item[key] for key in ("run_id", "request_id", "object_key", "object_level", "arrival_index", "predicted_action")}, ensure_ascii=False) + "\n")
    (output / "causal_profile_manifest.json").write_text(json.dumps({
        "schema": "astrakv-task1-causal-profile-v1", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id, "workload_id": args.workload_id, "workload_manifest": args.workload_manifest,
        "profile_mode": "causal_same_workload_history", "profile_evidence_kind": "modeled_dataset_metadata",
        "kv_bytes_per_token": args.kv_bytes_per_token, "gpu_capacity_bytes": args.gpu_capacity_bytes,
        "decision_count": len(decisions), "future_access_used": False,
        "claim_boundary": "Logical task-one access events are offline evidence, not vLLM/LMCache cache lifecycle events.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Task-one causal profile artifacts written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--gpu-capacity-bytes", type=int, required=True)
    parser.add_argument("--default-object-bytes", type=int, required=True)
    parser.add_argument("--output-dir", default="results/task1_causal_profile")
    return parser.parse_args()


def logical_size(row: Any, kv_bytes_per_token: int) -> int:
    if kv_bytes_per_token <= 0:
        raise SystemExit("--kv-bytes-per-token must be positive")
    return max(1, int(row.context_length or 0) * kv_bytes_per_token)


def decision_record(run_id: str, item: dict[str, Any], row: Any, action: str, size: int) -> dict[str, Any]:
    key = str(item.get("object_key") or row.prefix_id)
    return {
        "run_id": run_id, "request_id": row.request_id, "object_key": key,
        "object_level": str(item.get("object_level") or "prefix"), "arrival_index": row.arrival_index,
        "decision_after_arrival_index": row.arrival_index - 1,
        "profile_history_end_index": row.arrival_index - 1,
        "predicted_action": action, "action": action, "size_bytes": size,
        "size_source": "task1_estimated_context_tokens_x_kv_bytes_per_token",
        "legacy_unlinked": False,
        "metadata": {
            "run_id": run_id, "request_id": row.request_id, "prefix_id": key, "arrival_index": row.arrival_index,
            "reuse_ratio": row.reuse_ratio, "reuse_bucket": row.reuse_bucket, "legacy_unlinked": False,
            "profile_mode": "causal_same_workload_history", "profile_history_end_index": row.arrival_index - 1,
            "task1_modeled_metadata": True,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["run_id", "request_id", "object_key", "object_level", "arrival_index", "decision_after_arrival_index", "profile_history_end_index", "predicted_action", "action", "size_bytes", "size_source", "legacy_unlinked", "metadata"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["metadata"] = json.dumps(item["metadata"], ensure_ascii=False)
            writer.writerow(item)


if __name__ == "__main__":
    raise SystemExit(main())
