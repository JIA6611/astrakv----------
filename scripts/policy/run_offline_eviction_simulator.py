"""Replay a canonical workload under LRU/FIFO/AstraKV/Belady tier policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl  # noqa: E402
from astrakv.benchmarks.experiment_manifest import ExperimentManifest, input_hashes  # noqa: E402
from astrakv.prefetch.scorer import ChunkScorer  # noqa: E402
from astrakv.runtime.offline_eviction import (  # noqa: E402
    OfflineAccess, OfflineEvictionSimulator, OfflineObject, OfflinePolicy,
    PrefetchHint, ProxyCostModel, TierCapacities,
)
from astrakv.runtime.profile_db import ProfileDB  # noqa: E402
from astrakv.scheduler.object_scheduler import (  # noqa: E402
    ObjectSchedulerConfig, UnifiedObjectScheduler, candidates_from_profile_db,
)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    workload = load_runtime_workload_jsonl(args.workload_manifest)
    profile = ProfileDB.load(args.profile_db)
    trace_rows = load_jsonl(args.trace)
    benchmark_rows = index_jsonl(args.request_results) if args.request_results else {}
    objects, accesses = build_inputs(workload, trace_rows, profile, benchmark_rows, args.default_object_bytes)
    hints, decision_records = load_or_build_hints(args, profile, objects)
    capacities = TierCapacities(args.gpu_capacity_bytes, args.cpu_capacity_bytes, args.ssd_capacity_bytes)
    costs = ProxyCostModel(args.cpu_to_gpu_ms, args.ssd_to_gpu_ms, args.recompute_ms)
    results = [
        OfflineEvictionSimulator(
            policy=policy, capacities=capacities, cost_model=costs, run_id=args.run_id,
            workload_id=args.workload_id, objects=objects, accesses=accesses,
            prefetch_hints=hints,
        ).run()
        for policy in OfflinePolicy
    ]
    write_jsonl(output / "offline_eviction_events.jsonl", [event.to_record() for result in results for event in result.events])
    write_csv(output / "offline_eviction_policy_summary.csv", [result.metrics for result in results])
    write_jsonl(output / "offline_astrakv_decisions.jsonl", decision_records)
    manifest = build_manifest(args, capacities, costs, workload, objects, accesses, results)
    (output / "offline_eviction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output / "offline_eviction_report.md", args, manifest, results)
    print(f"Offline eviction artifacts written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--profile-db", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-capacity-bytes", type=int, required=True)
    parser.add_argument("--cpu-capacity-bytes", type=int, required=True)
    parser.add_argument("--ssd-capacity-bytes", type=int, required=True)
    parser.add_argument("--request-results", default="")
    parser.add_argument("--scheduler-decisions", default="")
    parser.add_argument("--default-object-bytes", type=int, required=True)
    parser.add_argument("--cpu-to-gpu-ms", type=float, default=2.0)
    parser.add_argument("--ssd-to-gpu-ms", type=float, default=12.0)
    parser.add_argument("--recompute-ms", type=float, default=40.0)
    parser.add_argument(
        "--profile-source", choices=("separate_profiling_run", "causal_same_workload_history", "self_profile"), required=True,
        help="causal_same_workload_history is valid offline evidence but is rejected by the runtime action safety gate.",
    )
    parser.add_argument("--output-dir", default="results/offline_eviction")
    return parser.parse_args()


def build_inputs(
    workload: list[Any], trace_rows: list[dict[str, Any]], profile: ProfileDB,
    benchmark_rows: dict[str, dict[str, Any]], default_object_bytes: int,
) -> tuple[list[OfflineObject], list[OfflineAccess]]:
    trace_by_request: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        request_id = str(row.get("request_id") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if request_id and not metadata.get("legacy_unlinked", False):
            trace_by_request.setdefault(request_id, []).append(row)
    profile_by_object: dict[tuple[str, str], Any] = {}
    for item in profile.chunks.values():
        if item.legacy_unlinked:
            continue
        if item.cache_key:
            profile_by_object[("cache_key", item.cache_key)] = item
        if item.prefix_id:
            profile_by_object[("prefix", item.prefix_id)] = item
    object_info: dict[tuple[str, str], dict[str, Any]] = {}
    accesses: list[OfflineAccess] = []
    for row in workload:
        level, key = ("cache_key", row.cache_key) if row.cache_key else ("prefix", row.prefix_id)
        associated = trace_by_request.get(row.request_id, [])
        byte_values = [as_int(item.get("bytes")) for item in associated if as_int(item.get("bytes")) > 0]
        load_values = [as_float(item.get("latency_ms")) for item in associated if str(item.get("event_type")) == "cache_load"]
        load_values = [value for value in load_values if value is not None]
        profile_item = profile_by_object.get((level, key))
        if byte_values:
            size, source = max(byte_values), "trace_bytes"
        elif profile_item is not None and profile_item.bytes_loaded > 0:
            size, source = int(profile_item.bytes_loaded), "profile_bytes_loaded"
        else:
            size, source = default_object_bytes, "default_object_bytes"
        observed = sum(load_values) / len(load_values) if load_values else (
            profile_item.avg_load_latency_ms if profile_item is not None else None
        )
        info = object_info.setdefault((level, key), {"size": size, "source": source, "observed": observed})
        if source == "trace_bytes" or info["source"] == "default_object_bytes":
            info.update({"size": max(int(info["size"]), size), "source": source, "observed": observed or info["observed"]})
        result = benchmark_rows.get(row.request_id, {})
        accesses.append(OfflineAccess(
            request_id=row.request_id, arrival_index=row.arrival_index, object_key=key, object_level=level,
            size_bytes=int(info["size"]), size_source=str(info["source"]),
            base_ttft_ms=as_float(result.get("ttft_ms")), base_tpot_ms=as_float(result.get("tpot_ms")),
            observed_load_ms=info["observed"],
        ))
    objects = [OfflineObject(key, level, int(info["size"]), str(info["source"]), info["observed"])
               for (level, key), info in sorted(object_info.items())]
    return objects, sorted(accesses, key=lambda item: item.arrival_index)


def load_or_build_hints(args: argparse.Namespace, profile: ProfileDB, objects: list[OfflineObject]) -> tuple[list[PrefetchHint], list[dict[str, Any]]]:
    scores = ChunkScorer().score_db(profile)
    score_lookup = {item.chunk_id: item.score for item in scores}
    by_key = {(item.object_level, item.object_key): item for item in objects}

    def attach_scores(records: list[dict[str, Any]]) -> None:
        for record in records:
            level = str(record.get("object_level") or "prefix")
            key = str(record.get("object_key") or "")
            if (level, key) in by_key:
                current = by_key[(level, key)]
                by_key[(level, key)] = OfflineObject(
                    current.object_key,
                    current.object_level,
                    current.size_bytes,
                    current.size_source,
                    current.observed_load_ms,
                    score_lookup.get(str(key), current.astrakv_score),
                )

    if args.scheduler_decisions:
        records = load_csv(args.scheduler_decisions)
        attach_scores(records)
        objects[:] = list(by_key.values())
        return hints_from_records(records), records
    score_by_chunk = {item.chunk_id: item.to_record() for item in scores}
    scheduler = UnifiedObjectScheduler(ObjectSchedulerConfig(gpu_budget_bytes=args.gpu_capacity_bytes, default_object_bytes=args.default_object_bytes))
    decisions = scheduler.schedule(candidates_from_profile_db(profile, chunk_scores=score_by_chunk, default_size_bytes=args.default_object_bytes))
    records = [item.to_record() for item in decisions if not item.metadata.get("legacy_unlinked", True)]
    updated: list[OfflineObject] = []
    attach_scores(records)
    objects[:] = list(by_key.values())
    return hints_from_records(records), records


def hints_from_records(records: list[dict[str, Any]]) -> list[PrefetchHint]:
    hints: list[PrefetchHint] = []
    for record in records:
        if str(record.get("action") or "") != "prefetch" or truthy(record.get("legacy_unlinked")):
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        key = str(record.get("object_key") or metadata.get("prefix_id") or record.get("cache_key") or "")
        level = str(record.get("object_level") or ("cache_key" if record.get("cache_key") else "prefix"))
        index = as_int(record.get("arrival_index") or metadata.get("arrival_index"))
        if key and index >= 0:
            hints.append(PrefetchHint(index, key, level, str(record.get("request_id") or "")))
    return hints


def build_manifest(args: argparse.Namespace, capacities: TierCapacities, costs: ProxyCostModel, workload: list[Any], objects: list[OfflineObject], accesses: list[OfflineAccess], results: list[Any]) -> dict[str, Any]:
    experiment = ExperimentManifest(
        run_id=args.run_id, workload_id=args.workload_id, workload_path=args.workload_manifest,
        workload_sha256=sha256_file(args.workload_manifest), cache_state="unknown",
        capacities=capacities.to_record(), command="run_offline_eviction_simulator.py",
        input_hashes=input_hashes((args.workload_manifest, args.trace, args.profile_db, args.request_results, args.scheduler_decisions)),
    ).to_record()
    return {
        "schema": "astrakv-offline-eviction-v1",
        "simulation_status": "valid",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "workload_id": args.workload_id,
        "workload_manifest": str(Path(args.workload_manifest)),
        "workload_sha256": sha256_file(args.workload_manifest),
        "trace": str(Path(args.trace)),
        "trace_sha256": sha256_file(args.trace),
        "profile_db": str(Path(args.profile_db)),
        "profile_db_sha256": sha256_file(args.profile_db),
        "profile_source": args.profile_source,
        "self_profile_leakage": args.profile_source == "self_profile",
        "capacities": capacities.to_record(),
        "cost_model": {**costs.to_record(), "all_costs_are_proxies": True},
        "request_results": str(Path(args.request_results)) if args.request_results else "",
        "scheduler_decisions": str(Path(args.scheduler_decisions)) if args.scheduler_decisions else "generated_from_profile_db",
        "object_count": len(objects),
        "access_count": len(accesses),
        "legacy_unlinked_in_denominator": False,
        "policies": [result.metrics for result in results],
        "experiment_manifest": experiment,
    }


def write_report(path: Path, args: argparse.Namespace, manifest: dict[str, Any], results: list[Any]) -> None:
    lines = ["# Offline Eviction Policy Report", "", "## Evidence Boundary", "", "All transfer and timing values are offline proxies. This report does not prove a vLLM/LMCache runtime action.", "", "## Inputs", "", f"- Workload: `{args.workload_manifest}`", f"- Trace: `{args.trace}`", f"- Profile source: `{args.profile_source}`", f"- Timing mode: `{next(iter(result.metrics for result in results))['timing_mode']}`", "", "## Policy Comparison", "", "| policy | hit rate | migration bytes | SSD read proxy | SSD write proxy | prefetch waste | OOM unavoided | TTFT proxy ms | TPOT proxy ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        row = result.metrics
        lines.append(f"| {row['policy']} | {row['total_hit_rate']:.4f} | {row['migration_bytes']} | {row['ssd_read_proxy_bytes']} | {row['ssd_write_proxy_bytes']} | {row['prefetch_waste']} | {row['oom_unavoided']} | {row['ttft_proxy_ms_mean']:.4f} | {fmt(row['tpot_proxy_ms_mean'])} |")
    lines.extend(["", "## Artifact Manifest", "", f"- Schema: `{manifest['schema']}`", f"- Workload SHA-256: `{manifest['workload_sha256']}`", f"- Self-profile leakage: `{manifest['self_profile_leakage']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def index_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("request_id")): row for row in load_jsonl(path) if row.get("request_id")}


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(str(row["metadata"]))
            except json.JSONDecodeError:
                row["metadata"] = {}
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def truthy(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
