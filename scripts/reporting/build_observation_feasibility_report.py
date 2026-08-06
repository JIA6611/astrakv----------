"""Assess whether exact-next candidates are runtime-feasible prefetch targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from astrakv.runtime.prediction_sidecar import PredictorCandidateRecord


DEFAULT_CANDIDATE_REPORT = Path("results/predictor_candidates/predictor_candidate_report.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/observation_feasibility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", default=str(DEFAULT_CANDIDATE_REPORT))
    parser.add_argument("--request-results", required=True)
    parser.add_argument("--runtime-events-raw", required=True)
    parser.add_argument("--runtime-command-receipts", required=True)
    parser.add_argument("--runtime-structured-events", required=True)
    parser.add_argument("--backend-binding-events", default="")
    parser.add_argument("--online-profile-checkpoint", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--minimum-lead-time-ms", type=float, default=250.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = [
        item
        for item in load_candidates(Path(args.candidate_report))
        if item.predicted_class == "exact-next"
    ]
    request_rows = load_jsonl(Path(args.request_results))
    raw_events = load_jsonl(Path(args.runtime_events_raw))
    _ = load_jsonl(Path(args.runtime_command_receipts))
    _ = load_jsonl(Path(args.runtime_structured_events))
    binding_rows = load_jsonl(Path(args.backend_binding_events)) if args.backend_binding_events else []
    profile = (
        json.loads(Path(args.online_profile_checkpoint).read_text(encoding="utf-8"))
        if args.online_profile_checkpoint and Path(args.online_profile_checkpoint).is_file()
        else {}
    )

    requests_by_id = {str(row.get("request_id") or ""): row for row in request_rows}
    requests_by_object = build_request_groups(request_rows)
    binding_by_request = latest_binding_by_request(binding_rows)
    binding_by_object = latest_binding_by_object(binding_rows)
    events_by_object = group_events_by_object(raw_events)

    report_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        report_rows.append(
            analyze_candidate(
                candidate,
                requests_by_id=requests_by_id,
                requests_by_object=requests_by_object,
                binding_by_request=binding_by_request,
                binding_by_object=binding_by_object,
                events_by_object=events_by_object,
                profile=profile,
                minimum_lead_time_ms=args.minimum_lead_time_ms,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "observation_feasibility_report.jsonl"
    csv_path = output_dir / "observation_feasibility_report.csv"
    summary_path = output_dir / "observation_feasibility_summary.json"

    write_jsonl(jsonl_path, report_rows)
    write_csv(csv_path, report_rows)
    summary_path.write_text(
        json.dumps(summarize(report_rows), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Observation feasibility report written to {jsonl_path}")
    print(f"Observation feasibility CSV written to {csv_path}")
    print(f"Observation feasibility summary written to {summary_path}")
    return 0


def analyze_candidate(
    candidate: PredictorCandidateRecord,
    *,
    requests_by_id: dict[str, dict[str, Any]],
    requests_by_object: dict[str, list[dict[str, Any]]],
    binding_by_request: dict[str, dict[str, Any]],
    binding_by_object: dict[tuple[str, str], dict[str, Any]],
    events_by_object: dict[str, list[dict[str, Any]]],
    profile: dict[str, Any],
    minimum_lead_time_ms: float,
) -> dict[str, Any]:
    seed_request = requests_by_id.get(candidate.request_id)
    next_request = next_same_object_request(
        requests_by_object.get(candidate.candidate_object_id, []),
        candidate.request_id,
    )
    binding = (
        binding_by_request.get(candidate.request_id)
        or binding_by_request.get("" if next_request is None else str(next_request.get("request_id") or ""))
        or binding_by_object.get((candidate.object_level.value, candidate.candidate_object_id))
    )
    object_state = profile_object_state(profile, binding)
    if object_state is None:
        object_state = profile_object_state_by_object_key(profile, candidate.candidate_object_id)
    object_exists = (
        bool(events_by_object.get(candidate.candidate_object_id))
        or binding is not None
        or object_state is not None
    )
    lead_time_ms = compute_lead_time_ms(seed_request, next_request)
    tier_before_demand = tier_before_request(
        events_by_object.get(candidate.candidate_object_id, []),
        next_request,
    )
    prefetch_ready = action_ready(binding, "prefetch")
    load_ready = action_ready(binding, "load")
    same_request_target = same_request_load_target(binding)
    pending_state = pending_binding_state(binding)
    active_binding_state = active_binding_state_from(binding, object_state)
    breaker_state = str(((profile.get("controller_state") or {}).get("breaker") or {}).get("state") or "closed")
    prefetch_waste = int((object_state or {}).get("prefetch_waste_count") or 0)
    load_target_state = str((object_state or {}).get("last_load_target_state") or "")
    try:
        load_target_consumed_at_ns = int((object_state or {}).get("last_load_target_consumed_at_ns") or 0)
    except (TypeError, ValueError):
        load_target_consumed_at_ns = 0
    load_target_stale = load_target_state in {"consumed", "unavailable", "missing", "expired"} or load_target_consumed_at_ns > 0

    if not object_exists:
        status = "not_runtime_ready"
        notes = "candidate object was not observed in runtime artifacts"
    elif tier_before_demand != "ssd":
        status = "not_ssd_resident"
        notes = "object was not SSD-resident immediately before the next demand"
    elif not prefetch_ready:
        status = "not_runtime_ready"
        notes = "prefetch action was not ready in the latest execution spec"
    elif same_request_target or active_binding_state or pending_state or breaker_state in {"open", "half_open"}:
        status = "unsafe_binding_state"
        notes = "binding, breaker, or same-request load state makes prefetch unsafe"
    elif lead_time_ms is None or lead_time_ms < minimum_lead_time_ms:
        status = "needs_more_lead_time"
        notes = "measured seed-to-demand gap is smaller than the configured prefetch window"
    else:
        status = "prefetchable_now"
        notes = "exact-next candidate has SSD residency, ready execution spec, and enough lead time"

    return {
        "request_id": candidate.request_id,
        "candidate_object_id": candidate.candidate_object_id,
        "object_level": candidate.object_level.value,
        "predicted_class": candidate.predicted_class,
        "estimated_reusable_tokens": candidate.estimated_reusable_tokens,
        "estimated_kv_bytes": candidate.estimated_kv_bytes,
        "feasibility_status": status,
        "lead_time_ms": lead_time_ms,
        "next_request_id": "" if next_request is None else str(next_request.get("request_id") or ""),
        "object_exists": object_exists,
        "tier_before_demand": tier_before_demand,
        "prefetch_ready": prefetch_ready,
        "load_ready": load_ready,
        "load_target_state": load_target_state,
        "load_target_consumed_at_ns": load_target_consumed_at_ns,
        "load_target_stale": load_target_stale,
        "same_request_load_target": same_request_target,
        "active_binding_state": active_binding_state,
        "pending_binding_state": pending_state,
        "breaker_state": breaker_state,
        "prefetch_waste_count": prefetch_waste,
        "notes": notes,
    }


def build_request_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        object_id = object_id_for_request(row)
        if not object_id:
            continue
        groups.setdefault(object_id, []).append(row)
    for object_id, items in groups.items():
        groups[object_id] = sorted(items, key=request_sort_key)
    return groups


def latest_binding_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("record_type") or "") != "binding":
            continue
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        result[request_id] = row
    return result


def latest_binding_by_object(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("record_type") or "") != "binding":
            continue
        object_key = str(row.get("object_key") or "")
        object_level = str(row.get("object_level") or "")
        if not object_key or not object_level:
            continue
        result[(object_level, object_key)] = row
    return result


def group_events_by_object(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        object_key = str(row.get("object_key") or "")
        if not object_key:
            continue
        grouped.setdefault(object_key, []).append(row)
    for object_key, items in grouped.items():
        grouped[object_key] = sorted(items, key=lambda item: int(item.get("timestamp_ns") or 0))
    return grouped


def next_same_object_request(
    rows: list[dict[str, Any]],
    request_id: str,
) -> dict[str, Any] | None:
    found = False
    for row in rows:
        if found:
            return row
        if str(row.get("request_id") or "") == request_id:
            found = True
    return None


def request_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    try:
        arrival = int(row.get("arrival_index"))
    except (TypeError, ValueError):
        arrival = 1 << 30
    try:
        started = float(row.get("request_started_s"))
    except (TypeError, ValueError):
        started = 0.0
    return (arrival, started)


def compute_lead_time_ms(
    seed_request: dict[str, Any] | None,
    next_request: dict[str, Any] | None,
) -> float | None:
    if seed_request is None or next_request is None:
        return None
    try:
        seed_end = float(seed_request.get("request_ended_s"))
        next_start = float(next_request.get("request_started_s"))
    except (TypeError, ValueError):
        return None
    return max(0.0, (next_start - seed_end) * 1000.0)


def tier_before_request(events: list[dict[str, Any]], request_row: dict[str, Any] | None) -> str:
    if request_row is None:
        return "unknown"
    try:
        started_ns = int(float(request_row.get("request_started_s")) * 1_000_000_000)
    except (TypeError, ValueError):
        return "unknown"
    candidate = "unknown"
    for row in events:
        try:
            timestamp_ns = int(row.get("timestamp_ns") or 0)
        except (TypeError, ValueError):
            continue
        if timestamp_ns > started_ns:
            break
        tier_after = str(row.get("tier_after") or row.get("tier") or "unknown")
        if tier_after:
            candidate = tier_after
    return candidate


def action_ready(binding: dict[str, Any] | None, action_name: str) -> bool:
    execution_spec = {} if binding is None else dict(binding.get("execution_spec") or {})
    actions = execution_spec.get("actions") if isinstance(execution_spec.get("actions"), dict) else {}
    return str((actions.get(action_name) or {}).get("status") or "") == "ready"


def same_request_load_target(binding: dict[str, Any] | None) -> bool:
    if binding is None:
        return False
    execution_spec = dict(binding.get("execution_spec") or {})
    actions = execution_spec.get("actions") if isinstance(execution_spec.get("actions"), dict) else {}
    load_action = actions.get("load") if isinstance(actions.get("load"), dict) else {}
    load_target_id = str(
        load_action.get("load_target_id")
        or (load_action.get("metadata") or {}).get("load_target_id")
        or binding.get("metadata", {}).get("load_target_id")
        or ""
    )
    load_target_reqmeta = str(
        load_action.get("runtime_reqmeta_id")
        or (load_action.get("metadata") or {}).get("runtime_reqmeta_id")
        or binding.get("metadata", {}).get("runtime_reqmeta_id")
        or ""
    )
    owner_reqmeta = str((binding.get("metadata") or {}).get("runtime_reqmeta_id") or "")
    return bool(load_target_id and load_target_reqmeta and owner_reqmeta and load_target_reqmeta == owner_reqmeta)


def pending_binding_state(binding: dict[str, Any] | None) -> bool:
    metadata = {} if binding is None else dict(binding.get("metadata") or {})
    try:
        pending_io = int(metadata.get("pending_io") or 0)
    except (TypeError, ValueError):
        pending_io = 0
    return pending_io > 0


def active_binding_state_from(binding: dict[str, Any] | None, object_state: dict[str, Any] | None) -> bool:
    metadata = {} if binding is None else dict(binding.get("metadata") or {})
    lifecycle = str(metadata.get("lifecycle") or "")
    active_refs = int((object_state or {}).get("active_reference_count") or 0)
    return lifecycle == "active" or active_refs > 0


def object_id_for_request(row: dict[str, Any]) -> str:
    for field in ("cache_key", "prefix_id", "workflow_id"):
        value = str(row.get(field) or "")
        if value:
            return value
    return ""


def profile_object_state(
    profile: dict[str, Any],
    binding: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if binding is None:
        return None
    execution_spec = dict(binding.get("execution_spec") or {})
    backend_object_id = str(
        binding.get("backend_object_id")
        or execution_spec.get("backend_object_id")
        or ""
    )
    objects = profile.get("objects") if isinstance(profile.get("objects"), dict) else {}
    value = objects.get(backend_object_id)
    return value if isinstance(value, dict) else None


def profile_object_state_by_object_key(
    profile: dict[str, Any],
    object_key: str,
) -> dict[str, Any] | None:
    objects = profile.get("objects") if isinstance(profile.get("objects"), dict) else {}
    for value in objects.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("last_object_key") or "") == object_key:
            return value
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("feasibility_status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "astrakv-observation-feasibility-report-v1",
        "record_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
    }


def load_candidates(path: Path) -> list[PredictorCandidateRecord]:
    rows: list[PredictorCandidateRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        rows.append(PredictorCandidateRecord.from_record(record))
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        rows.append(record)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "request_id",
        "candidate_object_id",
        "object_level",
        "predicted_class",
        "estimated_reusable_tokens",
        "estimated_kv_bytes",
        "feasibility_status",
        "lead_time_ms",
        "next_request_id",
        "object_exists",
        "tier_before_demand",
        "prefetch_ready",
        "load_ready",
        "load_target_state",
        "load_target_consumed_at_ns",
        "load_target_stale",
        "same_request_load_target",
        "active_binding_state",
        "pending_binding_state",
        "breaker_state",
        "prefetch_waste_count",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
