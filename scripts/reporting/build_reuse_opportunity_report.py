"""Build an auditable raw-workload reuse-pilot decision report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import yaml


DECISIONS = frozenset(
    {
        "raw_workload_selected",
        "raw_workload_selected_with_composed_stress",
        "observation_incomplete",
    }
)


def build_reuse_opportunity_report(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize modeled reuse opportunities without inferring backend behavior."""
    slots = config.get("required_dataset_slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("required_dataset_slots must be a non-empty list")
    sampling = _normalize_sampling(config.get("sampling"))

    raw_workloads: list[dict[str, Any]] = []
    missing_slots: list[dict[str, str]] = []
    for slot in slots:
        if not isinstance(slot, dict) or not isinstance(slot.get("slot"), str):
            raise ValueError("every dataset slot must have a string slot name")
        observation = slot.get("raw_observation")
        if observation:
            dataset_id = slot.get("dataset_id")
            source_id = slot.get("source_id")
            if not isinstance(dataset_id, str) or not dataset_id or not isinstance(source_id, str) or not source_id:
                raise ValueError("observed dataset slots require dataset_id and source_id")
            raw_workloads.append(
                summarize_observation_file(
                    observation,
                    dataset_id=dataset_id,
                    source_id=source_id,
                    namespace="raw",
                )
            )
            continue
        missing_slots.append(
            {
                "slot": slot["slot"],
                "reason": str(slot.get("reason") or slot.get("status") or "observation not provided"),
            }
        )

    composed_stress = []
    for item in config.get("composed_stress", []):
        if not isinstance(item, dict):
            raise ValueError("composed_stress entries must be objects")
        dataset_id = item.get("dataset_id")
        source_id = item.get("source_id")
        observation = item.get("observation")
        if not all(isinstance(value, str) and value for value in (dataset_id, source_id, observation)):
            raise ValueError("composed_stress entries require dataset_id, source_id, and observation")
        composed_stress.append(
            summarize_observation_file(
                observation,
                dataset_id=dataset_id,
                source_id=source_id,
                namespace="composed_stress",
            )
        )

    if missing_slots:
        decision = "observation_incomplete"
        decision_reason = "required semantic-independent dataset observations are unavailable"
    elif composed_stress:
        decision = "raw_workload_selected_with_composed_stress"
        decision_reason = "raw workloads remain primary; declared composed stress is reported separately"
    else:
        decision = "raw_workload_selected"
        decision_reason = "all required raw dataset observations are available"

    return {
        "schema": "astrakv-reuse-opportunity-report-v1",
        "evidence_class": "modeled_dataset_metadata",
        "decision": decision,
        "decision_reason": decision_reason,
        "raw_workloads": raw_workloads,
        "composed_stress_workloads": composed_stress,
        "missing_required_slots": missing_slots,
        "sampling": sampling,
        "selection_guardrails": {
            "raw_and_composed_aggregates_separate": True,
            "backend_cache_observations_included": False,
            "manual_request_selection_allowed": False,
        },
    }


def _normalize_sampling(value: Any) -> dict[str, Any]:
    defaults = {
        "seed": None,
        "smoke_request_count": 10,
        "pilot_request_count": 50,
        "selection_method": "published_arrival_order_prefix",
    }
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ValueError("sampling must be an object")
    sampling = {**defaults, **value}
    if not isinstance(sampling["seed"], int):
        raise ValueError("sampling seed must be an integer")
    for field in ("smoke_request_count", "pilot_request_count"):
        if not isinstance(sampling[field], int) or sampling[field] <= 0:
            raise ValueError(f"sampling {field} must be a positive integer")
    if sampling["selection_method"] != "published_arrival_order_prefix":
        raise ValueError("sampling selection_method must preserve published arrival order")
    return sampling


def summarize_observation_file(
    observation_path: str | Path,
    *,
    dataset_id: str,
    source_id: str,
    namespace: str,
) -> dict[str, Any]:
    path = Path(observation_path)
    records = list(_read_observations(path))
    if not records:
        raise ValueError(f"observation file is empty: {path}")
    if any(record["dataset_id"] != dataset_id for record in records):
        raise ValueError(f"observation dataset_id does not match configured dataset: {dataset_id}")

    token_counts = [record["token_count"] for record in records]
    block_hashes = [block for record in records for block in record["block_hashes"]]
    reused_tokens = sum(record["historical_reused_tokens"] for record in records)
    potential_kv_bytes = sum(record["potential_kv_bytes"] for record in records)
    reuse_histogram: dict[str, int] = {}
    for record in records:
        key = str(record["historical_reuse_count"])
        reuse_histogram[key] = reuse_histogram.get(key, 0) + 1
    total_blocks = len(block_hashes)
    unique_blocks = len(set(block_hashes))
    return {
        "namespace": namespace,
        "dataset_id": dataset_id,
        "source_id": source_id,
        "input_path": str(path),
        "input_sha256": sha256_file(path),
        "request_count": len(records),
        "input_token_distribution": distribution(token_counts),
        "reusable_token_ratio": reused_tokens / sum(token_counts) if token_counts else 0.0,
        "unique_prefix_block_count": unique_blocks,
        "historical_reuse_count_histogram": reuse_histogram,
        "potential_kv_bytes": potential_kv_bytes,
        "duplicated_prefix_block_ratio": (total_blocks - unique_blocks) / total_blocks if total_blocks else 0.0,
    }


def _read_observations(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"observation file not found: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid observation JSONL at line {line_number}") from error
        if not isinstance(record, dict) or record.get("evidence_class") != "modeled_dataset_metadata":
            raise ValueError(f"observation at line {line_number} is not modeled dataset metadata")
        _validate_observation(record, line_number)
        yield record


def _validate_observation(record: dict[str, Any], line_number: int) -> None:
    required_ints = (
        "token_count",
        "historical_reused_tokens",
        "historical_reuse_count",
        "potential_kv_bytes",
    )
    if not isinstance(record.get("dataset_id"), str) or not record["dataset_id"]:
        raise ValueError(f"observation at line {line_number} has no dataset_id")
    if not isinstance(record.get("block_hashes"), list) or not all(
        isinstance(value, str) for value in record["block_hashes"]
    ):
        raise ValueError(f"observation at line {line_number} has invalid block_hashes")
    for field in required_ints:
        if not isinstance(record.get(field), int) or record[field] < 0:
            raise ValueError(f"observation at line {line_number} has invalid {field}")


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0.0, "p95": 0.0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": float(median(ordered)),
        "p95": float(ordered[math.ceil(len(ordered) * 0.95) - 1]),
        "max": ordered[-1],
    }


def write_report(output_dir: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "reuse_opportunity_report.json"
    markdown_path = output / "reuse_opportunity_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Reuse Opportunity Pilot",
        "",
        f"- Evidence class: `{report['evidence_class']}`",
        f"- Decision: `{report['decision']}`",
        f"- Reason: {report['decision_reason']}",
        f"- Sampling: seed `{report['sampling']['seed']}`, smoke `{report['sampling']['smoke_request_count']}`, "
        f"pilot `{report['sampling']['pilot_request_count']}`, method `{report['sampling']['selection_method']}`",
        "",
        "## Raw Workloads",
        "",
        "| Dataset | Requests | Reusable token ratio | Unique prefix blocks | Potential KV bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["raw_workloads"]:
        lines.append(
            f"| {item['dataset_id']} | {item['request_count']} | {item['reusable_token_ratio']:.6f} | "
            f"{item['unique_prefix_block_count']} | {item['potential_kv_bytes']} |"
        )
    lines.extend(["", "## Composed Stress Workloads", "", "Reported separately and excluded from raw aggregates."])
    if report["missing_required_slots"]:
        lines.extend(["", "## Missing Required Slots", ""])
        lines.extend(f"- `{item['slot']}`: {item['reason']}" for item in report["missing_required_slots"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    report = build_reuse_opportunity_report(config)
    json_path, markdown_path = write_report(args.output_dir, report)
    print(f"Reuse opportunity report written to {json_path}")
    print(f"Reuse opportunity report written to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
