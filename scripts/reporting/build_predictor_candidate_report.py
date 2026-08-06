"""Build a benchmark-agnostic predictor candidate report from unified reuse rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.prediction_sidecar import (
    PREDICTOR_CANDIDATE_REPORT_SCHEMA,
    PredictorCandidateRecord,
)


DEFAULT_ANALYSIS_JSONL = Path("results/reuse_pattern_analysis/unified_reuse_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/predictor_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-jsonl", default=str(DEFAULT_ANALYSIS_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help="Optional grouped source_name filter such as qasper or multifieldqa_en.",
    )
    parser.add_argument(
        "--predicted-class",
        action="append",
        default=[],
        help="Optional reuse class filter. Defaults to all three classes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.analysis_jsonl))
    requested_sources = {item.strip() for item in args.source_name if item.strip()}
    requested_classes = {
        item.strip() for item in args.predicted_class if item.strip()
    } or {"exact-next", "fixed-revisit", "structural-partial"}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[PredictorCandidateRecord] = []
    for row in rows:
        if str(row.get("source_kind") or "") != "grouped_prompt":
            continue
        source_name = str(row.get("source_name") or "")
        if requested_sources and source_name not in requested_sources:
            continue
        predicted_class = str(row.get("reuse_class") or "")
        if predicted_class not in requested_classes:
            continue
        candidate_object_id = _candidate_object_id(row)
        if not candidate_object_id:
            continue
        records.append(
            PredictorCandidateRecord(
                request_id=str(row.get("request_id") or ""),
                candidate_object_id=candidate_object_id,
                object_level=_candidate_object_level(row, candidate_object_id),
                predicted_class=predicted_class,
                lead_distance_requests=_lead_distance_requests(row),
                estimated_reusable_tokens=_as_non_negative_int(
                    row.get("estimated_reusable_tokens")
                ),
                estimated_kv_bytes=_as_non_negative_int(row.get("estimated_kv_bytes")),
                confidence=_confidence_for(predicted_class, row),
                reason=_reason_for(predicted_class),
            )
        )

    jsonl_path = output_dir / "predictor_candidate_report.jsonl"
    csv_path = output_dir / "predictor_candidate_report.csv"
    summary_path = output_dir / "predictor_candidate_report_summary.json"

    write_jsonl(jsonl_path, [item.to_record() for item in records])
    write_csv(csv_path, [item.to_record() for item in records])
    summary = summarize(records)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Predictor candidate report written to {jsonl_path}")
    print(f"Predictor candidate CSV written to {csv_path}")
    print(f"Predictor candidate summary written to {summary_path}")
    return 0


def _candidate_object_id(row: dict[str, Any]) -> str:
    for field in (
        "runtime_object_key",
        "cache_key",
        "prefix_id",
        "workflow_id",
        "reuse_group",
        "request_id",
    ):
        value = str(row.get(field) or "")
        if value:
            return value
    return ""


def _candidate_object_level(row: dict[str, Any], candidate_object_id: str) -> ObjectLevel:
    level = str(row.get("runtime_object_level") or row.get("object_level") or "")
    if not level and candidate_object_id and candidate_object_id == str(row.get("cache_key") or ""):
        level = "cache_key"
    if not level:
        level = "prefix"
    return ObjectLevel("prefix" if level == "" else level)


def _lead_distance_requests(row: dict[str, Any]) -> int:
    for field in ("next_same_group_distance", "adjacent_distance", "prev_same_group_distance"):
        value = row.get(field)
        try:
            distance = int(value)
        except (TypeError, ValueError):
            continue
        if distance >= 0:
            return distance
    return 0


def _confidence_for(predicted_class: str, row: dict[str, Any]) -> float:
    group_size = _as_non_negative_int(row.get("group_size"))
    if predicted_class == "exact-next":
        return min(1.0, 0.95 + (0.01 if group_size > 2 else 0.0))
    if predicted_class == "fixed-revisit":
        return 0.7
    return 0.45


def _reason_for(predicted_class: str) -> str:
    if predicted_class == "exact-next":
        return "exact_next_locality"
    if predicted_class == "fixed-revisit":
        return "recent_same_object_revisit"
    return "structural_partial_reuse"


def summarize(records: list[PredictorCandidateRecord]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    source_object_count = len({item.candidate_object_id for item in records})
    for item in records:
        counts[item.predicted_class] = counts.get(item.predicted_class, 0) + 1
    return {
        "schema": PREDICTOR_CANDIDATE_REPORT_SCHEMA,
        "record_count": len(records),
        "distinct_candidate_object_count": source_object_count,
        "class_counts": dict(sorted(counts.items())),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        "schema",
        "request_id",
        "candidate_object_id",
        "object_level",
        "predicted_class",
        "lead_distance_requests",
        "estimated_reusable_tokens",
        "estimated_kv_bytes",
        "confidence",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _as_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
