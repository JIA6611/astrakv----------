"""Build LayerSensitivityRecord and QualityGuardRecord entries from offline artifacts.

This closes the offline-to-online policy loop for AstraKV-W MVP by turning
quality and hidden-state evaluation outputs into ProfileDB guard records that
the online controller can consume directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.profile_db import LayerSensitivityRecord, ProfileDB, QualityGuardRecord  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = ProfileDB.load(args.profile_db) if args.profile_db else ProfileDB()

    layer_records: list[LayerSensitivityRecord] = []
    quality_records: list[QualityGuardRecord] = []

    if args.layer_sensitivity_csv:
        layer_records.extend(
            _load_layer_sensitivity_csv(
                Path(args.layer_sensitivity_csv),
                workload_id=args.workload_id,
                sensitivity_partial_block_threshold=args.sensitivity_partial_block_threshold,
                sensitivity_recompute_block_threshold=args.sensitivity_recompute_block_threshold,
                high_priority_prefetch_boost=args.high_priority_prefetch_boost,
            )
        )
    if args.hidden_state_drift_csv:
        layer_records.extend(
            _load_hidden_state_drift_csv(
                Path(args.hidden_state_drift_csv),
                workload_id=args.workload_id,
                min_cka=args.min_cka,
                max_l2_drift=args.max_l2_drift,
                max_abs_diff=args.max_abs_diff,
                high_priority_prefetch_boost=args.high_priority_prefetch_boost,
            )
        )
    if args.quality_csv:
        quality_records.extend(
            _load_quality_csv(
                Path(args.quality_csv),
                workload_id=args.workload_id,
                max_ppl_delta=args.max_ppl_delta,
                high_priority_prefetch_boost=args.high_priority_prefetch_boost,
            )
        )

    db.merge_profile_guards(
        layer_sensitivity=_dedupe_layer_records(layer_records),
        quality_guards=_dedupe_quality_records(quality_records),
    )

    db_path = output_dir / args.db_name
    db.save(db_path)

    summary = {
        "schema": "astrakv-profile-guards-summary-v1",
        "workload_id": args.workload_id,
        "inputs": {
            "profile_db": args.profile_db,
            "quality_csv": args.quality_csv,
            "hidden_state_drift_csv": args.hidden_state_drift_csv,
            "layer_sensitivity_csv": args.layer_sensitivity_csv,
        },
        "thresholds": {
            "max_ppl_delta": args.max_ppl_delta,
            "min_cka": args.min_cka,
            "max_l2_drift": args.max_l2_drift,
            "max_abs_diff": args.max_abs_diff,
            "sensitivity_partial_block_threshold": args.sensitivity_partial_block_threshold,
            "sensitivity_recompute_block_threshold": args.sensitivity_recompute_block_threshold,
        },
        "counts": {
            "layer_sensitivity_records": len(db.layer_sensitivity),
            "quality_guard_records": len(db.quality_guards),
        },
        "output_profile_db": str(db_path),
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profile guards merged into {db_path}")
    print(f"Guard summary written to {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--profile-db", help="Optional base ProfileDB JSON to merge into.")
    parser.add_argument("--quality-csv", help="quality_results.csv from scripts/research/evaluate_quality.py")
    parser.add_argument("--hidden-state-drift-csv", help="hidden_state_drift_results.csv from scripts/research/evaluate_hidden_state_drift.py")
    parser.add_argument("--layer-sensitivity-csv", help="Optional CSV with layer_id,sensitivity_score columns.")
    parser.add_argument("--output-dir", default="results/profile_guards")
    parser.add_argument("--db-name", default="profile_db_with_guards.json")
    parser.add_argument("--summary-name", default="profile_guard_summary.json")
    parser.add_argument("--max-ppl-delta", type=float, default=0.2)
    parser.add_argument("--min-cka", type=float, default=0.95)
    parser.add_argument("--max-l2-drift", type=float, default=0.25)
    parser.add_argument("--max-abs-diff", type=float, default=0.15)
    parser.add_argument("--sensitivity-partial-block-threshold", type=float, default=0.8)
    parser.add_argument("--sensitivity-recompute-block-threshold", type=float, default=0.95)
    parser.add_argument("--high-priority-prefetch-boost", type=float, default=0.2)
    return parser.parse_args()


def _load_quality_csv(
    path: Path,
    *,
    workload_id: str,
    max_ppl_delta: float,
    high_priority_prefetch_boost: float,
) -> list[QualityGuardRecord]:
    rows = _read_csv(path)
    records: list[QualityGuardRecord] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id") or row.get("sample_id") or row.get("case") or "")
        if not chunk_id:
            continue
        ppl_delta = _as_float(row.get("ppl_delta"))
        exact_match = _as_float(row.get("exact_match"))
        normalized_match = _as_float(row.get("normalized_match"))
        partial_allowed = ppl_delta is None or ppl_delta <= max_ppl_delta
        recompute_allowed = partial_allowed and (exact_match is None or exact_match >= 1.0)
        quality_tier = "guarded" if not partial_allowed or not recompute_allowed else "open"
        boost = high_priority_prefetch_boost if normalized_match is not None and normalized_match >= 1.0 else 0.0
        records.append(
            QualityGuardRecord(
                workload_id=workload_id,
                chunk_id=chunk_id,
                quality_tier=quality_tier,
                max_ppl_delta=ppl_delta,
                partial_load_allowed=partial_allowed,
                recompute_allowed=recompute_allowed,
                prefetch_priority_boost=boost,
                metadata={
                    "source": str(path),
                    "exact_match": exact_match,
                    "normalized_match": normalized_match,
                    "token_divergence_rate": _as_float(row.get("token_divergence_rate")),
                    "char_divergence_rate": _as_float(row.get("char_divergence_rate")),
                },
            )
        )
    return records


def _load_hidden_state_drift_csv(
    path: Path,
    *,
    workload_id: str,
    min_cka: float,
    max_l2_drift: float,
    max_abs_diff: float,
    high_priority_prefetch_boost: float,
) -> list[LayerSensitivityRecord]:
    grouped: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "sensitivity_score": 0.0,
            "min_cka": 1.0,
            "max_l2_drift": 0.0,
            "max_abs_diff": 0.0,
        }
    )
    for row in _read_csv(path):
        layer_id = _as_int(row.get("layer_id"))
        if layer_id is None:
            continue
        cka = _as_float(row.get("cka"))
        l2 = _as_float(row.get("l2_drift")) or 0.0
        abs_diff = _as_float(row.get("max_abs_diff")) or 0.0
        sensitivity = max(
            0.0,
            1.0 - (cka if cka is not None else 1.0),
            l2,
            abs_diff,
        )
        bucket = grouped[layer_id]
        bucket["sensitivity_score"] = max(bucket["sensitivity_score"], sensitivity)
        bucket["min_cka"] = min(bucket["min_cka"], cka if cka is not None else 1.0)
        bucket["max_l2_drift"] = max(bucket["max_l2_drift"], l2)
        bucket["max_abs_diff"] = max(bucket["max_abs_diff"], abs_diff)

    records: list[LayerSensitivityRecord] = []
    for layer_id, bucket in sorted(grouped.items()):
        partial_allowed = bucket["min_cka"] >= min_cka and bucket["max_l2_drift"] <= max_l2_drift
        recompute_allowed = partial_allowed and bucket["max_abs_diff"] <= max_abs_diff
        boost = high_priority_prefetch_boost if partial_allowed and recompute_allowed else 0.0
        records.append(
            LayerSensitivityRecord(
                workload_id=workload_id,
                layer_id=layer_id,
                sensitivity_score=bucket["sensitivity_score"],
                partial_load_allowed=partial_allowed,
                recompute_allowed=recompute_allowed,
                prefetch_priority_boost=boost,
                metadata={
                    "source": str(path),
                    "min_cka": bucket["min_cka"],
                    "max_l2_drift": bucket["max_l2_drift"],
                    "max_abs_diff": bucket["max_abs_diff"],
                },
            )
        )
    return records


def _load_layer_sensitivity_csv(
    path: Path,
    *,
    workload_id: str,
    sensitivity_partial_block_threshold: float,
    sensitivity_recompute_block_threshold: float,
    high_priority_prefetch_boost: float,
) -> list[LayerSensitivityRecord]:
    records: list[LayerSensitivityRecord] = []
    for row in _read_csv(path):
        layer_id = _as_int(row.get("layer_id"))
        if layer_id is None:
            continue
        score = _as_float(row.get("sensitivity_score")) or 0.0
        partial_allowed = score < sensitivity_partial_block_threshold
        recompute_allowed = score < sensitivity_recompute_block_threshold
        records.append(
            LayerSensitivityRecord(
                workload_id=workload_id,
                layer_id=layer_id,
                sensitivity_score=score,
                partial_load_allowed=partial_allowed,
                recompute_allowed=recompute_allowed,
                prefetch_priority_boost=high_priority_prefetch_boost if partial_allowed else 0.0,
                metadata={"source": str(path)},
            )
        )
    return records


def _dedupe_layer_records(records: list[LayerSensitivityRecord]) -> list[LayerSensitivityRecord]:
    merged: dict[tuple[str, int], LayerSensitivityRecord] = {}
    for record in records:
        key = (record.workload_id, record.layer_id)
        current = merged.get(key)
        if current is None:
            merged[key] = record
            continue
        merged[key] = LayerSensitivityRecord(
            workload_id=record.workload_id,
            layer_id=record.layer_id,
            sensitivity_score=max(current.sensitivity_score, record.sensitivity_score),
            partial_load_allowed=current.partial_load_allowed and record.partial_load_allowed,
            recompute_allowed=current.recompute_allowed and record.recompute_allowed,
            prefetch_priority_boost=max(current.prefetch_priority_boost, record.prefetch_priority_boost),
            metadata={**current.metadata, **record.metadata},
        )
    return list(merged.values())


def _dedupe_quality_records(records: list[QualityGuardRecord]) -> list[QualityGuardRecord]:
    merged: dict[tuple[str, str, int | None], QualityGuardRecord] = {}
    for record in records:
        key = (record.workload_id, record.chunk_id, record.layer_id)
        current = merged.get(key)
        if current is None:
            merged[key] = record
            continue
        merged[key] = QualityGuardRecord(
            workload_id=record.workload_id,
            chunk_id=record.chunk_id,
            layer_id=record.layer_id,
            quality_tier="guarded" if "guarded" in {current.quality_tier, record.quality_tier} else "open",
            max_ppl_delta=_max_optional(current.max_ppl_delta, record.max_ppl_delta),
            min_cka=_min_optional(current.min_cka, record.min_cka),
            partial_load_allowed=current.partial_load_allowed and record.partial_load_allowed,
            recompute_allowed=current.recompute_allowed and record.recompute_allowed,
            prefetch_priority_boost=max(current.prefetch_priority_boost, record.prefetch_priority_boost),
            metadata={**current.metadata, **record.metadata},
        )
    return list(merged.values())


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


if __name__ == "__main__":
    raise SystemExit(main())
