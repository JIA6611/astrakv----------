"""Convert abstract predictor candidates into runtime advisory sidecar rows."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from astrakv.runtime.prediction_sidecar import (
    PredictorCandidateRecord,
    SIDECAR_PREDICTION_SCHEMA,
    SidecarPrediction,
)


DEFAULT_CANDIDATE_REPORT = Path("results/predictor_candidates/predictor_candidate_report.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/sidecar_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", default=str(DEFAULT_CANDIDATE_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lead-time-ms", type=float, default=250.0)
    parser.add_argument("--expires-after-ms", type=float, default=21_600_000.0)
    parser.add_argument(
        "--predicted-class",
        action="append",
        default=[],
        help="Optional candidate classes to include. Defaults to exact-next only.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum candidate confidence to emit.",
    )
    parser.add_argument(
        "--evidence-source",
        default="predictor_candidate_report",
        help="Opaque provenance string carried into runtime metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_classes = {
        item.strip() for item in args.predicted_class if item.strip()
    } or {"exact-next"}
    if args.lead_time_ms < 0 or args.expires_after_ms <= 0:
        raise ValueError("lead-time and expiry windows must be positive")

    candidate_rows = load_candidates(Path(args.candidate_report))
    expires_at_ns = time.time_ns() + int(args.expires_after_ms * 1_000_000)
    predictions: list[SidecarPrediction] = []
    for row in candidate_rows:
        if row.predicted_class not in allowed_classes:
            continue
        if row.confidence < args.min_confidence:
            continue
        predictions.append(
            SidecarPrediction(
                run_id=args.run_id,
                request_id=row.request_id,
                candidate_object_id=row.candidate_object_id,
                object_level=row.object_level,
                score=_score_for(row),
                recommended_lead_time_ms=float(args.lead_time_ms),
                confidence=row.confidence,
                reason=row.reason,
                evidence_source=str(args.evidence_source),
                predicted_class=row.predicted_class,
                expires_at_ns=expires_at_ns,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "sidecar_prediction.jsonl"
    csv_path = output_dir / "sidecar_prediction.csv"
    summary_path = output_dir / "sidecar_prediction_summary.json"

    write_jsonl(jsonl_path, [item.to_record() for item in predictions])
    write_csv(csv_path, [item.to_record() for item in predictions])
    summary_path.write_text(
        json.dumps(
            {
                "schema": SIDECAR_PREDICTION_SCHEMA,
                "run_id": args.run_id,
                "record_count": len(predictions),
                "lead_time_ms": args.lead_time_ms,
                "expires_at_ns": expires_at_ns,
                "predicted_classes": sorted(allowed_classes),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Sidecar predictions written to {jsonl_path}")
    print(f"Sidecar prediction CSV written to {csv_path}")
    print(f"Sidecar prediction summary written to {summary_path}")
    return 0


def _score_for(row: PredictorCandidateRecord) -> float:
    token_bonus = min(0.08, row.estimated_reusable_tokens / 100_000.0)
    return min(1.0, max(0.0, row.confidence + token_bonus))


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "schema",
        "run_id",
        "request_id",
        "candidate_object_id",
        "object_level",
        "score",
        "recommended_lead_time_ms",
        "confidence",
        "reason",
        "evidence_source",
        "predicted_class",
        "expires_at_ns",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
