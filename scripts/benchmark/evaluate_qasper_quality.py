"""Evaluate QASPER ground-truth Exact Match and token-F1 from benchmark output."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.request_results)
    records = [score(row) for row in rows]
    write_jsonl(output / "qasper_quality_records.jsonl", records)
    write_csv(output / "qasper_quality_summary.csv", records)
    write_report(output / "qasper_quality_report.md", records, args)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-results", required=True)
    parser.add_argument("--output-dir", default="results/qasper_quality")
    return parser.parse_args()


def score(row: dict[str, Any]) -> dict[str, Any]:
    output = str(row.get("output_text") or "")
    reference = str(row.get("ground_truth") or "")
    status = str(row.get("status") or "missing")
    prediction_tokens, reference_tokens = tokens(output), tokens(reference)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    valid = status == "ok" and bool(output) and bool(reference)
    return {
        "request_id": row.get("request_id", ""), "sample_id": row.get("sample_id", ""),
        "status": status, "ground_truth_available": bool(reference), "exact_match": bool(valid and normalize(output) == normalize(reference)),
        "token_f1": f1 if valid else 0.0, "token_precision": precision if valid else 0.0,
        "token_recall": recall if valid else 0.0, "output_text": output, "ground_truth": reference,
    }


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", value).lower()).split())


def tokens(value: str) -> list[str]:
    return normalize(value).split()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    values = {"samples": len(rows), "ok_samples": sum(item["status"] == "ok" for item in rows),
              "exact_match": mean([float(item["exact_match"]) for item in rows]), "token_f1": mean([float(item["token_f1"]) for item in rows])}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in values.items())


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.write_text("\n".join([
        "# QASPER Quality Report", "", f"- Request results: `{args.request_results}`",
        f"- Samples: `{len(rows)}`", f"- Exact match: `{mean([float(item['exact_match']) for item in rows]):.4f}`",
        f"- Token-F1: `{mean([float(item['token_f1']) for item in rows]):.4f}`", "",
        "Ground truth is treated as one literal reference; `|` is not split into alternatives.",
    ]) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
