"""Evaluate CKA and hidden-state drift from archived JSONL artifacts.

This script compares baseline and variant hidden states that were already
exported by a model/runtime hook. It does not run a model or allocate GPU
tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.evaluation.hidden_state import (  # noqa: E402
    HiddenStateComparison,
    compare_record_sets,
    load_hidden_state_jsonl,
    summarize_comparisons,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_hidden_state_jsonl(args.baseline_jsonl)
    variant = load_hidden_state_jsonl(args.variant_jsonl)
    comparisons = compare_record_sets(baseline, variant)

    records_path = output_dir / args.records_name
    csv_path = output_dir / args.csv_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_records_jsonl(records_path, comparisons)
    write_results_csv(csv_path, comparisons)
    write_report(report_path, args, comparisons, records_path, csv_path)
    write_manifest(manifest_path, args, comparisons, records_path, csv_path, report_path)

    print(f"Hidden-state drift records written to {records_path}")
    print(f"Hidden-state drift CSV written to {csv_path}")
    print(f"Hidden-state drift report written to {report_path}")
    print(f"Hidden-state drift manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", required=True, help="Baseline hidden-state JSONL.")
    parser.add_argument("--variant-jsonl", required=True, help="Variant hidden-state JSONL.")
    parser.add_argument("--output-dir", default="results/hidden_state_drift")
    parser.add_argument("--records-name", default="hidden_state_drift_records.jsonl")
    parser.add_argument("--csv-name", default="hidden_state_drift_results.csv")
    parser.add_argument("--report-name", default="hidden_state_drift_report.md")
    parser.add_argument("--manifest-name", default="hidden_state_drift_manifest.json")
    return parser.parse_args()


def write_records_jsonl(path: Path, comparisons: list[HiddenStateComparison]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for comparison in comparisons:
            handle.write(json.dumps(comparison.to_record(), ensure_ascii=False) + "\n")


def write_results_csv(path: Path, comparisons: list[HiddenStateComparison]) -> None:
    rows = [comparison.to_record() for comparison in comparisons]
    fieldnames = [
        "sample_id",
        "layer_id",
        "token_index",
        "status",
        "baseline_shape",
        "variant_shape",
        "element_count",
        "cka",
        "cosine_similarity",
        "mse",
        "l2_drift",
        "max_abs_diff",
        "reason",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_report(
    path: Path,
    args: argparse.Namespace,
    comparisons: list[HiddenStateComparison],
    records_path: Path,
    csv_path: Path,
) -> None:
    summary = summarize_comparisons(comparisons)
    summary_record = summary.to_record()
    lines = [
        "# Hidden-State Drift Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Baseline JSONL: `{args.baseline_jsonl}`",
        f"- Variant JSONL: `{args.variant_jsonl}`",
        "",
        "## Summary",
        "",
        f"- Comparisons: `{summary.comparison_count}`",
        f"- OK comparisons: `{summary.ok_count}`",
        f"- Mean CKA: `{summary_record['mean_cka']}`",
        f"- Mean cosine similarity: `{summary_record['mean_cosine_similarity']}`",
        f"- Mean MSE: `{summary_record['mean_mse']}`",
        f"- Mean L2 drift: `{summary_record['mean_l2_drift']}`",
        f"- Max L2 drift: `{summary_record['max_l2_drift']}`",
        f"- Max abs diff: `{summary_record['max_abs_diff']}`",
        "",
        "### Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in summary.status_counts.items():
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Per-Layer Results",
            "",
            "| sample | layer | token | status | shape | CKA | cosine | MSE | L2 drift | max abs diff |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in comparisons[:80]:
        record = comparison.to_record()
        lines.append(
            f"| {record['sample_id']} | {record['layer_id']} | {record['token_index']} | "
            f"{record['status']} | {record['baseline_shape']}->{record['variant_shape']} | "
            f"{record['cka']} | {record['cosine_similarity']} | {record['mse']} | "
            f"{record['l2_drift']} | {record['max_abs_diff']} |"
        )
    if len(comparisons) > 80:
        lines.append(f"| ... |  |  |  |  |  |  |  |  | {len(comparisons) - 80} more comparison(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CKA near 1.0 indicates similar representational geometry.",
            "- Cosine similarity near 1.0 indicates similar flattened hidden states.",
            "- MSE, L2 drift, and max absolute difference expose numeric drift.",
            "- Shape mismatches and missing layers are reported explicitly rather than coerced.",
            "- This evaluator consumes exported hidden states only; GPU/model hooks must create the inputs for official claims.",
            "",
            "## Artifacts",
            "",
            f"- `{records_path}`",
            f"- `{csv_path}`",
            "- `hidden_state_drift_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    comparisons: list[HiddenStateComparison],
    records_path: Path,
    csv_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-hidden-state-drift-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "baseline_jsonl": args.baseline_jsonl,
            "variant_jsonl": args.variant_jsonl,
        },
        "outputs": {
            "records_jsonl": str(records_path),
            "results_csv": str(csv_path),
            "report": str(report_path),
        },
        "summary": summarize_comparisons(comparisons).to_record(),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
