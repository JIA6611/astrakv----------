"""Analyze archived benchmark results across multiple models and backends.

This script reads existing `benchmark_results.csv` files. It does not launch
model servers, download checkpoints, or modify the benchmark runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.multi_model import (  # noqa: E402
    MultiModelRunInput,
    compare_runs,
    parse_run_spec,
    summarize_runs,
    write_csv,
    write_manifest,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [parse_run_spec(item, order=index) for index, item in enumerate(args.run)]
    summary_rows = summarize_runs(inputs)
    comparison_rows = compare_runs(inputs)

    summary_path = output_dir / args.summary_name
    comparison_path = output_dir / args.comparison_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_csv(summary_path, summary_rows)
    write_csv(comparison_path, comparison_rows)
    write_report(report_path, inputs, summary_rows, comparison_rows, summary_path, comparison_path)
    write_manifest(manifest_path, inputs, summary_rows, comparison_rows)

    print(f"Multi-model summary written to {summary_path}")
    print(f"Multi-model comparison written to {comparison_path}")
    print(f"Multi-model report written to {report_path}")
    print(f"Multi-model manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run as model_id:model_family:backend=path. First run per model is that model's baseline.",
    )
    parser.add_argument("--output-dir", default="results/multi_model_evaluation")
    parser.add_argument("--summary-name", default="multi_model_summary.csv")
    parser.add_argument("--comparison-name", default="multi_model_comparison.csv")
    parser.add_argument("--report-name", default="multi_model_report.md")
    parser.add_argument("--manifest-name", default="multi_model_manifest.json")
    return parser.parse_args()


def write_report(
    path: Path,
    inputs: list[MultiModelRunInput],
    summary_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    summary_path: Path,
    comparison_path: Path,
) -> None:
    families = sorted({item.model_family for item in inputs})
    models = sorted({item.model_id for item in inputs})
    lines = [
        "# Multi-Model Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Model families: `{', '.join(families)}`",
        f"- Models: `{', '.join(models)}`",
        "",
        "| model | family | backend | csv |",
        "| --- | --- | --- | --- |",
    ]
    for item in inputs:
        lines.append(f"| {item.model_id} | {item.model_family} | {item.backend} | `{item.csv_path}` |")

    lines.extend(
        [
            "",
            "## Per-Run Summary",
            "",
            "| model | family | backend | success | max ctx | max batch | TTFT ms | TPOT ms | throughput | RSS MB | disk read | disk write |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {model_id} | {model_family} | {backend} | {success_rate} | "
            "{max_context_length} | {max_batch_size} | {mean_ttft_ms} | {mean_tpot_ms} | "
            "{mean_throughput_tokens_s} | {max_process_rss_peak_mb} | {disk_read_delta_mb} | "
            "{disk_write_delta_mb} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Baseline-Aligned Comparison",
            "",
            "| model | family | backend | case | success | TTFT delta % | TPOT delta % | throughput delta % | RSS delta % | summary |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows[:80]:
        lines.append(
            "| {model_id} | {model_family} | {backend} | {case} | {success_rate} | "
            "{ttft_ms_delta_pct} | {tpot_ms_delta_pct} | {throughput_tokens_s_delta_pct} | "
            "{process_rss_peak_mb_delta_pct} | {help_hurt_summary} |".format(**row)
        )
    if len(comparison_rows) > 80:
        lines.append(f"| ... |  |  |  |  |  |  |  |  | {len(comparison_rows) - 80} more row(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The first provided run for each `model_id` is treated as that model's baseline.",
            "- RSS, disk IO, latency, throughput, and success rate are the primary portable comparison signals.",
            "- `gpu_memory_peak_mb` remains in CSV artifacts for platforms that expose it, but DGX Spark may leave it empty.",
            "- `help_hurt_summary` is a coarse compatibility signal; inspect raw CSVs before making final claims.",
            "- This analyzer summarizes archived artifacts only. Official multi-model claims require real GPU runs.",
            "",
            "## Artifacts",
            "",
            f"- `{summary_path}`",
            f"- `{comparison_path}`",
            "- `multi_model_report.md`",
            "- `multi_model_manifest.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
