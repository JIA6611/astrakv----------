"""Compare real endpoint benchmark runs.

The comparator reads multiple `benchmark_results.csv` files and aligns rows by
batch size, context length, and output tokens. The first run is treated as the
baseline for delta calculations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from astrakv.benchmarks.paired_run import PairedRunInput, PairedRunValidation, validate_paired_runs, write_paired_run_manifest


KEY_FIELDS = ("batch_size", "context_length", "output_tokens")
NUMERIC_FIELDS = (
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    "latency_p95_ms",
    "throughput_tokens_s",
    "process_rss_peak_mb",
    "cpu_memory_peak_mb",
    "gpu_memory_peak_mb",
    "gpu_util_peak_pct",
    "disk_read_delta_mb",
    "disk_write_delta_mb",
)


@dataclass(frozen=True, slots=True)
class RunInput:
    label: str
    csv_path: Path


def main() -> int:
    args = parse_args()
    runs = [parse_run_arg(item) for item in args.run]
    if len(runs) < 2:
        raise SystemExit("Provide at least two --run entries, with the baseline first.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_status, validation = comparison_claim(runs, unpaired=args.unpaired)
    if validation is not None:
        paired_manifest = output_dir / "paired_run_manifest.json"
        write_paired_run_manifest(paired_manifest, validation)
        if not validation.eligible:
            raise SystemExit(f"Paired-run evidence is ineligible; see {paired_manifest}")
    elif claim_status == "paired_claim_blocked":
        raise SystemExit("Paired comparisons require exactly baseline and variant runs; use --unpaired for a no-claims report.")

    comparison_rows = compare_runs(runs)
    csv_path = output_dir / "comparison_results.csv"
    report_path = output_dir / "comparison_report.md"
    write_csv(csv_path, comparison_rows)
    write_report(report_path, runs, comparison_rows, claim_status=claim_status)
    print(f"Comparison CSV written to {csv_path}")
    print(f"Comparison report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run input as label=path. Path can be a run directory or benchmark_results.csv.",
    )
    parser.add_argument("--output-dir", default="results/real_run_comparison")
    parser.add_argument("--unpaired", action="store_true", help="Write a non-paired report with no comparative claim.")
    parser.add_argument("--validate-pair", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_run_arg(value: str) -> RunInput:
    if "=" not in value:
        raise SystemExit(f"--run must use label=path format: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Run label cannot be empty: {value}")
    path = Path(raw_path.strip())
    csv_path = path / "benchmark_results.csv" if path.is_dir() else path
    if not csv_path.exists():
        raise SystemExit(f"benchmark_results.csv not found for run {label}: {csv_path}")
    return RunInput(label=label, csv_path=csv_path)


def validate_run_pair(runs: list[RunInput]) -> PairedRunValidation:
    """Validate the first two comparison inputs as baseline and variant runs."""
    if len(runs) != 2:
        raise ValueError("paired-run validation requires exactly baseline and variant inputs")
    baseline, variant = runs
    return validate_paired_runs(
        PairedRunInput(baseline.label, baseline.csv_path.parent),
        PairedRunInput(variant.label, variant.csv_path.parent),
    )


def comparison_claim(runs: list[RunInput], *, unpaired: bool) -> tuple[str, PairedRunValidation | None]:
    """Paired evidence is mandatory unless the caller explicitly opts out of claims."""
    if unpaired:
        return "non_paired_no_claims", None
    if len(runs) != 2:
        return "paired_claim_blocked", None
    validation = validate_run_pair(runs)
    return ("paired_claim_eligible" if validation.eligible else "paired_claim_blocked"), validation


def compare_runs(runs: list[RunInput]) -> list[dict[str, Any]]:
    loaded = [(run, load_rows(run.csv_path)) for run in runs]
    baseline_run, baseline_rows = loaded[0]
    baseline_by_key = {row_key(row): row for row in baseline_rows}

    rows: list[dict[str, Any]] = []
    for run, run_rows in loaded:
        for row in run_rows:
            key = row_key(row)
            baseline = baseline_by_key.get(key)
            output = {
                "run": run.label,
                "baseline_run": baseline_run.label,
                "case": row.get("case", ""),
                "backend": row.get("backend", ""),
                "model": row.get("model", ""),
                "batch_size": row.get("batch_size", ""),
                "context_length": row.get("context_length", ""),
                "output_tokens": row.get("output_tokens", ""),
                "request_count": row.get("request_count", ""),
                "success_count": row.get("success_count", ""),
                "success_rate": success_rate(row),
                "baseline_matched": bool(baseline),
            }
            for field in NUMERIC_FIELDS:
                value = as_float(row.get(field))
                base_value = as_float(baseline.get(field)) if baseline else None
                output[field] = "" if value is None else value
                output[f"{field}_baseline"] = "" if base_value is None else base_value
                output[f"{field}_delta"] = delta(value, base_value)
                output[f"{field}_delta_pct"] = delta_pct(value, base_value)
            output["gpu_memory_reduction_pct_vs_baseline"] = reduction_pct(
                as_float(row.get("gpu_memory_peak_mb")),
                as_float(baseline.get("gpu_memory_peak_mb")) if baseline else None,
            )
            output["errors"] = row.get("errors", "")
            rows.append(output)
    return sorted(
        rows,
        key=lambda item: (
            int_or_zero(item["batch_size"]),
            int_or_zero(item["context_length"]),
            int_or_zero(item["output_tokens"]),
            str(item["run"]),
        ),
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)  # type: ignore[return-value]


def success_rate(row: dict[str, Any]) -> float | str:
    success = as_float(row.get("success_count"))
    total = as_float(row.get("request_count"))
    if success is None or total in (None, 0):
        return ""
    return success / total


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline is None:
        return ""
    return value - baseline


def delta_pct(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline in (None, 0):
        return ""
    return (value - baseline) / baseline * 100.0


def reduction_pct(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline in (None, 0):
        return ""
    return (baseline - value) / baseline * 100.0


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, runs: list[RunInput], rows: list[dict[str, Any]], *, claim_status: str = "non_paired_no_claims") -> None:
    baseline = runs[0]
    lines = [
        "# Real Run Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Baseline run: `{baseline.label}` -> `{baseline.csv_path}`",
    ]
    for run in runs[1:]:
        lines.append(f"- Variant run: `{run.label}` -> `{run.csv_path}`")

    lines.extend(["", "## Claim Status", "", f"- Status: `{claim_status}`"])
    if claim_status == "non_paired_no_claims":
        lines.append("- This output is not a paired comparison and makes no controlled-comparison claim.")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| run | case | success | TTFT ms | TTFT delta % | TPOT ms | TPOT delta % | throughput tok/s | throughput delta % | RSS MB | GPU util % | disk read MB | disk write MB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        rss = row.get("process_rss_peak_mb") or row.get("cpu_memory_peak_mb")
        lines.append(
            "| {run} | {case} | {success_rate} | {ttft_ms} | {ttft_ms_delta_pct} | "
            "{tpot_ms} | {tpot_ms_delta_pct} | {throughput_tokens_s} | "
            "{throughput_tokens_s_delta_pct} | {rss} | {gpu_util_peak_pct} | "
            "{disk_read_delta_mb} | {disk_write_delta_mb} |".format(**row, rss=rss)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Negative TTFT/TPOT/latency deltas indicate the variant is faster than the baseline.",
            "- Positive throughput delta indicates the variant produced more tokens per second than the baseline.",
            "- RSS and disk increases are expected for CPU/disk-tier variants and should be interpreted together with latency.",
            "- `gpu_memory_peak_mb` is retained in the CSV only as a compatibility field. On DGX Spark it is usually unavailable because per-GPU framebuffer memory is not exposed.",
            "",
            "## Artifacts",
            "",
            "- `comparison_results.csv`",
            "- `comparison_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
