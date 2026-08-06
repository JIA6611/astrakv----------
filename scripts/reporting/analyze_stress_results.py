"""Analyze memory-constrained stress benchmark results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "oom",
    "cannot allocate memory",
    "memory allocation failed",
    "allocation failed",
    "cuda error 2",
)


@dataclass(frozen=True, slots=True)
class StressInput:
    label: str
    csv_path: Path


def main() -> int:
    args = parse_args()
    inputs = [parse_input_arg(item) for item in args.run]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(item) for item in inputs]
    csv_path = output_dir / "stress_summary.csv"
    report_path = output_dir / "stress_report.md"
    write_csv(csv_path, rows)
    write_report(report_path, inputs, rows)
    print(f"Stress summary written to {csv_path}")
    print(f"Stress report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Stress run as label=path. Path can be a run directory or benchmark_results.csv.",
    )
    parser.add_argument("--output-dir", default="results/stress_analysis")
    return parser.parse_args()


def parse_input_arg(value: str) -> StressInput:
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
    return StressInput(label=label, csv_path=csv_path)


def summarize_run(item: StressInput) -> dict[str, Any]:
    rows = load_rows(item.csv_path)
    total_cases = len(rows)
    total_requests = sum(as_int(row.get("request_count")) for row in rows)
    success_requests = sum(as_int(row.get("success_count")) for row in rows)
    failed_requests = max(0, total_requests - success_requests)
    successful_rows = [row for row in rows if as_int(row.get("success_count")) > 0]
    failed_rows = [row for row in rows if row.get("status") != "ok" or as_int(row.get("success_count")) < as_int(row.get("request_count"))]
    oom_rows = [row for row in rows if looks_oom(row.get("errors", ""))]

    return {
        "run": item.label,
        "csv_path": str(item.csv_path),
        "total_cases": total_cases,
        "total_requests": total_requests,
        "success_requests": success_requests,
        "failed_requests": failed_requests,
        "success_rate": ratio(success_requests, total_requests),
        "error_rate": ratio(failed_requests, total_requests),
        "failed_case_count": len(failed_rows),
        "oom_case_count": len(oom_rows),
        "oom_rate_by_case": ratio(len(oom_rows), total_cases),
        "max_success_context": max_or_blank(as_int(row.get("context_length")) for row in successful_rows),
        "max_success_batch": max_or_blank(as_int(row.get("batch_size")) for row in successful_rows),
        "max_success_context_at_batch1": max_context_for_batch(successful_rows, 1),
        "max_success_context_at_batch2": max_context_for_batch(successful_rows, 2),
        "max_success_context_at_batch4": max_context_for_batch(successful_rows, 4),
        "worst_latency_p95_ms": max_float_or_blank(row.get("latency_p95_ms") for row in successful_rows),
        "worst_tpot_p95_ms": max_float_or_blank(row.get("tpot_p95_ms") for row in successful_rows),
        "gpu_memory_peak_mb": max_float_or_blank(row.get("gpu_memory_peak_mb") for row in rows),
        "process_rss_peak_mb": max_float_or_blank((row.get("process_rss_peak_mb") or row.get("cpu_memory_peak_mb")) for row in rows),
        "cpu_memory_peak_mb": max_float_or_blank((row.get("process_rss_peak_mb") or row.get("cpu_memory_peak_mb")) for row in rows),
        "disk_read_delta_mb": sum_float_or_blank(row.get("disk_read_delta_mb") for row in rows),
        "disk_write_delta_mb": sum_float_or_blank(row.get("disk_write_delta_mb") for row in rows),
        "error_examples": " | ".join(sorted({row.get("errors", "") for row in failed_rows if row.get("errors", "")}))[:1000],
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def looks_oom(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(numerator: int, denominator: int) -> float | str:
    if denominator <= 0:
        return ""
    return numerator / denominator


def max_or_blank(values: Any) -> int | str:
    items = [int(value) for value in values if value not in ("", None)]
    return max(items) if items else ""


def max_float_or_blank(values: Any) -> float | str:
    items = [value for value in (as_float(item) for item in values) if value is not None]
    return max(items) if items else ""


def sum_float_or_blank(values: Any) -> float | str:
    items = [value for value in (as_float(item) for item in values) if value is not None]
    return sum(items) if items else ""


def max_context_for_batch(rows: list[dict[str, str]], batch_size: int) -> int | str:
    contexts = [
        as_int(row.get("context_length"))
        for row in rows
        if as_int(row.get("batch_size")) == batch_size
    ]
    return max(contexts) if contexts else ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, inputs: list[StressInput], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Memory-Constrained Stress Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
    ]
    for item in inputs:
        lines.append(f"- `{item.label}` -> `{item.csv_path}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| run | success rate | error rate | OOM cases | max ctx | max batch | ctx@bs1 | ctx@bs2 | ctx@bs4 | latency p95 worst | RSS MB peak | disk read MB | disk write MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run} | {success_rate} | {error_rate} | {oom_case_count} | "
            "{max_success_context} | {max_success_batch} | "
            "{max_success_context_at_batch1} | {max_success_context_at_batch2} | "
            "{max_success_context_at_batch4} | {worst_latency_p95_ms} | "
            "{cpu_memory_peak_mb} | {disk_read_delta_mb} | {disk_write_delta_mb} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Error Examples",
            "",
        ]
    )
    for row in rows:
        examples = row.get("error_examples") or "None."
        lines.append(f"- `{row['run']}`: {examples}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `max ctx` is the largest context length with at least one successful request.",
            "- `max batch` is the largest batch size with at least one successful request.",
            "- OOM detection is string-based and should be verified against server logs for official runs.",
            "- RSS, disk IO, GPU utilization, and vLLM startup KV-cache capacity are the stable memory-pressure evidence on DGX Spark.",
            "",
            "## Artifacts",
            "",
            "- `stress_summary.csv`",
            "- `stress_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
