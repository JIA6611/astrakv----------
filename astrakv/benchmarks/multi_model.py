"""Multi-model benchmark aggregation helpers.

The helpers read archived `benchmark_results.csv` files from real endpoint
runs. They do not start model servers or modify benchmark execution paths.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


KEY_FIELDS = ("batch_size", "context_length", "output_tokens")
NUMERIC_FIELDS = (
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    "latency_p95_ms",
    "throughput_tokens_s",
    "process_rss_peak_mb",
    "gpu_memory_peak_mb",
    "cpu_memory_peak_mb",
    "disk_read_delta_mb",
    "disk_write_delta_mb",
)


@dataclass(frozen=True, slots=True)
class MultiModelRunInput:
    model_id: str
    model_family: str
    backend: str
    csv_path: Path
    order: int = 0

    @property
    def label(self) -> str:
        return f"{self.model_id}:{self.backend}"


def parse_run_spec(value: str, *, order: int = 0) -> MultiModelRunInput:
    if "=" not in value:
        raise ValueError(f"run spec must use model_id:model_family:backend=path format: {value}")
    left, raw_path = value.split("=", 1)
    parts = left.split(":")
    if len(parts) != 3:
        raise ValueError(f"run spec must use model_id:model_family:backend=path format: {value}")
    model_id, model_family, backend = [part.strip() for part in parts]
    if not model_id or not model_family or not backend:
        raise ValueError(f"run spec fields cannot be empty: {value}")
    path = Path(raw_path.strip())
    csv_path = path if path.suffix.lower() == ".csv" else path / "benchmark_results.csv"
    return MultiModelRunInput(
        model_id=model_id,
        model_family=model_family,
        backend=backend,
        csv_path=csv_path,
        order=order,
    )


def summarize_runs(inputs: Iterable[MultiModelRunInput]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        run_rows = load_rows(item.csv_path)
        rows.append(summarize_one_run(item, run_rows))
    return rows


def summarize_one_run(item: MultiModelRunInput, rows: list[dict[str, str]]) -> dict[str, Any]:
    total_requests = sum(as_int(row.get("request_count")) for row in rows)
    success_requests = sum(as_int(row.get("success_count")) for row in rows)
    return {
        "model_id": item.model_id,
        "model_family": item.model_family,
        "backend": item.backend,
        "csv_path": str(item.csv_path),
        "case_count": len(rows),
        "request_count": total_requests,
        "success_count": success_requests,
        "success_rate": ratio(success_requests, total_requests),
        "max_context_length": max_int(row.get("context_length") for row in rows),
        "max_batch_size": max_int(row.get("batch_size") for row in rows),
        "mean_ttft_ms": mean_field(rows, "ttft_ms"),
        "mean_tpot_ms": mean_field(rows, "tpot_ms"),
        "mean_latency_p95_ms": mean_field(rows, "latency_p95_ms"),
        "mean_throughput_tokens_s": mean_field(rows, "throughput_tokens_s"),
        "max_process_rss_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
        "max_gpu_memory_peak_mb": max_field(rows, "gpu_memory_peak_mb"),
        "max_cpu_memory_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
        "disk_read_delta_mb": sum_field(rows, "disk_read_delta_mb"),
        "disk_write_delta_mb": sum_field(rows, "disk_write_delta_mb"),
        "error_count": sum(1 for row in rows if str(row.get("status", "")).lower() not in {"", "ok"}),
    }


def compare_runs(inputs: Iterable[MultiModelRunInput]) -> list[dict[str, Any]]:
    ordered_inputs = sorted(list(inputs), key=lambda item: item.order)
    loaded = [(item, load_rows(item.csv_path)) for item in ordered_inputs]
    baseline_by_model: dict[str, tuple[MultiModelRunInput, dict[tuple[str, str, str], dict[str, str]]]] = {}
    for item, rows in loaded:
        if item.model_id in baseline_by_model:
            continue
        baseline_by_model[item.model_id] = (item, {row_key(row): row for row in rows})

    output: list[dict[str, Any]] = []
    for item, rows in loaded:
        baseline_item, baseline_rows = baseline_by_model[item.model_id]
        for row in rows:
            key = row_key(row)
            baseline = baseline_rows.get(key)
            result = {
                "model_id": item.model_id,
                "model_family": item.model_family,
                "backend": item.backend,
                "baseline_backend": baseline_item.backend,
                "case": row.get("case", ""),
                "batch_size": row.get("batch_size", ""),
                "context_length": row.get("context_length", ""),
                "output_tokens": row.get("output_tokens", ""),
                "success_rate": success_rate(row),
                "baseline_matched": bool(baseline),
            }
            for field in NUMERIC_FIELDS:
                value = as_float(row.get(field))
                base_value = as_float(baseline.get(field)) if baseline else None
                result[field] = "" if value is None else value
                result[f"{field}_baseline"] = "" if base_value is None else base_value
                result[f"{field}_delta"] = delta(value, base_value)
                result[f"{field}_delta_pct"] = delta_pct(value, base_value)
            result["gpu_memory_reduction_pct_vs_baseline"] = reduction_pct(
                as_float(row.get("gpu_memory_peak_mb")),
                as_float(baseline.get("gpu_memory_peak_mb")) if baseline else None,
            )
            result["help_hurt_summary"] = classify_help_hurt(result)
            output.append(result)
    return sorted(
        output,
        key=lambda row: (
            str(row["model_family"]),
            str(row["model_id"]),
            int_or_zero(row["context_length"]),
            int_or_zero(row["batch_size"]),
            str(row["backend"]),
        ),
    )


def classify_help_hurt(row: dict[str, Any]) -> str:
    throughput_delta = as_float(row.get("throughput_tokens_s_delta_pct"))
    gpu_reduction = as_float(row.get("gpu_memory_reduction_pct_vs_baseline"))
    latency_delta = as_float(row.get("latency_p95_ms_delta_pct"))
    signals: list[str] = []
    if gpu_reduction is not None and gpu_reduction > 1.0:
        signals.append("less_gpu")
    elif gpu_reduction is not None and gpu_reduction < -1.0:
        signals.append("more_gpu")
    if throughput_delta is not None and throughput_delta > 1.0:
        signals.append("faster")
    elif throughput_delta is not None and throughput_delta < -1.0:
        signals.append("slower")
    if latency_delta is not None and latency_delta > 1.0:
        signals.append("higher_latency")
    elif latency_delta is not None and latency_delta < -1.0:
        signals.append("lower_latency")
    return ";".join(signals) if signals else "neutral_or_baseline"


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: str | Path, inputs: list[MultiModelRunInput], summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "astra-multi-model-evaluation-manifest-v1",
        "inputs": [
            {
                "model_id": item.model_id,
                "model_family": item.model_family,
                "backend": item.backend,
                "csv_path": str(item.csv_path),
            }
            for item in inputs
        ],
        "summary_count": len(summary_rows),
        "comparison_count": len(comparison_rows),
        "model_families": sorted({item.model_family for item in inputs}),
        "model_ids": sorted({item.model_id for item in inputs}),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)  # type: ignore[return-value]


def success_rate(row: dict[str, Any]) -> float | str:
    success = as_float(row.get("success_count"))
    total = as_float(row.get("request_count"))
    if success is None or total in (None, 0):
        return ""
    return success / total


def mean_field(rows: list[dict[str, str]], field: str) -> float | str:
    values = [value for value in (as_float(row.get(field)) for row in rows) if value is not None]
    return sum(values) / len(values) if values else ""


def max_field(rows: list[dict[str, str]], field: str) -> float | str:
    values = [value for value in (as_float(row.get(field)) for row in rows) if value is not None]
    return max(values) if values else ""


def max_field_with_fallback(rows: list[dict[str, str]], primary: str, fallback: str) -> float | str:
    values = [value for value in (as_float(row.get(primary)) for row in rows) if value is not None]
    if values:
        return max(values)
    return max_field(rows, fallback)


def sum_field(rows: list[dict[str, str]], field: str) -> float | str:
    values = [value for value in (as_float(row.get(field)) for row in rows) if value is not None]
    return sum(values) if values else ""


def max_int(values: Iterable[Any]) -> int | str:
    parsed = [as_int(value) for value in values]
    parsed = [value for value in parsed if value > 0]
    return max(parsed) if parsed else ""


def ratio(numerator: int, denominator: int) -> float | str:
    return numerator / denominator if denominator > 0 else ""


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


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    parsed = as_float(value)
    return 0 if parsed is None else int(parsed)


def int_or_zero(value: Any) -> int:
    return as_int(value)
