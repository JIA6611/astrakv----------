"""Build policy ablation tables from existing AstraKV-W artifacts.

The analyzer is intentionally read-only. It does not launch vLLM, LMCache, or
prefetch jobs. Instead, it aggregates already generated benchmark,
prefetch, and chunk-score CSVs into one competition-facing report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - config files are optional.
    yaml = None


BENCHMARK_NUMERIC_FIELDS = (
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

POLICY_KINDS = (
    "no_prefetch",
    "lmcache_default",
    "next_n",
    "lru_aware",
    "reuse_aware",
    "deadline_aware",
    "astrakv_combined",
)


@dataclass(frozen=True, slots=True)
class PolicyInput:
    label: str
    kind: str = "custom"
    benchmark_path: Path | None = None
    prefetch_path: Path | None = None
    chunk_scores_path: Path | None = None
    notes: str = ""


def main() -> int:
    args = parse_args()
    policies = load_policy_inputs(args)
    if not policies:
        raise SystemExit("No policy inputs were provided. Use --config or artifact arguments.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize_policies(policies)
    results_path = output_dir / "policy_ablation_results.csv"
    report_path = output_dir / "policy_ablation_report.md"
    manifest_path = output_dir / "policy_ablation_manifest.json"

    write_csv(results_path, rows)
    write_report(report_path, policies, rows)
    write_manifest(manifest_path, args, policies)
    print(f"Policy ablation CSV written to {results_path}")
    print(f"Policy ablation report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML policy matrix.")
    parser.add_argument("--output-dir", default="results/policy_ablation")
    parser.add_argument(
        "--benchmark-run",
        action="append",
        default=[],
        help="Benchmark artifact as label=path. Path can be a run dir or benchmark_results.csv.",
    )
    parser.add_argument(
        "--prefetch-run",
        action="append",
        default=[],
        help="Prefetch artifact as label=path. Path can be a run dir or prefetch_results.csv.",
    )
    parser.add_argument(
        "--chunk-scores",
        action="append",
        default=[],
        help="Chunk score artifact as label=path. Path can be a dir or chunk_scores.csv.",
    )
    return parser.parse_args()


def load_policy_inputs(args: argparse.Namespace) -> list[PolicyInput]:
    policies: dict[str, PolicyInput] = {}
    for policy in load_config_policies(args.config):
        policies[policy.label] = policy

    for raw in args.benchmark_run:
        label, path = parse_labeled_path(raw)
        policies[label] = merge_policy(
            policies.get(label),
            label=label,
            benchmark_path=resolve_artifact_path(path, "benchmark_results.csv"),
        )
    for raw in args.prefetch_run:
        label, path = parse_labeled_path(raw)
        policies[label] = merge_policy(
            policies.get(label),
            label=label,
            prefetch_path=resolve_artifact_path(path, "prefetch_results.csv"),
        )
    for raw in args.chunk_scores:
        label, path = parse_labeled_path(raw)
        policies[label] = merge_policy(
            policies.get(label),
            label=label,
            chunk_scores_path=resolve_artifact_path(path, "chunk_scores.csv"),
        )
    return list(policies.values())


def load_config_policies(path: str | None) -> list[PolicyInput]:
    if not path:
        return []
    if yaml is None:
        raise SystemExit("PyYAML is required for --config. Install requirements.txt first.")
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    raw_policies = config.get("policies", []) if isinstance(config, dict) else []
    if not isinstance(raw_policies, list):
        raise SystemExit(f"Config policies must be a list: {config_path}")

    policies: list[PolicyInput] = []
    for item in raw_policies:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        policies.append(
            PolicyInput(
                label=label,
                kind=str(item.get("kind") or infer_policy_kind(label)),
                benchmark_path=optional_artifact_path(item.get("benchmark"), "benchmark_results.csv"),
                prefetch_path=optional_artifact_path(item.get("prefetch"), "prefetch_results.csv"),
                chunk_scores_path=optional_artifact_path(item.get("chunk_scores"), "chunk_scores.csv"),
                notes=str(item.get("notes") or ""),
            )
        )
    return policies


def merge_policy(
    current: PolicyInput | None,
    *,
    label: str,
    benchmark_path: Path | None = None,
    prefetch_path: Path | None = None,
    chunk_scores_path: Path | None = None,
) -> PolicyInput:
    if current is None:
        return PolicyInput(
            label=label,
            kind=infer_policy_kind(label),
            benchmark_path=benchmark_path,
            prefetch_path=prefetch_path,
            chunk_scores_path=chunk_scores_path,
        )
    return PolicyInput(
        label=current.label,
        kind=current.kind,
        benchmark_path=benchmark_path or current.benchmark_path,
        prefetch_path=prefetch_path or current.prefetch_path,
        chunk_scores_path=chunk_scores_path or current.chunk_scores_path,
        notes=current.notes,
    )


def summarize_policies(policies: list[PolicyInput]) -> list[dict[str, Any]]:
    rows = [summarize_policy(policy) for policy in policies]
    baseline = next((row for row in rows if int_value(row.get("benchmark_cases")) > 0), None)
    for row in rows:
        attach_baseline_deltas(row, baseline)
    return rows


def summarize_policy(policy: PolicyInput) -> dict[str, Any]:
    benchmark = summarize_benchmark(policy.benchmark_path)
    prefetch = summarize_prefetch(policy.prefetch_path)
    scores = summarize_chunk_scores(policy.chunk_scores_path)
    missing = missing_metric_groups(benchmark, prefetch, scores)
    artifact_status = artifact_summary(policy)
    return {
        "policy": policy.label,
        "policy_kind": policy.kind,
        "artifact_status": artifact_status,
        "benchmark_cases": benchmark["case_count"],
        "prefetch_cases": prefetch["case_count"],
        "chunks_scored": scores["chunks_scored"],
        "request_count": benchmark["request_count"],
        "success_count": benchmark["success_count"],
        "success_rate": benchmark["success_rate"],
        "ttft_ms_mean": benchmark["ttft_ms_mean"],
        "tpot_ms_mean": benchmark["tpot_ms_mean"],
        "latency_ms_mean": benchmark["latency_ms_mean"],
        "latency_p95_ms_mean": benchmark["latency_p95_ms_mean"],
        "throughput_tokens_s_mean": benchmark["throughput_tokens_s_mean"],
        "gpu_memory_peak_mb": benchmark["gpu_memory_peak_mb"],
        "process_rss_peak_mb": benchmark["process_rss_peak_mb"],
        "cpu_memory_peak_mb": benchmark["cpu_memory_peak_mb"],
        "disk_read_delta_mb": benchmark["disk_read_delta_mb"],
        "disk_write_delta_mb": benchmark["disk_write_delta_mb"],
        "prefetch_submitted": prefetch["prefetch_submitted"],
        "prefetch_completed": prefetch["prefetch_completed"],
        "prefetch_failed": prefetch["prefetch_failed"],
        "prefetch_hit": prefetch["prefetch_hit"],
        "prefetch_waste": prefetch["prefetch_waste"],
        "prefetch_hit_rate": prefetch["prefetch_hit_rate"],
        "prefetch_waste_rate": prefetch["prefetch_waste_rate"],
        "chunk_action_prefetch": scores["action_prefetch"],
        "chunk_action_keep": scores["action_keep"],
        "chunk_action_offload": scores["action_offload"],
        "chunk_action_drop": scores["action_drop"],
        "chunk_score_mean": scores["score_mean"],
        "missing_metric_groups": ";".join(missing),
        "notes": policy.notes,
    }


def summarize_benchmark(path: Path | None) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return empty_benchmark_summary()

    request_count = sum(as_float(row.get("request_count")) or 0.0 for row in rows)
    success_count = sum(as_float(row.get("success_count")) or 0.0 for row in rows)
    summary = empty_benchmark_summary()
    summary.update(
        {
            "case_count": len(rows),
            "request_count": int_if_whole(request_count),
            "success_count": int_if_whole(success_count),
            "success_rate": safe_ratio(success_count, request_count),
        }
    )
    for field in BENCHMARK_NUMERIC_FIELDS:
        values = [value for value in (as_float(row.get(field)) for row in rows) if value is not None]
        if field == "process_rss_peak_mb":
            summary[field] = max(values) if values else ""
        elif field == "cpu_memory_peak_mb":
            if values:
                summary[field] = max(values)
        elif field == "gpu_memory_peak_mb":
            summary[field] = max(values) if values else ""
        elif field in {"disk_read_delta_mb", "disk_write_delta_mb"}:
            summary[field] = sum(values) if values else ""
        else:
            summary[f"{field}_mean"] = mean(values)
    if not summary["process_rss_peak_mb"]:
        summary["process_rss_peak_mb"] = summary["cpu_memory_peak_mb"]
    return summary


def empty_benchmark_summary() -> dict[str, Any]:
    return {
        "case_count": 0,
        "request_count": "",
        "success_count": "",
        "success_rate": "",
        "ttft_ms_mean": "",
        "tpot_ms_mean": "",
        "latency_ms_mean": "",
        "latency_p95_ms_mean": "",
        "throughput_tokens_s_mean": "",
        "gpu_memory_peak_mb": "",
        "process_rss_peak_mb": "",
        "cpu_memory_peak_mb": "",
        "disk_read_delta_mb": "",
        "disk_write_delta_mb": "",
    }


def summarize_prefetch(path: Path | None) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return empty_prefetch_summary()

    submitted = sum(as_float(row.get("prefetch_submitted")) or 0.0 for row in rows)
    completed = sum(as_float(row.get("prefetch_completed")) or 0.0 for row in rows)
    failed = sum(as_float(row.get("prefetch_failed")) or 0.0 for row in rows)
    hits = sum(as_float(row.get("prefetch_hit")) or 0.0 for row in rows)
    waste = sum(as_float(row.get("prefetch_waste")) or 0.0 for row in rows)
    return {
        "case_count": len(rows),
        "prefetch_submitted": int_if_whole(submitted),
        "prefetch_completed": int_if_whole(completed),
        "prefetch_failed": int_if_whole(failed),
        "prefetch_hit": int_if_whole(hits),
        "prefetch_waste": int_if_whole(waste),
        "prefetch_hit_rate": safe_ratio(hits, completed),
        "prefetch_waste_rate": safe_ratio(waste, completed),
    }


def empty_prefetch_summary() -> dict[str, Any]:
    return {
        "case_count": 0,
        "prefetch_submitted": "",
        "prefetch_completed": "",
        "prefetch_failed": "",
        "prefetch_hit": "",
        "prefetch_waste": "",
        "prefetch_hit_rate": "",
        "prefetch_waste_rate": "",
    }


def summarize_chunk_scores(path: Path | None) -> dict[str, Any]:
    rows = read_csv_rows(path)
    summary = {
        "chunks_scored": len(rows),
        "action_prefetch": 0,
        "action_keep": 0,
        "action_offload": 0,
        "action_drop": 0,
        "score_mean": "",
    }
    if not rows:
        return summary

    scores: list[float] = []
    for row in rows:
        action = str(row.get("action") or "").strip().lower()
        key = f"action_{action}"
        if key in summary:
            summary[key] = int(summary[key]) + 1
        value = as_float(row.get("score"))
        if value is not None:
            scores.append(value)
    summary["score_mean"] = mean(scores)
    return summary


def missing_metric_groups(
    benchmark: dict[str, Any],
    prefetch: dict[str, Any],
    scores: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if int_value(benchmark["case_count"]) == 0:
        missing.append("missing_benchmark")
    if int_value(prefetch["case_count"]) == 0:
        missing.append("missing_prefetch")
    if int_value(scores["chunks_scored"]) == 0:
        missing.append("missing_chunk_scores")
    return missing


def attach_baseline_deltas(row: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    if baseline is None or row is baseline:
        row.update(empty_delta_fields())
        return
    row["ttft_delta_pct_vs_baseline"] = delta_pct(
        as_float(row.get("ttft_ms_mean")),
        as_float(baseline.get("ttft_ms_mean")),
    )
    row["tpot_delta_pct_vs_baseline"] = delta_pct(
        as_float(row.get("tpot_ms_mean")),
        as_float(baseline.get("tpot_ms_mean")),
    )
    row["latency_delta_pct_vs_baseline"] = delta_pct(
        as_float(row.get("latency_ms_mean")),
        as_float(baseline.get("latency_ms_mean")),
    )
    row["throughput_delta_pct_vs_baseline"] = delta_pct(
        as_float(row.get("throughput_tokens_s_mean")),
        as_float(baseline.get("throughput_tokens_s_mean")),
    )
    row["process_rss_delta_pct_vs_baseline"] = delta_pct(
        as_float(row.get("process_rss_peak_mb") or row.get("cpu_memory_peak_mb")),
        as_float(baseline.get("process_rss_peak_mb") or baseline.get("cpu_memory_peak_mb")),
    )
    row["gpu_memory_reduction_pct_vs_baseline"] = reduction_pct(
        as_float(row.get("gpu_memory_peak_mb")),
        as_float(baseline.get("gpu_memory_peak_mb")),
    )


def empty_delta_fields() -> dict[str, Any]:
    return {
        "ttft_delta_pct_vs_baseline": "",
        "tpot_delta_pct_vs_baseline": "",
        "latency_delta_pct_vs_baseline": "",
        "throughput_delta_pct_vs_baseline": "",
        "process_rss_delta_pct_vs_baseline": "",
        "gpu_memory_reduction_pct_vs_baseline": "",
    }


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.is_dir():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, policies: list[PolicyInput], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Policy Ablation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Policy Registry",
        "",
        "| policy | kind | benchmark | prefetch | chunk scores | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for policy in policies:
        lines.append(
            "| {label} | {kind} | {benchmark} | {prefetch} | {scores} | {notes} |".format(
                label=policy.label,
                kind=policy.kind,
                benchmark=format_path(policy.benchmark_path),
                prefetch=format_path(policy.prefetch_path),
                scores=format_path(policy.chunk_scores_path),
                notes=policy.notes,
            )
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| policy | cases | success | TTFT ms | TPOT ms | latency ms | throughput tok/s | RSS MB | RSS delta % | disk read MB | disk write MB | prefetch hit rate | waste rate | chunk actions | missing |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        actions = (
            f"P:{row['chunk_action_prefetch']} "
            f"K:{row['chunk_action_keep']} "
            f"O:{row['chunk_action_offload']} "
            f"D:{row['chunk_action_drop']}"
        )
        lines.append(
            "| {policy} | {benchmark_cases} | {success_rate} | {ttft_ms_mean} | "
            "{tpot_ms_mean} | {latency_ms_mean} | {throughput_tokens_s_mean} | "
            "{process_rss_peak_mb} | {process_rss_delta_pct_vs_baseline} | "
            "{disk_read_delta_mb} | {disk_write_delta_mb} | "
            "{prefetch_hit_rate} | {prefetch_waste_rate} | {actions} | "
            "{missing_metric_groups} |".format(actions=actions, **row)
        )

    lines.extend(
        [
            "",
            "## Metric Meaning",
            "",
            "- `benchmark` metrics come from real endpoint `benchmark_results.csv` artifacts.",
            "- `prefetch` metrics come from endpoint-level selective prefetch `prefetch_results.csv` artifacts.",
            "- `chunk actions` come from ProfileDB-driven `chunk_scores.csv` and are advisory policy evidence.",
            "- The first policy with benchmark cases is used as the baseline for delta columns.",
            "- RSS and disk IO are the stable memory-pressure metrics on DGX Spark; per-GPU memory may be unavailable.",
            "- Empty cells mean the metric was unavailable in the provided artifacts, not zero.",
            "",
            "## Contest Interpretation",
            "",
            "- `no_prefetch` establishes cold or default demand serving behavior.",
            "- `lmcache_default` shows memory tiering without AstraKV-W policy guidance.",
            "- `next_n`, `lru_aware`, `reuse_aware`, and `deadline_aware` can be represented by separate scored/prefetch artifacts under the same workload.",
            "- `astrakv_combined` should combine ProfileDB scoring, selective prefetch evidence, and real benchmark deltas.",
            "",
            "## Missing Evidence",
            "",
        ]
    )
    missing_rows = [row for row in rows if row["missing_metric_groups"]]
    if missing_rows:
        lines.extend(
            [
                "| policy | missing groups |",
                "| --- | --- |",
            ]
        )
        for row in missing_rows:
            lines.append(f"| {row['policy']} | {row['missing_metric_groups']} |")
    else:
        lines.append("All policies have benchmark, prefetch, and chunk-score artifacts.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `policy_ablation_results.csv`",
            "- `policy_ablation_report.md`",
            "- `policy_ablation_manifest.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, policies: list[PolicyInput]) -> None:
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "policies": [
            {
                "label": policy.label,
                "kind": policy.kind,
                "benchmark_path": str(policy.benchmark_path or ""),
                "prefetch_path": str(policy.prefetch_path or ""),
                "chunk_scores_path": str(policy.chunk_scores_path or ""),
                "notes": policy.notes,
            }
            for policy in policies
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def artifact_summary(policy: PolicyInput) -> str:
    parts = []
    for name, path in (
        ("benchmark", policy.benchmark_path),
        ("prefetch", policy.prefetch_path),
        ("chunk_scores", policy.chunk_scores_path),
    ):
        if path is None:
            parts.append(f"{name}:none")
        elif path.exists():
            parts.append(f"{name}:ok")
        else:
            parts.append(f"{name}:missing")
    return ";".join(parts)


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(value.strip())
        label = path.stem
    if not label:
        raise SystemExit(f"Artifact label cannot be empty: {value}")
    return label, path


def optional_artifact_path(value: Any, default_name: str) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_artifact_path(Path(str(value)), default_name)


def resolve_artifact_path(path: Path, default_name: str) -> Path:
    if path.suffix.lower() == ".csv":
        return path
    return path / default_name


def infer_policy_kind(label: str) -> str:
    normalized = label.lower().replace("-", "_")
    for kind in POLICY_KINDS:
        if kind in normalized:
            return kind
    if "lmcache" in normalized:
        return "lmcache_default"
    if "astrakv" in normalized or "combined" in normalized:
        return "astrakv_combined"
    return "custom"


def format_path(path: Path | None) -> str:
    if path is None:
        return ""
    return f"`{path}`"


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: Any) -> int:
    number = as_float(value)
    if number is None:
        return 0
    return int(number)


def int_if_whole(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return value


def safe_ratio(numerator: float, denominator: float) -> float | str:
    if denominator == 0:
        return ""
    return numerator / denominator


def mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def delta_pct(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline in (None, 0):
        return ""
    return (value - baseline) / baseline * 100.0


def reduction_pct(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline in (None, 0):
        return ""
    return (baseline - value) / baseline * 100.0


if __name__ == "__main__":
    raise SystemExit(main())
