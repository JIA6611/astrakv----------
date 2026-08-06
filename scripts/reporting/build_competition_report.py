"""Build a competition-grade aggregate report from AstraKV-W artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    label: str
    kind: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def size_bytes(self) -> int:
        if not self.exists or self.path.is_dir():
            return 0
        return self.path.stat().st_size


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(args)
    summaries = summarize_artifacts(artifacts)

    inventory_path = output_dir / args.inventory_name
    manifest_path = output_dir / args.manifest_name
    report_path = output_dir / args.report_name
    write_inventory(inventory_path, artifacts, summaries)
    write_manifest(manifest_path, args, artifacts, summaries)
    write_report(report_path, args, artifacts, summaries, inventory_path, manifest_path)
    print(f"Competition report written to {report_path}")
    print(f"Competition manifest written to {manifest_path}")
    print(f"Artifact inventory written to {inventory_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/competition_report")
    parser.add_argument("--report-name", default="competition_report.md")
    parser.add_argument("--manifest-name", default="competition_report_manifest.json")
    parser.add_argument("--inventory-name", default="artifact_inventory.csv")
    parser.add_argument("--title", default="AstraKV-W Competition Report")
    parser.add_argument("--command", action="append", default=[], help="Command line to record in the report.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact as label:kind=path, for example vllm:benchmark=results/run/benchmark_results.csv.",
    )
    parser.add_argument("--benchmark", action="append", default=[], help="Shortcut for label=path benchmark artifact.")
    parser.add_argument("--comparison", default="", help="Comparison CSV or report path.")
    parser.add_argument("--policy-ablation", default="", help="Policy ablation CSV or report path.")
    parser.add_argument("--stress", default="", help="Stress summary CSV or report path.")
    parser.add_argument("--trace-summary", default="", help="Trace summary Markdown path.")
    parser.add_argument("--profile-db", default="", help="ProfileDB JSON path.")
    parser.add_argument("--chunk-scores", default="", help="Chunk scores CSV path.")
    parser.add_argument("--partial-load", default="", help="Partial KV load summary CSV/report path.")
    parser.add_argument("--load-recompute", default="", help="Load-vs-recompute decisions CSV/report path.")
    parser.add_argument("--object-schedule", default="", help="Unified object schedule CSV/report path.")
    parser.add_argument("--quality", default="", help="Quality results CSV/report path.")
    parser.add_argument("--workload-manifest", default="", help="Workload manifest JSON path.")
    parser.add_argument("--vm-evidence", action="append", default=[], help="VM evidence as label=path.")
    parser.add_argument("--prefetch", action="append", default=[], help="Prefetch artifact as label=path.")
    parser.add_argument("--cache-events", action="append", default=[], help="Cache event JSONL artifact as label=path.")
    parser.add_argument("--server-log", action="append", default=[], help="vLLM server log as label=path.")
    parser.add_argument("--eviction-agreement", action="append", default=[], help="Eviction agreement manifest as label=path.")
    parser.add_argument("--offline-policy-summary", action="append", default=[], help="Offline simulator CSV as label=path.")
    parser.add_argument("--offline-safety-gate", action="append", default=[], help="Offline safety gate JSON as label=path.")
    parser.add_argument("--experiment-manifest", action="append", default=[], help="Shared experiment manifest JSON as label=path.")
    parser.add_argument("--diagnostic", action="append", default=[], help="Runtime diagnostic manifest JSON as label=path.")
    return parser.parse_args()


def load_artifacts(args: argparse.Namespace) -> list[ArtifactInput]:
    artifacts: list[ArtifactInput] = []
    for item in args.artifact:
        artifacts.append(parse_artifact_arg(item))
    for item in args.benchmark:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="benchmark", path=resolve_path(path, "benchmark_results.csv")))
    for item in args.vm_evidence:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="vm_evidence", path=path))
    for item in args.prefetch:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="prefetch", path=resolve_path(path, "prefetch_results.csv")))
    for item in args.cache_events:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="cache_events", path=resolve_path(path, "cache_events.jsonl")))
    for item in args.server_log:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="server_log", path=path))
    for item in args.eviction_agreement:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="eviction_agreement", path=resolve_path(path, "eviction_agreement_manifest.json")))
    for item in args.offline_policy_summary:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="offline_policy", path=resolve_path(path, "offline_eviction_policy_summary.csv")))
    for item in args.offline_safety_gate:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="offline_safety_gate", path=path))
    for item in args.experiment_manifest:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="experiment_manifest", path=path))
    for item in args.diagnostic:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="diagnostic", path=path))
    for label, kind, raw_path in (
        ("comparison", "comparison", args.comparison),
        ("policy_ablation", "policy_ablation", args.policy_ablation),
        ("stress", "stress", args.stress),
        ("trace_summary", "trace", args.trace_summary),
        ("profile_db", "profile", args.profile_db),
        ("chunk_scores", "chunk_scores", args.chunk_scores),
        ("partial_load", "partial_load", args.partial_load),
        ("load_recompute", "load_recompute", args.load_recompute),
        ("object_schedule", "object_schedule", args.object_schedule),
        ("quality", "quality", args.quality),
        ("workload_manifest", "workload", args.workload_manifest),
    ):
        if raw_path:
            artifacts.append(ArtifactInput(label=label, kind=kind, path=Path(raw_path)))
    return artifacts


def parse_artifact_arg(value: str) -> ArtifactInput:
    if "=" not in value:
        raise SystemExit(f"--artifact must use label:kind=path format: {value}")
    left, raw_path = value.split("=", 1)
    if ":" in left:
        label, kind = left.split(":", 1)
    else:
        label, kind = left, "artifact"
    if not label.strip():
        raise SystemExit(f"Artifact label cannot be empty: {value}")
    return ArtifactInput(label=label.strip(), kind=kind.strip() or "artifact", path=Path(raw_path.strip()))


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        return Path(value).stem, Path(value)
    label, raw_path = value.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def resolve_path(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def summarize_artifacts(artifacts: list[ArtifactInput]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for artifact in artifacts:
        key = artifact_key(artifact)
        summaries[key] = summarize_artifact(artifact)
    return summaries


def summarize_artifact(artifact: ArtifactInput) -> dict[str, Any]:
    if not artifact.exists:
        return {"status": "missing", "path": str(artifact.path), "rows": 0}
    if artifact.path.suffix.lower() == ".csv":
        rows = read_csv_rows(artifact.path)
        return summarize_csv_artifact(artifact, rows)
    if artifact.path.suffix.lower() == ".json":
        return summarize_json_artifact(artifact)
    if artifact.path.suffix.lower() in {".jsonl", ".ndjson"}:
        return summarize_jsonl_artifact(artifact)
    if artifact.path.suffix.lower() in {".md", ".txt"}:
        text = artifact.path.read_text(encoding="utf-8", errors="replace")
        summary = {"status": "ok", "path": str(artifact.path), "lines": len(text.splitlines()), "size_bytes": artifact.size_bytes}
        if artifact.kind == "server_log":
            summary.update(summarize_server_log_text(text))
        return summary
    if artifact.path.suffix.lower() == ".log":
        text = artifact.path.read_text(encoding="utf-8", errors="replace")
        summary = {"status": "ok", "path": str(artifact.path), "lines": len(text.splitlines()), "size_bytes": artifact.size_bytes}
        if artifact.kind == "server_log":
            summary.update(summarize_server_log_text(text))
        return summary
    return {"status": "ok", "path": str(artifact.path), "size_bytes": artifact.size_bytes}


def summarize_csv_artifact(artifact: ArtifactInput, rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": "ok", "path": str(artifact.path), "rows": len(rows), "size_bytes": artifact.size_bytes}
    if artifact.kind == "benchmark":
        summary.update(
            {
                "success_rate": weighted_success_rate(rows),
                "mean_ttft_ms": mean_field(rows, "ttft_ms"),
                "mean_tpot_ms": mean_field(rows, "tpot_ms"),
                "mean_latency_p95_ms": mean_field(rows, "latency_p95_ms"),
                "max_gpu_memory_peak_mb": max_field(rows, "gpu_memory_peak_mb"),
                "max_process_rss_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
                "max_cpu_memory_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
                "max_gpu_util_peak_pct": max_field(rows, "gpu_util_peak_pct"),
                "disk_read_delta_mb": sum_field(rows, "disk_read_delta_mb"),
                "disk_write_delta_mb": sum_field(rows, "disk_write_delta_mb"),
            }
        )
    elif artifact.kind == "quality":
        summary.update(
            {
                "exact_match_rate": mean_field(rows, "exact_match"),
                "normalized_match_rate": mean_field(rows, "normalized_match"),
                "mean_token_divergence_rate": mean_field(rows, "token_divergence_rate"),
                "mean_ppl_delta": mean_field(rows, "ppl_delta"),
            }
        )
    elif artifact.kind == "object_schedule":
        summary.update({"action_counts": count_field(rows, "action"), "gpu_bytes_after_max": max_field(rows, "gpu_bytes_after")})
    elif artifact.kind == "chunk_scores":
        summary.update({"action_counts": count_field(rows, "action"), "mean_score": mean_field(rows, "score")})
    elif artifact.kind in {"policy_ablation", "comparison", "stress", "partial_load", "load_recompute"}:
        summary.update({"columns": list(rows[0].keys()) if rows else []})
        if "action" in (rows[0].keys() if rows else []):
            summary["action_counts"] = count_field(rows, "action")
        if artifact.kind == "stress":
            summary.update(summarize_stress_rows(rows))
    elif artifact.kind == "prefetch":
        summary.update(summarize_prefetch_rows(rows))
    elif artifact.kind == "offline_policy":
        summary.update({"policies": {str(row.get("policy")): row for row in rows}})
    return summary


def summarize_json_artifact(artifact: ArtifactInput) -> dict[str, Any]:
    try:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "path": str(artifact.path), "error": str(exc)}
    summary = {"status": "ok", "path": str(artifact.path), "size_bytes": artifact.size_bytes}
    if artifact.kind == "profile":
        summary["workload_count"] = len(payload.get("workloads", [])) if isinstance(payload, dict) else 0
        summary["chunk_count"] = len(payload.get("chunks", [])) if isinstance(payload, dict) else 0
    elif artifact.kind == "workload":
        nested = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if isinstance(nested, dict):
            summary.update(nested)
    elif artifact.kind == "vm_evidence":
        summary.update(flatten_vm_evidence(payload))
    elif artifact.kind == "eviction_agreement" and isinstance(payload, dict):
        agreement = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        metrics = agreement.get("metrics") if isinstance(agreement.get("metrics"), dict) else {}
        summary.update({
            "comparison_scope": payload.get("comparison_scope", "unknown"),
            "ground_truth_status": agreement.get("ground_truth_status", "missing"),
            "reason": agreement.get("reason", ""),
            "metrics": metrics,
        })
    elif artifact.kind == "offline_safety_gate" and isinstance(payload, dict):
        summary.update({"gate_status": payload.get("status", "missing"), "checks": payload.get("checks", {}), "reasons": payload.get("reasons", [])})
    elif artifact.kind == "experiment_manifest" and isinstance(payload, dict):
        summary.update({"run_id": payload.get("run_id", "unknown"), "model": payload.get("model", "unknown"), "dtype": payload.get("dtype", "unknown"), "quantization": payload.get("quantization", "unknown")})
    elif artifact.kind == "diagnostic" and isinstance(payload, dict):
        summary.update({"mode": payload.get("mode", "unknown"), "capabilities": payload.get("capabilities", {}), "raw_artifacts": payload.get("raw_artifacts", [])})
    else:
        summary["top_level_type"] = type(payload).__name__
    return summary


def summarize_jsonl_artifact(artifact: ArtifactInput) -> dict[str, Any]:
    count = 0
    event_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    with artifact.path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if artifact.kind == "cache_events":
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = str(payload.get("event_type") or payload.get("event") or payload.get("type") or "unknown")
                status = str(payload.get("status") or "unknown")
                event_counts[event] = event_counts.get(event, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1
    summary: dict[str, Any] = {"status": "ok", "path": str(artifact.path), "rows": count, "size_bytes": artifact.size_bytes}
    if artifact.kind == "cache_events":
        summary["event_counts"] = dict(sorted(event_counts.items()))
        summary["status_counts"] = dict(sorted(status_counts.items()))
    return summary


def summarize_prefetch_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    completed = sum(as_float(row.get("prefetch_completed")) or 0.0 for row in rows)
    submitted = sum(as_float(row.get("prefetch_submitted")) or 0.0 for row in rows)
    failed = sum(as_float(row.get("prefetch_failed")) or 0.0 for row in rows)
    hits = sum(as_float(row.get("prefetch_hit")) or 0.0 for row in rows)
    waste = sum(as_float(row.get("prefetch_waste")) or 0.0 for row in rows)
    return {
        "prefetch_submitted": int_if_whole(submitted),
        "prefetch_completed": int_if_whole(completed),
        "prefetch_failed": int_if_whole(failed),
        "prefetch_hit": int_if_whole(hits),
        "prefetch_waste": int_if_whole(waste),
        "prefetch_hit_rate": hits / completed if completed else "",
        "prefetch_waste_rate": waste / completed if completed else "",
        "mean_ttft_delta_pct": mean_field(rows, "ttft_delta_pct"),
        "mean_latency_delta_pct": mean_field(rows, "latency_delta_pct"),
    }


def summarize_stress_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if rows and "run" in rows[0] and "total_cases" in rows[0]:
        total_cases = sum(as_float(row.get("total_cases")) or 0.0 for row in rows)
        failed_cases = sum(as_float(row.get("failed_case_count")) or 0.0 for row in rows)
        return {
            "total_cases": int_if_whole(total_cases),
            "successful_cases": int_if_whole(max(0.0, total_cases - failed_cases)),
            "failed_cases": int_if_whole(failed_cases),
            "oom_case_count": int_if_whole(sum(as_float(row.get("oom_case_count")) or 0.0 for row in rows)),
            "case_success_rate": "" if total_cases == 0 else max(0.0, total_cases - failed_cases) / total_cases,
            "max_success_context": max_field(rows, "max_success_context"),
            "max_success_batch": max_field(rows, "max_success_batch"),
            "max_gpu_memory_peak_mb": max_field(rows, "gpu_memory_peak_mb"),
            "max_cpu_memory_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
            "disk_read_delta_mb": sum_field(rows, "disk_read_delta_mb"),
            "disk_write_delta_mb": sum_field(rows, "disk_write_delta_mb"),
        }

    total = len(rows)
    successful = 0
    failed = 0
    oom = 0
    max_success_context = ""
    max_success_batch = ""
    for row in rows:
        request_count = as_float(row.get("request_count")) or 0.0
        success_count = as_float(row.get("success_count")) or 0.0
        row_success = request_count > 0 and success_count >= request_count
        if row_success:
            successful += 1
            context = as_float(row.get("context_length"))
            batch = as_float(row.get("batch_size"))
            if context is not None:
                max_success_context = max(float(max_success_context or 0), context)
            if batch is not None:
                max_success_batch = max(float(max_success_batch or 0), batch)
        else:
            failed += 1
        if looks_oom(str(row.get("errors") or "")):
            oom += 1
    return {
        "total_cases": total,
        "successful_cases": successful,
        "failed_cases": failed,
        "oom_case_count": oom,
        "case_success_rate": successful / total if total else "",
        "max_success_context": int_if_whole(max_success_context) if isinstance(max_success_context, float) else "",
        "max_success_batch": int_if_whole(max_success_batch) if isinstance(max_success_batch, float) else "",
        "max_gpu_memory_peak_mb": max_field(rows, "gpu_memory_peak_mb"),
        "max_cpu_memory_peak_mb": max_field_with_fallback(rows, "process_rss_peak_mb", "cpu_memory_peak_mb"),
        "disk_read_delta_mb": sum_field(rows, "disk_read_delta_mb"),
        "disk_write_delta_mb": sum_field(rows, "disk_write_delta_mb"),
    }


def flatten_vm_evidence(payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not isinstance(payload, dict):
        summary["top_level_type"] = type(payload).__name__
        return summary
    source = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    for key, value in source.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
    for key, value in payload.items():
        if key in summary or key == "summary":
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    summary[f"{key}_{nested_key}"] = nested_value
    return summary


def summarize_server_log_text(text: str) -> dict[str, Any]:
    model_loading = extract_first_float(text, r"Model loading took\s+([0-9.]+)\s+GiB\s+memory")
    kv_available = extract_first_float(text, r"Available KV cache memory:\s*([0-9.]+)\s+GiB")
    kv_tokens = extract_first_int(text, r"GPU KV cache size:\s*([0-9,]+)\s+tokens")
    evidence_lines = []
    for line in text.splitlines():
        if any(pattern in line for pattern in ("Model loading took", "Available KV cache memory", "GPU KV cache size")):
            evidence_lines.append(line.strip())
        if len(evidence_lines) >= 6:
            break
    return {
        "model_loading_memory_gib": model_loading if model_loading is not None else "",
        "available_kv_cache_memory_gib": kv_available if kv_available is not None else "",
        "gpu_kv_cache_tokens": kv_tokens if kv_tokens is not None else "",
        "startup_evidence_lines": evidence_lines,
    }


def extract_first_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return as_float(match.group(1))


def extract_first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def looks_oom(text: str) -> bool:
    lowered = text.lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "oom" in lowered


def write_inventory(path: Path, artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    fieldnames = ["label", "kind", "path", "exists", "status", "rows", "size_bytes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            summary = summaries.get(artifact_key(artifact), {})
            writer.writerow(
                {
                    "label": artifact.label,
                    "kind": artifact.kind,
                    "path": artifact.path,
                    "exists": int(artifact.exists),
                    "status": summary.get("status", "missing"),
                    "rows": summary.get("rows", ""),
                    "size_bytes": summary.get("size_bytes", artifact.size_bytes),
                }
            )


def write_manifest(path: Path, args: argparse.Namespace, artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "title": args.title,
        "environment": collect_environment(),
        "git": collect_git_info(),
        "commands": args.command,
        "artifacts": [
            {
                "label": artifact.label,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "summary": summaries.get(artifact_key(artifact), {}),
            }
            for artifact in artifacts
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    artifacts: list[ArtifactInput],
    summaries: dict[str, Any],
    inventory_path: Path,
    manifest_path: Path,
) -> None:
    env = collect_environment()
    git = collect_git_info()
    lines = [
        f"# {args.title}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Environment",
        "",
        f"- Python: `{env['python']}`",
        f"- Platform: `{env['platform']}`",
        f"- Machine: `{env['machine']}`",
        f"- Git commit: `{git.get('commit', 'unknown')}`",
        f"- Git dirty: `{git.get('dirty', 'unknown')}`",
        "",
        "## Commands",
        "",
    ]
    if args.command:
        for command in args.command:
            lines.append(f"- `{command}`")
    else:
        lines.append("- No commands were provided. Add `--command` entries for official reports.")

    lines.extend(
        [
            "",
            "## Artifact Inventory",
            "",
            "| label | kind | status | rows | path |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for artifact in artifacts:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {artifact.kind} | {summary.get('status', 'missing')} | "
            f"{summary.get('rows', '')} | `{artifact.path}` |"
        )

    append_benchmark_section(lines, artifacts, summaries)
    append_quality_section(lines, artifacts, summaries)
    append_vm_evidence_section(lines, artifacts, summaries)
    append_eviction_agreement_section(lines, artifacts, summaries)
    append_offline_policy_section(lines, artifacts, summaries)
    append_offline_safety_gate_section(lines, artifacts, summaries)
    append_diagnostic_section(lines, artifacts, summaries)
    append_prefetch_section(lines, artifacts, summaries)
    append_cache_event_section(lines, artifacts, summaries)
    append_server_log_section(lines, artifacts, summaries)
    append_stress_boundary_section(lines, artifacts, summaries)
    append_policy_section(lines, artifacts, summaries)
    append_scheduler_section(lines, artifacts, summaries)
    append_profile_workload_section(lines, artifacts, summaries)
    append_missing_section(lines, artifacts, summaries)

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Empty or missing metrics mean no artifact was provided, not zero improvement.",
            "- Advisory scheduler, partial-load, and load-vs-recompute reports require backend adapter evidence before claiming physical execution.",
            "- Endpoint-level selective prefetch hit evidence is heuristic unless supported by cache events.",
            "- PPL is only reported when upstream records include PPL/loss/NLL evidence.",
            "",
            "## Artifacts",
            "",
            f"- `{inventory_path}`",
            f"- `{manifest_path}`",
            "- `competition_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def append_benchmark_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    benchmark_items = [artifact for artifact in artifacts if artifact.kind == "benchmark"]
    lines.extend(["", "## Benchmark Metrics", "", "| run | success | TTFT ms | TPOT ms | latency p95 ms | RSS MB | GPU util % | disk read MB | disk write MB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    if not benchmark_items:
        lines.append("| none |  |  |  |  |  |  |  |  |")
        return
    for artifact in benchmark_items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {fmt(summary.get('success_rate'))} | {fmt(summary.get('mean_ttft_ms'))} | "
            f"{fmt(summary.get('mean_tpot_ms'))} | {fmt(summary.get('mean_latency_p95_ms'))} | "
            f"{fmt(summary.get('max_process_rss_peak_mb'))} | {fmt(summary.get('max_gpu_util_peak_pct'))} | "
            f"{fmt(summary.get('disk_read_delta_mb'))} | {fmt(summary.get('disk_write_delta_mb'))} |"
        )


def append_quality_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "quality"]
    lines.extend(["", "## Quality Metrics", "", "| artifact | exact match | normalized match | token divergence | PPL delta |", "| --- | ---: | ---: | ---: | ---: |"])
    if not items:
        lines.append("| none |  |  |  |  |")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {fmt(summary.get('exact_match_rate'))} | "
            f"{fmt(summary.get('normalized_match_rate'))} | {fmt(summary.get('mean_token_divergence_rate'))} | "
            f"{fmt(summary.get('mean_ppl_delta'))} |"
        )


def append_vm_evidence_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "vm_evidence"]
    lines.extend(["", "## VM Evidence", "", "| artifact | status | rows | key evidence | path |", "| --- | --- | ---: | --- | --- |"])
    if not items:
        lines.append("| none |  |  |  |  |")
        return
    preferred = (
        "resident_ratio",
        "prefetch_coverage_rate",
        "cold_latency_ms",
        "warm_latency_ms",
        "speedup",
        "total_blocks",
        "eviction_requests",
        "prefetch_requests",
    )
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        facts = format_summary_facts(summary, preferred)
        lines.append(
            f"| {artifact.label} | {summary.get('status', 'missing')} | {summary.get('rows', '')} | {facts} | `{artifact.path}` |"
        )


def append_eviction_agreement_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "eviction_agreement"]
    lines.extend(["", "## 真实 Runtime 一致性", "", "| artifact | status | precision | recall | F1 | object coverage | reason |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    runtime_items = [item for item in items if summaries.get(artifact_key(item), {}).get("comparison_scope") == "runtime"]
    if not runtime_items:
        lines.append("| none | insufficient_ground_truth |  |  |  |  | no runtime agreement manifest provided |")
    for artifact in runtime_items:
        summary = summaries.get(artifact_key(artifact), {})
        metrics = summary.get("metrics", {}) if summary.get("ground_truth_status") == "valid" else {}
        lines.append(
            f"| {artifact.label} | {summary.get('ground_truth_status', 'missing')} | {fmt(metrics.get('precision'))} | "
            f"{fmt(metrics.get('recall'))} | {fmt(metrics.get('f1'))} | {fmt(metrics.get('object_coverage'))} | "
            f"{summary.get('reason', '')} |"
        )
    lines.extend(["", "## VM PoC 逻辑对象一致性", "", "| artifact | status | precision | recall | F1 | object coverage | reason |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    vm_items = [item for item in items if summaries.get(artifact_key(item), {}).get("comparison_scope") == "vm_poc"]
    if not vm_items:
        lines.append("| none | insufficient_ground_truth |  |  |  |  | no VM-PoC agreement manifest provided |")
    for artifact in vm_items:
        summary = summaries.get(artifact_key(artifact), {})
        metrics = summary.get("metrics", {}) if summary.get("ground_truth_status") == "valid" else {}
        lines.append(
            f"| {artifact.label} | {summary.get('ground_truth_status', 'missing')} | {fmt(metrics.get('precision'))} | "
            f"{fmt(metrics.get('recall'))} | {fmt(metrics.get('f1'))} | {fmt(metrics.get('object_coverage'))} | "
            f"{summary.get('reason', '')} |"
        )


def append_offline_policy_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [item for item in artifacts if item.kind == "offline_policy"]
    lines.extend(["", "## Offline Policy Comparison", "", "| artifact | policy | hit rate | migration bytes | OOM unavoided | timing mode |", "| --- | --- | ---: | ---: | ---: | --- |"])
    if not items:
        lines.append("| none |  |  |  |  |  |")
    for artifact in items:
        for policy, row in summaries.get(artifact_key(artifact), {}).get("policies", {}).items():
            lines.append(f"| {artifact.label} | {policy} | {fmt(row.get('total_hit_rate'))} | {fmt(row.get('migration_bytes'))} | {fmt(row.get('oom_unavoided'))} | {row.get('timing_mode', '')} |")


def append_offline_safety_gate_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [item for item in artifacts if item.kind == "offline_safety_gate"]
    lines.extend(["", "## Offline Safety Gate", "", "| artifact | status | checks | reasons |", "| --- | --- | --- | --- |"])
    if not items:
        lines.append("| none | missing |  | no gate artifact provided |")
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        checks = summary.get("checks", {})
        check_text = ", ".join(f"{key}={value}" for key, value in checks.items()) if isinstance(checks, dict) else ""
        reasons = "; ".join(str(item) for item in summary.get("reasons", []))
        lines.append(f"| {artifact.label} | {summary.get('gate_status', summary.get('status', 'missing'))} | {check_text} | {reasons} |")


def append_diagnostic_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [item for item in artifacts if item.kind == "diagnostic"]
    lines.extend(["", "## Runtime Diagnostics", "", "| artifact | mode | available tools | raw artifacts |", "| --- | --- | --- | --- |"])
    if not items:
        lines.append("| none | unavailable | no diagnostic manifest provided |  |")
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        caps = summary.get("capabilities", {})
        available = ", ".join(name for name, value in caps.items() if isinstance(value, dict) and value.get("available"))
        lines.append(f"| {artifact.label} | {summary.get('mode', 'unknown')} | {available or 'none'} | {len(summary.get('raw_artifacts', []))} |")


def append_prefetch_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "prefetch"]
    lines.extend(
        [
            "",
            "## Prefetch Evidence",
            "",
            "| artifact | submitted | completed | failed | hit rate | waste rate | TTFT delta % | latency delta % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not items:
        lines.append("| none |  |  |  |  |  |  |  |")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {fmt(summary.get('prefetch_submitted'))} | "
            f"{fmt(summary.get('prefetch_completed'))} | {fmt(summary.get('prefetch_failed'))} | "
            f"{fmt(summary.get('prefetch_hit_rate'))} | {fmt(summary.get('prefetch_waste_rate'))} | "
            f"{fmt(summary.get('mean_ttft_delta_pct'))} | {fmt(summary.get('mean_latency_delta_pct'))} |"
        )


def append_cache_event_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "cache_events"]
    lines.extend(["", "## Cache Event Evidence", ""])
    if not items:
        lines.append("- No cache event artifacts were provided.")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"- `{artifact.label}`: status `{summary.get('status', 'missing')}`, rows `{summary.get('rows', '')}`, "
            f"events `{summary.get('event_counts', {})}`, statuses `{summary.get('status_counts', {})}`"
        )


def append_server_log_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "server_log"]
    lines.extend(
        [
            "",
            "## Server Startup Memory Evidence",
            "",
            "| log | model loading GiB | available KV GiB | GPU KV tokens | path |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    if not items:
        lines.append("| none |  |  |  |  |")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {fmt(summary.get('model_loading_memory_gib'))} | "
            f"{fmt(summary.get('available_kv_cache_memory_gib'))} | {fmt(summary.get('gpu_kv_cache_tokens'))} | "
            f"`{artifact.path}` |"
        )
    lines.extend(
        [
            "",
            "These startup-level vLLM values are the authoritative GPU KV capacity evidence on DGX Spark. Case-level GPU framebuffer memory is not exposed by `nvidia-smi`/NVML on this unified-memory platform.",
        ]
    )


def append_stress_boundary_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind == "stress"]
    lines.extend(
        [
            "",
            "## Stress Boundary",
            "",
            "| artifact | cases | success cases | failed cases | OOM cases | max success ctx | max success batch | RSS MB | disk read MB | disk write MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not items:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {fmt(summary.get('total_cases'))} | {fmt(summary.get('successful_cases'))} | "
            f"{fmt(summary.get('failed_cases'))} | {fmt(summary.get('oom_case_count'))} | "
            f"{fmt(summary.get('max_success_context'))} | {fmt(summary.get('max_success_batch'))} | "
            f"{fmt(summary.get('max_cpu_memory_peak_mb'))} | {fmt(summary.get('disk_read_delta_mb'))} | "
            f"{fmt(summary.get('disk_write_delta_mb'))} |"
        )


def append_policy_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    lines.extend(["", "## Policy And Stress Evidence", ""])
    for kind in ("comparison", "policy_ablation", "stress", "partial_load", "load_recompute"):
        items = [artifact for artifact in artifacts if artifact.kind == kind]
        if not items:
            lines.append(f"- `{kind}`: missing")
            continue
        for artifact in items:
            summary = summaries.get(artifact_key(artifact), {})
            lines.append(f"- `{kind}` `{artifact.label}`: status `{summary.get('status', 'missing')}`, rows `{summary.get('rows', '')}`")


def append_scheduler_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    items = [artifact for artifact in artifacts if artifact.kind in {"object_schedule", "chunk_scores"}]
    lines.extend(["", "## Scheduler Evidence", ""])
    if not items:
        lines.append("- No scheduler artifacts were provided.")
        return
    for artifact in items:
        summary = summaries.get(artifact_key(artifact), {})
        action_counts = summary.get("action_counts", {})
        lines.append(f"- `{artifact.label}` `{artifact.kind}`: status `{summary.get('status', 'missing')}`, actions `{action_counts}`")


def append_profile_workload_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    lines.extend(["", "## Profile And Workload Evidence", ""])
    for kind in ("trace", "profile", "workload"):
        items = [artifact for artifact in artifacts if artifact.kind == kind]
        if not items:
            lines.append(f"- `{kind}`: missing")
            continue
        for artifact in items:
            summary = summaries.get(artifact_key(artifact), {})
            lines.append(f"- `{kind}` `{artifact.label}`: `{summary}`")


def append_missing_section(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    missing = [artifact for artifact in artifacts if summaries.get(artifact_key(artifact), {}).get("status") == "missing"]
    lines.extend(["", "## Missing Evidence", ""])
    if not missing:
        lines.append("- No provided artifact paths were missing.")
        return
    for artifact in missing:
        lines.append(f"- `{artifact.label}` `{artifact.kind}`: `{artifact.path}`")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_success_rate(rows: list[dict[str, str]]) -> float | str:
    success = sum(as_float(row.get("success_count")) or 0.0 for row in rows)
    total = sum(as_float(row.get("request_count")) or 0.0 for row in rows)
    if total == 0:
        return ""
    return success / total


def mean_field(rows: list[dict[str, str]], field: str) -> float | str:
    values = [value for value in (as_float(row.get(field)) for row in rows) if value is not None]
    if not values:
        return ""
    return sum(values) / len(values)


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


def count_field(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, "")).strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def format_summary_facts(summary: dict[str, Any], keys: tuple[str, ...]) -> str:
    facts = [f"{key}={fmt(summary.get(key))}" for key in keys if summary.get(key) not in (None, "")]
    return ", ".join(facts[:6])


def collect_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def collect_git_info() -> dict[str, str]:
    commit = run_git(["git", "rev-parse", "HEAD"])
    status = run_git(["git", "status", "--porcelain"])
    return {"commit": commit or "unknown", "dirty": "true" if status else "false"}


def run_git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except Exception:
        return ""


def artifact_key(artifact: ArtifactInput) -> str:
    return f"{artifact.label}:{artifact.kind}:{artifact.path}"


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_if_whole(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
