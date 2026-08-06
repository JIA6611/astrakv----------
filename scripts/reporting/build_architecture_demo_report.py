"""Build a fast architecture demo report from AstraKV-W evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reporting.build_competition_report import (  # noqa: E402
    ArtifactInput,
    artifact_key,
    as_float,
    collect_environment,
    collect_git_info,
    fmt,
    read_csv_rows,
    summarize_artifacts,
    write_inventory,
)


@dataclass(frozen=True, slots=True)
class DemoPaths:
    main_evidence: Path
    boundary_pass: Path
    boundary_fail: Path
    summary: Path
    command_log: Path | None
    include_vm_smoke: Path | None
    include_live_smoke: Path | None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = DemoPaths(
        main_evidence=Path(args.main_evidence),
        boundary_pass=Path(args.boundary_pass),
        boundary_fail=Path(args.boundary_fail),
        summary=Path(args.summary),
        command_log=Path(args.command_log) if args.command_log else None,
        include_vm_smoke=Path(args.include_vm_smoke) if args.include_vm_smoke else None,
        include_live_smoke=Path(args.include_live_smoke) if args.include_live_smoke else None,
    )
    artifacts = collect_demo_artifacts(paths)
    summaries = summarize_artifacts(artifacts)
    demo = build_demo_summary(paths, artifacts, summaries, args.command)

    selected_dir = output_dir / "selected_artifacts"
    copy_selected_artifacts(selected_dir, artifacts, paths)

    inventory_path = output_dir / "artifact_inventory.csv"
    summary_path = output_dir / "demo_summary.json"
    report_path = output_dir / "demo_report.md"
    write_inventory(inventory_path, artifacts, summaries)
    summary_path.write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, demo, artifacts, summaries, inventory_path, summary_path, selected_dir)

    print(f"Architecture demo report written to {report_path}")
    print(f"Architecture demo summary written to {summary_path}")
    print(f"Artifact inventory written to {inventory_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-evidence", default="results/extended_evidence_20260625_014917")
    parser.add_argument("--boundary-pass", default="results/extended_g016_ctx32k_b16_out256")
    parser.add_argument("--boundary-fail", default="results/extended_g015_ctx32k_b16_out256")
    parser.add_argument("--summary", default="results/project_implementation_evidence_summary.md")
    parser.add_argument("--command-log", default="")
    parser.add_argument("--include-vm-smoke", default="")
    parser.add_argument("--include-live-smoke", default="")
    parser.add_argument("--command", action="append", default=[])
    return parser.parse_args()


def collect_demo_artifacts(paths: DemoPaths) -> list[ArtifactInput]:
    main = paths.main_evidence
    e2e = main / "01_e2e"
    boundary = paths.boundary_pass / "02_boundary_32k"
    fail_boundary = paths.boundary_fail / "02_boundary_32k"
    artifacts = [
        ArtifactInput("main_final_report", "report", main / "07_final_report" / "competition_report.md"),
        ArtifactInput("project_summary", "report", paths.summary),
        ArtifactInput("vllm_baseline", "benchmark", first_match(e2e / "step3_vllm_benchmark", "benchmark_results.csv")),
        ArtifactInput("lmcache_cpu_baseline", "benchmark", first_match(e2e / "step4_lmcache_cpu_benchmark", "benchmark_results.csv")),
        ArtifactInput("lmcache_disk_baseline", "benchmark", first_match(e2e / "step4_lmcache_disk_benchmark", "benchmark_results.csv")),
        ArtifactInput("prefetch_results", "prefetch", e2e / "step5_prefetch" / "prefetch_results.csv"),
        ArtifactInput("prefetch_benchmark", "benchmark", e2e / "step5_prefetch" / "prefetch_benchmark_results.csv"),
        ArtifactInput("policy_ablation", "policy_ablation", e2e / "step7_policy_ablation" / "policy_ablation_results.csv"),
        ArtifactInput("chunk_scores", "chunk_scores", e2e / "step7_chunk_scores" / "chunk_scores.csv"),
        ArtifactInput("e2e_disk_cache_events", "cache_events", e2e / "cache_events" / "step4_lmcache_disk" / "cache_events.jsonl"),
        ArtifactInput("prefetch_cache_events", "cache_events", e2e / "cache_events" / "step5_prefetch" / "cache_events.jsonl"),
        ArtifactInput("dgx_vm_evidence", "vm_evidence", main / "04_os_vm" / "dgx_spark_vm" / "dgx_spark_vm_evidence_summary.json"),
        ArtifactInput("mmap_kv_evidence", "vm_evidence", main / "04_os_vm" / "mmap_kv_cache" / "mmap_kv_demo_summary.json"),
        ArtifactInput("quality", "quality", main / "05_quality" / "lmcache_disk_vs_vllm" / "quality_results.csv"),
        ArtifactInput("boundary_stress_pass", "stress", boundary / "stress_analysis" / "stress_summary.csv"),
        ArtifactInput("boundary_vllm_pass", "benchmark", first_match(boundary / "vllm", "benchmark_results.csv")),
        ArtifactInput("boundary_lmcache_disk_pass", "benchmark", first_match(boundary / "lmcache_disk", "benchmark_results.csv")),
        ArtifactInput("boundary_disk_cache_events", "cache_events", paths.boundary_pass / "03_cache_events" / "lmcache_disk_boundary" / "cache_events.jsonl"),
        ArtifactInput("boundary_vllm_pass_log", "server_log", boundary / "vllm_server.log"),
        ArtifactInput("boundary_lmcache_disk_pass_log", "server_log", boundary / "lmcache_disk_server.log"),
        ArtifactInput("boundary_vllm_fail_log", "server_log", fail_boundary / "vllm_server.log"),
    ]
    if paths.include_vm_smoke:
        artifacts.append(ArtifactInput("demo_vm_smoke", "vm_evidence", paths.include_vm_smoke / "mmap_kv_demo_summary.json"))
    if paths.include_live_smoke:
        artifacts.append(ArtifactInput("demo_live_smoke_report", "report", paths.include_live_smoke / "competition_report" / "competition_report.md"))
    return artifacts


def first_match(root: Path, name: str) -> Path:
    if root.is_file():
        return root
    if not root.exists():
        return root / name
    matches = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    return matches[0] if matches else root / name


def build_demo_summary(
    paths: DemoPaths,
    artifacts: list[ArtifactInput],
    summaries: dict[str, Any],
    commands: list[str],
) -> dict[str, Any]:
    prefetch_rows = safe_csv(paths.main_evidence / "01_e2e" / "step5_prefetch" / "prefetch_benchmark_results.csv")
    policy_rows = safe_csv(paths.main_evidence / "01_e2e" / "step7_policy_ablation" / "policy_ablation_results.csv")
    stress_rows = safe_csv(paths.boundary_pass / "02_boundary_32k" / "stress_analysis" / "stress_summary.csv")
    fail_log_path = paths.boundary_fail / "02_boundary_32k" / "vllm_server.log"
    fail_text = fail_log_path.read_text(encoding="utf-8", errors="replace") if fail_log_path.exists() else ""

    return {
        "schema": "astra-architecture-demo-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": collect_environment(),
        "git": collect_git_info(),
        "commands": commands,
        "evidence_roots": {
            "main_evidence": str(paths.main_evidence),
            "boundary_pass": str(paths.boundary_pass),
            "boundary_fail": str(paths.boundary_fail),
        },
        "artifact_status": artifact_status_rows(artifacts, summaries),
        "prefetch_comparison": summarize_prefetch_benchmark(prefetch_rows),
        "policy_ablation": summarize_policy_ablation(policy_rows),
        "boundary_pass": summarize_boundary_pass(stress_rows),
        "boundary_fail": summarize_boundary_failure(fail_text),
        "cache_events": summarize_cache_evidence(artifacts, summaries),
        "vm_evidence": summarize_named_artifacts(artifacts, summaries, "vm_evidence"),
        "quality": summarize_named_artifacts(artifacts, summaries, "quality"),
        "claim_boundaries": [
            "DGX Spark uses UMA; case-level GPU framebuffer memory is unavailable in these artifacts.",
            "Endpoint-level selective prefetch is demonstrated; vLLM internal KV scheduler replacement is not claimed.",
            "TTFT is the strongest performance claim; end-to-end latency is workload-dependent.",
        ],
    }


def artifact_status_rows(artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for artifact in artifacts:
        summary = summaries.get(artifact_key(artifact), {})
        rows.append(
            {
                "label": artifact.label,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "status": summary.get("status", "missing"),
                "rows": summary.get("rows", ""),
            }
        )
    return rows


def safe_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def summarize_prefetch_benchmark(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_context: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        context = str(row.get("context_length") or context_from_case(row.get("case", "")))
        backend = row.get("backend", "")
        if backend:
            by_context.setdefault(context, {})[backend] = row
    comparisons = []
    ttft_changes = []
    latency_changes = []
    for context in sorted(by_context, key=lambda item: int(item) if item.isdigit() else item):
        no_prefetch = by_context[context].get("astrakv_no_prefetch")
        prefetch = by_context[context].get("astrakv_prefetch_demand")
        if not no_prefetch or not prefetch:
            continue
        no_ttft = as_float(no_prefetch.get("ttft_ms"))
        yes_ttft = as_float(prefetch.get("ttft_ms"))
        no_latency = as_float(no_prefetch.get("latency_ms"))
        yes_latency = as_float(prefetch.get("latency_ms"))
        ttft_change = pct_change(no_ttft, yes_ttft)
        latency_change = pct_change(no_latency, yes_latency)
        if ttft_change is not None:
            ttft_changes.append(ttft_change)
        if latency_change is not None:
            latency_changes.append(latency_change)
        comparisons.append(
            {
                "context_length": context,
                "no_prefetch_ttft_ms": no_ttft,
                "prefetch_ttft_ms": yes_ttft,
                "ttft_change_pct": ttft_change,
                "no_prefetch_latency_ms": no_latency,
                "prefetch_latency_ms": yes_latency,
                "latency_change_pct": latency_change,
            }
        )
    return {
        "status": "ok" if comparisons else "missing",
        "comparisons": comparisons,
        "mean_ttft_change_pct": mean(ttft_changes),
        "mean_latency_change_pct": mean(latency_changes),
    }


def summarize_policy_ablation(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_policy = {row.get("policy", ""): row for row in rows}
    base = by_policy.get("no_prefetch")
    combined = by_policy.get("astrakv_combined")
    return {
        "status": "ok" if base and combined else "missing",
        "no_prefetch": pick_policy_fields(base),
        "astrakv_combined": pick_policy_fields(combined),
        "ttft_delta_pct_vs_baseline": as_float(combined.get("ttft_delta_pct_vs_baseline")) if combined else None,
        "latency_delta_pct_vs_baseline": as_float(combined.get("latency_delta_pct_vs_baseline")) if combined else None,
        "prefetch_hit_rate": as_float(combined.get("prefetch_hit_rate")) if combined else None,
        "chunks_scored": as_float(combined.get("chunks_scored")) if combined else None,
    }


def pick_policy_fields(row: dict[str, str] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "benchmark_cases": as_float(row.get("benchmark_cases")),
        "prefetch_cases": as_float(row.get("prefetch_cases")),
        "chunks_scored": as_float(row.get("chunks_scored")),
        "success_rate": as_float(row.get("success_rate")),
        "ttft_ms_mean": as_float(row.get("ttft_ms_mean")),
        "latency_ms_mean": as_float(row.get("latency_ms_mean")),
        "missing_metric_groups": row.get("missing_metric_groups", ""),
    }


def summarize_boundary_pass(rows: list[dict[str, str]]) -> dict[str, Any]:
    runs = []
    for row in rows:
        runs.append(
            {
                "run": row.get("run", ""),
                "success_rate": as_float(row.get("success_rate")),
                "total_requests": as_float(row.get("total_requests")),
                "success_requests": as_float(row.get("success_requests")),
                "max_success_context": as_float(row.get("max_success_context")),
                "max_success_batch": as_float(row.get("max_success_batch")),
                "process_rss_peak_mb": as_float(row.get("process_rss_peak_mb")),
                "disk_write_delta_mb": as_float(row.get("disk_write_delta_mb")),
                "worst_latency_p95_ms": as_float(row.get("worst_latency_p95_ms")),
            }
        )
    disk = next((row for row in runs if row.get("run") == "lmcache_disk"), {})
    return {
        "status": "ok" if runs else "missing",
        "runs": runs,
        "lmcache_disk_write_mb": disk.get("disk_write_delta_mb"),
        "max_context": max([row["max_success_context"] for row in runs if row.get("max_success_context") is not None], default=None),
        "max_batch": max([row["max_success_batch"] for row in runs if row.get("max_success_batch") is not None], default=None),
    }


def summarize_boundary_failure(text: str) -> dict[str, Any]:
    needed = extract_float(text, r"([0-9.]+)\s+GiB KV cache is needed")
    available = extract_float(text, r"available KV cache memory \(([0-9.]+)\s+GiB\)")
    estimated_len = extract_float(text, r"estimated maximum model length is\s+([0-9]+)")
    available_startup = extract_float(text, r"Available KV cache memory:\s*([0-9.]+)\s+GiB")
    return {
        "status": "ok" if needed or available or available_startup else "missing",
        "needed_kv_cache_gib": needed,
        "available_kv_cache_gib": available or available_startup,
        "estimated_max_model_len": estimated_len,
        "failure_summary": first_value_error_line(text),
    }


def summarize_cache_evidence(artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for artifact in artifacts:
        if artifact.kind != "cache_events":
            continue
        summary = summaries.get(artifact_key(artifact), {})
        result[artifact.label] = {
            "status": summary.get("status", "missing"),
            "rows": summary.get("rows", 0),
            "event_counts": summary.get("event_counts", {}),
        }
    return result


def summarize_named_artifacts(artifacts: list[ArtifactInput], summaries: dict[str, Any], kind: str) -> dict[str, Any]:
    result = {}
    for artifact in artifacts:
        if artifact.kind != kind:
            continue
        summary = dict(summaries.get(artifact_key(artifact), {}))
        result[artifact.label] = summary
    return result


def write_report(
    path: Path,
    demo: dict[str, Any],
    artifacts: list[ArtifactInput],
    summaries: dict[str, Any],
    inventory_path: Path,
    summary_path: Path,
    selected_dir: Path,
) -> None:
    lines = [
        "# AstraKV-W Architecture Demo Report",
        "",
        f"Generated: {demo['generated_at']}",
        "",
        "## Demo 入口与运行命令",
        "",
        f"- Python: `{demo['environment'].get('python', '')}`",
        f"- Platform: `{demo['environment'].get('platform', '')}`",
        f"- Git commit: `{demo['git'].get('commit', 'unknown')}`",
        f"- Git dirty: `{demo['git'].get('dirty', 'unknown')}`",
        "",
    ]
    for command in demo.get("commands") or ["bash scripts/entrypoints/run_architecture_demo.sh --skip-install"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "## 架构链路展示",
            "",
            "本 demo 展示的链路是：真实后端 vLLM/LMCache -> benchmark/request results -> cache events -> trace/ProfileDB -> policy scoring/ablation -> final report/archive。",
            "",
            "默认模式复用已归档重实验结果，并补跑轻量脚本检查和可选 VM smoke；这样可以在答辩现场快速说明架构如何工作，而不把时间消耗在模型启动和 32K 压力测试上。",
            "",
            "## 赛题要求对照",
            "",
            "- 访存行为分析：使用 benchmark CSV、server startup KV capacity、cache event JSONL 和 request/sample artifacts 解释 KV cache 行为。",
            "- 虚拟内存/分层缓存：使用 LMCache CPU/Disk、mmap/DGX VM evidence、disk write delta 和 32K boundary 展示分层存储路径。",
            "- 预取隐藏 I/O：使用 prefetch results、prefetch benchmark rows、cache events 和 policy ablation 展示 TTFT 改善与预取有效性。",
            "",
        ]
    )
    append_key_results(lines, demo)
    append_artifact_inventory(lines, artifacts, summaries)
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- DGX Spark 是 UMA 平台，本 demo 不声明 case-level GPU framebuffer memory 下降。",
            "- 当前 selective prefetch 是 endpoint-level warmup/prefetch，不声明已经替换 vLLM 内部 KV scheduler。",
            "- 当前最强性能结论聚焦 TTFT、KV capacity boundary、disk offload evidence 和 cache events；端到端 latency 不做全场景领先声明。",
            "",
            "## 如何复现完整结果",
            "",
            "```bash",
            "bash scripts/entrypoints/run_competition_extended_evidence.sh --skip-install --continue-on-failure",
            "```",
            "",
            "32K 可运行边界：",
            "",
            "```bash",
            "bash scripts/entrypoints/run_competition_extended_evidence.sh \\",
            "  --only boundary \\",
            "  --gpu-util-boundary 0.16 \\",
            "  --boundary-max-model-len 32768 \\",
            "  --boundary-context-lengths \"24576 32768\" \\",
            "  --boundary-batch-sizes \"4 8 12 16\" \\",
            "  --boundary-output-tokens 256 \\",
            "  --boundary-repeat 1 \\",
            "  --boundary-timeout 2400 \\",
            "  --output-root results/extended_g016_ctx32k_b16_out256 \\",
            "  --skip-install \\",
            "  --continue-on-failure",
            "```",
            "",
            "32K 失败下界：",
            "",
            "```bash",
            "bash scripts/entrypoints/run_competition_extended_evidence.sh \\",
            "  --only boundary \\",
            "  --gpu-util-boundary 0.15 \\",
            "  --boundary-max-model-len 32768 \\",
            "  --boundary-context-lengths \"24576 32768\" \\",
            "  --boundary-batch-sizes \"4 8 12 16\" \\",
            "  --boundary-output-tokens 256 \\",
            "  --boundary-repeat 1 \\",
            "  --boundary-timeout 2400 \\",
            "  --output-root results/extended_g015_ctx32k_b16_out256 \\",
            "  --skip-install \\",
            "  --continue-on-failure",
            "```",
            "",
            "## Demo 产物",
            "",
            f"- `{inventory_path}`",
            f"- `{summary_path}`",
            f"- `{selected_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def append_key_results(lines: list[str], demo: dict[str, Any]) -> None:
    prefetch = demo["prefetch_comparison"]
    policy = demo["policy_ablation"]
    boundary = demo["boundary_pass"]
    failure = demo["boundary_fail"]
    cache = demo["cache_events"]
    lines.extend(["## 关键结果摘要", ""])
    lines.extend(
        [
            "### Prefetch Demand vs No Prefetch",
            "",
            "| context | no-prefetch TTFT ms | prefetch TTFT ms | TTFT change | no-prefetch latency ms | prefetch latency ms | latency change |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if prefetch.get("comparisons"):
        for row in prefetch["comparisons"]:
            lines.append(
                f"| {row['context_length']} | {fmt(row['no_prefetch_ttft_ms'])} | {fmt(row['prefetch_ttft_ms'])} | "
                f"{fmt_pct(row['ttft_change_pct'])} | {fmt(row['no_prefetch_latency_ms'])} | "
                f"{fmt(row['prefetch_latency_ms'])} | {fmt_pct(row['latency_change_pct'])} |"
            )
    else:
        lines.append("| missing |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            f"平均 TTFT change: `{fmt_pct(prefetch.get('mean_ttft_change_pct'))}`；平均 latency change: `{fmt_pct(prefetch.get('mean_latency_change_pct'))}`。",
            "",
            "### Policy Ablation",
            "",
            "| policy | benchmark cases | prefetch cases | chunks scored | success rate | TTFT mean ms | latency mean ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_name in ("no_prefetch", "astrakv_combined"):
        row = policy.get(policy_name, {})
        lines.append(
            f"| {policy_name} | {fmt(row.get('benchmark_cases'))} | {fmt(row.get('prefetch_cases'))} | "
            f"{fmt(row.get('chunks_scored'))} | {fmt(row.get('success_rate'))} | "
            f"{fmt(row.get('ttft_ms_mean'))} | {fmt(row.get('latency_ms_mean'))} |"
        )
    lines.extend(
        [
            "",
            f"`astrakv_combined` TTFT vs baseline: `{fmt_pct(policy.get('ttft_delta_pct_vs_baseline'))}`；latency vs baseline: `{fmt_pct(policy.get('latency_delta_pct_vs_baseline'))}`。",
            "",
            "### 32K Boundary",
            "",
            "| run | success rate | requests | max context | max batch | RSS MB | disk write MB | worst p95 latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if boundary.get("runs"):
        for row in boundary["runs"]:
            requests = ""
            if row.get("success_requests") is not None and row.get("total_requests") is not None:
                requests = f"{fmt(row.get('success_requests'))}/{fmt(row.get('total_requests'))}"
            lines.append(
                f"| {row.get('run', '')} | {fmt(row.get('success_rate'))} | {requests} | "
                f"{fmt(row.get('max_success_context'))} | {fmt(row.get('max_success_batch'))} | "
                f"{fmt(row.get('process_rss_peak_mb'))} | {fmt(row.get('disk_write_delta_mb'))} | "
                f"{fmt(row.get('worst_latency_p95_ms'))} |"
            )
    else:
        lines.append("| missing |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            f"LMCache Disk 写盘: `{fmt(boundary.get('lmcache_disk_write_mb'))} MB`，用于证明 disk offload 路径真实触发。",
            "",
            "### 32K Failure Lower Bound",
            "",
            f"- status: `{failure.get('status')}`",
            f"- needed KV cache: `{fmt(failure.get('needed_kv_cache_gib'))} GiB`",
            f"- available KV cache: `{fmt(failure.get('available_kv_cache_gib'))} GiB`",
            f"- estimated max model length: `{fmt(failure.get('estimated_max_model_len'))}`",
            f"- failure summary: `{failure.get('failure_summary') or ''}`",
            "",
            "### Cache Events",
            "",
        ]
    )
    if cache:
        for label, row in cache.items():
            lines.append(f"- `{label}`: rows `{row.get('rows')}`, events `{row.get('event_counts')}`")
    else:
        lines.append("- missing")


def append_artifact_inventory(lines: list[str], artifacts: list[ArtifactInput], summaries: dict[str, Any]) -> None:
    lines.extend(["", "## Artifact Inventory", "", "| label | kind | status | rows | path |", "| --- | --- | --- | ---: | --- |"])
    for artifact in artifacts:
        summary = summaries.get(artifact_key(artifact), {})
        lines.append(
            f"| {artifact.label} | {artifact.kind} | {summary.get('status', 'missing')} | "
            f"{summary.get('rows', '')} | `{artifact.path}` |"
        )


def copy_selected_artifacts(selected_dir: Path, artifacts: list[ArtifactInput], paths: DemoPaths) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        if not artifact.exists or artifact.path.is_dir():
            continue
        safe_name = f"{artifact.label}{artifact.path.suffix or '.txt'}"
        shutil.copy2(artifact.path, selected_dir / safe_name)
    if paths.command_log and paths.command_log.exists():
        shutil.copy2(paths.command_log, selected_dir / "commands.log")


def context_from_case(case: str) -> str:
    match = re.search(r"ctx([0-9]+)", case)
    return match.group(1) if match else ""


def pct_change(baseline: float | None, variant: float | None) -> float | None:
    if baseline in (None, 0) or variant is None:
        return None
    return (variant - baseline) / baseline * 100.0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return as_float(match.group(1))


def first_value_error_line(text: str) -> str:
    for line in text.splitlines():
        if "ValueError:" in line:
            return line.strip()
    return ""


def fmt_pct(value: Any) -> str:
    if value in (None, ""):
        return ""
    number = as_float(value)
    if number is None:
        return str(value)
    return f"{number:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
