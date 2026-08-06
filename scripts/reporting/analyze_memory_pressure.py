"""Analyze memory pressure artifacts and emit passive scheduler hints.

This script reads archived benchmark CSV files, per-case sample CSVs, and
unified trace JSONL files. It does not modify vLLM, LMCache, CUDA, or a running
serving runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.memory_pressure import (  # noqa: E402
    MemoryPressureConfig,
    MemoryPressureController,
    MemoryPressureDecision,
    observations_from_benchmark_csv,
    observations_from_sample_csv,
    observations_from_trace_jsonl,
    summarize_decisions,
    write_decisions_csv,
    write_pressure_hints_jsonl,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = collect_observations(args)
    controller = MemoryPressureController(config_from_args(args))
    decisions = controller.assess_many(observations)

    decisions_path = output_dir / args.decisions_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_decisions_csv(decisions_path, decisions)
    write_pressure_hints_jsonl(hints_path, decisions)
    write_report(report_path, args, decisions, decisions_path, hints_path)
    write_manifest(manifest_path, args, decisions, decisions_path, hints_path, report_path)

    print(f"Memory pressure decisions written to {decisions_path}")
    print(f"Memory pressure hints written to {hints_path}")
    print(f"Memory pressure report written to {report_path}")
    print(f"Memory pressure manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-results",
        action="append",
        default=[],
        help="benchmark_results.csv path or a run directory. Can be repeated.",
    )
    parser.add_argument(
        "--samples",
        action="append",
        default=[],
        help="A *_samples.csv path or directory containing sample CSVs. Can be repeated.",
    )
    parser.add_argument(
        "--trace-events",
        action="append",
        default=[],
        help="Unified trace_events.jsonl path. Can be repeated.",
    )
    parser.add_argument("--run-id", default="", help="Optional run label for inputs without embedded labels.")
    parser.add_argument("--output-dir", default="results/memory_pressure")
    parser.add_argument("--decisions-name", default="memory_pressure_decisions.csv")
    parser.add_argument("--hints-name", default="memory_pressure_hints.jsonl")
    parser.add_argument("--report-name", default="memory_pressure_report.md")
    parser.add_argument("--manifest-name", default="memory_pressure_manifest.json")
    parser.add_argument("--gpu-capacity-mb", type=float, default=0.0)
    parser.add_argument("--cpu-capacity-mb", type=float, default=0.0)
    parser.add_argument("--gpu-medium-ratio", type=float, default=0.70)
    parser.add_argument("--gpu-high-ratio", type=float, default=0.85)
    parser.add_argument("--gpu-critical-ratio", type=float, default=0.95)
    parser.add_argument("--cpu-medium-ratio", type=float, default=0.70)
    parser.add_argument("--cpu-high-ratio", type=float, default=0.85)
    parser.add_argument("--cpu-critical-ratio", type=float, default=0.95)
    parser.add_argument("--disk-medium-mb", type=float, default=512.0)
    parser.add_argument("--disk-high-mb", type=float, default=4096.0)
    parser.add_argument("--disk-critical-mb", type=float, default=16384.0)
    parser.add_argument("--error-medium-rate", type=float, default=0.01)
    parser.add_argument("--error-high-rate", type=float, default=0.05)
    parser.add_argument("--error-critical-rate", type=float, default=0.20)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MemoryPressureConfig:
    return MemoryPressureConfig(
        gpu_capacity_mb=args.gpu_capacity_mb,
        cpu_capacity_mb=args.cpu_capacity_mb,
        gpu_medium_ratio=args.gpu_medium_ratio,
        gpu_high_ratio=args.gpu_high_ratio,
        gpu_critical_ratio=args.gpu_critical_ratio,
        cpu_medium_ratio=args.cpu_medium_ratio,
        cpu_high_ratio=args.cpu_high_ratio,
        cpu_critical_ratio=args.cpu_critical_ratio,
        disk_medium_mb=args.disk_medium_mb,
        disk_high_mb=args.disk_high_mb,
        disk_critical_mb=args.disk_critical_mb,
        error_medium_rate=args.error_medium_rate,
        error_high_rate=args.error_high_rate,
        error_critical_rate=args.error_critical_rate,
    )


def collect_observations(args: argparse.Namespace) -> list[object]:
    observations: list[object] = []
    for path in args.benchmark_results:
        observations.extend(observations_from_benchmark_csv(path, run_id=args.run_id))
    for path in args.samples:
        observations.extend(observations_from_sample_csv(path, run_id=args.run_id))
    for path in args.trace_events:
        observations.extend(observations_from_trace_jsonl(path, run_id=args.run_id))
    return observations


def write_report(
    path: Path,
    args: argparse.Namespace,
    decisions: list[MemoryPressureDecision],
    decisions_path: Path,
    hints_path: Path,
) -> None:
    summary = summarize_decisions(decisions)
    lines = [
        "# Memory Pressure Controller Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Benchmark results: `{', '.join(args.benchmark_results) if args.benchmark_results else 'none'}`",
        f"- Sample inputs: `{', '.join(args.samples) if args.samples else 'none'}`",
        f"- Trace events: `{', '.join(args.trace_events) if args.trace_events else 'none'}`",
        f"- Run id override: `{args.run_id or 'none'}`",
        f"- GPU capacity MB: `{args.gpu_capacity_mb or 'not configured'}`",
        f"- CPU capacity MB: `{args.cpu_capacity_mb or 'not configured'}`",
        "",
        "## Outputs",
        "",
        f"- Decisions CSV: `{decisions_path}`",
        f"- Passive hints JSONL: `{hints_path}`",
        "",
        "## Summary",
        "",
        f"- Decisions: `{summary['decision_count']}`",
        f"- Max memory pressure: `{summary['max_memory_pressure']:.6f}`",
        "",
        "### Level Counts",
        "",
        "| level | count |",
        "| --- | ---: |",
    ]
    append_count_table(lines, summary["level_counts"])
    lines.extend(["", "### Action Counts", "", "| action | count |", "| --- | ---: |"])
    append_count_table(lines, summary["action_counts"])
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| run | case | level | pressure | primary action | RSS MB | disk read MB | disk write MB | error rate | reason |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for decision in decisions[:100]:
        row = decision.to_record()
        lines.append(
            "| {run_id} | {case} | {level} | {memory_pressure} | {primary_action} | "
            "{cpu_memory_peak_mb} | {disk_read_delta_mb} | "
            "{disk_write_delta_mb} | {error_rate} | {reason} |".format(**row)
        )
    if len(decisions) > 100:
        lines.append(f"| ... |  |  |  |  |  |  |  |  | {len(decisions) - 100} more decision(s) omitted |")
    lines.extend(
        [
            "",
            "## How To Consume The Pressure Score",
            "",
            "- Use `memory_pressure` as the value for `--memory-pressure` in `score_chunks.py`, `decide_load_vs_recompute.py`, or `run_unified_object_scheduler.py`.",
            "- `reduce_prefetch_budget`, `offload_more`, `drop_low_reuse`, and `reduce_batch_or_context` are passive hints for adapters/operators.",
            "- RSS and disk IO are the primary portable pressure signals on DGX Spark; per-GPU memory is used only when the platform exposes it.",
            "- This report does not prove live runtime control until a real vLLM/LMCache adapter consumes the hints on GPU.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    decisions: list[MemoryPressureDecision],
    decisions_path: Path,
    hints_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-memory-pressure-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "benchmark_results": args.benchmark_results,
            "samples": args.samples,
            "trace_events": args.trace_events,
            "run_id": args.run_id,
        },
        "outputs": {
            "decisions_csv": str(decisions_path),
            "hints_jsonl": str(hints_path),
            "report": str(report_path),
        },
        "config": {
            "gpu_capacity_mb": args.gpu_capacity_mb,
            "cpu_capacity_mb": args.cpu_capacity_mb,
            "gpu_medium_ratio": args.gpu_medium_ratio,
            "gpu_high_ratio": args.gpu_high_ratio,
            "gpu_critical_ratio": args.gpu_critical_ratio,
            "cpu_medium_ratio": args.cpu_medium_ratio,
            "cpu_high_ratio": args.cpu_high_ratio,
            "cpu_critical_ratio": args.cpu_critical_ratio,
            "disk_medium_mb": args.disk_medium_mb,
            "disk_high_mb": args.disk_high_mb,
            "disk_critical_mb": args.disk_critical_mb,
            "error_medium_rate": args.error_medium_rate,
            "error_high_rate": args.error_high_rate,
            "error_critical_rate": args.error_critical_rate,
        },
        "summary": summarize_decisions(decisions),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def append_count_table(lines: list[str], counts: dict[str, int]) -> None:
    if not counts:
        lines.append("| none | 0 |")
        return
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")


if __name__ == "__main__":
    raise SystemExit(main())
