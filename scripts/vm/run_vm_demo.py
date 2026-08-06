"""Run the file-backed mmap virtual-memory demonstration.

The demo is independent from vLLM/LMCache. It produces page-like demand-load,
prefetch, latency, and RSS evidence for competition reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.vm_backend import (  # noqa: E402
    VMDemoConfig,
    VMDemoResult,
    VirtualMemoryDemoRunner,
    write_access_trace_csv,
    write_summary_json,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backing_file = Path(args.backing_file) if args.backing_file else output_dir / "vm_backing.bin"
    config = config_from_args(args)
    result = VirtualMemoryDemoRunner(config).run(backing_file)

    trace_path = output_dir / args.trace_name
    summary_path = output_dir / args.summary_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_access_trace_csv(trace_path, result.access_records)
    write_summary_json(summary_path, result.summary)
    write_report(report_path, result, trace_path, summary_path)
    write_manifest(manifest_path, args, result, trace_path, summary_path, report_path)

    print(f"VM access trace written to {trace_path}")
    print(f"VM summary written to {summary_path}")
    print(f"VM demo report written to {report_path}")
    print(f"VM demo manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/vm_demo")
    parser.add_argument("--backing-file", default="", help="Optional backing file path.")
    parser.add_argument("--trace-name", default="vm_access_trace.csv")
    parser.add_argument("--summary-name", default="vm_summary.json")
    parser.add_argument("--report-name", default="vm_demo_report.md")
    parser.add_argument("--manifest-name", default="vm_demo_manifest.json")
    parser.add_argument("--file-size-mb", type=int, default=16)
    parser.add_argument("--page-size-bytes", type=int, default=4096)
    parser.add_argument("--access-count", type=int, default=256)
    parser.add_argument("--pattern", choices=["sequential", "reverse", "stride", "random"], default="sequential")
    parser.add_argument("--prefetch-window", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3136859)
    parser.add_argument("--keep-backing-file", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> VMDemoConfig:
    return VMDemoConfig(
        file_size_mb=args.file_size_mb,
        page_size_bytes=args.page_size_bytes,
        access_count=args.access_count,
        pattern=args.pattern,
        prefetch_window=args.prefetch_window,
        seed=args.seed,
        keep_backing_file=args.keep_backing_file,
    )


def write_report(path: Path, result: VMDemoResult, trace_path: Path, summary_path: Path) -> None:
    summary = result.summary
    record = summary.to_record()
    lines = [
        "# OS Virtual Memory Demonstration Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- File size bytes: `{summary.file_size_bytes}`",
        f"- Page size bytes: `{summary.page_size_bytes}`",
        f"- Total pages: `{summary.total_pages}`",
        f"- Access count: `{summary.access_count}`",
        f"- Pattern: `{summary.metadata.get('pattern', '')}`",
        f"- Prefetch window: `{summary.prefetch_window}`",
        "",
        "## Summary",
        "",
        f"- Unique pages accessed: `{summary.unique_pages_accessed}`",
        f"- Prefetched pages accessed: `{summary.prefetched_page_count}`",
        f"- Prefetch coverage rate: `{summary.prefetch_coverage_rate:.6f}`",
        f"- Demand-fault-like count: `{summary.demand_fault_like_count}`",
        f"- Demand-fault-like rate: `{summary.demand_fault_like_rate:.6f}`",
        f"- First-touch count: `{summary.first_touch_count}`",
        f"- Average access latency us: `{summary.avg_latency_us:.6f}`",
        f"- P95 access latency us: `{summary.p95_latency_us:.6f}`",
        f"- Max access latency us: `{summary.max_latency_us:.6f}`",
        f"- RSS start MB: `{summary.rss_start_mb:.6f}`",
        f"- RSS peak MB: `{summary.rss_peak_mb:.6f}`",
        f"- RSS end MB: `{summary.rss_end_mb:.6f}`",
        f"- Backing file retained: `{summary.retained_backing_file}`",
        "",
        "## Metrics Table",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in record.items():
        if key == "metadata":
            continue
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The backing file represents lower-tier storage such as SSD.",
            "- First access to an untouched page is counted as demand-fault-like evidence.",
            "- Software prefetch touches future pages before demand access, reducing demand-fault-like accesses when the pattern is predictable.",
            "- This is an OS-style demonstration artifact, not a replacement for real vLLM/LMCache GPU benchmarks.",
            "",
            "## Artifacts",
            "",
            f"- Access trace: `{trace_path}`",
            f"- Summary JSON: `{summary_path}`",
            "- Report: `vm_demo_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    result: VMDemoResult,
    trace_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-vm-demo-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "file_size_mb": args.file_size_mb,
            "page_size_bytes": args.page_size_bytes,
            "access_count": args.access_count,
            "pattern": args.pattern,
            "prefetch_window": args.prefetch_window,
            "seed": args.seed,
            "keep_backing_file": args.keep_backing_file,
        },
        "outputs": {
            "access_trace_csv": str(trace_path),
            "summary_json": str(summary_path),
            "report": str(report_path),
        },
        "summary": result.summary.to_record(),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
