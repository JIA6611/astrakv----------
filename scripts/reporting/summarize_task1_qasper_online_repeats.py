"""Aggregate validated cold-start QASPER online-control repetitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scripts.reporting.summarize_task1_qasper_online_suite import WORKLOADS, summarize_suite


def aggregate_repeat_summaries(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    if len(repeats) < 2:
        raise ValueError("at least two independent cold-start repeats are required")
    workloads: dict[str, Any] = {}
    for workload in WORKLOADS:
        quality: dict[str, Any] = {}
        accepted = True
        for metric in ("exact_match", "token_f1"):
            baseline = [float(item["workloads"][workload]["baseline"]["quality"][metric]) for item in repeats]
            variant = [float(item["workloads"][workload]["variant"]["quality"][metric]) for item in repeats]
            delta = [right - left for left, right in zip(baseline, variant, strict=True)]
            accepted = accepted and all(value >= 0.0 for value in delta)
            quality[metric] = {"baseline": _stats(baseline), "variant": _stats(variant), "variant_minus_baseline": _stats(delta)}
        drops = [float(item["workloads"][workload]["variant"]["completed_drop_count"]) for item in repeats]
        workloads[workload] = {"quality_noninferior_zero_margin": accepted, "quality": quality, "completed_drop_count": _stats(drops)}
    return {"schema": "astrakv-task1-qasper-online-repeats-v1", "repeat_count": len(repeats), "workloads": workloads}


def _stats(values: list[float]) -> dict[str, Any]:
    return {"mean": mean(values), "sample_stddev": stdev(values) if len(values) > 1 else 0.0, "values": values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-suite", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = aggregate_repeat_summaries([summarize_suite(path) for path in args.repeat_suite])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "repeat_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# QASPER Online-Control Cold-Start Repeats", "", f"Repeats: `{summary['repeat_count']}`", ""]
    for workload, record in summary["workloads"].items():
        lines.extend([f"## {workload}", "", f"Quality non-inferior at zero margin: `{record['quality_noninferior_zero_margin']}`", "", "| metric | baseline mean | variant mean | delta mean | delta stddev |", "| --- | ---: | ---: | ---: | ---: |"])
        for metric, values in record["quality"].items():
            lines.append(f"| {metric} | {values['baseline']['mean']:.6f} | {values['variant']['mean']:.6f} | {values['variant_minus_baseline']['mean']:.6f} | {values['variant_minus_baseline']['sample_stddev']:.6f} |")
        lines.append("")
    (output / "repeat_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
