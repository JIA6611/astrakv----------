"""Generate a small controlled workload for AstraKV runtime action validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = "astra-runtime-workload-v1"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        context_length=args.context_length,
        expected_output_tokens=args.expected_output_tokens,
    )
    suite_path = output_dir / args.suite_name
    manifest_path = output_dir / args.manifest_name
    report_path = output_dir / args.report_name
    write_jsonl(suite_path, rows)
    write_manifest(manifest_path, suite_path, rows)
    write_report(report_path, suite_path, manifest_path, rows)
    print(f"Runtime action workload written to {suite_path}")
    print(f"Manifest written to {manifest_path}")
    print(f"Report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="astrakv/benchmarks/prompts")
    parser.add_argument("--suite-name", default="runtime_action_validation_workload.jsonl")
    parser.add_argument("--manifest-name", default="runtime_action_validation_manifest.json")
    parser.add_argument("--report-name", default="runtime_action_validation_report.md")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--expected-output-tokens", type=int, default=64)
    return parser.parse_args()


def build_rows(*, context_length: int, expected_output_tokens: int) -> list[dict[str, Any]]:
    shared = [
        {
            "schema": SCHEMA,
            "context_length": context_length,
            "expected_output_tokens": expected_output_tokens,
            "batch_size": 1,
        },
    ]
    rows = [
        row(
            request_id="hot-load-seed",
            prefix_id="prefix-hot-load-a",
            arrival_index=0,
            reuse_ratio=1.0,
            reuse_bucket="high",
            scenario="hot_load",
            prompt=prompt("hot_load", "seed", "Establish the reusable long-form context for a later revisit."),
        ),
        row(
            request_id="hot-load-revisit",
            prefix_id="prefix-hot-load-a",
            arrival_index=1,
            reuse_ratio=1.0,
            reuse_bucket="high",
            scenario="hot_load",
            prompt=prompt("hot_load", "revisit", "Return to the exact same shared context after a short gap."),
        ),
        row(
            request_id="cpu-offload",
            prefix_id="prefix-cpu-offload-b",
            arrival_index=2,
            reuse_ratio=0.55,
            reuse_bucket="medium",
            scenario="cpu_offload",
            prompt=prompt("cpu_offload", "demote", "Touch the prefix once, release it, and leave it cold enough for CPU-to-SSD offload."),
        ),
        row(
            request_id="ssd-prefetch-seed",
            prefix_id="prefix-prefetch-c",
            arrival_index=3,
            reuse_ratio=0.70,
            reuse_bucket="high",
            scenario="ssd_prefetch",
            prompt=prompt("ssd_prefetch", "seed", "Warm the prefix on disk so that the next access is predictable."),
        ),
        row(
            request_id="ssd-prefetch-revisit",
            prefix_id="prefix-prefetch-c",
            arrival_index=4,
            reuse_ratio=0.70,
            reuse_bucket="high",
            scenario="ssd_prefetch",
            prompt=prompt("ssd_prefetch", "revisit", "Revisit the same prefix soon enough that CPU prefetch is worthwhile."),
        ),
        row(
            request_id="cold-drop",
            prefix_id="prefix-cold-drop-d",
            arrival_index=5,
            reuse_ratio=0.0,
            reuse_bucket="none",
            scenario="cold_drop",
            prompt=prompt("cold_drop", "single", "Use a one-off prefix that should be dropped after release."),
        ),
        row(
            request_id="recompute-bias",
            prefix_id="prefix-recompute-e",
            arrival_index=6,
            reuse_ratio=0.15,
            reuse_bucket="low",
            scenario="recompute_bias",
            prompt=prompt("recompute_bias", "fallback", "Request a low-value revisit where waiting for recompute is acceptable."),
        ),
        row(
            request_id="evict-cold-disk-seed",
            prefix_id="prefix-evict-f",
            arrival_index=7,
            reuse_ratio=0.70,
            reuse_bucket="high",
            scenario="evict_cold_disk",
            prompt=prompt("evict_cold_disk", "seed", "Warm a disk-resident prefix so a later cold revisit can evict it after prefetch waste."),
        ),
        row(
            request_id="evict-cold-disk-followup",
            prefix_id="prefix-evict-f",
            arrival_index=8,
            reuse_ratio=0.05,
            reuse_bucket="none",
            scenario="evict_cold_disk",
            prompt=prompt("evict_cold_disk", "followup", "Touch the same prefix lightly so a later cold evict is justifiable."),
        ),
    ]
    return [{**shared[0], **item} for item in rows]


def row(
    *,
    request_id: str,
    prefix_id: str,
    arrival_index: int,
    reuse_ratio: float,
    reuse_bucket: str,
    scenario: str,
    prompt: str,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "prefix_id": prefix_id,
        "prefix_hash": prefix_id,
        "cache_key": prefix_id,
        "arrival_index": arrival_index,
        "reuse_ratio": reuse_ratio,
        "reuse_bucket": reuse_bucket,
        "case": scenario,
        "metadata": {
            "scenario": scenario,
            "workload_type": "runtime_action_validation",
            "reuse_group": prefix_id,
            "shared_context": reuse_ratio > 0,
            "expected_reuse": 1 if reuse_ratio > 0 else 0,
        },
    }


def prompt(scenario: str, phase: str, instruction: str) -> str:
    prefix = (
        "AstraKV runtime validation context. "
        f"Scenario: {scenario}. "
        "This prompt intentionally reuses a long stable prefix so runtime cache actions become observable. "
    )
    filler = " ".join(["shared-context"] * 768)
    return (
        f"{prefix}{filler}\n\n"
        f"Phase: {phase}. {instruction}\n\n"
        "Summarize the request in exactly three concise bullets."
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_item in rows:
            handle.write(json.dumps(row_item, ensure_ascii=False) + "\n")


def write_manifest(path: Path, suite_path: Path, rows: list[dict[str, Any]]) -> None:
    scenario_counts = Counter(str(row_item.get("metadata", {}).get("scenario") or "") for row_item in rows)
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite_path": str(suite_path),
        "row_count": len(rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "required_artifacts": [
            "request_results.jsonl",
            "runtime_events_raw.jsonl",
            "runtime_command_receipts.jsonl",
            "runtime_structured_events.jsonl",
            "backend server log",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(path: Path, suite_path: Path, manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    scenario_counts = Counter(str(row_item.get("metadata", {}).get("scenario") or "") for row_item in rows)
    lines = [
        "# Runtime Action Validation Workload",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scenarios",
        "",
        "| scenario | rows |",
        "| --- | ---: |",
    ]
    for scenario, count in sorted(scenario_counts.items()):
        lines.append(f"| {scenario} | {count} |")
    lines.extend([
        "",
        "## Intended Coverage",
        "",
        "- `hot_load`: exercise dynamic load-target registration and a hot revisit.",
        "- `cpu_offload`: keep a released object hot enough to reach CPU, then allow demotion to SSD.",
        "- `ssd_prefetch`: revisit an SSD object soon enough that prefetch is justifiable before direct load.",
        "- `cold_drop`: keep a one-off prefix cold enough for terminal drop.",
        "- `recompute_bias`: make recompute an acceptable no-dispatch outcome.",
        "- `evict_cold_disk`: accumulate a cold disk-resident prefix for evict verification.",
        "",
        "## Files",
        "",
        f"- `{suite_path}`",
        f"- `{manifest_path}`",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
