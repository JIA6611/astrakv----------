"""Validate and summarize one QASPER online-control experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKLOADS = ("random", "grouped")
ROLES = ("baseline", "variant")


def summarize_suite(suite_dir: str | Path) -> dict[str, Any]:
    root = Path(suite_dir)
    summaries: dict[str, Any] = {}
    for workload in WORKLOADS:
        workload_summary: dict[str, Any] = {}
        for role in ROLES:
            run = root / workload / role
            state = root / workload / f"{role}-state"
            requests = _read_jsonl(run / "request_results.jsonl")
            if not requests:
                raise ValueError(f"{workload}/{role}: missing request results")
            if any(row.get("status") != "ok" for row in requests):
                raise ValueError(f"{workload}/{role}: request failures present")
            if not (run / "experiment_manifest.json").is_file():
                raise ValueError(f"{workload}/{role}: missing experiment manifest")
            quality = _quality_metrics(run / "qasper_quality_summary.csv")
            for filename in (
                "backend_capabilities.json",
                "backend_binding_events.jsonl",
                "runtime_events_raw.jsonl",
                "runtime_command_receipts.jsonl",
            ):
                if not (state / filename).is_file():
                    raise ValueError(f"{workload}/{role}: missing runtime artifact {filename}")
            receipts = _read_jsonl(state / "runtime_command_receipts.jsonl")
            if role == "variant" and not any(
                row.get("status") == "completed" and int((row.get("metadata") or {}).get("removed") or 0) > 0
                for row in receipts
            ):
                raise ValueError(f"{workload}/{role}: missing completed receipt with removed object")
            workload_summary[role] = {
                "request_count": len(requests),
                "success_count": sum(row.get("status") == "ok" for row in requests),
                "quality": quality,
                "runtime_state_dir": str(state),
                "receipt_count": len(receipts),
                "completed_drop_count": sum(
                    row.get("status") == "completed" and int((row.get("metadata") or {}).get("removed") or 0) > 0
                    for row in receipts
                ),
            }
        comparison_dir = root / workload / "comparison"
        workload_summary["comparison"] = _comparison_summary(comparison_dir, workload=workload)
        summaries[workload] = workload_summary
    return {"schema": "astrakv-task1-qasper-online-suite-v1", "workloads": summaries}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quality_metrics(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing QASPER quality summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty QASPER quality summary: {path}")
    return {str(row.get("metric") or ""): str(row.get("value") or "") for row in rows}


def _comparison_summary(path: Path, *, workload: str) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"{workload}: missing comparison directory")
    status_path = path / "compare_exit_status.txt"
    if not status_path.is_file():
        raise ValueError(f"{workload}: missing compare_exit_status.txt")
    status_text = status_path.read_text(encoding="utf-8").strip()
    try:
        exit_status = int(status_text)
    except ValueError as exc:
        raise ValueError(f"{workload}: invalid compare exit status") from exc
    if exit_status != 0:
        raise ValueError(f"{workload}: compare_real_runs failed with exit status {exit_status}")
    paired_manifest_path = path / "paired_run_manifest.json"
    if not paired_manifest_path.is_file():
        raise ValueError(f"{workload}: missing paired_run_manifest.json")
    try:
        paired_manifest = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{workload}: invalid paired_run_manifest.json") from exc
    if not isinstance(paired_manifest, dict):
        raise ValueError(f"{workload}: invalid paired_run_manifest.json")
    if paired_manifest.get("eligible") is not True:
        raise ValueError(f"{workload}: paired run evidence is ineligible")
    if paired_manifest.get("claim_scope") != "online_control":
        raise ValueError(f"{workload}: paired run claim scope is not online_control")
    for filename in ("comparison_results.csv", "comparison_report.md"):
        if not (path / filename).is_file():
            raise ValueError(f"{workload}: missing comparison artifact {filename}")
    return {
        "compare_exit_status": exit_status,
        "paired_claim_eligible": True,
        "claim_scope": "online_control",
        "paired_run_manifest": str(paired_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    summary = summarize_suite(args.suite_dir)
    output = Path(args.output_dir or args.suite_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "suite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# QASPER Online Control Suite Report", ""]
    for workload, roles in summary["workloads"].items():
        lines.extend([f"## {workload}", "", "| role | requests | success | completed DROP | quality |", "| --- | ---: | ---: | ---: | --- |"])
        for role in ROLES:
            record = roles[role]
            quality = ", ".join(f"{key}={value}" for key, value in sorted(record["quality"].items()))
            lines.append(f"| {role} | {record['request_count']} | {record['success_count']} | {record['completed_drop_count']} | {quality} |")
        comparison = roles["comparison"]
        lines.extend([
            "",
            f"Paired comparison eligible: `{comparison['paired_claim_eligible']}`",
            f"Claim scope: `{comparison['claim_scope']}`",
        ])
        lines.append("")
    (output / "suite_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
