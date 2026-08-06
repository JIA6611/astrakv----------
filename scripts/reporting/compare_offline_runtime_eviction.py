"""Compare offline object-scheduler eviction decisions with runtime evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.eviction import (  # noqa: E402
    offline_decision_from_record,
    load_runtime_events_jsonl,
)
from astrakv.runtime.eviction_agreement import compare_eviction  # noqa: E402


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions = [offline_decision_from_record(row, run_id=args.run_id, ordinal=index) for index, row in enumerate(read_csv(args.offline_decisions), start=1)]
    events = load_runtime_events_jsonl(args.runtime_events)
    indexes = load_request_indexes(args.workload_manifest)
    summary = compare_eviction(
        decisions,
        events,
        request_indexes=indexes,
        prediction_window_requests=args.prediction_window_requests,
        comparison_scope=args.comparison_scope,
    )
    write_rows(output / "eviction_agreement.csv", [row.to_record() for row in summary.rows])
    write_confusion(output / "eviction_confusion_matrix.csv", summary.metrics)
    write_manifest(output / "eviction_agreement_manifest.json", args, summary.to_record())
    write_report(output / "eviction_agreement_report.md", args, summary)
    print(f"Eviction agreement report written to {output}")
    return 0 if summary.ground_truth_status == "valid" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-decisions", required=True, help="object_schedule_decisions.csv path")
    parser.add_argument("--runtime-events", required=True, help="Normalized RuntimeEvictionEvent JSONL path")
    parser.add_argument("--run-id", required=True, help="Run identifier shared by policy decisions and runtime events")
    parser.add_argument("--workload-manifest", default="", help="Optional request/workload JSONL with request_id and arrival_index")
    parser.add_argument("--prediction-window-requests", type=int, default=10)
    parser.add_argument(
        "--comparison-scope",
        choices=("runtime", "vm_poc"),
        default="runtime",
        help="Keep real structured runtime evidence separate from mmap VM-PoC execution evidence.",
    )
    parser.add_argument("--output-dir", default="results/eviction_agreement")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_request_indexes(path: str | Path) -> dict[str, int]:
    if not path or not Path(path).exists():
        return {}
    result: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = str(row.get("request_id") or "")
        try:
            result[request_id] = int(row.get("arrival_index"))
        except (TypeError, ValueError):
            continue
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "object_key", "object_level", "predicted_action", "actual_action", "classification",
        "tier_match", "decision_time_ns", "runtime_time_ns", "lead_time_ns", "decision_bytes", "runtime_bytes", "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["metadata"] = json.dumps(payload.get("metadata", {}), ensure_ascii=False)
            writer.writerow(payload)


def write_confusion(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "exact_action_agreement", "tier_transition_accuracy", "byte_weighted_agreement"):
            writer.writerow({"metric": key, "value": metrics.get(key, "")})


def write_manifest(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "offline_decisions": args.offline_decisions,
        "runtime_events": args.runtime_events,
        "run_id": args.run_id,
        "workload_manifest": args.workload_manifest,
        "prediction_window_requests": args.prediction_window_requests,
        "comparison_scope": args.comparison_scope,
        "summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(path: Path, args: argparse.Namespace, summary: Any) -> None:
    metrics = summary.metrics
    lines = [
        "# Offline / Runtime Eviction Agreement Report", "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
        "## Inputs", "",
        f"- Offline decisions: `{args.offline_decisions}`",
        f"- Runtime events: `{args.runtime_events}`",
        f"- Run id: `{args.run_id}`",
        f"- Prediction window: `{args.prediction_window_requests}` requests", "",
        f"- Comparison scope: `{args.comparison_scope}`", "",
        "## Ground Truth", "",
        f"- Status: `{summary.ground_truth_status}`",
        f"- Reason: {summary.reason}", "",
        "## Metrics", "",
        "| metric | value |", "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "", "## Interpretation", "",
        "- `runtime` uses only successful `runtime_structured` events; `vm_poc` uses only successful `vm_poc_execution` events.",
        "- These scopes are intentionally separate and must not be combined into one accuracy conclusion.",
        "- Log-derived events are preserved as evidence but do not create a real-runtime accuracy claim.",
        "- Disk I/O, cache store, and cache hit are not treated as eviction ground truth.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
