"""Plan selective MoE expert loading from activation traces.

The output is an adapter-facing plan. This script does not load expert weights
or modify a serving runtime.
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

from astrakv.moe.expert_loader import (  # noqa: E402
    ExpertLoadDecision,
    ExpertLoadPlannerConfig,
    ExpertProfile,
    SelectiveExpertLoaderPlanner,
    load_expert_catalog,
    load_expert_profiles_from_summary,
    profile_from_summary_row,
    summarize_decisions,
    write_decisions_csv,
    write_hints_jsonl,
)
from astrakv.runtime.moe_events import parse_events_jsonl, summarize_events  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(args)
    catalog = load_expert_catalog(args.expert_catalog) if args.expert_catalog else {}
    planner = SelectiveExpertLoaderPlanner(config_from_args(args))
    decisions = planner.plan(profiles, catalog=catalog)

    plan_path = output_dir / args.plan_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_decisions_csv(plan_path, decisions)
    write_hints_jsonl(hints_path, decisions)
    write_report(report_path, args, profiles, decisions, plan_path, hints_path)
    write_manifest(manifest_path, args, profiles, decisions, plan_path, hints_path, report_path)

    print(f"MoE expert load plan written to {plan_path}")
    print(f"MoE expert load hints written to {hints_path}")
    print(f"MoE expert load report written to {report_path}")
    print(f"MoE expert load manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-summary", default="", help="P2-1 moe_expert_summary.csv path.")
    parser.add_argument("--moe-events", default="", help="P2-1 moe_expert_events.jsonl path.")
    parser.add_argument("--expert-catalog", default="", help="Optional expert weight catalog CSV/JSON/JSONL.")
    parser.add_argument("--output-dir", default="results/moe_expert_loading")
    parser.add_argument("--plan-name", default="moe_expert_load_plan.csv")
    parser.add_argument("--hints-name", default="moe_expert_load_hints.jsonl")
    parser.add_argument("--report-name", default="moe_expert_load_report.md")
    parser.add_argument("--manifest-name", default="moe_expert_load_manifest.json")
    parser.add_argument("--gpu-budget-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--cpu-budget-bytes", type=int, default=16 * 1024 * 1024 * 1024)
    parser.add_argument("--default-expert-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--hot-threshold", type=float, default=0.45)
    parser.add_argument("--warm-threshold", type=float, default=0.18)
    parser.add_argument("--drop-threshold", type=float, default=0.02)
    parser.add_argument("--disable-ssd", action="store_true")
    return parser.parse_args()


def load_profiles(args: argparse.Namespace) -> list[ExpertProfile]:
    profiles: list[ExpertProfile] = []
    if args.expert_summary:
        profiles.extend(load_expert_profiles_from_summary(args.expert_summary))
    if args.moe_events:
        events = parse_events_jsonl(args.moe_events)
        summary = summarize_events(events)
        profiles.extend(profile_from_summary_row(row) for row in summary["expert_rows"])
    if not profiles:
        raise SystemExit("Provide --expert-summary or --moe-events with at least one expert record.")
    return merge_profiles(profiles)


def merge_profiles(profiles: list[ExpertProfile]) -> list[ExpertProfile]:
    by_key: dict[str, dict[str, Any]] = {}
    total_activations = 0
    for profile in profiles:
        total_activations += profile.activation_count
        item = by_key.setdefault(
            profile.key,
            {
                "layer_id": profile.layer_id,
                "expert_id": profile.expert_id,
                "activation_count": 0,
                "token_count": 0,
                "score_sum": 0.0,
                "score_weight": 0,
                "bytes_observed": 0,
                "latency_ms": 0.0,
                "metadata": {},
            },
        )
        item["activation_count"] += profile.activation_count
        item["token_count"] += profile.token_count
        item["score_sum"] += profile.avg_score * max(1, profile.activation_count)
        item["score_weight"] += max(1, profile.activation_count)
        item["bytes_observed"] += profile.bytes_observed
        item["latency_ms"] += profile.latency_ms
        item["metadata"].update(profile.metadata)
    merged: list[ExpertProfile] = []
    for item in by_key.values():
        score_weight = max(1, int(item["score_weight"]))
        activations = int(item["activation_count"])
        merged.append(
            ExpertProfile(
                layer_id=item["layer_id"],
                expert_id=item["expert_id"],
                activation_count=activations,
                token_count=int(item["token_count"]),
                avg_score=float(item["score_sum"]) / score_weight,
                bytes_observed=int(item["bytes_observed"]),
                latency_ms=float(item["latency_ms"]),
                hotness_share=activations / max(1, total_activations),
                metadata=dict(item["metadata"]),
            )
        )
    return merged


def config_from_args(args: argparse.Namespace) -> ExpertLoadPlannerConfig:
    return ExpertLoadPlannerConfig(
        gpu_budget_bytes=args.gpu_budget_bytes,
        cpu_budget_bytes=args.cpu_budget_bytes,
        default_expert_bytes=args.default_expert_bytes,
        hot_threshold=args.hot_threshold,
        warm_threshold=args.warm_threshold,
        drop_threshold=args.drop_threshold,
        ssd_enabled=not args.disable_ssd,
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    profiles: list[ExpertProfile],
    decisions: list[ExpertLoadDecision],
    plan_path: Path,
    hints_path: Path,
) -> None:
    summary = summarize_decisions(decisions)
    lines = [
        "# MoE Expert Selective Loading Plan",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Expert summary: `{args.expert_summary or 'none'}`",
        f"- MoE events: `{args.moe_events or 'none'}`",
        f"- Expert catalog: `{args.expert_catalog or 'none'}`",
        f"- GPU budget bytes: `{args.gpu_budget_bytes}`",
        f"- CPU budget bytes: `{args.cpu_budget_bytes}`",
        f"- Default expert bytes: `{args.default_expert_bytes}`",
        "",
        "## Outputs",
        "",
        f"- Load plan CSV: `{plan_path}`",
        f"- Scheduler hints JSONL: `{hints_path}`",
        "",
        "## Summary",
        "",
        f"- Expert profiles: `{len(profiles)}`",
        f"- Decisions: `{summary['total_experts']}`",
        f"- Total expert bytes: `{summary['total_expert_bytes']}`",
        f"- Planned GPU bytes: `{summary['planned_gpu_bytes']}`",
        f"- Planned CPU bytes: `{summary['planned_cpu_bytes']}`",
        f"- Planned SSD bytes: `{summary['planned_ssd_bytes']}`",
        f"- Estimated GPU bytes saved: `{summary['estimated_gpu_bytes_saved']}`",
        f"- Estimated GPU saving rate: `{summary['estimated_gpu_saving_rate']:.6f}`",
        "",
        "### Actions",
        "",
        "| action | count |",
        "| --- | ---: |",
    ]
    for action, count in summary["action_counts"].items():
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Top Decisions",
            "",
            "| layer | expert | action | tier | priority | size bytes | activations | hotness | reason |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for decision in decisions[:50]:
        layer = "" if decision.layer_id is None else decision.layer_id
        lines.append(
            f"| {layer} | {decision.expert_id} | {decision.action.value} | {decision.target_tier} | "
            f"{decision.priority:.6f} | {decision.size_bytes} | {decision.activation_count} | "
            f"{decision.hotness_share:.6f} | {decision.reason} |"
        )
    if len(decisions) > 50:
        lines.append(f"| ... |  |  |  |  |  |  |  | {len(decisions) - 50} more decision(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `load_gpu` and `keep_gpu` represent hot experts that should be resident in GPU memory when adapter support exists.",
            "- `keep_cpu` represents warm experts that should stay in CPU memory for faster demand loading.",
            "- `offload_ssd` represents cold experts that are useful to retain outside GPU/CPU budgets.",
            "- `on_demand` and `drop` are fallback choices when lower tiers are unavailable or the profile is weak.",
            "- This script emits passive adapter hints. A GPU runtime adapter must consume them before claiming real expert weight movement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    profiles: list[ExpertProfile],
    decisions: list[ExpertLoadDecision],
    plan_path: Path,
    hints_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-moe-expert-load-plan-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "expert_summary": args.expert_summary,
            "moe_events": args.moe_events,
            "expert_catalog": args.expert_catalog,
        },
        "outputs": {
            "plan_csv": str(plan_path),
            "hints_jsonl": str(hints_path),
            "report": str(report_path),
        },
        "config": {
            "gpu_budget_bytes": args.gpu_budget_bytes,
            "cpu_budget_bytes": args.cpu_budget_bytes,
            "default_expert_bytes": args.default_expert_bytes,
            "hot_threshold": args.hot_threshold,
            "warm_threshold": args.warm_threshold,
            "drop_threshold": args.drop_threshold,
            "ssd_enabled": not args.disable_ssd,
        },
        "profile_count": len(profiles),
        "summary": summarize_decisions(decisions),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
