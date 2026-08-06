"""Predict next-token MoE experts and emit passive prefetch hints.

This script evaluates predictions from existing MoE route traces. It does not
call a model router, prefetch expert weights, or modify a serving runtime.
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

from astrakv.moe.expert_predictor import (  # noqa: E402
    ExpertPrediction,
    ExpertPredictorConfig,
    RouterAwareExpertPredictor,
    load_expert_load_plan,
    observations_from_events,
    summarize_predictions,
    write_predictions_csv,
    write_prefetch_hints_jsonl,
)
from astrakv.runtime.moe_events import parse_events_jsonl  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = parse_events_jsonl(args.moe_events)
    observations = observations_from_events(events)
    load_plan = load_expert_load_plan(args.expert_load_plan) if args.expert_load_plan else {}
    predictor = RouterAwareExpertPredictor(config_from_args(args))
    predictions = predictor.predict(observations, load_plan=load_plan)

    predictions_path = output_dir / args.predictions_name
    hints_path = output_dir / args.hints_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_predictions_csv(predictions_path, predictions)
    write_prefetch_hints_jsonl(hints_path, predictions)
    write_report(report_path, args, observations, predictions, predictions_path, hints_path)
    write_manifest(manifest_path, args, observations, predictions, predictions_path, hints_path, report_path)

    print(f"MoE expert predictions written to {predictions_path}")
    print(f"MoE expert prefetch hints written to {hints_path}")
    print(f"MoE expert prediction report written to {report_path}")
    print(f"MoE expert prediction manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-events", required=True, help="P2-1 moe_expert_events.jsonl path.")
    parser.add_argument("--expert-load-plan", default="", help="Optional P2-2 moe_expert_load_plan.csv path.")
    parser.add_argument("--output-dir", default="results/moe_expert_predictions")
    parser.add_argument("--predictions-name", default="moe_expert_predictions.csv")
    parser.add_argument("--hints-name", default="moe_expert_prefetch_hints.jsonl")
    parser.add_argument("--report-name", default="moe_expert_prediction_report.md")
    parser.add_argument("--manifest-name", default="moe_expert_prediction_manifest.json")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--predictor-name",
        choices=["next_token", "history_window", "profile_guided"],
        default="next_token",
        help="Prediction policy name for ablation reports.",
    )
    parser.add_argument("--history-window", type=int, default=1, help="Recent route window used by advanced predictors.")
    parser.add_argument("--previous-token-weight", type=float, default=0.55)
    parser.add_argument("--history-window-weight", type=float, default=0.20)
    parser.add_argument("--transition-weight", type=float, default=0.0)
    parser.add_argument("--hot-expert-weight", type=float, default=0.30)
    parser.add_argument("--load-plan-weight", type=float, default=0.15)
    parser.add_argument("--gpu-resident-bonus", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExpertPredictorConfig:
    return ExpertPredictorConfig(
        top_k=args.top_k,
        predictor_name=args.predictor_name,
        history_window=args.history_window,
        previous_token_weight=args.previous_token_weight,
        history_window_weight=args.history_window_weight,
        transition_weight=args.transition_weight,
        hot_expert_weight=args.hot_expert_weight,
        load_plan_weight=args.load_plan_weight,
        gpu_resident_bonus=args.gpu_resident_bonus,
        min_score=args.min_score,
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    observations: list[object],
    predictions: list[ExpertPrediction],
    predictions_path: Path,
    hints_path: Path,
) -> None:
    summary = summarize_predictions(predictions)
    lines = [
        "# MoE Expert Predictor Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- MoE events: `{args.moe_events}`",
        f"- Expert load plan: `{args.expert_load_plan or 'none'}`",
        f"- Top-k: `{args.top_k}`",
        f"- Predictor name: `{args.predictor_name}`",
        f"- History window: `{args.history_window}`",
        f"- Previous-token weight: `{args.previous_token_weight}`",
        f"- History-window weight: `{args.history_window_weight}`",
        f"- Transition weight: `{args.transition_weight}`",
        f"- Hot expert weight: `{args.hot_expert_weight}`",
        f"- Load plan weight: `{args.load_plan_weight}`",
        f"- GPU resident bonus: `{args.gpu_resident_bonus}`",
        "",
        "## Outputs",
        "",
        f"- Predictions CSV: `{predictions_path}`",
        f"- Prefetch hints JSONL: `{hints_path}`",
        "",
        "## Summary",
        "",
        f"- Route observations: `{len(observations)}`",
        f"- Predictions: `{summary['prediction_count']}`",
        f"- Evaluated predictions: `{summary['evaluated_prediction_count']}`",
        f"- Predicted expert total: `{summary['predicted_expert_total']}`",
        f"- Evaluated predicted expert total: `{summary['evaluated_predicted_expert_total']}`",
        f"- Actual expert total: `{summary['actual_expert_total']}`",
        f"- Hit expert total: `{summary['hit_expert_total']}`",
        f"- Wasted expert total: `{summary['wasted_expert_total']}`",
        f"- Missed expert total: `{summary['missed_expert_total']}`",
        f"- Expert prefetch hit rate: `{summary['expert_prefetch_hit_rate']:.6f}`",
        f"- Expert prefetch waste rate: `{summary['expert_prefetch_waste_rate']:.6f}`",
        f"- Expert coverage rate: `{summary['expert_coverage_rate']:.6f}`",
        "",
        "## Top Predictions",
        "",
        "| request | layer | predictor | window | transition | source token | target token | predicted | actual | hit | wasted | missed | coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for prediction in predictions[:50]:
        layer = "" if prediction.layer_id is None else prediction.layer_id
        lines.append(
            f"| {prediction.request_id or 'n/a'} | {layer} | {prediction.predictor_name} | "
            f"{prediction.window_size} | {prediction.transition_score:.6f} | "
            f"{prediction.source_token_index} | {prediction.target_token_index} | "
            f"{','.join(prediction.predicted_experts)} | "
            f"{','.join(prediction.actual_experts)} | {','.join(prediction.hit_experts)} | "
            f"{','.join(prediction.wasted_experts)} | {','.join(prediction.missed_experts)} | "
            f"{prediction.coverage:.6f} |"
        )
    if len(predictions) > 50:
        lines.append(f"| ... |  |  |  |  |  |  |  |  | {len(predictions) - 50} more prediction(s) omitted |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `next_token` preserves the original previous-token predictor.",
            "- `history_window` adds recent same-request route history and transition statistics.",
            "- `profile_guided` uses the same local signals but labels the output for policy ablation with ProfileDB/load-plan artifacts.",
            "- Hit and waste rates are computed only for predictions whose next-token route is present in the trace.",
            "- `moe_expert_prefetch_hints.jsonl` is passive metadata. Runtime adapters must consume it before claiming real expert prefetch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    observations: list[object],
    predictions: list[ExpertPrediction],
    predictions_path: Path,
    hints_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-moe-expert-prediction-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "moe_events": args.moe_events,
            "expert_load_plan": args.expert_load_plan,
        },
        "outputs": {
            "predictions_csv": str(predictions_path),
            "hints_jsonl": str(hints_path),
            "report": str(report_path),
        },
        "config": {
            "top_k": args.top_k,
            "predictor_name": args.predictor_name,
            "history_window": args.history_window,
            "previous_token_weight": args.previous_token_weight,
            "history_window_weight": args.history_window_weight,
            "transition_weight": args.transition_weight,
            "hot_expert_weight": args.hot_expert_weight,
            "load_plan_weight": args.load_plan_weight,
            "gpu_resident_bonus": args.gpu_resident_bonus,
            "min_score": args.min_score,
        },
        "observation_count": len(observations),
        "summary": summarize_predictions(predictions),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
