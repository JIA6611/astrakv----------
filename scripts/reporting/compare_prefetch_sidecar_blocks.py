"""Compare repeated Sidecar-B blocks while preserving prompt identity.

Independent service blocks may have strong order drift.  This report keeps
each block visible and also averages repeated observations for the same
request before computing paired percentile/bootstrap statistics, so repeated
prompts are not treated as independent samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reporting.summarize_prefetch_phase_ttft import (
    load_jsonl,
    paired_summary,
    phase_rows,
)


def compare(
    blocks: list[tuple[str, Path, Path]], *, phase: str = "far",
) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    block_summaries: dict[str, dict[str, Any]] = {}
    for name, baseline_path, variant_path in blocks:
        baseline = phase_rows(load_jsonl(baseline_path), phase)
        variant = phase_rows(load_jsonl(variant_path), phase)
        loaded[name] = (baseline, variant)
        block_summaries[name] = paired_summary(baseline, variant)

    common_ids: set[str] | None = None
    for baseline, variant in loaded.values():
        ids = set(baseline) & set(variant)
        common_ids = ids if common_ids is None else common_ids & ids
    ordered_ids = sorted(common_ids or ())
    if not ordered_ids:
        raise ValueError("blocks have no common paired request ids")

    averaged_baseline: dict[str, dict[str, Any]] = {}
    averaged_variant: dict[str, dict[str, Any]] = {}
    for request_id in ordered_ids:
        baseline_rows = [pair[0][request_id] for pair in loaded.values()]
        variant_rows = [pair[1][request_id] for pair in loaded.values()]
        baseline_pair_ids = {
            str(row.get("prefetch_pair_id") or "") for row in baseline_rows
        }
        variant_pair_ids = {
            str(row.get("prefetch_pair_id") or "") for row in variant_rows
        }
        if len(baseline_pair_ids | variant_pair_ids) != 1:
            raise ValueError(f"pair-id mismatch across blocks for {request_id}")
        pair_id = next(iter(baseline_pair_ids | variant_pair_ids))
        averaged_baseline[request_id] = {
            "request_id": request_id,
            "prefetch_pair_id": pair_id,
            "ttft_ms": sum(float(row["ttft_ms"]) for row in baseline_rows)
            / len(baseline_rows),
        }
        averaged_variant[request_id] = {
            "request_id": request_id,
            "prefetch_pair_id": pair_id,
            "ttft_ms": sum(float(row["ttft_ms"]) for row in variant_rows)
            / len(variant_rows),
        }

    averaged = paired_summary(averaged_baseline, averaged_variant)
    block_p50_deltas = [
        summary.get("p50_delta_percent") for summary in block_summaries.values()
    ]
    block_directions = [
        "improve" if float(delta) < 0 else "regress"
        for delta in block_p50_deltas if delta is not None
    ]
    ci = averaged.get("p50_delta_bootstrap_ci_percent") or [None, None]
    ci_all_negative = (
        len(ci) == 2 and ci[0] is not None and ci[1] is not None
        and float(ci[1]) < 0
    )
    return {
        "schema": "astrakv-prefetch-sidecar-block-comparison-v1",
        "phase": phase,
        "block_count": len(blocks),
        "repeated_prompt_count": len(ordered_ids),
        "blocks": block_summaries,
        "prompt_averaged_across_blocks": averaged,
        "interpretation": {
            "block_p50_directions": block_directions,
            "block_direction_consistent": len(set(block_directions)) <= 1,
            "prompt_averaged_p50_ci_all_negative": ci_all_negative,
            "stable_end_to_end_claim": (
                len(set(block_directions)) <= 1 and ci_all_negative
            ),
        },
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Sidecar-B two-order block comparison",
        "",
        f"phase: `{data['phase']}`; blocks: {data['block_count']}; "
        f"repeated prompts: {data['repeated_prompt_count']}",
        "",
        "| block | baseline P50 | variant P50 | delta | baseline P95 | variant P95 | delta | wins | P50 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in data["blocks"].items():
        lines.append(
            f"| {name} | {row.get('baseline_p50_ms'):.2f} | "
            f"{row.get('variant_p50_ms'):.2f} | {row.get('p50_delta_percent'):.2f}% | "
            f"{row.get('baseline_p95_ms'):.2f} | {row.get('variant_p95_ms'):.2f} | "
            f"{row.get('p95_delta_percent'):.2f}% | {row.get('variant_wins')}/{row.get('paired_count')} | "
            f"`{row.get('p50_delta_bootstrap_ci_percent')}` |"
        )
    avg = data["prompt_averaged_across_blocks"]
    lines.extend([
        "",
        "## Prompt-averaged across blocks",
        "",
        f"- P50: {avg.get('baseline_p50_ms'):.2f} ms -> {avg.get('variant_p50_ms'):.2f} ms "
        f"({avg.get('p50_delta_percent'):.2f}%)",
        f"- P95: {avg.get('baseline_p95_ms'):.2f} ms -> {avg.get('variant_p95_ms'):.2f} ms "
        f"({avg.get('p95_delta_percent'):.2f}%)",
        f"- wins: {avg.get('variant_wins')}/{avg.get('paired_count')}",
        f"- P50 bootstrap 95% CI: `{avg.get('p50_delta_bootstrap_ci_percent')}`",
        "",
        "## Interpretation",
        "",
        f"- block directions: `{data['interpretation']['block_p50_directions']}`",
        f"- block direction consistent: `{data['interpretation']['block_direction_consistent']}`",
        f"- prompt-averaged P50 CI all negative: `{data['interpretation']['prompt_averaged_p50_ci_all_negative']}`",
        f"- stable end-to-end claim: `{data['interpretation']['stable_end_to_end_claim']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block", nargs=3, action="append", metavar=("NAME", "BASELINE", "VARIANT"),
        required=True,
    )
    parser.add_argument("--phase", choices=("first", "far", "near", "all"), default="far")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = compare(
        [(name, Path(baseline), Path(variant)) for name, baseline, variant in args.block],
        phase=args.phase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
