"""Evaluate the E11 workload-regime matrix without forcing one global winner."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.reporting.analyze_e11_request_attribution import analyze
from scripts.reporting.evaluate_e11_native_cpu_mvp import evaluate as evaluate_native
from scripts.reporting.evaluate_e11_target import (
    ARMS,
    artifact_path,
    as_float,
    as_str,
    load_jsonl,
    selected_run_dirs,
)


REGIMES = (
    "recency_aligned",
    "scan_pollution_past_observed",
    "profile_shift_or_stale",
)
FORMAL_CELLS_PER_REGIME = 6


def regime_for_cell(cell: str) -> str:
    dataset = cell.split("/", 1)[-1]
    return dataset.rsplit("__", 1)[-1] if "__" in dataset else ""


def _bootstrap_mean_ci(values: list[float], *, samples: int = 4000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(0)
    boot = []
    for _ in range(samples):
        boot.append(statistics.mean(values[rng.randrange(len(values))] for _ in values))
    boot.sort()
    return [
        round(boot[max(0, int(0.025 * len(boot)) - 1)], 4),
        round(boot[min(len(boot) - 1, int(0.975 * len(boot)))], 4),
    ]


def paired_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [as_float(row.get("delta_percent")) for row in rows]
    return {
        "paired_count": len(rows),
        "mean_delta_percent": round(statistics.mean(deltas), 4) if deltas else None,
        "median_delta_percent": round(statistics.median(deltas), 4) if deltas else None,
        "mean_delta_ci_percent": _bootstrap_mean_ci(deltas),
    }


def classify_winner(
    summary: dict[str, Any],
    *,
    correctness_valid: bool,
    astrakv_quality_not_worse: bool,
    lru_quality_not_worse: bool,
    astrakv_quality_improved: bool = False,
    lru_quality_improved: bool = False,
    formal_design_ready: bool = True,
    equivalence_band_percent: float = 2.0,
) -> str:
    if not correctness_valid or not formal_design_ready or not summary.get("paired_count"):
        return "inconclusive"
    lower, upper = summary["mean_delta_ci_percent"]
    if lower is None or upper is None:
        return "inconclusive"
    if upper < 0.0 and astrakv_quality_not_worse and astrakv_quality_improved:
        return "astrakv_wins"
    if lower > 0.0 and lru_quality_not_worse and lru_quality_improved:
        return "lru_wins"
    if lower >= -equivalence_band_percent and upper <= equivalence_band_percent:
        return "tie"
    return "inconclusive"


def regime_delta_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    cell_deltas = [
        as_float(cell["paired_ttft"].get("mean_delta_percent"))
        for cell in cells
        if cell["paired_ttft"].get("mean_delta_percent") is not None
    ]
    return {
        "paired_count": sum(int(cell["paired_ttft"].get("paired_count") or 0) for cell in cells),
        "independent_cell_count": len(cell_deltas),
        "mean_delta_percent": round(statistics.mean(cell_deltas), 4) if cell_deltas else None,
        "median_cell_delta_percent": (
            round(statistics.median(cell_deltas), 4) if cell_deltas else None
        ),
        "cell_mean_delta_ci_percent": _bootstrap_mean_ci(cell_deltas),
        # Keep the common field name for classify_winner/report consumers.
        "mean_delta_ci_percent": _bootstrap_mean_ci(cell_deltas),
    }


def _request_success(run_dir: Path) -> bool:
    rows = load_jsonl(artifact_path(run_dir, "request_results.jsonl"))
    return bool(rows) and all(as_str(row.get("status")) == "ok" for row in rows)


def _profile_contract(run_dir: Path, regime: str) -> bool:
    rows = load_jsonl(artifact_path(run_dir, "workload_source.jsonl"))
    if not rows:
        return False
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if as_str(metadata.get("e11_regime")) != regime:
            return False
        if as_float(metadata.get("e11_policy_reuse_ratio")) != 0.0:
            return False
        if metadata.get("e11_policy_reuse_ratio_overridden") is not True:
            return False
    return True


def _analysis_rows(regime: str, cell: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(cell.get("requests") or [])
    if regime == "recency_aligned":
        return [row for row in rows if as_float(row.get("reuse_ratio")) > 0.0]
    return [row for row in rows if as_str(row.get("phase")) == "post_divergence"]


def _sum_quality(cells: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        bad_completed = sum(int(cell["bad_eviction"][arm].get("completed") or 0) for cell in cells)
        bad_reaccessed = sum(
            int(cell["bad_eviction"][arm].get("reaccessed_within_window") or 0)
            for cell in cells
        )
        load_values = [cell["native_load"][arm].get("bytes_loaded") for cell in cells]
        result[arm] = {
            "bad_eviction_completed": bad_completed,
            "bad_eviction_reaccessed": bad_reaccessed,
            "bad_eviction_rate": (
                round(bad_reaccessed / bad_completed, 6) if bad_completed else None
            ),
            "native_load_bytes": (
                sum(int(value) for value in load_values)
                if load_values and all(value is not None for value in load_values) else None
            ),
        }
    return result


def _quality_not_worse(quality: dict[str, Any], left: str, right: str) -> bool:
    comparisons: list[bool] = []
    for field in ("bad_eviction_rate", "native_load_bytes"):
        left_value = quality[left].get(field)
        right_value = quality[right].get(field)
        if left_value is not None and right_value is not None:
            comparisons.append(float(left_value) <= float(right_value))
    return bool(comparisons) and all(comparisons)


def _quality_improved(quality: dict[str, Any], left: str, right: str) -> bool:
    comparable: list[tuple[float, float]] = []
    for field in ("bad_eviction_rate", "native_load_bytes"):
        left_value = quality[left].get(field)
        right_value = quality[right].get(field)
        if left_value is not None and right_value is not None:
            comparable.append((float(left_value), float(right_value)))
    return bool(comparable) and all(left <= right for left, right in comparable) and any(
        left < right for left, right in comparable
    )


def evaluate_regimes(root: Path, role: str = "baseline") -> dict[str, Any]:
    native = evaluate_native(root, role)
    attribution = analyze(root, role)
    run_dirs = {
        arm: selected_run_dirs(root / arm, role) for arm in ARMS
    }
    pairing_by_cell = {
        as_str(row.get("cell")): row for row in native["pairing"].get("cells", [])
    }
    cell_results: dict[str, Any] = {}
    for cell_name, cell_attr in attribution["cells"].items():
        regime = regime_for_cell(cell_name)
        gates: dict[str, bool] = {
            "known_regime": regime in REGIMES,
            "pairing_eligible": pairing_by_cell.get(cell_name, {}).get("eligible") is True,
        }
        for arm in ARMS:
            run_dir = run_dirs[arm].get(cell_name)
            install = native["installations"][arm]["per_cell"].get(cell_name, {})
            evictions = native["native_evictions"][arm]["per_cell"].get(cell_name, {})
            load = cell_attr["native_load"][arm]
            timing = cell_attr["selector_overhead"][arm]
            gates[f"{arm}_run_present"] = run_dir is not None
            gates[f"{arm}_request_success"] = bool(run_dir and _request_success(run_dir))
            gates[f"{arm}_profile_contract"] = bool(
                run_dir and regime in REGIMES and _profile_contract(run_dir, regime)
            )
            gates[f"{arm}_installation_valid"] = install.get("valid") is True
            gates[f"{arm}_native_completion_exact"] = bool(
                evictions.get("selected", 0) > 0
                and evictions.get("selected") == evictions.get("completed")
            )
            gates[f"{arm}_native_load_artifact"] = load.get("artifact_present") is True
            gates[f"{arm}_selector_timing"] = timing.get("instrumented") is True
        if regime in {"scan_pollution_past_observed", "profile_shift_or_stale"}:
            gates["victim_sequence_diverged"] = (
                native["victim_sequence_comparison"]["per_cell"]
                .get(cell_name, {}).get("diverged") is True
            )
        correctness_valid = all(gates.values())
        rows = _analysis_rows(regime, cell_attr)
        cell_results[cell_name] = {
            "regime": regime,
            "correctness_valid": correctness_valid,
            "gates": gates,
            "analysis_subset": (
                "reuse_requests" if regime == "recency_aligned" else "post_divergence"
            ),
            "paired_ttft": paired_delta_summary(rows),
            "native_load": cell_attr["native_load"],
            "selector_overhead": cell_attr["selector_overhead"],
            "bad_eviction": {
                arm: native["bad_eviction"][arm]["per_cell"].get(cell_name, {})
                for arm in ARMS
            },
            "first_divergence_ordinal": cell_attr.get("first_divergence_ordinal"),
        }

    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        cells = [value for value in cell_results.values() if value["regime"] == regime]
        rows: list[dict[str, Any]] = []
        for cell_name, cell in cell_results.items():
            if cell["regime"] == regime:
                rows.extend(_analysis_rows(regime, attribution["cells"][cell_name]))
        request_summary = paired_delta_summary(rows)
        summary = regime_delta_summary(cells)
        quality = _sum_quality(cells) if cells else {arm: {} for arm in ARMS}
        correctness_valid = bool(cells) and all(cell["correctness_valid"] for cell in cells)
        formal_design_ready = len(cells) >= FORMAL_CELLS_PER_REGIME
        astra_not_worse = _quality_not_worse(quality, "arm-evict-b", "arm-lru")
        lru_not_worse = _quality_not_worse(quality, "arm-lru", "arm-evict-b")
        astra_improved = _quality_improved(quality, "arm-evict-b", "arm-lru")
        lru_improved = _quality_improved(quality, "arm-lru", "arm-evict-b")
        regimes[regime] = {
            "cell_count": len(cells),
            "correctness_valid": correctness_valid,
            "paired_ttft": summary,
            "exploratory_request_level_ttft": request_summary,
            "formal_design_ready": formal_design_ready,
            "required_independent_cells": FORMAL_CELLS_PER_REGIME,
            "quality": quality,
            "astrakv_quality_not_worse": astra_not_worse,
            "lru_quality_not_worse": lru_not_worse,
            "astrakv_quality_improved": astra_improved,
            "lru_quality_improved": lru_improved,
            "winner": classify_winner(
                summary,
                correctness_valid=correctness_valid,
                astrakv_quality_not_worse=astra_not_worse,
                lru_quality_not_worse=lru_not_worse,
                astrakv_quality_improved=astra_improved,
                lru_quality_improved=lru_improved,
                formal_design_ready=formal_design_ready,
            ),
        }
    return {
        "schema": "astrakv-e11-workload-regime-evaluation-v1",
        "root": str(root),
        "role": role,
        "cells": cell_results,
        "regimes": regimes,
        "all_cells_correctness_valid": bool(cell_results) and all(
            cell["correctness_valid"] for cell in cell_results.values()
        ),
        "global_winner_intentionally_undefined": True,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# E11 workload-regime 评估报告",
        "",
        f"- 结果目录：`{result['root']}`",
        f"- 全部 cell 正确性：`{result['all_cells_correctness_valid']}`",
        "- 全局 winner：不定义；结论按 workload regime 给出。",
        "",
        "| regime | cells | formal ready | correctness | pairs | mean cell delta | cell 95% CI | winner |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for regime, row in result["regimes"].items():
        ttft = row["paired_ttft"]
        lines.append(
            f"| {regime} | {row['cell_count']} | {row['formal_design_ready']} | "
            f"{row['correctness_valid']} | "
            f"{ttft['paired_count']} | {ttft['mean_delta_percent']}% | "
            f"{ttft['mean_delta_ci_percent']} | {row['winner']} |"
        )
    lines += ["", "## Cell 门禁", ""]
    for cell, row in result["cells"].items():
        lines += [
            f"### {cell}",
            "",
            f"- correctness_valid：`{row['correctness_valid']}`",
            f"- gates：`{row['gates']}`",
            f"- paired TTFT：`{row['paired_ttft']}`",
            f"- native load：`{row['native_load']}`",
            f"- selector overhead：`{row['selector_overhead']}`",
            f"- bad eviction：`{row['bad_eviction']}`",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline",), default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = evaluate_regimes(args.root, args.role)
    output = args.output or args.root / "e11_regime_report.md"
    json_output = args.json_output or args.root / "e11_regime_result.json"
    output.write_text(render_report(result), encoding="utf-8")
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(output),
        "json": str(json_output),
        "all_cells_correctness_valid": result["all_cells_correctness_valid"],
        "winners": {name: row["winner"] for name, row in result["regimes"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
