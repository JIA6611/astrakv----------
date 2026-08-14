#!/usr/bin/env python3
"""Build the load-vs-recompute regime matrix report.

Consumes ``regime_cells.jsonl`` produced by
``run_load_recompute_regime_suite.sh`` and aggregates, per cell, TTFT p95,
UMA peak, KV block budget and actually loaded tokens.  Cross-arm paired TTFT
deltas (vs the recompute-only arm) and memory verdicts answer which workload
regime favors load and which favors recompute/partial.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reporting.validate_kv_core_acceptance import (  # noqa: E402
    kv_block_budget,
    load_run_artifact,
    paired_ttft_bootstrap,
    percentile,
    resource_peaks,
)


SCHEMA = "astrakv-load-recompute-regime-v1"
TTFT_WIN_WORKLOADS = {"repeated_long_prefix", "queued_concurrency"}
MEMORY_WIN_WORKLOADS = {"constrained_kv_churn", "random_no_reuse"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True, help="regime_cells.jsonl")
    parser.add_argument("--output", required=True, help="Markdown report path (a .json twin is written too).")
    args = parser.parse_args()
    cells = _read_cells(Path(args.cells))
    summary = build(cells)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary["markdown"], encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(summary["record"], indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary["record"], sort_keys=True))
    return 0


def _read_cells(path: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("variant_dir") and row.get("baseline_dir"):
            cells.append(row)
    return cells


def build(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: dict[str, dict[str, Any]] = {}
    for cell in cells:
        workload = str(cell.get("workload") or "")
        arm = str(cell.get("arm") or "")
        variant_dir = Path(str(cell["variant_dir"]))
        stats[(workload, arm)] = _cell_stats(variant_dir)
        by_workload[workload].append({**cell, "stats": stats[(workload, arm)]})

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "cells": [
            {
                "workload": cell.get("workload"),
                "arm": cell.get("arm"),
                "phase": cell.get("phase"),
                "stats": stats[(str(cell.get("workload") or ""), str(cell.get("arm") or ""))],
            }
            for cell in cells
        ],
        "comparisons": {},
        "verdicts": {},
    }
    lines = [
        "# Load-vs-Recompute Regime Matrix",
        "",
        f"- Cells: `{len(cells)}`",
        "",
        "## Per-Cell Metrics",
        "",
        "| workload | arm | eligible | TTFT p95 (ms) | TTFT delta vs recompute (95% CI) | UMA peak (GB) | KV block budget | loaded tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparisons: dict[str, dict[str, Any]] = {}
    for workload, workload_cells in sorted(by_workload.items()):
        recompute = next((c for c in workload_cells if c.get("arm") == "recompute_only"), None)
        workload_comparisons: dict[str, Any] = {}
        if recompute is not None:
            recompute_rows = _requests(Path(str(recompute["variant_dir"])))
            for other in workload_cells:
                if other.get("arm") == "recompute_only":
                    continue
                point, interval = paired_ttft_bootstrap(
                    recompute_rows, _requests(Path(str(other["variant_dir"]))),
                )
                workload_comparisons[str(other.get("arm"))] = {
                    "ttft_p95_delta_percent": point,
                    "bootstrap_ci_percent": list(interval),
                }
        comparisons[workload] = workload_comparisons
        for cell in workload_cells:
            s = cell["stats"]
            delta = workload_comparisons.get(str(cell.get("arm")))
            delta_text = "reference"
            if delta is not None and delta["ttft_p95_delta_percent"] is not None:
                lo, hi = delta["bootstrap_ci_percent"]
                delta_text = f"{delta['ttft_p95_delta_percent']:.2f}% [{lo:.2f}, {hi:.2f}]"
            uma_gb = (
                f"{s['uma_peak_bytes'] / 1024**3:.2f}"
                if s.get("uma_peak_bytes") is not None else "-"
            )
            eligible = "-"
            if s.get("acceptance_eligible") is True:
                eligible = "✓"
            elif s.get("acceptance_eligible") is False:
                eligible = "✗"
            lines.append(
                f"| {workload} | {cell.get('arm')} | {eligible} | {_fmt(s.get('ttft_p95_ms'))} | {delta_text} | "
                f"{uma_gb} | {_fmt(s.get('kv_block_budget'))} | {s.get('loaded_tokens')} |"
            )
    record["comparisons"] = comparisons

    verdicts: dict[str, Any] = {}
    for workload, workload_cells in sorted(by_workload.items()):
        verdicts[workload] = _verdict(workload, workload_cells, stats, comparisons)
    record["verdicts"] = verdicts

    lines.extend(["", "## Verdicts", ""])
    for workload, verdict in sorted(verdicts.items()):
        lines.append(f"- **{workload}**: {verdict['label']}")
        lines.append(f"  - Evidence: `{json.dumps(verdict['evidence'], sort_keys=True)}`")
    summary = {"markdown": "\n".join(lines) + "\n", "record": record}
    return summary


def _cell_stats(variant_dir: Path) -> dict[str, Any]:
    requests = _requests(variant_dir)
    ok = [row for row in requests if str(row.get("status") or "") == "ok"]
    ttfts = sorted(
        float(row["ttft_ms"]) for row in ok
        if row.get("ttft_ms") not in (None, "")
    )
    uma = resource_peaks(load_run_artifact(variant_dir, "uma_resource_samples.jsonl"))
    accounting = load_run_artifact(variant_dir, "kv_core_request_accounting.jsonl")
    loaded_tokens = sum(
        max(0, int(row.get("actual_loaded_tokens") or 0))
        for row in accounting if row.get("terminal") is True
    )
    acceptance = _load_json(variant_dir / "acceptance.json")
    if acceptance is None:
        acceptance = _load_json(variant_dir.parent / "acceptance.json")
    return {
        "request_count": len(requests),
        "ok_count": len(ok),
        "ttft_p95_ms": percentile(ttfts, 95),
        "uma_peak_bytes": uma.get("cgroup_memory_current_bytes"),
        "kv_block_budget": kv_block_budget(variant_dir),
        "loaded_tokens": loaded_tokens,
        "acceptance_eligible": None if acceptance is None else acceptance.get("eligible"),
        "acceptance_errors": [] if acceptance is None else list(acceptance.get("errors") or [])[:5],
    }


def _verdict(
    workload: str,
    workload_cells: list[dict[str, Any]],
    stats: dict[tuple[str, str], dict[str, Any]],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    cells = {str(c.get("arm")): c for c in workload_cells}
    recompute = cells.get("recompute_only")
    full = cells.get("full")
    partial = cells.get("partial")
    off = cells.get("off")
    evidence: dict[str, Any] = {}
    if workload in TTFT_WIN_WORKLOADS and recompute is not None:
        deltas = {
            arm: comparisons[workload][arm]
            for arm in ("full", "partial")
            if arm in comparisons[workload]
        }
        evidence["ttft_deltas_vs_recompute"] = deltas
        if deltas and all(
            d["ttft_p95_delta_percent"] is not None
            and d["ttft_p95_delta_percent"] < 0.0
            and d["bootstrap_ci_percent"][1] is not None
            and d["bootstrap_ci_percent"][1] < 0.0
            for d in deltas.values()
        ):
            return {"label": "load wins TTFT (both full and partial beat recompute-only)", "evidence": evidence}
        return {"label": "inconclusive TTFT (CI crosses zero or evidence missing)", "evidence": evidence}

    if workload in MEMORY_WIN_WORKLOADS and recompute is not None and full is not None:
        rec_uma = stats[(workload, "recompute_only")].get("uma_peak_bytes")
        full_uma = stats[(workload, "full")].get("uma_peak_bytes")
        partial_uma = stats[(workload, "partial")].get("uma_peak_bytes") if partial is not None else None
        evidence["uma_peak_bytes"] = {
            "recompute_only": rec_uma,
            "full": full_uma,
            "partial": partial_uma,
            "off": stats[(workload, "off")].get("uma_peak_bytes") if off is not None else None,
        }
        memory_ok = (
            rec_uma is not None and full_uma is not None
            and rec_uma <= full_uma
            and (partial_uma is None or rec_uma <= partial_uma)
        )
        off_ok = True
        if off is not None:
            off_uma = stats[(workload, "off")].get("uma_peak_bytes")
            off_ok = rec_uma is not None and off_uma is not None and rec_uma <= off_uma * 1.02
        evidence["within_off_plus_2pct"] = off_ok
        if memory_ok and off_ok:
            return {"label": "recompute-only wins memory (UMA <= full/partial and <= off + 2%)", "evidence": evidence}
        if memory_ok:
            return {"label": "recompute-only wins memory vs load arms (off gate not applied)", "evidence": evidence}
        return {"label": "recompute-only does NOT win memory", "evidence": evidence}
    return {"label": "no verdict defined for this workload/arms", "evidence": evidence}


def _requests(run_dir: Path) -> list[dict[str, Any]]:
    return load_run_artifact(run_dir, "request_results.jsonl")


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
