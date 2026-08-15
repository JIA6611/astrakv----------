"""Summarize paired TTFT for a fire-consume workload phase."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("first", "far", "near", "all"), default="near",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        rows.append(record)
    return rows


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def phase_rows(rows: Iterable[dict[str, Any]], phase: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if phase != "all" and str(row.get("prefetch_phase") or "") != phase:
            continue
        request_id = str(row.get("request_id") or "")
        try:
            ttft = float(row.get("ttft_ms"))
        except (TypeError, ValueError):
            continue
        if not request_id or ttft <= 0.0:
            continue
        selected[request_id] = row
    return selected


def paired_summary(
    baseline: dict[str, dict[str, Any]], variant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_ids = sorted(set(baseline) & set(variant))
    pairs: list[tuple[float, float]] = []
    missing_pair_id: list[str] = []
    for request_id in common_ids:
        left, right = baseline[request_id], variant[request_id]
        left_pair = str(left.get("prefetch_pair_id") or "")
        right_pair = str(right.get("prefetch_pair_id") or "")
        if left_pair != right_pair:
            missing_pair_id.append(request_id)
            continue
        pairs.append((float(left["ttft_ms"]), float(right["ttft_ms"])))

    base_values = [pair[0] for pair in pairs]
    variant_values = [pair[1] for pair in pairs]
    base_p50 = percentile(base_values, 0.50)
    variant_p50 = percentile(variant_values, 0.50)
    base_p95 = percentile(base_values, 0.95)
    variant_p95 = percentile(variant_values, 0.95)

    def delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None or left <= 0.0:
            return None
        return (right - left) / left * 100.0

    wins = sum(1 for left, right in pairs if right < left)
    return {
        "paired_count": len(pairs),
        "missing_or_mismatched_pair_id_count": len(missing_pair_id),
        "missing_or_mismatched_pair_ids": missing_pair_id,
        "baseline_p50_ms": base_p50,
        "variant_p50_ms": variant_p50,
        "p50_delta_percent": delta(base_p50, variant_p50),
        "baseline_p95_ms": base_p95,
        "variant_p95_ms": variant_p95,
        "p95_delta_percent": delta(base_p95, variant_p95),
        "variant_wins": wins,
        "variant_win_rate": (wins / len(pairs)) if pairs else None,
        "paired_request_ids": [request_id for request_id in common_ids if request_id not in missing_pair_id],
        "p50_delta_bootstrap_ci_percent": bootstrap_percentile_ci(pairs, 0.50),
        "p95_delta_bootstrap_ci_percent": bootstrap_percentile_ci(pairs, 0.95),
    }


def bootstrap_percentile_ci(
    pairs: list[tuple[float, float]], quantile: float, *, samples: int = 2000,
) -> list[float | None]:
    if not pairs:
        return [None, None]
    rng = random.Random(0)
    deltas: list[float] = []
    for _ in range(samples):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        left = percentile((pair[0] for pair in sampled), quantile)
        right = percentile((pair[1] for pair in sampled), quantile)
        if left is not None and right is not None and left > 0.0:
            deltas.append((right - left) / left * 100.0)
    return [percentile(deltas, 0.025), percentile(deltas, 0.975)]


def main() -> int:
    args = parse_args()
    baseline_rows = load_jsonl(args.baseline)
    variant_rows = load_jsonl(args.variant)
    baseline = phase_rows(baseline_rows, args.phase)
    variant = phase_rows(variant_rows, args.phase)
    record = {
        "schema": "astrakv-prefetch-phase-ttft-v1",
        "phase": args.phase,
        "baseline_request_count": len(baseline),
        "variant_request_count": len(variant),
        **paired_summary(baseline, variant),
    }
    encoded = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
