"""Build a compact SLO-oriented E11 demo from a real paired run."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.reporting.analyze_e11_request_attribution import analyze


REGIME = "scan_pollution_past_observed"


def _round(value: float) -> float:
    return round(float(value), 2)


def build_demo_result(
    attribution: dict[str, Any], *, slo_ms: float, min_requests: int = 3
) -> dict[str, Any]:
    matching = [
        (name, cell)
        for name, cell in attribution.get("cells", {}).items()
        if name.split("/", 1)[-1].endswith(f"__{REGIME}")
    ]
    if len(matching) != 1:
        return {
            "schema": "astrakv-e11-slo-demo-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "inconclusive",
            "reason": f"expected_one_scan_pollution_cell_found_{len(matching)}",
            "regime": REGIME,
            "slo_ms": _round(slo_ms),
            "eligible": False,
        }

    cell_name, cell = matching[0]
    rows = [
        row
        for row in cell.get("requests", [])
        if row.get("phase") == "post_divergence"
        and float(row.get("lru_ttft_ms") or 0) > 0
        and float(row.get("astrakv_ttft_ms") or 0) > 0
    ]
    lru_values = [float(row["lru_ttft_ms"]) for row in rows]
    astra_values = [float(row["astrakv_ttft_ms"]) for row in rows]
    diverged = cell.get("first_divergence_ordinal") is not None
    eligible = diverged and len(rows) >= min_requests

    result: dict[str, Any] = {
        "schema": "astrakv-e11-slo-demo-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "inconclusive",
        "reason": "insufficient_post_divergence_evidence",
        "regime": REGIME,
        "cell": cell_name,
        "slo_ms": _round(slo_ms),
        "eligible": eligible,
        "evidence": {
            "victim_sequence_diverged": diverged,
            "first_divergence_ordinal": cell.get("first_divergence_ordinal"),
            "post_divergence_paired_requests": len(rows),
            "minimum_required_requests": min_requests,
            "source_evidence_gaps": attribution.get("evidence_gaps", []),
        },
    }
    if not eligible:
        return result

    lru_p50 = statistics.median(lru_values)
    astra_p50 = statistics.median(astra_values)
    improvement_ms = lru_p50 - astra_p50
    improvement_percent = improvement_ms / lru_p50 * 100.0
    lru_pass = lru_p50 <= slo_ms
    astra_pass = astra_p50 <= slo_ms
    improved = improvement_ms > 0
    result.update(
        {
            "status": "pass" if improved and astra_pass else "no_improvement",
            "reason": (
                "astrakv_lower_p50_and_meets_slo"
                if improved and astra_pass
                else "astrakv_did_not_both_improve_p50_and_meet_slo"
            ),
            "metrics": {
                "lru_ttft_p50_ms": _round(lru_p50),
                "astrakv_ttft_p50_ms": _round(astra_p50),
                "improvement_ms": _round(improvement_ms),
                "improvement_percent": _round(improvement_percent),
            },
            "slo": {
                "lru_pass": lru_pass,
                "astrakv_pass": astra_pass,
                "lru_headroom_ms": _round(slo_ms - lru_p50),
                "astrakv_headroom_ms": _round(slo_ms - astra_p50),
                "headroom_gain_ms": _round(improvement_ms),
            },
        }
    )
    return result


def render_terminal(result: dict[str, Any]) -> str:
    metrics = result.get("metrics")
    border = "=" * 58
    if not metrics:
        return "\n".join(
            [
                border,
                "E11 Scan-Pollution TTFT P50 Demo",
                "Result: INCONCLUSIVE (insufficient post-divergence evidence)",
                border,
            ]
        )
    return "\n".join(
        [
            border,
            "E11 Scan-Pollution TTFT P50 Demo",
            f"LMCache LRU          {metrics['lru_ttft_p50_ms']:10.2f} ms",
            f"AstraKV-W evict-B    {metrics['astrakv_ttft_p50_ms']:10.2f} ms",
            f"P50 reduction        {metrics['improvement_percent']:10.2f} %",
            border,
        ]
    )


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# E11 Scan-Pollution SLO Demo",
        "",
        f"- 结果状态：`{result['status']}`",
        f"- 证据状态：`{'有效' if result.get('eligible') else '证据不足'}`",
        f"- TTFT P50 运行目标：`<= {result['slo_ms']:.2f} ms`",
    ]
    metrics = result.get("metrics")
    if metrics:
        slo = result["slo"]
        lines += [
            "",
            "| 策略 | TTFT P50 | SLO | SLO 余量 |",
            "| --- | ---: | --- | ---: |",
            f"| LMCache LRU | {metrics['lru_ttft_p50_ms']:.2f} ms | "
            f"{'通过' if slo['lru_pass'] else '未通过'} | {slo['lru_headroom_ms']:+.2f} ms |",
            f"| AstraKV-W evict-B | {metrics['astrakv_ttft_p50_ms']:.2f} ms | "
            f"{'通过' if slo['astrakv_pass'] else '未通过'} | {slo['astrakv_headroom_ms']:+.2f} ms |",
            "",
            f"evict-B 的 TTFT P50 相对 LRU 下降 {metrics['improvement_percent']:.2f}% "
            f"（{metrics['improvement_ms']:.2f} ms），SLO 余量增加 "
            f"{slo['headroom_gain_ms']:.2f} ms。",
        ]
    else:
        lines += ["", f"当前不能形成 P50 对比：`{result['reason']}`。"]
    lines += [
        "",
        "说明：该结果来自单次现场配对 Demo，用于展示真机执行与方向性收益；正式结论仍以多轮实验为准。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline",), default="baseline")
    parser.add_argument("--slo-ms", type=float, default=1600.0)
    parser.add_argument("--min-requests", type=int, default=3)
    args = parser.parse_args()
    if args.slo_ms <= 0 or args.min_requests <= 0:
        parser.error("--slo-ms and --min-requests must be positive")

    result = build_demo_result(
        analyze(args.root, args.role),
        slo_ms=args.slo_ms,
        min_requests=args.min_requests,
    )
    json_path = args.root / "e11_slo_demo.json"
    md_path = args.root / "e11_slo_demo.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_terminal(result))


if __name__ == "__main__":
    main()
