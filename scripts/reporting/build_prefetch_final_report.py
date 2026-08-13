"""Build the final Prefetch-A/B acceptance report from run artifacts.

Consumes the aggregator outputs produced after the DGX runs:

- E3 acceptance:              validate_e3_prefetch_acceptance.py output
- standard 2x2 (Sidecar-B):  validate_prefetch_2x2_ablation.py output
- transfer (Profile-B):       same validator output from the transfer runner
- adaptation (hybrid):        analyze_prefetch_adaptation.py output

The report contains the acceptance checklist, the four-grid comparison table
(workload x strategy x TTFT P50/P95 x hit/waste), the B generalization tiers
(Sidecar-B oracle vs Profile-B transfer), the online-adaptation windows, and
the Phase-2 conflict counters.  Missing inputs simply omit their section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _hit_rate(consumed: int, wasted: int) -> str:
    total = consumed + wasted
    return "-" if total == 0 else f"{consumed / total * 100:.1f}%"


def _cell_rows(cells: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for cell in sorted(
        cells,
        key=lambda item: (str(item.get("dataset") or ""), str(item.get("cell") or "")),
    ):
        benchmark = cell.get("benchmark") or {}
        a = cell.get("prefetch_a") or {}
        b = cell.get("prefetch_b") or {}
        rows.append([
            str(cell.get("dataset") or ""),
            str(cell.get("cell") or ""),
            _fmt(benchmark.get("ttft_p50_ms")),
            _fmt(benchmark.get("ttft_p95_ms")),
            _hit_rate(int(a.get("tickets_consumed") or 0), int(a.get("tickets_wasted") or 0)),
            f"{int(b.get('completed_with_bytes') or 0)}/{int(b.get('receipt_count') or 0)}",
        ])
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    *,
    e3: dict[str, Any] | None,
    ablation: dict[str, Any] | None,
    transfer: dict[str, Any] | None,
    adaptation: dict[str, Any] | None,
) -> str:
    sections: list[str] = ["# AstraKV 双预取（A/B）验收报告"]

    sections.append("## 1. 功能层验收（E3）")
    if e3 is None:
        sections.append("未提供 E3 acceptance JSON。")
    else:
        acceptance = e3.get("acceptance") or {}
        sections.append(
            "- B completed receipt: `%s`\n"
            "- A CONSUMED ticket: `%s`\n"
            "- B 失败均带诊断: `%s`\n"
            "- 失败原因: `%s`" % (
                acceptance.get("b_completed_receipt_found"),
                acceptance.get("a_consumed_ticket_found"),
                acceptance.get("b_failures_have_diagnostics"),
                json.dumps(
                    {
                        role.get("role"): role.get("prefetch_b", {}).get("failure_reasons", {})
                        for role in e3.get("roles", [])
                    },
                    ensure_ascii=False,
                ),
            )
        )

    sections.append("## 2. 四格对比表（标准 2×2，B-only = Sidecar-B 上界）")
    if ablation is None:
        sections.append("未提供标准 2×2 validation JSON。")
    else:
        sections.append(_markdown_table(
            ["workload", "策略", "TTFT P50(ms)", "TTFT P95(ms)", "A hit rate", "B completed/receipts"],
            _cell_rows(ablation.get("cells", [])),
        ))
        sections.append("> 说明：标准 2×2 的 B-only 格使用同源 exact-next sidecar，属 oracle 上界。")

    sections.append("## 3. B 泛化（Profile-B，训练/评测分离 + 部分重叠）")
    if transfer is None:
        sections.append("未提供 transfer validation JSON。")
    else:
        sections.append(_markdown_table(
            ["workload", "策略", "TTFT P50(ms)", "TTFT P95(ms)", "A hit rate", "B completed/receipts"],
            _cell_rows(transfer.get("cells", [])),
        ))
        sections.append("> Profile-B：离线画像 + 结构 hints，测试集与训练集共享热 context 但请求不同。")

    sections.append("## 4. 在线学习自适应（hybrid，cold → after-N）")
    if adaptation is None:
        sections.append("未提供 adaptation JSON。")
    else:
        rows: list[list[str]] = []
        for window in adaptation.get("windows", []):
            a = window.get("prefetch_a") or {}
            b = window.get("prefetch_b") or {}
            rows.append([
                str(window.get("window")),
                "-".join(str(v) for v in window.get("arrival_range", [])),
                str(window.get("request_count")),
                _fmt(window.get("ttft_p50_ms")),
                _fmt(window.get("ttft_p95_ms")),
                str(a.get("decision_count")),
                str(b.get("completed_with_bytes")),
            ])
        sections.append(_markdown_table(
            ["window", "arrival", "requests", "TTFT P50(ms)", "TTFT P95(ms)", "A decisions", "B completed"],
            rows,
        ))

    sections.append("## 5. 阶段二冲突计数（both 格）")
    conflict: dict[str, Any] = {}
    for payload in (ablation, transfer):
        if payload is None:
            continue
        totals = payload.get("both_cell_conflict_totals") or {}
        for key, value in totals.items():
            conflict[key] = int(conflict.get(key, 0)) + int(value or 0)
    if not conflict:
        sections.append("未提供冲突计数。")
    else:
        for key, value in sorted(conflict.items()):
            sections.append(f"- {key}: {value}")
        sections.append("> 归零或有明确解释后，both 模式方可宣称生产可用。")

    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3", default="")
    parser.add_argument("--ablation-2x2", default="")
    parser.add_argument("--transfer", default="")
    parser.add_argument("--adaptation", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        e3=_load(Path(args.e3)) if args.e3 else None,
        ablation=_load(Path(args.ablation_2x2)) if args.ablation_2x2 else None,
        transfer=_load(Path(args.transfer)) if args.transfer else None,
        adaptation=_load(Path(args.adaptation)) if args.adaptation else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
