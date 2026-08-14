"""Build the final evict-B vs LRU two-layer comparison report.

Inputs (DGX run outputs):
  --evict-b-root   arm root for evict-B (contains <dataset>/*-state dirs)
  --lru-root       arm root for LMCache-native LRU
  --offline-root   kv-core offline eviction root (workload/offline_eviction manifests)

The report covers the acceptance gates from the plan:
  1. hard gate: evict-B arm produced >= 1 action=evict, status=completed receipt
  2. primary metric: TTFT p95 (and p50/mean) per arm
  3. bad-eviction rate per arm (evict-B side is receipt-backed; LRU side is
     LMCache-internal and therefore not observable)
  4. offline LRU/FIFO/AstraKV/Belady comparison (Belady labeled offline oracle)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGG = load_module("aggregate_evict_ablation", Path(__file__).with_name("aggregate_evict_ablation.py"))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return round(ordered[index], 4)


def collect_arm(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"present": False}
    state_dirs = sorted(
        path for path in root.rglob("*")
        if path.is_dir() and path.name.endswith("-state")
    )
    runs = []
    for state_dir in state_dirs:
        run_dir = state_dir.with_name(state_dir.name[: -len("-state")])
        runs.append(
            AGG.aggregate_run(
                state_dir,
                run_dir if run_dir.exists() else None,
                reaccess_window_ms=30_000,
            )
        )
    merged = AGG.merge_runs(runs)
    raw_ttft: list[float] = []
    evict_receipts: list[dict[str, Any]] = []
    for path in root.rglob("request_results.jsonl"):
        if "/warmup-" in path.as_posix() or "\\warmup-" in path.as_posix():
            continue
        for row in AGG.load_jsonl(path):
            value = AGG.as_float(row.get("ttft_ms"))
            if value > 0:
                raw_ttft.append(value)
    for path in root.rglob("runtime_command_receipts.jsonl"):
        for row in AGG.load_jsonl(path):
            if AGG.as_str(row.get("action")) == "evict" and AGG.as_str(row.get("status")) == "completed":
                evict_receipts.append(row)
    tier_after = Counter(AGG.as_str(row.get("tier_after")) for row in evict_receipts)
    if raw_ttft:
        merged["ttft_ms"] = {
            "count": len(raw_ttft),
            "mean": round(statistics.mean(raw_ttft), 4),
            "p50": percentile(raw_ttft, 50),
            "p95": percentile(raw_ttft, 95),
        }
    merged["evict_completed_tier_after"] = dict(tier_after)
    return {"present": True, "state_dirs": len(state_dirs), "metrics": merged}


def collect_offline(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for manifest in sorted(root.rglob("offline_eviction_manifest.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        workload = manifest.parent.parent.name
        for policy in payload.get("policies", []):
            rows.append({
                "workload": workload,
                "policy": policy.get("policy"),
                "hit": round(float(policy.get("total_hit_rate") or 0.0), 4),
                "ssd_read": int(policy.get("ssd_read_proxy_bytes") or 0),
                "ttft": round(float(policy.get("ttft_proxy_ms_mean") or 0.0), 2),
                "oracle": bool(policy.get("is_offline_oracle")),
            })
    return rows


def fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def render(evict_b: dict[str, Any], lru: dict[str, Any], offline: list[dict[str, Any]]) -> str:
    lines = [
        "# evict-B vs LRU 双层对照实验报告",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## 硬门槛检查",
        "",
    ]
    if evict_b.get("present"):
        completed = evict_b["metrics"].get("evict_completed", 0)
        tier = evict_b["metrics"].get("evict_completed_tier_after", {})
        gate = "PASS" if completed >= 1 else "FAIL"
        lines += [
            f"- evict-B arm `action=evict, status=completed` receipts: **{completed}** → **{gate}**",
            f"- tier_after 分布: `{tier}`",
            "",
        ]
    else:
        lines += ["- evict-B arm: **无数据**（等待 DGX 结果）", ""]

    lines += ["## 真机对比（主指标 TTFT p95）", "", "| arm | evict receipts | TTFT count | TTFT mean | p50 | p95 | 坏驱逐率 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, arm in (("evict-B", evict_b), ("LRU(原生)", lru)):
        if not arm.get("present"):
            lines.append(f"| {name} | — | — | — | — | — | — |")
            continue
        m = arm["metrics"]
        t = m.get("ttft_ms", {})
        bad = m.get("bad_eviction", {})
        rate = fmt(bad.get("rate"))
        lines.append(
            f"| {name} | {m.get('evict_completed', 0)} | {fmt(t.get('count'))} | "
            f"{fmt(t.get('mean'))} | {fmt(t.get('p50'))} | {fmt(t.get('p95'))} | {rate} |"
        )

    lines += ["", "## 离线层（LRU/FIFO/AstraKV/Belady）", "", "| workload | policy | hit | SSD read (B) | TTFT proxy (ms) | oracle |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    if not offline:
        lines.append("| （无离线数据） | | | | | |")
    for row in offline:
        lines.append(
            f"| {row['workload']} | {row['policy']} | {row['hit']} | {row['ssd_read']} | "
            f"{row['ttft']} | {'yes' if row['oracle'] else ''} |"
        )

    lines += [
        "",
        "## 验收清单",
        "",
        "- [ ] 硬门槛：evict-B arm ≥1 条 `action=evict, status=completed` receipt（分层：tier_after=ssd 为 CPU 层、cpu/none 为 SSD 层）",
        "- [ ] 离线：AstraKV（evict-B 离线化身）不差于 LRU，Belady 为最低上界",
        "- [ ] 真机：evict-B TTFT p95 ≤ LRU（3 次重复出均值±方差）",
        "- [ ] 真机：evict-B 坏驱逐率（30s 窗口）作为决策质量指标报告",
        "",
        "## 结论",
        "",
        "（等待 DGX 真机数据后填充：离线增益 → 真机一致性 → 跨臂 TTFT/坏驱逐率结论。）",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evict-b-root", type=Path, required=True)
    parser.add_argument("--lru-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/evict_final_report.md"))
    args = parser.parse_args()
    evict_b = collect_arm(args.evict_b_root)
    lru = collect_arm(args.lru_root)
    offline = collect_offline(args.offline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(evict_b, lru, offline), encoding="utf-8")
    print(f"Final report written to {args.output}")


if __name__ == "__main__":
    main()
