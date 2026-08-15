"""Evaluate an E11 evict-B vs LRU run against the target acceptance gates.

Usage:
  python scripts/reporting/evaluate_e11_target.py --root DIR [--output PATH]

DIR is an evict suite root (e.g. results/evict-e11-<ts> or
results/evict-b-vs-lru-<ts>) containing arm-evict-b/ and arm-lru/ with
rep-*/<dataset>/{baseline,variant}/... artifacts.

Outputs a markdown acceptance report with the three-tier conclusion:
  1) real-machine evict-B < LRU (completed receipts + p95 + repeat direction)
  2) chain executed but advantage not demonstrated
  3) inconclusive (missing receipts / too few effective evictions / high not_found)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def as_str(v: Any) -> str:
    return "" if v is None else str(v)


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return round(ordered[idx], 4)


def is_warmup(path: Path) -> bool:
    return "/warmup-" in path.as_posix() or "\\warmup-" in path.as_posix()


def collect_ttft(arm_root: Path) -> dict[str, list[float]]:
    """Return ttft lists grouped by dataset."""
    by_dataset: dict[str, list[float]] = {}
    for path in arm_root.rglob("request_results.jsonl"):
        if is_warmup(path):
            continue
        dataset = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        for row in load_jsonl(path):
            value = as_float(row.get("ttft_ms"))
            if value > 0:
                by_dataset.setdefault(dataset, []).append(value)
    return by_dataset


def collect_receipts(arm_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for path in arm_root.rglob("runtime_command_receipts.jsonl"):
        dataset = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        bucket = counts.setdefault(dataset, {})
        for row in load_jsonl(path):
            key = f"{as_str(row.get('action'))}:{as_str(row.get('status'))}"
            bucket[key] = bucket.get(key, 0) + 1
    return counts


def collect_completed_evicts(arm_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in arm_root.rglob("runtime_command_receipts.jsonl"):
        dataset = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        tiers = result.setdefault(dataset, [])
        for row in load_jsonl(path):
            if as_str(row.get("action")) == "evict" and as_str(row.get("status")) == "completed":
                tiers.append(f"{as_str(row.get('tier_before'))}->{as_str(row.get('tier_after'))}")
    return result


def evict_metrics(arm_root: Path) -> dict[str, Any]:
    total = 0
    completed = 0
    not_found = 0
    for path in arm_root.rglob("runtime_command_receipts.jsonl"):
        for row in load_jsonl(path):
            if as_str(row.get("action")) != "evict":
                continue
            total += 1
            status = as_str(row.get("status"))
            if status == "completed":
                completed += 1
            elif status == "not_found":
                not_found += 1
    return {
        "evict_attempts": total,
        "evict_completed": completed,
        "evict_not_found": not_found,
        "effective_eviction_rate": round(completed / total, 4) if total else None,
        "not_found_rate": round(not_found / total, 4) if total else None,
    }


def bad_eviction(arm_root: Path, window_ms: int = 30_000) -> dict[str, Any]:
    evicted_at: dict[str, int] = {}
    for path in arm_root.rglob("runtime_command_receipts.jsonl"):
        for row in load_jsonl(path):
            if as_str(row.get("action")) == "evict" and as_str(row.get("status")) == "completed":
                key = as_str(row.get("backend_object_id") or row.get("object_key"))
                ts = as_int(row.get("timestamp_ns"))
                if key:
                    evicted_at[key] = max(evicted_at.get(key, 0), ts)
    reaccessed: set[str] = set()
    for path in arm_root.rglob("runtime_events_raw.jsonl"):
        for row in load_jsonl(path):
            action = as_str(row.get("action"))
            if action not in {"cache_hit", "cache_load"}:
                continue
            key = as_str(row.get("backend_object_id") or row.get("object_key"))
            ts = as_int(row.get("timestamp_ns"))
            evict_ts = evicted_at.get(key)
            if key and evict_ts is not None and evict_ts < ts and (ts - evict_ts) <= window_ms * 1_000_000:
                reaccessed.add(key)
    return {
        "evicted_keys": len(evicted_at),
        "reaccessed_within_window": len(reaccessed),
        "rate": round(len(reaccessed) / len(evicted_at), 4) if evicted_at else None,
        "window_ms": window_ms,
    }


def crash_check(root: Path) -> list[str]:
    problems = []
    for path in root.rglob("*-server.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for needle in ("FileNotFoundError", "EngineDeadError", "OutOfMemoryError", "CUDA out of memory"):
            if needle in text:
                problems.append(f"{path.name}: {needle}")
                break
    return problems


def request_success(arm_root: Path) -> dict[str, Any]:
    ok = 0
    failed = 0
    for path in arm_root.rglob("request_results.jsonl"):
        if is_warmup(path):
            continue
        for row in load_jsonl(path):
            status = as_str(row.get("status"))
            if status == "ok":
                ok += 1
            else:
                failed += 1
    total = ok + failed
    return {
        "ok": ok,
        "failed": failed,
        "success_rate": round(ok / total, 4) if total else None,
    }


def bootstrap_ci(deltas: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in deltas]
        means.append(statistics.mean(sample))
    means.sort()
    return round(means[int(0.025 * len(means))], 4), round(means[int(0.975 * len(means))], 4)


def paired_ttft(evict_b: dict[str, list[float]], lru: dict[str, list[float]]) -> dict[str, Any]:
    """Per-dataset per-repeat direction plus overall paired bootstrap CI on p95 deltas."""
    # p95 per dataset per arm
    eb_p95 = {ds: pct(vals, 95) or 0.0 for ds, vals in evict_b.items()}
    lru_p95 = {ds: pct(vals, 95) or 0.0 for ds, vals in lru.items()}
    deltas = [eb_p95[ds] - lru_p95[ds] for ds in eb_p95 if ds in lru_p95]
    lower, upper = bootstrap_ci(deltas) if deltas else (None, None)
    direction = [ds for ds in eb_p95 if ds in lru_p95 and eb_p95[ds] < lru_p95[ds]]
    return {
        "evict_b_p95_by_dataset": {k: round(v, 4) for k, v in eb_p95.items()},
        "lru_p95_by_dataset": {k: round(v, 4) for k, v in lru_p95.items()},
        "p95_delta_mean": round(statistics.mean(deltas), 4) if deltas else None,
        "p95_delta_ci_lower": lower,
        "p95_delta_ci_upper": upper,
        "datasets_where_evict_b_lower": direction,
        "datasets": sorted(set(eb_p95) | set(lru_p95)),
    }


def verify_completeness(root: Path) -> dict[str, Any]:
    missing = []
    for arm in ("arm-evict-b", "arm-lru"):
        arm_root = root / arm
        if not arm_root.exists():
            missing.append(f"{arm}: missing")
            continue
        if not any(arm_root.rglob("request_results.jsonl")):
            missing.append(f"{arm}: no request_results.jsonl")
        if not (arm_root / "arm_metrics.json").exists():
            missing.append(f"{arm}: no arm_metrics.json")
    return {"complete": not missing, "missing": missing}


def mainline_chain_evidence(root: Path) -> dict[str, Any]:
    evict_b = root / "arm-evict-b"
    # Look for kv-core artifacts (A/lookup) and prefetch receipts (B).
    a_tickets = len(list(evict_b.rglob("kv_core_prefetch_tickets.jsonl")))
    lookup_decisions = len(list(evict_b.rglob("kv_core_policy_decisions.jsonl")))
    evict_receipts = len(list(evict_b.rglob("runtime_command_receipts.jsonl")))
    prefetch_b = 0
    for path in evict_b.rglob("runtime_command_receipts.jsonl"):
        prefetch_b += sum(
            1 for row in load_jsonl(path) if as_str(row.get("action")) == "prefetch"
        )
    return {
        "prefetch_a_ticket_files": a_tickets,
        "lookup_decision_files": lookup_decisions,
        "evict_receipt_files": evict_receipts,
        "prefetch_b_receipts": prefetch_b,
    }


def render_report(root: Path, result: dict[str, Any]) -> str:
    lines = [
        "# E11 Target 验收报告",
        "",
        f"- 结果根目录: `{root}`",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## 结论",
        "",
        f"**{result['conclusion']['tier']}**: {result['conclusion']['message']}",
        "",
        "## 1. 运行完整性",
        "",
        f"- 完整: {result['completeness']['complete']}；缺失: `{result['completeness']['missing']}`",
        f"- 崩溃/异常日志: `{result['crashes'] or '无'}`",
        f"- 请求成功率: evict-B `{result['success']['arm-evict-b']}`, LRU `{result['success']['arm-lru']}`",
        "",
        "## 2. evict-B 执行证据",
        "",
        f"- evict 尝试/完成/not_found: `{result['evict']['arm-evict-b']}`",
        f"- 有效驱逐率: `{result['evict']['arm-evict-b'].get('effective_eviction_rate')}`",
        f"- not_found 率: `{result['evict']['arm-evict-b'].get('not_found_rate')}`",
        f"- completed tier 分布: `{result['completed_tiers']['arm-evict-b']}`",
        f"- LRU 臂 evict: `{result['evict']['arm-lru']}`（原生驱逐不可观测）",
        "",
        "## 3. TTFT（主指标，warmup 已排除）",
        "",
        "| dataset | evict-B p95 | LRU p95 | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    paired = result["ttft"]
    for ds in paired["datasets"]:
        lines.append(
            f"| {ds} | {paired['evict_b_p95_by_dataset'].get(ds)} | "
            f"{paired['lru_p95_by_dataset'].get(ds)} | "
            f"{round((paired['evict_b_p95_by_dataset'].get(ds) or 0) - (paired['lru_p95_by_dataset'].get(ds) or 0), 4)} |"
        )
    lines += [
        "",
        f"- p95 delta mean: `{paired['p95_delta_mean']}`",
        f"- p95 delta bootstrap CI: `{paired['p95_delta_ci_lower']} .. {paired['p95_delta_ci_upper']}`",
        f"- evict-B p95 更低的 dataset: `{paired['datasets_where_evict_b_lower']}`",
        "",
        "## 4. 驱逐质量",
        "",
        f"- evict-B 坏驱逐率(30s): `{result['bad_eviction']['arm-evict-b']}`（LRU 侧不可观测，不强行比较）",
        "",
        "## 5. 证据与主线链",
        "",
        f"- 主线链: `{result['chain']}`（legacy 模式无 KV-Core A tickets 为预期）",
        f"- 证据完整性: `{result['completeness']}`",
        "",
        "## 6. 判定依据",
        "",
        f"- `{result['gates']}`",
        "",
    ]
    return "\n".join(lines)


def decide_conclusion(result: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    ev = result["evict"]["arm-evict-b"]
    completed = ev["evict_completed"]
    not_found_rate = ev["not_found_rate"]
    paired = result["ttft"]
    ci_upper = paired["p95_delta_ci_upper"]
    delta = paired["p95_delta_mean"]
    direction = paired["datasets_where_evict_b_lower"]
    datasets = paired["datasets"]

    gates = {
        "completed_receipts": completed > 0,
        "completed_per_dataset": len(result["completed_tiers"]["arm-evict-b"]) == len(datasets),
        "no_crash": not result["crashes"],
        "ttft_p95_evict_b_lower": bool(direction) and len(direction) == len(datasets) if datasets else False,
        "delta_ci_upper_below_zero": ci_upper is not None and ci_upper < 0,
        "delta_at_least_5pct": delta is not None and datasets and delta / max(1e-9, sum(paired["lru_p95_by_dataset"].values()) / len(datasets)) <= -0.05,
        "not_found_not_excessive": not_found_rate is not None and not_found_rate < 0.8,
        "bad_eviction_low": (result["bad_eviction"]["arm-evict-b"].get("rate") or 0.0) <= 0.10,
    }

    if gates["completed_receipts"] and gates["ttft_p95_evict_b_lower"] and gates["delta_ci_upper_below_zero"]:
        tier = "① 真机优于 LRU（可声称）"
        msg = "有 completed receipts，TTFT p95 evict-B < LRU 且重复方向一致、配对 CI 上界 < 0。"
    elif gates["completed_receipts"] and not gates["not_found_not_excessive"]:
        tier = "③ inconclusive"
        msg = "链路执行了，但 not_found 过高，有效驱逐不足，不能形成稳定性能结论。"
    elif gates["completed_receipts"]:
        tier = "② 已接入执行，优势未证实"
        msg = "evict-B 已真实接入并执行，但 TTFT p95 优势未达到统计可解释（相近或 CI 跨 0）。"
    else:
        tier = "③ inconclusive"
        msg = "缺少 completed receipts 或有效样本不足，需要检查容量/存在性校验后重跑。"
    return tier, msg, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root

    result: dict[str, Any] = {}
    result["crashes"] = crash_check(root)
    result["completeness"] = verify_completeness(root)
    result["success"] = {
        "arm-evict-b": request_success(root / "arm-evict-b"),
        "arm-lru": request_success(root / "arm-lru"),
    }
    result["evict"] = {
        "arm-evict-b": evict_metrics(root / "arm-evict-b"),
        "arm-lru": evict_metrics(root / "arm-lru"),
    }
    result["completed_tiers"] = {
        "arm-evict-b": collect_completed_evicts(root / "arm-evict-b"),
        "arm-lru": collect_completed_evicts(root / "arm-lru"),
    }
    result["bad_eviction"] = {
        "arm-evict-b": bad_eviction(root / "arm-evict-b"),
        "arm-lru": bad_eviction(root / "arm-lru"),
    }
    result["ttft"] = paired_ttft(
        collect_ttft(root / "arm-evict-b"),
        collect_ttft(root / "arm-lru"),
    )
    result["chain"] = mainline_chain_evidence(root)
    result["conclusion"] = {}
    result["gates"] = {}
    tier, msg, gates = decide_conclusion(result)
    result["conclusion"] = {"tier": tier, "message": msg}
    result["gates"] = gates

    output = args.output or (root / "e11_target_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(root, result), encoding="utf-8")
    print(json.dumps({
        "tier": tier,
        "message": msg,
        "completed": result["evict"]["arm-evict-b"]["evict_completed"],
        "not_found_rate": result["evict"]["arm-evict-b"]["not_found_rate"],
        "ttft_p95_delta_ci": [result["ttft"]["p95_delta_ci_lower"], result["ttft"]["p95_delta_ci_upper"]],
        "gates": gates,
        "report": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
