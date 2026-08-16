"""Evaluate E11 CPU native-policy LRU vs AstraKV-W MVP evidence.

Unlike the legacy E11 evaluator, this script treats LMCache's native CPU
capacity-reclaim events as ground truth.  External action-command receipts are
intentionally ignored because both arms disable that execution architecture.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.reporting.evaluate_e11_target import (
    ARMS,
    artifact_path,
    as_float,
    as_int,
    as_str,
    build_pairing,
    crash_check,
    load_jsonl,
    paired_p95,
    request_success,
    selected_run_dirs,
)


EXPECTED_POLICY = {
    "arm-evict-b": ("astrakv", "astrakv_native_cpu"),
    "arm-lru": ("lru", "lmcache_lru"),
}


def _native_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(artifact_path(run_dir, "native_cache_policy_evictions.jsonl"))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        selection_id = as_str(row.get("selection_id"))
        status = as_str(row.get("status"))
        if selection_id and status in {"selected", "completed"}:
            unique.setdefault((selection_id, status), row)
    return list(unique.values())


def native_eviction_metrics(arm_root: Path, role: str) -> dict[str, Any]:
    selected_total = completed_total = 0
    per_cell: dict[str, dict[str, Any]] = {}
    score_sources: dict[str, int] = {}
    for cell, run_dir in selected_run_dirs(arm_root, role).items():
        rows = _native_rows(run_dir)
        selected = [row for row in rows if as_str(row.get("status")) == "selected"]
        completed = [row for row in rows if as_str(row.get("status")) == "completed"]
        selected_ids = {as_str(row.get("selection_id")) for row in selected}
        completed_ids = {as_str(row.get("selection_id")) for row in completed}
        selected_total += len(selected_ids)
        completed_total += len(completed_ids)
        for row in selected:
            source = as_str(row.get("score_source")) or "unknown"
            score_sources[source] = score_sources.get(source, 0) + 1
        per_cell[cell] = {
            "selected": len(selected_ids),
            "completed": len(completed_ids),
            "completion_rate": round(len(completed_ids) / len(selected_ids), 6) if selected_ids else None,
            "orphan_selected": sorted(selected_ids - completed_ids),
            "orphan_completed": sorted(completed_ids - selected_ids),
            "effective_policies": sorted({as_str(row.get("effective_policy")) for row in rows}),
            "terminal_conditions": sorted({
                as_str(row.get("terminal_condition")) for row in completed
                if as_str(row.get("terminal_condition"))
            }),
        }
    return {
        "selected": selected_total,
        "completed": completed_total,
        "completion_rate": round(completed_total / selected_total, 6) if selected_total else None,
        "not_found": 0,
        "score_sources": score_sources,
        "per_cell": per_cell,
    }


def installation_evidence(arm: str, arm_root: Path, role: str) -> dict[str, Any]:
    expected_requested, expected_effective = EXPECTED_POLICY[arm]
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, run_dir in selected_run_dirs(arm_root, role).items():
        rows = [
            row for row in load_jsonl(artifact_path(run_dir, "native_policy_installation.jsonl"))
            if as_str(row.get("status")) == "installed"
        ]
        row = rows[-1] if rows else {}
        valid = bool(
            row
            and as_str(row.get("cpu_requested_policy")) == expected_requested
            and as_str(row.get("cpu_effective_policy")) == expected_effective
            and row.get("cpu_same_native_capacity_path") is True
            and row.get("ssd_policy_unchanged") is True
            and "LRU" in as_str(row.get("ssd_effective_policy_class")).upper()
        )
        per_cell[cell] = {
            "valid": valid,
            "cpu_requested_policy": row.get("cpu_requested_policy"),
            "cpu_effective_policy": row.get("cpu_effective_policy"),
            "cpu_delegate_policy_class": row.get("cpu_delegate_policy_class"),
            "ssd_effective_policy_class": row.get("ssd_effective_policy_class"),
            "ssd_policy_unchanged": row.get("ssd_policy_unchanged"),
            "installation_records": len(rows),
        }
    return {
        "expected_requested_policy": expected_requested,
        "expected_effective_policy": expected_effective,
        "valid": bool(per_cell) and all(row["valid"] for row in per_cell.values()),
        "per_cell": per_cell,
    }


def _logical_key(row: dict[str, Any]) -> str:
    signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
    return as_str(signals.get("logical_object_key"))


def bad_eviction(arm_root: Path, role: str, window_ms: int = 30_000) -> dict[str, Any]:
    completed = matchable = reaccessed = 0
    per_cell: dict[str, dict[str, Any]] = {}
    window_ns = window_ms * 1_000_000
    for cell, run_dir in selected_run_dirs(arm_root, role).items():
        evictions: list[tuple[str, int, str, str]] = []
        for row in _native_rows(run_dir):
            if as_str(row.get("status")) != "completed":
                continue
            completed += 1
            logical_key = _logical_key(row)
            if logical_key:
                matchable += 1
                signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
                evictions.append((
                    logical_key,
                    as_int(row.get("timestamp_ns")),
                    as_str(row.get("selection_id")),
                    as_str(signals.get("request_id")),
                ))
        hit_events = []
        for row in load_jsonl(artifact_path(run_dir, "runtime_events_raw.jsonl")):
            if as_str(row.get("action")) in {"cache_hit", "cache_load"} and as_str(row.get("status")) in {
                "completed", "available", "ok", "executed",
            }:
                hit_events.append((
                    as_str(row.get("object_key")),
                    as_int(row.get("timestamp_ns")),
                    as_str(row.get("request_id") or row.get("logical_request_id")),
                ))
        bad_ids: set[str] = set()
        for logical_key, evicted_ns, selection_id, victim_request_id in evictions:
            if any(
                hit_key == logical_key
                and evicted_ns < hit_ns <= evicted_ns + window_ns
                and (not victim_request_id or hit_request_id != victim_request_id)
                for hit_key, hit_ns, hit_request_id in hit_events
            ):
                bad_ids.add(selection_id)
        reaccessed += len(bad_ids)
        per_cell[cell] = {
            "completed": len([row for row in _native_rows(run_dir) if as_str(row.get("status")) == "completed"]),
            "matchable": len(evictions),
            "reaccessed_within_window": len(bad_ids),
            "rate": round(len(bad_ids) / len(evictions), 6) if evictions else None,
        }
    return {
        "completed": completed,
        "matchable": matchable,
        "unmatchable": completed - matchable,
        "reaccessed_within_window": reaccessed,
        "rate": round(reaccessed / matchable, 6) if matchable else None,
        "window_ms": window_ms,
        "per_cell": per_cell,
    }


def paired_request_summary(pairs_by_cell: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    pairs = [pair for cell_pairs in pairs_by_cell.values() for pair in cell_pairs if pair[0] > 0]
    if not pairs:
        return {
            "paired_count": 0,
            "lru_mean_ms": None,
            "astrakv_mean_ms": None,
            "mean_delta_percent": None,
            "median_paired_delta_percent": None,
            "mean_paired_delta_ci_percent": [None, None],
        }
    deltas = [(astrakv - lru) / lru * 100.0 for lru, astrakv in pairs]
    rng = random.Random(0)
    boot = []
    for _ in range(4000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        boot.append(statistics.mean(sample))
    boot.sort()
    low = boot[max(0, int(0.025 * len(boot)) - 1)]
    high = boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    lru_mean = statistics.mean(pair[0] for pair in pairs)
    astrakv_mean = statistics.mean(pair[1] for pair in pairs)
    return {
        "paired_count": len(pairs),
        "lru_mean_ms": round(lru_mean, 4),
        "astrakv_mean_ms": round(astrakv_mean, 4),
        "mean_delta_percent": round((astrakv_mean - lru_mean) / lru_mean * 100.0, 4),
        "median_paired_delta_percent": round(statistics.median(deltas), 4),
        "mean_paired_delta_ci_percent": [round(low, 4), round(high, 4)],
    }


def workload_shape(root: Path, role: str) -> dict[str, Any]:
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, run_dir in selected_run_dirs(root / "arm-evict-b", role).items():
        rows = load_jsonl(artifact_path(run_dir, "request_results.jsonl"))
        ratios = sorted({round(as_float(row.get("reuse_ratio")), 6) for row in rows})
        keys = [as_str(row.get("cache_key") or row.get("prefix_id")) for row in rows]
        counts = {key: keys.count(key) for key in sorted(set(keys)) if key}
        valid = bool(
            len(rows) >= 6
            and 0.0 in ratios
            and any(ratio > 0.0 for ratio in ratios)
            and any(count >= 3 for count in counts.values())
            and any(count == 1 for count in counts.values())
        )
        per_cell[cell] = {
            "valid": valid,
            "request_count": len(rows),
            "distinct_reuse_ratios": ratios,
            "reuse_group_visit_counts": counts,
        }
    return {
        "valid": bool(per_cell) and all(value["valid"] for value in per_cell.values()),
        "per_cell": per_cell,
    }


def victim_sequence_comparison(root: Path, role: str) -> dict[str, Any]:
    astra_dirs = selected_run_dirs(root / "arm-evict-b", role)
    lru_dirs = selected_run_dirs(root / "arm-lru", role)
    per_cell: dict[str, dict[str, Any]] = {}
    for cell in sorted(set(astra_dirs) | set(lru_dirs)):
        astra = [
            as_str(row.get("backend_key_identity"))
            for row in _native_rows(astra_dirs[cell])
            if as_str(row.get("status")) == "selected"
        ] if cell in astra_dirs else []
        lru = [
            as_str(row.get("backend_key_identity"))
            for row in _native_rows(lru_dirs[cell])
            if as_str(row.get("status")) == "selected"
        ] if cell in lru_dirs else []
        first_divergence = next(
            (index for index, pair in enumerate(zip(astra, lru)) if pair[0] != pair[1]),
            None,
        )
        diverged = bool(astra and lru and (len(astra) != len(lru) or first_divergence is not None))
        per_cell[cell] = {
            "diverged": diverged,
            "astrakv_selected": len(astra),
            "lru_selected": len(lru),
            "same_positions": sum(left == right for left, right in zip(astra, lru)),
            "first_divergence_index": first_divergence,
            "same_sequence": astra == lru,
            "same_victim_multiset": sorted(astra) == sorted(lru),
        }
    return {
        "diverged": bool(per_cell) and all(value["diverged"] for value in per_cell.values()),
        "per_cell": per_cell,
    }


def evaluate(root: Path, role: str) -> dict[str, Any]:
    pairing, pairs_by_cell = build_pairing(root, role)
    installations = {
        arm: installation_evidence(arm, root / arm, role) for arm in ARMS
    }
    evictions = {
        arm: native_eviction_metrics(root / arm, role) for arm in ARMS
    }
    bad = {arm: bad_eviction(root / arm, role) for arm in ARMS}
    success = {arm: request_success(root / arm, role) for arm in ARMS}
    crashes = crash_check(root, role)
    p95 = paired_p95(pairs_by_cell)
    paired = paired_request_summary(pairs_by_cell)
    shape = workload_shape(root, role)
    victim_sequences = victim_sequence_comparison(root, role)
    expected_cells = {cell["cell"] for cell in pairing.get("cells", [])}
    completed_per_cell = {
        arm: all(
            evictions[arm]["per_cell"].get(cell, {}).get("completed", 0) > 0
            for cell in expected_cells
        ) if expected_cells else False
        for arm in ARMS
    }
    gates = {
        "pairing_eligible": pairing.get("eligible") is True,
        "policy_installation_valid": all(value["valid"] for value in installations.values()),
        "native_completed_both_arms": all(value["completed"] > 0 for value in evictions.values()),
        "native_completed_every_cell": all(completed_per_cell.values()),
        "native_completion_exact": all(
            value["selected"] > 0 and value["selected"] == value["completed"]
            for value in evictions.values()
        ),
        "no_crash": not crashes,
        "request_success": all(value.get("success_rate") == 1.0 for value in success.values()),
        "workload_reuse_heterogeneous": shape["valid"],
        "victim_sequence_diverged": victim_sequences["diverged"],
        "bad_eviction_fully_matchable": all(value["unmatchable"] == 0 for value in bad.values()),
        "astrakv_bad_eviction_not_worse": (
            bad["arm-evict-b"]["rate"] is not None
            and bad["arm-lru"]["rate"] is not None
            and bad["arm-evict-b"]["rate"] <= bad["arm-lru"]["rate"]
        ),
        "directional_mean_ttft_better": (
            paired["mean_delta_percent"] is not None and paired["mean_delta_percent"] < 0
        ),
        "directional_p95_better": (
            p95["p95_delta_mean_percent"] is not None and p95["p95_delta_mean_percent"] < 0
        ),
        "mean_delta_ci_upper_below_zero": (
            paired["mean_paired_delta_ci_percent"][1] is not None
            and paired["mean_paired_delta_ci_percent"][1] < 0
        ),
        "p95_delta_ci_upper_below_zero": (
            p95["p95_delta_ci_upper_percent"] is not None
            and p95["p95_delta_ci_upper_percent"] < 0
        ),
    }
    correctness_names = (
        "pairing_eligible", "policy_installation_valid", "native_completed_both_arms",
        "native_completed_every_cell", "native_completion_exact", "no_crash", "request_success",
        "workload_reuse_heterogeneous", "victim_sequence_diverged",
        "bad_eviction_fully_matchable",
    )
    correctness_valid = all(gates[name] for name in correctness_names)
    quality_valid = correctness_valid and gates["astrakv_bad_eviction_not_worse"]
    directional = quality_valid and gates["directional_mean_ttft_better"] and gates["directional_p95_better"]
    independent_cell_count = len(pairing.get("cells", []))
    formal_repeat_ready = independent_cell_count >= 3
    claimable = (
        directional
        and formal_repeat_ready
        and gates["mean_delta_ci_upper_below_zero"]
        and gates["p95_delta_ci_upper_below_zero"]
    )
    if claimable:
        conclusion = {
            "tier": "claimable_improvement",
            "message": "原生同层 A/B 有效，AstraKV-W 的平均 TTFT 与 p95 均改善，且配对 bootstrap CI 上界低于 0。",
        }
    elif directional:
        conclusion = {
            "tier": "directional_improvement",
            "message": "原生同层 A/B 正确且当前小样本方向上优于 LRU，但置信区间尚不足以形成正式性能声明。",
        }
    elif correctness_valid:
        conclusion = {
            "tier": "valid_no_improvement",
            "message": "原生同层 A/B 已正确执行，但当前样本未显示 AstraKV-W 同时改善平均 TTFT 和 p95。",
        }
    else:
        conclusion = {
            "tier": "invalid_or_inconclusive",
            "message": "安装、原生驱逐、配对、冷热异质性、victim 序列分叉、请求成功或崩溃门禁未通过，不能解释性能差异。",
        }
    return {
        "schema": "astrakv-e11-native-cpu-mvp-evaluation-v2",
        "root": str(root),
        "role": role,
        "pairing": pairing,
        "workload_shape": shape,
        "victim_sequence_comparison": victim_sequences,
        "installations": installations,
        "native_evictions": evictions,
        "bad_eviction": bad,
        "success": success,
        "crashes": crashes,
        "ttft_p95": p95,
        "paired_ttft": paired,
        "gates": gates,
        "correctness_valid": correctness_valid,
        "quality_valid": quality_valid,
        "independent_cell_count": independent_cell_count,
        "formal_repeat_ready": formal_repeat_ready,
        "directional_improvement": directional,
        "claimable_improvement": claimable,
        "conclusion": conclusion,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# E11 原生 CPU 回收策略 MVP 报告",
        "",
        f"- 结果目录：`{result['root']}`",
        f"- 对比角色：`{result['role']}`（两臂均禁用 Prefetch-B 与旧 evict 扫描）",
        "- arm-evict-b：CPU AstraKV-W；SSD 原生 LRU",
        "- arm-lru：CPU 原生 LRU；SSD 原生 LRU",
        "",
        "## 结论",
        "",
        f"**{result['conclusion']['tier']}**：{result['conclusion']['message']}",
        "",
        "## 正确性与安装证据",
        "",
        f"- correctness_valid：`{result['correctness_valid']}`",
        f"- 配对：eligible=`{result['pairing']['eligible']}`，errors=`{result['pairing']['errors']}`",
        f"- workload 异质性：`{result['workload_shape']}`",
        f"- victim 序列分叉：`{result['victim_sequence_comparison']}`",
        f"- 安装：`{result['installations']}`",
        f"- 原生驱逐：`{result['native_evictions']}`",
        f"- 请求成功：`{result['success']}`",
        f"- 崩溃：`{result['crashes'] or '无'}`",
        "",
        "## 性能",
        "",
        f"- 配对请求 TTFT：`{result['paired_ttft']}`",
        f"- 分 cell p95：`{result['ttft_p95']}`",
        "",
        "| cell | pairs | AstraKV-W p95 (ms) | LRU p95 (ms) | delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in result["ttft_p95"]["cells"]:
        lines.append(
            f"| {cell['cell']} | {cell['paired_count']} | {cell['evict_b_p95_ms']} | "
            f"{cell['lru_p95_ms']} | {cell['delta_percent']}% |"
        )
    lines += [
        "",
        "## 驱逐质量",
        "",
        f"- 30 秒内重访问：`{result['bad_eviction']}`",
        "",
        "## 判定门禁",
        "",
        f"- `{result['gates']}`",
        "",
        "说明：`directional_improvement` 只表示当前小样本方向；只有 `claimable_improvement` 才满足本报告的统计声明门禁。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline",), default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--paired-manifest-output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, args.role)
    report_path = args.output or args.root / "e11_target_report.md"
    json_path = args.json_output or args.root / "e11_native_cpu_result.json"
    pairing_path = args.paired_manifest_output or args.root / "e11_paired_run_manifest.json"
    report_path.write_text(render_report(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pairing_path.write_text(
        json.dumps(result["pairing"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tier": result["conclusion"]["tier"],
        "message": result["conclusion"]["message"],
        "correctness_valid": result["correctness_valid"],
        "directional_improvement": result["directional_improvement"],
        "claimable_improvement": result["claimable_improvement"],
        "native_completed": {
            arm: result["native_evictions"][arm]["completed"] for arm in ARMS
        },
        "mean_ttft_delta_percent": result["paired_ttft"]["mean_delta_percent"],
        "p95_delta_percent": result["ttft_p95"]["p95_delta_mean_percent"],
        "report": str(report_path),
        "json": str(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
