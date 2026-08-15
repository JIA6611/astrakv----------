"""Evaluate an E11 evict-B vs native-LRU run against the target gates.

The comparison is intentionally role-isolated.  By default it compares the
``baseline`` role in each arm, which keeps Prefetch-B disabled in both arms
and makes evict policy the only changed variable.  Evidence is grouped by
``repeat/dataset`` and request keys are paired across arms before computing
TTFT p95 or a bootstrap confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ARMS = ("arm-evict-b", "arm-lru")
CRASH_NEEDLES = (
    "FileNotFoundError",
    "EngineDeadError",
    "OutOfMemoryError",
    "CUDA out of memory",
    "Engine core initialization failed",
)


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_str(value: Any) -> str:
    return "" if value is None else str(value)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_run_dirs(arm_root: Path, role: str) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for run_dir in sorted(arm_root.glob(f"rep-*/*/{role}")):
        if not run_dir.is_dir():
            continue
        repeat = run_dir.parent.parent.name
        dataset = run_dir.parent.name
        selected[f"{repeat}/{dataset}"] = run_dir
    return selected


def artifact_path(run_dir: Path, name: str) -> Path | None:
    """Prefer the exported run artifact and use state only as a fallback.

    The ablation copies state artifacts into the run directory.  Recursively
    reading both locations double-counts every command receipt, which was the
    source of inflated E11 completed/not_found counts in earlier reports.
    """

    exported = run_dir / name
    if exported.is_file():
        return exported
    state = run_dir.parent / f"{run_dir.name}-state" / name
    return state if state.is_file() else None


def receipt_identity(row: dict[str, Any]) -> str:
    explicit = as_str(row.get("receipt_id"))
    if explicit:
        return explicit
    fields = (
        as_str(row.get("command_id")),
        as_str(row.get("action")),
        as_str(row.get("status")),
        as_str(row.get("backend_object_id") or row.get("object_key")),
        as_str(row.get("timestamp_ns")),
    )
    if any(fields):
        return "|".join(fields)
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def unique_receipts(run_dir: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(artifact_path(run_dir, "runtime_command_receipts.jsonl")):
        unique.setdefault(receipt_identity(row), row)
    return list(unique.values())


def request_pair_key(row: dict[str, Any]) -> str:
    for field in ("sample_id", "prefetch_pair_id", "workload_case", "case", "request_id"):
        value = as_str(row.get(field))
        if value:
            return value
    return ""


def request_index(run_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in load_jsonl(artifact_path(run_dir, "request_results.jsonl")):
        if as_str(row.get("status")) != "ok" or as_float(row.get("ttft_ms")) <= 0:
            continue
        key = request_pair_key(row)
        if not key:
            continue
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
    return indexed, duplicates


def build_pairing(root: Path, role: str) -> tuple[dict[str, Any], dict[str, list[tuple[float, float]]]]:
    evict_dirs = selected_run_dirs(root / "arm-evict-b", role)
    lru_dirs = selected_run_dirs(root / "arm-lru", role)
    all_cells = sorted(set(evict_dirs) | set(lru_dirs))
    errors: list[str] = []
    cells: list[dict[str, Any]] = []
    pairs_by_cell: dict[str, list[tuple[float, float]]] = {}

    for cell in all_cells:
        evict_dir = evict_dirs.get(cell)
        lru_dir = lru_dirs.get(cell)
        if evict_dir is None or lru_dir is None:
            errors.append(f"cell_missing:{cell}")
            continue
        evict_rows, evict_duplicates = request_index(evict_dir)
        lru_rows, lru_duplicates = request_index(lru_dir)
        evict_keys, lru_keys = set(evict_rows), set(lru_rows)
        common = sorted(evict_keys & lru_keys)
        workload_evict = sha256_file(artifact_path(evict_dir, "workload_source.jsonl"))
        workload_lru = sha256_file(artifact_path(lru_dir, "workload_source.jsonl"))
        workload_match = bool(workload_evict and workload_evict == workload_lru)
        keys_match = bool(common) and evict_keys == lru_keys
        if evict_duplicates:
            errors.append(f"duplicate_pair_keys:arm-evict-b:{cell}")
        if lru_duplicates:
            errors.append(f"duplicate_pair_keys:arm-lru:{cell}")
        if not workload_match:
            errors.append(f"workload_hash_mismatch:{cell}")
        if not keys_match:
            errors.append(f"request_pair_mismatch:{cell}")
        pairs = [
            (as_float(lru_rows[key].get("ttft_ms")), as_float(evict_rows[key].get("ttft_ms")))
            for key in common
        ]
        pairs_by_cell[cell] = pairs
        cells.append({
            "cell": cell,
            "arm_evict_b_count": len(evict_rows),
            "arm_lru_count": len(lru_rows),
            "paired_count": len(pairs),
            "workload_sha256": workload_evict if workload_match else "",
            "workload_match": workload_match,
            "request_keys_match": keys_match,
            "eligible": workload_match and keys_match and not evict_duplicates and not lru_duplicates,
        })

    record = {
        "schema": "astrakv-e11-paired-run-manifest-v1",
        "role": role,
        "eligible": bool(cells) and not errors and all(cell["eligible"] for cell in cells),
        "errors": list(dict.fromkeys(errors)),
        "cells": cells,
    }
    return record, pairs_by_cell


def paired_p95(pairs_by_cell: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    deltas: list[float] = []
    usable: dict[str, list[tuple[float, float]]] = {}
    for cell, pairs in sorted(pairs_by_cell.items()):
        if not pairs:
            continue
        lru_p95 = percentile((pair[0] for pair in pairs), 0.95)
        evict_p95 = percentile((pair[1] for pair in pairs), 0.95)
        if lru_p95 is None or evict_p95 is None or lru_p95 <= 0:
            continue
        delta = (evict_p95 - lru_p95) / lru_p95 * 100.0
        usable[cell] = pairs
        deltas.append(delta)
        cells.append({
            "cell": cell,
            "paired_count": len(pairs),
            "evict_b_p95_ms": round(evict_p95, 4),
            "lru_p95_ms": round(lru_p95, 4),
            "delta_percent": round(delta, 4),
            "evict_b_lower": evict_p95 < lru_p95,
        })

    lower, upper = bootstrap_p95_ci(usable) if usable else (None, None)
    return {
        "cells": cells,
        "cell_count": len(cells),
        "p95_delta_mean_percent": round(statistics.mean(deltas), 4) if deltas else None,
        "p95_delta_ci_lower_percent": lower,
        "p95_delta_ci_upper_percent": upper,
        "cells_where_evict_b_lower": [cell["cell"] for cell in cells if cell["evict_b_lower"]],
    }


def bootstrap_p95_ci(
    pairs_by_cell: dict[str, list[tuple[float, float]]], *, samples: int = 2000, seed: int = 0,
) -> tuple[float | None, float | None]:
    if not pairs_by_cell:
        return None, None
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        cell_deltas: list[float] = []
        for pairs in pairs_by_cell.values():
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            lru_p95 = percentile((pair[0] for pair in sampled), 0.95)
            evict_p95 = percentile((pair[1] for pair in sampled), 0.95)
            if lru_p95 is not None and evict_p95 is not None and lru_p95 > 0:
                cell_deltas.append((evict_p95 - lru_p95) / lru_p95 * 100.0)
        if cell_deltas:
            boot.append(statistics.mean(cell_deltas))
    if not boot:
        return None, None
    return (
        round(percentile(boot, 0.025) or 0.0, 4),
        round(percentile(boot, 0.975) or 0.0, 4),
    )


def receipts_by_cell(arm_root: Path, role: str) -> dict[str, list[dict[str, Any]]]:
    return {
        cell: unique_receipts(run_dir)
        for cell, run_dir in selected_run_dirs(arm_root, role).items()
    }


def evict_metrics(arm_root: Path, role: str) -> dict[str, Any]:
    cells = receipts_by_cell(arm_root, role)
    attempts = completed = not_found = 0
    per_cell: dict[str, dict[str, int]] = {}
    for cell, rows in cells.items():
        evicts = [row for row in rows if as_str(row.get("action")) == "evict"]
        cell_completed = sum(as_str(row.get("status")) == "completed" for row in evicts)
        cell_not_found = sum(as_str(row.get("status")) == "not_found" for row in evicts)
        attempts += len(evicts)
        completed += cell_completed
        not_found += cell_not_found
        per_cell[cell] = {
            "attempts": len(evicts),
            "completed": cell_completed,
            "not_found": cell_not_found,
        }
    return {
        "evict_attempts": attempts,
        "evict_completed": completed,
        "evict_not_found": not_found,
        "effective_eviction_rate": round(completed / attempts, 4) if attempts else None,
        "not_found_rate": round(not_found / attempts, 4) if attempts else None,
        "per_cell": per_cell,
    }


def completed_tiers(arm_root: Path, role: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for cell, rows in receipts_by_cell(arm_root, role).items():
        result[cell] = [
            f"{as_str(row.get('tier_before'))}->{as_str(row.get('tier_after'))}"
            for row in rows
            if as_str(row.get("action")) == "evict" and as_str(row.get("status")) == "completed"
        ]
    return result


def bad_eviction(arm_root: Path, role: str, window_ms: int = 30_000) -> dict[str, Any]:
    evicted = 0
    reaccessed = 0
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, run_dir in selected_run_dirs(arm_root, role).items():
        evicted_at: dict[str, int] = {}
        for row in unique_receipts(run_dir):
            if as_str(row.get("action")) != "evict" or as_str(row.get("status")) != "completed":
                continue
            key = as_str(row.get("backend_object_id") or row.get("object_key"))
            if key:
                evicted_at[key] = max(evicted_at.get(key, 0), as_int(row.get("timestamp_ns")))
        cell_reaccessed: set[str] = set()
        for row in load_jsonl(artifact_path(run_dir, "runtime_events_raw.jsonl")):
            if as_str(row.get("action")) not in {"cache_hit", "cache_load"}:
                continue
            key = as_str(row.get("backend_object_id") or row.get("object_key"))
            timestamp = as_int(row.get("timestamp_ns"))
            evict_timestamp = evicted_at.get(key)
            if key and evict_timestamp is not None and evict_timestamp < timestamp:
                if timestamp - evict_timestamp <= window_ms * 1_000_000:
                    cell_reaccessed.add(key)
        evicted += len(evicted_at)
        reaccessed += len(cell_reaccessed)
        per_cell[cell] = {
            "evicted_keys": len(evicted_at),
            "reaccessed_within_window": len(cell_reaccessed),
            "rate": round(len(cell_reaccessed) / len(evicted_at), 4) if evicted_at else None,
        }
    return {
        "evicted_keys": evicted,
        "reaccessed_within_window": reaccessed,
        "rate": round(reaccessed / evicted, 4) if evicted else None,
        "window_ms": window_ms,
        "per_cell": per_cell,
    }


def request_success(arm_root: Path, role: str) -> dict[str, Any]:
    ok = failed = 0
    for run_dir in selected_run_dirs(arm_root, role).values():
        for row in load_jsonl(artifact_path(run_dir, "request_results.jsonl")):
            if as_str(row.get("status")) == "ok":
                ok += 1
            else:
                failed += 1
    total = ok + failed
    return {"ok": ok, "failed": failed, "success_rate": round(ok / total, 4) if total else None}


def crash_check(root: Path, role: str) -> list[str]:
    problems: list[str] = []
    for arm in ARMS:
        for cell, run_dir in selected_run_dirs(root / arm, role).items():
            log = run_dir.parent / f"{role}-server.log"
            if not log.is_file():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            needle = _fatal_log_needle(text)
            if needle is not None:
                problems.append(f"{arm}/{cell}/{log.name}: {needle}")
    return problems


def _fatal_log_needle(text: str) -> str | None:
    """Return a fatal marker, excluding vLLM's SIGTERM shutdown artifact.

    vLLM 0.23 may log ``EngineDeadError`` after the runner has deliberately
    sent SIGTERM and entered ``[shutdown] ... mode=abort``.  The measured
    requests are already complete in that sequence, so treating it as a
    runtime crash rejects valid runs.  An EngineDeadError without that local
    causal prelude remains fatal.
    """

    lines = text.splitlines()
    for needle in CRASH_NEEDLES:
        for index, line in enumerate(lines):
            if needle not in line:
                continue
            if needle == "EngineDeadError":
                prelude = lines[max(0, index - 80):index + 1]
                signal_seen = any("trigger received signal=SIGTERM" in item for item in prelude)
                shutdown_seen = any("[shutdown]" in item for item in prelude)
                if signal_seen and shutdown_seen:
                    continue
            return needle
    return None


def verify_completeness(root: Path, role: str, pairing: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    for arm in ARMS:
        arm_root = root / arm
        if not arm_root.is_dir():
            missing.append(f"{arm}: missing")
            continue
        if not selected_run_dirs(arm_root, role):
            missing.append(f"{arm}: no {role} request runs")
        if not (arm_root / "arm_metrics.json").is_file():
            missing.append(f"{arm}: no arm_metrics.json")
    if not pairing.get("eligible"):
        missing.append("cross-arm pairing ineligible")
    return {"complete": not missing, "missing": missing}


def mainline_chain_evidence(root: Path, role: str) -> dict[str, Any]:
    run_dirs = selected_run_dirs(root / "arm-evict-b", role).values()
    a_tickets = lookup_decisions = receipt_files = prefetch_b = 0
    for run_dir in run_dirs:
        if artifact_path(run_dir, "kv_core_prefetch_tickets.jsonl") is not None:
            a_tickets += 1
        if artifact_path(run_dir, "kv_core_policy_decisions.jsonl") is not None:
            lookup_decisions += 1
        if artifact_path(run_dir, "runtime_command_receipts.jsonl") is not None:
            receipt_files += 1
        prefetch_b += sum(as_str(row.get("action")) == "prefetch" for row in unique_receipts(run_dir))
    return {
        "prefetch_a_ticket_files": a_tickets,
        "lookup_decision_files": lookup_decisions,
        "evict_receipt_files": receipt_files,
        "prefetch_b_receipts": prefetch_b,
    }


def decide_conclusion(result: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    evict = result["evict"]["arm-evict-b"]
    ttft = result["ttft"]
    cells = ttft["cells"]
    completed_cells = {
        cell for cell, values in result["completed_tiers"]["arm-evict-b"].items() if values
    }
    not_found_rate = evict.get("not_found_rate")
    delta = ttft.get("p95_delta_mean_percent")
    ci_upper = ttft.get("p95_delta_ci_upper_percent")
    bad_rate = result["bad_eviction"]["arm-evict-b"].get("rate")
    gates = {
        "pairing_eligible": result["pairing"].get("eligible") is True,
        "complete_artifacts": result["completeness"].get("complete") is True,
        "completed_receipts": evict.get("evict_completed", 0) > 0,
        "completed_per_cell": bool(cells) and len(completed_cells) == len(cells),
        "no_crash": not result["crashes"],
        "request_success": all(
            result["success"][arm].get("success_rate") == 1.0 for arm in ARMS
        ),
        "ttft_p95_evict_b_lower": bool(cells) and all(cell["evict_b_lower"] for cell in cells),
        "delta_ci_upper_below_zero": ci_upper is not None and ci_upper < 0,
        "delta_at_least_5pct": delta is not None and delta <= -5.0,
        "not_found_not_excessive": not_found_rate is not None and not_found_rate < 0.8,
        "bad_eviction_low": bad_rate is not None and bad_rate <= 0.10,
    }
    if all(gates.values()):
        return (
            "① 真机优于 LRU（可声称）",
            "跨臂严格配对、每个 repeat/dataset 均有 completed evict，且 TTFT p95 改善的配对 CI 上界小于 0。",
            gates,
        )
    if gates["completed_receipts"] and not gates["not_found_not_excessive"]:
        return "③ inconclusive", "链路执行了，但 not_found 过高，有效驱逐不足。", gates
    if gates["completed_receipts"] and not gates["pairing_eligible"]:
        return "③ inconclusive", "链路执行了，但跨臂 workload/request 配对不合格。", gates
    if gates["completed_receipts"]:
        return "② 已接入执行，优势未证实", "evict-B 已真实执行，但性能、稳定性或坏驱逐门禁尚未同时通过。", gates
    return "③ inconclusive", "缺少去重后的 completed evict receipts 或有效样本。", gates


def render_report(root: Path, role: str, result: dict[str, Any]) -> str:
    lines = [
        "# E11 Target 验收报告",
        "",
        f"- 结果根目录: `{root}`",
        f"- 对比角色: `{role}`（两臂相同，隔离 evict 策略）",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## 结论",
        "",
        f"**{result['conclusion']['tier']}**: {result['conclusion']['message']}",
        "",
        "## 1. 运行与配对",
        "",
        f"- 完整性: `{result['completeness']}`",
        f"- 跨臂配对: eligible=`{result['pairing']['eligible']}` errors=`{result['pairing']['errors']}`",
        f"- 崩溃/异常: `{result['crashes'] or '无'}`",
        f"- 请求成功率: evict-B `{result['success']['arm-evict-b']}`, LRU `{result['success']['arm-lru']}`",
        "",
        "## 2. evict-B 执行证据",
        "",
        f"- evict-B: `{result['evict']['arm-evict-b']}`",
        f"- LRU receipt: `{result['evict']['arm-lru']}`（原生 LRU victim 若未插桩则可能不可观测）",
        f"- completed tier: `{result['completed_tiers']['arm-evict-b']}`",
        "",
        "## 3. TTFT p95（按 repeat/dataset 配对）",
        "",
        "| cell | pairs | evict-B p95 (ms) | LRU p95 (ms) | delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in result["ttft"]["cells"]:
        lines.append(
            f"| {cell['cell']} | {cell['paired_count']} | {cell['evict_b_p95_ms']} | "
            f"{cell['lru_p95_ms']} | {cell['delta_percent']}% |"
        )
    lines += [
        "",
        f"- mean p95 delta: `{result['ttft']['p95_delta_mean_percent']}%`",
        f"- paired bootstrap CI: `[{result['ttft']['p95_delta_ci_lower_percent']}, "
        f"{result['ttft']['p95_delta_ci_upper_percent']}]%`",
        "",
        "## 4. 驱逐质量",
        "",
        f"- evict-B 30s 坏驱逐: `{result['bad_eviction']['arm-evict-b']}`",
        f"- LRU 30s 坏驱逐: `{result['bad_eviction']['arm-lru']}`",
        "",
        "## 5. 判定门",
        "",
        f"- `{result['gates']}`",
        "",
        "## 6. 主线证据",
        "",
        f"- `{result['chain']}`",
        "",
    ]
    return "\n".join(lines)


def evaluate(root: Path, role: str) -> dict[str, Any]:
    pairing, pairs_by_cell = build_pairing(root, role)
    result: dict[str, Any] = {
        "pairing": pairing,
        "crashes": crash_check(root, role),
        "success": {arm: request_success(root / arm, role) for arm in ARMS},
        "evict": {arm: evict_metrics(root / arm, role) for arm in ARMS},
        "completed_tiers": {arm: completed_tiers(root / arm, role) for arm in ARMS},
        "bad_eviction": {arm: bad_eviction(root / arm, role) for arm in ARMS},
        "ttft": paired_p95(pairs_by_cell),
        "chain": mainline_chain_evidence(root, role),
    }
    result["completeness"] = verify_completeness(root, role, pairing)
    tier, message, gates = decide_conclusion(result)
    result["conclusion"] = {"tier": tier, "message": message}
    result["gates"] = gates
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "variant"), default="baseline")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--paired-manifest-output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(args.root, args.role)
    output = args.output or (args.root / "e11_target_report.md")
    paired_output = args.paired_manifest_output or (args.root / "e11_paired_run_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(args.root, args.role, result), encoding="utf-8")
    paired_output.parent.mkdir(parents=True, exist_ok=True)
    paired_output.write_text(
        json.dumps(result["pairing"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tier": result["conclusion"]["tier"],
        "message": result["conclusion"]["message"],
        "completed": result["evict"]["arm-evict-b"]["evict_completed"],
        "not_found_rate": result["evict"]["arm-evict-b"]["not_found_rate"],
        "ttft_p95_delta_percent": result["ttft"]["p95_delta_mean_percent"],
        "ttft_p95_delta_ci_percent": [
            result["ttft"]["p95_delta_ci_lower_percent"],
            result["ttft"]["p95_delta_ci_upper_percent"],
        ],
        "pairing_eligible": result["pairing"]["eligible"],
        "gates": result["gates"],
        "report": str(output),
        "paired_manifest": str(paired_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
