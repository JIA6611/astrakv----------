"""Aggregate evict-B vs LRU arm metrics from DGX run artifacts.

For each arm the suite produces one or more state directories:
  <arm-root>/<dataset>/<role>-state        (receipts, commands, hook events)
  <arm-root>/<dataset>/<role>              (benchmark request_results.jsonl)

This script computes per-arm:
  - receipt action x status counts (evict/drop/prefetch)
  - prefetch-A ticket statuses (kv_core_prefetch_tickets.jsonl, KV-core mode)
  - lookup load-vs-recompute decisions (kv_core_policy_decisions.jsonl)
  - TTFT p50/p95/mean from request_results.jsonl
  - bad-eviction rate: evict-completed keys re-accessed within a window
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
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


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return round(ordered[index], 4)


def event_key(row: dict[str, Any]) -> str:
    return as_str(
        row.get("object_key")
        or row.get("cache_key")
        or row.get("prefix_hash")
        or row.get("backend_object_id")
        or row.get("chunk_id")
    )


def aggregate_run(state_dir: Path, run_dir: Path | None, *, reaccess_window_ms: int) -> dict[str, Any]:
    def load_first(*candidates: Path) -> list[dict[str, Any]]:
        for path in candidates:
            if path.exists():
                return load_jsonl(path)
        return []

    # The ablation copies control-plane artifacts from the *-state dir into the
    # run dir (baseline|variant); prefer the run dir, fall back to state dir.
    receipts = load_first(
        run_dir / "runtime_command_receipts.jsonl" if run_dir is not None else state_dir,
        state_dir / "runtime_command_receipts.jsonl",
    )
    commands = load_first(
        run_dir / "astrakv_runtime_commands.jsonl" if run_dir is not None else state_dir,
        state_dir / "astrakv_runtime_commands.jsonl",
    )
    tickets = load_first(
        run_dir / "kv_core_prefetch_tickets.jsonl" if run_dir is not None else state_dir,
        state_dir / "kv_core_prefetch_tickets.jsonl",
    )
    decisions = load_first(
        run_dir / "kv_core_policy_decisions.jsonl" if run_dir is not None else state_dir,
        state_dir / "kv_core_policy_decisions.jsonl",
    )
    native_evictions = load_first(
        run_dir / "native_cache_policy_evictions.jsonl" if run_dir is not None else state_dir,
        state_dir / "native_cache_policy_evictions.jsonl",
    )
    native_installations = load_first(
        run_dir / "native_policy_installation.jsonl" if run_dir is not None else state_dir,
        state_dir / "native_policy_installation.jsonl",
    )

    receipt_counts: Counter[str] = Counter()
    evict_completed: list[dict[str, Any]] = []
    for row in receipts:
        receipt_counts[f"{as_str(row.get('action'))}:{as_str(row.get('status'))}"] += 1
        if as_str(row.get("action")) == "evict" and as_str(row.get("status")) == "completed":
            evict_completed.append(row)

    native_selected = {
        as_str(row.get("selection_id")): row
        for row in native_evictions
        if as_str(row.get("status")) == "selected" and as_str(row.get("selection_id"))
    }
    native_completed = {
        as_str(row.get("selection_id")): row
        for row in native_evictions
        if as_str(row.get("status")) == "completed" and as_str(row.get("selection_id"))
    }
    if native_evictions:
        evict_completed = list(native_completed.values())

    ticket_counts: Counter[str] = Counter()
    for row in tickets:
        ticket_counts[as_str(row.get("status"))] += 1

    lookup_counts: Counter[str] = Counter()
    for row in decisions:
        lookup_counts[as_str(row.get("action"))] += 1

    cold_scores: list[float] = []
    for row in commands:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if as_str(row.get("action")) == "evict" and meta.get("evict_cold_score") is not None:
            cold_scores.append(as_float(meta.get("evict_cold_score")))
    if native_evictions:
        cold_scores = [
            as_float(row.get("cold_score"))
            for row in native_selected.values()
            if row.get("cold_score") is not None
        ]

    ttft: list[float] = []
    request_results = load_jsonl(run_dir / "request_results.jsonl") if run_dir is not None else []
    for row in request_results:
        value = as_float(row.get("ttft_ms"))
        if value > 0:
            ttft.append(value)

    # Bad eviction: an evict-completed key later re-accessed within the window.
    evicted_at: dict[str, int] = {}
    for row in evict_completed:
        signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
        key = as_str(signals.get("logical_object_key")) or event_key(row)
        ts = as_int(row.get("timestamp_ns"))
        if key:
            evicted_at[key] = max(evicted_at.get(key, 0), ts)
    reaccess_after_evict: set[str] = set()
    if evicted_at:
        for path in (
            run_dir / "runtime_structured_events.jsonl" if run_dir is not None else state_dir,
            state_dir / "runtime_structured_events.jsonl",
            run_dir / "hook_raw.jsonl" if run_dir is not None else state_dir,
            state_dir / "hook_raw.jsonl",
        ):
            for row in load_jsonl(path):
                action = as_str(row.get("action"))
                if action not in {"cache_hit", "cache_load", "load", "hit"}:
                    continue
                key = event_key(row)
                ts = as_int(row.get("timestamp_ns"))
                if not key or not ts:
                    continue
                evict_ts = evicted_at.get(key)
                if evict_ts is not None and evict_ts < ts and (ts - evict_ts) <= reaccess_window_ms * 1_000_000:
                    reaccess_after_evict.add(key)

    return {
        "state_dir": str(state_dir),
        "run_dir": str(run_dir) if run_dir is not None else "",
        "receipts": dict(receipt_counts),
        "evict_completed": len(evict_completed),
        "native_eviction": {
            "selected": len(native_selected),
            "completed": len(native_completed),
            "completion_rate": (
                round(len(native_completed) / len(native_selected), 4)
                if native_selected else None
            ),
        },
        "native_policy_installation": native_installations[-1] if native_installations else None,
        "prefetch_a_tickets": dict(ticket_counts),
        "lookup_actions": dict(lookup_counts),
        "ttft_ms": {
            "count": len(ttft),
            "mean": round(statistics.mean(ttft), 4) if ttft else None,
            "p50": percentile(ttft, 50),
            "p95": percentile(ttft, 95),
        },
        "evict_cold_scores": {
            "count": len(cold_scores),
            "mean": round(statistics.mean(cold_scores), 4) if cold_scores else None,
            "min": round(min(cold_scores), 4) if cold_scores else None,
            "max": round(max(cold_scores), 4) if cold_scores else None,
        },
        "bad_eviction": {
            "evicted_keys": len(evicted_at),
            "reaccessed_within_window": len(reaccess_after_evict),
            "rate": round(len(reaccess_after_evict) / len(evicted_at), 4) if evicted_at else None,
            "window_ms": reaccess_window_ms,
        },
    }


def merge_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    receipts: Counter[str] = Counter()
    tickets: Counter[str] = Counter()
    lookup: Counter[str] = Counter()
    ttft: list[float] = []
    cold: list[float] = []
    evicted_keys = 0
    reaccessed = 0
    evict_completed = 0
    native_selected = 0
    native_completed = 0
    native_installations: list[dict[str, Any]] = []
    for run in runs:
        receipts.update(run["receipts"])
        tickets.update(run["prefetch_a_tickets"])
        lookup.update(run["lookup_actions"])
        evict_completed += run["evict_completed"]
        native_selected += run.get("native_eviction", {}).get("selected", 0)
        native_completed += run.get("native_eviction", {}).get("completed", 0)
        if run.get("native_policy_installation"):
            native_installations.append(run["native_policy_installation"])
        ttft_values = run["ttft_ms"]
        if ttft_values.get("count"):
            # Recompute percentiles from per-run aggregate is approximate; the
            # suite keeps raw per-request rows, so this merge marks best-effort.
            pass
        cold.append(run["evict_cold_scores"]["mean"] if run["evict_cold_scores"].get("count") else 0.0)
        evicted_keys += run["bad_eviction"]["evicted_keys"]
        reaccessed += run["bad_eviction"]["reaccessed_within_window"]
    return {
        "run_count": len(runs),
        "receipts": dict(receipts),
        "evict_completed": evict_completed,
        "native_eviction": {
            "selected": native_selected,
            "completed": native_completed,
            "completion_rate": round(native_completed / native_selected, 4) if native_selected else None,
        },
        "native_policy_installations": native_installations,
        "prefetch_a_tickets": dict(tickets),
        "lookup_actions": dict(lookup),
        "ttft_ms": {"count": len(ttft), "mean": None, "p50": None, "p95": None},
        "evict_cold_scores": {
            "count": sum(1 for run in runs if run["evict_cold_scores"].get("count")),
            "mean": round(statistics.mean(cold), 4) if cold else None,
            "min": None,
            "max": None,
        },
        "bad_eviction": {
            "evicted_keys": evicted_keys,
            "reaccessed_within_window": reaccessed,
            "rate": round(reaccessed / evicted_keys, 4) if evicted_keys else None,
            "window_ms": runs[0]["bad_eviction"]["window_ms"] if runs else None,
        },
    }


def measured_state_dirs(arm_root: Path) -> list[Path]:
    """Return only measured role state directories from the E11 arm layout.

    A recursive search also enters ``*-lmcache-store`` and nested
    ``warmup-state`` directories. Besides being very slow on real artifacts,
    treating the nested warmup directory as a measured state makes
    ``aggregate_run`` attempt to read the directory itself as JSONL.
    """
    return sorted(
        path
        for path in arm_root.glob("rep-*/*/*-state")
        if path.is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, help="Single run state directory")
    parser.add_argument("--run-dir", type=Path, help="Single run benchmark directory (request_results.jsonl)")
    parser.add_argument("--arm-root", type=Path, help="Arm root: glob <root>/*/*-state with sibling run dirs")
    parser.add_argument("--output", type=Path, help="JSON output path")
    parser.add_argument("--reaccess-window-ms", type=int, default=30_000)
    args = parser.parse_args()

    if args.state_dir is not None:
        runs = [aggregate_run(args.state_dir, args.run_dir, reaccess_window_ms=args.reaccess_window_ms)]
        merged = runs[0]
    elif args.arm_root is not None:
        runs = []
        for state_dir in measured_state_dirs(args.arm_root):
            run_dir = state_dir.with_name(state_dir.name[: -len("-state")])
            # Warmup passes share the state dir; only the measured run_dir
            # should contribute request_results to TTFT aggregation.
            measured_run_dir = run_dir if run_dir.exists() else None
            runs.append(aggregate_run(state_dir, measured_run_dir, reaccess_window_ms=args.reaccess_window_ms))
        merged = merge_runs(runs)
        # Recompute TTFT percentiles from all raw request rows across the arm.
        raw_ttft: list[float] = []
        for path in args.arm_root.rglob("request_results.jsonl"):
            if "/warmup-" in path.as_posix() or "\\warmup-" in path.as_posix():
                continue
            for row in load_jsonl(path):
                value = as_float(row.get("ttft_ms"))
                if value > 0:
                    raw_ttft.append(value)
        if raw_ttft:
            merged["ttft_ms"] = {
                "count": len(raw_ttft),
                "mean": round(statistics.mean(raw_ttft), 4),
                "p50": percentile(raw_ttft, 50),
                "p95": percentile(raw_ttft, 95),
            }
    else:
        parser.error("one of --state-dir or --arm-root is required")

    output = args.output or (args.state_dir or args.arm_root) / "arm_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "astrakv-evict-ablation-arm-v1",
        "merged": merged,
        "runs": runs if args.arm_root is not None else [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
