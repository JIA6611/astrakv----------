"""Compare native-load vs forced-recompute arms from a KV equivalence run.

The equivalence suite materializes three request roles for one exact prefix:
seed (initial write), loaded (native SSD->GPU load), recompute (scheduler
declines the load and recomputes the prefix).  This comparator reads the run's
``request_results.jsonl`` (split by ``case``) and reports, per arm, TTFT
P50/P95 and CPU/GPU memory peaks, plus the load-vs-recompute deltas.  It also
reads ``kv_equivalence.json`` when present for the output-equivalence verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-load-vs-recompute-v1"
ROLES = ("seed", "loaded", "recompute")


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, int(round(len(sorted_values) * percentile / 100.0 - 0.5))))
    return sorted_values[index]


def _row_role(row: dict[str, Any]) -> str | None:
    case = str(row.get("case") or "")
    for role in ROLES:
        if case == f"kv_equivalence_{role}":
            return role
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def compare(run_dir: Path, state_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "request_results.jsonl"
    if not results_path.is_file():
        raise SystemExit(f"missing request_results.jsonl under {run_dir}")
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = _row_role(row)
        if role is not None:
            by_role[role].append(row)

    arms: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        rows = by_role.get(role, [])
        ok = [row for row in rows if str(row.get("status") or "") == "ok"]
        ttfts = sorted(
            float(row["ttft_ms"]) for row in ok
            if row.get("ttft_ms") not in (None, "")
        )
        gpu_after = [
            float(row["gpu_memory_mb_after"]) for row in ok
            if row.get("gpu_memory_mb_after") not in (None, "")
        ]
        cpu_after = [
            float(row["cpu_memory_mb_after"]) for row in ok
            if row.get("cpu_memory_mb_after") not in (None, "")
        ]
        arms[role] = {
            "request_count": len(rows),
            "ok_count": len(ok),
            "ttft_p50_ms": _percentile(ttfts, 50),
            "ttft_p95_ms": _percentile(ttfts, 95),
            "gpu_memory_mb_peak": max(gpu_after) if gpu_after else None,
            "cpu_memory_mb_peak": max(cpu_after) if cpu_after else None,
        }

    def delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
        av = a.get(key)
        bv = b.get(key)
        if av in (None, "") or bv in (None, "") or not bv:
            return None
        return (float(av) - float(bv)) / float(bv) * 100.0

    loaded = arms.get("loaded", {})
    recompute = arms.get("recompute", {})
    return {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "arms": arms,
        "comparison": {
            "ttft_p50_delta_percent": delta(loaded, recompute, "ttft_p50_ms"),
            "ttft_p95_delta_percent": delta(loaded, recompute, "ttft_p95_ms"),
            "gpu_memory_mb_peak_delta_percent": delta(loaded, recompute, "gpu_memory_mb_peak"),
            "cpu_memory_mb_peak_delta_percent": delta(loaded, recompute, "cpu_memory_mb_peak"),
            "note": (
                "negative delta = load is faster / lower memory than recompute; "
                "loaded arm = native SSD->GPU load, recompute arm = forced recompute"
            ),
        },
        "equivalence": _load_json(run_dir / "kv_equivalence.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Equivalence suite run dir (contains request_results.jsonl).")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    state_dir = Path(args.state_dir) if args.state_dir else run_dir / "state"
    summary = compare(run_dir, state_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
