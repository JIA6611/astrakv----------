"""Validate an externally supplied runtime-eviction workload JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.runtime_workload import (  # noqa: E402
    WorkloadContractError,
    load_runtime_workload_jsonl,
)


def main() -> int:
    args = parse_args()
    try:
        rows = load_runtime_workload_jsonl(args.workload_jsonl)
    except WorkloadContractError as exc:
        print(f"Runtime workload validation failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "workload_jsonl": args.workload_jsonl,
        "row_count": len(rows),
        "arrival_index_min": rows[0].arrival_index,
        "arrival_index_max": rows[-1].arrival_index,
        "reuse_bucket_counts": dict(sorted(Counter(row.reuse_bucket for row in rows).items())),
        "prefix_count": len({row.prefix_id for row in rows}),
        "cache_key_count": len({row.cache_key for row in rows if row.cache_key}),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--output", default="", help="Optional validation summary JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
