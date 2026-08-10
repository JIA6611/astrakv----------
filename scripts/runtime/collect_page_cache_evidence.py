#!/usr/bin/env python3
"""Collect OS page-cache residency evidence alongside a KV-Core run.

Point ``--path`` at the LMCache disk store (a directory or one backing file).
The collector samples mincore residency of a bounded file prefix every
``--interval-s`` seconds and pairs it with cgroup-v2 ``memory.current`` and
process RSS.  ``--madvise-willneed`` issues an explicit OS prefetch on the
sampled region at start; ``--madvise-dontneed-on-exit`` hints reclamation at
the end.  Output is JSONL (and an optional CSV) suitable for the f5 capacity
figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.os_page_cache_evidence import (  # noqa: E402
    PageCacheEvidenceCollector,
    write_csv,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="LMCache disk store dir or backing file")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--max-mapped-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--madvise-willneed", action="store_true")
    parser.add_argument("--madvise-dontneed-on-exit", action="store_true")
    args = parser.parse_args()

    collector = PageCacheEvidenceCollector(
        path=Path(args.path),
        sample_interval_s=args.interval_s,
        duration_s=args.duration_s,
        max_mapped_bytes=args.max_mapped_bytes,
        madvise_willneed_at_start=args.madvise_willneed,
        madvise_dontneed_on_exit=args.madvise_dontneed_on_exit,
    )
    print(f"Sampling {collector.path}", flush=True)
    samples = collector.run()
    write_jsonl(Path(args.output), samples)
    if args.csv:
        write_csv(Path(args.csv), samples)
    valid = [sample for sample in samples if sample.get("mincore_status") == "valid"]
    if valid:
        fractions = [sample["resident_fraction"] for sample in valid if sample.get("resident_fraction") is not None]
        mean = sum(fractions) / len(fractions) if fractions else None
        summary = {
            "schema": "astrakv-os-page-cache-evidence-summary-v1",
            "sample_count": len(samples),
            "valid_sample_count": len(valid),
            "mean_resident_fraction": mean,
        }
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps({"schema": "astrakv-os-page-cache-evidence-summary-v1", "sample_count": len(samples), "valid_sample_count": 0, "mean_resident_fraction": None, "status": samples[0].get("mincore_status") if samples else "no_samples"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
