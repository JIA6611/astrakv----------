"""Capture and compare LMCache SSD-store artifacts without inferring cache hits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def snapshot_store(store_path: str | Path, *, label: str, run_id: str) -> dict[str, Any]:
    store = Path(store_path).resolve()
    if not store.is_dir():
        raise ValueError(f"LMCache SSD store does not exist: {store}")
    file_count = 0
    pt_file_count = 0
    total_bytes = 0
    for root, _, filenames in os.walk(store):
        for filename in filenames:
            path = Path(root) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            file_count += 1
            total_bytes += size
            if path.suffix == ".pt":
                pt_file_count += 1
    return {
        "schema": "astrakv-lmcache-ssd-store-snapshot-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "label": label,
        "store_path": str(store),
        "mount": mount_record(store),
        "file_count": file_count,
        "pt_file_count": pt_file_count,
        "total_bytes": total_bytes,
        "claim_boundary": "Store files are disk artifacts; they do not prove cache hit, request-to-object mapping, or eviction success.",
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for field in ("schema", "run_id", "store_path"):
        if before.get(field) != after.get(field):
            raise ValueError(f"snapshot mismatch for {field}")
    file_delta = int(after["file_count"]) - int(before["file_count"])
    pt_delta = int(after["pt_file_count"]) - int(before["pt_file_count"])
    byte_delta = int(after["total_bytes"]) - int(before["total_bytes"])
    return {
        "schema": "astrakv-lmcache-ssd-store-comparison-v1",
        "run_id": after["run_id"],
        "store_path": after["store_path"],
        "before_snapshot": before,
        "after_snapshot": after,
        "file_count_delta": file_delta,
        "pt_file_count_delta": pt_delta,
        "byte_delta": byte_delta,
        "disk_artifact_growth_observed": file_delta > 0 or byte_delta > 0,
        "claim_boundary": "Observed store growth does not prove cache hit, request-to-object mapping, or eviction success.",
    }


def mount_record(path: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "-T", str(path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except OSError:
        return {"source": "", "fstype": "", "target": ""}
    fields = result.stdout.strip().split()
    if len(fields) < 3:
        return {"source": "", "fstype": "", "target": ""}
    return {"source": fields[0], "fstype": fields[1], "target": fields[2]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--before", help="Optional before snapshot to compare with this capture.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = snapshot_store(args.store_path, label=args.label, run_id=args.run_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"LMCache SSD snapshot written to {output}")
    if args.before:
        before = json.loads(Path(args.before).read_text(encoding="utf-8"))
        comparison_path = output.with_name("lmcache_ssd_store_comparison.json")
        comparison_path.write_text(
            json.dumps(compare_snapshots(before, snapshot), indent=2) + "\n", encoding="utf-8"
        )
        print(f"LMCache SSD comparison written to {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
