"""Materialize a recompute-only arm variant of a canonical KV workload.

Every revisit row (``reuse_bucket != "none"``) receives the test-only
``kv_core_decision_probe.force_recompute`` override so the runtime scheduler
declines the external prefix and natively recomputes it.  Seed rows are left
untouched so the LMCache object is really written to the disk store first.
Request identity (``request_id``/``case``/``arrival_index``) is preserved so
the arm pairs with the load-allowed arms by ``sample_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl  # noqa: E402


SCHEMA = "astrakv-recompute-only-workload-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize(Path(args.source_workload), Path(args.output_dir))
    print(json.dumps(manifest, sort_keys=True))
    return 0


def materialize(source_path: Path, output_dir: Path) -> dict:
    rows = load_runtime_workload_jsonl(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source_path.name
    modified = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = row.to_record()
            metadata = dict(record.get("metadata") or {})
            if str(record.get("reuse_bucket") or "none") != "none":
                metadata["kv_core_decision_probe"] = {"force_recompute": True}
                modified += 1
            record["metadata"] = metadata
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "source_workload": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "output_workload": str(target),
        "output_sha256": _sha256_file(target),
        "rows": len(rows),
        "force_recompute_rows": modified,
        "seed_rows_unchanged": True,
    }
    (output_dir / "recompute_only_workload.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
