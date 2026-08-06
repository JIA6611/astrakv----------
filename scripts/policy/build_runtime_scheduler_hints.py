"""Merge offline scheduler artifacts into one runtime-consumable hint stream."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.scheduler_hints import SchedulerHintIndex  # noqa: E402
from astrakv.scheduler.hints import SchedulerHint  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_hints: list[SchedulerHint] = []
    for path in args.hints:
        all_hints.extend(_load_hint_jsonl(path))

    index = SchedulerHintIndex.from_hints(all_hints)
    merged = _merge_by_request(index)
    output_path = output_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as handle:
        for hint in merged:
            handle.write(json.dumps(asdict(hint), ensure_ascii=False) + "\n")
    print(f"Runtime scheduler hints written to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hints", action="append", required=True, help="Hint JSONL path. Repeatable.")
    parser.add_argument("--output-dir", default="results/runtime_scheduler_hints")
    parser.add_argument("--output-name", default="runtime_scheduler_hints.jsonl")
    return parser.parse_args()


def _load_hint_jsonl(path: str | Path) -> list[SchedulerHint]:
    rows: list[SchedulerHint] = []
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"Hint JSONL not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid hint JSON at {input_path}:{line_number}: {exc}") from exc
            rows.append(SchedulerHint(
                request_id=str(record.get("request_id") or ""),
                action=str(record.get("action") or ""),
                reason=str(record.get("reason") or ""),
                priority=int(record.get("priority") or 0),
                metadata=dict(record.get("metadata") or {}),
            ))
    return rows


def _merge_by_request(index: SchedulerHintIndex) -> list[SchedulerHint]:
    merged: list[SchedulerHint] = []
    for request_id in sorted(index.by_request):
        seen_objects: set[str] = set()
        for hint in index.hints_for_request(request_id):
            metadata = hint.metadata
            object_id = str(
                metadata.get("chunk_id")
                or metadata.get("object_key")
                or metadata.get("cache_key")
                or metadata.get("prefix_id")
                or request_id
            )
            dedupe_key = f"{hint.action}:{object_id}"
            if dedupe_key in seen_objects:
                continue
            seen_objects.add(dedupe_key)
            merged.append(hint)
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
