"""Split grouped prompts into train/test subsets that share hot contexts.

The transfer methodology requires the test workload to contain requests the
train workload never saw, while sharing the same hot chunks/contexts.  This
tool splits one ``grouped_prompts.jsonl`` by context: questions over the same
document context are distributed across both subsets, so the two subsets
overlap at the chunk level but not at the request level.

Output layout (consumed by run_prefetch_transfer_ablation.sh):

    <output-root>/train/<dataset>/grouped_prompts.jsonl
    <output-root>/test/<dataset>/grouped_prompts.jsonl
    <output-root>/split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-transfer-split-v1"


def _context_key(row: dict[str, Any]) -> str:
    for field in ("context_hash", "shared_context", "reuse_group"):
        value = row.get(field)
        if value:
            return str(value)
    return "unknown"


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("order") or 0))


def split_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_context_key(row)].append(row)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    shared_contexts = 0
    single_side_contexts = 0
    for context, context_rows in grouped.items():
        ordered = _sorted_rows(context_rows)
        if len(ordered) == 1:
            single_side_contexts += 1
            if rng.random() < train_ratio:
                train_rows.append(ordered[0])
            else:
                test_rows.append(ordered[0])
            continue
        rng.shuffle(ordered)
        train_count = max(1, min(len(ordered) - 1, round(len(ordered) * train_ratio)))
        train_rows.extend(ordered[:train_count])
        test_rows.extend(ordered[train_count:])
        shared_contexts += 1

    train_rows = _sorted_rows(train_rows)
    test_rows = _sorted_rows(test_rows)
    stats = {
        "contexts_total": len(grouped),
        "contexts_shared_both_sides": shared_contexts,
        "contexts_single_side": single_side_contexts,
        "train_questions": len(train_rows),
        "test_questions": len(test_rows),
        "train_contexts": len({_context_key(row) for row in train_rows}),
        "test_contexts": len({_context_key(row) for row in test_rows}),
        "overlap_contexts": len(
            {_context_key(row) for row in train_rows}
            & {_context_key(row) for row in test_rows}
        ),
    }
    return train_rows, test_rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-prompts-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in (0, 1)")
    source = Path(args.grouped_prompts_jsonl)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no rows in {source}")

    train_rows, test_rows, stats = split_rows(
        rows, train_ratio=args.train_ratio, seed=args.seed,
    )
    output_root = Path(args.output_root)
    train_dir = output_root / "train" / args.dataset
    test_dir = output_root / "test" / args.dataset
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, subset: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in subset),
            encoding="utf-8",
        )

    write(train_dir / "grouped_prompts.jsonl", train_rows)
    write(test_dir / "grouped_prompts.jsonl", test_rows)
    manifest = {
        "schema": SCHEMA,
        "source": str(source),
        "dataset": args.dataset,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stats": stats,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
