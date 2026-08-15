"""Build a small, auditable Profile-B train/test transfer workload.

The source split contains different questions over some shared contexts.  For
each selected context this tool chooses one train question and one different
test question, then emits repeated visits inside each split so the existing
fire-consume runner can seed SSD state and measure a far/near pair.  Repeated
visits never cross the train/test boundary: the predictor sees only the train
question while the measured test question remains unseen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-profile-transfer-subset-v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"grouped prompt file is empty: {path}")
    return rows


def _context_key(row: dict[str, Any]) -> str:
    return str(
        row.get("context_hash")
        or row.get("reuse_group")
        or row.get("shared_context")
        or ""
    )


def _prefix_key(row: dict[str, Any]) -> str:
    return str(row.get("reuse_group") or row.get("context_hash") or "")


def _request_id(row: dict[str, Any]) -> str:
    return str(row.get("request_id") or row.get("sample_id") or "")


def _prompt_fingerprint(row: dict[str, Any]) -> str:
    declared = str(row.get("prompt_hash") or "")
    if declared:
        return declared
    payload: Any = row.get("messages")
    if not isinstance(payload, list):
        payload = str(row.get("prompt") or "")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _row_cost(row: dict[str, Any]) -> int:
    prompt = str(row.get("prompt") or "")
    if prompt:
        return len(prompt)
    messages = row.get("messages")
    return len(json.dumps(messages, ensure_ascii=False)) if isinstance(messages, list) else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_pairs(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    train_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        train_by_context[_context_key(row)].append(row)
    for row in test_rows:
        test_by_context[_context_key(row)].append(row)

    candidates: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for context in sorted(set(train_by_context) & set(test_by_context)):
        if not context:
            continue
        compatible: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for train_row in train_by_context[context]:
            for test_row in test_by_context[context]:
                if not _prefix_key(train_row) or _prefix_key(train_row) != _prefix_key(test_row):
                    continue
                if _request_id(train_row) == _request_id(test_row):
                    continue
                if _prompt_fingerprint(train_row) == _prompt_fingerprint(test_row):
                    continue
                compatible.append((
                    max(_row_cost(train_row), _row_cost(test_row)),
                    train_row,
                    test_row,
                ))
        if not compatible:
            continue
        cost, train_row, test_row = min(
            compatible,
            key=lambda item: (
                item[0], _request_id(item[1]), _request_id(item[2]),
            ),
        )
        candidates.append((cost, context, train_row, test_row))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < limit:
        raise ValueError(
            f"only {len(candidates)} disjoint shared-context pairs are available; "
            f"requested {limit}"
        )
    return [(context, train, test) for _, context, train, test in candidates[:limit]]


def _clone_visits(
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    side: str,
    visits: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair_index, (_, train_row, test_row) in enumerate(selected):
        source = train_row if side == "train" else test_row
        source_request_id = _request_id(source) or f"profile-transfer-{side}-{pair_index:03d}"
        for visit in range(visits):
            row = dict(source)
            row["request_id"] = f"{source_request_id}-transfer-{side}-visit-{visit + 1}"
            if row.get("sample_id"):
                row["sample_id"] = f"{row['sample_id']}-transfer-{side}-visit-{visit + 1}"
            row["order"] = len(output)
            row["transfer_source_request_id"] = source_request_id
            row["transfer_split"] = side
            row["transfer_visit"] = visit + 1
            output.append(row)
    return output


def materialize(
    train_path: Path,
    test_path: Path,
    output_root: Path,
    *,
    dataset: str,
    limit: int,
    visits: int,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if visits < 3:
        raise ValueError("visits must be at least 3 for fire-consume")
    train_rows = _load_jsonl(train_path)
    test_rows = _load_jsonl(test_path)
    selected = select_pairs(train_rows, test_rows, limit=limit)
    output_train = _clone_visits(selected, side="train", visits=visits)
    output_test = _clone_visits(selected, side="test", visits=visits)

    train_dir = output_root / "train" / dataset
    test_dir = output_root / "test" / dataset
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    output_train_path = train_dir / "grouped_prompts.jsonl"
    output_test_path = test_dir / "grouped_prompts.jsonl"
    write(output_train_path, output_train)
    write(output_test_path, output_test)

    train_original_ids = {_request_id(train) for _, train, _ in selected}
    test_original_ids = {_request_id(test) for _, _, test in selected}
    train_prompt_hashes = {_prompt_fingerprint(train) for _, train, _ in selected}
    test_prompt_hashes = {_prompt_fingerprint(test) for _, _, test in selected}
    train_prefixes = {_prefix_key(train) for _, train, _ in selected}
    test_prefixes = {_prefix_key(test) for _, _, test in selected}
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "limit": limit,
        "visits_per_split": visits,
        "sources": {
            "train": str(train_path),
            "test": str(test_path),
            "train_sha256": _sha256(train_path),
            "test_sha256": _sha256(test_path),
        },
        "outputs": {
            "train": str(output_train_path),
            "test": str(output_test_path),
            "train_rows": len(output_train),
            "test_rows": len(output_test),
        },
        "anti_leakage": {
            "train_test_original_request_id_overlap": sorted(
                train_original_ids & test_original_ids
            ),
            "train_test_prompt_hash_overlap": sorted(
                train_prompt_hashes & test_prompt_hashes
            ),
            "shared_prefixes": sorted(train_prefixes & test_prefixes),
            "shared_prefix_count": len(train_prefixes & test_prefixes),
            "passed": (
                not (train_original_ids & test_original_ids)
                and not (train_prompt_hashes & test_prompt_hashes)
                and train_prefixes == test_prefixes
                and len(train_prefixes) == limit
            ),
        },
        "selected": [
            {
                "context_key": context,
                "prefix_key": _prefix_key(train),
                "train_request_id": _request_id(train),
                "test_request_id": _request_id(test),
                "train_prompt_hash": _prompt_fingerprint(train),
                "test_prompt_hash": _prompt_fingerprint(test),
                "train_prompt_chars": _row_cost(train),
                "test_prompt_chars": _row_cost(test),
            }
            for context, train, test in selected
        ],
    }
    if not manifest["anti_leakage"]["passed"]:
        raise ValueError(f"anti-leakage validation failed: {manifest['anti_leakage']}")
    (output_root / "transfer_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-grouped-prompts", required=True)
    parser.add_argument("--test-grouped-prompts", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument("--limit", type=int, default=9)
    parser.add_argument("--visits", type=int, default=3)
    args = parser.parse_args()
    manifest = materialize(
        Path(args.train_grouped_prompts),
        Path(args.test_grouped_prompts),
        Path(args.output_root),
        dataset=args.dataset,
        limit=args.limit,
        visits=args.visits,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
