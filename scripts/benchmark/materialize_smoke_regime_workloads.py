"""Materialize small closed-loop smoke workloads for the regime suite.

The canonical regime workloads use long Qasper/MultifieldQA contexts and many
rows, which makes a first DGX loop-check expensive.  This generator builds the
same four canonical workload files (same names and schema) with a handful of
rows and short deterministic prompts so the full chain
(materialize -> server -> benchmark -> accounting -> acceptance -> report)
can be verified end to end before the real matrix runs.

The default context length (3000 tokens) is deliberately above the 2048-token
partial-prefix cap so the E4 arm really splits load vs recompute instead of
degenerating into a full load; otherwise the partial-prefix evidence gate
would fail for the wrong reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "astrakv-smoke-regime-workloads-v1"
WORKLOAD_SCHEMA = "astra-runtime-workload-v1"


@dataclass(frozen=True, slots=True)
class SmokeWorkloads:
    output_dir: Path

    def write(self, rows_by_name: dict[str, list[dict[str, Any]]]) -> dict:
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "workloads": {},
        }
        for name, rows in sorted(rows_by_name.items()):
            path = self.output_dir / f"{name}.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest["workloads"][name] = {
                "row_count": len(rows),
                "total_context_tokens": sum(int(row["context_length"] or 0) for row in rows),
                "sha256": _sha256_file(path),
            }
        manifest_path = self.output_dir / "smoke_regime_workloads.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=3000)
    parser.add_argument("--groups", type=int, default=2, help="Shared-context groups for repeated/queued/churn workloads.")
    parser.add_argument("--revisits", type=int, default=3)
    parser.add_argument("--churn-groups", type=int, default=4)
    parser.add_argument("--churn-revisits", type=int, default=2)
    parser.add_argument("--random-rows", type=int, default=6)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prefetch-lead-s", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(materialize(args), sort_keys=True))
    return 0


def materialize(args: argparse.Namespace) -> dict:
    if args.context_tokens < 2304:
        raise SystemExit("--context-tokens must exceed the 2048 partial cap plus one chunk so the E4 split is real")
    rng = random.Random(args.seed)
    rows = {
        "repeated_long_prefix": build_repeated_long_prefix(
            args, rng, groups=args.groups, revisits=args.revisits,
        ),
        "constrained_kv_churn": build_constrained_kv_churn(
            args, rng, groups=args.churn_groups, revisits=args.churn_revisits,
        ),
        "queued_concurrency": build_queued_concurrency(
            args, rng, groups=args.groups, revisits=args.revisits,
        ),
        "random_no_reuse": build_random_no_reuse(args, rng, count=args.random_rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SmokeWorkloads(output_dir).write(rows)


def build_repeated_long_prefix(args: argparse.Namespace, rng: random.Random, *, groups: int, revisits: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for group in range(groups):
        prefix = _prefix(args, group)
        for visit in range(revisits):
            index += 1
            rows.append(_row(
                args=args,
                workload="repeated_long_prefix",
                index=index,
                prompt=prefix["prompt"],
                prefix_id=prefix["prefix_id"],
                reuse_bucket="none" if visit == 0 else "high",
                reuse_ratio=0.0 if visit == 0 else 0.95,
                prefetch_lead_s=0.0 if visit == 0 else args.prefetch_lead_s,
                metadata={"group": group, "visit": visit},
            ))
    rng.shuffle(rows)
    return rows


def build_constrained_kv_churn(args: argparse.Namespace, rng: random.Random, *, groups: int, revisits: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    prefixes = [_prefix(args, group) for group in range(groups)]
    # Visit-major interleaving forces SSD reloads between revisits.
    for visit in range(revisits):
        for group in range(groups):
            index += 1
            rows.append(_row(
                args=args,
                workload="constrained_kv_churn",
                index=index,
                prompt=prefixes[group]["prompt"],
                prefix_id=prefixes[group]["prefix_id"],
                reuse_bucket="none" if visit == 0 else "low",
                reuse_ratio=0.0 if visit == 0 else 0.25,
                prefetch_lead_s=0.0 if visit == 0 else args.prefetch_lead_s,
                metadata={"group": group, "visit": visit},
            ))
    rng.shuffle(rows)
    return rows


def build_queued_concurrency(args: argparse.Namespace, rng: random.Random, *, groups: int, revisits: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    prefixes = [_prefix(args, group) for group in range(groups)]
    for visit in range(revisits):
        for group in range(groups):
            index += 1
            rows.append(_row(
                args=args,
                workload="queued_concurrency",
                index=index,
                prompt=prefixes[group]["prompt"],
                prefix_id=prefixes[group]["prefix_id"],
                reuse_bucket="none" if visit == 0 else "high",
                reuse_ratio=0.0 if visit == 0 else 0.9,
                prefetch_lead_s=0.0 if visit == 0 else args.prefetch_lead_s,
                metadata={"group": group, "visit": visit},
            ))
    rng.shuffle(rows)
    return rows


def build_random_no_reuse(args: argparse.Namespace, rng: random.Random, *, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        prompt = _unique_prompt(args, index)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        rows.append(_row(
            args=args,
            workload="random_no_reuse",
            index=index,
            prompt=prompt,
            prefix_id=f"random:{digest[:16]}",
            reuse_bucket="none",
            reuse_ratio=0.0,
            prefetch_lead_s=0.0,
            metadata={"unique": index},
        ))
    rng.shuffle(rows)
    return rows


def _prefix(args: argparse.Namespace, group: int) -> dict[str, str]:
    target_chars = max(64, int(args.context_tokens * 4))
    base = f"Smoke shared context for group {group}. Each sentence is deterministic and reusable. "
    repeat = max(1, target_chars // len(base))
    prompt = (base * repeat) + "Question: what does the smoke workload verify?"
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "prompt": prompt,
        "prefix_id": f"smoke-group-{group}",
        "prefix_hash": f"sha256:{digest}",
    }


def _unique_prompt(args: argparse.Namespace, index: int) -> str:
    base = f"Unique smoke prompt number {index} with no shared prefix. "
    target_chars = max(64, int(args.context_tokens * 4))
    return (base * max(1, target_chars // len(base))) + f"End of unique row {index}."


def _row(
    *,
    args: argparse.Namespace,
    workload: str,
    index: int,
    prompt: str,
    prefix_id: str,
    reuse_bucket: str,
    reuse_ratio: float,
    prefetch_lead_s: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "schema": WORKLOAD_SCHEMA,
        "request_id": f"{workload}-{index:06d}",
        "prompt": prompt,
        "prefix_id": prefix_id,
        "prefix_hash": f"sha256:{digest}",
        "cache_key": "",
        "arrival_index": index,
        "reuse_ratio": reuse_ratio,
        "reuse_bucket": reuse_bucket,
        "context_length": max(1, len(prompt) // 4),
        "expected_output_tokens": args.output_tokens,
        "batch_size": 1,
        "sleep_before_s": 0.5 if prefetch_lead_s > 0 else 0.0,
        "prefetch_lead_s": prefetch_lead_s,
        "case": workload,
        "metadata": {
            "workload_type": workload,
            "generation_seed": 0,
            "smoke": True,
            **metadata,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
