#!/usr/bin/env python3
"""Materialize the four canonical KV-Core runtime workloads.

``run_kv_core_controlled_suite.sh`` consumes ``repeated_long_prefix.jsonl``,
``random_no_reuse.jsonl``, ``constrained_kv_churn.jsonl`` and
``queued_concurrency.jsonl`` from ``--workload-dir``.  This generator builds
those files deterministically from the audited prompt sources in
``workload_prompts/<dataset>/`` (``grouped_prompts.jsonl`` and
``random_prompts.jsonl``) into the canonical ``astra-runtime-workload-v1``
contract validated by ``astrakv.benchmarks.runtime_workload``.

Semantics per workload:

- ``repeated_long_prefix``: a few shared-context groups, each revisited many
  times with the identical prompt so the exact token prefix resolves to one
  LMCache object.  Revisits carry a prefetch lead so E3/E4 have a real
  SSD-to-CPU promotion window and E3C measures the same rows without it.
- ``random_no_reuse``: unique prompts with unique prefix ids.  Regression
  control: admission/prefetch must not help and must not regress TTFT.
- ``constrained_kv_churn``: many shared-context groups with long contexts,
  interleaved so GPU/LMCache capacity pressure forces SSD reloads between
  revisits.  Reuse is intentionally low-to-medium.
- ``queued_concurrency``: a few shared-context groups arriving round-robin
  with zero sleep so the serving queue is exercised back-to-back.

The workload-driven benchmark executes rows sequentially (``run_batch`` is
only used by the matrix path); "concurrency" here means rapid back-to-back
arrivals that share prefixes and therefore queue on the same LMCache object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.runtime_workload import (  # noqa: E402
    RUNTIME_WORKLOAD_SCHEMA_VERSION,
    load_runtime_workload_jsonl,
)

WORKLOAD_MANIFEST_SCHEMA = "astrakv-kv-core-workload-manifest-v1"
CANONICAL_WORKLOADS = (
    "repeated_long_prefix",
    "random_no_reuse",
    "constrained_kv_churn",
    "queued_concurrency",
)


@dataclass(frozen=True, slots=True)
class PromptSource:
    dataset: str
    prompt: str
    prompt_hash: str
    reuse_group: str
    shared_context: bool
    estimated_prompt_tokens: int


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_sources(grouped_path: Path, random_path: Path, dataset: str) -> list[PromptSource]:
    sources: list[PromptSource] = []
    for path, is_random in ((grouped_path, False), (random_path, True)):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = raw.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    continue
                metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                estimated = int(metadata.get("estimated_prompt_tokens") or 0)
                reuse_group = str(raw.get("reuse_group") or "")
                if is_random and not reuse_group:
                    reuse_group = f"random:{_sha256(prompt)[:16]}"
                sources.append(
                    PromptSource(
                        dataset=dataset,
                        prompt=prompt,
                        prompt_hash=str(raw.get("prompt_hash") or _sha256(prompt)),
                        reuse_group=reuse_group,
                        shared_context=bool(raw.get("shared_context")) or not is_random,
                        estimated_prompt_tokens=estimated or max(1, len(prompt) // 4),
                    )
                )
    return sources


def _grouped_sources(
    sources: Iterable[PromptSource],
    *,
    rng: random.Random,
    max_context_tokens: int,
    count: int,
    min_context_tokens: int = 0,
) -> list[list[PromptSource]]:
    by_group: dict[str, list[PromptSource]] = {}
    for source in sources:
        if not source.shared_context or not source.reuse_group:
            continue
        if source.estimated_prompt_tokens > max_context_tokens:
            continue
        if source.estimated_prompt_tokens < min_context_tokens:
            continue
        by_group.setdefault(source.reuse_group, []).append(source)
    candidates = [group for group in by_group.values() if group]
    rng.shuffle(candidates)
    return candidates[:count]


def _row(
    *,
    workload: str,
    request_index: int,
    source: PromptSource,
    prefix_id: str,
    prefix_hash: str,
    reuse_ratio: float,
    reuse_bucket: str,
    expected_output_tokens: int,
    prefetch_lead_s: float,
    batch_size: int = 1,
    case_suffix: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_dataset": source.dataset,
        "source_prompt_hash": source.prompt_hash,
        "generation_seed": 0,
        "visit": request_index,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "schema": RUNTIME_WORKLOAD_SCHEMA_VERSION,
        "request_id": f"{workload}-{request_index:06d}",
        "prompt": source.prompt,
        "prefix_id": prefix_id,
        "prefix_hash": prefix_hash,
        "cache_key": "",
        "arrival_index": request_index,
        "reuse_ratio": reuse_ratio,
        "reuse_bucket": reuse_bucket,
        "context_length": source.estimated_prompt_tokens,
        "expected_output_tokens": expected_output_tokens,
        "batch_size": batch_size,
        "sleep_before_s": 0.0,
        "prefetch_lead_s": prefetch_lead_s,
        "case": f"{workload}{case_suffix}",
        "metadata": metadata,
    }


def build_repeated_long_prefix(
    grouped: list[list[PromptSource]],
    *,
    groups: int = 8,
    revisits: int = 8,
    output_tokens: int = 128,
    prefetch_lead_s: float = 0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for group in grouped[:groups]:
        seed = group[0]
        prefix_id = f"{seed.dataset}:{seed.reuse_group}"
        prefix_hash = f"sha256:{_sha256(seed.prompt)}"
        for visit in range(revisits):
            index += 1
            rows.append(
                _row(
                    workload="repeated_long_prefix",
                    request_index=index,
                    source=seed,
                    prefix_id=prefix_id,
                    prefix_hash=prefix_hash,
                    reuse_ratio=0.0 if visit == 0 else 0.95,
                    reuse_bucket="none" if visit == 0 else "high",
                    expected_output_tokens=output_tokens,
                    prefetch_lead_s=0.0 if visit == 0 else prefetch_lead_s,
                    extra_metadata={"group_visit": visit},
                )
            )
    return rows


def build_random_no_reuse(
    unique: list[PromptSource],
    *,
    requests: int = 48,
    output_tokens: int = 128,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(unique[:requests], start=1):
        rows.append(
            _row(
                workload="random_no_reuse",
                request_index=index,
                source=source,
                prefix_id=f"{source.dataset}:random:{source.reuse_group}",
                prefix_hash=f"sha256:{_sha256(source.prompt)}",
                reuse_ratio=0.0,
                reuse_bucket="none",
                expected_output_tokens=output_tokens,
                prefetch_lead_s=0.0,
            )
        )
    return rows


def build_constrained_kv_churn(
    grouped: list[list[PromptSource]],
    *,
    groups: int = 24,
    revisits: int = 3,
    output_tokens: int = 16,
    prefetch_lead_s: float = 0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    selected = grouped[:groups]
    # Interleave groups so no group is revisited before many others arrive,
    # forcing SSD reloads under capacity pressure.
    for visit in range(revisits):
        for group in selected:
            index += 1
            seed = group[0]
            rows.append(
                _row(
                    workload="constrained_kv_churn",
                    request_index=index,
                    source=seed,
                    prefix_id=f"{seed.dataset}:{seed.reuse_group}",
                    prefix_hash=f"sha256:{_sha256(seed.prompt)}",
                    reuse_ratio=0.0 if visit == 0 else 0.25,
                    reuse_bucket="none" if visit == 0 else "low",
                    expected_output_tokens=output_tokens,
                    prefetch_lead_s=0.0 if visit == 0 else prefetch_lead_s,
                    extra_metadata={"group_visit": visit},
                )
            )
    return rows


def build_queued_concurrency(
    grouped: list[list[PromptSource]],
    *,
    groups: int = 4,
    revisits: int = 8,
    output_tokens: int = 16,
    prefetch_lead_s: float = 0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    selected = grouped[:groups]
    seeds = [group[0] for group in selected]
    # Round-robin zero-sleep arrivals: every request hits the queue back-to-back
    # and all revisits of one group share the same LMCache object.
    for visit in range(revisits):
        for seed in seeds:
            index += 1
            rows.append(
                _row(
                    workload="queued_concurrency",
                    request_index=index,
                    source=seed,
                    prefix_id=f"{seed.dataset}:{seed.reuse_group}",
                    prefix_hash=f"sha256:{_sha256(seed.prompt)}",
                    reuse_ratio=0.0 if visit == 0 else 0.9,
                    reuse_bucket="none" if visit == 0 else "high",
                    expected_output_tokens=output_tokens,
                    prefetch_lead_s=0.0 if visit == 0 else prefetch_lead_s,
                    batch_size=1,
                    extra_metadata={"group_visit": visit},
                )
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reuse_buckets: dict[str, int] = {}
    for row in rows:
        bucket = str(row["reuse_bucket"])
        reuse_buckets[bucket] = reuse_buckets.get(bucket, 0) + 1
    return {
        "row_count": len(rows),
        "prefix_group_count": len({row["prefix_id"] for row in rows}),
        "reuse_buckets": reuse_buckets,
        "total_context_tokens": sum(int(row["context_length"] or 0) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", default=str(PROJECT_ROOT / "workload_prompts"))
    parser.add_argument("--datasets", default="qasper,multifieldqa_en")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "kv-core-workloads"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=128)
    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir)
    datasets = [name for name in (item.strip() for item in args.datasets.split(",")) if name]
    if not datasets:
        raise SystemExit("--datasets must name at least one prompt dataset")

    grouped_sources: list[PromptSource] = []
    unique_sources: list[PromptSource] = []
    for dataset in datasets:
        grouped_sources.extend(
            _prompt_sources(
                prompts_dir / dataset / "grouped_prompts.jsonl",
                prompts_dir / dataset / "random_prompts.jsonl",
                dataset,
            )
        )
    if not grouped_sources:
        raise SystemExit(f"no prompt sources found under {prompts_dir}")

    rng = random.Random(args.seed)
    grouped = _grouped_sources(
        grouped_sources,
        rng=rng,
        max_context_tokens=args.max_context_tokens,
        count=max(24, 4),
    )
    unique = [
        source
        for source in grouped_sources
        if not source.shared_context or not source.reuse_group
    ]
    rng.shuffle(unique)

    builders: dict[str, list[dict[str, Any]]] = {
        "repeated_long_prefix": build_repeated_long_prefix(grouped, output_tokens=args.output_tokens),
        "random_no_reuse": build_random_no_reuse(unique, output_tokens=args.output_tokens),
        "constrained_kv_churn": build_constrained_kv_churn(grouped, output_tokens=args.output_tokens),
        "queued_concurrency": build_queued_concurrency(grouped, output_tokens=args.output_tokens),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_WORKLOADS:
        rows = builders[name]
        path = output_dir / f"{name}.jsonl"
        _write_jsonl(path, rows)
        # Contract validation doubles as the generator's self-check.
        load_runtime_workload_jsonl(path)
        summaries[name] = _summary(rows)
        print(f"{name}: {len(rows)} rows -> {path}")

    manifest = {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "prompts_dir": str(prompts_dir),
            "datasets": datasets,
            "seed": args.seed,
            "max_context_tokens": args.max_context_tokens,
            "output_tokens": args.output_tokens,
        },
        "workloads": summaries,
    }
    manifest_path = output_dir / "kv_core_workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
