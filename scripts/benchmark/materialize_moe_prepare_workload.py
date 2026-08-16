#!/usr/bin/env python3
"""Build deterministic 2K/8K exact-prefix rows for the MoE prepare demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.runtime_workload import (  # noqa: E402
    RUNTIME_WORKLOAD_SCHEMA_VERSION,
    load_runtime_workload_jsonl,
)
from scripts.benchmark.run_real_benchmark import tokenize_chat_messages  # noqa: E402


SYSTEM_PROMPT = "You are a concise assistant for MoE request-ahead prefill validation."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/opt/models/Qwen3.6-35B-A3B")
    parser.add_argument("--output", default="results/moe-prepare-workload.jsonl")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[2048, 8192])
    parser.add_argument("--visits", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--prefetch-lead-s", type=float, default=0.25)
    args = parser.parse_args()
    if args.visits <= 0 or args.output_tokens <= 0:
        raise SystemExit("--visits and --output-tokens must be positive")

    rows: list[dict[str, Any]] = []
    arrival_index = 0
    for target in args.context_lengths:
        messages, exact_ids = build_exact_messages(args.model, target)
        prompt = str(messages[-1]["content"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for visit in range(args.visits):
            arrival_index += 1
            rows.append({
                "schema": RUNTIME_WORKLOAD_SCHEMA_VERSION,
                "request_id": f"moe-ctx{target}-visit{visit}",
                "prompt": prompt,
                "prefix_id": f"moe-exact-ctx{target}",
                "prefix_hash": f"sha256:{digest}",
                "cache_key": f"moe-exact-ctx{target}",
                "arrival_index": arrival_index,
                "reuse_ratio": 0.0 if visit == 0 else 1.0,
                "reuse_bucket": "none" if visit == 0 else "high",
                "context_length": len(exact_ids),
                "expected_output_tokens": args.output_tokens,
                "batch_size": 1,
                "sleep_before_s": 0.0,
                "prefetch_lead_s": args.prefetch_lead_s,
                "case": f"moe_ctx{target}",
                "metadata": {
                    "messages": messages,
                    "exact_token_ids": list(exact_ids),
                    "generation_seed": 0,
                    "target_context_length": target,
                    "group_visit": visit,
                    "scenario": "moe_request_ahead_prefill",
                },
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    load_runtime_workload_jsonl(output)
    print(f"MoE prepare workload written to {output} ({len(rows)} rows)")
    return 0


def build_exact_messages(model: str, target_tokens: int) -> tuple[list[dict[str, str]], tuple[int, ...]]:
    if target_tokens <= 128:
        raise ValueError("target context length must exceed chat-template overhead")
    word_count = max(1, target_tokens - 96)
    messages: list[dict[str, str]] = []
    exact_ids: tuple[int, ...] = ()
    for _ in range(12):
        prompt = (
            "Use this repeated context for an exact-prefix MoE routing and KV validation.\n"
            + ("the " * word_count)
            + "\nSummarize the context in one short sentence."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        exact_ids = tokenize_chat_messages(
            messages, tokenizer_path=model, tokenizer_revision="",
        )
        delta = target_tokens - len(exact_ids)
        if delta == 0:
            break
        word_count = max(1, word_count + delta)
    if abs(len(exact_ids) - target_tokens) > 8:
        raise RuntimeError(
            f"could not materialize context near {target_tokens} tokens; got {len(exact_ids)}"
        )
    return messages, exact_ids


if __name__ == "__main__":
    raise SystemExit(main())
