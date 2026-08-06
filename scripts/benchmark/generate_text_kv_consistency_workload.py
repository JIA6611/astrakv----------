"""Generate the controlled workload bundle for target-1 text/KV consistency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.text_kv_consistency import (
    DEFAULT_BLOCK_SIZE_TOKENS,
    DEFAULT_CONTEXT_LENGTHS,
    build_workload_bundle,
    write_workload_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/text-kv-consistency-workload")
    parser.add_argument(
        "--context-lengths",
        default=",".join(str(item) for item in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated context lengths to generate. Default: 8192,16384",
    )
    parser.add_argument("--expected-output-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size-tokens", type=int, default=DEFAULT_BLOCK_SIZE_TOKENS)
    parser.add_argument("--model-path", default=os.environ.get("ASTRAKV_MODEL", ""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    context_lengths = parse_context_lengths(args.context_lengths)
    suite_manifest = {
        "schema": "astrakv-text-kv-consistency-workload-suite-v1",
        "target_context_lengths": context_lengths,
        "target_block_size_tokens": int(args.block_size_tokens),
        "expected_output_tokens": int(args.expected_output_tokens),
        "batch_size": int(args.batch_size),
        "model_path": str(args.model_path or ""),
        "contexts": {},
    }
    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    for context_length in context_lengths:
        bundle = build_workload_bundle(
            context_length=context_length,
            expected_output_tokens=args.expected_output_tokens,
            batch_size=args.batch_size,
            block_size_tokens=args.block_size_tokens,
            tokenizer=tokenizer,
        )
        context_dir = output_root / context_label(context_length)
        written = write_workload_bundle(context_dir, bundle)
        suite_manifest["contexts"][str(context_length)] = {
            "label": context_dir.name,
            "analysis_workload": str(written["analysis_workload"]),
            "warmup_workload": str(written["warmup_workload"]),
            "pairwise_reference_replay": str(written["pairwise_reference_replay"]),
            "manifest": str(written["manifest"]),
            "report": str(written["report"]),
        }
        print(f"[ctx={context_length}] Analysis workload written to {written['analysis_workload']}")
        print(f"[ctx={context_length}] Warmup workload written to {written['warmup_workload']}")
        print(f"[ctx={context_length}] Replay workload written to {written['pairwise_reference_replay']}")
        print(f"[ctx={context_length}] Manifest written to {written['manifest']}")
        print(f"[ctx={context_length}] Report written to {written['report']}")
    (output_root / "text_kv_consistency_workload_suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_context_lengths(value: str) -> list[int]:
    items = [item.strip() for item in str(value).split(",")]
    result: list[int] = []
    for item in items:
        if not item:
            continue
        result.append(int(item))
    if not result:
        raise SystemExit("at least one context length is required")
    return result


def context_label(context_length: int) -> str:
    mapping = {8192: "ctx8k", 16384: "ctx16k"}
    if context_length in mapping:
        return mapping[context_length]
    if context_length >= 1024:
        return f"ctx{context_length // 1024}k"
    return f"ctx{context_length}"


if __name__ == "__main__":
    raise SystemExit(main())
