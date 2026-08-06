"""Generate the strategy-level AstraKV ON/OFF TTFT validation workload."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.policy_ab_ttft import (
    DEFAULT_ANCHOR_COUNT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHURN_VARIANTS,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EXPECTED_OUTPUT_TOKENS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_PROMPT_TOKENS,
    DEFAULT_SAMPLE_CYCLES,
    DEFAULT_WARMUP_CYCLES,
    build_workload_bundle,
    write_workload_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/policy-ab-ttft-workload")
    parser.add_argument("--anchor-count", type=int, default=DEFAULT_ANCHOR_COUNT)
    parser.add_argument("--churn-variants", type=int, default=DEFAULT_CHURN_VARIANTS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--warmup-cycles", type=int, default=DEFAULT_WARMUP_CYCLES)
    parser.add_argument("--sample-cycles", type=int, default=DEFAULT_SAMPLE_CYCLES)
    parser.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS)
    parser.add_argument("--expected-output-tokens", type=int, default=DEFAULT_EXPECTED_OUTPUT_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_workload_bundle(
        anchor_count=args.anchor_count,
        churn_variants=args.churn_variants,
        prompt_tokens=args.prompt_tokens,
        warmup_cycles=args.warmup_cycles,
        sample_cycles=args.sample_cycles,
        idle_seconds=args.idle_seconds,
        expected_output_tokens=args.expected_output_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    written = write_workload_bundle(args.output_dir, bundle)
    print(f"Workload written to {written['workload']}")
    print(f"Manifest written to {written['manifest']}")
    print(f"Report written to {written['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
