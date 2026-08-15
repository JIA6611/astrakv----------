#!/usr/bin/env python3
"""Small local demo of AstraKV-W's mainline latency logic.

This is a deterministic tiering simulation, not a real vLLM/LMCache run.  It
shows why the mainline policies reduce demand misses: Prefetch-A and
Prefetch-B promote reusable SSD/CPU blocks before the next access, while a
no-op baseline pays the full CPU-miss latency every time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.prefetch.selective_kv import (
    KVBlockRef,
    KVAccessSource,
    SelectiveKVPrefetchConfig,
    SelectiveKVPrefetchMVP,
)


async def run_policy(*, trace: list[str], prefetch: bool, prefetch_window: int) -> dict[str, object]:
    blocks = sorted(set(trace), key=trace.index)
    config = SelectiveKVPrefetchConfig(
        gpu_capacity_blocks=max(2, len(blocks) // 2),
        block_size_bytes=1,
        prefetch_window=prefetch_window,
        prefetch_latency_ms=0.5,
        cpu_miss_latency_ms=4.0,
        gpu_hit_latency_ms=0.05,
        prefetch_hit_latency_ms=0.10,
    )
    engine = SelectiveKVPrefetchMVP(config)
    engine.add_cpu_blocks([KVBlockRef(block_id=block_id, size_bytes=1) for block_id in blocks])
    await engine.start()
    sources: list[str] = []
    total_latency_ms = 0.0
    try:
        for position, block_id in enumerate(trace):
            if prefetch:
                await engine.submit_predictions(trace, position)
            result = await engine.access(block_id)
            total_latency_ms += result.latency_ms
            sources.append(result.source.value)
    finally:
        await engine.close()
    metrics = engine.to_metrics_record()
    return {
        "policy": "mainline_prefetch" if prefetch else "baseline_no_policy",
        "total_access_latency_ms": round(total_latency_ms, 3),
        "access_sources": sources,
        "metrics": metrics,
    }


def build_demo_trace(blocks: int) -> list[str]:
    sequence = [f"block-{index}" for index in range(blocks)]
    # Two passes over the same prompt: the first pass warms the lower tier,
    # the second pass measures whether the mainline promoted it in advance.
    return sequence + sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--prefetch-window", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="results/mainline_mini_demo",
        help="Directory for report.json and report.md",
    )
    args = parser.parse_args()

    trace = build_demo_trace(max(2, args.blocks))
    baseline = asyncio.run(run_policy(trace=trace, prefetch=False, prefetch_window=0))
    mainline = asyncio.run(
        run_policy(trace=trace, prefetch=True, prefetch_window=max(1, args.prefetch_window))
    )

    baseline_latency = float(baseline["total_access_latency_ms"])
    mainline_latency = float(mainline["total_access_latency_ms"])
    speedup = baseline_latency / max(1e-9, mainline_latency)
    improvement_pct = 100.0 * (baseline_latency - mainline_latency) / max(1e-9, baseline_latency)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "demo": "mainline_mini_demo",
        "mode": "synthetic_tiering_proxy",
        "blocks": max(2, args.blocks),
        "trace_length": len(trace),
        "baseline": baseline,
        "mainline": mainline,
        "baseline_total_latency_ms": baseline_latency,
        "mainline_total_latency_ms": mainline_latency,
        "speedup_x": round(speedup, 3),
        "latency_improvement_pct": round(improvement_pct, 3),
        "note": "Local deterministic proxy. Real DGX evidence must use benchmark_results.csv and runtime receipts.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        (
            "# AstraKV-W 主线 Mini Demo\n\n"
            f"- baseline 总访问延迟：{baseline_latency:.3f} ms\n"
            f"- mainline 总访问延迟：{mainline_latency:.3f} ms\n"
            f"- 提升：{improvement_pct:.1f}%（{speedup:.2f}x）\n\n"
            "说明：这是本地时序代理，证明的是预取/补位策略能减少需求访问 miss；"
            "不是真实 vLLM TTFT。\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
