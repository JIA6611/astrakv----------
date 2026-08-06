#!/usr/bin/env python3
"""Generate DGX Spark mmap-backed KV chunk evidence.

This runner is intentionally standalone. It does not patch vLLM or LMCache; it
proves the DGX Spark-relevant execution path: KVChunkMeta -> mmap block ->
MADV_DONTNEED/MADV_WILLNEED -> mincore residency -> cold/warm read evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.vm.dgx_spark_adapter import DgxSparkKVAdapter, DgxSparkKVAdapterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dgx_spark_vm_evidence")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--block-size-mb", type=float, default=1.0)
    parser.add_argument("--total-blocks", type=int, default=64)
    parser.add_argument("--hot-chunks", type=int, default=3)
    parser.add_argument("--keep-backing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("dgx_spark_vm_evidence")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_size = max(4096, int(args.block_size_mb * 1024 * 1024))
    backing = output_dir / "dgx_spark_kv_backing.bin"

    config = DgxSparkKVAdapterConfig(
        backing_file=str(backing),
        total_blocks=max(args.total_blocks, args.chunks),
        block_size_bytes=block_size,
        keep_backing_file=args.keep_backing,
    )

    logger.info("Creating DGX Spark KV mmap evidence in %s", output_dir)
    hot_count = min(args.hot_chunks, args.chunks)
    with DgxSparkKVAdapter(config) as adapter:
        chunks = [
            KVChunkMeta(
                request_id="dgx-spark-smoke",
                layer_id=index,
                start_token=index * 128,
                end_token=(index + 1) * 128,
                chunk_id=f"chunk-{index}",
                tier=MemoryTier.SSD,
                dtype=config.dtype_str,
                size_bytes=block_size,
                cache_key=f"dgx-spark-smoke:{index}",
            )
            for index in range(args.chunks)
        ]
        records = [adapter.register_chunk(chunk) for chunk in chunks]

        items = block_size // config.dtype.itemsize
        for index, record in enumerate(records):
            data = np.full(items, index + 1, dtype=config.dtype)
            adapter.write_chunk(record.chunk.chunk_id, data)

        evict_events = adapter.evict_chunks([record.chunk.chunk_id for record in records])
        time.sleep(0.2)
        resident_after_evict = {
            record.chunk.chunk_id: adapter.resident_ratio(record.chunk.chunk_id)
            for record in records
        }

        hot_records = records[:hot_count]
        prefetch_events = [adapter.prefetch_chunk(record.chunk.chunk_id) for record in hot_records]
        time.sleep(0.2)
        resident_after_prefetch = {
            record.chunk.chunk_id: adapter.resident_ratio(record.chunk.chunk_id)
            for record in records
        }

        cold_target = records[min(hot_count, len(records) - 1)]
        cold_data, cold_event = adapter.read_chunk(cold_target.chunk.chunk_id)
        warm_data, warm_event = adapter.read_chunk(cold_target.chunk.chunk_id)

        checksum = int(cold_data[: min(1024, cold_data.size)].sum() + warm_data[: min(1024, warm_data.size)].sum())
        final_stats = adapter.cache.collect_stats().to_record()
        summary = {
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "target_architecture": "DGX Spark / unified memory / local NVMe",
            },
            "config": {
                "chunks": args.chunks,
                "hot_chunks": hot_count,
                "total_blocks": config.total_blocks,
                "block_size_bytes": block_size,
                "backing_file": str(backing),
                "keep_backing_file": args.keep_backing,
            },
            "chunk_records": adapter.chunk_records(),
            "residency": {
                "after_evict": resident_after_evict,
                "after_prefetch": resident_after_prefetch,
            },
            "latency": {
                "cold_read_us": round(cold_event.latency_us, 3),
                "warm_read_us": round(warm_event.latency_us, 3),
                "warm_over_cold_speedup": round(cold_event.latency_us / max(warm_event.latency_us, 1e-6), 3),
            },
            "events": [event.to_record() for event in adapter.events],
            "event_counts": {
                "evict": len(evict_events),
                "prefetch": len(prefetch_events),
                "read": 2,
            },
            "stats": final_stats,
            "checksum": checksum,
        }

    summary_path = output_dir / "dgx_spark_vm_evidence_summary.json"
    report_path = output_dir / "dgx_spark_vm_evidence_report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)
    logger.info("Report written to %s", report_path)
    return 0


def build_report(summary: dict) -> str:
    config = summary["config"]
    latency = summary["latency"]
    stats = summary["stats"]
    return "\n".join(
        [
            "# DGX Spark VM-Backed KV Evidence",
            "",
            "## Scope",
            "",
            "This run maps logical KV chunks onto mmap-backed blocks and exercises",
            "OS virtual-memory controls relevant to DGX Spark: local backing store,",
            "MADV_DONTNEED eviction, MADV_WILLNEED prefetch, and mincore residency.",
            "",
            "## Configuration",
            "",
            f"- Chunks: `{config['chunks']}`",
            f"- Hot chunks: `{config['hot_chunks']}`",
            f"- Total blocks: `{config['total_blocks']}`",
            f"- Block size bytes: `{config['block_size_bytes']}`",
            f"- Backing file: `{config['backing_file']}`",
            "",
            "## Evidence",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Cold read latency us | {latency['cold_read_us']} |",
            f"| Warm read latency us | {latency['warm_read_us']} |",
            f"| Warm/cold speedup | {latency['warm_over_cold_speedup']} |",
            f"| Resident blocks | {stats['resident_blocks']} |",
            f"| Resident ratio | {stats['resident_ratio']} |",
            f"| Prefetch requests | {stats['prefetch_requests']} |",
            f"| Evict requests | {stats['evict_requests']} |",
            "",
            "## Interpretation",
            "",
            "- This is chunk-level OS VM evidence, not a vLLM internal KV hook.",
            "- On DGX Spark, the most meaningful path is NVMe-backed mmap plus OS",
            "  residency control; LMCache CPU tier should be reported separately as",
            "  application-level cache ownership, not traditional PCIe offload.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
