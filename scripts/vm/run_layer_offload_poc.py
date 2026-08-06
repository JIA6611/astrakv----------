"""CLI entry point for the layer offloading PoC.

Runs the model-weight layer offloading experiment to produce quantitative
evidence for the competition report: how much GPU memory can be saved by
layer-by-layer offloading, and what the latency cost is.

Usage:
    # Quick smoke test with 1.5B model
    python scripts/vm/run_layer_offload_poc.py --model Qwen/Qwen2.5-1.5B-Instruct

    # Full experiment with 7B model
    python scripts/vm/run_layer_offload_poc.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from transformers import AutoTokenizer
    _HAS_DEPS = True
except ImportError as e:
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _HAS_DEPS = False
    _MISSING_DEPS_MSG = str(e)  # noqa: F841

from astrakv.vm.layer_offload import (
    LayerOffloadConfig,
    LayerOffloadManager,
    LayerOffloadResult,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer Offloading PoC")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Model name (default: Qwen/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--output-dir", default="results/gpu/layer_offload_poc",
                   help="Output directory")
    p.add_argument("--windows", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32],
                   help="GPU layer window sizes to test (default: 1 2 4 8 16 32)")
    p.add_argument("--prompt-multiplier", type=int, default=10,
                   help="Repeat prompt N times for longer input (default: 10)")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip full-model GPU baseline measurement")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("run_layer_offload_poc")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check dependencies ──
    if not _HAS_DEPS:
        logger.error("Missing dependencies (torch, transformers). Install with: pip install torch transformers")
        return 1

    # ── Check CUDA availability ──
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Layer offloading PoC requires a GPU.")
        logger.error("On CPU-only systems, this experiment cannot produce GPU memory metrics.")
        return 1

    logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_mem / 1e9)

    # ── Initialize ──
    config = LayerOffloadConfig(model_name=args.model)
    manager = LayerOffloadManager(config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Load model to CPU ──
    model = manager.load_model_to_cpu()

    # ── Prepare input ──
    prompt = "Explain the concept of virtual memory in operating systems." * args.prompt_multiplier
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    logger.info("Input: %d tokens", input_ids.shape[1])

    # ── Baseline: full model on GPU ──
    if not args.no_baseline:
        logger.info("=== Baseline: full model on GPU ===")
        try:
            full_gpu_mb = manager.measure_gpu_memory_no_offload(model)
            logger.info("Full-model GPU memory: %.0f MB", full_gpu_mb)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Full model OOM on GPU — this is expected for constrained devices!")
            full_gpu_mb = float("inf")
    else:
        full_gpu_mb = None

    # ── Offloading experiments ──
    logger.info("=== Layer offloading experiments ===")
    results = manager.measure_gpu_memory_with_offload(
        model, input_ids, window_sizes=args.windows
    )

    # ── Save results ──
    LayerOffloadManager.save_results(results, output_dir)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print(" Layer Offloading PoC — Summary")
    print("=" * 60)
    print(f"{'Window':>8}  {'GPU Peak MB':>12}  {'Latency ms':>12}  {'Load Avg ms':>12}")
    print("-" * 60)
    for w in args.windows:
        r = results[w]
        print(f"{w:>8}  {r.gpu_peak_mb:>12.0f}  {r.latency_ms:>12.0f}  {r.layer_load_avg_ms:>12.3f}")

    if full_gpu_mb is not None and full_gpu_mb != float("inf"):
        min_mem = min(r.gpu_peak_mb for r in results.values())
        print(f"\nFull model GPU: {full_gpu_mb:.0f} MB")
        print(f"Best offload (w=1): {min_mem:.0f} MB (saves {1 - min_mem/full_gpu_mb:.1%})")
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
