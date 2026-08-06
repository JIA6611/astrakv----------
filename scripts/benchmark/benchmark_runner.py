"""AstraKV-W benchmark runner.

This runner provides a complete benchmark infrastructure with a synthetic
baseline backend. It is deliberately isolated from third-party runtimes: it
does not import or modify vLLM, LMCache, FlashAttention, llama.cpp, SGLang, or
TensorRT-LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback is for minimal environments.
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics_collector import DiskIOMetrics, MemorySnapshot, MetricsCollector
from scripts.reporting.plot_benchmarks import plot_csv

from astrakv.prefetch import KVBlockRef, SelectiveKVPrefetchConfig, SelectiveKVPrefetchMVP


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch_size: int
    context_length: int
    output_tokens: int
    repeat: int


@dataclass
class RequestMetrics:
    ttft_ms: float
    tpot_ms: float
    total_time_ms: float
    output_tokens: int


@dataclass
class SyntheticRunResult:
    request_metrics: list[RequestMetrics]
    wall_time_seconds: float
    total_output_tokens: int
    kv_cache_hits: int
    kv_cache_lookups: int
    prefetch_hits: int
    prefetch_lookups: int
    prefetch_waste_rate: float
    gpu_kv_peak_bytes: int
    gpu_kv_full_bytes: int
    memory_samples: list[MemorySnapshot]


class SyntheticKVCache:
    """Small LRU cache used to produce baseline cache hit-rate metrics."""

    def __init__(self, capacity: int, prefetch_distance: int):
        self.capacity = max(1, capacity)
        self.prefetch_distance = max(0, prefetch_distance)
        self.cache: OrderedDict[str, None] = OrderedDict()
        self.prefetch: set[str] = set()

    def lookup(self, key: str, next_keys: list[str]) -> tuple[bool, bool]:
        kv_hit = key in self.cache
        prefetch_hit = key in self.prefetch
        if kv_hit:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = None
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

        self.prefetch.discard(key)
        for future_key in next_keys[: self.prefetch_distance]:
            if future_key not in self.cache:
                self.prefetch.add(future_key)
        return kv_hit, prefetch_hit


class SyntheticBackend:
    """Runtime-free baseline that simulates prefill/decode timing."""

    def __init__(self, config: dict[str, Any], collector: MetricsCollector):
        synthetic = config.get("synthetic_backend", {})
        cache_cfg = config.get("cache", {})
        self.collector = collector
        self.prefill_ms_per_token = float(synthetic.get("prefill_ms_per_token", 0.018))
        self.decode_ms_per_token = float(synthetic.get("decode_ms_per_token", 0.9))
        self.batch_overhead_ms = float(synthetic.get("batch_overhead_ms", 1.5))
        self.jitter_ratio = float(synthetic.get("deterministic_jitter_ratio", 0.04))
        self.block_size_bytes = int(
            config.get("prefetch_mvp", {}).get("block_size_bytes", 16 * 1024 * 1024)
        )
        self.cache = SyntheticKVCache(
            capacity=int(cache_cfg.get("kv_cache_capacity", 16)),
            prefetch_distance=int(cache_cfg.get("prefetch_distance", 1)),
        )

    def run_case(self, case: BenchmarkCase) -> SyntheticRunResult:
        request_keys = [_request_key(case, index) for index in range(case.batch_size)]
        request_metrics: list[RequestMetrics] = []
        memory_samples: list[MemorySnapshot] = [self.collector.snapshot()]
        kv_hits = 0
        prefetch_hits = 0
        wall_start = time.perf_counter()

        for index, key in enumerate(request_keys):
            next_keys = request_keys[index + 1 :]
            kv_hit, prefetch_hit = self.cache.lookup(key, next_keys)
            kv_hits += int(kv_hit)
            prefetch_hits += int(prefetch_hit)

            prefill_tokens = max(1, case.context_length)
            cache_discount = 0.35 if kv_hit else 1.0
            prefill_ms = (
                self.batch_overhead_ms
                + prefill_tokens * self.prefill_ms_per_token * cache_discount
                + _stable_jitter_ms(key, self.jitter_ratio)
            )
            decode_ms = max(1, case.output_tokens) * self.decode_ms_per_token
            total_ms = prefill_ms + decode_ms
            ttft_ms = prefill_ms
            tpot_ms = decode_ms / max(1, case.output_tokens)

            _sleep_ms(total_ms)
            request_metrics.append(
                RequestMetrics(
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    total_time_ms=total_ms,
                    output_tokens=case.output_tokens,
                )
            )
            memory_samples.append(self.collector.snapshot())

        wall_time = max(time.perf_counter() - wall_start, 1e-9)
        return SyntheticRunResult(
            request_metrics=request_metrics,
            wall_time_seconds=wall_time,
            total_output_tokens=sum(item.output_tokens for item in request_metrics),
            kv_cache_hits=kv_hits,
            kv_cache_lookups=max(1, len(request_keys)),
            prefetch_hits=prefetch_hits,
            prefetch_lookups=max(1, len(request_keys)),
            prefetch_waste_rate=0.0,
            gpu_kv_peak_bytes=len(set(request_keys)) * self.block_size_bytes,
            gpu_kv_full_bytes=len(set(request_keys)) * self.block_size_bytes,
            memory_samples=memory_samples,
        )


class SelectivePrefetchBenchmarkBackend:
    """Synthetic decode benchmark for the Selective KV Prefetch MVP."""

    def __init__(self, config: dict[str, Any], collector: MetricsCollector):
        synthetic = config.get("synthetic_backend", {})
        prefetch_cfg = config.get("prefetch_mvp", {})
        self.collector = collector
        self.prefill_ms_per_token = float(synthetic.get("prefill_ms_per_token", 0.018))
        self.decode_compute_ms_per_token = float(synthetic.get("decode_ms_per_token", 0.9))
        self.batch_overhead_ms = float(synthetic.get("batch_overhead_ms", 1.5))
        self.jitter_ratio = float(synthetic.get("deterministic_jitter_ratio", 0.04))
        self.block_size_bytes = int(prefetch_cfg.get("block_size_bytes", 16 * 1024 * 1024))
        self.blocks_per_context = int(prefetch_cfg.get("blocks_per_context", 8))
        self.reuse_stride = int(prefetch_cfg.get("reuse_stride", 4))
        self.config = SelectiveKVPrefetchConfig(
            gpu_capacity_blocks=int(prefetch_cfg.get("gpu_capacity_blocks", 4)),
            block_size_bytes=self.block_size_bytes,
            prefetch_window=int(prefetch_cfg.get("prefetch_window", 2)),
            max_queue_size=int(prefetch_cfg.get("max_queue_size", 16)),
            prefetch_latency_ms=float(prefetch_cfg.get("prefetch_latency_ms", 0.15)),
            cpu_miss_latency_ms=float(prefetch_cfg.get("cpu_miss_latency_ms", 1.2)),
            gpu_hit_latency_ms=float(prefetch_cfg.get("gpu_hit_latency_ms", 0.02)),
            prefetch_hit_latency_ms=float(prefetch_cfg.get("prefetch_hit_latency_ms", 0.03)),
        )

    def run_case_pair(self, case: BenchmarkCase) -> tuple[SyntheticRunResult, SyntheticRunResult]:
        trace = self._decode_trace(case)
        baseline = asyncio.run(self._run_case(case, trace, enable_prefetch=False))
        selective = asyncio.run(self._run_case(case, trace, enable_prefetch=True))
        return baseline, selective

    async def _run_case(
        self,
        case: BenchmarkCase,
        trace: list[str],
        enable_prefetch: bool,
    ) -> SyntheticRunResult:
        engine = SelectiveKVPrefetchMVP(self.config)
        engine.add_cpu_blocks(
            KVBlockRef(block_id=block_id, size_bytes=self.block_size_bytes)
            for block_id in sorted(set(trace))
        )
        await engine.start()

        memory_samples: list[MemorySnapshot] = [self.collector.snapshot()]
        wall_start = time.perf_counter()
        request_metrics: list[RequestMetrics] = []
        total_decode_ms = 0.0

        prefill_ms = (
            self.batch_overhead_ms
            + max(1, case.context_length) * self.prefill_ms_per_token
            + _stable_jitter_ms(case.name, self.jitter_ratio)
        )

        for position, block_id in enumerate(trace):
            if enable_prefetch:
                await engine.submit_predictions(trace, position)
                await asyncio.sleep(self.config.prefetch_latency_ms / 1000.0)
            access = await engine.access(block_id)
            total_decode_ms += self.decode_compute_ms_per_token + access.latency_ms
            memory_samples.append(self.collector.snapshot())

        await engine.close()
        wall_time = max(time.perf_counter() - wall_start, 1e-9)
        total_output_tokens = case.output_tokens * case.batch_size
        tpot_ms = total_decode_ms / max(1, total_output_tokens)
        total_time_ms = prefill_ms + total_decode_ms

        for _ in range(case.batch_size):
            request_metrics.append(
                RequestMetrics(
                    ttft_ms=prefill_ms,
                    tpot_ms=tpot_ms,
                    total_time_ms=total_time_ms,
                    output_tokens=case.output_tokens,
                )
            )

        metrics = engine.metrics
        full_gpu_bytes = len(set(trace)) * self.block_size_bytes
        return SyntheticRunResult(
            request_metrics=request_metrics,
            wall_time_seconds=wall_time,
            total_output_tokens=total_output_tokens,
            kv_cache_hits=metrics.gpu_hits + metrics.prefetch_hits,
            kv_cache_lookups=max(1, metrics.demand_lookups),
            prefetch_hits=metrics.prefetch_hits,
            prefetch_lookups=max(1, metrics.demand_lookups),
            prefetch_waste_rate=metrics.prefetch_waste_rate,
            gpu_kv_peak_bytes=metrics.gpu_bytes_peak,
            gpu_kv_full_bytes=full_gpu_bytes,
            memory_samples=memory_samples,
        )

    def _decode_trace(self, case: BenchmarkCase) -> list[str]:
        block_count = max(1, math.ceil(case.context_length / max(1, case.context_length // self.blocks_per_context)))
        base_blocks = [f"{case.name}:block:{idx}" for idx in range(block_count)]
        trace: list[str] = []
        for token_idx in range(case.output_tokens):
            for batch_idx in range(case.batch_size):
                cursor = (token_idx + batch_idx) % len(base_blocks)
                if token_idx > 0 and token_idx % max(1, self.reuse_stride) == 0:
                    cursor = max(0, cursor - 1)
                trace.append(base_blocks[cursor])
        return trace


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark config: {config_path}")
    return data


def build_cases(config: dict[str, Any]) -> list[BenchmarkCase]:
    matrix = config.get("matrix", {})
    batch_sizes = [int(item) for item in matrix.get("batch_sizes", [1])]
    context_lengths = [int(item) for item in matrix.get("context_lengths", [128])]
    output_tokens = int(matrix.get("output_tokens", 16))
    repeat = int(matrix.get("repeat", 1))

    cases: list[BenchmarkCase] = []
    for batch_size in batch_sizes:
        for context_length in context_lengths:
            cases.append(
                BenchmarkCase(
                    name=f"bs{batch_size}_ctx{context_length}",
                    batch_size=batch_size,
                    context_length=context_length,
                    output_tokens=output_tokens,
                    repeat=repeat,
                )
            )
    return cases


def run_benchmark(config: dict[str, Any], output_root: str | Path) -> dict[str, Path]:
    run_name = str(config.get("run_name", "baseline_synthetic"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{timestamp}"
    charts_dir = run_dir / "charts"
    scratch_dir = run_dir / "scratch"
    run_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    collector = MetricsCollector(scratch_dir)
    benchmark_kind = str(config.get("benchmark", {}).get("type", "synthetic"))
    synthetic_backend = SyntheticBackend(config, collector)
    selective_backend = SelectivePrefetchBenchmarkBackend(config, collector)
    rows: list[dict[str, Any]] = []
    disk_cfg = config.get("disk_io", {})
    disk_size_mb = int(disk_cfg.get("scratch_file_mb", 16))
    disk_block_mb = int(disk_cfg.get("block_mb", 1))

    for case in build_cases(config):
        repeat_rows = []
        for repeat_index in range(case.repeat):
            disk = collector.measure_scratch_io(disk_size_mb, disk_block_mb)
            if benchmark_kind == "selective_prefetch_mvp":
                baseline_result, selective_result = selective_backend.run_case_pair(case)
                baseline_peak = collector.sample_peak_memory(baseline_result.memory_samples)
                selective_peak = collector.sample_peak_memory(selective_result.memory_samples)
                baseline_row = summarize_case(
                    case,
                    repeat_index,
                    baseline_result,
                    baseline_peak,
                    disk,
                    backend_name="synthetic_no_prefetch",
                )
                selective_row = summarize_case(
                    case,
                    repeat_index,
                    selective_result,
                    selective_peak,
                    disk,
                    backend_name="selective_prefetch_mvp",
                    baseline=baseline_row,
                )
                rows.extend([baseline_row, selective_row])
                repeat_rows.extend([baseline_row, selective_row])
            else:
                run_result = synthetic_backend.run_case(case)
                peak = collector.sample_peak_memory(run_result.memory_samples)
                row = summarize_case(
                    case,
                    repeat_index,
                    run_result,
                    peak,
                    disk,
                    backend_name="synthetic",
                )
                rows.append(row)
                repeat_rows.append(row)
        _ = repeat_rows

    csv_path = run_dir / "benchmark_results.csv"
    write_csv(csv_path, rows)
    chart_paths = plot_csv(csv_path, charts_dir)
    md_path = run_dir / "benchmark_report.md"
    write_markdown(md_path, rows, chart_paths, config)
    config_copy = run_dir / "benchmark_config.json"
    config_copy.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "csv": csv_path,
        "markdown": md_path,
        "config": config_copy,
        "charts_dir": charts_dir,
    }


def summarize_case(
    case: BenchmarkCase,
    repeat_index: int,
    run_result: SyntheticRunResult,
    peak: MemorySnapshot,
    disk: DiskIOMetrics,
    backend_name: str,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reqs = run_result.request_metrics
    ttft = [item.ttft_ms for item in reqs]
    tpot = [item.tpot_ms for item in reqs]
    total_time = [item.total_time_ms for item in reqs]
    throughput = run_result.total_output_tokens / run_result.wall_time_seconds
    ttft_ms = _mean(ttft)
    tpot_ms = _mean(tpot)
    gpu_kv_peak_mb = run_result.gpu_kv_peak_bytes / (1024 * 1024)
    gpu_kv_full_mb = run_result.gpu_kv_full_bytes / (1024 * 1024)
    gpu_reduction = 1.0 - (run_result.gpu_kv_peak_bytes / max(1, run_result.gpu_kv_full_bytes))
    row = {
        "case": case.name,
        "repeat": repeat_index,
        "backend": backend_name,
        "batch_size": case.batch_size,
        "context_length": case.context_length,
        "output_tokens": case.output_tokens,
        "request_count": len(reqs),
        "ttft_ms": ttft_ms,
        "ttft_p50_ms": _quantile(ttft, 0.50),
        "ttft_p95_ms": _quantile(ttft, 0.95),
        "tpot_ms": tpot_ms,
        "tpot_p50_ms": _quantile(tpot, 0.50),
        "tpot_p95_ms": _quantile(tpot, 0.95),
        "latency_ms": _mean(total_time),
        "throughput_tokens_s": throughput,
        "cpu_memory_peak_mb": peak.cpu_rss_mb,
        "gpu_memory_peak_mb": peak.gpu_used_mb if peak.gpu_used_mb is not None else "",
        "gpu_probe": peak.gpu_probe,
        "ssd_write_mb": disk.write_mb,
        "ssd_read_mb": disk.read_mb,
        "ssd_write_mb_s": disk.write_mb_s,
        "ssd_read_mb_s": disk.read_mb_s,
        "kv_cache_hits": run_result.kv_cache_hits,
        "kv_cache_lookups": run_result.kv_cache_lookups,
        "kv_cache_hit_rate": run_result.kv_cache_hits / max(1, run_result.kv_cache_lookups),
        "prefetch_hits": run_result.prefetch_hits,
        "prefetch_lookups": run_result.prefetch_lookups,
        "prefetch_hit_rate": run_result.prefetch_hits / max(1, run_result.prefetch_lookups),
        "prefetch_waste_rate": run_result.prefetch_waste_rate,
        "gpu_kv_peak_mb": gpu_kv_peak_mb,
        "gpu_kv_full_mb": gpu_kv_full_mb,
        "gpu_memory_reduction_pct": gpu_reduction * 100.0,
        "ttft_change_pct": 0.0,
        "tpot_change_pct": 0.0,
        "wall_time_seconds": run_result.wall_time_seconds,
    }
    if baseline is not None:
        row["ttft_change_pct"] = _pct_change(ttft_ms, float(baseline["ttft_ms"]))
        row["tpot_change_pct"] = _pct_change(tpot_ms, float(baseline["tpot_ms"]))
    return row


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No benchmark rows to write.")
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: str | Path,
    rows: list[dict[str, Any]],
    chart_paths: list[Path],
    config: dict[str, Any],
) -> None:
    lines = [
        "# AstraKV-W Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Backend: synthetic benchmark mode. No third-party runtime or scheduler was modified.",
        "",
        "## Summary",
        "",
        "| case | backend | batch | context | TTFT ms | TTFT change % | TPOT ms | TPOT change % | throughput tok/s | CPU MB | GPU KV MB | GPU reduction % | KV hit | Prefetch hit | Prefetch waste |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {backend} | {batch_size} | {context_length} | {ttft_ms:.3f} | "
            "{ttft_change_pct:.3f} | {tpot_ms:.3f} | {tpot_change_pct:.3f} | "
            "{throughput_tokens_s:.3f} | {cpu_memory_peak_mb:.3f} | {gpu_kv_peak_mb:.3f} | "
            "{gpu_memory_reduction_pct:.3f} | {kv_cache_hit_rate:.3f} | "
            "{prefetch_hit_rate:.3f} | {prefetch_waste_rate:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- TTFT: mean time to first token per request.",
            "- TPOT: mean decode time per output token.",
            "- Throughput: generated output tokens divided by measured wall time.",
            "- GPU memory: best-effort `nvidia-smi` total used memory across visible GPUs; blank means unavailable.",
            "- CPU memory: process RSS peak during the case.",
            "- SSD read/write: temporary scratch-file read/write bandwidth under the run directory.",
            "- KV cache hit rate and prefetch hit rate: synthetic decode trace counters.",
            "- Prefetch waste rate: completed prefetched blocks that were evicted or left unused.",
            "- TTFT/TPOT change: percentage change versus the same case without selective prefetch.",
            "- GPU memory reduction: estimated KV memory reduction versus keeping all referenced KV blocks resident on GPU.",
            "",
            "## Charts",
            "",
        ]
    )
    for chart in chart_paths:
        rel = chart.relative_to(Path(path).parent).as_posix()
        lines.append(f"- ![{chart.stem}]({rel})")

    lines.extend(["", "## Config", "", "```json", json.dumps(config, indent=2), "```", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _request_key(case: BenchmarkCase, index: int) -> str:
    bucket = index % max(1, math.ceil(case.batch_size / 2))
    raw = f"{case.context_length}:{bucket}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _stable_jitter_ms(key: str, ratio: float) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:6], 16) / float(0xFFFFFF)
    centered = value - 0.5
    return centered * ratio


def _sleep_ms(ms: float) -> None:
    time.sleep(max(0.0, ms / 1000.0))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0


def _fmt_optional(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AstraKV-W benchmark baseline.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "benchmarks" / "configs" / "baseline.yaml"),
        help="Benchmark config path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Output root directory.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_benchmark(config, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
