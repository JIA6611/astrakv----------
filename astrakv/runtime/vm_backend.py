"""File-backed mmap virtual-memory backend helpers.

This module models OS-style demand loading with a backing file, page-like
access units, and optional software prefetch. It is intentionally independent
from vLLM, LMCache, CUDA, and the real benchmark runners so experiments and
future adapters can reuse the same VM evidence path.
"""

from __future__ import annotations

import csv
import ctypes
import json
import mmap
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class VMDemoConfig:
    file_size_mb: int = 16
    page_size_bytes: int = 4096
    access_count: int = 256
    pattern: str = "sequential"
    prefetch_window: int = 0
    seed: int = 3136859
    keep_backing_file: bool = False


@dataclass(frozen=True, slots=True)
class VMAccessRecord:
    step: int
    page_index: int
    byte_offset: int
    prefetched_before_access: bool
    first_touch: bool
    demand_fault_like: bool
    value: int
    latency_us: float
    rss_mb: float

    def to_record(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "page_index": self.page_index,
            "byte_offset": self.byte_offset,
            "prefetched_before_access": self.prefetched_before_access,
            "first_touch": self.first_touch,
            "demand_fault_like": self.demand_fault_like,
            "value": self.value,
            "latency_us": round(self.latency_us, 6),
            "rss_mb": round(self.rss_mb, 6),
        }


@dataclass(frozen=True, slots=True)
class VMDemoSummary:
    file_size_bytes: int
    page_size_bytes: int
    total_pages: int
    access_count: int
    unique_pages_accessed: int
    prefetch_window: int
    prefetched_page_count: int
    demand_fault_like_count: int
    first_touch_count: int
    avg_latency_us: float
    p95_latency_us: float
    max_latency_us: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_end_mb: float
    backing_file: str
    retained_backing_file: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def demand_fault_like_rate(self) -> float:
        return self.demand_fault_like_count / max(1, self.access_count)

    @property
    def prefetch_coverage_rate(self) -> float:
        return self.prefetched_page_count / max(1, self.unique_pages_accessed)

    def to_record(self) -> dict[str, Any]:
        return {
            "file_size_bytes": self.file_size_bytes,
            "page_size_bytes": self.page_size_bytes,
            "total_pages": self.total_pages,
            "access_count": self.access_count,
            "unique_pages_accessed": self.unique_pages_accessed,
            "prefetch_window": self.prefetch_window,
            "prefetched_page_count": self.prefetched_page_count,
            "prefetch_coverage_rate": round(self.prefetch_coverage_rate, 6),
            "demand_fault_like_count": self.demand_fault_like_count,
            "demand_fault_like_rate": round(self.demand_fault_like_rate, 6),
            "first_touch_count": self.first_touch_count,
            "avg_latency_us": round(self.avg_latency_us, 6),
            "p95_latency_us": round(self.p95_latency_us, 6),
            "max_latency_us": round(self.max_latency_us, 6),
            "rss_start_mb": round(self.rss_start_mb, 6),
            "rss_peak_mb": round(self.rss_peak_mb, 6),
            "rss_end_mb": round(self.rss_end_mb, 6),
            "backing_file": self.backing_file,
            "retained_backing_file": self.retained_backing_file,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class VMDemoResult:
    config: VMDemoConfig
    access_records: list[VMAccessRecord]
    summary: VMDemoSummary


class VirtualMemoryDemoRunner:
    def __init__(self, config: VMDemoConfig | None = None) -> None:
        self.config = config or VMDemoConfig()

    def run(self, backing_file: str | Path) -> VMDemoResult:
        path = Path(backing_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_size_bytes = max(1, int(self.config.file_size_mb)) * BYTES_PER_MB
        page_size = max(1, int(self.config.page_size_bytes))
        total_pages = max(1, file_size_bytes // page_size)
        access_order = build_access_order(
            total_pages=total_pages,
            access_count=self.config.access_count,
            pattern=self.config.pattern,
            seed=self.config.seed,
        )

        rss_start = cpu_rss_mb()
        touched: set[int] = set()
        prefetched: set[int] = set()
        records: list[VMAccessRecord] = []
        checksum = 0

        create_backing_file(path, file_size_bytes)
        try:
            with path.open("r+b") as handle:
                with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapping:
                    for step, page_index in enumerate(access_order):
                        self._prefetch(mapping, page_index, page_size, total_pages, prefetched, touched)
                        offset = page_index * page_size
                        first_touch = page_index not in touched
                        prefetched_before = page_index in prefetched
                        start = time.perf_counter()
                        value = mapping[offset]
                        latency_us = (time.perf_counter() - start) * 1_000_000
                        touched.add(page_index)
                        checksum = (checksum + int(value) + page_index) % 1_000_000_007
                        records.append(
                            VMAccessRecord(
                                step=step,
                                page_index=page_index,
                                byte_offset=offset,
                                prefetched_before_access=prefetched_before,
                                first_touch=first_touch,
                                demand_fault_like=first_touch and not prefetched_before,
                                value=int(value),
                                latency_us=latency_us,
                                rss_mb=cpu_rss_mb(),
                            )
                        )
        finally:
            retained = self.config.keep_backing_file
            if not retained:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        summary = summarize_records(
            records,
            config=self.config,
            file_size_bytes=file_size_bytes,
            total_pages=total_pages,
            prefetched_pages=prefetched,
            backing_file=path,
            retained_backing_file=self.config.keep_backing_file,
            rss_start_mb=rss_start,
            metadata={"checksum": checksum, "pattern": self.config.pattern},
        )
        return VMDemoResult(config=self.config, access_records=records, summary=summary)

    def _prefetch(
        self,
        mapping: mmap.mmap,
        page_index: int,
        page_size: int,
        total_pages: int,
        prefetched: set[int],
        touched: set[int],
    ) -> None:
        if self.config.prefetch_window <= 0:
            return
        for next_page in range(page_index + 1, min(total_pages, page_index + 1 + self.config.prefetch_window)):
            if next_page in touched or next_page in prefetched:
                continue
            _ = mapping[next_page * page_size]
            prefetched.add(next_page)
            touched.add(next_page)


def build_access_order(total_pages: int, access_count: int, pattern: str, seed: int) -> list[int]:
    count = max(1, int(access_count))
    pages = max(1, int(total_pages))
    if pattern == "sequential":
        return [index % pages for index in range(count)]
    if pattern == "reverse":
        return [(pages - 1 - index) % pages for index in range(count)]
    if pattern == "stride":
        stride = max(1, pages // max(1, min(pages, count // 2 or 1)))
        return [(index * stride) % pages for index in range(count)]
    if pattern == "random":
        rng = random.Random(seed)
        return [rng.randrange(pages) for _ in range(count)]
    raise ValueError(f"Unsupported access pattern: {pattern}")


def create_backing_file(path: Path, size_bytes: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size_bytes)
        # Write sparse anchors so the file exists and deterministic byte values
        # are visible without filling large files in local smoke tests.
        step = max(1, size_bytes // 16)
        for offset in range(0, size_bytes, step):
            handle.seek(offset)
            handle.write(bytes([(offset // step) % 251]))
        handle.flush()
        os.fsync(handle.fileno())


def summarize_records(
    records: list[VMAccessRecord],
    *,
    config: VMDemoConfig,
    file_size_bytes: int,
    total_pages: int,
    prefetched_pages: set[int],
    backing_file: Path,
    retained_backing_file: bool,
    rss_start_mb: float,
    metadata: dict[str, Any],
) -> VMDemoSummary:
    latencies = [record.latency_us for record in records]
    rss_values = [record.rss_mb for record in records]
    accessed_pages = {record.page_index for record in records}
    prefetched_accessed_pages = accessed_pages.intersection(prefetched_pages)
    return VMDemoSummary(
        file_size_bytes=file_size_bytes,
        page_size_bytes=config.page_size_bytes,
        total_pages=total_pages,
        access_count=len(records),
        unique_pages_accessed=len(accessed_pages),
        prefetch_window=config.prefetch_window,
        prefetched_page_count=len(prefetched_accessed_pages),
        demand_fault_like_count=sum(1 for record in records if record.demand_fault_like),
        first_touch_count=sum(1 for record in records if record.first_touch),
        avg_latency_us=statistics.mean(latencies) if latencies else 0.0,
        p95_latency_us=percentile(latencies, 95.0),
        max_latency_us=max(latencies) if latencies else 0.0,
        rss_start_mb=rss_start_mb,
        rss_peak_mb=max([rss_start_mb, *rss_values]) if rss_values else rss_start_mb,
        rss_end_mb=rss_values[-1] if rss_values else rss_start_mb,
        backing_file=str(backing_file),
        retained_backing_file=retained_backing_file,
        metadata=metadata,
    )


def write_access_trace_csv(path: str | Path, records: Iterable[VMAccessRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_record() for record in records]
    fieldnames = [
        "step",
        "page_index",
        "byte_offset",
        "prefetched_before_access",
        "first_touch",
        "demand_fault_like",
        "value",
        "latency_us",
        "rss_mb",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: str | Path, summary: VMDemoSummary) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_record(), indent=2, ensure_ascii=False), encoding="utf-8")


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cpu_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / BYTES_PER_MB
    except Exception:
        pass

    if platform.system().lower() == "windows":
        return _windows_rss_mb()

    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system().lower() == "darwin":
            return rss / BYTES_PER_MB
        return rss / 1024.0
    except Exception:
        return 0.0


def _windows_rss_mb() -> float:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return 0.0
    return float(counters.WorkingSetSize) / BYTES_PER_MB
