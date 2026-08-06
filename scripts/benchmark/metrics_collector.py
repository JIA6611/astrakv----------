"""Metrics helpers for AstraKV-W benchmark runs.

The collector is intentionally runtime-agnostic. It does not import vLLM,
LMCache, SGLang, or any third-party runtime code. Hardware-specific metrics use
best-effort probes and fall back to explicit unavailable values.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    cpu_rss_mb: float
    gpu_used_mb: float | None
    gpu_probe: str


@dataclass(frozen=True)
class DiskIOMetrics:
    write_mb: float
    read_mb: float
    write_seconds: float
    read_seconds: float
    write_mb_s: float
    read_mb_s: float


class MetricsCollector:
    """Collect process memory, GPU memory, and scratch SSD IO metrics."""

    def __init__(self, scratch_dir: str | Path):
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> MemorySnapshot:
        gpu_used, gpu_probe = self._gpu_memory_mb()
        return MemorySnapshot(
            cpu_rss_mb=self._cpu_rss_mb(),
            gpu_used_mb=gpu_used,
            gpu_probe=gpu_probe,
        )

    def sample_peak_memory(self, samples: Iterable[MemorySnapshot]) -> MemorySnapshot:
        samples = list(samples)
        if not samples:
            return self.snapshot()

        peak_cpu = max(item.cpu_rss_mb for item in samples)
        gpu_values = [item.gpu_used_mb for item in samples if item.gpu_used_mb is not None]
        peak_gpu = max(gpu_values) if gpu_values else None
        probe = next((item.gpu_probe for item in samples if item.gpu_probe != "unavailable"), samples[-1].gpu_probe)
        return MemorySnapshot(cpu_rss_mb=peak_cpu, gpu_used_mb=peak_gpu, gpu_probe=probe)

    def measure_scratch_io(self, size_mb: int, block_mb: int = 1) -> DiskIOMetrics:
        """Write and read a temporary file to estimate local scratch IO.

        The file is created under results scratch space and removed afterwards.
        """

        size_bytes = max(1, int(size_mb)) * BYTES_PER_MB
        block_bytes = max(1, int(block_mb)) * BYTES_PER_MB
        buffer = b"\x5a" * min(block_bytes, size_bytes)

        fd, path_str = tempfile.mkstemp(prefix="astrakv_io_", suffix=".bin", dir=self.scratch_dir)
        path = Path(path_str)
        try:
            written = 0
            write_start = time.perf_counter()
            with os.fdopen(fd, "wb", buffering=0) as handle:
                while written < size_bytes:
                    chunk = buffer[: min(len(buffer), size_bytes - written)]
                    handle.write(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            write_seconds = max(time.perf_counter() - write_start, 1e-9)

            read_bytes = 0
            read_start = time.perf_counter()
            with path.open("rb", buffering=0) as handle:
                while True:
                    chunk = handle.read(block_bytes)
                    if not chunk:
                        break
                    read_bytes += len(chunk)
            read_seconds = max(time.perf_counter() - read_start, 1e-9)

            write_mb = written / BYTES_PER_MB
            read_mb = read_bytes / BYTES_PER_MB
            return DiskIOMetrics(
                write_mb=write_mb,
                read_mb=read_mb,
                write_seconds=write_seconds,
                read_seconds=read_seconds,
                write_mb_s=write_mb / write_seconds,
                read_mb_s=read_mb / read_seconds,
            )
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _cpu_rss_mb() -> float:
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

    @staticmethod
    def _gpu_memory_mb() -> tuple[float | None, str]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            values = [float(line.strip()) for line in output.splitlines() if line.strip()]
            if values:
                return sum(values), "nvidia-smi"
            return None, "nvidia-smi-empty"
        except Exception:
            return None, "unavailable"


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
