"""DGX-oriented metrics sampler for real AstraKV-W endpoint benchmarks."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class DgxSample:
    timestamp_s: float
    cpu_rss_mb: float
    gpu_used_mb: float | None
    gpu_util_pct: float | None
    disk_read_mb: float | None
    disk_write_mb: float | None
    cpu_util_pct: float | None = None
    pss_mb: float | None = None
    minor_faults: int | None = None
    major_faults: int | None = None
    voluntary_ctx_switches: int | None = None
    involuntary_ctx_switches: int | None = None
    process_read_mb: float | None = None
    process_write_mb: float | None = None
    disk_read_iops: float | None = None
    disk_write_iops: float | None = None
    disk_queue_ms: float | None = None
    disk_read_latency_ms: float | None = None
    disk_write_latency_ms: float | None = None
    gpu_process_used_mb: float | None = None
    run_id: str = ""
    case: str = ""
    request_id: str = ""
    request_started_s: float | None = None
    request_ended_s: float | None = None
    sample_path: str = ""
    active_request_ids: tuple[str, ...] = ()
    shared_request_ids: tuple[str, ...] = ()
    shared_boundary_ids: tuple[str, ...] = ()
    attribution_mode: str = "case_boundary"


@dataclass(frozen=True)
class DgxSummary:
    cpu_rss_peak_mb: float
    gpu_used_peak_mb: float | None
    gpu_util_peak_pct: float | None
    disk_read_delta_mb: float | None
    disk_write_delta_mb: float | None
    sample_count: int
    gpu_probe: str
    disk_probe: str


class DgxMetricsCollector:
    """Samples host, GPU, and disk counters while real requests are running."""

    def __init__(
        self,
        output_csv: str | Path,
        interval_seconds: float = 0.5,
        disk_device: str = "",
        process_name_filters: Iterable[str] | None = None,
        run_id: str = "",
        case: str = "",
    ) -> None:
        self.output_csv = Path(output_csv)
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.disk_device = disk_device
        self.process_name_filters = tuple(process_name_filters or ())
        self.run_id = str(run_id)
        self.case = str(case)
        self.samples: list[DgxSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_probe = "unavailable"
        self.disk_probe = "unavailable"
        self._previous_disk_extended: tuple[float, tuple[int, int, int, int, int]] | None = None
        self._attribution_lock = threading.RLock()
        self._active_requests: dict[str, int] = {}
        self._request_windows: dict[str, tuple[float, float | None]] = {}
        self._shared_boundaries: dict[int, tuple[str, ...]] = {}
        self._next_shared_boundary_token = 0

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dgx-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> DgxSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
            self._thread = None
        self._write_csv()
        return self.summary()

    def summary(self) -> DgxSummary:
        if not self.samples:
            return DgxSummary(0.0, None, None, None, None, 0, self.gpu_probe, self.disk_probe)

        cpu_peak = max(sample.cpu_rss_mb for sample in self.samples)
        gpu_values = [sample.gpu_used_mb for sample in self.samples if sample.gpu_used_mb is not None]
        gpu_util_values = [
            sample.gpu_util_pct for sample in self.samples if sample.gpu_util_pct is not None
        ]
        read_values = [sample.disk_read_mb for sample in self.samples if sample.disk_read_mb is not None]
        write_values = [
            sample.disk_write_mb for sample in self.samples if sample.disk_write_mb is not None
        ]
        read_delta = (read_values[-1] - read_values[0]) if len(read_values) >= 2 else None
        write_delta = (write_values[-1] - write_values[0]) if len(write_values) >= 2 else None
        return DgxSummary(
            cpu_rss_peak_mb=cpu_peak,
            gpu_used_peak_mb=max(gpu_values) if gpu_values else None,
            gpu_util_peak_pct=max(gpu_util_values) if gpu_util_values else None,
            disk_read_delta_mb=read_delta,
            disk_write_delta_mb=write_delta,
            sample_count=len(self.samples),
            gpu_probe=self.gpu_probe,
            disk_probe=self.disk_probe,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_seconds)

    def _sample(self) -> DgxSample:
        with self._attribution_lock:
            active_request_ids = tuple(sorted(self._active_requests))
            active_boundaries = tuple(sorted(self._shared_boundaries.items()))
            shared_boundary_ids = tuple(f"shared-batch-{token}" for token, _ in active_boundaries)
            shared_request_ids = tuple(sorted({
                request_id
                for _, boundary in active_boundaries
                for request_id in boundary
            }))
            has_independent_active_request = bool(shared_boundary_ids) and bool(set(active_request_ids) - set(shared_request_ids))
            if not active_request_ids:
                request_id = ""
                attribution_mode = "case_boundary"
                shared_request_ids = ()
                shared_boundary_ids = ()
            elif len(shared_boundary_ids) > 1 or has_independent_active_request:
                request_id = ""
                attribution_mode = "shared_batch_ambiguous"
            elif shared_request_ids:
                request_id = ""
                attribution_mode = "shared_batch"
            elif len(active_request_ids) == 1:
                request_id = active_request_ids[0]
                attribution_mode = "exclusive_request"
            else:
                request_id = ""
                attribution_mode = "case_boundary"
            window_ids = (request_id,) if request_id else (shared_request_ids or active_request_ids)
            windows = [
                window
                for identity in window_ids
                if (window := self._request_windows.get(identity)) is not None
            ]
        request_started_s = min((window[0] for window in windows), default=None)
        request_ended_s = max((window[1] for window in windows), default=None) if windows and all(window[1] is not None for window in windows) else None
        resource = self._sample_resource_metrics()
        return DgxSample(
            timestamp_s=time.time(),
            run_id=self.run_id,
            case=self.case,
            request_id=request_id,
            request_started_s=request_started_s,
            request_ended_s=request_ended_s,
            sample_path=str(self.output_csv),
            active_request_ids=active_request_ids,
            shared_request_ids=shared_request_ids,
            shared_boundary_ids=shared_boundary_ids,
            attribution_mode=attribution_mode,
            **resource,
        )

    def _sample_resource_metrics(self) -> dict[str, float | int | None]:
        gpu_used, gpu_util, gpu_probe = _gpu_metrics()
        disk_read, disk_write, disk_probe = _disk_metrics(self.disk_device)
        disk_extra = self._disk_extra()
        self.gpu_probe = gpu_probe
        self.disk_probe = disk_probe
        process = _process_diagnostics(self.process_name_filters)
        process["gpu_process_used_mb"] = _gpu_process_memory_mb(self.process_name_filters)
        return {
            "cpu_rss_mb": _cpu_rss_mb(self.process_name_filters),
            "gpu_used_mb": gpu_used,
            "gpu_util_pct": gpu_util,
            "disk_read_mb": disk_read,
            "disk_write_mb": disk_write,
            **process,
            **disk_extra,
        }

    def active_request_ids(self) -> tuple[str, ...]:
        with self._attribution_lock:
            return tuple(sorted(self._active_requests))

    @contextmanager
    def request_scope(self, request_id: int | str) -> Iterator[None]:
        identity = str(request_id)
        with self._attribution_lock:
            count = self._active_requests.get(identity, 0)
            if count == 0:
                self._request_windows[identity] = (time.time(), None)
            self._active_requests[identity] = count + 1
        try:
            yield
        finally:
            with self._attribution_lock:
                count = self._active_requests.get(identity, 0) - 1
                if count > 0:
                    self._active_requests[identity] = count
                else:
                    self._active_requests.pop(identity, None)
                    self._request_windows.pop(identity, None)

    @contextmanager
    def shared_batch_scope(self, request_ids: Iterable[int | str]) -> Iterator[None]:
        boundary = tuple(sorted({str(request_id) for request_id in request_ids}))
        if len(boundary) < 2:
            raise ValueError("shared batch attribution requires at least two request IDs")
        with self._attribution_lock:
            token = self._next_shared_boundary_token
            self._next_shared_boundary_token += 1
            self._shared_boundaries[token] = boundary
        try:
            yield
        finally:
            with self._attribution_lock:
                self._shared_boundaries.pop(token, None)

    def _write_csv(self) -> None:
        with self.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_s",
                    "run_id",
                    "case",
                    "request_id",
                    "request_started_s",
                    "request_ended_s",
                    "sample_path",
                    "active_request_ids",
                    "shared_request_ids",
                    "shared_boundary_ids",
                    "attribution_mode",
                    "cpu_rss_mb",
                    "gpu_used_mb",
                    "gpu_util_pct",
                    "disk_read_mb",
                    "disk_write_mb",
                    "cpu_util_pct", "pss_mb", "minor_faults", "major_faults", "voluntary_ctx_switches", "involuntary_ctx_switches", "process_read_mb", "process_write_mb",
                    "disk_read_iops", "disk_write_iops", "disk_queue_ms", "disk_read_latency_ms", "disk_write_latency_ms", "gpu_process_used_mb",
                ],
            )
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(
                    {
                        "timestamp_s": f"{sample.timestamp_s:.6f}",
                        "run_id": sample.run_id,
                        "case": sample.case,
                        "request_id": sample.request_id,
                        "request_started_s": "" if sample.request_started_s is None else f"{sample.request_started_s:.6f}",
                        "request_ended_s": "" if sample.request_ended_s is None else f"{sample.request_ended_s:.6f}",
                        "sample_path": sample.sample_path,
                        "active_request_ids": ",".join(sample.active_request_ids),
                        "shared_request_ids": ",".join(sample.shared_request_ids),
                        "shared_boundary_ids": ",".join(sample.shared_boundary_ids),
                        "attribution_mode": sample.attribution_mode,
                        "cpu_rss_mb": f"{sample.cpu_rss_mb:.6f}",
                        "gpu_used_mb": "" if sample.gpu_used_mb is None else f"{sample.gpu_used_mb:.6f}",
                        "gpu_util_pct": ""
                        if sample.gpu_util_pct is None
                        else f"{sample.gpu_util_pct:.6f}",
                        "disk_read_mb": ""
                        if sample.disk_read_mb is None
                        else f"{sample.disk_read_mb:.6f}",
                        "disk_write_mb": ""
                        if sample.disk_write_mb is None
                        else f"{sample.disk_write_mb:.6f}",
                        "cpu_util_pct": "" if sample.cpu_util_pct is None else f"{sample.cpu_util_pct:.6f}",
                        "pss_mb": "" if sample.pss_mb is None else f"{sample.pss_mb:.6f}",
                        "minor_faults": "" if sample.minor_faults is None else sample.minor_faults,
                        "major_faults": "" if sample.major_faults is None else sample.major_faults,
                        "voluntary_ctx_switches": "" if sample.voluntary_ctx_switches is None else sample.voluntary_ctx_switches,
                        "involuntary_ctx_switches": "" if sample.involuntary_ctx_switches is None else sample.involuntary_ctx_switches,
                        "process_read_mb": "" if sample.process_read_mb is None else f"{sample.process_read_mb:.6f}",
                        "process_write_mb": "" if sample.process_write_mb is None else f"{sample.process_write_mb:.6f}",
                        "disk_read_iops": "" if sample.disk_read_iops is None else f"{sample.disk_read_iops:.6f}",
                        "disk_write_iops": "" if sample.disk_write_iops is None else f"{sample.disk_write_iops:.6f}",
                        "disk_queue_ms": "" if sample.disk_queue_ms is None else f"{sample.disk_queue_ms:.6f}",
                        "disk_read_latency_ms": "" if sample.disk_read_latency_ms is None else f"{sample.disk_read_latency_ms:.6f}",
                        "disk_write_latency_ms": "" if sample.disk_write_latency_ms is None else f"{sample.disk_write_latency_ms:.6f}",
                        "gpu_process_used_mb": "" if sample.gpu_process_used_mb is None else f"{sample.gpu_process_used_mb:.6f}",
                    }
                )

    def _disk_extra(self) -> dict[str, float | None]:
        counters = _disk_extended_counters(self.disk_device)
        now = time.time()
        empty = {"disk_read_iops": None, "disk_write_iops": None, "disk_queue_ms": None, "disk_read_latency_ms": None, "disk_write_latency_ms": None}
        if counters is None or self._previous_disk_extended is None:
            self._previous_disk_extended = (now, counters) if counters is not None else None
            return empty
        previous_time, previous = self._previous_disk_extended
        self._previous_disk_extended = (now, counters)
        dt = max(0.001, now - previous_time)
        reads, writes, read_ms, write_ms, weighted_ms = (max(0, current - old) for current, old in zip(counters, previous))
        total_ios = reads + writes
        return {"disk_read_iops": reads / dt, "disk_write_iops": writes / dt, "disk_queue_ms": weighted_ms / max(1, total_ios), "disk_read_latency_ms": read_ms / max(1, reads), "disk_write_latency_ms": write_ms / max(1, writes)}


def _gpu_metrics() -> tuple[float | None, float | None, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        memory_values: list[float] = []
        util_values: list[float] = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                memory = _float_or_none(parts[0])
                util = _float_or_none(parts[1])
                if memory is not None:
                    memory_values.append(memory)
                if util is not None:
                    util_values.append(util)
        if memory_values:
            return sum(memory_values), max(util_values) if util_values else None, "nvidia-smi"
        nvml_memory, nvml_util, nvml_probe = _nvml_metrics()
        if nvml_memory is not None or nvml_util is not None:
            return nvml_memory, nvml_util, nvml_probe
        return None, None, f"nvidia-smi-empty/{nvml_probe}"
    except Exception:
        nvml_memory, nvml_util, nvml_probe = _nvml_metrics()
        if nvml_memory is not None or nvml_util is not None:
            return nvml_memory, nvml_util, nvml_probe
        return None, None, nvml_probe


def _nvml_metrics() -> tuple[float | None, float | None, str]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None, None, "nvml-unavailable"

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        count = int(pynvml.nvmlDeviceGetCount())
        memory_values: list[float] = []
        util_values: list[float] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_values.append(float(memory.used) / BYTES_PER_MB)
            except Exception:
                pass
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_values.append(float(utilization.gpu))
            except Exception:
                pass
        if memory_values or util_values:
            return (
                sum(memory_values) if memory_values else None,
                max(util_values) if util_values else None,
                "nvml",
            )
        return None, None, "nvml-empty"
    except Exception:
        return None, None, "nvml-unavailable"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _float_or_none(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A", "nan", "NaN"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _disk_metrics(device: str) -> tuple[float | None, float | None, str]:
    if platform.system().lower() != "linux":
        return None, None, "unavailable"
    diskstats = Path("/proc/diskstats")
    if not diskstats.exists():
        return None, None, "unavailable"
    try:
        rows = diskstats.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None, "unavailable"

    selected: list[list[str]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 14:
            continue
        name = parts[2]
        if device and name != device:
            continue
        if not device and _looks_like_whole_disk(name):
            selected.append(parts)
        elif device:
            selected.append(parts)

    if not selected:
        return None, None, "unavailable"

    sectors_read = sum(int(parts[5]) for parts in selected)
    sectors_written = sum(int(parts[9]) for parts in selected)
    # Linux diskstats sectors are conventionally 512 bytes.
    return (
        sectors_read * 512 / BYTES_PER_MB,
        sectors_written * 512 / BYTES_PER_MB,
        "/proc/diskstats",
    )


def _looks_like_whole_disk(name: str) -> bool:
    if name.startswith(("loop", "ram", "dm-")):
        return False
    if name.startswith("nvme") and "p" in name:
        return False
    if name.startswith(("sd", "vd", "xvd")) and name[-1].isdigit():
        return False
    return True


def _disk_extended_counters(device: str) -> tuple[int, int, int, int, int] | None:
    if platform.system().lower() != "linux" or not Path("/proc/diskstats").exists():
        return None
    selected: list[list[str]] = []
    for row in Path("/proc/diskstats").read_text(encoding="utf-8", errors="replace").splitlines():
        parts = row.split()
        if len(parts) < 14:
            continue
        if device and parts[2] != device:
            continue
        if device or _looks_like_whole_disk(parts[2]):
            selected.append(parts)
    if not selected:
        return None
    return (sum(int(parts[3]) for parts in selected), sum(int(parts[7]) for parts in selected), sum(int(parts[6]) for parts in selected), sum(int(parts[10]) for parts in selected), sum(int(parts[13]) for parts in selected))


def _cpu_rss_mb(filters: tuple[str, ...]) -> float:
    if not filters:
        return _current_process_rss_mb()

    try:
        import psutil  # type: ignore

        total = 0.0
        lowered = tuple(item.lower() for item in filters)
        for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if any(item in name or item in cmdline for item in lowered):
                    total += proc.info["memory_info"].rss / BYTES_PER_MB
            except Exception:
                continue
        return total if total > 0 else _current_process_rss_mb()
    except Exception:
        return _current_process_rss_mb()


def _current_process_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / BYTES_PER_MB
    except Exception:
        return 0.0


def _process_diagnostics(filters: tuple[str, ...]) -> dict[str, float | int | None]:
    empty: dict[str, float | int | None] = {"cpu_util_pct": None, "pss_mb": None, "minor_faults": None, "major_faults": None, "voluntary_ctx_switches": None, "involuntary_ctx_switches": None, "process_read_mb": None, "process_write_mb": None}
    try:
        import psutil  # type: ignore
        processes = [psutil.Process(os.getpid())]
        if filters:
            processes = []
            lowered = tuple(item.lower() for item in filters)
            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    text = f"{proc.info.get('name') or ''} {' '.join(proc.info.get('cmdline') or [])}".lower()
                    if any(item in text for item in lowered):
                        processes.append(proc)
                except Exception:
                    continue
        if not processes:
            return empty
        totals = {key: 0.0 for key in empty}
        found = {key: False for key in empty}
        for proc in processes:
            try:
                totals["cpu_util_pct"] += float(proc.cpu_percent(interval=None)); found["cpu_util_pct"] = True
                full = proc.memory_full_info(); pss = getattr(full, "pss", None)
                if pss is not None: totals["pss_mb"] += float(pss) / BYTES_PER_MB; found["pss_mb"] = True
                ctx = proc.num_ctx_switches(); totals["voluntary_ctx_switches"] += float(ctx.voluntary); totals["involuntary_ctx_switches"] += float(ctx.involuntary); found["voluntary_ctx_switches"] = found["involuntary_ctx_switches"] = True
                io = proc.io_counters(); totals["process_read_mb"] += float(io.read_bytes) / BYTES_PER_MB; totals["process_write_mb"] += float(io.write_bytes) / BYTES_PER_MB; found["process_read_mb"] = found["process_write_mb"] = True
                mem = proc.memory_info()
                if hasattr(mem, "pfaults"): totals["minor_faults"] += float(mem.pfaults); found["minor_faults"] = True
                if hasattr(mem, "pageins"): totals["major_faults"] += float(mem.pageins); found["major_faults"] = True
            except Exception:
                continue
        return {key: (int(totals[key]) if key.endswith("faults") or key.endswith("switches") else totals[key]) if found[key] else None for key in empty}
    except Exception:
        return empty


def _gpu_process_memory_mb(filters: tuple[str, ...]) -> float | None:
    if not filters:
        return None
    try:
        import psutil  # type: ignore
        lowered = tuple(item.lower() for item in filters)
        pids = set()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            text = f"{proc.info.get('name') or ''} {' '.join(proc.info.get('cmdline') or [])}".lower()
            if any(item in text for item in lowered):
                pids.add(str(proc.info["pid"]))
        if not pids:
            return None
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        return sum(float(parts[1].strip()) for line in output.splitlines() if (parts := line.split(",")) and len(parts) >= 2 and parts[0].strip() in pids)
    except Exception:
        return None
