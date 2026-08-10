"""OS page-cache residency evidence for the AstraKV-W KV-Core runs.

The collector answers a question the LMCache tiering numbers cannot: which
pages of the file-backed disk store are actually resident in physical memory,
and how does residency evolve under prefetch (``madvise(MADV_WILLNEED)``) and
eviction (``madvise(MADV_DONTNEED)``) pressure.  Each sample pairs mincore
residency with cgroup-v2 ``memory.current`` and process RSS so the report can
show OS-level evidence alongside the logical CPU/SSD tier occupancy.

The module is a measurement companion, not part of the request path.  On
non-Linux hosts (or kernels without the needed syscalls) it degrades to
``mincore_status="unsupported"`` records instead of crashing, so the same
tooling can be smoke-tested on the development machine and run for real on the
DGX host.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.uma_metrics import _rss_bytes, current_cgroup_memory_evidence


PAGE_CACHE_EVIDENCE_SCHEMA = "astrakv-os-page-cache-evidence-v1"

_LINUX = sys.platform.startswith("linux")
_LIBC = None
if _LINUX:
    try:
        import ctypes

        _LIBC = ctypes.CDLL(None, use_errno=True)
    except (OSError, AttributeError):
        _LIBC = None

_OS_MADVISE = getattr(os, "madvise", None)


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return 4096


def _mincore_resident(mapping: mmap.mmap, page_size: int) -> int | None:
    """Count resident pages under ``mapping`` via ``mincore``."""
    if _LIBC is None:
        return None
    size = len(mapping)
    if size <= 0:
        return 0
    page_count = (size + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * ((page_count + 7) // 8))()  # type: ignore[name-defined]
    buffer_array = (ctypes.c_ubyte * size).from_buffer(mapping)  # type: ignore[name-defined]
    address = ctypes.addressof(buffer_array)  # type: ignore[name-defined]
    result = _LIBC.mincore(
        ctypes.c_void_p(address),  # type: ignore[name-defined]
        ctypes.c_size_t(size),  # type: ignore[name-defined]
        vector,
    )
    if result != 0:
        return None
    resident = 0
    for page_index in range(page_count):
        if vector[page_index // 8] & (1 << (page_index % 8)):
            resident += 1
    return resident


def _madvise_region(fd: int, offset: int, length: int, advice: int) -> bool:
    if _OS_MADVISE is None:
        return False
    try:
        _OS_MADVISE(fd, offset, length, advice)
        return True
    except (OSError, ValueError):
        return False


def _madvise_constant(name: str) -> int | None:
    if name == "willneed":
        return getattr(os, "MADV_WILLNEED", None)
    if name == "dontneed":
        return getattr(os, "MADV_DONTNEED", None)
    return None


def _process_rss_bytes() -> int | None:
    return _rss_bytes(Path("/proc/self/status"))


def collect_sample(
    *,
    path: Path,
    page_size: int,
    max_mapped_bytes: int,
    madvise: str | None = None,
) -> dict[str, Any]:
    """Sample one file: mincore residency + cgroup/RSS evidence."""
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "schema": PAGE_CACHE_EVIDENCE_SCHEMA,
            "timestamp_ns": time.time_ns(),
            "path": str(path),
            "file_size_bytes": None,
            "mapped_bytes": 0,
            "page_size_bytes": page_size,
            "total_pages": 0,
            "resident_pages": None,
            "resident_fraction": None,
            "mincore_status": f"error:{type(exc).__name__}",
            "cgroup_memory_current_bytes": None,
            "cgroup_status": "unavailable",
            "cgroup_path": "",
            "process_rss_bytes": None,
        }
    file_size = max(0, int(stat.st_size))
    mapped = min(file_size, max(0, int(max_mapped_bytes)))
    resident: int | None = None
    mincore_status = "unsupported"
    madvise_status = "not_requested"
    if mapped > 0:
        fd = os.open(path, os.O_RDONLY)
        try:
            if madvise is not None:
                constant = _madvise_constant(madvise)
                madvise_status = (
                    "unsupported"
                    if constant is None
                    else ("issued" if _madvise_region(fd, 0, mapped, constant) else "failed")
                )
            mapping = mmap.mmap(fd, mapped, access=mmap.ACCESS_READ)
            try:
                resident = _mincore_resident(mapping, page_size)
                mincore_status = "valid" if resident is not None else "unsupported"
            finally:
                mapping.close()
        except (OSError, ValueError, TypeError) as exc:
            mincore_status = f"error:{type(exc).__name__}"
        finally:
            os.close(fd)
    cgroup_bytes, cgroup_status, cgroup_path = current_cgroup_memory_evidence()
    total_pages = (mapped + page_size - 1) // page_size if mapped > 0 else 0
    return {
        "schema": PAGE_CACHE_EVIDENCE_SCHEMA,
        "timestamp_ns": time.time_ns(),
        "path": str(path),
        "file_size_bytes": file_size,
        "mapped_bytes": mapped,
        "page_size_bytes": page_size,
        "total_pages": total_pages,
        "resident_pages": resident,
        "resident_fraction": None if resident is None else round(resident / max(1, total_pages), 6),
        "mincore_status": mincore_status,
        "madvise_status": madvise_status,
        "cgroup_memory_current_bytes": cgroup_bytes,
        "cgroup_status": cgroup_status,
        "cgroup_path": cgroup_path,
        "process_rss_bytes": _process_rss_bytes(),
    }


@dataclass(slots=True)
class PageCacheEvidenceCollector:
    path: Path
    sample_interval_s: float = 1.0
    duration_s: float | None = 60.0
    max_mapped_bytes: int = 512 * 1024 * 1024
    page_size: int = 0
    madvise_willneed_at_start: bool = False
    madvise_dontneed_on_exit: bool = False

    def __post_init__(self) -> None:
        if not self.page_size:
            self.page_size = _page_size()
        target = self._target()
        self.path = target

    def _target(self) -> Path:
        if self.path.is_dir():
            files = [path for path in self.path.iterdir() if path.is_file()]
            if not files:
                raise ValueError(f"no regular files under evidence dir: {self.path}")
            return max(files, key=lambda item: item.stat().st_size)
        return self.path

    def sample(self) -> dict[str, Any]:
        return collect_sample(
            path=self.path,
            page_size=self.page_size,
            max_mapped_bytes=self.max_mapped_bytes,
            madvise="willneed" if self.madvise_willneed_at_start else None,
        )

    def run(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        deadline = None if self.duration_s is None else time.monotonic() + self.duration_s
        first = True
        while True:
            if first and self.madvise_willneed_at_start:
                collect_sample(
                    path=self.path,
                    page_size=self.page_size,
                    max_mapped_bytes=self.max_mapped_bytes,
                    madvise="willneed",
                )
                first = False
            samples.append(self.sample())
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, self.sample_interval_s))
        if self.madvise_dontneed_on_exit:
            collect_sample(
                path=self.path,
                page_size=self.page_size,
                max_mapped_bytes=self.max_mapped_bytes,
                madvise="dontneed",
            )
        return samples


def write_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "timestamp_ns,file_size_bytes,resident_pages,total_pages,resident_fraction,"
        "process_rss_bytes,cgroup_memory_current_bytes,mincore_status\n"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        for sample in samples:
            handle.write(
                ",".join(
                    str(
                        sample.get(name)
                        if sample.get(name) is not None
                        else ""
                    )
                    for name in (
                        "timestamp_ns",
                        "file_size_bytes",
                        "resident_pages",
                        "total_pages",
                        "resident_fraction",
                        "process_rss_bytes",
                        "cgroup_memory_current_bytes",
                        "mincore_status",
                    )
                )
                + "\n"
            )


__all__ = [
    "PAGE_CACHE_EVIDENCE_SCHEMA",
    "PageCacheEvidenceCollector",
    "collect_sample",
    "write_csv",
    "write_jsonl",
]
