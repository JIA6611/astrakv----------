"""File-backed mmap KV-cache manager with OS virtual-memory integration.

This module implements a KV-cache manager that directly leverages Linux virtual
memory mechanisms (mmap + madvise) for on-demand loading, prefetching, and
eviction of KV-cache blocks.  It is the **recommended** implementation path for
satisfying competition Task 2 ("use virtual memory techniques").

Core design:
- KV-cache backing store is a sparse file on NVMe/SSD.
- mmap maps the file into the process virtual address space.
- OS page cache automatically handles demand paging (on-demand load).
- madvise(MADV_WILLNEED) controls prefetch — tell the OS which blocks to
  read ahead.
- madvise(MADV_DONTNEED) controls eviction — tell the OS which blocks to
  reclaim under memory pressure.
- mincore queries which blocks are currently resident in physical memory.

This directly reuses the OS virtual-memory subsystem, satisfying the
competition's core requirement.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # Optional: importing the VM boundary must work on non-PoC hosts.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - covered by import-only tests.
    np = None  # type: ignore[assignment]


# ── madvise constants ───────────────────────────────────────────────
MADV_NORMAL     = 0
MADV_RANDOM     = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED   = 3   # prefetch: tell OS we will need these pages soon
MADV_DONTNEED   = 4   # evict: tell OS we no longer need these pages
MADV_FREE       = 8   # Linux 4.5+: lazy free
MADV_REMOVE     = 9   # Linux 2.6.16+: free and remove backing

_libc: Any | None = None


def vm_platform_available() -> bool:
    """Return whether this host exposes the Linux VM syscalls used by the PoC."""

    return sys.platform.startswith("linux")


def vm_platform_reason() -> str:
    if vm_platform_available():
        return "available"
    return f"unsupported_platform: Linux madvise/mincore required, current platform is {sys.platform}"


def vm_numpy_available() -> bool:
    return np is not None


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("missing_optional_dependency: numpy is required for the mmap VM PoC")
    return np


def _linux_libc() -> Any:
    global _libc
    if not vm_platform_available():
        raise OSError(vm_platform_reason())
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return _libc


def _madvise(addr: int, length: int, advice: int) -> int:
    """Wrap madvise(2).  Returns 0 on success, -1 on error."""
    return _linux_libc().madvise(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(length),
        ctypes.c_int(advice),
    )


def _mincore(addr: int, length: int) -> bytes:
    """Wrap mincore(2).  Returns a byte vector (1 bit per page = resident)."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    vec_size = (length + page_size - 1) // page_size
    vec = ctypes.create_string_buffer(vec_size)
    ret = _linux_libc().mincore(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(length),
        vec,
    )
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return vec.raw


# ── config / stats dataclasses ──────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MMapKVCacheConfig:
    """Configuration for the mmap-backed KV cache."""

    total_blocks: int
    block_size_bytes: int
    backing_file: str
    dtype_str: str = "float16"

    @property
    def total_size_bytes(self) -> int:
        return self.total_blocks * self.block_size_bytes

    @property
    def dtype(self) -> np.dtype:
        return _require_numpy().dtype(self.dtype_str)


@dataclass(slots=True)
class MMapKVCacheStats:
    """Snapshot of mmap KV-cache runtime statistics."""

    total_blocks: int = 0
    block_size_bytes: int = 0
    resident_blocks: int = 0
    resident_ratio: float = 0.0
    prefetch_requests: int = 0
    evict_requests: int = 0
    cold_read_count: int = 0
    warm_read_count: int = 0
    cold_read_total_us: float = 0.0
    warm_read_total_us: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_cold_read_us(self) -> float:
        return self.cold_read_total_us / max(1, self.cold_read_count)

    @property
    def avg_warm_read_us(self) -> float:
        return self.warm_read_total_us / max(1, self.warm_read_count)

    def to_record(self) -> dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "block_size_bytes": self.block_size_bytes,
            "resident_blocks": self.resident_blocks,
            "resident_ratio": round(self.resident_ratio, 6),
            "prefetch_requests": self.prefetch_requests,
            "evict_requests": self.evict_requests,
            "cold_read_count": self.cold_read_count,
            "warm_read_count": self.warm_read_count,
            "avg_cold_read_us": round(self.avg_cold_read_us, 3),
            "avg_warm_read_us": round(self.avg_warm_read_us, 3),
            "metadata": dict(self.metadata),
        }


# ── main class ──────────────────────────────────────────────────────

class MMapKVCache:
    """MMap-backed KV-cache manager.

    Each KV block is a contiguous region of a sparse backing file that is
    memory-mapped into the process address space.  Reads from blocks that
    are not currently in the OS page cache will trigger a minor/major page
    fault handled transparently by the kernel — this is **on-demand
    loading** at the OS level.

    Prefetch and eviction are controlled explicitly via madvise hints,
    enabling profile-guided virtual-memory management.

    Usage::

        cache = MMapKVCache(
            backing_file="/data/kv_cache.bin",
            total_blocks=1024,
            block_size_bytes=2 * 1024 * 1024,  # 2 MB per block
        )
        # Write KV data
        cache.write_block(5, kv_tensor)
        # Evict everything (MADV_DONTNEED)
        cache.evict_batch(range(1024))
        # Prefetch hot blocks (MADV_WILLNEED)
        cache.prefetch_batch([5, 7, 12])
        # On-demand read (OS handles page fault)
        data = cache.read_block(5)
        # Check residency
        stats = cache.collect_stats()
    """

    def __init__(self, config: MMapKVCacheConfig | None = None, **kwargs: Any) -> None:
        if not vm_platform_available():
            raise RuntimeError(vm_platform_reason())
        _require_numpy()
        if config is not None:
            self._cfg = config
        else:
            self._cfg = MMapKVCacheConfig(**kwargs)
        self._backing = Path(self._cfg.backing_file)
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._base_addr: int = 0
        self._stats = MMapKVCacheStats(
            total_blocks=self._cfg.total_blocks,
            block_size_bytes=self._cfg.block_size_bytes,
        )
        self._logger = logging.getLogger("MMapKVCache")
        self._init()

    # ── lifecycle ───────────────────────────────────────────────

    def _init(self) -> None:
        self._init_backing_file()
        self._fd = os.open(str(self._backing), os.O_RDWR)
        self._mm = mmap.mmap(self._fd, self._cfg.total_size_bytes, access=mmap.ACCESS_WRITE)
        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        self._logger.info(
            "MMapKVCache ready: %d blocks × %.1f KB = %.2f GB, backing=%s",
            self._cfg.total_blocks,
            self._cfg.block_size_bytes / 1024,
            self._cfg.total_size_bytes / 1e9,
            self._backing,
        )

    def _init_backing_file(self) -> None:
        """Create sparse backing file if it does not exist."""
        if self._backing.exists():
            actual = os.path.getsize(self._backing)
            if actual != self._cfg.total_size_bytes:
                self._logger.warning(
                    "Backing file size mismatch: expected %d, got %d; recreating",
                    self._cfg.total_size_bytes,
                    actual,
                )
                self._backing.unlink()
        if not self._backing.exists():
            self._logger.info(
                "Creating sparse backing file: %s (%.2f GB)",
                self._backing,
                self._cfg.total_size_bytes / 1e9,
            )
            with open(self._backing, "wb") as fh:
                fh.seek(self._cfg.total_size_bytes - 1)
                fh.write(b"\x00")

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "MMapKVCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── read / write ────────────────────────────────────────────

    def read_block(self, block_id: int) -> np.ndarray:
        """Read KV data for *block_id*.

        If the backing pages are not resident, the OS triggers a page fault
        and loads them from the backing file automatically — on-demand
        loading at the OS level.
        """
        self._validate_block(block_id)
        offset = block_id * self._cfg.block_size_bytes
        t0 = time.perf_counter()
        raw = self._mm[offset : offset + self._cfg.block_size_bytes]
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        # Heuristic: > 100 µs is likely a cold read (page fault)
        if elapsed_us > 100:
            self._stats.cold_read_count += 1
            self._stats.cold_read_total_us += elapsed_us
        else:
            self._stats.warm_read_count += 1
            self._stats.warm_read_total_us += elapsed_us

        return np.frombuffer(raw, dtype=self._cfg.dtype).copy()

    def write_block(self, block_id: int, data: np.ndarray) -> None:
        """Write KV data into *block_id* (write-back to mmap, OS flushes)."""
        self._validate_block(block_id)
        if data.nbytes != self._cfg.block_size_bytes:
            raise ValueError(
                f"Data size {data.nbytes} does not match block size "
                f"{self._cfg.block_size_bytes}"
            )
        offset = block_id * self._cfg.block_size_bytes
        self._mm[offset : offset + self._cfg.block_size_bytes] = (
            data.astype(self._cfg.dtype).tobytes()
        )

    # ── virtual-memory control (madvise) ─────────────────────────

    def prefetch_block(self, block_id: int) -> bool:
        """Prefetch: tell OS to read *block_id* ahead (MADV_WILLNEED).

        Returns True if the madvise call succeeded.
        """
        offset = block_id * self._cfg.block_size_bytes
        addr = self._base_addr + offset
        ret = _madvise(addr, self._cfg.block_size_bytes, MADV_WILLNEED)
        self._stats.prefetch_requests += 1
        if ret != 0:
            self._logger.debug("madvise WILLNEED failed for block %d", block_id)
            return False
        self._logger.debug("Prefetch requested: block=%d", block_id)
        return True

    def evict_block(self, block_id: int) -> bool:
        """Evict: tell OS to reclaim pages for *block_id* (MADV_DONTNEED).

        The data stays in the backing file; a future read will trigger a
        fresh page fault.

        Returns True if the madvise call succeeded.
        """
        offset = block_id * self._cfg.block_size_bytes
        addr = self._base_addr + offset
        ret = _madvise(addr, self._cfg.block_size_bytes, MADV_DONTNEED)
        self._stats.evict_requests += 1
        if ret != 0:
            self._logger.debug("madvise DONTNEED failed for block %d", block_id)
            return False
        self._logger.debug("Evicted: block=%d", block_id)
        return True

    def prefetch_batch(self, block_ids: list[int]) -> int:
        """Batch prefetch for profile-guided prefetching."""
        ok = sum(1 for bid in block_ids if self.prefetch_block(bid))
        self._logger.info("Batch prefetch: %d/%d blocks", ok, len(block_ids))
        return ok

    def evict_batch(self, block_ids: list[int]) -> int:
        """Batch evict for memory-pressure scenarios."""
        ok = sum(1 for bid in block_ids if self.evict_block(bid))
        self._logger.info("Batch evict: %d/%d blocks", ok, len(block_ids))
        return ok

    # ── residency query (mincore) ────────────────────────────────

    def get_resident_blocks(self) -> dict[int, float]:
        """Return per-block residency ratio via mincore(2).

        Each value is in [0.0, 1.0] — the fraction of the block's pages
        that are currently resident in physical memory.
        """
        try:
            vec = _mincore(self._base_addr, self._cfg.total_size_bytes)
        except OSError:
            self._logger.warning("mincore failed; returning empty residency")
            return {}

        page_size = os.sysconf("SC_PAGE_SIZE")
        resident: dict[int, float] = {}
        for bid in range(self._cfg.total_blocks):
            page_start = (bid * self._cfg.block_size_bytes) // page_size
            page_end = (
                (bid + 1) * self._cfg.block_size_bytes + page_size - 1
            ) // page_size
            total_pages = page_end - page_start
            pages_in_mem = sum(
                (vec[p] & 1) for p in range(page_start, min(page_end, len(vec)))
            )
            resident[bid] = pages_in_mem / max(1, total_pages)
        return resident

    def collect_stats(self) -> MMapKVCacheStats:
        """Gather current residency statistics."""
        resident_map = self.get_resident_blocks()
        if resident_map:
            self._stats.resident_blocks = sum(
                1 for v in resident_map.values() if v > 0.5
            )
            self._stats.resident_ratio = self._stats.resident_blocks / max(
                1, self._cfg.total_blocks
            )
        return self._stats

    # ── helpers ─────────────────────────────────────────────────

    def _validate_block(self, block_id: int) -> None:
        if not (0 <= block_id < self._cfg.total_blocks):
            raise IndexError(
                f"block_id {block_id} out of range [0, {self._cfg.total_blocks})"
            )
        if self._mm is None:
            raise RuntimeError("MMapKVCache is closed")

    @property
    def config(self) -> MMapKVCacheConfig:
        return self._cfg

    @property
    def stats(self) -> MMapKVCacheStats:
        return self._stats
