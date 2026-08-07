"""UMA-aware resource evidence for the AstraKV-W KV-Core experiments.

GB10 exposes unified physical memory.  GPU and CPU cache residency remain
useful logical tiers, but neither is accepted here as an independent physical
memory measurement.  This module intentionally collects only evidence which
can be audited from the OS, vLLM, and LMCache snapshots.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


UMA_RESOURCE_SCHEMA = "astrakv-uma-resource-snapshot-v1"


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value in {"", "max"}:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _rss_bytes(status_path: Path) -> int | None:
    try:
        rows = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for row in rows:
        if not row.startswith("VmRSS:"):
            continue
        fields = row.split()
        if len(fields) >= 2:
            try:
                return max(0, int(fields[1])) * 1024
            except ValueError:
                return None
    return None


def _disk_usage_bytes(path: Path) -> int | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return max(0, int(stat.f_blocks - stat.f_bfree) * int(stat.f_frsize))


@dataclass(frozen=True, slots=True)
class UMAResourceSnapshot:
    timestamp_ns: int
    cgroup_memory_current_bytes: int | None
    process_rss_bytes: int | None
    vllm_available_kv_blocks: int | None
    lmcache_cpu_used_bytes: int | None
    lmcache_ssd_used_bytes: int | None
    disk_read_bytes: int | None
    disk_write_bytes: int | None
    topology: str

    def to_record(self) -> dict[str, Any]:
        return {"schema": UMA_RESOURCE_SCHEMA, **asdict(self)}


class UMAResourceCollector:
    """Collect one snapshot without inventing a separate GPU-memory number."""

    def __init__(
        self,
        *,
        cgroup_memory_current_path: Path | str = "/sys/fs/cgroup/memory.current",
        process_status_path: Path | str = "/proc/self/status",
        ssd_path: Path | str | None = None,
        topology: str = "gpu_ssd",
    ) -> None:
        if topology not in {"gpu_ssd", "gpu_cpu_ssd"}:
            raise ValueError("unsupported KV-Core topology")
        self.cgroup_memory_current_path = Path(cgroup_memory_current_path)
        self.process_status_path = Path(process_status_path)
        self.ssd_path = None if ssd_path is None else Path(ssd_path)
        self.topology = topology

    def snapshot(self, *, timestamp_ns: int, lmcache: Mapping[str, Any] | None = None, vllm: Mapping[str, Any] | None = None, disk_io: Mapping[str, Any] | None = None) -> UMAResourceSnapshot:
        lmcache = lmcache or {}
        vllm = vllm or {}
        disk_io = disk_io or {}
        return UMAResourceSnapshot(
            timestamp_ns=int(timestamp_ns),
            cgroup_memory_current_bytes=_read_int(self.cgroup_memory_current_path),
            process_rss_bytes=_rss_bytes(self.process_status_path),
            vllm_available_kv_blocks=_non_negative_optional(vllm.get("available_kv_blocks")),
            lmcache_cpu_used_bytes=_non_negative_optional(lmcache.get("cpu_used_bytes")),
            lmcache_ssd_used_bytes=_non_negative_optional(lmcache.get("ssd_used_bytes"), fallback_path=self.ssd_path),
            disk_read_bytes=_non_negative_optional(disk_io.get("read_bytes")),
            disk_write_bytes=_non_negative_optional(disk_io.get("write_bytes")),
            topology=self.topology,
        )


def _non_negative_optional(value: Any, *, fallback_path: Path | None = None) -> int | None:
    if value is None:
        return _disk_usage_bytes(fallback_path) if fallback_path is not None else None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


__all__ = ["UMA_RESOURCE_SCHEMA", "UMAResourceCollector", "UMAResourceSnapshot"]
