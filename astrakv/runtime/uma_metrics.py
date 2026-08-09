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


def resolve_cgroup_memory_current_path(
    *,
    proc_cgroup_path: Path | str = "/proc/self/cgroup",
    proc_mountinfo_path: Path | str = "/proc/self/mountinfo",
) -> Path | None:
    """Resolve this process's cgroup-v2 ``memory.current`` file.

    A cgroup-v2 mount need not expose memory-controller files at its root.
    On the DGX host, for example, an SSH or tmux child belongs to a nested
    ``user.slice/...scope`` while ``/sys/fs/cgroup/memory.current`` does not
    exist.  Reading the root path and converting failure to zero would turn a
    missing measurement into false UMA evidence, so resolve the process's
    actual cgroup before each snapshot instead.
    """
    try:
        cgroup_rows = Path(proc_cgroup_path).read_text(encoding="utf-8").splitlines()
        mount_rows = Path(proc_mountinfo_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    cgroup_path: str | None = None
    for row in cgroup_rows:
        parts = row.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            cgroup_path = parts[2]
            break
    if not cgroup_path:
        return None

    for row in mount_rows:
        fields = row.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 1 >= len(fields) or fields[separator + 1] != "cgroup2":
            continue
        if len(fields) < 5:
            continue
        mount_root, mount_point = fields[3], fields[4]
        try:
            relative = Path(cgroup_path).relative_to(mount_root)
        except ValueError:
            continue
        candidate = Path(mount_point) / relative / "memory.current"
        if candidate.is_file():
            return candidate
    return None


def current_cgroup_memory_evidence() -> tuple[int | None, str, str]:
    """Return ``(bytes, status, path)`` for the current process cgroup.

    Zero is intentionally not considered valid evidence for an active runtime
    callback.  It is returned as ``unavailable`` rather than silently
    promoted to a physical-memory measurement.
    """
    path = resolve_cgroup_memory_current_path()
    if path is None:
        return None, "unavailable", ""
    value = _read_int(path)
    if value is None or value <= 0:
        return value, "unavailable", str(path)
    return value, "valid", str(path)


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
    cgroup_memory_status: str
    cgroup_memory_current_path: str
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
        cgroup_memory_current_path: Path | str | None = None,
        process_status_path: Path | str = "/proc/self/status",
        ssd_path: Path | str | None = None,
        topology: str = "gpu_ssd",
    ) -> None:
        if topology not in {"gpu_ssd", "gpu_cpu_ssd"}:
            raise ValueError("unsupported KV-Core topology")
        self.cgroup_memory_current_path = (
            None if cgroup_memory_current_path is None else Path(cgroup_memory_current_path)
        )
        self.process_status_path = Path(process_status_path)
        self.ssd_path = None if ssd_path is None else Path(ssd_path)
        self.topology = topology

    def snapshot(self, *, timestamp_ns: int, lmcache: Mapping[str, Any] | None = None, vllm: Mapping[str, Any] | None = None, disk_io: Mapping[str, Any] | None = None) -> UMAResourceSnapshot:
        lmcache = lmcache or {}
        vllm = vllm or {}
        disk_io = disk_io or {}
        if self.cgroup_memory_current_path is None:
            cgroup_value, cgroup_status, cgroup_path = current_cgroup_memory_evidence()
        else:
            cgroup_value = _read_int(self.cgroup_memory_current_path)
            cgroup_status = "valid" if cgroup_value is not None and cgroup_value > 0 else "unavailable"
            cgroup_path = str(self.cgroup_memory_current_path)
        return UMAResourceSnapshot(
            timestamp_ns=int(timestamp_ns),
            cgroup_memory_current_bytes=cgroup_value,
            cgroup_memory_status=cgroup_status,
            cgroup_memory_current_path=cgroup_path,
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


__all__ = [
    "UMA_RESOURCE_SCHEMA", "UMAResourceCollector", "UMAResourceSnapshot",
    "current_cgroup_memory_evidence", "resolve_cgroup_memory_current_path",
]
