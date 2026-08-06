"""Compatibility wrapper for the OS virtual-memory demo.

The reusable file-backed mmap implementation lives in ``runtime.vm_backend``.
This module keeps older imports working while experiments remain responsible
for runnable demonstrations and report artifacts.
"""

from astrakv.runtime.vm_backend import (
    BYTES_PER_MB,
    VMAccessRecord,
    VMDemoConfig,
    VMDemoResult,
    VMDemoSummary,
    VirtualMemoryDemoRunner,
    build_access_order,
    cpu_rss_mb,
    create_backing_file,
    percentile,
    summarize_records,
    write_access_trace_csv,
    write_summary_json,
)

__all__ = [
    "BYTES_PER_MB",
    "VMAccessRecord",
    "VMDemoConfig",
    "VMDemoResult",
    "VMDemoSummary",
    "VirtualMemoryDemoRunner",
    "build_access_order",
    "cpu_rss_mb",
    "create_backing_file",
    "percentile",
    "summarize_records",
    "write_access_trace_csv",
    "write_summary_json",
]
