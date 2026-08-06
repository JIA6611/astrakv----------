"""Low-perturbation Linux/DGX diagnostic sidecar and optional deep-tool capture."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


TOOLS = ("pidstat", "iostat", "sar", "perf", "bpftrace", "dcgmi", "nsys", "ncu")


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capabilities = tool_capabilities()
    samples = sample_process(args.pid, args.duration_seconds, args.interval_seconds)
    write_csv(output / "diagnostic_samples.csv", samples)
    raw = run_optional_tools(args, output, capabilities)
    manifest = {
        "schema": "astrakv-runtime-diagnostic-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "deep_diagnostic" if args.collect_tools else "sidecar_diagnostic",
        "platform": platform.platform(),
        "pid": args.pid,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "capabilities": capabilities,
        "samples": str(output / "diagnostic_samples.csv"),
        "raw_artifacts": raw,
        "nsys_report": args.nsys_report or "",
        "ncu_report": args.ncu_report or "",
        "claim_boundary": "Diagnostic artifacts are separate from formal endpoint throughput benchmarks.",
    }
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Runtime diagnostic artifacts written to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--collect-tools", action="store_true", help="Run available pidstat/iostat/sar/perf/DCGM tools as a separate diagnostic stage.")
    parser.add_argument("--enable-ebpf", action="store_true", help="Record bpftrace capability; eBPF is never required for benchmark success.")
    parser.add_argument("--nsys-report", default="", help="Existing Nsight Systems report path to index.")
    parser.add_argument("--ncu-report", default="", help="Existing Nsight Compute report path to index.")
    parser.add_argument("--output-dir", default="results/runtime_diagnostic")
    return parser.parse_args()


def tool_capabilities() -> dict[str, dict[str, Any]]:
    capabilities = {
        name: {"available": bool(shutil.which(name)), "path": shutil.which(name) or ""}
        for name in TOOLS
    }
    capabilities["perf"]["perf_event_paranoid"] = perf_event_paranoid()
    return capabilities


def perf_event_paranoid() -> int | str:
    path = Path("/proc/sys/kernel/perf_event_paranoid")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return "not_available"


def sample_process(pid: int, duration: float, interval: float) -> list[dict[str, Any]]:
    if not platform.system().lower().startswith("linux"):
        return [{"timestamp_s": time.time(), "status": "unsupported_platform", "pid": pid}]
    target = pid or 0
    deadline = time.monotonic() + max(0.1, duration)
    result: list[dict[str, Any]] = []
    while True:
        result.append(process_snapshot(target))
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.1, interval))
    return result


def process_snapshot(pid: int) -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp_s": time.time(), "pid": pid, "status": "ok"}
    if not pid:
        row["status"] = "pid_not_provided"
        return row
    base = Path(f"/proc/{pid}")
    try:
        status = parse_key_values(base / "status")
        io = parse_key_values(base / "io")
        stat = (base / "stat").read_text(encoding="utf-8").split()
        row.update({
            "rss_kb": number(status.get("VmRSS")), "pss_kb": pss_kb(base / "smaps_rollup"),
            "minor_faults": int(stat[9]), "major_faults": int(stat[11]),
            "voluntary_ctx_switches": number(status.get("voluntary_ctxt_switches")),
            "involuntary_ctx_switches": number(status.get("nonvoluntary_ctxt_switches")),
            "process_read_bytes": number(io.get("read_bytes")), "process_write_bytes": number(io.get("write_bytes")),
        })
    except (OSError, ValueError, IndexError) as exc:
        row["status"] = f"unavailable:{type(exc).__name__}"
    return row


def run_optional_tools(args: argparse.Namespace, output: Path, capabilities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not args.collect_tools or not platform.system().lower().startswith("linux"):
        return []
    specs: list[tuple[str, list[str]]] = []
    seconds = str(max(1, int(args.duration_seconds)))
    if capabilities["pidstat"]["available"] and args.pid:
        specs.append(("pidstat", ["pidstat", "-r", "-u", "-w", "-d", "-p", str(args.pid), "1", seconds]))
    if capabilities["iostat"]["available"]:
        specs.append(("iostat", ["iostat", "-dx", "1", seconds]))
    if capabilities["sar"]["available"]:
        specs.append(("sar", ["sar", "-u", "1", seconds]))
    if capabilities["perf"]["available"] and args.pid:
        specs.append(("perf", ["perf", "stat", "-p", str(args.pid), "-e", "page-faults,context-switches,cache-misses", "--", "sleep", seconds]))
    if capabilities["dcgmi"]["available"]:
        specs.append(("dcgmi", ["dcgmi", "dmon", "-e", "1001,1002,1003", "-d", "1000", "-c", seconds]))
    artifacts: list[dict[str, Any]] = []
    for name, command in specs:
        target = output / f"{name}.txt"
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=max(15, int(args.duration_seconds) + 15), check=False)
            target.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            artifacts.append(tool_artifact_record(name, target, result))
        except Exception as exc:
            artifacts.append({"tool": name, "status": f"failed:{type(exc).__name__}"})
    if args.enable_ebpf:
        artifacts.append({"tool": "bpftrace", "status": "capability_recorded_only", "available": capabilities["bpftrace"]["available"]})
    return artifacts


def tool_artifact_record(name: str, path: Path, result: Any) -> dict[str, Any]:
    stderr = str(getattr(result, "stderr", "") or "").strip()
    returncode = int(getattr(result, "returncode", 1))
    if returncode == 0:
        status = "collected"
    elif name == "perf" and (
        "Access to performance monitoring" in stderr
        or "perf_event_paranoid" in stderr
        or "CAP_PERFMON" in stderr
    ):
        status = "not_available"
    else:
        status = "failed"
    return {
        "tool": name,
        "path": str(path),
        "returncode": returncode,
        "collection_status": status,
        "stderr_summary": stderr[:500],
    }


def parse_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().split()[0]
    return result


def pss_kb(path: Path) -> int:
    try:
        return number(parse_key_values(path).get("Pss"))
    except OSError:
        return 0


def number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
