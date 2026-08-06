#!/usr/bin/env python3
"""AstraKV-W unified command-line interface.

Usage::

    python cli.py benchmark --config configs/baseline.yaml
    python cli.py prefetch --config configs/astrakv_selective_prefetch.yaml
    python cli.py analyze stress --results-dir results/stress_test
    python cli.py report --output-dir results/report
    python cli.py vm mmap --blocks 100
    python cli.py test --pattern test_vm

See ``docs/guides/cli_usage.md`` for details.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run_script(script_name: str, *args: str) -> int:
    """Run a script from the scripts/ directory."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def _run_shell(script_name: str, *args: str) -> int:
    """Run a shell script from scripts/."""
    script_path = SCRIPTS_DIR / script_name
    cmd = ["bash", str(script_path), *args]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


# ── subcommand: benchmark ──────────────────────────────────────

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks (synthetic or real)."""
    if args.real:
        script = "benchmark/run_real_benchmark.py"
    elif args.selective:
        script = "benchmark/run_selective_prefetch_real.py"
    else:
        script = "benchmark/benchmark_runner.py"
    extra = []
    if args.config:
        extra += ["--config", args.config]
    if args.output_dir:
        extra += ["--output-dir", args.output_dir]
    return _run_script(script, *extra)


# ── subcommand: prefetch ───────────────────────────────────────

def cmd_prefetch(args: argparse.Namespace) -> int:
    """Run selective KV prefetch."""
    return _run_script(
        "benchmark/run_selective_prefetch_real.py",
        *(["--config", args.config] if args.config else []),
        *(["--output-dir", args.output_dir] if args.output_dir else []),
    )


# ── subcommand: analyze ────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run analysis tools."""
    script_map = {
        "stress": "reporting/analyze_stress_results.py",
        "memory": "reporting/analyze_memory_pressure.py",
        "ablation": "policy/analyze_policy_ablation.py",
        "failure": "reporting/analyze_failure_recovery.py",
        "multi-model": "reporting/analyze_multi_model_evaluation.py",
    }
    script = script_map.get(args.tool, "reporting/analyze_stress_results.py")
    extra = []
    if args.results_dir:
        extra += ["--results-dir", args.results_dir]
    if args.output_dir:
        extra += ["--output-dir", args.output_dir]
    return _run_script(script, *extra)


# ── subcommand: report ─────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> int:
    """Build competition report from artifacts."""
    return _run_script(
        "reporting/build_competition_report.py",
        *(["--output-dir", args.output_dir] if args.output_dir else []),
    )


# ── subcommand: vm ─────────────────────────────────────────────

def cmd_vm(args: argparse.Namespace) -> int:
    """Run virtual memory demos."""
    if args.demo == "mmap":
        extra = ["--blocks", str(args.blocks), "--block-size-mb", str(args.block_size_mb)]
        if args.output_dir:
            extra += ["--output-dir", args.output_dir]
        return _run_script("vm/run_mmap_kv_cache.py", *extra)
    elif args.demo == "demo":
        return _run_script(
            "vm/run_vm_demo.py",
            *(["--output-dir", args.output_dir] if args.output_dir else []),
        )
    elif args.demo == "layer":
        extra = []
        if args.model:
            extra += ["--model", args.model]
        if args.output_dir:
            extra += ["--output-dir", args.output_dir]
        return _run_script("vm/run_layer_offload_poc.py", *extra)
    else:
        print(f"Unknown VM demo: {args.demo}")
        return 1


# ── subcommand: test ───────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    """Run test suite."""
    cmd_parts = [sys.executable, "-m", "pytest", "tests/", "-v"]
    if args.pattern:
        cmd_parts += ["-k", args.pattern]
    return subprocess.run(cmd_parts, cwd=str(PROJECT_ROOT)).returncode


# ── subcommand: edge ───────────────────────────────────────────

def cmd_edge(args: argparse.Namespace) -> int:
    """Edge device simulation."""
    if args.setup:
        return _run_shell("archive/setup_edge_sim.sh", str(args.mem_gb))
    else:
        return _run_shell("archive/run_edge_sim_tests.sh")


# ── subcommand: ablation ───────────────────────────────────────

def cmd_ablation(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Run ablation experiment suite."""
    return _run_shell("archive/run_ablation.sh")


# ── main parser ────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrakv",
        description="AstraKV-W: Memory-Constrained LLM Inference Toolkit",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run benchmarks")
    p_bench.add_argument("--config", help="Benchmark config YAML")
    p_bench.add_argument("--output-dir", help="Output directory")
    p_bench.add_argument("--real", action="store_true", help="Run real vLLM benchmark")
    p_bench.add_argument("--selective", action="store_true", help="Run selective prefetch benchmark")

    # prefetch
    p_pref = sub.add_parser("prefetch", help="Run selective KV prefetch")
    p_pref.add_argument("--config", help="Prefetch config YAML")
    p_pref.add_argument("--output-dir", help="Output directory")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze benchmark results")
    p_analyze.add_argument("tool", nargs="?", default="stress",
                           choices=["stress", "memory", "ablation", "failure", "multi-model"])
    p_analyze.add_argument("--results-dir")
    p_analyze.add_argument("--output-dir")

    # report
    p_report = sub.add_parser("report", help="Build competition report")
    p_report.add_argument("--output-dir")

    # vm
    p_vm = sub.add_parser("vm", help="Virtual memory demos")
    p_vm.add_argument("demo", nargs="?", default="mmap",
                      choices=["mmap", "demo", "layer"])
    p_vm.add_argument("--blocks", type=int, default=100)
    p_vm.add_argument("--block-size-mb", type=float, default=1.0)
    p_vm.add_argument("--model", help="Model name (for layer offload demo)")
    p_vm.add_argument("--output-dir")

    # test
    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("--pattern", "-k", help="Pytest pattern filter")

    # edge
    p_edge = sub.add_parser("edge", help="Edge device simulation")
    p_edge.add_argument("--setup", action="store_true", help="Setup cgroup (requires root)")
    p_edge.add_argument("--mem-gb", type=int, default=16, help="Memory limit in GB")

    # ablation
    sub.add_parser("ablation", help="Run ablation experiment suite")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "benchmark": cmd_benchmark,
        "prefetch": cmd_prefetch,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "vm": cmd_vm,
        "test": cmd_test,
        "edge": cmd_edge,
        "ablation": cmd_ablation,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
