#!/usr/bin/env python3
"""Build publication/demo figures from AstraKV-W evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_COLORS = {
    "vLLM": "#1f4e79",
    "LMCache CPU": "#f28e2b",
    "LMCache Disk": "#2ca02c",
    "Prefetch": "#7b61ff",
    "无 Prefetch": "#9aa0a6",
}

BACKEND_LABELS = {
    "vllm": "vLLM",
    "vllm_extreme": "vLLM",
    "lmcache_cpu": "LMCache CPU",
    "lmcache_cpu_extreme": "LMCache CPU",
    "lmcache_disk": "LMCache Disk",
    "lmcache_disk_extreme": "LMCache Disk",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_num(*values: Any, default: float = 0.0) -> float:
    for value in values:
        parsed = fnum(value, default=math.nan)
        if not math.isnan(parsed):
            return parsed
    return default


def nested_get(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ensure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from pathlib import Path

    windows_fonts = Path("C:/Windows/Fonts")
    for font_file in [
        "simhei.ttf",
        "simsun.ttc",
        "simsunb.ttf",
        "NotoSansSC-VF.ttf",
        "NotoSerifSC-VF.ttf",
    ]:
        font_path = windows_fonts / font_file
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass

    # Prefer common Windows and CJK fonts so Chinese text renders correctly on
    # both Windows and Linux without depending on a specific font package.
    for font_name in [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Microsoft JhengHei",
        "Microsoft JhengHei UI",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
    ]:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["font.family"] = "sans-serif"
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    return plt


def savefig(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt = ensure_matplotlib()
    plt.close(fig)


def fmt(value: float, digits: int = 1, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}"


def bar_colors(labels: list[str]) -> list[str]:
    return [BACKEND_COLORS.get(label, "#4c78a8") for label in labels]


def annotate_bars(ax: Any, bars: Any, suffix: str = "", digits: int = 1, dy: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + span * dy,
            fmt(float(height), digits, suffix),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def style_axis(ax: Any, ylabel: str = "") -> None:
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_box(ax: Any, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str = "#f3f4f6", ec: str = "#374151") -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def draw_arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str = "#4b5563") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": color, "shrinkA": 4, "shrinkB": 4},
    )


def draw_flow_card(
    ax: Any,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    subtitle: str,
    fc: str = "#f8fafc",
    accent: str = "#94a3b8",
) -> None:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    x, y = xy
    w, h = wh
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#cbd5e1",
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(card)
    
    # 顶部装饰条微调，适配更大的字体
    ax.add_patch(
        Rectangle(
            (x + 0.01, y + h - 0.018),
            max(w - 0.02, 0.01),
            0.014,
            linewidth=0,
            facecolor=accent,
            alpha=0.95,
            zorder=3,
        )
    )
    
    # 调大主标题字体并微调高度
    ax.text(
        x + w / 2,
        y + h * 0.60,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        weight="bold",
        color="#0f172a",
        zorder=4,
    )
    
    # 调大副标题字体
    ax.text(
        x + w / 2,
        y + h * 0.32,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#475569",
        linespacing=1.15,
        zorder=4,
    )


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def artifact_paths(main: Path, demo: Path | None, boundary_pass: Path | None, boundary_fail: Path | None) -> dict[str, Path | None]:
    selected = demo / "selected_artifacts" if demo else None
    return {
        "comparison": (selected / "comparison_results.csv") if selected and (selected / "comparison_results.csv").exists() else main / "01_e2e/step7_comparison/comparison_results.csv",
        "vllm_baseline": (selected / "vllm_baseline.csv") if selected and (selected / "vllm_baseline.csv").exists() else find_first(main, ["01_e2e/step3_vllm_benchmark/*/benchmark_results.csv"]),
        "lmcache_cpu_baseline": (selected / "lmcache_cpu_baseline.csv") if selected and (selected / "lmcache_cpu_baseline.csv").exists() else find_first(main, ["01_e2e/step4_lmcache_cpu_benchmark/*/benchmark_results.csv"]),
        "lmcache_disk_baseline": (selected / "lmcache_disk_baseline.csv") if selected and (selected / "lmcache_disk_baseline.csv").exists() else find_first(main, ["01_e2e/step4_lmcache_disk_benchmark/*/benchmark_results.csv"]),
        "prefetch_results": (selected / "prefetch_results.csv") if selected and (selected / "prefetch_results.csv").exists() else main / "01_e2e/step5_prefetch/prefetch_results.csv",
        "policy_ablation": (selected / "policy_ablation.csv") if selected and (selected / "policy_ablation.csv").exists() else main / "01_e2e/step7_policy_ablation/policy_ablation_results.csv",
        "chunk_scores": (selected / "chunk_scores.csv") if selected and (selected / "chunk_scores.csv").exists() else main / "06_policy_chain/chunk_scores/chunk_scores.csv",
        "quality": (selected / "quality.csv") if selected and (selected / "quality.csv").exists() else main / "05_quality/lmcache_disk_vs_vllm/quality_results.csv",
        "official_stress": main / "01_e2e/step6_stress_analysis/stress_summary.csv",
        "extreme_stress": main / "01_e2e/extreme_stress_analysis/stress_summary.csv",
        "boundary_32k": main / "02_boundary_32k/stress_analysis/stress_summary.csv",
        "boundary_pass": (selected / "boundary_stress_pass.csv") if selected and (selected / "boundary_stress_pass.csv").exists() else (boundary_pass / "02_boundary_32k/stress_analysis/stress_summary.csv" if boundary_pass else None),
        "boundary_pass_report": boundary_pass / "07_final_report/competition_report.md" if boundary_pass else None,
        "boundary_fail_log": boundary_fail / "02_boundary_32k/vllm_server.log" if boundary_fail else None,
        "cache_events_boundary_disk": (selected / "boundary_disk_cache_events.jsonl") if selected and (selected / "boundary_disk_cache_events.jsonl").exists() else (boundary_pass / "03_cache_events/lmcache_disk_boundary/cache_events.jsonl" if boundary_pass else main / "03_cache_events/lmcache_disk_boundary/cache_events.jsonl"),
        "cache_events_prefetch": (selected / "prefetch_cache_events.jsonl") if selected and (selected / "prefetch_cache_events.jsonl").exists() else main / "03_cache_events/prefetch_e2e/cache_events.jsonl",
        "mmap": (selected / "mmap_kv_evidence.json") if selected and (selected / "mmap_kv_evidence.json").exists() else main / "04_os_vm/mmap_kv_cache/mmap_kv_demo_summary.json",
        "cli_mmap": main / "04_os_vm/cli_mmap/mmap_kv_demo_summary.json",
        "dgx_vm": (selected / "dgx_vm_evidence.json") if selected and (selected / "dgx_vm_evidence.json").exists() else main / "04_os_vm/dgx_spark_vm/dgx_spark_vm_evidence_summary.json",
        "trace_events": main / "01_e2e/step7_trace_store/trace_events.jsonl",
        "load_recompute": main / "06_policy_chain/load_recompute/load_recompute_decisions.csv",
        "object_scheduler": main / "06_policy_chain/object_scheduler/object_schedule_decisions.csv",
    }


def summarize_backend(path: Path | None, label: str) -> dict[str, float | str]:
    rows = read_csv(path) if path else []
    ok = [r for r in rows if (r.get("status") or r.get("success") or "ok").lower() in {"ok", "success", "1", "true"}]
    use = ok or rows
    return {
        "backend": label,
        "cases": len(rows),
        "ttft_ms": mean([fnum(r.get("ttft_ms")) for r in use]),
        "latency_ms": mean([fnum(r.get("latency_ms")) for r in use]),
        "latency_p95_ms": mean([fnum(r.get("latency_p95_ms")) for r in use]),
        "throughput_tokens_s": mean([fnum(r.get("throughput_tokens_s")) for r in use]),
        "rss_mb": max([fnum(r.get("cpu_memory_peak_mb") or r.get("process_rss_peak_mb")) for r in use] or [0.0]),
        "disk_write_mb": max([fnum(r.get("disk_write_delta_mb")) for r in use] or [0.0]),
    }


def summarize_comparison(path: Path | None) -> list[dict[str, Any]]:
    rows = read_csv(path) if path else []
    if not rows:
        return []
    label_by_run = {
        "vllm": "vLLM",
        "lmcache_cpu": "LMCache CPU",
        "lmcache_disk": "LMCache Disk",
    }
    out: list[dict[str, Any]] = []
    for run, label in label_by_run.items():
        use = [r for r in rows if r.get("run") == run and fnum(r.get("success_rate"), 0.0) > 0]
        if not use:
            continue
        out.append({
            "backend": label,
            "cases": len(use),
            "ttft_ms": mean([fnum(r.get("ttft_ms")) for r in use]),
            "tpot_ms": mean([fnum(r.get("tpot_ms")) for r in use]),
            "latency_ms": mean([fnum(r.get("latency_ms")) for r in use]),
            "latency_p95_ms": mean([fnum(r.get("latency_p95_ms")) for r in use]),
            "throughput_tokens_s": mean([fnum(r.get("throughput_tokens_s")) for r in use]),
            "rss_mb": mean([fnum(r.get("process_rss_peak_mb") or r.get("cpu_memory_peak_mb")) for r in use]),
            "disk_read_mb": mean([fnum(r.get("disk_read_delta_mb")) for r in use]),
            "disk_write_mb": mean([fnum(r.get("disk_write_delta_mb")) for r in use]),
        })
    return out


def backend_display(name: str) -> str:
    if name in BACKEND_COLORS:
        return name
    return BACKEND_LABELS.get(name, name)


def summarize_quality(path: Path | None) -> dict[str, Any]:
    rows = read_csv(path) if path else []
    ok = [r for r in rows if (r.get("status") or "ok").lower() == "ok"]
    use = ok or rows
    total = len(use)
    exact = sum(1 for r in use if fnum(r.get("exact_match")) > 0)
    normalized = sum(1 for r in use if fnum(r.get("normalized_match")) > 0)
    token_div = mean([fnum(r.get("token_divergence_rate")) for r in use])
    return {
        "rows": total,
        "exact_match_rate": exact / total if total else 0.0,
        "normalized_match_rate": normalized / total if total else 0.0,
        "token_divergence_rate": token_div,
    }


def find_boundary_benchmark(path: Path | None, backend: str) -> Path | None:
    if not path:
        return None
    matches = sorted(path.glob(f"02_boundary_32k/{backend}/*/benchmark_results.csv"))
    return matches[0] if matches else None


def summarize_boundary_32k(path: Path | None) -> list[dict[str, Any]]:
    specs = [
        ("vllm", "vLLM"),
        ("lmcache_cpu", "LMCache CPU"),
        ("lmcache_disk", "LMCache Disk"),
    ]
    rows: list[dict[str, Any]] = []
    for backend_dir, label in specs:
        csv_path = find_boundary_benchmark(path, backend_dir)
        raw = read_csv(csv_path) if csv_path else []
        selected = [
            r for r in raw
            if int(fnum(r.get("batch_size"))) == 16 and int(fnum(r.get("context_length"))) == 32768
        ]
        use = selected or raw[-1:]
        if not use:
            continue
        r = use[0]
        rows.append({
            "backend": label,
            "ttft_ms": fnum(r.get("ttft_ms")),
            "tpot_ms": fnum(r.get("tpot_ms")),
            "p95_ms": fnum(r.get("latency_p95_ms")),
            "rss_mb": fnum(r.get("process_rss_peak_mb") or r.get("cpu_memory_peak_mb")),
            "disk_write_mb": fnum(r.get("disk_write_delta_mb")),
            "case": r.get("case", ""),
        })
    return rows


def parse_markdown_table(lines: list[str], header_starts: str) -> list[dict[str, str]]:
    for idx, line in enumerate(lines):
        if not line.strip().startswith(header_starts):
            continue
        if idx + 1 >= len(lines) or "---" not in lines[idx + 1]:
            continue
        headers = [h.strip() for h in line.strip().strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for raw in lines[idx + 2:]:
            if not raw.strip().startswith("|"):
                break
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
        return rows
    return []


def summarize_boundary_report(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    table = parse_markdown_table(lines, "| run | success | TTFT ms")
    labels = {"vllm": "vLLM", "lmcache_cpu": "LMCache CPU", "lmcache_disk": "LMCache Disk"}
    out: list[dict[str, Any]] = []
    for row in table:
        run = row.get("run", "")
        if run not in labels:
            continue
        out.append({
            "backend": labels[run],
            "success": fnum(row.get("success")),
            "ttft_ms": fnum(row.get("TTFT ms")),
            "tpot_ms": fnum(row.get("TPOT ms")),
            "p95_ms": fnum(row.get("latency p95 ms")),
            "rss_mb": fnum(row.get("RSS MB")),
            "disk_read_mb": fnum(row.get("disk read MB")),
            "disk_write_mb": fnum(row.get("disk write MB")),
            "case": "final_report_32k_b16_out256",
        })
    return out


def vm_metrics(paths: dict[str, Path | None]) -> dict[str, Any]:
    mmap_path = paths.get("cli_mmap") if paths.get("cli_mmap") and paths["cli_mmap"].exists() else paths.get("mmap")
    mmap = read_json(mmap_path) if mmap_path else {}
    dgx = read_json(paths["dgx_vm"]) if paths["dgx_vm"] else {}
    cold_ms = first_num(
        nested_get(mmap, "latency", "cold_read_ms"),
        mmap.get("cold_read_ms"),
        fnum(mmap.get("cold_read_us"), default=math.nan) / 1000.0,
    )
    warm_ms = first_num(
        nested_get(mmap, "latency", "warm_read_ms"),
        mmap.get("warm_read_ms"),
        fnum(mmap.get("warm_read_us"), default=math.nan) / 1000.0,
    )
    cold_us = first_num(dgx.get("cold_read_us"), nested_get(dgx, "latency", "cold_read_us"))
    warm_us = first_num(dgx.get("warm_read_us"), nested_get(dgx, "latency", "warm_read_us"))
    resident = first_num(nested_get(dgx, "stats", "resident_ratio"), dgx.get("resident_ratio"))
    return {
        "mmap": mmap,
        "dgx_vm": dgx,
        "mmap_cold_ms": cold_ms,
        "mmap_warm_ms": warm_ms,
        "mmap_speedup": cold_ms / warm_ms if warm_ms else 0.0,
        "dgx_cold_us": cold_us,
        "dgx_warm_us": warm_us,
        "dgx_resident_ratio": resident,
        "dgx_prefetch_requests": fnum(nested_get(dgx, "stats", "prefetch_requests")),
        "dgx_evict_requests": fnum(nested_get(dgx, "stats", "evict_requests")),
    }


def plot_ch10_1_flow(out: Path) -> dict[str, Any]:
    plt = ensure_matplotlib()
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(16.0, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # 底层主面板
    panel = FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.90,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
        zorder=0,
    )
    ax.add_patch(panel)

    # 绘制上下层的背景区域
    ax.add_patch(Rectangle((0.04, 0.50), 0.92, 0.28, facecolor="#f8fbff", edgecolor="#e2e8f0", linewidth=1.0, zorder=0, alpha=0.95))
    ax.add_patch(Rectangle((0.04, 0.12), 0.92, 0.28, facecolor="#fbf9f4", edgecolor="#e2e8f0", linewidth=1.0, zorder=0, alpha=0.85))

    # 主标题与副标题
    ax.text(0.04, 0.90, "图 10-1 DGX 真实实验总体流程", fontsize=16, weight="bold", color="#0f172a")
    ax.text(
        0.04,
        0.85,
        "上层展示端到端运行链路，下层展示证据沉淀、策略分析与虚拟内存验证。",
        fontsize=10.5,
        color="#475569",
    )

    def tag(x: float, y: float, text: str, fc: str, ec: str) -> None:
        ax.text(
            x,
            y,
            text,
            ha="left",
            va="top",  # 改为 top，紧贴背景区域左上角往下渲染
            fontsize=11.5,
            weight="bold",
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15", fc=fc, ec=ec, lw=1.2),
            zorder=5   # 提高渲染层级，绝对防止被卡片遮盖
        )

    # 标签现在严格贴合各自背景区域的左上角 (Y轴分别是 0.775 和 0.395)
    tag(0.045, 0.775, "在线执行链路", "#eef2ff", "#a5b4fc")
    tag(0.045, 0.395, "离线分析与验证", "#fef3c7", "#fde047")

    top_cards = [
        {"title": "请求集", "subtitle": "上下文 / batch / 输出", "fc": "#eef2ff", "accent": "#7c8db5"},
        {"title": "后端执行", "subtitle": "vLLM / CPU / Disk", "fc": "#eff6ff", "accent": "#5b8def"},
        {"title": "性能基准", "subtitle": "TTFT / TPOT / P95", "fc": "#e0f2fe", "accent": "#56a3d9"},
        {"title": "缓存事件", "subtitle": "Trace / Cache", "fc": "#ecfdf5", "accent": "#5ea77a"},
    ]
    bottom_cards = [
        {"title": "策略分析", "subtitle": "ProfileDB / chunk score", "fc": "#fff7ed", "accent": "#d59a5e"},
        {"title": "报告归档", "subtitle": "图表 / 结论", "fc": "#f8fafc", "accent": "#9ca3af"},
        {"title": "虚拟内存", "subtitle": "mmap / madvise", "fc": "#f5f3ff", "accent": "#9b8cf2"},
    ]

    top_w, top_h = 0.145, 0.15
    top_y = 0.565
    top_xs = [0.09, 0.315, 0.54, 0.765]

    for idx, card in enumerate(top_cards):
        x = top_xs[idx]
        draw_flow_card(ax, (x, top_y), (top_w, top_h), card["title"], card["subtitle"], card["fc"], card["accent"])
        if idx < len(top_cards) - 1:
            draw_arrow(ax, (x + top_w, top_y + top_h / 2), (top_xs[idx + 1], top_y + top_h / 2), color="#64748b")

    bottom_w, bottom_h = 0.145, 0.15
    bottom_y = 0.185
    bottom_xs = [0.17, 0.4275, 0.685]

    for idx, card in enumerate(bottom_cards):
        x = bottom_xs[idx]
        draw_flow_card(ax, (x, bottom_y), (bottom_w, bottom_h), card["title"], card["subtitle"], card["fc"], card["accent"])
        if idx < len(bottom_cards) - 1:
            draw_arrow(ax, (x + bottom_w, bottom_y + bottom_h / 2), (bottom_xs[idx + 1], bottom_y + bottom_h / 2), color="#64748b")

    # 分界虚线
    ax.plot([0.06, 0.94], [0.45, 0.45], color="#cbd5e1", lw=1.5, linestyle="--", alpha=0.8, zorder=1)

    trans_x, trans_y = 0.50, 0.45
    ax.text(
        trans_x,
        trans_y,
        "Trace / Cache 数据进入策略分析层",
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", fc="#ffffff", ec="#cbd5e1", lw=1.2),
        zorder=3
    )

    draw_arrow(ax, (top_xs[-1] + top_w / 2, top_y), (trans_x + 0.14, trans_y), color="#94a3b8")
    draw_arrow(ax, (trans_x - 0.14, trans_y), (bottom_xs[0] + bottom_w / 2, bottom_y + bottom_h), color="#94a3b8")

    savefig(fig, out / "10-1_dgx_experiment_flow.png")
    return {"nodes": len(top_cards) + len(bottom_cards)}


def plot_ch10_2_baseline(rows: list[dict[str, Any]], out: Path) -> None:
    plt = ensure_matplotlib()
    labels = [str(r["backend"]) for r in rows]
    metrics = [
        ("ttft_ms", "TTFT（ms）", "平均 TTFT"),
        ("latency_ms", "Latency（ms）", "平均端到端延迟"),
        ("rss_mb", "RSS（MB）", "峰值 RSS"),
        ("disk_write_mb", "写盘量（MB）", "写盘量"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (key, ylabel, title) in zip(axes.ravel(), metrics):
        values = [float(r[key]) for r in rows]
        bars = ax.bar(labels, values, color=bar_colors(labels), width=0.58)
        ax.set_title(title)
        style_axis(ax, ylabel)
        ax.tick_params(axis="x", rotation=12)
        annotate_bars(ax, bars, digits=1)
    fig.suptitle("图 10-2 基线对比实验性能与资源结果", fontsize=15, weight="bold")
    savefig(fig, out / "10-2_baseline_perf_resource.png")


def plot_ch10_3_prefetch(summary: list[dict[str, Any]], out: Path) -> None:
    if not summary:
        return
    plt = ensure_matplotlib()
    labels = [str(r["context"]) for r in summary]
    x = list(range(len(labels)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specs = [
        ("no_prefetch_ttft_ms", "prefetch_ttft_ms", "ttft_delta_pct", "TTFT（ms）", "TTFT 对比"),
        ("no_prefetch_latency_ms", "prefetch_latency_ms", "latency_delta_pct", "端到端延迟（ms）", "端到端延迟对比"),
    ]
    for ax, (base_key, pre_key, delta_key, ylabel, title) in zip(axes, specs):
        no_vals = [r[base_key] for r in summary]
        pre_vals = [r[pre_key] for r in summary]
        b1 = ax.bar([i - width / 2 for i in x], no_vals, width, label="无 Prefetch", color=BACKEND_COLORS["无 Prefetch"])
        b2 = ax.bar([i + width / 2 for i in x], pre_vals, width, label="Prefetch", color=BACKEND_COLORS["Prefetch"])
        ax.set_xticks(x, labels)
        ax.set_xlabel("上下文长度")
        ax.set_title(title)
        style_axis(ax, ylabel)
        annotate_bars(ax, b1, digits=1)
        annotate_bars(ax, b2, digits=1)
        ymax = ax.get_ylim()[1]
        for i, r in enumerate(summary):
            y = max(no_vals[i], pre_vals[i]) + ymax * 0.10
            ax.text(i, y, fmt(r[delta_key], 1, "%"), ha="center", color=BACKEND_COLORS["Prefetch"], fontsize=9, weight="bold")
        ax.legend()
    fig.suptitle("图 10-3 选择性预取的 TTFT 与端到端延迟对比", fontsize=15, weight="bold")
    savefig(fig, out / "10-3_selective_prefetch_ttft_latency.png")


def plot_ch10_4_stress(stress: list[dict[str, Any]], out: Path) -> None:
    rows = [r for r in stress if r["scenario"] in {"Official", "Extreme16K"}]
    if not rows:
        return
    plt = ensure_matplotlib()
    scenarios = ["Official", "Extreme16K"]
    backends = ["vLLM", "LMCache CPU", "LMCache Disk"]
    x = list(range(len(scenarios)))
    
    # 每个后端占据的总宽度槽位
    width = 0.24 
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # --- 左侧子图：最差 P95 延迟 ---
    # 为左侧柱子稍微缩减一点宽度（原槽宽的 92%），留出明确的物理缝隙，避免边缘像素粘连
    bar_width_left = width * 0.92
    
    for bidx, backend in enumerate(backends):
        vals = []
        for scenario in scenarios:
            row = next((r for r in rows if r["scenario"] == scenario and backend_display(str(r["backend"])) == backend), None)
            vals.append(row["worst_p95_ms"] if row else 0)
        
        # 坐标中心点保持不变： i + (bidx - 1) * width，但传入缩减后的 bar_width_left
        axes[0].bar([i + (bidx - 1) * width for i in x], vals, bar_width_left, label=backend, color=BACKEND_COLORS[backend])
    
    axes[0].set_xticks(x, ["常规压力", "极限压力"])
    axes[0].set_title("最差 P95 延迟")
    style_axis(axes[0], "P95（ms）")
    axes[0].legend()

    # --- 右侧子图：RSS 与写盘量 ---
    # 将后端的总宽度分为两半，一半给RSS，一半给写盘量
    sub_width = width / 2.0
    # 同样缩减单根柱子的宽度，保持同后端两根柱子中间有一丝缝隙
    bar_width_right = sub_width * 0.92 
    
    for bidx, backend in enumerate(backends):
        rss_vals = []
        disk_vals = []
        for scenario in scenarios:
            row = next((r for r in rows if r["scenario"] == scenario and backend_display(str(r["backend"])) == backend), None)
            rss_vals.append(row["rss_mb"] if row else 0)
            disk_vals.append(row["disk_write_mb"] if row else 0)
        
        # 当前后端的基准中心坐标偏移
        offset = (bidx - 1) * width
        
        # RSS 往左偏半个子宽度，写盘往右偏半个子宽度
        axes[1].bar([i + offset - sub_width / 2 for i in x], rss_vals, bar_width_right, label=f"{backend} RSS", color=BACKEND_COLORS[backend], alpha=0.90)
        axes[1].bar([i + offset + sub_width / 2 for i in x], disk_vals, bar_width_right, label=f"{backend} 写盘", color=BACKEND_COLORS[backend], alpha=0.42, hatch="//")
        
    axes[1].set_xticks(x, ["常规压力", "极限压力"])
    axes[1].set_title("RSS 与写盘量")
    axes[1].set_yscale("symlog", linthresh=10)
    style_axis(axes[1], "MB（对数刻度）")
    axes[1].legend(fontsize=7, ncol=2)
    
    fig.suptitle("图 10-4 分级压力测试结果", fontsize=15, weight="bold")
    savefig(fig, out / "10-4_stress_hierarchy.png")


def plot_ch10_5_boundary_32k(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    plt = ensure_matplotlib()
    labels = [r["backend"] for r in rows]
    metrics = [
        ("ttft_ms", "TTFT（ms）", "TTFT"),
        ("tpot_ms", "TPOT（ms/token）", "TPOT"),
        ("p95_ms", "P95 延迟（ms）", "P95 延迟"),
        ("rss_mb", "RSS（MB）", "RSS"),
        ("disk_write_mb", "写盘量（MB）", "写盘量"),
    ]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    for ax, (key, ylabel, title) in zip(axes, metrics):
        values = [r[key] for r in rows]
        bars = ax.bar(labels, values, color=bar_colors(labels), width=0.58)
        ax.set_title(title)
        if key == "disk_write_mb":
            ax.set_yscale("symlog", linthresh=10)
        style_axis(ax, ylabel)
        ax.tick_params(axis="x", rotation=12)
        annotate_bars(ax, bars, digits=1)
    fig.suptitle("图 10-5 32K 长上下文边界实验性能对比（batch=16, context=32768）", fontsize=15, weight="bold")
    savefig(fig, out / "10-5_boundary_32k_five_panel.png")


def plot_ch10_6_boundary_threshold(boundary: dict[str, Any], out: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0.145, 0.165)
    ax.set_ylim(0, 1)
    ax.axvspan(0.145, 0.15, color="#fee2e2", alpha=0.9)
    ax.axvspan(0.16, 0.165, color="#dcfce7", alpha=0.9)
    ax.hlines(0.5, 0.145, 0.165, color="#6b7280", linewidth=2)
    ax.scatter([0.15], [0.5], s=260, marker="x", color="#dc2626", linewidths=4, label="0.15 启动失败")
    ax.scatter([0.16], [0.5], s=260, marker="o", color="#16a34a", linewidths=2, label="0.16 可运行")
    ax.text(0.15, 0.66, "0.15\n失败", ha="center", color="#991b1b", fontsize=12, weight="bold")
    ax.text(0.16, 0.66, "0.16\n可运行", ha="center", color="#166534", fontsize=12, weight="bold")
    fail = boundary.get("g015", {})
    available = fail.get("available_kv_gib", "1.67")
    ax.text(0.155, 0.20, f"可用 KV Cache {available} GiB < 所需 1.75 GiB\n估计最大长度：{fail.get('estimated_max_len', '31328')}", ha="center", fontsize=11)
    ax.set_xlabel("GPU 显存利用率参数")
    ax.set_yticks([])
    ax.set_title("图 10-6 32K 启动上下界示意图", fontsize=15, weight="bold")
    ax.legend(loc="upper center", ncol=2)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    savefig(fig, out / "10-6_boundary_startup_threshold.png")


def plot_ch10_7_policy_flow(paths: dict[str, Path | None], cache: dict[str, dict[str, int]], policy: dict[str, Any], out: Path) -> None:
    trace_count = 0
    trace_path = paths.get("trace_events")
    if trace_path and trace_path.exists():
        trace_count = sum(1 for _ in trace_path.open(encoding="utf-8"))
    chunk_actions = Counter(policy.get("chunk_actions", {}))
    load_path = paths.get("load_recompute")
    load_actions = Counter(r.get("action", "unknown") for r in read_csv(load_path))
    object_path = paths.get("object_scheduler")
    object_actions = Counter(r.get("action", "unknown") for r in read_csv(object_path))
    cache_total = sum(cache.get("prefetch", {}).values()) + sum(cache.get("boundary_disk", {}).values())
    
    plt = ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=(13.6, 3.05))
    ax.set_xlim(0, 0.96)
    ax.set_ylim(0.30, 0.92)
    ax.axis("off")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    
    # 定义丰富的色彩主题 (背景色, 边框色)，使图表更具活力
    colors = {
        "req": ("#e0f2fe", "#7dd3fc"),       # 浅蓝
        "trace": ("#f3e8ff", "#d8b4fe"),     # 浅紫
        "profile": ("#fef9c3", "#fde047"),   # 浅黄
        "chunk": ("#dcfce7", "#86efac"),     # 浅绿
        "load": ("#ffe4e6", "#fda4af"),      # 浅粉
        "sched": ("#ffedd5", "#fdba74"),     # 浅橙
        "cache": ("#e0e7ff", "#a5b4fc"),     # 浅靛
    }
    
    def draw_colored_box(x, y, w, h, text, fc, ec):
        from matplotlib.patches import FancyBboxPatch
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#1e293b", zorder=3)

    def draw_arrow_line(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b", shrinkA=3, shrinkB=3), zorder=1
        )

    # 计算顶部主流程节点坐标 (X坐标, Y中心点, 文本, 颜色配置)
    w, h = 0.11, 0.16
    main_y = 0.58
    nodes = [
        (0.02, main_y, "请求", colors["req"]),
        (0.16, main_y, f"Trace 事件\n{trace_count} 条", colors["trace"]),
        (0.30, main_y, "画像库\n复用/命中画像", colors["profile"]),
        (0.44, main_y, f"Chunk 评分\n{sum(chunk_actions.values())} 个", colors["chunk"]),
        (0.62, 0.70, f"加载/重算\n加载 {load_actions.get('load', 0)}", colors["load"]),
        (0.62, 0.46, f"对象调度\nkeep {object_actions.get('keep', 0)} / drop {object_actions.get('drop', 0)}", colors["sched"]),
        (0.83, main_y, f"缓存事件\n{cache_total} 条", colors["cache"]),
    ]
    
    # 绘制顶部节点
    for x, cy, text, (fc, ec) in nodes:
        draw_colored_box(x, cy - h / 2, w, h, text, fc, ec)
        
    # 主线连接 (0->1, 1->2, 2->3)
    for i in range(3):
        draw_arrow_line(nodes[i][0] + w, nodes[i][1], nodes[i+1][0], nodes[i+1][1])
        
    # 分支连接 (3 -> 4, 3 -> 5)
    draw_arrow_line(nodes[3][0] + w, nodes[3][1], nodes[4][0], nodes[4][1])
    draw_arrow_line(nodes[3][0] + w, nodes[3][1], nodes[5][0], nodes[5][1])
    
    # 汇聚连接 (4 -> 6, 5 -> 6)
    draw_arrow_line(nodes[4][0] + w, nodes[4][1], nodes[6][0], nodes[6][1])
    draw_arrow_line(nodes[5][0] + w, nodes[5][1], nodes[6][0], nodes[6][1])

        
    # 补充标题与注释
    ax.text(0.02, 0.88, "图 10-7 缓存事件与策略链流程图", fontsize=15, weight="bold", color="#0f172a")
    
    sub_text = f"策略分支：chunk offload {chunk_actions.get('offload', 0)} / drop {chunk_actions.get('drop', 0)}；调度 keep {object_actions.get('keep', 0)} / drop {object_actions.get('drop', 0)}"
    # 将副文本的 X 锚点设置在最右侧节点的右边缘对齐 (0.94)
    ax.text(0.94, 0.88, sub_text, fontsize=10.5, color="#475569", ha="right")
    
    fig.savefig(out / "10-7_cache_policy_flow.png", dpi=180, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_ch10_8_vm(paths: dict[str, Path | None], out: Path) -> dict[str, Any]:
    metrics = vm_metrics(paths)
    plt = ensure_matplotlib()
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1])
    
    # ================= 1. 左侧机制路径 =================
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    
    # 调整1：为四个方框配置更加鲜艳、有区分度的背景色与边框色
    flow_nodes = [
        (0.15, 0.78, "后备存储\n文件映射", "#e0f2fe", "#7dd3fc"), # 浅蓝
        (0.15, 0.55, "驻留页\nmincore 统计", "#f3e8ff", "#d8b4fe"), # 浅紫
        (0.15, 0.32, "冷热读取\n缺页 / 命中", "#dcfce7", "#86efac"), # 浅绿
        (0.15, 0.10, "预取 / 换出\nmadvise", "#ffedd5", "#fdba74"), # 浅橙
    ]
    for x, y, text, fc, ec in flow_nodes:
        draw_box(ax_flow, (x, y), (0.70, 0.14), text, fc=fc, ec=ec)
    for i in range(len(flow_nodes) - 1):
        draw_arrow(ax_flow, (0.50, flow_nodes[i][1]), (0.50, flow_nodes[i + 1][1] + 0.14))
    ax_flow.set_title("机制路径")

    # ================= 2. mmap KV 缓存 =================
    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(["冷读", "热读"], [metrics["mmap_cold_ms"], metrics["mmap_warm_ms"]], color=["#cc6677", "#228833"])
    ax1.set_title("mmap KV 缓存")
    style_axis(ax1, "ms")
    annotate_bars(ax1, bars, digits=2)
    
    # 调整2：将 "2.0x 加速" 的 X 轴坐标从 0.5 改为 0.75，使其完美居中悬浮于右侧绿色柱子上方
    ax1.text(0.75, 0.86, f"{metrics['mmap_speedup']:.1f}x 加速", transform=ax1.transAxes, ha="center", color="#166534", weight="bold")

    # ================= 3. 其他图表 =================
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(["驻留率"], [metrics["dgx_resident_ratio"] * 100], color=BACKEND_COLORS["Prefetch"])
    ax2.set_title("DGX Spark 虚拟内存")
    style_axis(ax2, "%")
    ax2.set_ylim(0, max(20, metrics["dgx_resident_ratio"] * 120))
    annotate_bars(ax2, bars, suffix="%", digits=1)

    ax3 = fig.add_subplot(gs[0, 3])
    bars = ax3.bar(["预取", "换出"], [metrics["dgx_prefetch_requests"], metrics["dgx_evict_requests"]], color=["#7b61ff", "#6b7280"])
    ax3.set_title("VM 操作计数")
    style_axis(ax3, "次")
    annotate_bars(ax3, bars, digits=0)

    ax4 = fig.add_subplot(gs[1, 1:3])
    bars = ax4.bar(["DGX 冷读", "DGX 热读"], [metrics["dgx_cold_us"], metrics["dgx_warm_us"]], color=["#cc6677", "#228833"])
    ax4.set_title("DGX Spark 读取延迟")
    style_axis(ax4, "us")
    annotate_bars(ax4, bars, digits=1)

    # ================= 4. 右下角结论文本 =================
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis("off")
    
    # 调整3：强制使用顶部对齐 (va="top")，并拉大 Y 轴坐标间距，彻底防止文本挤压重叠
    ax5.text(0.05, 0.90, "结论", fontsize=13, weight="bold", va="top")
    ax5.text(0.05, 0.70, "文件后备 mmap\nMADV_WILLNEED 预取\nMADV_DONTNEED 换出\nmincore 驻留率", fontsize=11, linespacing=1.8, va="top")
    
    fig.suptitle("图 10-8 OS 虚拟内存机制实验结果图", fontsize=15, weight="bold")
    savefig(fig, out / "10-8_os_vm_mechanism_results.png")
    return metrics


def plot_ch10_9_quality(paths: dict[str, Path | None], out: Path) -> dict[str, Any]:
    summary = summarize_quality(paths["quality"])
    plt = ensure_matplotlib()
    labels = ["精确匹配", "归一化匹配", "Token 分歧率"]
    values = [
        summary["exact_match_rate"] * 100,
        summary["normalized_match_rate"] * 100,
        summary["token_divergence_rate"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=["#1f4e79", "#7b61ff", "#cc6677"], width=0.55)
    style_axis(ax, "%")
    ax.set_ylim(0, max(values) * 1.35 if values else 1)
    annotate_bars(ax, bars, suffix="%", digits=1)
    ax.set_title("图 10-9 输出一致性结果图", fontsize=15, weight="bold")
    ax.text(0.02, 0.92, f"样本数：{summary['rows']}", transform=ax.transAxes, fontsize=10, color="#4b5563")
    savefig(fig, out / "10-9_output_consistency.png")
    return summary


def load_boundary_heatmap_rows(boundary_root: Path | None) -> list[dict[str, Any]]:
    if not boundary_root:
        return []
    specs = [
        ("vllm", "vLLM"),
        ("lmcache_cpu", "LMCache CPU"),
        ("lmcache_disk", "LMCache Disk"),
    ]
    rows: list[dict[str, Any]] = []
    for backend_dir, label in specs:
        csv_path = find_boundary_benchmark(boundary_root, backend_dir)
        for row in read_csv(csv_path) if csv_path else []:
            request_count = fnum(row.get("request_count"))
            success_count = fnum(row.get("success_count"))
            rows.append({
                "backend": label,
                "case": row.get("case", ""),
                "context_length": int(fnum(row.get("context_length"))),
                "batch_size": int(fnum(row.get("batch_size"))),
                "output_tokens": int(fnum(row.get("output_tokens"))),
                "success_rate": success_count / request_count if request_count else (1.0 if row.get("status") == "ok" else 0.0),
                "latency_p95_ms": fnum(row.get("latency_p95_ms")),
                "ttft_ms": fnum(row.get("ttft_ms")),
                "tpot_ms": fnum(row.get("tpot_ms")),
                "disk_write_delta_mb": fnum(row.get("disk_write_delta_mb")),
                "process_rss_peak_mb": fnum(row.get("process_rss_peak_mb") or row.get("cpu_memory_peak_mb")),
            })
    return rows


def heat_matrix(rows: list[dict[str, Any]], backend: str, metric: str) -> tuple[list[int], list[int], list[list[float]]]:
    use = [r for r in rows if r.get("backend") == backend]
    contexts = sorted({int(r["context_length"]) for r in use})
    batches = sorted({int(r["batch_size"]) for r in use})
    lookup = {(int(r["batch_size"]), int(r["context_length"])): float(r.get(metric, 0.0)) for r in use}
    matrix = [[lookup.get((batch, context), math.nan) for context in contexts] for batch in batches]
    return contexts, batches, matrix


def annotate_heatmap(
    ax: Any,
    matrix: list[list[float]],
    fmt_spec: str,
    image: Any | None = None,
    suffix: str = "",
) -> None:
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if math.isnan(value):
                text = "-"
                color = "#111827"
            else:
                text = f"{format(value, fmt_spec)}{suffix}"
                color = "#111827"
                if image is not None:
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = "#f9fafb" if luminance < 0.42 else "#111827"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color)


def plot_single_heatmap(
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    colorbar_label: str,
    output_file: str,
    out: Path,
    cmap: str,
    fmt_spec: str,
    log_scale: bool = False,
) -> None:
    if not rows:
        return
    import numpy as np
    from matplotlib.colors import LogNorm

    plt = ensure_matplotlib()
    backends = ["vLLM", "LMCache CPU", "LMCache Disk"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
    image = None
    positive_values: list[float] = []
    for backend in backends:
        _, _, matrix = heat_matrix(rows, backend, metric)
        for row in matrix:
            positive_values.extend([v for v in row if not math.isnan(v) and v > 0])
    norm = None
    if log_scale and positive_values:
        norm = LogNorm(vmin=max(min(positive_values), 1e-3), vmax=max(positive_values))
    for ax, backend in zip(axes, backends):
        contexts, batches, matrix = heat_matrix(rows, backend, metric)
        arr = np.array(matrix, dtype=float) if matrix else np.empty((0, 0))
        image = ax.imshow(arr, cmap=cmap, aspect="auto", norm=norm)
        ax.set_title(backend)
        ax.set_xticks(range(len(contexts)), [str(c) for c in contexts], rotation=30, ha="right")
        ax.set_yticks(range(len(batches)), [str(b) for b in batches])
        ax.set_xlabel("context")
        if ax is axes[0]:
            ax.set_ylabel("batch")
        annotate_heatmap(ax, matrix, fmt_spec, image)
    if image is not None:
        cax = fig.add_axes([0.925, 0.20, 0.018, 0.58])
        fig.colorbar(image, cax=cax, label=colorbar_label)
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.06, right=0.90, wspace=0.22)
    fig.savefig(out / output_file, dpi=180)
    plt.close(fig)


def plot_disk_write_offload_heatmap(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    import numpy as np
    from matplotlib.colors import LogNorm

    plt = ensure_matplotlib()
    contexts, batches, disk_matrix = heat_matrix(rows, "LMCache Disk", "disk_write_delta_mb")
    _, _, vllm_matrix = heat_matrix(rows, "vLLM", "disk_write_delta_mb")
    if not disk_matrix:
        return

    ratio_matrix: list[list[float]] = []
    for disk_row, vllm_row in zip(disk_matrix, vllm_matrix):
        ratio_row: list[float] = []
        for disk_value, vllm_value in zip(disk_row, vllm_row):
            if math.isnan(disk_value) or math.isnan(vllm_value) or vllm_value <= 0:
                ratio_row.append(math.nan)
            else:
                ratio_row.append(disk_value / vllm_value)
        ratio_matrix.append(ratio_row)

    disk_arr = np.array(disk_matrix, dtype=float)
    ratio_arr = np.array(ratio_matrix, dtype=float)
    ratio_values = [v for row in ratio_matrix for v in row if not math.isnan(v) and v > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)

    disk_image = axes[0].imshow(disk_arr, cmap="Blues", aspect="auto")
    axes[0].set_title("LMCache Disk 写盘量")
    axes[0].set_xticks(range(len(contexts)), [str(c) for c in contexts], rotation=30, ha="right")
    axes[0].set_yticks(range(len(batches)), [str(b) for b in batches])
    axes[0].set_xlabel("context")
    axes[0].set_ylabel("batch")
    annotate_heatmap(axes[0], disk_matrix, ".1f", disk_image)

    ratio_norm = None
    if ratio_values:
        ratio_norm = LogNorm(vmin=max(min(ratio_values), 1e-3), vmax=max(ratio_values))
    ratio_image = axes[1].imshow(ratio_arr, cmap="PuRd", aspect="auto", norm=ratio_norm)
    axes[1].set_title("相对 vLLM 写盘放大倍数")
    axes[1].set_xticks(range(len(contexts)), [str(c) for c in contexts], rotation=30, ha="right")
    axes[1].set_yticks(range(len(batches)), [str(b) for b in batches])
    axes[1].set_xlabel("context")
    annotate_heatmap(axes[1], ratio_matrix, ".0f", ratio_image, suffix="x")

    cax_disk = fig.add_axes([0.47, 0.20, 0.016, 0.56])
    fig.colorbar(disk_image, cax=cax_disk, label="disk write MB")
    cax_ratio = fig.add_axes([0.925, 0.20, 0.016, 0.56])
    fig.colorbar(ratio_image, cax=cax_ratio, label="Disk / vLLM write")

    fig.suptitle("图 10-12 Context × Batch Disk Offload 写盘证据（gpu_memory_utilization=0.16）", fontsize=15, weight="bold")
    fig.text(
        0.06,
        0.06,
        "注：vLLM/LMCache CPU 写盘通常只有约 1-4 MB；本图突出 LMCache Disk tier 的真实写盘与相对放大倍数。",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.07, right=0.90, wspace=0.34)
    fig.savefig(out / "10-12_disk_write_heatmap.png", dpi=180)
    plt.close(fig)


def plot_ch10_heatmaps(boundary_root: Path | None, out: Path) -> list[dict[str, Any]]:
    rows = load_boundary_heatmap_rows(boundary_root)
    if not rows:
        return []
    plot_single_heatmap(
        rows,
        "ttft_ms",
        "图 10-10 Context × Batch TTFT 热力图（gpu_memory_utilization=0.16）",
        "TTFT ms",
        "10-10_ttft_heatmap.png",
        out,
        "YlGnBu",
        ".0f",
        log_scale=True,
    )
    plot_single_heatmap(
        rows,
        "latency_p95_ms",
        "图 10-11 Context × Batch P95 延迟热力图（gpu_memory_utilization=0.16）",
        "P95 latency ms",
        "10-11_p95_latency_heatmap.png",
        out,
        "YlOrRd",
        ".0f",
        log_scale=True,
    )
    plot_disk_write_offload_heatmap(rows, out)
    return rows


def plot_baseline(paths: dict[str, Path | None], out: Path) -> list[dict[str, Any]]:
    rows = [
        summarize_backend(paths["vllm_baseline"], "vLLM"),
        summarize_backend(paths["lmcache_cpu_baseline"], "LMCache CPU"),
        summarize_backend(paths["lmcache_disk_baseline"], "LMCache Disk"),
    ]
    plt = ensure_matplotlib()
    labels = [str(r["backend"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(labels))
    width = 0.25
    axes[0].bar([i - width for i in x], [float(r["ttft_ms"]) for r in rows], width, label="TTFT")
    axes[0].bar(list(x), [float(r["latency_ms"]) for r in rows], width, label="Latency")
    axes[0].bar([i + width for i in x], [float(r["latency_p95_ms"]) for r in rows], width, label="P95")
    axes[0].set_xticks(list(x), labels, rotation=15, ha="right")
    axes[0].set_ylabel("ms")
    axes[0].set_title("Baseline latency metrics")
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], [float(r["rss_mb"]) for r in rows], width, label="Process RSS MB")
    axes[1].bar([i + width / 2 for i in x], [float(r["disk_write_mb"]) for r in rows], width, label="Disk write MB")
    axes[1].set_xticks(list(x), labels, rotation=15, ha="right")
    axes[1].set_ylabel("MB")
    axes[1].set_title("Resource trade-off")
    axes[1].legend()
    savefig(fig, out / "01_baseline_backend_comparison.png")
    return rows


def plot_prefetch(path: Path | None, out: Path) -> list[dict[str, Any]]:
    rows = read_csv(path) if path else []
    by_ctx: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_ctx.setdefault(int(fnum(row.get("context_length"))), []).append(row)
    summary = []
    for ctx, ctx_rows in sorted(by_ctx.items()):
        no_ttft = mean([fnum(r.get("no_prefetch_ttft_ms")) for r in ctx_rows])
        pre_ttft = mean([fnum(r.get("prefetch_demand_ttft_ms")) for r in ctx_rows])
        no_lat = mean([fnum(r.get("no_prefetch_latency_ms")) for r in ctx_rows])
        pre_lat = mean([fnum(r.get("prefetch_demand_latency_ms")) for r in ctx_rows])
        summary.append({
            "context": ctx,
            "no_prefetch_ttft_ms": no_ttft,
            "prefetch_ttft_ms": pre_ttft,
            "ttft_delta_pct": (pre_ttft - no_ttft) / no_ttft * 100.0 if no_ttft else 0.0,
            "no_prefetch_latency_ms": no_lat,
            "prefetch_latency_ms": pre_lat,
            "latency_delta_pct": (pre_lat - no_lat) / no_lat * 100.0 if no_lat else 0.0,
            "hit_rate": mean([fnum(r.get("prefetch_hit")) for r in ctx_rows]),
            "waste_rate": mean([fnum(r.get("prefetch_waste")) for r in ctx_rows]),
        })
    if not summary:
        return []
    plt = ensure_matplotlib()
    labels = [str(r["context"]) for r in summary]
    x = range(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar([i - width / 2 for i in x], [r["no_prefetch_ttft_ms"] for r in summary], width, label="No prefetch")
    axes[0].bar([i + width / 2 for i in x], [r["prefetch_ttft_ms"] for r in summary], width, label="Prefetch demand")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_xlabel("Context length")
    axes[0].set_ylabel("TTFT ms")
    axes[0].set_title("Selective prefetch TTFT")
    axes[0].legend()

    axes[1].plot(labels, [r["ttft_delta_pct"] for r in summary], marker="o", label="TTFT delta %")
    axes[1].plot(labels, [r["latency_delta_pct"] for r in summary], marker="o", label="Latency delta %")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Context length")
    axes[1].set_ylabel("Delta % (negative is better)")
    axes[1].set_title("Prefetch effect by context")
    axes[1].legend()
    savefig(fig, out / "02_prefetch_ttft_latency.png")
    return summary


def plot_stress(paths: dict[str, Path | None], out: Path) -> list[dict[str, Any]]:
    scenarios = [
        ("Official", paths["official_stress"]),
        ("Extreme16K", paths["extreme_stress"]),
        ("Boundary32K", paths["boundary_32k"]),
        ("G016_B32K", paths["boundary_pass"]),
    ]
    rows: list[dict[str, Any]] = []
    for scenario, path in scenarios:
        for row in read_csv(path) if path else []:
            rows.append({
                "scenario": scenario,
                "backend": row.get("run", ""),
                "success_rate": fnum(row.get("success_rate")),
                "max_context": fnum(row.get("max_success_context")),
                "max_batch": fnum(row.get("max_success_batch")),
                "worst_p95_ms": fnum(row.get("worst_latency_p95_ms")),
                "rss_mb": fnum(row.get("process_rss_peak_mb")),
                "disk_write_mb": fnum(row.get("disk_write_delta_mb")),
            })
    if not rows:
        return []
    plt = ensure_matplotlib()
    labels = [f"{r['scenario']}\n{r['backend']}" for r in rows]
    x = range(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 0.75), 8))
    axes[0].bar(list(x), [r["worst_p95_ms"] for r in rows])
    axes[0].set_ylabel("Worst P95 ms")
    axes[0].set_title("Stress and boundary latency")
    axes[0].set_yscale("log")
    axes[0].set_xticks(list(x), labels, rotation=45, ha="right")
    axes[1].bar(list(x), [r["disk_write_mb"] for r in rows], color="#4477aa")
    axes[1].set_ylabel("Disk write MB")
    axes[1].set_title("Stress and boundary disk write")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_xticks(list(x), labels, rotation=45, ha="right")
    savefig(fig, out / "03_stress_boundary_latency_disk.png")
    return rows


def parse_boundary_fail(log_path: Path | None) -> dict[str, Any]:
    if not log_path or not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    out: dict[str, Any] = {"log": str(log_path), "startup_failed": "ValueError" in text or "max seq len" in text}
    patterns = {
        "available_kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
        "required_kv_gib": r"Try increasing.*?KV cache.*?([0-9.]+) GiB|requires ([0-9.]+) GiB",
        "estimated_max_len": r"maximum number of tokens.*?([0-9,]+)|maximum model length.*?([0-9,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        value = next((g for g in match.groups() if g), "")
        out[key] = value.replace(",", "")
    return out


def plot_boundary_startup(paths: dict[str, Path | None], out: Path) -> dict[str, Any]:
    fail = parse_boundary_fail(paths["boundary_fail_log"])
    pass_rows = read_csv(paths["boundary_pass"]) if paths["boundary_pass"] else []
    pass_ok = all(fnum(r.get("success_rate")) >= 1.0 for r in pass_rows) if pass_rows else False
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["0.15 startup", "0.16 workload"], [0 if fail.get("startup_failed") else 1, 1 if pass_ok else 0], color=["#cc6677", "#228833"])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Pass = 1, Fail = 0")
    ax.set_title("32K boundary startup result")
    note = []
    if fail.get("available_kv_gib"):
        note.append(f"0.15 available KV: {fail['available_kv_gib']} GiB")
    if fail.get("required_kv_gib"):
        note.append(f"required: {fail['required_kv_gib']} GiB")
    if note:
        ax.text(0.02, 0.92, "\n".join(note), transform=ax.transAxes, va="top")
    savefig(fig, out / "04_boundary_startup_g015_g016.png")
    return {"g015": fail, "g016_success": pass_ok}


def event_counts(path: Path | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path or not path.exists():
        return counts
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event_type") or row.get("type") or row.get("event") or "unknown"
            counts[str(event)] += 1
    return counts


def plot_cache_events(paths: dict[str, Path | None], out: Path) -> dict[str, dict[str, int]]:
    series = {
        "boundary_disk": event_counts(paths["cache_events_boundary_disk"]),
        "prefetch": event_counts(paths["cache_events_prefetch"]),
    }
    wanted = sorted({k for counts in series.values() for k in counts if k.startswith("cache_") or k == "request_result"})
    if not wanted:
        return {k: dict(v) for k, v in series.items()}
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = list(series)
    bottoms = [0] * len(labels)
    for event in wanted:
        values = [series[label][event] for label in labels]
        ax.bar(labels, values, bottom=bottoms, label=event)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("Event count")
    ax.set_title("Cache event evidence")
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, out / "05_cache_event_counts.png")
    return {k: dict(v) for k, v in series.items()}


def plot_policy(paths: dict[str, Path | None], out: Path) -> dict[str, Any]:
    rows = read_csv(paths["policy_ablation"]) if paths["policy_ablation"] else []
    chunk_rows = read_csv(paths["chunk_scores"]) if paths["chunk_scores"] else []
    actions = Counter((r.get("action") or r.get("recommended_action") or r.get("decision") or "unknown") for r in chunk_rows)
    if not rows and not actions:
        return {}
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    policies = [r.get("policy", r.get("name", "")) for r in rows]
    ttft = [
        first_num(
            r.get("ttft_delta_pct"),
            r.get("ttft_vs_baseline_pct"),
            r.get("ttft_delta_vs_baseline_pct"),
            r.get("ttft_delta_pct_vs_baseline"),
        )
        for r in rows
    ]
    latency = [
        first_num(
            r.get("latency_delta_pct"),
            r.get("latency_vs_baseline_pct"),
            r.get("latency_delta_vs_baseline_pct"),
            r.get("latency_delta_pct_vs_baseline"),
        )
        for r in rows
    ]
    if policies:
        x = range(len(policies))
        width = 0.35
        axes[0].bar([i - width / 2 for i in x], ttft, width, label="TTFT delta %")
        axes[0].bar([i + width / 2 for i in x], latency, width, label="Latency delta %")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xticks(list(x), policies, rotation=35, ha="right")
        axes[0].set_title("Policy ablation")
        axes[0].set_ylabel("Delta %")
        axes[0].legend()
    if actions:
        axes[1].bar(list(actions), list(actions.values()), color="#66c2a5")
        axes[1].set_title("Chunk action counts")
        axes[1].set_ylabel("Chunks")
        axes[1].tick_params(axis="x", rotation=35)
    savefig(fig, out / "06_policy_ablation_actions.png")
    return {"policy_rows": rows, "chunk_actions": dict(actions)}


def plot_vm(paths: dict[str, Path | None], out: Path) -> dict[str, Any]:
    mmap = read_json(paths["mmap"]) if paths["mmap"] else {}
    dgx = read_json(paths["dgx_vm"]) if paths["dgx_vm"] else {}
    cold = first_num(
        mmap.get("cold_read_ms"),
        mmap.get("cold_read_latency_ms"),
        mmap.get("cold_read_block_ms"),
        nested_get(mmap, "latency", "cold_read_ms"),
        fnum(mmap.get("cold_read_us"), default=math.nan) / 1000.0,
        fnum(nested_get(mmap, "stats", "avg_cold_read_us"), default=math.nan) / 1000.0,
    )
    warm = first_num(
        mmap.get("warm_read_ms"),
        mmap.get("warm_read_latency_ms"),
        mmap.get("warm_read_block_ms"),
        nested_get(mmap, "latency", "warm_read_ms"),
        fnum(mmap.get("warm_read_us"), default=math.nan) / 1000.0,
        fnum(nested_get(mmap, "stats", "avg_warm_read_us"), default=math.nan) / 1000.0,
    )
    resident = first_num(
        dgx.get("resident_ratio"),
        nested_get(dgx, "stats", "resident_ratio"),
        mmap.get("resident_ratio_after_eviction"),
        nested_get(mmap, "stats", "resident_ratio"),
    )
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["cold", "warm"], [cold, warm], color=["#cc6677", "#228833"])
    axes[0].set_ylabel("ms")
    axes[0].set_title("mmap cold/warm read")
    axes[1].bar(["resident ratio"], [resident])
    axes[1].set_ylim(0, max(1.0, resident * 1.2))
    axes[1].set_title("VM resident evidence")
    savefig(fig, out / "07_vm_mmap_evidence.png")
    return {"mmap": mmap, "dgx_vm": dgx}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(out: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# AstraKV-W 第 10 章正文图报告",
        "",
        "## 图清单",
        "",
    ]
    for item in manifest["figures"]:
        lines.append(f"- `{item['file']}`: {item['title']}")
    lines += [
        "",
        "## 解读边界",
        "",
        "- 基线图只说明 vLLM、LMCache CPU、LMCache Disk 的性能与资源权衡，不应解读为 AstraKV 整体加速。",
        "- 选择性预取图主要支撑 TTFT 降低；端到端延迟是辅助指标。",
        "- 压力与 32K 边界图支撑可运行边界、P95、RSS、写盘量，不把不同量纲放在同一坐标轴。",
        "- 热力图展示 context length 与 batch size 对 TTFT、P95 延迟和写盘量的影响；成功率在该组 g016 边界数据中均为 1.0，因此不单独绘制成功率热力图。",
        "- 缓存事件/策略链图支撑 Trace 到策略决策的链路完整性。",
        "- VM/mmap 图是 OS 虚拟内存机制证据，不等同于真实 vLLM KV cache 已经 mmap 化。",
        "",
        "## 数据来源",
        "",
    ]
    for key, value in manifest["artifacts"].items():
        lines.append(f"- `{key}`: `{value or 'missing'}`")
    (out / "figure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-evidence", default="results/extended_evidence_20260625_014917")
    parser.add_argument("--demo-evidence", default="")
    parser.add_argument("--boundary-pass", default="results/extended_g016_ctx32k_b16_out256")
    parser.add_argument("--boundary-fail", default="results/extended_g015_ctx32k_b16_out256")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(
        Path(args.main_evidence),
        Path(args.demo_evidence) if args.demo_evidence else None,
        Path(args.boundary_pass) if args.boundary_pass else None,
        Path(args.boundary_fail) if args.boundary_fail else None,
    )

    manifest: dict[str, Any] = {
        "artifacts": {k: str(v) if v else "" for k, v in paths.items()},
        "figures": [],
        "summaries": {},
    }

    flow = plot_ch10_1_flow(out)
    manifest["figures"].append({"file": "10-1_dgx_experiment_flow.png", "title": "图 10-1 DGX 真实实验总体流程"})
    manifest["summaries"]["flow"] = flow

    baseline = summarize_comparison(paths["comparison"]) or plot_baseline(paths, out)
    write_csv(out / "baseline_summary.csv", baseline)
    plot_ch10_2_baseline(baseline, out)
    manifest["figures"].append({"file": "10-2_baseline_perf_resource.png", "title": "图 10-2 基线对比实验性能与资源结果"})
    manifest["summaries"]["baseline"] = baseline

    prefetch = plot_prefetch(paths["prefetch_results"], out)
    write_csv(out / "prefetch_summary.csv", prefetch)
    if prefetch:
        plot_ch10_3_prefetch(prefetch, out)
        manifest["figures"].append({"file": "10-3_selective_prefetch_ttft_latency.png", "title": "图 10-3 选择性预取的 TTFT 与端到端延迟对比"})
    manifest["summaries"]["prefetch"] = prefetch

    stress = plot_stress(paths, out)
    write_csv(out / "stress_boundary_summary.csv", stress)
    if stress:
        plot_ch10_4_stress(stress, out)
        manifest["figures"].append({"file": "10-4_stress_hierarchy.png", "title": "图 10-4 分级压力测试结果"})
    manifest["summaries"]["stress"] = stress

    boundary32k = summarize_boundary_report(paths.get("boundary_pass_report")) or summarize_boundary_32k(Path(args.boundary_pass) if args.boundary_pass else None)
    write_csv(out / "boundary32k_summary.csv", boundary32k)
    if boundary32k:
        plot_ch10_5_boundary_32k(boundary32k, out)
        manifest["figures"].append({"file": "10-5_boundary_32k_five_panel.png", "title": "图 10-5 32K 长上下文边界实验性能对比"})
    manifest["summaries"]["boundary32k"] = boundary32k

    boundary = plot_boundary_startup(paths, out)
    plot_ch10_6_boundary_threshold(boundary, out)
    manifest["figures"].append({"file": "10-6_boundary_startup_threshold.png", "title": "图 10-6 32K 启动上下界示意图"})
    manifest["summaries"]["boundary"] = boundary

    heatmap_rows = plot_ch10_heatmaps(Path(args.boundary_pass) if args.boundary_pass else None, out)
    write_csv(out / "heatmap_summary.csv", heatmap_rows)
    if heatmap_rows:
        manifest["figures"].extend([
            {"file": "10-10_ttft_heatmap.png", "title": "图 10-10 Context × Batch TTFT 热力图"},
            {"file": "10-11_p95_latency_heatmap.png", "title": "图 10-11 Context × Batch P95 延迟热力图"},
            {"file": "10-12_disk_write_heatmap.png", "title": "图 10-12 Context × Batch Disk Offload 写盘证据图"},
        ])
    manifest["summaries"]["heatmaps"] = heatmap_rows

    cache = plot_cache_events(paths, out)
    manifest["summaries"]["cache_events"] = cache

    policy = plot_policy(paths, out)
    if policy or cache:
        plot_ch10_7_policy_flow(paths, cache, policy, out)
        manifest["figures"].append({"file": "10-7_cache_policy_flow.png", "title": "图 10-7 缓存事件与策略链流程图"})
    manifest["summaries"]["policy"] = policy

    vm = plot_vm(paths, out)
    vm_ch10 = plot_ch10_8_vm(paths, out)
    manifest["figures"].append({"file": "10-8_os_vm_mechanism_results.png", "title": "图 10-8 OS 虚拟内存机制实验结果图"})
    manifest["summaries"]["vm"] = vm
    manifest["summaries"]["vm_ch10"] = vm_ch10

    quality = plot_ch10_9_quality(paths, out)
    manifest["figures"].append({"file": "10-9_output_consistency.png", "title": "图 10-9 输出一致性结果图"})
    manifest["summaries"]["quality"] = quality

    for old_png in out.glob("0[1-7]_*.png"):
        old_png.unlink()

    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(out, manifest)
    print(f"Figures written to {out}")
    print(f"Report written to {out / 'figure_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
