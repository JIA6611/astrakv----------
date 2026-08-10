"""Draw code-grounded KV-Core mechanism figures for the report.

The figures are structural diagrams, not experimental plots. They intentionally
avoid invented latency, capacity, hit-rate, or memory-saving numbers.

Usage:
    python scripts/reporting/draw_core_mechanism_figures.py --out-dir figures/core_mechanisms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


COLORS = {
    "ink": "#243447",
    "muted": "#65758B",
    "grid": "#D8E0EA",
    "identity": "#3B5BDB",
    "native": "#0B7285",
    "prefetch": "#2B8A3E",
    "action": "#E67700",
    "evidence": "#7048E8",
    "error": "#C92A2A",
    "surface": "#F8FAFC",
    "white": "#FFFFFF",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, width, height, text, color="native", *, fontsize=10, dashed=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=COLORS[color],
        facecolor=COLORS["white"] if not dashed else COLORS["surface"],
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=COLORS["ink"], fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end, color="ink", *, dashed=False, label=None, rad=0.0):
    style = "-|>"
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=1.5, color=COLORS[color],
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.07, label, ha="center", va="bottom",
                fontsize=8.5, color=COLORS["muted"])


def finish(ax, title: str, *, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"], pad=14)


def identity_chain(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    finish(ax, "图1  精确Token、LMCache原生键与请求绑定代际")
    labels = [
        ("Tokenizer / chat template\nexact_token_ids", "identity"),
        ("prefix_hash\n整数Token稳定哈希", "identity"),
        ("KVCompatibilityKey\n模型/精度/布局/粒度/层组", "identity"),
        ("TokenDatabase\nnative chunk keys", "native"),
        ("PhysicalKVObject\n控制面身份记录", "native"),
        ("RequestKVBinding\nnative request + generation", "evidence"),
    ]
    xs = [0.2, 1.85, 3.5, 5.15, 6.8, 8.45]
    for x, (label, color) in zip(xs, labels):
        box(ax, (x, 2.35), 1.35, 1.15, label, color, fontsize=8.5)
    for x in xs[:-1]:
        arrow(ax, (x + 1.35, 2.92), (x + 1.62, 2.92), "ink")
    ax.text(0.2, 1.55, "一致性检查：model revision · dtype · RoPE · KV layout · block/chunk · layer group · prefix hash",
            fontsize=9, color=COLORS["muted"])
    ax.text(7.0, 1.55, "generation递增用于拒绝过期ticket、command和receipt",
            fontsize=9, color=COLORS["evidence"])
    box(ax, (3.5, 0.45), 2.5, 0.62, "任一身份不一致 → fail-closed / 不归因", "error", fontsize=9, dashed=True)
    arrow(ax, (4.15, 2.35), (4.15, 1.08), "error", dashed=True)
    arrow(ax, (7.45, 2.35), (5.95, 0.78), "error", dashed=True)
    save(fig, out_dir, "fig01_identity_binding")


def prefetch_window(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    finish(ax, "图2  请求提交前的确定性SSD→CPU预取窗口", xlim=(0, 12), ylim=(0, 7))
    stages = [(0.7, "上下文发布"), (3.2, "prefetch_lead_s\n提交前窗口"), (6.0, "HTTP提交"), (8.6, "scheduler lookup"), (10.2, "native retrieve")]
    for x, label in stages:
        ax.plot([x, x], [0.8, 5.8], color=COLORS["grid"], linewidth=1)
        ax.text(x, 0.48, label, ha="center", va="top", fontsize=9, color=COLORS["ink"])
    ax.plot([0.7, 10.8], [4.9, 4.9], color=COLORS["identity"], linewidth=4, solid_capstyle="round")
    ax.text(0.7, 5.15, "Benchmark / RuntimeRequestContext", color=COLORS["identity"], fontsize=10)
    ax.plot([0.7, 6.0], [3.3, 3.3], color=COLORS["prefetch"], linewidth=4, solid_capstyle="round")
    ax.text(0.7, 3.55, "exact token context → TokenDatabase → SSD keys → CPU hot cache", color=COLORS["prefetch"], fontsize=10)
    arrow(ax, (1.0, 2.8), (5.55, 2.8), "prefetch", label="只选择连续、SSD存在且CPU缺失的前缀")
    ax.text(6.0, 2.55, "不写GPU；由native connector在请求上下文中消费", ha="left", fontsize=9, color=COLORS["native"])
    ax.text(0.7, 1.3, "ticket: submitted → completed → consumed", fontsize=9, color=COLORS["evidence"])
    ax.text(6.75, 1.3, "未消费：wasted / expired；部分读取：failed", fontsize=9, color=COLORS["muted"])
    save(fig, out_dir, "fig02_prefetch_window")


def accounting_flow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    finish(ax, "图3  Lookup—Admission—Load—Accounting四阶段证据链")
    nodes = [
        (0.4, "Lookup\nN_lookup", "identity"),
        (2.75, "Admission\nN_alloc", "action"),
        (5.1, "Native retrieve\nN_load", "native"),
        (7.45, "Prefill gap\nN_recompute", "muted"),
        (9.8, "Request finished\nAccounting", "evidence"),
    ]
    for x, label, color in nodes:
        box(ax, (x, 2.45), 1.65, 1.1, label, color, fontsize=9)
    for (x, _, _), (nx, _, _) in zip(nodes, nodes[1:]):
        arrow(ax, (x + 1.65, 3.0), (nx, 3.0), "ink")
    ax.text(0.55, 1.55, "N_lookup ≥ N_alloc ≥ N_load", fontsize=12, color=COLORS["ink"], fontweight="bold")
    ax.text(0.55, 0.95, "N_missing = N_req − N_local − N_load = N_recompute(unalloc) + N_shortfall",
            fontsize=10, color=COLORS["evidence"])
    box(ax, (7.15, 0.65), 2.35, 0.85, "load_shortfall\n显式记录，不静默补齐", "error", fontsize=9, dashed=True)
    arrow(ax, (5.95, 2.45), (8.15, 1.5), "error", dashed=True)
    save(fig, out_dir, "fig03_accounting_chain")


def ticket_state_machine(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    finish(ax, "图4  PrefetchTicket因果状态机", xlim=(0, 12), ylim=(0, 8))
    box(ax, (0.8, 3.25), 1.6, 0.9, "submitted", "prefetch")
    box(ax, (3.2, 3.25), 1.6, 0.9, "completed\n完整promotion", "prefetch")
    box(ax, (5.7, 3.25), 1.6, 0.9, "consumed\n目标请求消费", "evidence")
    box(ax, (3.1, 5.7), 1.8, 0.9, "failed\n部分读取", "error")
    box(ax, (5.9, 5.7), 1.8, 0.9, "expired\nTTL到期", "muted")
    box(ax, (8.6, 5.7), 1.8, 0.9, "wasted\n请求结束未消费", "action")
    box(ax, (8.6, 2.0), 1.8, 0.9, "cancelled\n主动取消", "muted")
    arrow(ax, (2.4, 3.7), (3.2, 3.7), "prefetch")
    arrow(ax, (4.8, 3.7), (5.7, 3.7), "evidence")
    arrow(ax, (1.9, 4.15), (3.35, 5.7), "error", dashed=True, label="字节不完整")
    arrow(ax, (4.7, 4.15), (6.25, 5.7), "muted", dashed=True, label="TTL")
    arrow(ax, (4.8, 3.25), (8.9, 5.7), "action", dashed=True, label="请求完成")
    arrow(ax, (2.4, 3.25), (8.6, 2.45), "muted", dashed=True, label="取消")
    ax.text(0.8, 1.0, "说明：late不是代码状态；是否在需求前完成由时间戳与prefetch window派生。",
            fontsize=9, color=COLORS["muted"])
    save(fig, out_dir, "fig04_prefetch_ticket_state")


def planes_and_actions(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    finish(ax, "图5  三平面架构与受保护动作路径", xlim=(0, 14), ylim=(0, 8))
    box(ax, (0.4, 5.7), 3.0, 1.0, "Benchmark / Request Context\nexact tokens + HMAC", "identity")
    box(ax, (4.2, 5.7), 3.0, 1.0, "RuntimeControlHost\nVendorCallbackBridge", "evidence")
    box(ax, (8.0, 5.7), 2.6, 1.0, "vLLM Scheduler\nexternal admission", "native")
    box(ax, (11.2, 5.7), 2.3, 1.0, "LMCache Connector\nCPU / SSD", "native")
    arrow(ax, (3.4, 6.2), (4.2, 6.2), "identity")
    arrow(ax, (7.2, 6.2), (8.0, 6.2), "native")
    arrow(ax, (10.6, 6.2), (11.2, 6.2), "native")
    box(ax, (2.0, 3.4), 3.0, 1.05, "OnlinePolicyController\nProfileDB / hints", "action", dashed=True)
    box(ax, (5.8, 3.4), 3.0, 1.05, "RuntimeExecutionGate\nreservation + generation", "action", dashed=True)
    box(ax, (9.6, 3.4), 2.7, 1.05, "Protected Action Service\ncommand + HMAC", "action", dashed=True)
    arrow(ax, (5.0, 5.7), (3.5, 4.45), "action", dashed=True)
    arrow(ax, (5.0, 3.93), (5.8, 3.93), "action", dashed=True)
    arrow(ax, (8.8, 3.93), (9.6, 3.93), "action", dashed=True)
    arrow(ax, (10.95, 4.45), (11.95, 5.7), "action", dashed=True, label="owner action")
    box(ax, (3.8, 1.0), 6.6, 0.8, "Evidence Plane: callbacks · tickets · commands · receipts · accounting · manifest", "evidence")
    arrow(ax, (5.0, 5.7), (6.8, 1.8), "evidence", dashed=True)
    arrow(ax, (10.8, 3.4), (8.4, 1.8), "evidence", dashed=True)
    save(fig, out_dir, "fig05_three_planes_actions")


def fail_closed(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.8))
    finish(ax, "图6  按故障时机划分的fail-closed路径", xlim=(0, 14), ylim=(0, 7))
    box(ax, (0.5, 3.2), 2.8, 1.1, "准入前检查\n身份 / 能力 / 容量 / deadline", "identity")
    box(ax, (4.0, 3.2), 2.8, 1.1, "准入前失败\ncap=0 / skip prefetch", "error", dashed=True)
    box(ax, (7.5, 3.2), 2.8, 1.1, "准入后load shortfall\nterminal failure receipt", "error", dashed=True)
    box(ax, (11.0, 3.2), 2.5, 1.1, "原生Prefill\n保持数据面正确性", "native")
    arrow(ax, (3.3, 3.75), (4.0, 3.75), "ink")
    arrow(ax, (2.9, 3.2), (7.5, 3.2), "error", dashed=True)
    ax.text(5.2, 4.5, "另一条时序：scheduler已分配外部Token", ha="center", fontsize=8.5, color=COLORS["muted"])
    arrow(ax, (6.8, 3.35), (11.0, 3.35), "native")
    ax.text(8.9, 2.85, "未准入外部KV → 原生计算", ha="center", fontsize=8.5, color=COLORS["muted"])
    box(ax, (4.0, 1.0), 2.8, 1.0, "通用action失败\nrejected / failed receipt", "error", dashed=True)
    box(ax, (7.5, 1.0), 2.8, 1.0, "数据面保持原状\n不越权删除或写GPU", "native")
    arrow(ax, (9.0, 3.2), (5.5, 2.0), "error", dashed=True)
    ax.text(7.6, 2.35, "reservation / generation / breaker", ha="center", fontsize=8.5, color=COLORS["muted"])
    arrow(ax, (6.8, 1.5), (7.5, 1.5), "native")
    save(fig, out_dir, "fig06_fail_closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figures/core_mechanisms"))
    args = parser.parse_args()
    configure()
    for fn in (identity_chain, prefetch_window, accounting_flow, ticket_state_machine, planes_and_actions, fail_closed):
        fn(args.out_dir)
    print(f"wrote 6 figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
