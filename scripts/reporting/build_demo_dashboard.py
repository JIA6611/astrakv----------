"""Build a static HTML dashboard from AstraKV-W archived artifacts.

The dashboard is intentionally static: it has no server process, no frontend
build chain, and no dependency on real runtime internals.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reporting.build_competition_report import (  # noqa: E402
    ArtifactInput,
    collect_environment,
    collect_git_info,
    parse_artifact_arg,
    parse_label_path,
    resolve_path,
    summarize_artifacts,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_dashboard_artifacts(args)
    summaries = summarize_artifacts(artifacts)
    dashboard_data = build_dashboard_data(args, artifacts, summaries)

    html_path = output_dir / args.html_name
    data_path = output_dir / args.data_name
    manifest_path = output_dir / args.manifest_name

    write_dashboard_data(data_path, dashboard_data)
    write_dashboard_html(html_path, dashboard_data)
    write_manifest(manifest_path, args, dashboard_data, html_path, data_path)

    print(f"Demo dashboard written to {html_path}")
    print(f"Dashboard data written to {data_path}")
    print(f"Dashboard manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/demo_dashboard")
    parser.add_argument("--html-name", default="dashboard.html")
    parser.add_argument("--data-name", default="dashboard_data.json")
    parser.add_argument("--manifest-name", default="dashboard_manifest.json")
    parser.add_argument("--title", default="AstraKV-W Competition Dashboard")
    parser.add_argument("--command", action="append", default=[], help="Command line to record.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact as label:kind=path, for example moe:moe_trace=results/moe/report.md.",
    )
    parser.add_argument("--benchmark", action="append", default=[], help="Shortcut for label=path benchmark artifact.")
    parser.add_argument("--quality", default="", help="Quality CSV/report artifact.")
    parser.add_argument("--hidden-state", default="", help="Hidden-state drift CSV/report artifact.")
    parser.add_argument("--vm-demo", default="", help="VM demo summary/report artifact.")
    parser.add_argument("--moe-trace", default="", help="MoE expert trace CSV/report artifact.")
    parser.add_argument("--moe-loading", default="", help="MoE expert load plan/report artifact.")
    parser.add_argument("--moe-prediction", default="", help="MoE expert prediction CSV/report artifact.")
    parser.add_argument("--competition-report", default="", help="Optional aggregate competition report path.")
    return parser.parse_args()


def load_dashboard_artifacts(args: argparse.Namespace) -> list[ArtifactInput]:
    artifacts: list[ArtifactInput] = []
    for item in args.artifact:
        artifacts.append(parse_artifact_arg(item))
    for item in args.benchmark:
        label, path = parse_label_path(item)
        artifacts.append(ArtifactInput(label=label, kind="benchmark", path=resolve_path(path, "benchmark_results.csv")))
    for label, kind, raw_path in (
        ("quality", "quality", args.quality),
        ("hidden_state", "hidden_state", args.hidden_state),
        ("vm_demo", "vm_demo", args.vm_demo),
        ("moe_trace", "moe_trace", args.moe_trace),
        ("moe_loading", "moe_loading", args.moe_loading),
        ("moe_prediction", "moe_prediction", args.moe_prediction),
        ("competition_report", "report", args.competition_report),
    ):
        if raw_path:
            artifacts.append(ArtifactInput(label=label, kind=kind, path=Path(raw_path)))
    return artifacts


def build_dashboard_data(
    args: argparse.Namespace,
    artifacts: list[ArtifactInput],
    summaries: dict[str, Any],
) -> dict[str, Any]:
    artifact_rows = []
    for artifact in artifacts:
        summary = summaries.get(artifact_key(artifact), {})
        artifact_rows.append(
            {
                "label": artifact.label,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "exists": artifact.exists,
                "status": summary.get("status", "missing"),
                "rows": summary.get("rows", ""),
                "size_bytes": summary.get("size_bytes", artifact.size_bytes),
                "summary": summary,
            }
        )

    benchmark_cards = []
    for artifact in artifacts:
        if artifact.kind != "benchmark":
            continue
        summary = summaries.get(artifact_key(artifact), {})
        benchmark_cards.append(
            {
                "label": artifact.label,
                "success_rate": summary.get("success_rate", ""),
                "ttft_ms": summary.get("mean_ttft_ms", ""),
                "tpot_ms": summary.get("mean_tpot_ms", ""),
                "latency_p95_ms": summary.get("mean_latency_p95_ms", ""),
                "gpu_memory_peak_mb": summary.get("max_gpu_memory_peak_mb", ""),
                "cpu_memory_peak_mb": summary.get("max_cpu_memory_peak_mb", ""),
                "disk_read_delta_mb": summary.get("disk_read_delta_mb", ""),
                "disk_write_delta_mb": summary.get("disk_write_delta_mb", ""),
            }
        )

    evidence = group_evidence(artifact_rows)
    return {
        "schema": "astra-demo-dashboard-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "title": args.title,
        "environment": collect_environment(),
        "git": collect_git_info(),
        "commands": list(args.command),
        "benchmark_cards": benchmark_cards,
        "artifacts": artifact_rows,
        "evidence": evidence,
        "missing": [row for row in artifact_rows if row["status"] == "missing"],
    }


def group_evidence(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "quality": [],
        "hidden_state": [],
        "moe": [],
        "vm": [],
        "scheduler": [],
        "reports": [],
        "other": [],
    }
    for row in rows:
        kind = str(row["kind"])
        if kind == "quality":
            groups["quality"].append(row)
        elif kind == "hidden_state":
            groups["hidden_state"].append(row)
        elif kind.startswith("moe"):
            groups["moe"].append(row)
        elif kind == "vm_demo":
            groups["vm"].append(row)
        elif kind in {"object_schedule", "chunk_scores", "load_recompute", "partial_load"}:
            groups["scheduler"].append(row)
        elif kind in {"report", "comparison", "policy_ablation", "stress"}:
            groups["reports"].append(row)
        elif kind != "benchmark":
            groups["other"].append(row)
    return groups


def write_dashboard_data(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_dashboard_html(path: Path, data: dict[str, Any]) -> None:
    title = escape(data["title"])
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{title}</h1>",
        f"<p class=\"muted\">Generated {escape(data['generated_at'])}. Static artifact dashboard for competition review.</p>",
        render_environment(data),
        render_commands(data),
        render_benchmark_cards(data),
        render_artifact_table(data),
        render_evidence_sections(data),
        render_missing(data),
        "</main>",
        "</body>",
        "</html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_environment(data: dict[str, Any]) -> str:
    env = data.get("environment", {})
    git = data.get("git", {})
    rows = [
        ("Python", env.get("python", "")),
        ("Platform", env.get("platform", "")),
        ("Machine", env.get("machine", "")),
        ("Git commit", git.get("commit", "unknown")),
        ("Git dirty", git.get("dirty", "unknown")),
    ]
    return section("Environment", definition_list(rows))


def render_commands(data: dict[str, Any]) -> str:
    commands = data.get("commands", [])
    if not commands:
        return section("Commands", '<p class="muted">No commands were recorded for this dashboard.</p>')
    return section("Commands", "<ul>" + "".join(f"<li><code>{escape(command)}</code></li>" for command in commands) + "</ul>")


def render_benchmark_cards(data: dict[str, Any]) -> str:
    cards = data.get("benchmark_cards", [])
    if not cards:
        return section("Benchmark Metrics", '<p class="muted">No benchmark artifacts were provided.</p>')
    body = ['<div class="card-grid">']
    for card in cards:
        body.append(
            '<article class="metric-card">'
            f"<h3>{escape(card['label'])}</h3>"
            f"<p><span>Success</span><strong>{fmt(card.get('success_rate'))}</strong></p>"
            f"<p><span>TTFT ms</span><strong>{fmt(card.get('ttft_ms'))}</strong></p>"
            f"<p><span>TPOT ms</span><strong>{fmt(card.get('tpot_ms'))}</strong></p>"
            f"<p><span>Latency p95 ms</span><strong>{fmt(card.get('latency_p95_ms'))}</strong></p>"
            f"<p><span>GPU MB</span><strong>{fmt(card.get('gpu_memory_peak_mb'))}</strong></p>"
            f"<p><span>CPU MB</span><strong>{fmt(card.get('cpu_memory_peak_mb'))}</strong></p>"
            "</article>"
        )
    body.append("</div>")
    return section("Benchmark Metrics", "\n".join(body))


def render_artifact_table(data: dict[str, Any]) -> str:
    rows = data.get("artifacts", [])
    if not rows:
        return section("Artifacts", '<p class="muted">No artifacts were provided.</p>')
    header = "<tr><th>Label</th><th>Kind</th><th>Status</th><th>Rows</th><th>Size</th><th>Path</th></tr>"
    body = []
    for row in rows:
        status_class = "ok" if row["status"] == "ok" else "warn"
        body.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{escape(row['kind'])}</td>"
            f"<td><span class=\"pill {status_class}\">{escape(row['status'])}</span></td>"
            f"<td>{escape(row['rows'])}</td>"
            f"<td>{escape(row['size_bytes'])}</td>"
            f"<td><code>{escape(row['path'])}</code></td>"
            "</tr>"
        )
    return section("Artifacts", f"<table>{header}{''.join(body)}</table>")


def render_evidence_sections(data: dict[str, Any]) -> str:
    evidence = data.get("evidence", {})
    parts = []
    for title, key in (
        ("Quality Evidence", "quality"),
        ("Hidden-State Evidence", "hidden_state"),
        ("MoE Evidence", "moe"),
        ("Virtual-Memory Evidence", "vm"),
        ("Scheduler Evidence", "scheduler"),
        ("Reports And Ablations", "reports"),
    ):
        rows = evidence.get(key, [])
        if not rows:
            parts.append(section(title, '<p class="muted">No artifact provided.</p>'))
            continue
        parts.append(section(title, render_summary_list(rows)))
    return "\n".join(parts)


def render_missing(data: dict[str, Any]) -> str:
    missing = data.get("missing", [])
    if not missing:
        return section("Missing Evidence", '<p class="muted">No provided artifact paths were missing.</p>')
    return section("Missing Evidence", render_summary_list(missing))


def render_summary_list(rows: list[dict[str, Any]]) -> str:
    items = []
    for row in rows:
        summary = row.get("summary", {})
        compact = {key: value for key, value in summary.items() if key not in {"path", "size_bytes"}}
        items.append(
            "<li>"
            f"<strong>{escape(row['label'])}</strong> "
            f"<span class=\"muted\">{escape(row['kind'])}</span>"
            f"<pre>{escape(json.dumps(compact, ensure_ascii=False, indent=2))}</pre>"
            "</li>"
        )
    return "<ul class=\"summary-list\">" + "".join(items) + "</ul>"


def definition_list(rows: list[tuple[str, Any]]) -> str:
    body = []
    for key, value in rows:
        body.append(f"<dt>{escape(key)}</dt><dd>{escape(value)}</dd>")
    return "<dl>" + "".join(body) + "</dl>"


def section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def write_manifest(path: Path, args: argparse.Namespace, data: dict[str, Any], html_path: Path, data_path: Path) -> None:
    payload = {
        "schema": "astra-demo-dashboard-manifest-v1",
        "generated_at": data["generated_at"],
        "title": args.title,
        "outputs": {
            "html": str(html_path),
            "data_json": str(data_path),
        },
        "artifact_count": len(data.get("artifacts", [])),
        "missing_count": len(data.get("missing", [])),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def css() -> str:
    return """
:root { color-scheme: light; font-family: Segoe UI, Arial, sans-serif; }
body { margin: 0; background: #f6f7f9; color: #17202a; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
h1 { font-size: 32px; margin: 0 0 8px; letter-spacing: 0; }
h2 { font-size: 20px; margin: 0 0 14px; letter-spacing: 0; }
h3 { font-size: 16px; margin: 0 0 10px; letter-spacing: 0; }
section { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin-top: 16px; }
.muted { color: #607080; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.metric-card { border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; background: #fbfcfd; }
.metric-card p { display: flex; justify-content: space-between; gap: 10px; margin: 8px 0; }
.metric-card span { color: #5f6f7d; }
.metric-card strong { font-weight: 650; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid #e5e9ef; padding: 8px; text-align: left; vertical-align: top; }
th { background: #f1f4f7; }
code { font-family: Consolas, monospace; font-size: 13px; }
pre { overflow: auto; background: #f4f6f8; border-radius: 6px; padding: 10px; font-size: 12px; }
dl { display: grid; grid-template-columns: 160px 1fr; gap: 8px 14px; margin: 0; }
dt { color: #5f6f7d; }
dd { margin: 0; }
.pill { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; }
.pill.ok { background: #e5f3ea; color: #17663a; }
.pill.warn { background: #fff2d7; color: #75520b; }
.summary-list { padding-left: 18px; }
.summary-list li { margin: 10px 0; }
"""


def artifact_key(artifact: ArtifactInput) -> str:
    return f"{artifact.label}:{artifact.kind}:{artifact.path}"


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
