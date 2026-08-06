"""Plan partial KV loads from runtime-agnostic chunk metadata.

The planner emits adapter-consumable intent records. It does not load tensors
or call vLLM/LMCache internals.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier  # noqa: E402
from astrakv.kv_cache.partial_load import (  # noqa: E402
    PARTIAL_LOAD_SCHEMA_VERSION,
    PartialKVLoadDecision,
    PartialKVLoadPlanner,
    PartialKVLoadRequest,
    TokenSpan,
    chunk_meta_from_record,
    memory_tier_from,
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks)
    if args.default_bytes_per_token > 0:
        chunks = [with_default_size(chunk, args.default_bytes_per_token) for chunk in chunks]
    request_id = args.request_id or infer_request_id(chunks)
    if not request_id:
        raise SystemExit("No request id found. Provide --request-id or chunks with request_id.")

    request = PartialKVLoadRequest(
        request_id=request_id,
        target_layers=tuple(args.layers or ()),
        token_spans=tuple(parse_token_span(item) for item in args.token_span),
        target_tier=memory_tier_from(args.target_tier),
        metadata={"source_chunks": args.chunks},
    )
    planner = PartialKVLoadPlanner()
    decisions = planner.plan(chunks, request)
    summary = planner.summarize(decisions, plan_id=request.plan_id, request_id=request.request_id)

    plan_path = output_dir / args.plan_name
    decision_csv_path = output_dir / args.decisions_csv_name
    summary_path = output_dir / args.summary_name
    report_path = output_dir / args.report_name
    write_jsonl(plan_path, [decision.to_record() for decision in decisions])
    write_decisions_csv(decision_csv_path, decisions)
    write_summary_csv(summary_path, summary.to_record())
    write_report(report_path, args, chunks, request, decisions, summary.to_record())

    print(f"Partial KV plan written to {plan_path}")
    print(f"Partial KV summary written to {summary_path}")
    print(f"Partial KV report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Chunk metadata JSON, JSONL, CSV, or runtime snapshot JSON.")
    parser.add_argument("--request-id", default="", help="Request id to plan. Defaults to first request id in chunks.")
    parser.add_argument("--layers", nargs="*", type=int, default=[], help="Selected layer ids. Empty means all layers.")
    parser.add_argument(
        "--token-span",
        action="append",
        default=[],
        help="Selected token span as start:end. Can be repeated. Empty means full chunk span.",
    )
    parser.add_argument("--target-tier", default="gpu", help="Target tier for selected ranges.")
    parser.add_argument(
        "--default-bytes-per-token",
        type=int,
        default=0,
        help="Fill missing chunk size_bytes as token_count * this value.",
    )
    parser.add_argument("--output-dir", default="results/partial_kv_load")
    parser.add_argument("--plan-name", default="partial_kv_plan.jsonl")
    parser.add_argument("--decisions-csv-name", default="partial_kv_decisions.csv")
    parser.add_argument("--summary-name", default="partial_kv_summary.csv")
    parser.add_argument("--report-name", default="partial_kv_report.md")
    return parser.parse_args()


def load_chunks(path: str | Path) -> list[KVChunkMeta]:
    chunk_path = Path(path)
    if not chunk_path.exists():
        raise SystemExit(f"Chunk metadata file not found: {chunk_path}")
    if chunk_path.suffix.lower() == ".csv":
        with chunk_path.open("r", encoding="utf-8", newline="") as handle:
            return [chunk_meta_from_record(row) for row in csv.DictReader(handle)]
    text = chunk_path.read_text(encoding="utf-8")
    if chunk_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            records = list(parsed.get("chunks") or parsed.get("block_table") or [parsed])
        elif isinstance(parsed, list):
            records = parsed
        else:
            records = []
    return [chunk_meta_from_record(record) for record in records if isinstance(record, dict)]


def parse_token_span(value: str) -> TokenSpan:
    raw = value.strip().replace(",", ":")
    if ":" not in raw:
        raise SystemExit(f"--token-span must use start:end format: {value}")
    start, end = raw.split(":", 1)
    return TokenSpan(int(start), int(end))


def with_default_size(chunk: KVChunkMeta, bytes_per_token: int) -> KVChunkMeta:
    if chunk.size_bytes is not None:
        return chunk
    return KVChunkMeta(
        request_id=chunk.request_id,
        layer_id=chunk.layer_id,
        start_token=chunk.start_token,
        end_token=chunk.end_token,
        block_ids=chunk.block_ids,
        chunk_id=chunk.chunk_id,
        tier=chunk.tier,
        dtype=chunk.dtype,
        device=chunk.device,
        size_bytes=max(0, chunk.token_count * bytes_per_token),
        cache_key=chunk.cache_key,
        adapter_name=chunk.adapter_name,
        metadata=dict(chunk.metadata, size_bytes_estimated=True, bytes_per_token=bytes_per_token),
    )


def infer_request_id(chunks: list[KVChunkMeta]) -> str:
    for chunk in chunks:
        if chunk.request_id:
            return chunk.request_id
    return ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_decisions_csv(path: Path, decisions: list[PartialKVLoadDecision]) -> None:
    rows = [decision.to_record() for decision in decisions]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        key for key in rows[0].keys() if key != "metadata"
    ] + ["metadata"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["selected_spans"] = json.dumps(output.get("selected_spans", []), ensure_ascii=False)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_report(
    path: Path,
    args: argparse.Namespace,
    chunks: list[KVChunkMeta],
    request: PartialKVLoadRequest,
    decisions: list[PartialKVLoadDecision],
    summary: dict[str, Any],
) -> None:
    action_counts: dict[str, int] = {}
    for decision in decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1

    lines = [
        "# Partial KV Load Plan Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Chunk metadata: `{args.chunks}`",
        f"- Request id: `{request.request_id}`",
        f"- Target layers: `{', '.join(str(item) for item in request.target_layers) if request.target_layers else 'all'}`",
        f"- Token spans: `{', '.join(f'{span.start_token}:{span.end_token}' for span in request.token_spans) if request.token_spans else 'full'}`",
        f"- Target tier: `{request.target_tier.value}`",
        f"- Default bytes per token: `{args.default_bytes_per_token}`",
        "",
        "## Summary",
        "",
        f"- Schema: `{PARTIAL_LOAD_SCHEMA_VERSION}`",
        f"- Total input chunks: `{len(chunks)}`",
        f"- Planned chunks for request: `{summary['total_chunks']}`",
        f"- Full loads: `{summary['load_full']}`",
        f"- Partial loads: `{summary['load_partial']}`",
        f"- Skipped chunks: `{summary['skip']}`",
        f"- Loaded tokens: `{summary['loaded_tokens']}`",
        f"- Skipped tokens: `{summary['skipped_tokens']}`",
        f"- Loaded bytes: `{summary['loaded_bytes']}`",
        f"- Skipped bytes: `{summary['skipped_bytes']}`",
        f"- Estimated byte saving rate: `{summary['byte_saving_rate']:.4f}`",
        "",
        "## Decisions",
        "",
        "| chunk | layer | chunk span | selected span | action | loaded bytes | skipped bytes | reason |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for decision in decisions[:50]:
        selected = (
            f"{decision.selected_start_token}:{decision.selected_end_token}"
            if decision.selected_start_token is not None and decision.selected_end_token is not None
            else ""
        )
        lines.append(
            f"| {decision.chunk_id} | {decision.layer_id} | "
            f"{decision.chunk_start_token}:{decision.chunk_end_token} | {selected} | "
            f"{decision.action.value} | {decision.loaded_bytes} | {decision.skipped_bytes} | "
            f"{decision.reason} |"
        )
    if len(decisions) > 50:
        lines.append(f"| ... |  |  |  |  |  |  | {len(decisions) - 50} more decision(s) omitted |")

    lines.extend(
        [
            "",
            "## Adapter Boundary",
            "",
            "- This artifact is a partial-load intent plan, not proof that tensors were partially loaded.",
            "- Runtime adapters can consume `partial_kv_plan.jsonl` and map ranges to backend block/page operations.",
            "- Official claims about real partial loading require GPU validation with adapter logs or cache events.",
            "",
            "## Artifacts",
            "",
            "- `partial_kv_plan.jsonl`",
            "- `partial_kv_decisions.csv`",
            "- `partial_kv_summary.csv`",
            "- `partial_kv_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
