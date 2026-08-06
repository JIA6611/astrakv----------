"""Evaluate output quality and consistency for baseline vs variant runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.evaluation.quality import (  # noqa: E402
    QualityComparison,
    compare_output_records,
    infer_sample_id,
    summarize_quality,
)
from astrakv.runtime.endpoint_prefetch import classify_endpoint_error, normalize_base_url  # noqa: E402


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_jsonl and args.variant_jsonl:
        baseline_records = load_jsonl(args.baseline_jsonl)
        variant_records = load_jsonl(args.variant_jsonl)
        comparisons = compare_record_sets(baseline_records, variant_records)
        input_mode = "offline_jsonl"
    elif args.prompts and args.baseline_base_url and args.variant_base_url:
        prompts = load_prompts(args.prompts)
        baseline_records = collect_endpoint_outputs(args, prompts, mode="baseline")
        variant_records = collect_endpoint_outputs(args, prompts, mode="variant")
        comparisons = compare_record_sets(baseline_records, variant_records)
        input_mode = "endpoint"
    else:
        raise SystemExit(
            "Provide either --baseline-jsonl and --variant-jsonl, "
            "or --prompts with --baseline-base-url and --variant-base-url."
        )

    records_path = output_dir / args.records_name
    csv_path = output_dir / args.csv_name
    report_path = output_dir / args.report_name
    write_records_jsonl(records_path, comparisons)
    write_results_csv(csv_path, comparisons)
    write_report(report_path, args, comparisons, input_mode, records_path, csv_path)
    print(f"Quality records written to {records_path}")
    print(f"Quality CSV written to {csv_path}")
    print(f"Quality report written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", default="", help="Offline baseline output JSONL.")
    parser.add_argument("--variant-jsonl", default="", help="Offline variant output JSONL.")
    parser.add_argument("--prompts", default="", help="Prompt JSONL or text file for endpoint evaluation.")
    parser.add_argument("--baseline-base-url", default="", help="Baseline OpenAI-compatible base URL.")
    parser.add_argument("--variant-base-url", default="", help="Variant OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--baseline-api-key", default="")
    parser.add_argument("--variant-api-key", default="")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--variant-model", default="")
    parser.add_argument("--system-prompt", default="You are a deterministic evaluation assistant.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/quality_evaluation")
    parser.add_argument("--records-name", default="quality_records.jsonl")
    parser.add_argument("--csv-name", default="quality_results.csv")
    parser.add_argument("--report-name", default="quality_report.md")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {jsonl_path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def compare_record_sets(
    baseline_records: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> list[QualityComparison]:
    baseline_by_id = index_records(baseline_records)
    variant_by_id = index_records(variant_records)
    ordered_ids = ordered_union([infer_sample_id(record) or f"row{index}" for index, record in enumerate(baseline_records)])
    ordered_ids.extend(item for item in variant_by_id if item not in set(ordered_ids))

    comparisons: list[QualityComparison] = []
    for sample_id in ordered_ids:
        baseline = baseline_by_id.get(sample_id, {"sample_id": sample_id, "status": "missing"})
        variant = variant_by_id.get(sample_id, {"sample_id": sample_id, "status": "missing"})
        comparisons.append(compare_output_records(baseline, variant, sample_id=sample_id))
    return comparisons


def index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        sample_id = infer_sample_id(record) or f"row{index}"
        indexed[sample_id] = record
    return indexed


def ordered_union(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    if prompt_path.suffix.lower() == ".jsonl":
        records = load_jsonl(prompt_path)
        return [normalize_prompt_record(record, index) for index, record in enumerate(records)]
    prompts: list[dict[str, Any]] = []
    for index, line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            prompts.append({"sample_id": f"prompt{index}", "prompt": line.strip()})
    return prompts


def normalize_prompt_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    sample_id = infer_sample_id(record) or f"prompt{index}"
    prompt = record.get("prompt", record.get("text", record.get("input", "")))
    messages = record.get("messages")
    return {
        "sample_id": sample_id,
        "prompt": str(prompt) if prompt is not None else "",
        "messages": messages if isinstance(messages, list) else None,
    }


def collect_endpoint_outputs(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    is_baseline = mode == "baseline"
    base_url = normalize_base_url(args.baseline_base_url if is_baseline else args.variant_base_url)
    api_key = args.baseline_api_key or args.api_key if is_baseline else args.variant_api_key or args.api_key
    model = args.baseline_model or args.model if is_baseline else args.variant_model or args.model
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        try:
            text, latency_ms = chat_completion_text(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    "output": text,
                    "latency_ms": latency_ms,
                    "mode": mode,
                    "model": model,
                    "base_url": base_url,
                }
            )
        except Exception as exc:  # noqa: BLE001 - evaluation diagnostics need concrete endpoint errors.
            records.append(
                {
                    "sample_id": sample_id,
                    "status": "error",
                    "output": "",
                    "error": classify_endpoint_error(exc),
                    "mode": mode,
                    "model": model,
                    "base_url": base_url,
                }
            )
    return records


def chat_completion_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: dict[str, Any],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> tuple[str, float]:
    started = time.perf_counter()
    messages = prompt.get("messages")
    if not isinstance(messages, list):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(prompt.get("prompt", ""))},
        ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    chunks: list[str] = []
    for event in stream_chat_completion(f"{base_url}/v1/chat/completions", payload, api_key, timeout):
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str):
                chunks.append(content)
    return "".join(chunks), (time.perf_counter() - started) * 1000.0


def stream_chat_completion(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> Iterator[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def write_records_jsonl(path: Path, comparisons: list[QualityComparison]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for comparison in comparisons:
            handle.write(json.dumps(comparison.to_record(), ensure_ascii=False) + "\n")


def write_results_csv(path: Path, comparisons: list[QualityComparison]) -> None:
    rows = [comparison.to_record() for comparison in comparisons]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0].keys() if key != "metadata"] + ["metadata"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_report(
    path: Path,
    args: argparse.Namespace,
    comparisons: list[QualityComparison],
    input_mode: str,
    records_path: Path,
    csv_path: Path,
) -> None:
    summary = summarize_quality(comparisons)
    summary_record = summary.to_record()
    ppl_available = summary.mean_baseline_ppl is not None and summary.mean_variant_ppl is not None
    lines = [
        "# Quality Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Input mode: `{input_mode}`",
        f"- Baseline JSONL: `{args.baseline_jsonl or 'none'}`",
        f"- Variant JSONL: `{args.variant_jsonl or 'none'}`",
        f"- Prompts: `{args.prompts or 'none'}`",
        f"- Baseline endpoint: `{args.baseline_base_url or 'none'}`",
        f"- Variant endpoint: `{args.variant_base_url or 'none'}`",
        f"- Model: `{args.model}`",
        f"- Temperature: `{args.temperature}`",
        f"- Top-p: `{args.top_p}`",
        "",
        "## Summary",
        "",
        f"- Samples: `{summary.sample_count}`",
        f"- OK samples: `{summary.ok_count}`",
        f"- Exact match rate: `{summary.exact_match_rate:.4f}`",
        f"- Normalized match rate: `{summary.normalized_match_rate:.4f}`",
        f"- Mean token divergence rate: `{summary.mean_token_divergence_rate:.4f}`",
        f"- Mean char divergence rate: `{summary.mean_char_divergence_rate:.4f}`",
        f"- PPL available: `{ppl_available}`",
        f"- Mean baseline PPL: `{summary_record['mean_baseline_ppl']}`",
        f"- Mean variant PPL: `{summary_record['mean_variant_ppl']}`",
        f"- Mean PPL delta: `{summary_record['mean_ppl_delta']}`",
        "",
        "## Per-Sample Results",
        "",
        "| sample | status | exact | normalized | token divergence | char divergence | PPL delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in comparisons[:50]:
        ppl_delta = "" if comparison.ppl_delta is None else f"{comparison.ppl_delta:.6f}"
        lines.append(
            f"| {comparison.sample_id} | {comparison.status} | {int(comparison.exact_match)} | "
            f"{int(comparison.normalized_match)} | {comparison.token_divergence_rate:.6f} | "
            f"{comparison.char_divergence_rate:.6f} | {ppl_delta} |"
        )
    if len(comparisons) > 50:
        lines.append(f"| ... |  |  |  |  |  | {len(comparisons) - 50} more sample(s) omitted |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact match is strict string equality.",
            "- Normalized match lowercases text and collapses whitespace.",
            "- Token divergence is edit distance over whitespace tokens divided by the longer token sequence.",
            "- PPL is reported only when input records already include `ppl`, `perplexity`, `loss`, `nll`, or `negative_log_likelihood` fields.",
            "- Endpoint mode uses deterministic settings by default, but backend kernels may still have small nondeterminism.",
            "",
            "## Artifacts",
            "",
            f"- `{records_path}`",
            f"- `{csv_path}`",
            "- `quality_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
