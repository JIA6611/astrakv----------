"""Run prompt workload JSONL against an OpenAI-compatible chat endpoint.

This is a standalone server-side helper for the Qasper random/grouped workload
package. It reads prompt records containing `messages` or `prompt`, sends them
to `/v1/chat/completions`, and writes request-level metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_jsonl(args.prompts)
    if args.limit > 0:
        prompts = prompts[: args.limit]
    write_json(output_dir / "run_config.json", vars(args) | {"prompt_count": len(prompts)})

    results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt.get('request_id', prompt.get('sample_id', index))}")
        if args.dry_run:
            result = dry_run_result(prompt, index=index, model=args.model)
        else:
            result = run_one_request(prompt, index=index, args=args)
        results.append(result)
        append_jsonl(output_dir / "request_results.jsonl", result)

    rows = summarize_results(results, backend=args.backend)
    write_csv(output_dir / "benchmark_results.csv", rows)
    write_report(output_dir / "benchmark_report.md", args, results, rows)
    failed = sum(1 for item in results if item["status"] == "error")
    print(f"Finished {len(results)} request(s), failed={failed}. Outputs: {output_dir}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="Prompt JSONL with `messages` or `prompt` fields.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Endpoint base URL, without /v1 suffix.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=0, help="Override prompt max_tokens when > 0.")
    parser.add_argument("--temperature", type=float, default=None, help="Override prompt temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override prompt top_p.")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming chat completions.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write skipped results without HTTP.")
    return parser.parse_args()


def run_one_request(prompt: dict[str, Any], *, index: int, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    ttft_ms: float | None = None
    output_text = ""
    output_tokens = 0
    status = "ok"
    error = ""
    try:
        payload = build_payload(prompt, args)
        if args.no_stream:
            output_text, output_tokens = call_non_streaming(args.base_url, payload, args.api_key, args.timeout)
            ttft_ms = None
        else:
            parts: list[str] = []
            for event in stream_chat_completion(args.base_url, payload, args.api_key, args.timeout):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                content = extract_delta_content(event)
                if content:
                    parts.append(content)
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    output_tokens = int(usage["completion_tokens"])
            output_text = "".join(parts)
            if output_tokens == 0:
                output_tokens = max(1, len(output_text.split())) if output_text else 0
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostics should not hide errors.
        status = "error"
        error = classify_error(exc)

    ended = time.perf_counter()
    latency_ms = (ended - started) * 1000.0
    generated_after_first = max(0, output_tokens - 1)
    tpot_ms = None
    if status == "ok" and ttft_ms is not None and generated_after_first > 0:
        tpot_ms = max(0.0, latency_ms - ttft_ms) / generated_after_first
    return base_result(prompt, index=index, model=args.model) | {
        "backend": args.backend,
        "status": status,
        "error": error,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "output_text": output_text,
    }


def dry_run_result(prompt: dict[str, Any], *, index: int, model: str) -> dict[str, Any]:
    return base_result(prompt, index=index, model=model) | {
        "backend": "dry_run",
        "status": "skipped",
        "error": "",
        "ttft_ms": None,
        "tpot_ms": None,
        "latency_ms": 0.0,
        "output_tokens": 0,
        "output_text": "",
    }


def base_result(prompt: dict[str, Any], *, index: int, model: str) -> dict[str, Any]:
    return {
        "request_index": index,
        "request_id": prompt.get("request_id", f"request-{index}"),
        "sample_id": prompt.get("sample_id", f"sample-{index}"),
        "dataset": prompt.get("dataset", ""),
        "task": prompt.get("task", ""),
        "workload_type": prompt.get("workload_type", ""),
        "reuse_group": prompt.get("reuse_group", ""),
        "shared_context": bool(prompt.get("shared_context", False)),
        "model": model,
        "answer": prompt.get("answer", ""),
        "ground_truth": prompt.get("ground_truth", ""),
        "prompt_hash": prompt.get("prompt_hash", ""),
        "metadata_ref": prompt.get("metadata_ref", ""),
    }


def build_payload(prompt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = prompt.get("messages")
    if not isinstance(messages, list):
        messages = [
            {"role": "system", "content": "You are a deterministic benchmark assistant."},
            {"role": "user", "content": str(prompt.get("prompt", ""))},
        ]
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens if args.max_tokens > 0 else int(prompt.get("max_tokens", 128) or 128),
        "temperature": args.temperature if args.temperature is not None else float(prompt.get("temperature", 0.0) or 0.0),
        "top_p": args.top_p if args.top_p is not None else float(prompt.get("top_p", 1.0) or 1.0),
        "stream": not args.no_stream,
        "stream_options": {"include_usage": True} if not args.no_stream else None,
    }


def stream_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> Iterable[dict[str, Any]]:
    url = normalize_base_url(base_url) + "/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
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
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def call_non_streaming(
    base_url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> tuple[str, int]:
    url = normalize_base_url(base_url) + "/v1/chat/completions"
    payload = dict(payload)
    payload["stream"] = False
    payload.pop("stream_options", None)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    choices = data.get("choices", [])
    text = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        text = str(message.get("content", "")) if isinstance(message, dict) else ""
    usage = data.get("usage", {})
    tokens = int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
    return text, tokens


def extract_delta_content(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            parts.append(str(delta["content"]))
    return "".join(parts)


def summarize_results(results: list[dict[str, Any]], *, backend: str) -> list[dict[str, Any]]:
    ok = [item for item in results if item["status"] == "ok"]
    errors = [item for item in results if item["status"] == "error"]
    latencies = [float(item["latency_ms"]) for item in ok]
    ttfts = [float(item["ttft_ms"]) for item in ok if item["ttft_ms"] is not None]
    tpots = [float(item["tpot_ms"]) for item in ok if item["tpot_ms"] is not None]
    return [
        {
            "backend": backend,
            "request_count": len(results),
            "ok_count": len(ok),
            "error_count": len(errors),
            "avg_latency_ms": avg(latencies),
            "avg_ttft_ms": avg(ttfts),
            "avg_tpot_ms": avg(tpots),
            "p95_latency_ms": percentile(latencies, 0.95),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "shared_context_count": sum(1 for item in results if item.get("shared_context")),
        }
    ]


def write_report(path: Path, args: argparse.Namespace, results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    row = rows[0] if rows else {}
    lines = [
        "# Prompt Workload Endpoint Report",
        "",
        f"- Prompts: `{args.prompts}`",
        f"- Base URL: `{args.base_url}`",
        f"- Model: `{args.model}`",
        f"- Backend: `{args.backend}`",
        f"- Request count: `{row.get('request_count', 0)}`",
        f"- OK count: `{row.get('ok_count', 0)}`",
        f"- Error count: `{row.get('error_count', 0)}`",
        f"- Avg TTFT ms: `{row.get('avg_ttft_ms', 0.0):.3f}`",
        f"- Avg TPOT ms: `{row.get('avg_tpot_ms', 0.0):.3f}`",
        f"- Avg latency ms: `{row.get('avg_latency_ms', 0.0):.3f}`",
        f"- P95 latency ms: `{row.get('p95_latency_ms', 0.0):.3f}`",
        "",
        "## Failed Requests",
        "",
    ]
    failed = [item for item in results if item["status"] == "error"]
    if failed:
        for item in failed[:20]:
            lines.append(f"- `{item['request_id']}`: {item['error']}")
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def classify_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        return f"HTTPError {exc.code}: {body[:500]}"
    if isinstance(exc, urllib.error.URLError):
        return f"URLError: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def normalize_base_url(base_url: str) -> str:
    return str(base_url).rstrip("/")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
