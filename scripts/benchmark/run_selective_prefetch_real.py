"""Run endpoint-level selective prefetch against a real vLLM/LMCache backend.

The runner uses only the OpenAI-compatible HTTP endpoint. It does not import or
modify vLLM, LMCache, or connector internals. A prefetch is a real short-output
warmup request for a repeated prefix; the following demand request uses the
same prefix so LMCache/vLLM can reuse stored KV through its normal runtime path.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - config files are optional for CLI smoke runs.
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.kv_cache.metadata import MemoryTier  # noqa: E402
from astrakv.prefetch.async_engine import AsyncPrefetchEngine, PrefetchStatus  # noqa: E402
from astrakv.runtime.endpoint_prefetch import (  # noqa: E402
    EndpointPrefetchAdapter,
    EndpointRequest,
    EndpointResult,
    OpenAIEndpointClient,
    make_prefetch_request,
    normalize_base_url,
)


EVENT_SCHEMA = "astra-prefetch-event-v1"


@dataclass(frozen=True, slots=True)
class PrefetchCase:
    case_id: str
    context_length: int
    repeat_index: int
    control_prompt: str
    prefetch_prompt: str


def main() -> int:
    raw_args = parse_args()
    config = load_config(raw_args.config)
    args = resolve_args(raw_args, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, events = asyncio.run(run_cases(args))
    write_csv(output_dir / "prefetch_results.csv", rows)
    write_csv(output_dir / "prefetch_benchmark_results.csv", build_prefetch_benchmark_rows(rows, args))
    write_jsonl(output_dir / "prefetch_events.jsonl", events)
    write_effective_config(output_dir / "prefetch_config.json", args, config)
    write_report(output_dir / "prefetch_report.md", args, rows, events)

    failures = [row for row in rows if row["no_prefetch_status"] != "ok" or row["prefetch_demand_status"] != "ok"]
    if failures:
        print(f"Selective prefetch run completed with {len(failures)} failed case(s).", file=sys.stderr)
        return 1
    print(f"Selective prefetch outputs written to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config path.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--warmup-output-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompt-token-scale", type=float, default=None)
    parser.add_argument(
        "--cache-events",
        action="append",
        default=[],
        help="Optional P0-5 cache_events.jsonl file. Can be repeated for log evidence in the report.",
    )
    parser.add_argument(
        "--hit-improvement-threshold-pct",
        type=float,
        default=None,
        help="Latency improvement threshold used when cache-event logs cannot be matched to rows.",
    )
    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise SystemExit("PyYAML is required for --config. Install requirements.txt first.")
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def resolve_args(raw: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    backend_cfg = _dict(config.get("backend"))
    matrix_cfg = _dict(config.get("matrix"))
    workload_cfg = _dict(config.get("workload"))
    prefetch_cfg = _dict(config.get("prefetch"))
    run_name = str(config.get("run_name") or "astrakv_real_selective_prefetch")

    if raw.output_dir:
        output_dir = Path(raw.output_dir)
    else:
        output_dir = Path("results") / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return argparse.Namespace(
        config=raw.config,
        run_name=run_name,
        backend=str(first_non_empty(raw.backend, backend_cfg.get("name"), "vllm_lmcache_endpoint")),
        base_url=normalize_base_url(
            str(first_non_empty(raw.base_url, backend_cfg.get("base_url"), "http://127.0.0.1:8000"))
        ),
        api_key=str(first_non_empty(raw.api_key, backend_cfg.get("api_key"), "EMPTY")),
        model=str(first_non_empty(raw.model, backend_cfg.get("model"), "Qwen/Qwen2.5-7B-Instruct")),
        output_dir=str(output_dir),
        context_lengths=list(raw.context_lengths or matrix_cfg.get("context_lengths") or [1024, 2048]),
        repeat=int(raw.repeat or matrix_cfg.get("repeat") or 3),
        output_tokens=int(raw.output_tokens or matrix_cfg.get("output_tokens") or 64),
        warmup_output_tokens=int(raw.warmup_output_tokens or prefetch_cfg.get("warmup_output_tokens") or 1),
        timeout=float(raw.timeout or backend_cfg.get("timeout_seconds") or 600.0),
        temperature=float(
            raw.temperature if raw.temperature is not None else workload_cfg.get("temperature", 0.0)
        ),
        top_p=float(raw.top_p if raw.top_p is not None else workload_cfg.get("top_p", 1.0)),
        system_prompt=str(
            first_non_empty(
                workload_cfg.get("system_prompt"),
                "You are a concise assistant for cache prefetch validation.",
            )
        ),
        prompt_seed=str(
            first_non_empty(
                workload_cfg.get("prompt_seed"),
                "AstraKV-W real selective prefetch repeated-prefix validation.",
            )
        ),
        prompt_token_scale=float(
            raw.prompt_token_scale
            if raw.prompt_token_scale is not None
            else workload_cfg.get("prompt_token_scale", 0.70)
        ),
        hit_improvement_threshold_pct=float(
            raw.hit_improvement_threshold_pct
            if raw.hit_improvement_threshold_pct is not None
            else prefetch_cfg.get("hit_improvement_threshold_pct", 5.0)
        ),
        target_tier=str(prefetch_cfg.get("target_tier", "gpu")),
        cache_events=list(raw.cache_events or prefetch_cfg.get("cache_events", [])),
        raw_config={key: value for key, value in config.items() if key != "_config_path"},
    )


async def run_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    client = OpenAIEndpointClient(args.base_url, api_key=args.api_key, timeout_seconds=args.timeout)
    engine = AsyncPrefetchEngine(adapter=EndpointPrefetchAdapter(client))
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for case in build_cases(args):
        no_prefetch_request = make_endpoint_request(
            args=args,
            request_id=f"{case.case_id}:no_prefetch",
            prompt=case.control_prompt,
            max_tokens=args.output_tokens,
            mode="no_prefetch",
        )
        no_prefetch_result = await asyncio.to_thread(client.chat_completion, no_prefetch_request)
        events.append(make_event("demand_completed", case, mode="no_prefetch", result=no_prefetch_result))

        prefetch_request = make_endpoint_request(
            args=args,
            request_id=f"{case.case_id}:prefetch",
            prompt=case.prefetch_prompt,
            max_tokens=args.warmup_output_tokens,
            mode="astrakv_prefetch",
        )
        target_tier = parse_tier(args.target_tier)
        request = make_prefetch_request(
            chunk_id=case.case_id,
            endpoint_request=prefetch_request,
            target_tier=target_tier,
            priority=max(0, max(args.context_lengths) - case.context_length),
            metadata={
                "case": case.case_id,
                "context_length": case.context_length,
                "repeat_index": case.repeat_index,
                "policy": "repeated_prefix_warmup",
            },
        )
        events.append(
            {
                "schema": EVENT_SCHEMA,
                "event_type": "prefetch_submitted",
                "case": case.case_id,
                "mode": "astrakv_prefetch",
                "request_id": request.request_id,
                "chunk_id": request.chunk_id,
                "target_tier": request.target_tier.value,
                "status": "submitted",
                "metadata": dict(
                    json_safe_metadata(request.metadata),
                    endpoint_request_id=prefetch_request.request_id,
                ),
            }
        )
        prefetch_result = await engine.submit(request)
        endpoint_result = prefetch_result.metadata.get("endpoint_result", {})
        events.append(
            {
                "schema": EVENT_SCHEMA,
                "event_type": "prefetch_completed"
                if prefetch_result.status == PrefetchStatus.COMPLETED
                else "prefetch_failed",
                "case": case.case_id,
                "mode": "astrakv_prefetch",
                "request_id": prefetch_result.request_id,
                "chunk_id": prefetch_result.chunk_id,
                "target_tier": target_tier.value,
                "status": prefetch_result.status.value,
                "message": prefetch_result.message,
                "metadata": dict(endpoint_result) if isinstance(endpoint_result, dict) else {},
            }
        )

        demand_request = make_endpoint_request(
            args=args,
            request_id=f"{case.case_id}:prefetch_demand",
            prompt=case.prefetch_prompt,
            max_tokens=args.output_tokens,
            mode="astrakv_prefetch_demand",
        )
        demand_result = await asyncio.to_thread(client.chat_completion, demand_request)
        events.append(make_event("demand_completed", case, mode="astrakv_prefetch", result=demand_result))

        row = summarize_case(
            args=args,
            case=case,
            no_prefetch=no_prefetch_result,
            prefetch_status=prefetch_result.status,
            prefetch_endpoint=endpoint_result if isinstance(endpoint_result, dict) else {},
            demand=demand_result,
        )
        rows.append(row)
        events.append(
            {
                "schema": EVENT_SCHEMA,
                "event_type": "prefetch_hit" if row["prefetch_hit"] else "prefetch_waste",
                "case": case.case_id,
                "mode": "astrakv_prefetch",
                "request_id": demand_result.request_id,
                "chunk_id": case.case_id,
                "target_tier": target_tier.value,
                "status": "observed",
                "metadata": {
                    "hit_evidence": row["hit_evidence"],
                    "latency_delta_pct": row["latency_delta_pct"],
                    "ttft_delta_pct": row["ttft_delta_pct"],
                },
            }
        )
    return rows, events


def build_cases(args: argparse.Namespace) -> list[PrefetchCase]:
    cases: list[PrefetchCase] = []
    for context_length in args.context_lengths:
        for repeat_index in range(args.repeat):
            case_id = f"ctx{context_length}_rep{repeat_index}"
            control_prompt = build_prompt(
                context_length=context_length,
                repeat_index=repeat_index,
                namespace="control",
                prompt_seed=args.prompt_seed,
                prompt_token_scale=args.prompt_token_scale,
            )
            prefetch_prompt = build_prompt(
                context_length=context_length,
                repeat_index=repeat_index,
                namespace="astrakv_prefetch",
                prompt_seed=args.prompt_seed,
                prompt_token_scale=args.prompt_token_scale,
            )
            cases.append(
                PrefetchCase(
                    case_id=case_id,
                    context_length=context_length,
                    repeat_index=repeat_index,
                    control_prompt=control_prompt,
                    prefetch_prompt=prefetch_prompt,
                )
            )
    return cases


def build_prompt(
    *,
    context_length: int,
    repeat_index: int,
    namespace: str,
    prompt_seed: str,
    prompt_token_scale: float,
) -> str:
    scaled_length = int(max(1, context_length) * max(0.05, prompt_token_scale))
    words = ["the"] * max(1, scaled_length - 96)
    context = " ".join(words)
    return (
        "Use the following synthetic context for repeated-prefix cache validation.\n"
        f"Seed: {prompt_seed}\n"
        f"Namespace: {namespace}\n"
        f"Repeat id: {repeat_index}\n"
        f"Context length target: {context_length}\n"
        f"{context}\n\n"
        "Summarize the context in exactly three short bullet points."
    )


def make_endpoint_request(
    *,
    args: argparse.Namespace,
    request_id: str,
    prompt: str,
    max_tokens: int,
    mode: str,
) -> EndpointRequest:
    return EndpointRequest(
        request_id=request_id,
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        metadata={"mode": mode, "backend": args.backend},
    )


def summarize_case(
    *,
    args: argparse.Namespace,
    case: PrefetchCase,
    no_prefetch: EndpointResult,
    prefetch_status: PrefetchStatus,
    prefetch_endpoint: dict[str, Any],
    demand: EndpointResult,
) -> dict[str, Any]:
    latency_delta = optional_delta(no_prefetch.latency_ms, demand.latency_ms)
    latency_delta_pct = optional_delta_pct(no_prefetch.latency_ms, demand.latency_ms)
    ttft_delta = optional_delta(no_prefetch.ttft_ms, demand.ttft_ms)
    ttft_delta_pct = optional_delta_pct(no_prefetch.ttft_ms, demand.ttft_ms)
    hit, evidence = infer_prefetch_hit(
        no_prefetch=no_prefetch,
        demand=demand,
        prefetch_status=prefetch_status,
        improvement_threshold_pct=args.hit_improvement_threshold_pct,
    )
    prefetch_latency = as_float(prefetch_endpoint.get("latency_ms"))
    prefetch_ttft = as_float(prefetch_endpoint.get("ttft_ms"))
    return {
        "case": case.case_id,
        "backend": args.backend,
        "model": args.model,
        "context_length": case.context_length,
        "repeat_index": case.repeat_index,
        "output_tokens": args.output_tokens,
        "warmup_output_tokens": args.warmup_output_tokens,
        "prefetch_submitted": 1,
        "prefetch_completed": int(prefetch_status == PrefetchStatus.COMPLETED),
        "prefetch_failed": int(prefetch_status == PrefetchStatus.FAILED),
        "prefetch_hit": int(hit),
        "prefetch_waste": int(prefetch_status == PrefetchStatus.COMPLETED and not hit),
        "hit_evidence": evidence,
        "prefetch_latency_ms": "" if prefetch_latency is None else prefetch_latency,
        "prefetch_ttft_ms": "" if prefetch_ttft is None else prefetch_ttft,
        "no_prefetch_status": no_prefetch.status,
        "prefetch_demand_status": demand.status,
        "no_prefetch_ttft_ms": "" if no_prefetch.ttft_ms is None else no_prefetch.ttft_ms,
        "prefetch_demand_ttft_ms": "" if demand.ttft_ms is None else demand.ttft_ms,
        "ttft_delta_ms": "" if ttft_delta is None else ttft_delta,
        "ttft_delta_pct": "" if ttft_delta_pct is None else ttft_delta_pct,
        "no_prefetch_latency_ms": no_prefetch.latency_ms,
        "prefetch_demand_latency_ms": demand.latency_ms,
        "latency_delta_ms": "" if latency_delta is None else latency_delta,
        "latency_delta_pct": "" if latency_delta_pct is None else latency_delta_pct,
        "no_prefetch_output_tokens": no_prefetch.output_tokens_observed,
        "prefetch_demand_output_tokens": demand.output_tokens_observed,
        "no_prefetch_error": no_prefetch.error,
        "prefetch_demand_error": demand.error,
    }


def infer_prefetch_hit(
    *,
    no_prefetch: EndpointResult,
    demand: EndpointResult,
    prefetch_status: PrefetchStatus,
    improvement_threshold_pct: float,
) -> tuple[bool, str]:
    if prefetch_status != PrefetchStatus.COMPLETED:
        return False, "prefetch_not_completed"
    if not no_prefetch.ok or not demand.ok:
        return False, "request_failed"
    latency_delta_pct = optional_delta_pct(no_prefetch.latency_ms, demand.latency_ms)
    ttft_delta_pct = optional_delta_pct(no_prefetch.ttft_ms, demand.ttft_ms)
    threshold = max(0.0, improvement_threshold_pct)
    if latency_delta_pct is not None and latency_delta_pct >= threshold:
        return True, "latency_improvement_heuristic"
    if ttft_delta_pct is not None and ttft_delta_pct >= threshold:
        return True, "ttft_improvement_heuristic"
    return False, "no_latency_or_ttft_improvement"


def make_event(
    event_type: str,
    case: PrefetchCase,
    *,
    mode: str,
    result: EndpointResult,
) -> dict[str, Any]:
    return {
        "schema": EVENT_SCHEMA,
        "event_type": event_type,
        "case": case.case_id,
        "mode": mode,
        "request_id": result.request_id,
        "chunk_id": case.case_id,
        "target_tier": "unknown",
        "status": result.status,
        "metadata": result.to_record(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prefetch_benchmark_rows(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    benchmark_rows: list[dict[str, Any]] = []
    context_lengths = sorted({int(row["context_length"]) for row in rows})
    for context_length in context_lengths:
        context_rows = [row for row in rows if int(row["context_length"]) == context_length]
        benchmark_rows.append(
            summarize_prefetch_mode(
                rows=context_rows,
                mode="no_prefetch",
                backend="astrakv_no_prefetch",
                case=f"ctx{context_length}_no_prefetch",
                args=args,
            )
        )
        benchmark_rows.append(
            summarize_prefetch_mode(
                rows=context_rows,
                mode="prefetch_demand",
                backend="astrakv_prefetch_demand",
                case=f"ctx{context_length}_prefetch_demand",
                args=args,
            )
        )
    return benchmark_rows


def summarize_prefetch_mode(
    *,
    rows: list[dict[str, Any]],
    mode: str,
    backend: str,
    case: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    status_key = "no_prefetch_status" if mode == "no_prefetch" else "prefetch_demand_status"
    ttft_key = "no_prefetch_ttft_ms" if mode == "no_prefetch" else "prefetch_demand_ttft_ms"
    latency_key = "no_prefetch_latency_ms" if mode == "no_prefetch" else "prefetch_demand_latency_ms"
    tokens_key = "no_prefetch_output_tokens" if mode == "no_prefetch" else "prefetch_demand_output_tokens"

    ok_rows = [row for row in rows if row.get(status_key) == "ok"]
    latencies = [value for value in (as_float(row.get(latency_key)) for row in ok_rows) if value is not None]
    ttfts = [value for value in (as_float(row.get(ttft_key)) for row in ok_rows) if value is not None]
    observed_tokens = sum(int(as_float(row.get(tokens_key)) or 0) for row in ok_rows)
    total_seconds = sum(latencies) / 1000.0
    context_length = int(rows[0]["context_length"]) if rows else ""
    errors = [
        str(row.get("no_prefetch_error" if mode == "no_prefetch" else "prefetch_demand_error") or "")
        for row in rows
    ]
    return {
        "case": case,
        "backend": backend,
        "model": args.model,
        "batch_size": 1,
        "context_length": context_length,
        "output_tokens": args.output_tokens,
        "request_count": len(rows),
        "success_count": len(ok_rows),
        "ttft_ms": mean(ttfts),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "tpot_ms": "",
        "tpot_p50_ms": "",
        "tpot_p95_ms": "",
        "latency_ms": mean(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "throughput_tokens_s": observed_tokens / max(total_seconds, 1e-9),
        "process_rss_peak_mb": "",
        "cpu_memory_peak_mb": "",
        "gpu_memory_peak_mb": "",
        "gpu_util_peak_pct": "",
        "disk_read_delta_mb": "",
        "disk_write_delta_mb": "",
        "sample_count": 0,
        "gpu_probe": "unavailable",
        "disk_probe": "unavailable",
        "status": "ok" if len(ok_rows) == len(rows) else "error",
        "errors": " | ".join(sorted({error for error in errors if error})),
    }


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_effective_config(path: Path, args: argparse.Namespace, config: dict[str, Any]) -> None:
    payload = {
        "source_config": config.get("_config_path", ""),
        "run_name": args.run_name,
        "backend": args.backend,
        "base_url": args.base_url,
        "model": args.model,
        "context_lengths": args.context_lengths,
        "repeat": args.repeat,
        "output_tokens": args.output_tokens,
        "warmup_output_tokens": args.warmup_output_tokens,
        "target_tier": args.target_tier,
        "hit_improvement_threshold_pct": args.hit_improvement_threshold_pct,
        "cache_events": args.cache_events,
        "raw_config": args.raw_config,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    submitted = sum(int(row["prefetch_submitted"]) for row in rows)
    completed = sum(int(row["prefetch_completed"]) for row in rows)
    failed = sum(int(row["prefetch_failed"]) for row in rows)
    hits = sum(int(row["prefetch_hit"]) for row in rows)
    waste = sum(int(row["prefetch_waste"]) for row in rows)
    hit_rate = hits / max(1, completed)
    waste_rate = waste / max(1, completed)
    log_summary = summarize_cache_event_files(args.cache_events)

    lines = [
        "# Real Selective Prefetch Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Backend: `{args.backend}`",
        f"- Model: `{args.model}`",
        f"- Endpoint: `{args.base_url}/v1`",
        f"- Context lengths: `{', '.join(str(item) for item in args.context_lengths)}`",
        f"- Repeat: `{args.repeat}`",
        f"- Demand output tokens: `{args.output_tokens}`",
        f"- Warmup output tokens: `{args.warmup_output_tokens}`",
        f"- Hit threshold: `{args.hit_improvement_threshold_pct}%` latency or TTFT improvement when logs are not row-matchable",
        "",
        "## Summary",
        "",
        f"- Prefetch submitted: `{submitted}`",
        f"- Prefetch completed: `{completed}`",
        f"- Prefetch failed: `{failed}`",
        f"- Prefetch hit: `{hits}`",
        f"- Prefetch waste: `{waste}`",
        f"- Prefetch hit rate: `{hit_rate:.4f}`",
        f"- Prefetch waste rate: `{waste_rate:.4f}`",
        f"- Event rows: `{len(events)}`",
        "",
        "## Case Comparison",
        "",
        "| case | context | prefetch | hit | evidence | no-prefetch TTFT | prefetch TTFT | TTFT delta % | no-prefetch latency | prefetch latency | latency delta % |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {context_length} | {prefetch_completed} | {prefetch_hit} | "
            "{hit_evidence} | {no_prefetch_ttft_ms} | {prefetch_demand_ttft_ms} | "
            "{ttft_delta_pct} | {no_prefetch_latency_ms} | {prefetch_demand_latency_ms} | "
            "{latency_delta_pct} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Cache Event Evidence",
            "",
        ]
    )
    if log_summary:
        lines.extend(
            [
                "| file | total | cache hit | cache miss | cache load | cache store | cache offload |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in log_summary:
            counts = item["event_type_counts"]
            lines.append(
                f"| `{item['path']}` | {item['total_events']} | "
                f"{counts.get('cache_hit', 0)} | {counts.get('cache_miss', 0)} | "
                f"{counts.get('cache_load', 0)} | {counts.get('cache_store', 0)} | "
                f"{counts.get('cache_offload', 0)} |"
            )
    else:
        lines.extend(
            [
                "No P0-5 `cache_events.jsonl` file was provided.",
                "",
                "For official claims, parse vLLM/LMCache server logs with `scripts/benchmark/extract_cache_events.py` and rerun or attach this report with cache-event evidence.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `prefetch_submitted` means AstraKV-W issued a real endpoint warmup request.",
            "- `prefetch_completed` means the warmup request completed successfully at the backend endpoint.",
            "- `prefetch_hit` is a latency/TTFT heuristic unless a cache-event file is supplied as external evidence.",
            "- `prefetch_waste` means a completed warmup did not show the configured latency or TTFT improvement.",
            "- This runner intentionally does not modify vLLM or LMCache internals.",
            "",
            "## Artifacts",
            "",
            "- `prefetch_results.csv`",
            "- `prefetch_benchmark_results.csv`",
            "- `prefetch_events.jsonl`",
            "- `prefetch_config.json`",
            "- `prefetch_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_cache_event_files(paths: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            summaries.append(
                {
                    "path": str(path),
                    "total_events": 0,
                    "event_type_counts": {"missing": 1},
                }
            )
            continue
        counts: dict[str, int] = {}
        total = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counts["parse_error"] = counts.get("parse_error", 0) + 1
                    continue
                total += 1
                event_type = str(record.get("event_type", "unknown"))
                counts[event_type] = counts.get(event_type, 0) + 1
        summaries.append(
            {
                "path": str(path),
                "total_events": total,
                "event_type_counts": dict(sorted(counts.items())),
            }
        )
    return summaries


def json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "endpoint_request" and is_json_scalar(value)
    }


def is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def optional_delta(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None:
        return None
    return baseline - value


def optional_delta_pct(baseline: float | None, value: float | None) -> float | None:
    if baseline in (None, 0) or value is None:
        return None
    return ((baseline - value) / baseline) * 100.0


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_tier(value: str) -> MemoryTier:
    try:
        return MemoryTier(value)
    except ValueError:
        return MemoryTier.GPU


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
