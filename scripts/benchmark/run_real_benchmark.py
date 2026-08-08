"""Run a real vLLM OpenAI-compatible endpoint smoke benchmark.

This script intentionally benchmarks an already-running HTTP endpoint. It does
not import or modify vLLM, LMCache, runtime, scheduler, or prefetch code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4, uuid5

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - config files are optional for CLI smoke runs.
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgx_metrics_collector import DgxMetricsCollector, DgxSummary
from astrakv.benchmarks.runtime_workload import WorkloadContractError, load_runtime_workload_jsonl
from astrakv.benchmarks.experiment_manifest import ExperimentManifest, file_sha256, input_hashes, redact_command_text, write_experiment_manifest
from astrakv.benchmarks.runtime_artifacts import export_online_control_artifacts
from astrakv.runtime.request_context import (
    AuthenticatedJsonHttpRequestContextClient,
    JsonHttpRequestContextClient,
    RequestContextArtifact,
    RequestContextClient,
    RequestContextJsonlArtifact,
    RuntimeRequestContext,
)
from scripts.reporting.plot_benchmarks import plot_csv


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_BATCH_NONCE_LOCK = threading.Lock()
_USED_BATCH_NONCES: set[str] = set()
_USED_REQUEST_NONCES: set[str] = set()
_DISPATCHED_REQUEST_CONTEXTS: set[tuple[str, str, str]] = set()
_SENSITIVE_CONFIG_KEY_PARTS = {"api_key", "authorization", "password", "secret"}


@dataclass(frozen=True)
class RequestResult:
    case: str
    backend: str
    request_id: int | str
    model: str
    batch_size: int
    context_length: int
    output_tokens_target: int
    output_tokens_observed: int
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    throughput_tokens_s: float
    output_text: str
    cpu_memory_mb_before: float
    cpu_memory_mb_after: float
    gpu_memory_mb_before: float | None
    gpu_memory_mb_after: float | None
    gpu_probe: str
    status: str
    error: str
    run_id: str = ""
    prefix_id: str = ""
    prefix_hash: str = ""
    arrival_index: int | None = None
    reuse_bucket: str = ""
    reuse_ratio: float | None = None
    cache_key: str = ""
    workload_case: str = ""
    prompt_hash: str = ""
    sample_id: str = ""
    ground_truth: str = ""
    dataset: str = ""
    task: str = ""
    workload_type: str = ""
    reuse_group: str = ""
    reuse_group_size: int | None = None
    shared_context: bool | None = None
    expected_reuse: int | None = None
    estimated_kv_tokens: int | None = None
    estimated_reusable_tokens: int | None = None
    workflow_id: str = ""
    parent_request_id: str = ""
    subtask_index: int = 0
    cache_state: str = "unknown"
    first_sse_ms: float | None = None
    endpoint_response_id: str = ""
    request_started_s: float = 0.0
    request_ended_s: float = 0.0
    sample_path: str = ""
    attribution_mode: str = "unattributed"
    request_nonce: str = ""
    runtime_association_status: str = "unlinked"
    runtime_request_id: str = ""
    runtime_event_id: str = ""
    request_context_error: str = ""
    # vLLM returns these only when its token-evidence options are enabled.
    # They make KV-Core correctness comparisons about generated token IDs, not
    # merely decoded text that can hide tokenizer-boundary differences.
    output_token_ids: tuple[int, ...] = ()
    finish_reason: str = ""
    deterministic_logprob: float | None = None


def main() -> int:
    raw_args = parse_args()
    config = load_config(raw_args.config)
    args = resolve_args(raw_args, config)
    validate_configured_workload(args.workload_jsonl, args.require_canonical_workload)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_effective_config(output_dir / "benchmark_config.json", args, config)
    request_results: list[RequestResult] = []
    failures: list[str] = []
    metric_summaries: dict[str, DgxSummary] = {}
    request_context_artifact: RequestContextArtifact = RequestContextJsonlArtifact(
        output_dir / "request_context.jsonl"
    )
    request_context_client = build_runtime_request_context_client(
        args.request_context_url, run_id=args.run_id,
        session_id=args.request_context_session_id, secret_hex=args.request_context_secret_hex,
    )

    workload_rows = load_workload_rows(args.workload_jsonl)
    if workload_rows:
        request_results, metric_summaries = run_workload_rows(
            args,
            output_dir,
            workload_rows,
            request_context_client=request_context_client,
            request_context_artifact=request_context_artifact,
        )
        failures.extend(item.error for item in request_results if item.status != "ok")
    else:
        for context_length in args.context_lengths:
            for batch_size in args.batch_sizes:
                case = f"bs{batch_size}_ctx{context_length}_out{args.output_tokens}"
                collector = start_metrics_collector(args, output_dir, case)
                for repeat_index in range(args.repeat):
                    batch_results = run_batch(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    backend=args.backend,
                    case=case,
                    repeat_index=repeat_index,
                    batch_size=batch_size,
                    context_length=context_length,
                    output_tokens=args.output_tokens,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    system_prompt=args.system_prompt,
                    prompt_seed=args.prompt_seed,
                    prompt_token_scale=args.prompt_token_scale,
                    run_id=args.run_id,
                    metrics_collector=collector,
                    request_context_client=request_context_client,
                    request_context_artifact=request_context_artifact,
                )
                    request_results.extend(batch_results)
                    failures.extend(item.error for item in batch_results if item.status != "ok")
                if collector is not None:
                    metric_summaries[case] = collector.stop()

    write_jsonl(output_dir / "request_results.jsonl", request_results)
    rows = summarize_results(request_results, metric_summaries)
    csv_path = output_dir / "benchmark_results.csv"
    write_csv(csv_path, rows)
    write_charts(csv_path, output_dir / "charts")
    write_report(output_dir / "benchmark_report.md", args, rows, request_results, failures)
    finalize_experiment_manifest(output_dir, args, config)

    if failures:
        print(f"Benchmark completed with {len(failures)} failed request(s).", file=sys.stderr)
        return 1

    print(f"Benchmark outputs written to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config file for real endpoint runs.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None, help="Backend label written into result rows.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--prompt-seed", default=None)
    parser.add_argument("--prompt-token-scale", type=float, default=None)
    parser.add_argument("--metrics-interval", type=float, default=None)
    parser.add_argument("--disk-device", default=None)
    parser.add_argument("--process-name-filter", action="append", default=None)
    parser.add_argument("--workload-jsonl", default=None, help="Optional manifest with prompt, request_id, prefix_id, and arrival_index.")
    parser.add_argument("--run-id", default=None, help="Stable identifier written to every request result.")
    parser.add_argument("--workload-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--random-seed", default=None)
    parser.add_argument("--cache-state", choices=("cold", "warm", "unknown"), default=None)
    parser.add_argument("--connector-version", default=None)
    parser.add_argument("--pair-id", default=None, help="Required identity shared by a baseline/variant pair.")
    parser.add_argument("--pair-role", choices=("baseline", "variant"), default=None)
    parser.add_argument("--claim-scope", choices=("benchmark", "online_control", "kv_core"), default=None)
    parser.add_argument(
        "--online-artifact", action="append", default=None,
        help=(
            "Online evidence as role=path; canonical roles are "
            "backend_capabilities,backend_binding_events,runtime_events_raw,"
            "astrakv_runtime_commands,runtime_command_receipts,runtime_structured_events,"
            "online_profile_checkpoint."
        ),
    )
    parser.add_argument(
        "--runtime-state-dir", default=None,
        help="RuntimeControlHost state directory; exports normalized online evidence automatically.",
    )
    parser.add_argument("--request-context-url", default=None, help="Optional loopback URL for runtime request-context handoff.")
    parser.add_argument("--request-context-session-id", default=None)
    parser.add_argument("--request-context-secret-hex", default=None)
    parser.add_argument(
        "--enable-samples",
        action="store_true",
        default=None,
        help="Write continuous samples/<case>_samples.csv metrics during each case.",
    )
    parser.add_argument(
        "--disable-samples",
        action="store_false",
        dest="enable_samples",
        help="Disable continuous DGX metrics sampling even when a config has metrics.",
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
    metrics_cfg = _dict(config.get("metrics"))
    runtime_cfg = _dict(config.get("runtime"))

    run_name = str(config.get("run_name") or "real_vllm_endpoint")
    if config:
        output_root = raw.output_dir or "results"
        output_dir = Path(output_root) / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(raw.output_dir or "results/local_smoke_test")

    base_url = first_non_empty(
        raw.base_url,
        os.environ.get("VLLM_BASE_URL"),
        backend_cfg.get("base_url"),
        DEFAULT_BASE_URL,
    )
    api_key = first_non_empty(
        raw.api_key,
        os.environ.get("OPENAI_API_KEY"),
        backend_cfg.get("api_key"),
        "EMPTY",
    )
    model = first_non_empty(
        raw.model,
        os.environ.get("VLLM_MODEL"),
        backend_cfg.get("model"),
        DEFAULT_MODEL,
    )

    enable_samples = raw.enable_samples
    if enable_samples is None:
        enable_samples = bool(config and metrics_cfg)

    return argparse.Namespace(
        config=raw.config,
        run_name=run_name,
        backend=str(first_non_empty(raw.backend, backend_cfg.get("name"), "vllm_openai_endpoint")),
        base_url=normalize_base_url(str(base_url)),
        api_key=str(api_key),
        model=str(model),
        output_dir=str(output_dir),
        context_lengths=list(raw.context_lengths or matrix_cfg.get("context_lengths") or [512, 1024]),
        batch_sizes=list(raw.batch_sizes or matrix_cfg.get("batch_sizes") or [1]),
        output_tokens=int(raw.output_tokens or matrix_cfg.get("output_tokens") or 64),
        repeat=int(raw.repeat or matrix_cfg.get("repeat") or 1),
        timeout=float(raw.timeout or backend_cfg.get("timeout_seconds") or 600.0),
        temperature=float(raw.temperature if raw.temperature is not None else workload_cfg.get("temperature", 0.0)),
        top_p=float(raw.top_p if raw.top_p is not None else workload_cfg.get("top_p", 1.0)),
        system_prompt=str(
            first_non_empty(
                raw.system_prompt,
                workload_cfg.get("system_prompt"),
                "You are a concise benchmark assistant. Answer directly.",
            )
        ),
        prompt_seed=str(
            first_non_empty(
                raw.prompt_seed,
                workload_cfg.get("prompt_seed"),
                "AstraKV-W evaluates memory-constrained large language model inference.",
            )
        ),
        prompt_token_scale=float(
            raw.prompt_token_scale
            if raw.prompt_token_scale is not None
            else workload_cfg.get("prompt_token_scale", 1.0)
        ),
        enable_samples=bool(enable_samples),
        metrics_interval=float(raw.metrics_interval or metrics_cfg.get("sample_interval_seconds") or 0.5),
        disk_device=str(raw.disk_device if raw.disk_device is not None else metrics_cfg.get("disk_device", "")),
        process_name_filters=tuple(
            raw.process_name_filter
            if raw.process_name_filter is not None
            else metrics_cfg.get("process_name_filters", ())
        ),
        workload_jsonl=str(raw.workload_jsonl or config.get("workload_jsonl") or ""),
        run_id=str(raw.run_id or config.get("run_id") or run_name),
        workload_id=str(raw.workload_id or config.get("workload_id") or "unknown"),
        model_revision=str(raw.model_revision or config.get("model_revision") or "unknown"),
        tokenizer_revision=str(raw.tokenizer_revision or config.get("tokenizer_revision") or "unknown"),
        dtype=str(raw.dtype or config.get("dtype") or "unknown"),
        quantization=str(raw.quantization or config.get("quantization") or "unknown"),
        random_seed=str(raw.random_seed or config.get("random_seed") or "unknown"),
        cache_state=str(raw.cache_state or config.get("cache_state") or "unknown"),
        connector_version=str(raw.connector_version or config.get("connector_version") or "unknown"),
        pair_id=str(raw.pair_id or config.get("pair_id") or ""),
        pair_role=str(raw.pair_role or config.get("pair_role") or ""),
        claim_scope=str(raw.claim_scope or config.get("claim_scope") or "benchmark"),
        online_artifact=tuple(raw.online_artifact or config.get("online_artifact") or ()),
        runtime_state_dir=str(raw.runtime_state_dir or config.get("runtime_state_dir") or ""),
        request_context_url=str(raw.request_context_url or runtime_cfg.get("request_context_url") or ""),
        request_context_session_id=str(raw.request_context_session_id or runtime_cfg.get("request_context_session_id") or os.environ.get("ASTRAKV_RUNTIME_CONTROL_SESSION_ID", "")),
        request_context_secret_hex=str(raw.request_context_secret_hex or runtime_cfg.get("request_context_secret_hex") or os.environ.get("ASTRAKV_RUNTIME_CONTROL_SECRET_HEX", "")),
        require_canonical_workload=bool(config.get("require_canonical_workload", False)),
    )


def build_runtime_request_context_client(
    context_url: str, *, run_id: str, session_id: str, secret_hex: str,
) -> RequestContextClient | None:
    if not context_url:
        return None
    if bool(session_id) != bool(secret_hex):
        raise ValueError("runtime request context authentication requires both session ID and secret")
    if not session_id:
        return JsonHttpRequestContextClient(context_url)
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise ValueError("runtime request context secret must be hexadecimal") from exc
    return AuthenticatedJsonHttpRequestContextClient(
        context_url, run_id=run_id, session_id=session_id, secret=secret,
    )


def run_batch(
    *,
    base_url: str,
    api_key: str,
    model: str,
    backend: str,
    case: str,
    repeat_index: int,
    batch_size: int,
    context_length: int,
    output_tokens: int,
    timeout: float,
    temperature: float,
    top_p: float,
    system_prompt: str,
    prompt_seed: str,
    prompt_token_scale: float,
    run_id: str = "",
    metrics_collector: DgxMetricsCollector | None = None,
    request_context_client: RequestContextClient | None = None,
    request_context_artifact: RequestContextArtifact | None = None,
    batch_nonce: str | None = None,
) -> list[RequestResult]:
    invocation_nonce = _reserve_batch_nonce(batch_nonce)
    request_ids = tuple(
        f"{run_id or 'run'}:{case}:{invocation_nonce}:r{repeat_index}:i{index}"
        for index in range(batch_size)
    )
    request_nonces = tuple(
        str(uuid5(UUID(invocation_nonce), f"request-{index}"))
        for index in range(batch_size)
    )
    _reserve_request_contexts(run_id, request_ids, request_nonces)
    attribution_mode = "shared_batch" if batch_size > 1 else "exclusive_request"
    boundary = (
        metrics_collector.shared_batch_scope(request_ids)
        if metrics_collector is not None and batch_size > 1
        else nullcontext()
    )
    with boundary:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(
                    run_one_request,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    backend=backend,
                    case=case,
                    request_id=request_id,
                    batch_size=batch_size,
                    context_length=context_length,
                    output_tokens=output_tokens,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                    prompt_seed=prompt_seed,
                    prompt_token_scale=prompt_token_scale,
                    request_metadata={"run_id": run_id},
                    metrics_collector=metrics_collector,
                    attribution_mode=attribution_mode,
                    request_nonce=request_nonces[request_index],
                    request_context_client=request_context_client,
                    request_context_artifact=request_context_artifact,
                    request_context_reserved=True,
                )
                for request_index, request_id in enumerate(request_ids)
            ]
            return [future.result() for future in as_completed(futures)]


def run_one_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    backend: str,
    case: str,
    request_id: int | str,
    batch_size: int,
    context_length: int,
    output_tokens: int,
    timeout: float,
    temperature: float,
    top_p: float,
    system_prompt: str,
    prompt_seed: str,
    prompt_token_scale: float,
    prompt: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    metrics_collector: DgxMetricsCollector | None = None,
    attribution_mode: str | None = None,
    request_nonce: str | None = None,
    request_context_client: RequestContextClient | None = None,
    request_context_artifact: RequestContextArtifact | None = None,
    request_context_reserved: bool = False,
) -> RequestResult:
    request_metadata = request_metadata or {}
    cpu_before = cpu_rss_mb()
    gpu_before, gpu_probe = gpu_memory_mb()
    request_started_s = time.time()
    started = time.perf_counter()
    ttft_ms: float | None = None
    first_sse_ms: float | None = None
    endpoint_response_id = ""
    observed_tokens = 0
    usage_tokens: int | None = None
    output_parts: list[str] = []
    output_token_ids: list[int] = []
    finish_reason = ""
    deterministic_logprob = 0.0
    logprob_observed = False
    status = "ok"
    error = ""
    nonce = _canonical_request_nonce(request_nonce)
    runtime_association_status = "unlinked"
    runtime_request_id = ""
    runtime_event_id = ""
    request_context_error = ""
    run_id = str(request_metadata.get("run_id") or "")
    task_metadata = request_metadata.get("metadata") if isinstance(request_metadata.get("metadata"), dict) else {}
    if not request_context_reserved:
        _reserve_request_contexts(run_id, (str(request_id),), (nonce,))
    if request_context_client is not None or request_context_artifact is not None:
        try:
            context = RuntimeRequestContext(
                run_id=run_id,
                request_id=str(request_id),
                case=case,
                request_nonce=nonce,
                request_started_s=request_started_s,
                metadata={
                    "prefix_id": str(request_metadata.get("prefix_id") or ""),
                    "cache_key": str(request_metadata.get("cache_key") or ""),
                    "prefix_hash": str(request_metadata.get("prefix_hash") or ""),
                    "reuse_bucket": str(request_metadata.get("reuse_bucket") or ""),
                    "reuse_ratio": request_metadata.get("reuse_ratio"),
                    "arrival_index": request_metadata.get("arrival_index"),
                    "context_length": request_metadata.get("context_length"),
                    "context_length_source": request_metadata.get("context_length_source", ""),
                    "scenario": str(task_metadata.get("scenario") or request_metadata.get("scenario") or ""),
                    "workload_case": str(request_metadata.get("case") or ""),
                    "cache_state": str(request_metadata.get("cache_state") or ""),
                    "object_key": str(
                        request_metadata.get("cache_key")
                        or request_metadata.get("prefix_id")
                        or request_metadata.get("workflow_id")
                        or request_id
                    ),
                },
            )
            if request_context_artifact is not None:
                request_context_artifact.append(context)
            if request_context_client is not None:
                receipt = request_context_client.publish(context)
                if receipt.matches(context):
                    runtime_association_status = "linked"
                    runtime_request_id = receipt.runtime_request_id
                    runtime_event_id = receipt.runtime_event_id
        except Exception as exc:  # Context linkage must not suppress endpoint diagnostics.
            request_context_error = f"{type(exc).__name__}: {exc}"

    request_scope = metrics_collector.request_scope(request_id) if metrics_collector is not None else nullcontext()
    with request_scope:
        try:
            supplied_messages = task_metadata.get("messages")
            messages = supplied_messages if isinstance(supplied_messages, list) and supplied_messages else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt or build_prompt(context_length, request_id, prompt_seed, prompt_token_scale)},
            ]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
                "user": str(request_id),
                "_astrakv_request_id": str(request_id),
            }
            if backend == "vllm-lmcache-kv-core":
                # vLLM's documented compatibility extension returns stable
                # token identifiers in streamed choices.  Other benchmark
                # backends retain their existing request shape.
                payload.update({"return_token_ids": True, "logprobs": True, "top_logprobs": 1})
            for event in stream_chat_completion(
                f"{base_url}/v1/chat/completions",
                payload,
                api_key,
                timeout,
            ):
                if first_sse_ms is None:
                    first_sse_ms = (time.perf_counter() - started) * 1000.0
                if not endpoint_response_id and isinstance(event.get("id"), str):
                    endpoint_response_id = event["id"]
                usage = event.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens")
                    if isinstance(completion_tokens, int):
                        usage_tokens = completion_tokens

                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    token_ids = extract_choice_token_ids(choice)
                    if token_ids:
                        output_token_ids.extend(token_ids)
                    logprobs = choice.get("logprobs")
                    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
                    if isinstance(content_logprobs, list):
                        for token in content_logprobs:
                            if not isinstance(token, dict):
                                continue
                            try:
                                deterministic_logprob += float(token["logprob"])
                                logprob_observed = True
                            except (KeyError, TypeError, ValueError):
                                continue
                    delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if content:
                        output_parts.append(str(content))
                        observed_tokens += 1
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - this is a smoke-test diagnostic.
            status = "error"
            error = classify_error(exc)

    ended = time.perf_counter()
    request_ended_s = time.time()
    latency_ms = (ended - started) * 1000.0
    output_tokens_observed = max(observed_tokens, usage_tokens or 0)
    if status == "ok" and backend == "vllm-lmcache-kv-core" and not output_token_ids:
        status = "error"
        error = (
            "KV-Core benchmark requires real output token IDs from the vLLM response; "
            "the endpoint returned no token-id evidence."
        )
    if status == "ok":
        status, error = finalize_request_status(
            observed_tokens=observed_tokens,
            output_tokens_observed=output_tokens_observed,
            ttft_ms=ttft_ms,
            first_sse_ms=first_sse_ms,
        )
    generated_after_first = max(0, output_tokens_observed - 1)
    if status == "ok" and ttft_ms is not None and generated_after_first > 0:
        tpot_ms = max(0.0, latency_ms - ttft_ms) / generated_after_first
    else:
        tpot_ms = None
    throughput = output_tokens_observed / max((ended - started), 1e-9)
    gpu_after, gpu_probe_after = gpu_memory_mb()

    return RequestResult(
        case=case,
        backend=backend,
        request_id=request_id,
        model=model,
        batch_size=batch_size,
        context_length=context_length,
        output_tokens_target=output_tokens,
        output_tokens_observed=output_tokens_observed,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=latency_ms,
        throughput_tokens_s=throughput,
        output_text="".join(output_parts),
        cpu_memory_mb_before=cpu_before,
        cpu_memory_mb_after=cpu_rss_mb(),
        gpu_memory_mb_before=gpu_before,
        gpu_memory_mb_after=gpu_after,
        gpu_probe=gpu_probe_after if gpu_probe_after != "unavailable" else gpu_probe,
        status=status,
        error=error,
        run_id=run_id,
        prefix_id=str(request_metadata.get("prefix_id") or ""),
        prefix_hash=str(request_metadata.get("prefix_hash") or ""),
        arrival_index=as_int_or_none(request_metadata.get("arrival_index")),
        reuse_bucket=str(request_metadata.get("reuse_bucket") or ""),
        reuse_ratio=as_float_or_none(request_metadata.get("reuse_ratio")),
        cache_key=str(request_metadata.get("cache_key") or ""),
        workload_case=str(request_metadata.get("case") or ""),
        prompt_hash=str(request_metadata.get("prompt_hash") or ""),
        sample_id=str(task_metadata.get("sample_id") or ""),
        ground_truth=str(task_metadata.get("ground_truth") or ""),
        dataset=str(task_metadata.get("dataset") or ""),
        task=str(task_metadata.get("task") or ""),
        workload_type=str(task_metadata.get("workload_type") or ""),
        reuse_group=str(task_metadata.get("reuse_group") or ""),
        reuse_group_size=as_int_or_none(task_metadata.get("reuse_group_size")),
        shared_context=task_metadata.get("shared_context") if isinstance(task_metadata.get("shared_context"), bool) else None,
        expected_reuse=as_int_or_none(task_metadata.get("expected_reuse")),
        estimated_kv_tokens=as_int_or_none(task_metadata.get("estimated_kv_tokens")),
        estimated_reusable_tokens=as_int_or_none(task_metadata.get("estimated_reusable_tokens")),
        workflow_id=str(request_metadata.get("workflow_id") or request_id),
        parent_request_id=str(request_metadata.get("parent_request_id") or request_id),
        subtask_index=as_int_or_none(request_metadata.get("subtask_index")) or 0,
        cache_state=str(request_metadata.get("cache_state") or "unknown"),
        first_sse_ms=first_sse_ms,
        endpoint_response_id=endpoint_response_id,
        request_started_s=request_started_s,
        request_ended_s=request_ended_s,
        sample_path=str(metrics_collector.output_csv) if metrics_collector is not None else "",
        attribution_mode=attribution_mode or ("exclusive_request" if metrics_collector is not None else "unattributed"),
        request_nonce=nonce,
        runtime_association_status=runtime_association_status,
        runtime_request_id=runtime_request_id,
        runtime_event_id=runtime_event_id,
        request_context_error=request_context_error,
        output_token_ids=tuple(output_token_ids),
        finish_reason=finish_reason,
        deterministic_logprob=deterministic_logprob if logprob_observed else None,
    )


def extract_choice_token_ids(choice: dict[str, Any]) -> list[int]:
    """Extract token IDs from a vLLM/OpenAI streamed choice.

    vLLM 0.23 may expose IDs through its compatibility ``token_ids`` field,
    while its OpenAI logprobs response carries the same evidence as
    ``logprobs.content[].token_id``.  Prefer the explicit extension and use
    logprobs only as a structured fallback; never reconstruct IDs from text.
    """
    explicit = choice.get("token_ids")
    if isinstance(explicit, list):
        normalized = [
            token_id
            for token_id in explicit
            if isinstance(token_id, int) and not isinstance(token_id, bool)
        ]
        if normalized:
            return normalized

    delta = choice.get("delta")
    if isinstance(delta, dict):
        delta_ids = delta.get("token_ids")
        if isinstance(delta_ids, list):
            normalized = [
                token_id
                for token_id in delta_ids
                if isinstance(token_id, int) and not isinstance(token_id, bool)
            ]
            if normalized:
                return normalized

    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return []
    return [
        token_id
        for token in content
        if isinstance(token, dict)
        for token_id in [token.get("token_id")]
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    ]


def finalize_request_status(
    *,
    observed_tokens: int,
    output_tokens_observed: int,
    ttft_ms: float | None,
    first_sse_ms: float | None,
) -> tuple[str, str]:
    if observed_tokens <= 0 or output_tokens_observed <= 0:
        if first_sse_ms is not None:
            return "error", "Streamed response ended before any generated content tokens were observed."
        return "error", "Request completed without any generated content tokens."
    if ttft_ms is None:
        return "error", "Generated content was observed but TTFT could not be measured."
    return "ok", ""


def load_workload_rows(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        return [row.to_record() for row in load_runtime_workload_jsonl(path)]
    except WorkloadContractError as exc:
        raise SystemExit(f"invalid runtime workload: {exc}") from exc


def validate_configured_workload(path: str, required: bool) -> None:
    if required and not path:
        raise SystemExit("configured endpoint benchmarks require a canonical workload JSONL")


def run_workload_rows(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    request_context_client: RequestContextClient | None = None,
    request_context_artifact: RequestContextArtifact | None = None,
) -> tuple[list[RequestResult], dict[str, DgxSummary]]:
    results: list[RequestResult] = []
    summaries: dict[str, DgxSummary] = {}
    if args.backend == "vllm-lmcache-kv-core":
        for row in rows:
            context_length, _ = resolve_workload_context_length(row)
            if context_length <= 0:
                request_id = str(row.get("request_id") or "<unknown>")
                raise SystemExit(
                    f"KV-Core workload row {request_id} is missing a positive context_length; "
                    "provide tokenizer-derived context_length in the workload or metadata."
                )
    for index, row in enumerate(rows):
        sleep_before_s = as_float_or_none(row.get("sleep_before_s"))
        if sleep_before_s is not None and sleep_before_s > 0.0:
            time.sleep(sleep_before_s)
        request_id = str(row["request_id"])
        # Sample paths are keyed by case.  Preserve a caller-provided case while
        # ensuring repeated manifest cases do not overwrite one another.
        original_case = str(row.get("case") or request_id)
        case = f"{original_case}_{index:05d}"
        collector = start_metrics_collector(args, output_dir, case)
        prompt = str(row["prompt"])
        context_length, context_length_source = resolve_workload_context_length(row)
        result = run_one_request(
            base_url=args.base_url, api_key=args.api_key, model=args.model, backend=args.backend,
            case=case, request_id=request_id, batch_size=max(1, as_int_or_none(row.get("batch_size")) or 1),
            context_length=context_length,
            output_tokens=max(1, as_int_or_none(row.get("expected_output_tokens")) or args.output_tokens),
            timeout=args.timeout, temperature=args.temperature, top_p=args.top_p, system_prompt=args.system_prompt,
            prompt_seed=args.prompt_seed, prompt_token_scale=args.prompt_token_scale, prompt=prompt,
            request_metadata={
                **row,
                "run_id": args.run_id,
                "cache_state": args.cache_state,
                "context_length": context_length,
                "context_length_source": context_length_source,
                "prompt_hash": row.get("prompt_hash") or stable_hash(prompt),
            },
            metrics_collector=collector,
            attribution_mode="exclusive_request",
            request_context_client=request_context_client,
            request_context_artifact=request_context_artifact,
        )
        results.append(result)
        if collector is not None:
            summaries[case] = collector.stop()
    return results, summaries


def build_prompt(
    context_length: int,
    request_id: int,
    prompt_seed: str,
    prompt_token_scale: float,
) -> str:
    # Use common words rather than synthetic IDs: Qwen tokenizes strings like
    # "ctx0_0001" into many pieces, which can accidentally exceed max_model_len.
    scaled_length = int(max(1, context_length) * max(0.05, prompt_token_scale))
    words = ["the"] * max(1, scaled_length - 96)
    context = " ".join(words)
    return (
        "Use the following synthetic context for a local inference smoke test.\n"
        f"Seed: {prompt_seed}\n"
        f"Request id: {request_id}\n"
        f"{context}\n\n"
        "Summarize the context in exactly three short bullet points."
    )


def resolve_workload_context_length(row: dict[str, Any]) -> tuple[int, str]:
    """Resolve a workload's declared prompt-token count without text guessing."""
    direct = as_int_or_none(row.get("context_length"))
    if direct is not None and direct > 0:
        return direct, "workload.context_length"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for field in ("context_length", "context_token_estimate", "estimated_prompt_tokens", "estimated_context_tokens"):
        value = as_int_or_none(metadata.get(field))
        if value is not None and value > 0:
            return value, f"workload.metadata.{field}"
    return 0, "missing"


def stream_chat_completion(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> Iterator[dict[str, Any]]:
    payload = dict(payload)
    request_id = str(payload.pop("_astrakv_request_id", "") or "")
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if request_id:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
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


def summarize_results(
    results: list[RequestResult],
    metric_summaries: dict[str, DgxSummary] | None = None,
) -> list[dict[str, Any]]:
    metric_summaries = metric_summaries or {}
    grouped: dict[str, list[RequestResult]] = {}
    for result in results:
        grouped.setdefault(result.case, []).append(result)

    rows: list[dict[str, Any]] = []
    for case, items in sorted(grouped.items()):
        ok_items = [item for item in items if item.status == "ok"]
        source = ok_items if ok_items else items
        batch_size = source[0].batch_size
        context_length = source[0].context_length
        output_tokens = source[0].output_tokens_target
        observed_total = sum(item.output_tokens_observed for item in ok_items)
        total_seconds = sum(item.latency_ms for item in ok_items) / 1000.0
        cpu_peak = max((item.cpu_memory_mb_after for item in source), default=0.0)
        metrics = metric_summaries.get(case)
        if metrics is not None and metrics.cpu_rss_peak_mb > 0:
            cpu_peak = max(cpu_peak, metrics.cpu_rss_peak_mb)
        gpu_values = [
            value
            for item in source
            for value in (item.gpu_memory_mb_before, item.gpu_memory_mb_after)
            if value is not None
        ]
        if metrics is not None and metrics.gpu_used_peak_mb is not None:
            gpu_values.append(metrics.gpu_used_peak_mb)
        gpu_peak = max(gpu_values) if gpu_values else None
        rows.append(
            {
                "case": case,
                "backend": source[0].backend,
                "model": source[0].model,
                "batch_size": batch_size,
                "context_length": context_length,
                "output_tokens": output_tokens,
                "request_count": len(items),
                "success_count": len(ok_items),
                "ttft_ms": mean([item.ttft_ms for item in ok_items]),
                "ttft_p50_ms": percentile([item.ttft_ms for item in ok_items], 50),
                "ttft_p95_ms": percentile([item.ttft_ms for item in ok_items], 95),
                "tpot_ms": mean([item.tpot_ms for item in ok_items]),
                "tpot_p50_ms": percentile([item.tpot_ms for item in ok_items], 50),
                "tpot_p95_ms": percentile([item.tpot_ms for item in ok_items], 95),
                "latency_ms": mean([item.latency_ms for item in ok_items]),
                "latency_p50_ms": percentile([item.latency_ms for item in ok_items], 50),
                "latency_p95_ms": percentile([item.latency_ms for item in ok_items], 95),
                "throughput_tokens_s": observed_total / max(total_seconds, 1e-9),
                "process_rss_peak_mb": cpu_peak,
                "cpu_memory_peak_mb": cpu_peak,
                "gpu_memory_peak_mb": gpu_peak if gpu_peak is not None else "",
                "gpu_util_peak_pct": ""
                if metrics is None or metrics.gpu_util_peak_pct is None
                else metrics.gpu_util_peak_pct,
                "disk_read_delta_mb": ""
                if metrics is None or metrics.disk_read_delta_mb is None
                else metrics.disk_read_delta_mb,
                "disk_write_delta_mb": ""
                if metrics is None or metrics.disk_write_delta_mb is None
                else metrics.disk_write_delta_mb,
                "sample_count": 0 if metrics is None else metrics.sample_count,
                "gpu_probe": next((item.gpu_probe for item in source if item.gpu_probe != "unavailable"), "unavailable"),
                "disk_probe": "unavailable" if metrics is None else metrics.disk_probe,
                "status": "ok" if len(ok_items) == len(items) else "error",
                "errors": " | ".join(sorted({item.error for item in items if item.error})),
            }
        )
    return rows


def write_jsonl(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            record = {"schema": "astrakv-benchmark-request-v2", **result.__dict__}
            record["sample_id"] = result.sample_id or str(result.request_id)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_experiment_manifest(output_dir: Path, args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Write the final, content-addressed v2 manifest after all run outputs exist."""
    workload = output_dir / "workload_source.jsonl"
    source_workload = Path(str(getattr(args, "workload_jsonl", "") or ""))
    if source_workload.is_file():
        shutil.copyfile(source_workload, workload)
    else:
        request_rows = _read_jsonl_records(output_dir / "request_results.jsonl")
        workload.write_text(
            "".join(json.dumps({"case": row.get("case", ""), "sample_id": row.get("sample_id") or row.get("request_id", ""), "request_id": row.get("request_id", "")}, ensure_ascii=False) + "\n" for row in request_rows),
            encoding="utf-8",
        )
    matrix = output_dir / "benchmark_matrix.json"
    matrix.write_text(json.dumps({
        "context_lengths": list(args.context_lengths), "batch_sizes": list(args.batch_sizes), "output_tokens": args.output_tokens,
        "repeat": args.repeat, "temperature": args.temperature, "top_p": args.top_p,
    }, sort_keys=True, indent=2), encoding="utf-8")
    environment = output_dir / "environment_source.json"
    environment.write_text(json.dumps({
        "backend": args.backend, "base_url": args.base_url, "connector_version": args.connector_version,
        "software": ExperimentManifest(run_id="environment").to_record()["software"],
        "gpu": ExperimentManifest(run_id="environment").to_record()["gpu"],
        "config_sha256": file_sha256(args.config),
    }, sort_keys=True, indent=2), encoding="utf-8")
    # Runtime mode and request-specific secrets must differ between pair
    # members.  Keep them in the full environment artifact above, but derive a
    # separate immutable control fingerprint for paired comparisons.
    control_environment = output_dir / "control_environment_source.json"
    control_environment.write_text(json.dumps({
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "random_seed": args.random_seed,
        "cache_state": args.cache_state,
        "connector_version": args.connector_version,
        "base_url": args.base_url,
        "max_model_len": os.environ.get("ASTRAKV_MAX_MODEL_LEN", ""),
        "gpu_memory_utilization": os.environ.get("ASTRAKV_GPU_MEMORY_UTILIZATION", ""),
        "prefix_caching": os.environ.get("ASTRAKV_PREFIX_CACHING", ""),
        "kv_transfer_config": os.environ.get("ASTRAKV_KV_TRANSFER_CONFIG", ""),
        "lmcache_config_sha256": file_sha256(os.environ.get("LMCACHE_CONFIG_FILE", "")),
        "software": ExperimentManifest(run_id="control-environment").to_record()["software"],
        "gpu": ExperimentManifest(run_id="control-environment").to_record()["gpu"],
        "config_sha256": file_sha256(args.config),
    }, sort_keys=True, indent=2), encoding="utf-8")
    quality = output_dir / "quality_results.csv"
    _write_quality_provenance(quality, output_dir / "request_results.jsonl")
    artifact_paths = {
        "workload": workload.name, "matrix": matrix.name, "environment": environment.name,
        "control_environment": control_environment.name,
        "benchmark": "benchmark_results.csv", "requests": "request_results.jsonl", "quality": quality.name,
    }
    runtime_state_dir = str(getattr(args, "runtime_state_dir", "") or "")
    online_artifacts = tuple(getattr(args, "online_artifact", ()) or ())
    if runtime_state_dir and online_artifacts:
        raise ValueError("runtime-state-dir and online-artifact cannot be combined")
    if runtime_state_dir:
        for role, target in export_online_control_artifacts(runtime_state_dir, output_dir).items():
            artifact_paths[role] = target.name
    else:
        for role, source in _parse_online_artifacts(online_artifacts).items():
            target = output_dir / _canonical_online_filename(role, source)
            shutil.copyfile(source, target)
            artifact_paths[role] = target.name
    write_experiment_manifest(
        output_dir / "experiment_manifest.json",
        ExperimentManifest(
            run_id=args.run_id, workload_id=args.workload_id, workload_path=workload.name,
            workload_sha256=file_sha256(workload), model=args.model, model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision, dtype=args.dtype, quantization=args.quantization,
            random_seed=args.random_seed, cache_state=args.cache_state, command=redacted_command(sys.argv),
            connector_version=args.connector_version, input_hashes=input_hashes((args.workload_jsonl, args.config)),
            pair_id=str(getattr(args, "pair_id", "") or ""), pair_role=str(getattr(args, "pair_role", "") or ""),
            matrix_sha256=file_sha256(matrix), environment_sha256=file_sha256(environment),
            control_environment_sha256=file_sha256(control_environment), artifact_paths=artifact_paths,
            claim_scope=str(getattr(args, "claim_scope", "benchmark") or "benchmark"),
        ),
    )


def _parse_online_artifacts(values: tuple[str, ...]) -> dict[str, Path]:
    aliases = {
        "bindings": "backend_binding_events",
        "events": "runtime_events_raw",
        "commands": "astrakv_runtime_commands",
        "receipts": "runtime_command_receipts",
        "preflight": "backend_capabilities",
    }
    allowed = {
        "backend_capabilities",
        "backend_binding_events",
        "runtime_events_raw",
        "astrakv_runtime_commands",
        "runtime_command_receipts",
        "runtime_structured_events",
        "online_profile_checkpoint",
        "trace",
        *aliases,
    }
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("online artifact must use role=path")
        role, raw_path = value.split("=", 1)
        source = Path(raw_path)
        canonical_role = aliases.get(role, role)
        if role not in allowed or canonical_role in result or not source.is_file():
            raise ValueError("online artifact role/path is invalid")
        result[canonical_role] = source
    return result


def _canonical_online_filename(role: str, source: Path) -> str:
    names = {
        "backend_capabilities": "backend_capabilities.json",
        "backend_binding_events": "backend_binding_events.jsonl",
        "runtime_events_raw": "runtime_events_raw.jsonl",
        "astrakv_runtime_commands": "astrakv_runtime_commands.jsonl",
        "runtime_command_receipts": "runtime_command_receipts.jsonl",
        "runtime_structured_events": "runtime_structured_events.jsonl",
        "online_profile_checkpoint": "online_profile_checkpoint.json",
        "trace": "trace_events.jsonl",
    }
    try:
        return names[role]
    except KeyError as exc:
        raise ValueError(f"unknown online artifact role: {role}") from exc


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def _write_quality_provenance(path: Path, request_path: Path) -> None:
    rows = _read_jsonl_records(request_path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "request_id", "output_sha256", "status"])
        writer.writeheader()
        for row in rows:
            sample_id = str(row.get("sample_id") or row.get("request_id") or "")
            writer.writerow({
                "sample_id": sample_id, "request_id": row.get("request_id", ""),
                "output_sha256": stable_hash(str(row.get("output_text") or "")), "status": row.get("status", ""),
            })


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [
            {
                "case": "",
                "backend": "vllm_openai_endpoint",
                "model": "",
                "batch_size": "",
                "context_length": "",
                "output_tokens": "",
                "request_count": 0,
                "success_count": 0,
                "ttft_ms": "",
                "ttft_p50_ms": "",
                "ttft_p95_ms": "",
                "tpot_ms": "",
                "tpot_p50_ms": "",
                "tpot_p95_ms": "",
                "latency_ms": "",
                "latency_p50_ms": "",
                "latency_p95_ms": "",
                "throughput_tokens_s": "",
                "process_rss_peak_mb": "",
                "cpu_memory_peak_mb": "",
                "gpu_memory_peak_mb": "",
                "gpu_util_peak_pct": "",
                "disk_read_delta_mb": "",
                "disk_write_delta_mb": "",
                "sample_count": 0,
                "gpu_probe": "unavailable",
                "disk_probe": "unavailable",
                "status": "error",
                "errors": "No benchmark results were produced.",
            }
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    results: list[RequestResult],
    failures: list[str],
) -> None:
    total_requests = len(results)
    total_success = sum(1 for item in results if item.status == "ok")
    success_rate = (total_success / total_requests * 100.0) if total_requests else 0.0
    lines = [
        "# Real Endpoint Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Run id: `{args.run_id}`",
        f"- Backend: `{args.backend}`",
        f"- Model: `{args.model}`",
        f"- Endpoint: `{args.base_url.rstrip('/')}/v1`",
        f"- Context lengths: `{', '.join(str(item) for item in args.context_lengths)}`",
        f"- Batch sizes: `{', '.join(str(item) for item in args.batch_sizes)}`",
        f"- Output tokens: `{args.output_tokens}`",
        f"- Repeat: `{args.repeat}`",
        f"- Workload manifest: `{args.workload_jsonl or 'synthetic prompt matrix'}`",
        f"- Continuous samples: `{args.enable_samples}`",
        f"- Total requests: `{total_requests}`",
        f"- Successful requests: `{total_success}`",
        f"- Success rate: `{success_rate:.2f}%`",
        "",
        "## Summary",
        "",
        "| case | status | requests | success | TTFT ms | TPOT ms | latency p95 ms | throughput tok/s | process RSS MB | GPU util % | disk read MB | disk write MB | samples | memory probe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        gpu_util = row["gpu_util_peak_pct"] if row["gpu_util_peak_pct"] != "" else "n/a"
        disk_read = row["disk_read_delta_mb"] if row["disk_read_delta_mb"] != "" else "n/a"
        disk_write = row["disk_write_delta_mb"] if row["disk_write_delta_mb"] != "" else "n/a"
        rss_peak = row.get("process_rss_peak_mb", row.get("cpu_memory_peak_mb", 0.0))
        rss_display = f"{float(rss_peak):.3f}" if rss_peak != "" else "n/a"
        lines.append(
            "| {case} | {status} | {request_count} | {success_count} | {ttft_ms} | "
            "{tpot_ms} | {latency_p95_ms} | {throughput_tokens_s:.3f} | "
            f"{rss_display} | "
            f"{gpu_util} | {disk_read} | {disk_write} | "
            "{{sample_count}} | {{gpu_probe}} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Continuous Samples",
            "",
        ]
    )
    if args.enable_samples:
        lines.extend(
            [
                "Each benchmark case should produce one continuous sampling file under `samples/`.",
                "",
                "| case | sample file | samples | GPU probe | disk probe | GPU util peak % | disk read MB | disk write MB |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            case = row["case"]
            sample_file = f"samples/{case}_samples.csv" if case else "samples/<case>_samples.csv"
            gpu_util = row["gpu_util_peak_pct"] if row["gpu_util_peak_pct"] != "" else "n/a"
            disk_read = row["disk_read_delta_mb"] if row["disk_read_delta_mb"] != "" else "n/a"
            disk_write = row["disk_write_delta_mb"] if row["disk_write_delta_mb"] != "" else "n/a"
            lines.append(
                "| {case} | `{sample_file}` | {sample_count} | {gpu_probe} | "
                "{disk_probe} | "
                f"{gpu_util} | {disk_read} | {disk_write} |".format(
                    **row,
                    sample_file=sample_file,
                )
            )
        lines.extend(
            [
                "",
                "Validation rule: for official real runs, every non-empty case above should have a matching sample CSV and `sample_count > 0`.",
            ]
        )
    else:
        lines.extend(
            [
                "Continuous sampling was disabled for this run.",
                "",
                "Enable it with `--enable-samples` or by using a config file with a `metrics` section.",
            ]
        )

    lines.extend(
        [
            "",
            "## Endpoint Checks",
            "",
            "- `GET /v1/models`: run separately before this script or verify with `curl http://127.0.0.1:8000/v1/models`.",
            "- `POST /v1/chat/completions`: measured by this script using streaming responses.",
            "",
            "## Notes",
            "",
            "- TTFT is measured from request start to the first streamed content delta.",
            "- TPOT is measured from first streamed token to request completion, divided by generated tokens after the first token.",
            "- Process RSS, GPU utilization, and disk IO are the stable case-level resource metrics.",
            "- `gpu_memory_peak_mb` remains a compatibility field only. On DGX Spark unified-memory systems, `nvidia-smi`/NVML may not expose per-GPU memory counters.",
            "- Continuous samples are real host/GPU/disk probes. KV hit, prefetch hit, and offload events remain blank until a real LMCache/vLLM event adapter is added.",
            "- Canonical workload runs use the positive context_length declared by the workload or its tokenizer profile metadata; synthetic matrix runs use the generated prompt target.",
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        for failure in sorted(set(failures)):
            lines.append(f"- {failure}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `benchmark_results.csv`",
            "- `benchmark_config.json`",
            "- `request_results.jsonl`",
            "- `benchmark_report.md`",
            "- `samples/<case>_samples.csv` when continuous sampling is enabled",
            "",
            f"Total request rows: `{len(results)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def mean(values: list[float | None]) -> float | str:
    real_values = [float(value) for value in values if value is not None]
    if not real_values:
        return ""
    return sum(real_values) / len(real_values)


def percentile(values: list[float | None], pct: float) -> float | str:
    real_values = sorted(float(value) for value in values if value is not None)
    if not real_values:
        return ""
    if len(real_values) == 1:
        return real_values[0]
    rank = (len(real_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(real_values) - 1)
    weight = rank - lower
    return real_values[lower] * (1.0 - weight) + real_values[upper] * weight


def cpu_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def gpu_memory_mb() -> tuple[float | None, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        value, probe = nvml_gpu_memory_mb()
        return value, probe
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            # Some platforms (e.g. DGX Spark / Tegra) return "[N/A]" for
            # certain queries like memory.used — skip those gracefully.
            continue
    if not values:
        value, probe = nvml_gpu_memory_mb()
        if value is not None:
            return value, probe
        return None, f"nvidia-smi-empty/{probe}"
    return sum(values), "nvidia-smi"


def nvml_gpu_memory_mb() -> tuple[float | None, str]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None, "nvml-unavailable"

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        values: list[float] = []
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                values.append(float(memory.used) / (1024.0 * 1024.0))
            except Exception:
                continue
        if values:
            return sum(values), "nvml"
        return None, "nvml-empty"
    except Exception:
        return None, "nvml-unavailable"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def classify_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        return f"Endpoint HTTP {exc.code}: {detail}"
    if isinstance(exc, urllib.error.URLError):
        return f"Endpoint connection error: {exc.reason}"
    return f"Benchmark script/runtime error: {type(exc).__name__}: {exc}"


def start_metrics_collector(
    args: argparse.Namespace,
    output_dir: Path,
    case: str,
) -> DgxMetricsCollector | None:
    if not args.enable_samples:
        return None
    collector = DgxMetricsCollector(
        output_csv=output_dir / "samples" / f"{case}_samples.csv",
        interval_seconds=args.metrics_interval,
        disk_device=args.disk_device,
        process_name_filters=args.process_name_filters,
        run_id=args.run_id,
        case=case,
    )
    collector.start()
    return collector


def write_effective_config(path: Path, args: argparse.Namespace, config: dict[str, Any]) -> None:
    payload = {
        "source_config": config.get("_config_path", ""),
        "run_name": args.run_name,
        "backend": args.backend,
        "base_url": args.base_url,
        "model": args.model,
        "context_lengths": args.context_lengths,
        "batch_sizes": args.batch_sizes,
        "output_tokens": args.output_tokens,
        "repeat": args.repeat,
        "timeout": args.timeout,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_token_scale": args.prompt_token_scale,
        "enable_samples": args.enable_samples,
        "metrics_interval": args.metrics_interval,
        "disk_device": args.disk_device,
        "process_name_filters": list(args.process_name_filters),
        "workload_jsonl": args.workload_jsonl,
        "run_id": args.run_id,
        "request_context_url": args.request_context_url,
        "request_context_artifact": "request_context.jsonl",
        "samples_dir": "samples" if args.enable_samples else "",
        "samples_file_pattern": "samples/<case>_samples.csv" if args.enable_samples else "",
        "raw_config": redact_config({key: value for key, value in config.items() if key != "_config_path"}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def redacted_command(arguments: list[str]) -> str:
    """Render a reproducible command without retaining credential values."""
    return redact_command_text(shlex.join(arguments))


def redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_config_key(str(key)) else redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_CONFIG_KEY_PARTS) or normalized.endswith("_token")


def write_charts(csv_path: Path, charts_dir: Path) -> list[Path]:
    try:
        return plot_csv(csv_path, charts_dir)
    except Exception as exc:  # noqa: BLE001 - charts must not hide benchmark results.
        print(f"Chart generation skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[:-3].rstrip("/")
    return cleaned


def _reserve_batch_nonce(batch_nonce: str | None) -> str:
    if batch_nonce is None:
        nonce = str(uuid4())
    else:
        try:
            nonce = str(UUID(batch_nonce))
        except (TypeError, ValueError) as exc:
            raise ValueError("batch_nonce must be a UUID") from exc
        if nonce != batch_nonce.lower():
            raise ValueError("batch_nonce must use canonical UUID form")
    with _BATCH_NONCE_LOCK:
        if nonce in _USED_BATCH_NONCES:
            raise ValueError("batch_nonce was already used")
        _USED_BATCH_NONCES.add(nonce)
    return nonce


def _canonical_request_nonce(request_nonce: str | None) -> str:
    if request_nonce is None:
        return str(uuid4())
    try:
        nonce = str(UUID(request_nonce))
    except (TypeError, ValueError) as exc:
        raise ValueError("request_nonce must be a UUID") from exc
    if nonce != request_nonce.lower():
        raise ValueError("request_nonce must use canonical UUID form")
    return nonce


def _reserve_request_contexts(
    run_id: str,
    request_ids: tuple[str, ...],
    request_nonces: tuple[str, ...],
) -> None:
    identities = {(str(run_id), request_id, request_nonce) for request_id, request_nonce in zip(request_ids, request_nonces)}
    with _BATCH_NONCE_LOCK:
        if any(identity in _DISPATCHED_REQUEST_CONTEXTS for identity in identities) or any(
            request_nonce in _USED_REQUEST_NONCES for request_nonce in request_nonces
        ):
            raise ValueError("duplicate (run_id, request_id, request_nonce) before dispatch")
        _DISPATCHED_REQUEST_CONTEXTS.update(identities)
        _USED_REQUEST_NONCES.update(request_nonces)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_int_or_none(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def as_float_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _arrival_sort_key(row: dict[str, Any], fallback: int) -> tuple[int, int]:
    arrival_index = as_int_or_none(row.get("arrival_index"))
    return (0, arrival_index) if arrival_index is not None else (1, fallback)


if __name__ == "__main__":
    raise SystemExit(main())
