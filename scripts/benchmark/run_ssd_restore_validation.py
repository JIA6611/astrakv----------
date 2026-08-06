"""Run a single-request LMCache SSD restore validation phase.

This runner deliberately does not manage vLLM processes. The caller launches a
specific A/B/C/D server configuration, then this script records request-level
HTTP timing plus phase-scoped LMCache metrics and server-log evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.parse
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_real_benchmark import run_one_request  # noqa: E402


RETRIEVE_REQUEST_METRIC = "lmcache:num_retrieve_requests_total"
HIT_TOKEN_METRIC = "lmcache:num_hit_tokens_total"


def build_reset_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/reset_prefix_cache?reset_external=false"


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for name, value in after.items():
        if not isinstance(value, (int, float)):
            continue
        previous = before.get(name, 0.0)
        if not isinstance(previous, (int, float)):
            previous = 0.0
        delta[name] = float(value) - float(previous)
    return delta


def evaluate_ssd_restore_evidence(
    *,
    vllm_cached_tokens: int | None,
    lmcache_loaded_tokens: int | None,
    retrieve_requests_delta: float | None,
    need_to_load_tokens: int | None,
    disk_read_observed: bool,
    request_status: str,
) -> dict[str, Any]:
    missing: list[str] = []
    if request_status != "ok":
        missing.append("request_not_ok")
    if vllm_cached_tokens is None or vllm_cached_tokens != 0:
        missing.append("vllm_prefix_cache_not_isolated")
    if lmcache_loaded_tokens is None or lmcache_loaded_tokens <= 0:
        missing.append("lmcache_loaded_tokens_not_positive")
    if retrieve_requests_delta is None or retrieve_requests_delta <= 0:
        missing.append("retrieve_requests_not_positive")
    if need_to_load_tokens is None or need_to_load_tokens <= 0:
        missing.append("need_to_load_not_positive")
    if not disk_read_observed:
        missing.append("disk_read_not_observed")
    return {
        "status": "ssd_restore_evidence_complete" if not missing else "insufficient_ssd_restore_evidence",
        "missing": missing,
    }


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        name, value = fields
        base_name, _, labels = name.partition("{")
        if base_name.endswith("_created") or 'role="scheduler"' in labels:
            continue
        try:
            metrics[base_name] = float(value)
        except ValueError:
            continue
    return metrics


def scrape_metrics(url: str) -> dict[str, float]:
    if not url:
        return {}
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - explicit local endpoint.
        return parse_prometheus_metrics(response.read().decode("utf-8", errors="replace"))


def reset_local_prefix_cache(url: str) -> None:
    if not url.endswith("reset_external=false"):
        raise ValueError("restore validation may only reset local prefix cache with reset_external=false")
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - explicit local endpoint.
        if response.status >= 300:
            raise RuntimeError(f"prefix reset failed with HTTP {response.status}")


def exact_prefix_prompt(prefix_tokens: int, epoch: int) -> str:
    prefix = " ".join(["context"] * prefix_tokens)
    return f"{prefix}\n\nQuestion: summarize the shared context.\nEpoch suffix: {epoch}."


def new_log_text(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        return handle.read()


def log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def extract_need_to_load(text: str) -> int | None:
    matches = re.findall(r"need to load:\s*(\d+)", text)
    return int(matches[-1]) if matches else None


def request_record(result: Any) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "endpoint_response_id": result.endpoint_response_id,
        "status": result.status,
        "ttft_ms": result.ttft_ms,
        "tpot_ms": result.tpot_ms,
        "latency_ms": result.latency_ms,
        "output_tokens_observed": result.output_tokens_observed,
        "error": result.error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="vLLM base URL without /v1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", choices=("baseline", "ssd_restore", "gpu_prefix", "combined"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--lmcache-metrics-url", default="")
    parser.add_argument("--reset-prefix-url", default="")
    parser.add_argument("--prefix-tokens", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed-wait-seconds", type=float, default=3.0)
    parser.add_argument("--api-key", default="EMPTY")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "ssd_restore" and not args.reset_prefix_url:
        raise SystemExit("ssd_restore phase requires --reset-prefix-url ending in reset_external=false")
    if args.reset_prefix_url and not args.reset_prefix_url.endswith("reset_external=false"):
        raise SystemExit("--reset-prefix-url must preserve connector state with reset_external=false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = Path(args.server_log)
    rows: list[dict[str, Any]] = []
    for prefix_tokens in args.prefix_tokens:
        for epoch in range(args.epochs):
            prompt = exact_prefix_prompt(prefix_tokens, epoch)
            metrics_before_seed = scrape_metrics(args.lmcache_metrics_url)
            seed_offset = log_offset(server_log)
            seed = run_one_request(
                base_url=args.base_url.rstrip("/"), api_key=args.api_key, model=args.model,
                backend=args.phase, case=f"seed_{prefix_tokens}_{epoch}", request_id=f"seed-{prefix_tokens}-{epoch}",
                batch_size=1, context_length=prefix_tokens, output_tokens=args.output_tokens,
                timeout=600, temperature=0.0, top_p=1.0, system_prompt="", prompt_seed="", prompt_token_scale=1.0,
                prompt=prompt, request_metadata={"cache_state": "seed"},
            )
            time.sleep(args.seed_wait_seconds)
            metrics_after_seed = scrape_metrics(args.lmcache_metrics_url)

            if args.phase == "ssd_restore":
                reset_local_prefix_cache(args.reset_prefix_url)

            replay_offset = log_offset(server_log)
            metrics_before_replay = scrape_metrics(args.lmcache_metrics_url)
            replay = run_one_request(
                base_url=args.base_url.rstrip("/"), api_key=args.api_key, model=args.model,
                backend=args.phase, case=f"replay_{prefix_tokens}_{epoch}", request_id=f"replay-{prefix_tokens}-{epoch}",
                batch_size=1, context_length=prefix_tokens, output_tokens=args.output_tokens,
                timeout=600, temperature=0.0, top_p=1.0, system_prompt="", prompt_seed="", prompt_token_scale=1.0,
                prompt=prompt, request_metadata={"cache_state": "replay"},
            )
            time.sleep(args.seed_wait_seconds)
            metrics_after_replay = scrape_metrics(args.lmcache_metrics_url)
            replay_log = new_log_text(server_log, replay_offset)
            metrics = metric_delta(metrics_before_replay, metrics_after_replay)
            need_to_load = extract_need_to_load(replay_log)
            loaded_tokens = int(metrics.get(HIT_TOKEN_METRIC, 0.0))
            retrieve_delta = metrics.get(RETRIEVE_REQUEST_METRIC, 0.0)
            evidence = evaluate_ssd_restore_evidence(
                vllm_cached_tokens=0 if args.phase in {"baseline", "ssd_restore"} else None,
                lmcache_loaded_tokens=loaded_tokens,
                retrieve_requests_delta=retrieve_delta,
                need_to_load_tokens=need_to_load,
                disk_read_observed="Disk read size:" in replay_log,
                request_status=replay.status,
            )
            rows.append({
                "schema": "astrakv-ssd-restore-validation-v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "phase": args.phase,
                "prefix_tokens": prefix_tokens,
                "epoch": epoch,
                "prompt_hash": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "seed": request_record(seed),
                "replay": request_record(replay),
                "metrics_seed_delta": metric_delta(metrics_before_seed, metrics_after_seed),
                "metrics_replay_delta": metrics,
                "need_to_load_tokens": need_to_load,
                "disk_read_observed": "Disk read size:" in replay_log,
                "server_log_byte_range": [replay_offset, log_offset(server_log)],
                "evidence": evidence,
            })

    output = output_dir / "ssd_restore_rows.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": "astrakv-ssd-restore-manifest-v1", "phase": args.phase,
        "command": " ".join(sys.argv), "rows": len(rows), "server_log": str(server_log),
        "metrics_url": args.lmcache_metrics_url, "reset_prefix_url": args.reset_prefix_url,
        "claim_boundary": "Only rows with ssd_restore_evidence_complete may be used for SSD restore latency attribution.",
    }
    (output_dir / "ssd_restore_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"SSD restore validation rows written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
