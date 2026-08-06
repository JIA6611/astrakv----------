"""Run a small MoE forward pass and export router expert-route events.

This script is intended for offline/local MoE checkpoints. It does not start a
serving runtime, move expert weights between tiers, or claim expert prefetch.
It only records router decisions exposed by Hugging Face model outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_PROMPTS = [
    "Explain why memory-constrained LLM inference needs KV cache management.",
    "Summarize how expert routing in MoE models affects memory traffic.",
    "Describe a virtual memory analogy for loading model objects on demand.",
]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)
    if not prompts:
        raise SystemExit("No prompts provided.")

    torch, auto_tokenizer, auto_model = import_hf_deps()
    dtype = resolve_dtype(torch, args.dtype)
    device = resolve_device(torch, args.device)

    tokenizer = auto_tokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = auto_model.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()
    if device != "auto":
        model.to(device)

    top_k = args.top_k or infer_top_k(getattr(model, "config", None))
    all_records: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []

    for index, prompt in enumerate(prompts):
        request_id = f"{args.request_prefix}-{index}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        if device != "auto":
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True, use_cache=False)

        router_logits = extract_router_logits(outputs)
        if not router_logits:
            raise SystemExit(
                "Model output did not expose router_logits. Use a supported MoE model/checkpoint "
                "or enable trust_remote_code if the model implementation requires it."
            )

        input_ids = inputs["input_ids"].detach().cpu().numpy()[0].tolist()
        records = router_records_from_logits(
            router_logits,
            input_ids=input_ids,
            request_id=request_id,
            model_name=args.model,
            top_k=top_k,
        )
        all_records.extend(records)
        prompt_summaries.append(
            {
                "request_id": request_id,
                "prompt_chars": len(prompt),
                "input_tokens": len(input_ids),
                "route_events": len(records),
            }
        )

    events_path = output_dir / args.events_name
    summary_path = output_dir / args.summary_name
    report_path = output_dir / args.report_name
    manifest_path = output_dir / args.manifest_name

    write_jsonl(events_path, all_records)
    summary = summarize_records(all_records)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, args, prompts, prompt_summaries, summary, events_path)
    write_manifest(manifest_path, args, prompt_summaries, summary, events_path, summary_path, report_path)

    print(f"MoE route events written to {events_path}")
    print(f"MoE route summary written to {summary_path}")
    print(f"MoE route report written to {report_path}")
    print(f"MoE route manifest written to {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MoE model directory or HF id.")
    parser.add_argument("--output-dir", default="results/moe_route_trace")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt text. Can be repeated.")
    parser.add_argument("--prompt-file", default="", help="Text file with one prompt per line.")
    parser.add_argument("--events-name", default="moe_route_events.jsonl")
    parser.add_argument("--summary-name", default="moe_route_summary.json")
    parser.add_argument("--report-name", default="moe_route_report.md")
    parser.add_argument("--manifest-name", default="moe_route_manifest.json")
    parser.add_argument("--request-prefix", default="moe-route")
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=0, help="Override experts per token. Defaults to model config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Use local model files only. This is the default; kept as an explicit offline flag.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default the script uses local files only.",
    )
    args = parser.parse_args()
    args.local_files_only = not args.allow_download
    return args


def import_hf_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only without optional deps
        raise SystemExit("Missing optional dependencies. Install with: pip install torch transformers") from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file:
        path = Path(args.prompt_file)
        prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return prompts or list(DEFAULT_PROMPTS)


def resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda but torch.cuda.is_available() is false.")
    return device


def resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return "auto"


def infer_top_k(config: Any) -> int:
    for name in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    return 2


def extract_router_logits(outputs: Any) -> list[Any]:
    value = getattr(outputs, "router_logits", None)
    if value is None and isinstance(outputs, dict):
        value = outputs.get("router_logits")
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def router_records_from_logits(
    router_logits: Iterable[Any],
    *,
    input_ids: list[int],
    request_id: str,
    model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    token_count = len(input_ids)
    for layer_id, logits in enumerate(router_logits):
        arr = to_numpy(logits)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[-1] == 0:
            continue
        if arr.shape[0] == 0:
            continue
        usable_tokens = min(token_count, arr.shape[0])
        for token_index in range(usable_tokens):
            row = arr[token_index]
            probs = softmax(row)
            k = max(1, min(top_k, probs.shape[0]))
            expert_indices = np.argsort(probs)[-k:][::-1]
            records.append(
                {
                    "event_type": "expert_route",
                    "request_id": request_id,
                    "layer_id": layer_id,
                    "token_index": token_index,
                    "token_id": input_ids[token_index],
                    "experts": [int(item) for item in expert_indices.tolist()],
                    "scores": [round(float(probs[item]), 8) for item in expert_indices.tolist()],
                    "top_k": k,
                    "tier": "unknown",
                    "metadata": {
                        "source": "hf_forward_router_logits",
                        "model": model_name,
                    },
                }
            )
    return records


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def softmax(row: np.ndarray) -> np.ndarray:
    row = row.astype(np.float64)
    row = row - np.max(row)
    exp = np.exp(row)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(exp)
    return exp / denom


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    unique_requests = {str(row.get("request_id", "")) for row in records}
    unique_layers = {int(row.get("layer_id", -1)) for row in records}
    unique_experts = {
        str(expert)
        for row in records
        for expert in row.get("experts", [])
    }
    layer_experts = {
        (int(row.get("layer_id", -1)), str(expert))
        for row in records
        for expert in row.get("experts", [])
    }
    return {
        "schema": "astra-moe-route-trace-summary-v1",
        "total_records": len(records),
        "unique_requests": len(unique_requests),
        "unique_layers": len(unique_layers),
        "unique_experts": len(unique_experts),
        "unique_layer_experts": len(layer_experts),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    args: argparse.Namespace,
    prompts: list[str],
    prompt_summaries: list[dict[str, Any]],
    summary: dict[str, Any],
    events_path: Path,
) -> None:
    lines = [
        "# MoE Route Trace Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Run",
        "",
        f"- Model: `{args.model}`",
        f"- Local files only: `{args.local_files_only}`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Prompts: `{len(prompts)}`",
        f"- Events JSONL: `{events_path}`",
        "",
        "## Summary",
        "",
        f"- Total route records: `{summary['total_records']}`",
        f"- Unique requests: `{summary['unique_requests']}`",
        f"- Unique layers: `{summary['unique_layers']}`",
        f"- Unique experts: `{summary['unique_experts']}`",
        f"- Unique layer-expert pairs: `{summary['unique_layer_experts']}`",
        "",
        "## Prompt Summary",
        "",
        "| request | prompt chars | input tokens | route events |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in prompt_summaries:
        lines.append(
            f"| {item['request_id']} | {item['prompt_chars']} | {item['input_tokens']} | {item['route_events']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This report records router choices exposed by a Hugging Face MoE forward pass.",
            "- It does not prove serving-time expert weight prefetch or runtime tier movement.",
            "- Feed `moe_route_events.jsonl` into `extract_moe_expert_events.py` for the AstraKV-W MoE evidence chain.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    prompt_summaries: list[dict[str, Any]],
    summary: dict[str, Any],
    events_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    manifest = {
        "schema": "astra-moe-route-trace-manifest-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "model": args.model,
            "prompt_count": len(prompt_summaries),
            "local_files_only": args.local_files_only,
            "max_input_tokens": args.max_input_tokens,
        },
        "outputs": {
            "events_jsonl": str(events_path),
            "summary_json": str(summary_path),
            "report": str(report_path),
        },
        "summary": summary,
        "prompt_summary": prompt_summaries,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
