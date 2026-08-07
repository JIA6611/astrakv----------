#!/usr/bin/env python3
"""Build a teacher-forced, compatibility-aware Qwen3-8B KV-Core profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.runtime.kv_runtime_core import KVCompatibilityKey, exact_token_prefix_hash
from astrakv.runtime.offline_kv_profile import OfflinePrefixProfile, build_manifest, validate_qwen3_8b_target


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid workload JSON at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"workload row {number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("workload is empty")
    return rows


def _metrics(reference, candidate) -> tuple[float, float, float, float]:
    """CKA, cosine, and max-abs between two same-shape hidden-state tensors."""
    import torch
    x, y = reference.float().reshape(-1, reference.shape[-1]), candidate.float().reshape(-1, candidate.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = torch.sum((x.T @ y) ** 2)
    norm = torch.sqrt(torch.sum((x.T @ x) ** 2) * torch.sum((y.T @ y) ** 2))
    cka = float((hsic / norm.clamp_min(1e-12)).item())
    cosine = float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item())
    l2 = float((x - y).norm().item())
    max_abs = float((x - y).abs().max().item())
    return cka, cosine, l2, max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template-revision", default="qwen3-default")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--engine-id", default="offline-profile")
    parser.add_argument("--worker-id", default="worker-0")
    parser.add_argument("--native-key-namespace", default="qwen3-8b-profile")
    parser.add_argument("--hardware", default="DGX Spark GB10")
    parser.add_argument("--quantization", default="unquantized")
    args = parser.parse_args()
    validate_qwen3_8b_target(model_id="Qwen3-8B", quantization=args.quantization)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("build_qwen3_kv_core_profile requires torch and transformers") from exc
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
    rows = jsonl(Path(args.workload_jsonl))
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    profiles: list[OfflinePrefixProfile] = []
    for index, row in enumerate(rows):
        prompt, continuation = str(row.get("prompt") or ""), str(row.get("continuation") or row.get("ground_truth") or "")
        if not prompt or not continuation:
            raise ValueError("every profile workload row requires prompt and teacher-forced continuation/ground_truth")
        prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
        input_ids = torch.tensor([prefix_ids + continuation_ids], device=args.device)
        labels = input_ids.clone()
        labels[:, :len(prefix_ids)] = -100
        with torch.inference_mode():
            baseline = model(input_ids=input_ids, labels=labels, output_hidden_states=True, use_cache=False)
            repeat = model(input_ids=input_ids, labels=labels, output_hidden_states=True, use_cache=False)
        loss = float(baseline.loss.item())
        layer_activity: dict[str, float] = {}
        cka: dict[str, float] = {}
        cosine: dict[str, float] = {}
        l2: dict[str, float] = {}
        max_abs: dict[str, float] = {}
        for layer, (left, right) in enumerate(zip(baseline.hidden_states or (), repeat.hidden_states or ())):
            name = str(layer)
            layer_activity[name] = float(left.float().norm().item())
            cka[name], cosine[name], l2[name], max_abs[name] = _metrics(left, right)
        key = KVCompatibilityKey(
            model_id="Qwen3-8B", model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
            chat_template_revision=args.chat_template_revision, dtype=args.dtype, rope_config=json.dumps(getattr(model.config, "rope_scaling", None) or {}, sort_keys=True),
            adapter_namespace="base", kv_layout="vllm-paged-kv-v1", block_size_tokens=16, chunk_size_tokens=256,
            layer_group="all-layers", prefix_hash=exact_token_prefix_hash(prefix_ids), engine_id=args.engine_id, worker_id=args.worker_id,
        )
        profiles.append(OfflinePrefixProfile(
            sample_id=str(row.get("sample_id") or index), compatibility_key=key,
            native_key=f"{args.native_key_namespace}:{key.identity}", prefix_tokens=len(prefix_ids),
            teacher_forced_loss=loss, teacher_forced_ppl=math.exp(min(loss, 80.0)),
            layer_activity_l2=layer_activity, layer_cka=cka, layer_cosine=cosine, layer_l2=l2, layer_max_abs=max_abs,
        ))
    workload = Path(args.workload_jsonl)
    manifest = build_manifest(
        target={"model_id": "Qwen3-8B", "model_revision": args.model_revision, "tokenizer_revision": args.tokenizer_revision, "quantization": args.quantization, "dtype": args.dtype},
        hardware={"platform": args.hardware, "memory_model": "uma"}, workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(), profiles=profiles,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
