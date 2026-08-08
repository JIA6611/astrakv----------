#!/usr/bin/env python3
"""Build a teacher-forced, compatibility-aware Qwen3-8B KV-Core profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def _continuation_loss(logits, continuation_ids) -> float:
    import torch
    labels = torch.tensor([continuation_ids], device=logits.device)
    return float(torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
    ).item())


def _run_teacher_forced_path(
    model,
    prefix_ids,
    continuation_ids,
    *,
    cached_prefix_tokens: int | None,
    collect_hidden: bool = True,
):
    """Run full recompute or cache-prefix-plus-native-recompute continuation."""
    import torch
    device = next(model.parameters()).device
    prefix_len, continuation_len = len(prefix_ids), len(continuation_ids)
    if cached_prefix_tokens is None:
        full = torch.tensor([prefix_ids + continuation_ids], device=device)
        output = model(
            input_ids=full, output_hidden_states=collect_hidden, use_cache=False,
        )
        logits = output.logits[
            :, prefix_len - 1 : prefix_len - 1 + continuation_len
        ].detach().clone()
        hidden = tuple(
            state[:, prefix_len : prefix_len + continuation_len].detach().clone()
            for state in output.hidden_states or ()
        )
        return logits, hidden, _continuation_loss(logits, continuation_ids)

    cached = max(0, min(prefix_len, int(cached_prefix_tokens)))
    if cached == 0:
        return _run_teacher_forced_path(
            model, prefix_ids, continuation_ids,
            cached_prefix_tokens=None, collect_hidden=collect_hidden,
        )
    cached_input = torch.tensor([prefix_ids[:cached]], device=device)
    prefix_output = model(input_ids=cached_input, output_hidden_states=False, use_cache=True)
    suffix_tokens = prefix_ids[cached:] + continuation_ids
    suffix_input = torch.tensor([suffix_tokens], device=device)
    suffix_output = model(
        input_ids=suffix_input, past_key_values=prefix_output.past_key_values,
        output_hidden_states=collect_hidden, use_cache=False,
    )
    recomputed_prefix = prefix_len - cached
    if recomputed_prefix == 0:
        logits = torch.cat(
            [prefix_output.logits[:, -1:], suffix_output.logits[:, : max(0, continuation_len - 1)]],
            dim=1,
        )
    else:
        start = recomputed_prefix - 1
        logits = suffix_output.logits[
            :, start : start + continuation_len
        ].detach().clone()
    hidden = tuple(
        state[:, recomputed_prefix : recomputed_prefix + continuation_len].detach().clone()
        for state in suffix_output.hidden_states or ()
    )
    return logits, hidden, _continuation_loss(logits, continuation_ids)


def _native_lmcache_key(prefix_ids: list[int], *, model_name: str, chunk_size: int, dtype, hash_algorithm: str) -> str:
    """Generate the same ordered CacheEngineKey list used by LMCache 0.4.7."""
    try:
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache.v1.token_database import ChunkedTokenDatabase
    except ImportError as exc:
        raise RuntimeError("LMCache 0.4.7 is required to generate native offline keys") from exc
    config = LMCacheEngineConfig.from_defaults()
    config.chunk_size = int(chunk_size)
    config.pre_caching_hash_algorithm = hash_algorithm
    if hash_algorithm == "builtin" and os.environ.get("PYTHONHASHSEED") is None:
        raise RuntimeError("PYTHONHASHSEED must be fixed when LMCache uses builtin native-key hashing")
    metadata = LMCacheMetadata(
        model_name=model_name, world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=dtype,
        kv_shape=(1, 2, int(chunk_size), 1, 1), role="offline-profile",
        served_model_name="Qwen3-8B", chunk_size=int(chunk_size),
    )
    database = ChunkedTokenDatabase(config, metadata)
    keys = [key.to_string() for _start, _end, key in database.process_tokens(tokens=prefix_ids)]
    if not keys:
        raise ValueError("LMCache token database produced no native keys")
    return json.dumps(keys, separators=(",", ":"), ensure_ascii=True)


def _perturb_layer_cache(past_key_values, layer: int):
    import torch
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
    else:
        legacy = tuple(past_key_values)
    result = []
    for index, values in enumerate(legacy):
        if index == layer:
            result.append(tuple(torch.zeros_like(value) for value in values))
        else:
            result.append(tuple(value for value in values))
    legacy_result = tuple(result)
    factory = getattr(type(past_key_values), "from_legacy_cache", None)
    if callable(factory):
        return factory(legacy_result)
    # Transformers 5.x removed from_legacy_cache but DynamicCache accepts the
    # per-layer tensor tuples as its first constructor argument.  Preserve the
    # native cache type because Qwen3 no longer accepts a raw legacy tuple.
    try:
        return type(past_key_values)(legacy_result)
    except (TypeError, ValueError):
        return legacy_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template-revision", default="qwen3-default")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--lmcache-model-name", default="",
        help="Exact LMCache metadata model_name; defaults to the --model path used by vLLM.",
    )
    parser.add_argument("--lmcache-hash-algorithm", default="builtin")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--correctness-atol", type=float, default=1e-3)
    parser.add_argument("--sensitivity-max-layers", type=int, default=0)
    parser.add_argument("--hardware", default="DGX Spark GB10")
    parser.add_argument("--quantization", default="unquantized")
    args = parser.parse_args()
    if not args.lmcache_model_name:
        args.lmcache_model_name = str(Path(args.model).resolve())
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
        messages = row.get("messages")
        if not isinstance(messages, list):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            messages = metadata.get("messages")
        if isinstance(messages, list) and messages:
            prefix_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
        if not continuation_ids:
            raise ValueError("teacher-forced continuation produced no tokens")
        partial_tokens = (len(prefix_ids) // 2 // args.chunk_size) * args.chunk_size
        partial_tokens = min(partial_tokens, max(0, len(prefix_ids) - args.chunk_size))
        with torch.inference_mode():
            baseline_logits, baseline_hidden, loss = _run_teacher_forced_path(
                model, prefix_ids, continuation_ids, cached_prefix_tokens=None,
            )
            exact_logits, _exact_hidden, exact_loss = _run_teacher_forced_path(
                model, prefix_ids, continuation_ids,
                cached_prefix_tokens=len(prefix_ids), collect_hidden=False,
            )
            partial_logits, partial_hidden, partial_loss = _run_teacher_forced_path(
                model, prefix_ids, continuation_ids, cached_prefix_tokens=partial_tokens,
            )
        exact_error = float((baseline_logits.float() - exact_logits.float()).abs().max().item())
        partial_error = float((baseline_logits.float() - partial_logits.float()).abs().max().item())
        correctness = (
            exact_error <= args.correctness_atol
            and partial_error <= args.correctness_atol
            and abs(loss - exact_loss) <= args.correctness_atol
            and abs(loss - partial_loss) <= args.correctness_atol
        )
        layer_activity: dict[str, float] = {}
        cka: dict[str, float] = {}
        cosine: dict[str, float] = {}
        l2: dict[str, float] = {}
        max_abs: dict[str, float] = {}
        for layer, (left, right) in enumerate(zip(baseline_hidden, partial_hidden)):
            name = str(layer)
            layer_activity[name] = float(left.float().norm().item())
            cka[name], cosine[name], l2[name], max_abs[name] = _metrics(left, right)
        layer_sensitivity: dict[str, dict[str, float]] = {}
        prefix_tensor = torch.tensor([prefix_ids], device=args.device)
        continuation_tensor = torch.tensor([continuation_ids], device=args.device)
        with torch.inference_mode():
            prefix_output = model(input_ids=prefix_tensor, use_cache=True)
            layer_count = len(prefix_output.past_key_values)
            if args.sensitivity_max_layers > 0:
                layer_count = min(layer_count, args.sensitivity_max_layers)
            for layer in range(layer_count):
                perturbed = _perturb_layer_cache(prefix_output.past_key_values, layer)
                candidate = model(input_ids=continuation_tensor, past_key_values=perturbed, use_cache=False)
                candidate_logits = torch.cat(
                    [prefix_output.logits[:, -1:], candidate.logits[:, : max(0, len(continuation_ids) - 1)]],
                    dim=1,
                )
                candidate_loss = _continuation_loss(candidate_logits, continuation_ids)
                layer_sensitivity[str(layer)] = {
                    "teacher_forced_loss_delta": candidate_loss - loss,
                    "logit_max_abs": float((baseline_logits.float() - candidate_logits.float()).abs().max().item()),
                }
        key = KVCompatibilityKey(
            model_id="Qwen3-8B", model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
            chat_template_revision=args.chat_template_revision, dtype=args.dtype, rope_config=json.dumps(getattr(model.config, "rope_scaling", None) or {}, sort_keys=True),
            adapter_namespace="base", kv_layout="vllm-paged-kv-v1", block_size_tokens=args.block_size, chunk_size_tokens=args.chunk_size,
            layer_group="all-kv-layers", prefix_hash=exact_token_prefix_hash(prefix_ids),
        )
        profiles.append(OfflinePrefixProfile(
            sample_id=str(row.get("sample_id") or index), compatibility_key=key,
            native_key=_native_lmcache_key(prefix_ids, model_name=args.lmcache_model_name, chunk_size=args.chunk_size, dtype=dtype, hash_algorithm=args.lmcache_hash_algorithm), prefix_tokens=len(prefix_ids),
            teacher_forced_loss=loss, teacher_forced_ppl=math.exp(min(loss, 80.0)),
            layer_activity_l2=layer_activity, layer_cka=cka, layer_cosine=cosine, layer_l2=l2, layer_max_abs=max_abs,
            path_quality={
                "exact_reuse": {"teacher_forced_loss": exact_loss, "logit_max_abs": exact_error},
                "partial_load_recompute": {"teacher_forced_loss": partial_loss, "logit_max_abs": partial_error, "loaded_prefix_tokens": float(partial_tokens)},
                "ssd_miss_recompute": {"teacher_forced_loss": loss, "logit_max_abs": 0.0},
            },
            layer_sensitivity=layer_sensitivity,
            correctness_passed=correctness,
        ))
    workload = Path(args.workload_jsonl)
    manifest = build_manifest(
        target={"model_id": "Qwen3-8B", "model_revision": args.model_revision, "tokenizer_revision": args.tokenizer_revision, "chat_template_revision": args.chat_template_revision, "quantization": args.quantization, "dtype": args.dtype, "block_size_tokens": args.block_size, "chunk_size_tokens": args.chunk_size, "lmcache_model_name": args.lmcache_model_name, "lmcache_hash_algorithm": args.lmcache_hash_algorithm, "workload_id": args.workload_id or workload.stem},
        hardware={"platform": args.hardware, "memory_model": "uma", "device": args.device, "torch_dtype": args.dtype}, workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(), profiles=profiles,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
