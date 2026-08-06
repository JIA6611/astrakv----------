"""Emit tokenizer-backed reuse observations for immutable Task 1 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.task1_qasper_adapter import load_task1_qasper_directory
from astrakv.benchmarks.workflow_observer import (
    load_replay_workflow_rows,
    load_task1_prompt_records,
    observe_workflow_rows,
    task1_prompt_records_to_workflow_rows,
)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    model_path = str(args.model_path or config.get("model_path", ""))
    block_size = int(args.block_size_tokens or config.get("block_size_tokens", 0))
    kv_bytes = int(args.kv_bytes_per_token or config.get("kv_bytes_per_token", 0))
    if not model_path or block_size <= 0 or kv_bytes <= 0:
        raise SystemExit("model_path, block_size_tokens, and kv_bytes_per_token must be explicit positive values")

    if args.task1_dir:
        if not args.workload:
            raise SystemExit("--workload is required with --task1-dir")
        # Validate source manifests before observing the original messages.
        load_task1_qasper_directory(args.task1_dir, args.workload)
        records = load_task1_prompt_records(
            args.task1_dir, workload_type=args.workload, limit=args.limit
        )
        rows = task1_prompt_records_to_workflow_rows(records, workload_type=args.workload)
        source_path = Path(args.task1_dir) / "prompts" / f"qasper_{args.workload}_prompts.jsonl"
    else:
        rows = load_replay_workflow_rows(args.replay_jsonl)
        if args.limit is not None:
            rows = rows[: args.limit]
        source_path = Path(args.replay_jsonl)
    from transformers import AutoTokenizer  # Imported only when real tokenization is requested.

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    observations = observe_workflow_rows(
        rows,
        tokenizer=tokenizer,
        block_size_tokens=block_size,
        kv_bytes_per_token=kv_bytes,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observation_path = output / "workflow_reuse_observation_v1.jsonl"
    with observation_path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_record(), ensure_ascii=False) + "\n")
    manifest = {
        "schema": "astrakv-workflow-observation-manifest-v1",
        "evidence_class": "modeled_dataset_metadata",
        "dataset_id": observations[0].dataset_id if observations else None,
        "workload_id": observations[0].workload_id if observations else None,
        "input_path": str(source_path),
        "input_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(observation_path),
        "model_path": model_path,
        "tokenizer_class": type(tokenizer).__name__,
        "block_size_tokens": block_size,
        "kv_bytes_per_token": kv_bytes,
        "request_count": len(observations),
    }
    manifest.update(tokenizer_manifest_metadata(tokenizer))
    if args.task1_dir:
        manifest["input_directory"] = str(Path(args.task1_dir))
        manifest["input_prompt_sha256"] = manifest["input_sha256"]
    else:
        manifest["input_replay_jsonl"] = str(source_path)
    (output / "workflow_reuse_observation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/workflow_observation_qwen3_8b.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task1-dir")
    source.add_argument("--replay-jsonl")
    parser.add_argument("--workload", choices=("random", "grouped"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--block-size-tokens", type=int)
    parser.add_argument("--kv-bytes-per-token", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_manifest_metadata(tokenizer: object) -> dict[str, str]:
    """Record the tokenizer identity and chat-template bytes used for observation."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, dict):
        init_kwargs = {}
    revision = init_kwargs.get("_commit_hash") or init_kwargs.get("revision") or "unknown"
    template = getattr(tokenizer, "chat_template", "") or ""
    if not isinstance(template, str):
        template = str(template)
    return {
        "tokenizer_identifier": str(getattr(tokenizer, "name_or_path", "") or "unknown"),
        "tokenizer_revision": str(revision),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
