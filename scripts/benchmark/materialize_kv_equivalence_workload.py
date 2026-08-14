#!/usr/bin/env python3
"""Materialize a same-engine native-load versus recompute KV probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow, load_runtime_workload_jsonl


WORKLOAD_NAME = "kv_equivalence_single_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", default="", help="Existing runtime workload JSONL with exact-prefix source rows.")
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Audited grouped_prompts.jsonl (workload_prompts/<dataset>/) used to synthesize an exact-prefix source row.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-request-id", default="")
    parser.add_argument("--output-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_tokens < 1:
        raise SystemExit("--output-tokens must be positive")
    source_path = Path(args.source_workload) if args.source_workload else None
    prompts_path = Path(args.prompts_file) if args.prompts_file else None
    if (source_path is None) == (prompts_path is None):
        raise SystemExit("exactly one of --source-workload or --prompts-file is required")
    if prompts_path is not None:
        source = select_prompt_source(prompts_path, args.source_request_id)
        source_path = prompts_path
    else:
        assert source_path is not None
        source = select_source(source_path, args.source_request_id)
    rows = materialize(source, output_tokens=args.output_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = output_dir / f"{WORKLOAD_NAME}.jsonl"
    workload.write_text("".join(json.dumps(row.to_record(), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema": "astrakv-kv-equivalence-workload-v1",
        "workload": WORKLOAD_NAME,
        "source_workload": str(source_path),
        "source_workload_sha256": sha256_file(source_path),
        "source_request_id": source.request_id,
        "source_prompt_sha256": hashlib.sha256(source.prompt.encode()).hexdigest(),
        "workload_sha256": sha256_file(workload),
        "same_engine_required": True,
        "test_only_force_recompute": True,
    }
    (output_dir / "kv_equivalence_workload.manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def select_source(path: Path, requested_id: str) -> RuntimeWorkloadRow:
    raw_metadata: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                raw_metadata[str(item.get("request_id") or "")] = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    candidates = [
        row for row in load_runtime_workload_jsonl(path)
        if raw_metadata.get(row.request_id, {}).get("exact_prefix") is True
        and isinstance(raw_metadata.get(row.request_id, {}).get("messages"), list)
        and row.context_length and row.context_length > 0
        and row.cache_key and row.prefix_hash
    ]
    if requested_id:
        candidates = [row for row in candidates if row.request_id == requested_id]
    if not candidates:
        raise ValueError("no eligible exact-prefix source row with messages and cache identity")
    return min(candidates, key=lambda row: row.arrival_index)


def select_prompt_source(path: Path, requested_id: str) -> RuntimeWorkloadRow:
    """Synthesize an exact-prefix source row from an audited grouped prompt.

    The equivalence probe needs the exact token sequence and a real chat
    message so the benchmark can compute deterministic token ids.  Grouped
    prompts carry the full context plus question, so we wrap the prompt as a
    single user message and derive cache identity from the prompt digest.
    """

    candidates: list[tuple[dict[str, Any], str, int, int]] = []
    for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            continue
        if item.get("shared_context") is not True:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        estimated = int(metadata.get("estimated_prompt_tokens") or 0)
        if estimated <= 0:
            estimated = max(1, len(prompt) // 4)
        candidates.append((item, prompt, estimated, line_number))
    if requested_id:
        candidates = [
            candidate for candidate in candidates
            if str(candidate[0].get("request_id") or "") == requested_id
            or str(candidate[0].get("sample_id") or "") == requested_id
        ]
    if not candidates:
        raise ValueError("no eligible grouped prompt source with shared_context and prompt text")
    item, prompt, estimated, _ = min(candidates, key=lambda candidate: (
        int(candidate[0].get("order") or 0), candidate[3],
    ))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    metadata = dict(item.get("metadata") if isinstance(item.get("metadata"), dict) else {})
    metadata.update({
        "exact_prefix": True,
        "messages": [{"role": "user", "content": prompt}],
    })
    return RuntimeWorkloadRow(
        request_id=str(item.get("request_id") or f"equivalence-source-{digest[:12]}"),
        prompt=prompt,
        prefix_id=str(item.get("reuse_group") or f"prompt:{digest[:16]}"),
        prefix_hash=f"sha256:{digest}",
        cache_key=f"sha256:{digest}",
        arrival_index=0,
        reuse_ratio=1.0,
        reuse_bucket="high",
        context_length=estimated,
        expected_output_tokens=128,
        batch_size=1,
        sleep_before_s=0.0,
        prefetch_lead_s=0.0,
        case="kv_equivalence_source",
        metadata=metadata,
    )


def materialize(source: RuntimeWorkloadRow, *, output_tokens: int) -> list[RuntimeWorkloadRow]:
    messages = source.metadata.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("source metadata.messages is required")
    digest = hashlib.sha256(source.request_id.encode()).hexdigest()[:12]
    rows: list[RuntimeWorkloadRow] = []
    for index, role in enumerate(("seed", "loaded", "recompute")):
        request_id = f"kv-equivalence-{digest}-{role}"
        metadata = copy.deepcopy(source.metadata)
        metadata.update({
            "workload_type": WORKLOAD_NAME,
            "scenario": "same_engine_kv_equivalence",
            "equivalence_role": role,
            "sample_id": request_id,
            "generation_seed": 0,
            "source_request_id": source.request_id,
            "exact_prefix": True,
            "kv_core_equivalence_mode": "force_recompute" if role == "recompute" else "",
        })
        rows.append(RuntimeWorkloadRow(
            request_id=request_id, prompt=source.prompt, prefix_id=source.prefix_id,
            prefix_hash=source.prefix_hash, cache_key=source.cache_key,
            arrival_index=index, reuse_ratio=0.0 if role == "seed" else 1.0,
            reuse_bucket="none" if role == "seed" else "high",
            context_length=source.context_length, expected_output_tokens=output_tokens,
            batch_size=1, sleep_before_s=1.0 if role == "loaded" else None,
            prefetch_lead_s=0.0, case=f"kv_equivalence_{role}", metadata=metadata,
        ))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
