"""Controlled workload and report helpers for text/KV reuse consistency."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl


RUNTIME_WORKLOAD_SCHEMA = "astra-runtime-workload-v1"
WORKFLOW_REPLAY_SCHEMA = "astrakv-workflow-trace-v1"
REPORT_SCHEMA = "astrakv-text-kv-consistency-report-v3"
WORKLOAD_MANIFEST_SCHEMA = "astrakv-text-kv-consistency-workload-v2"
SIMILARITY_BUCKETS = ("exact", "sim90", "sim80")
WARMUP_CONDITIONS = ("cold", "warm", "hot")
CONDITION_WARMUP_REPEATS = {"cold": 0, "warm": 1, "hot": 2}
TARGET_PREFIX_RATIOS = {"exact": 1.0, "sim90": 0.9, "sim80": 0.8}
DEFAULT_CONTEXT_LENGTHS = (8192, 16384)
DEFAULT_BLOCK_SIZE_TOKENS = 16
DEFAULT_BENCHMARK_SYSTEM_PROMPT = "You are a concise benchmark assistant. Answer directly."
CONTEXT_LABEL_OVERRIDES = {8192: "ctx8k", 16384: "ctx16k"}
_BUCKET_CODES = {"exact": "ex", "sim90": "n9", "sim80": "n8"}
_TOKEN_ATOM_CANDIDATES = (
    " alpha",
    " beta",
    " gamma",
    " delta",
    " epsilon",
    " zeta",
    " theta",
    " lambda",
    " omega",
    " reuse",
    " cache",
    " prefix",
    " block",
    " token",
    " vector",
    " memory",
)


@dataclass(frozen=True, slots=True)
class WorkloadBundle:
    context_length: int
    block_size_tokens: int
    analysis_rows: list[dict[str, Any]]
    warmup_rows: list[dict[str, Any]]
    replay_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def build_workload_bundle(
    *,
    context_length: int,
    expected_output_tokens: int,
    batch_size: int = 1,
    block_size_tokens: int = DEFAULT_BLOCK_SIZE_TOKENS,
    tokenizer: Any | None = None,
    system_prompt: str = DEFAULT_BENCHMARK_SYSTEM_PROMPT,
) -> WorkloadBundle:
    if tokenizer is not None:
        return _build_tokenizer_aligned_workload_bundle(
            context_length=context_length,
            expected_output_tokens=expected_output_tokens,
            batch_size=batch_size,
            block_size_tokens=block_size_tokens,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
    block_count = _block_count_for_context(context_length, block_size_tokens)
    analysis_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    expected_blocks: dict[str, int] = {}
    mutation_start_blocks: dict[str, int] = {}

    for arrival_index, bucket in enumerate(SIMILARITY_BUCKETS):
        target_ratio = TARGET_PREFIX_RATIOS[bucket]
        prefix_id = f"text-kv-{bucket}-ctx{context_length}"
        anchor_request_id = f"{bucket}-anchor"
        probe_request_id = f"{bucket}-probe"
        expected_reusable_blocks = _shared_block_count(block_count, target_ratio)
        mutation_start_block = expected_reusable_blocks if target_ratio < 1.0 else block_count
        expected_blocks[bucket] = expected_reusable_blocks
        mutation_start_blocks[bucket] = mutation_start_block

        anchor_prompt = _build_prompt(
            bucket=bucket,
            variant="anchor",
            target_ratio=1.0,
            context_length=context_length,
            block_size_tokens=block_size_tokens,
            block_count=block_count,
        )
        probe_prompt = _build_prompt(
            bucket=bucket,
            variant="probe",
            target_ratio=target_ratio,
            context_length=context_length,
            block_size_tokens=block_size_tokens,
            block_count=block_count,
        )

        common_metadata = {
            "scenario": "text_kv_consistency",
            "similarity_bucket": bucket,
            "target_prefix_ratio": target_ratio,
            "workload_type": "text_kv_consistency",
            "shared_context": True,
            "reuse_group": prefix_id,
            "context_length": context_length,
            "block_size_tokens": block_size_tokens,
            "block_count": block_count,
            "target_mutation_start_block": mutation_start_block,
            "expected_reusable_blocks": expected_reusable_blocks,
            "expected_reusable_tokens": expected_reusable_blocks * block_size_tokens,
        }
        warmup_rows.append(
            _workload_row(
                request_id=anchor_request_id,
                prompt=anchor_prompt,
                prefix_id=prefix_id,
                arrival_index=arrival_index,
                reuse_ratio=0.0,
                reuse_bucket="none",
                case=f"{bucket}_anchor",
                metadata={
                    **common_metadata,
                    "role": "warmup_anchor",
                    "expected_reuse": 0,
                    "reference_request_id": "",
                },
                context_length=context_length,
                expected_output_tokens=expected_output_tokens,
                batch_size=batch_size,
            )
        )
        analysis_rows.append(
            _workload_row(
                request_id=probe_request_id,
                prompt=probe_prompt,
                prefix_id=prefix_id,
                arrival_index=arrival_index,
                reuse_ratio=target_ratio,
                reuse_bucket=_bucket_label(target_ratio),
                case=f"{bucket}_probe",
                metadata={
                    **common_metadata,
                    "role": "analysis_probe",
                    "expected_reuse": 1 if target_ratio > 0 else 0,
                    "reference_request_id": anchor_request_id,
                },
                context_length=context_length,
                expected_output_tokens=expected_output_tokens,
                batch_size=batch_size,
            )
        )
        replay_rows.extend(
            [
                _workflow_replay_row(
                    request_id=anchor_request_id,
                    arrival_index=len(replay_rows),
                    prompt=anchor_prompt,
                ),
                _workflow_replay_row(
                    request_id=probe_request_id,
                    arrival_index=len(replay_rows) + 1,
                    prompt=probe_prompt,
                ),
            ]
        )

    manifest = {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "context_length": context_length,
        "target_context_lengths": [context_length],
        "target_block_size_tokens": block_size_tokens,
        "analysis_request_count": len(analysis_rows),
        "warmup_request_count": len(warmup_rows),
        "replay_request_count": len(replay_rows),
        "similarity_buckets": list(SIMILARITY_BUCKETS),
        "warmup_conditions": list(WARMUP_CONDITIONS),
        "target_prefix_ratios": dict(TARGET_PREFIX_RATIOS),
        "target_mutation_start_block": mutation_start_blocks,
        "expected_reusable_blocks": expected_blocks,
        "analysis_request_ids": [row["request_id"] for row in analysis_rows],
        "warmup_request_ids": [row["request_id"] for row in warmup_rows],
    }
    return WorkloadBundle(
        context_length=context_length,
        block_size_tokens=block_size_tokens,
        analysis_rows=analysis_rows,
        warmup_rows=warmup_rows,
        replay_rows=replay_rows,
        manifest=manifest,
    )


def _build_tokenizer_aligned_workload_bundle(
    *,
    context_length: int,
    expected_output_tokens: int,
    batch_size: int,
    block_size_tokens: int,
    tokenizer: Any,
    system_prompt: str,
) -> WorkloadBundle:
    max_prompt_tokens = max(block_size_tokens, int(context_length) - int(expected_output_tokens))
    shared_atom, sim90_atom, sim80_atom, pad_atom = _select_token_atoms(tokenizer)
    analysis_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    expected_blocks: dict[str, int] = {}
    mutation_start_blocks: dict[str, int] = {}
    observed_prompt_tokens: dict[str, int] = {}
    observed_prompt_blocks: dict[str, int] = {}
    prompt_token_budgets: dict[str, int] = {bucket: max_prompt_tokens for bucket in SIMILARITY_BUCKETS}
    resolved_bucket_specs: dict[str, dict[str, Any]] = {}
    resolved_prompt_tokens: int | None = None

    for _ in range(max_prompt_tokens - block_size_tokens + 1):
        candidate_specs = {
            bucket: _build_tokenizer_bucket_spec(
                bucket=bucket,
                context_length=context_length,
                block_size_tokens=block_size_tokens,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                target_prompt_tokens=prompt_token_budgets[bucket],
                shared_atom=shared_atom,
                sim90_atom=sim90_atom,
                sim80_atom=sim80_atom,
                pad_atom=pad_atom,
            )
            for bucket in SIMILARITY_BUCKETS
        }
        observed_candidates = [
            int(spec["observed_anchor_tokens"])
            for spec in candidate_specs.values()
        ] + [
            int(spec["observed_probe_tokens"])
            for spec in candidate_specs.values()
        ]
        common_prompt_tokens = min(observed_candidates)
        mismatched_buckets = [
            bucket
            for bucket, spec in candidate_specs.items()
            if int(spec["observed_anchor_tokens"]) != common_prompt_tokens
            or int(spec["observed_probe_tokens"]) != common_prompt_tokens
        ]
        if not mismatched_buckets:
            resolved_bucket_specs = candidate_specs
            resolved_prompt_tokens = common_prompt_tokens
            break
        for bucket in mismatched_buckets:
            if prompt_token_budgets[bucket] <= block_size_tokens:
                raise ValueError(
                    f"context_length={context_length} could not align bucket={bucket} "
                    "to a shared tokenizer-backed prompt length"
                )
            prompt_token_budgets[bucket] -= 1
    if resolved_prompt_tokens is None:
        raise ValueError(
            f"context_length={context_length} could not resolve tokenizer-aligned prompt lengths "
            f"within max_prompt_tokens={max_prompt_tokens}"
        )

    for arrival_index, bucket in enumerate(SIMILARITY_BUCKETS):
        prefix_id = f"text-kv-{bucket}-ctx{context_length}"
        anchor_request_id = f"{bucket}-anchor"
        probe_request_id = f"{bucket}-probe"
        bucket_spec = resolved_bucket_specs[bucket]
        observed_total_tokens = int(bucket_spec["observed_anchor_tokens"])
        total_block_count = _blocks_from_tokens(observed_total_tokens, block_size_tokens)
        target_ratio = float(bucket_spec["target_ratio"])
        mutation_start_block = int(bucket_spec["mutation_start_block"])
        shared_block_count = int(bucket_spec["shared_block_count"])
        expected_blocks[bucket] = mutation_start_block if target_ratio < 1.0 else total_block_count
        mutation_start_blocks[bucket] = mutation_start_block if target_ratio < 1.0 else total_block_count
        observed_prompt_tokens[bucket] = observed_total_tokens
        observed_prompt_blocks[bucket] = total_block_count
        anchor_messages = list(bucket_spec["anchor_messages"])
        probe_messages = list(bucket_spec["probe_messages"])
        anchor_prompt = str(bucket_spec["anchor_prompt"])
        probe_prompt = str(bucket_spec["probe_prompt"])
        common_metadata = {
            "scenario": "text_kv_consistency",
            "similarity_bucket": bucket,
            "target_prefix_ratio": target_ratio,
            "workload_type": "text_kv_consistency",
            "shared_context": True,
            "reuse_group": prefix_id,
            "context_length": context_length,
            "block_size_tokens": block_size_tokens,
            "block_count": total_block_count,
            "target_mutation_start_block": mutation_start_blocks[bucket],
            "expected_reusable_blocks": expected_blocks[bucket],
            "expected_reusable_tokens": expected_blocks[bucket] * block_size_tokens,
            "prompt_token_budget": prompt_token_budgets[bucket],
            "observed_prompt_tokens": observed_total_tokens,
            "messages": anchor_messages,
        }
        warmup_rows.append(
            _workload_row(
                request_id=anchor_request_id,
                prompt=anchor_prompt,
                prefix_id=prefix_id,
                arrival_index=arrival_index,
                reuse_ratio=0.0,
                reuse_bucket="none",
                case=f"{bucket}_anchor",
                metadata={
                    **common_metadata,
                    "role": "warmup_anchor",
                    "expected_reuse": 0,
                    "reference_request_id": "",
                },
                context_length=context_length,
                expected_output_tokens=expected_output_tokens,
                batch_size=batch_size,
            )
        )
        analysis_rows.append(
            _workload_row(
                request_id=probe_request_id,
                prompt=probe_prompt,
                prefix_id=prefix_id,
                arrival_index=arrival_index,
                reuse_ratio=target_ratio,
                reuse_bucket=_bucket_label(target_ratio),
                case=f"{bucket}_probe",
                metadata={
                    **common_metadata,
                    "role": "analysis_probe",
                    "expected_reuse": 1,
                    "reference_request_id": anchor_request_id,
                    "messages": probe_messages,
                },
                context_length=context_length,
                expected_output_tokens=expected_output_tokens,
                batch_size=batch_size,
            )
        )
        replay_rows.extend(
            [
                _workflow_replay_row(
                    request_id=anchor_request_id,
                    arrival_index=len(replay_rows),
                    messages=anchor_messages,
                ),
                _workflow_replay_row(
                    request_id=probe_request_id,
                    arrival_index=len(replay_rows) + 1,
                    messages=probe_messages,
                ),
            ]
        )

    manifest = {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "context_length": context_length,
        "target_context_lengths": [context_length],
        "target_block_size_tokens": block_size_tokens,
        "prompt_token_budget": resolved_prompt_tokens,
        "prompt_token_budget_by_bucket": prompt_token_budgets,
        "tokenizer_aligned": True,
        "analysis_request_count": len(analysis_rows),
        "warmup_request_count": len(warmup_rows),
        "replay_request_count": len(replay_rows),
        "similarity_buckets": list(SIMILARITY_BUCKETS),
        "warmup_conditions": list(WARMUP_CONDITIONS),
        "target_prefix_ratios": dict(TARGET_PREFIX_RATIOS),
        "target_mutation_start_block": mutation_start_blocks,
        "expected_reusable_blocks": expected_blocks,
        "observed_prompt_tokens": observed_prompt_tokens,
        "observed_prompt_blocks": observed_prompt_blocks,
        "analysis_request_ids": [row["request_id"] for row in analysis_rows],
        "warmup_request_ids": [row["request_id"] for row in warmup_rows],
    }
    return WorkloadBundle(
        context_length=context_length,
        block_size_tokens=block_size_tokens,
        analysis_rows=analysis_rows,
        warmup_rows=warmup_rows,
        replay_rows=replay_rows,
        manifest=manifest,
    )


def _build_tokenizer_bucket_spec(
    *,
    bucket: str,
    context_length: int,
    block_size_tokens: int,
    tokenizer: Any,
    system_prompt: str,
    target_prompt_tokens: int,
    shared_atom: str,
    sim90_atom: str,
    sim80_atom: str,
    pad_atom: str,
) -> dict[str, Any]:
    header = (
        "AstraKV text/KV consistency validation. "
        f"Bucket={bucket}. "
        f"Target prompt tokens={target_prompt_tokens}. "
        f"Tokenizer block size={block_size_tokens}. "
        "The next section is intentionally repetitive and block aligned.\n\n"
    )
    footer = "\n\nAnswer in two short sentences about why prefix reuse matters."
    prefix_messages = _messages_for_prompt(system_prompt, header)
    prefix_token_count = len(_chat_token_ids(tokenizer, prefix_messages))
    alignment_tokens = (-prefix_token_count) % block_size_tokens
    alignment_text = shared_atom * alignment_tokens
    fixed_messages = _messages_for_prompt(system_prompt, header + alignment_text + footer)
    fixed_token_count = len(_chat_token_ids(tokenizer, fixed_messages))
    filler_token_budget = target_prompt_tokens - fixed_token_count
    if filler_token_budget < block_size_tokens:
        raise ValueError(
            f"context_length={context_length} leaves no room for aligned filler under the tokenizer-backed budget"
        )
    body_block_count = filler_token_budget // block_size_tokens
    trailing_pad_tokens = filler_token_budget - (body_block_count * block_size_tokens)
    shared_block = shared_atom * block_size_tokens
    mutated_block = (sim90_atom if bucket == "sim90" else sim80_atom) * block_size_tokens
    trailing_pad = pad_atom * trailing_pad_tokens
    shared_prefix_before_body = prefix_token_count + alignment_tokens
    exact_body = shared_block * body_block_count
    exact_prompt = header + alignment_text + exact_body + trailing_pad + footer
    exact_messages = _messages_for_prompt(system_prompt, exact_prompt)
    observed_anchor_tokens = len(_chat_token_ids(tokenizer, exact_messages))
    target_ratio = TARGET_PREFIX_RATIOS[bucket]
    desired_shared_tokens = (
        observed_anchor_tokens
        if target_ratio >= 1.0
        else int(round(observed_anchor_tokens * target_ratio))
    )
    shared_block_count = body_block_count if target_ratio >= 1.0 else max(
        0,
        min(
            body_block_count,
            int((max(shared_prefix_before_body, desired_shared_tokens) - shared_prefix_before_body) // block_size_tokens),
        ),
    )
    if target_ratio < 1.0 and shared_block_count >= body_block_count:
        shared_block_count = max(0, body_block_count - 1)
    mutation_start_block = _blocks_from_tokens(shared_prefix_before_body, block_size_tokens) + shared_block_count
    probe_body = (shared_block * shared_block_count) + (mutated_block * max(0, body_block_count - shared_block_count))
    probe_prompt = header + alignment_text + probe_body + trailing_pad + footer
    probe_messages = _messages_for_prompt(system_prompt, probe_prompt)
    observed_probe_tokens = len(_chat_token_ids(tokenizer, probe_messages))
    return {
        "target_ratio": target_ratio,
        "body_block_count": body_block_count,
        "shared_block_count": shared_block_count,
        "mutation_start_block": mutation_start_block,
        "anchor_prompt": exact_prompt,
        "probe_prompt": probe_prompt,
        "anchor_messages": exact_messages,
        "probe_messages": probe_messages,
        "observed_anchor_tokens": observed_anchor_tokens,
        "observed_probe_tokens": observed_probe_tokens,
    }


def write_workload_bundle(
    output_dir: str | Path,
    bundle: WorkloadBundle,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    analysis_path = root / "analysis_workload.jsonl"
    warmup_path = root / "warmup_workload.jsonl"
    replay_path = root / "pairwise_reference_replay.jsonl"
    manifest_path = root / "text_kv_consistency_workload_manifest.json"
    report_path = root / "text_kv_consistency_workload_report.md"
    _write_jsonl(analysis_path, bundle.analysis_rows)
    _write_jsonl(warmup_path, bundle.warmup_rows)
    _write_jsonl(replay_path, bundle.replay_rows)
    manifest_path.write_text(json.dumps(bundle.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(_workload_report(bundle), encoding="utf-8")
    return {
        "analysis_workload": analysis_path,
        "warmup_workload": warmup_path,
        "pairwise_reference_replay": replay_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def build_suite_report(suite_dir: str | Path) -> dict[str, Any]:
    root = Path(suite_dir)
    context_specs = _discover_context_specs(root)
    contexts: dict[str, dict[str, Any]] = {}
    context_labels: list[str] = []
    consistency_labels: list[str] = []

    for spec in context_specs:
        context_report = _build_context_report(root, spec)
        context_key = str(spec["context_length"])
        contexts[context_key] = context_report
        context_labels.append(context_key)
        consistency_labels.append(str(context_report.get("classification", {}).get("label") or "inconsistent"))

    classification = _classify_suite(contexts)
    report = {
        "schema": REPORT_SCHEMA,
        "suite_dir": str(root),
        "context_lengths": [int(item) for item in context_labels],
        "contexts": contexts,
        "text_similarity_definition": "shared_prefix_token_ratio_via_tokenizer_blocks",
        "classification": classification,
    }
    if len(context_specs) == 1:
        only_report = contexts[context_labels[0]]
        report["conditions"] = only_report.get("conditions", {})
        report["consistency_checks"] = only_report.get("consistency_checks", {})
    return report


def classify_consistency(inputs: dict[str, Any]) -> dict[str, str]:
    missing = list(inputs.get("missing_artifacts") or [])
    if missing:
        return {
            "label": "inconsistent",
            "summary": "Missing benchmark or cache-evidence artifacts prevented a trustworthy consistency claim.",
        }

    cold_total = int(inputs.get("cold_expected_miss_count") or 0)
    cold_miss = int(inputs.get("cold_observed_miss_count") or 0)
    warm_hot_total = int(inputs.get("warm_hot_rows") or 0)
    warm_hot_signals = int(inputs.get("warm_hot_rows_with_cache_signal") or 0)
    warm_hot_block_evidence = int(inputs.get("warm_hot_rows_with_block_evidence") or 0)
    monotonic_ok = bool(inputs.get("warm_hot_monotonic_ok"))
    ttft_improved = int(inputs.get("warm_hot_ttft_improved_count") or 0)
    fallback_only_rows = int(inputs.get("fallback_only_rows") or 0)
    missing_block_evidence_rows = int(inputs.get("warm_hot_rows_missing_block_evidence") or 0)

    if (
        cold_total > 0
        and cold_miss == cold_total
        and warm_hot_total > 0
        and warm_hot_signals == warm_hot_total
        and warm_hot_block_evidence == warm_hot_total
        and monotonic_ok
        and ttft_improved > 0
        and fallback_only_rows == 0
        and missing_block_evidence_rows == 0
    ):
        return {
            "label": "consistent",
            "summary": "Text-level prefix similarity, block-level KV reuse evidence, and TTFT improvements aligned across all observed long-context runs.",
        }
    if (
        cold_total > 0
        and cold_miss >= max(1, cold_total - 1)
        and warm_hot_total > 0
        and warm_hot_signals >= max(1, warm_hot_total // 2)
        and (
            warm_hot_block_evidence >= max(1, warm_hot_total // 2)
            or missing_block_evidence_rows > 0
            or fallback_only_rows > 0
        )
    ):
        return {
            "label": "partially_consistent",
            "summary": "The expected trend was visible, but part of the block-level KV evidence was incomplete or depended on coarse fallback signals.",
        }
    return {
        "label": "inconsistent",
        "summary": "Observed cache signals did not reliably track the text-level prefix-reuse expectation.",
    }


def write_suite_report(output_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "text_kv_consistency_report.json"
    markdown_path = root / "text_kv_consistency_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_suite_report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _build_context_report(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    workload_dir = Path(spec["workload_dir"])
    text_observation_dir = Path(spec["text_observation_dir"])
    analysis_rows = load_runtime_workload_jsonl(workload_dir / "analysis_workload.jsonl")
    text_observations = _read_jsonl(text_observation_dir / "workflow_reuse_observation_v1.jsonl")
    text_by_request = {str(item.get("workflow_id") or ""): item for item in text_observations}
    block_size_tokens = _resolve_block_size_tokens(analysis_rows, text_observations, workload_dir)

    summary_by_condition: dict[str, Any] = {}
    classification_inputs = {
        "cold_expected_miss_count": 0,
        "cold_observed_miss_count": 0,
        "warm_hot_rows": 0,
        "warm_hot_rows_with_cache_signal": 0,
        "warm_hot_rows_with_block_evidence": 0,
        "warm_hot_rows_missing_block_evidence": 0,
        "warm_hot_monotonic_ok": True,
        "warm_hot_ttft_improved_count": 0,
        "fallback_only_rows": 0,
        "missing_artifacts": [],
    }

    for condition in WARMUP_CONDITIONS:
        condition_root = Path(spec["condition_roots"][condition])
        run_root = condition_root / "run"
        cache_events_path = condition_root / "cache_events" / "cache_events.jsonl"
        request_rows = _read_jsonl(run_root / "request_results.jsonl")
        cache_events = _read_jsonl(cache_events_path)
        runtime_events_raw = _read_jsonl(run_root / "runtime_events_raw.jsonl")
        runtime_structured_events = _read_jsonl(run_root / "runtime_structured_events.jsonl")
        runtime_receipts = _read_jsonl(run_root / "runtime_command_receipts.jsonl")

        if not request_rows:
            classification_inputs["missing_artifacts"].append(str(run_root / "request_results.jsonl"))
        if not cache_events and not runtime_events_raw and not runtime_structured_events:
            classification_inputs["missing_artifacts"].append(str(cache_events_path))

        request_by_id = {str(item.get("request_id") or ""): item for item in request_rows}
        cache_by_id = _group_by_request(cache_events)
        runtime_raw_by_id = _group_by_request(runtime_events_raw)
        runtime_structured_by_id = _group_by_request(runtime_structured_events)
        receipts_by_id = _group_by_request(runtime_receipts)

        request_entries: list[dict[str, Any]] = []
        monotonic_candidates: list[tuple[str, int]] = []
        for row in analysis_rows:
            request_id = row.request_id
            request_result = request_by_id.get(request_id, {})
            text_item = text_by_request.get(request_id, {})
            evidence = summarize_reuse_evidence(
                runtime_raw_by_id.get(request_id, []),
                runtime_structured_by_id.get(request_id, []),
                cache_by_id.get(request_id, []),
                block_size_tokens=block_size_tokens,
            )
            target_ratio = float((row.metadata or {}).get("target_prefix_ratio") or 0.0)
            text_ratio = _safe_ratio(
                text_item.get("historical_reused_tokens"),
                text_item.get("token_count"),
            )
            ttft_ms = _as_float(request_result.get("ttft_ms"))
            latency_ms = _as_float(request_result.get("latency_ms"))
            expected_reusable_blocks = _as_int((row.metadata or {}).get("expected_reusable_blocks"))
            expected_reusable_tokens = _as_int((row.metadata or {}).get("expected_reusable_tokens"))
            observed_kv_reuse_blocks = _observed_kv_reuse_blocks(evidence)
            observed_kv_reuse_tokens = _observed_kv_reuse_tokens(evidence)
            request_entry = {
                "request_id": request_id,
                "status": str(request_result.get("status") or "missing"),
                "error": str(request_result.get("error") or ""),
                "ttft_ms": ttft_ms,
                "latency_ms": latency_ms,
                "output_tokens_observed": _as_int(request_result.get("output_tokens_observed")),
                "similarity_bucket": str((row.metadata or {}).get("similarity_bucket") or ""),
                "target_prefix_ratio": target_ratio,
                "observed_text_prefix_ratio": text_ratio,
                "text_token_count": _as_int(text_item.get("token_count")),
                "text_historical_reused_tokens": _as_int(text_item.get("historical_reused_tokens")),
                "expected_reusable_blocks": expected_reusable_blocks,
                "expected_reusable_tokens": expected_reusable_tokens,
                "cache_signal": evidence["cache_signal"],
                "cache_hit_tokens": evidence["cache_hit_tokens"],
                "cache_load_tokens": evidence["cache_load_tokens"],
                "cache_event_types": evidence["event_types"],
                "structured_cache_hit_blocks": evidence["structured_cache_hit_blocks"],
                "structured_cache_hit_tokens": evidence["structured_cache_hit_tokens"],
                "structured_cache_load_blocks": evidence["structured_cache_load_blocks"],
                "structured_cache_load_tokens": evidence["structured_cache_load_tokens"],
                "structured_store_blocks": evidence["structured_store_blocks"],
                "observed_kv_hit_blocks": evidence["structured_cache_hit_blocks"],
                "observed_kv_load_blocks": evidence["structured_cache_load_blocks"],
                "observed_kv_store_blocks": evidence["structured_store_blocks"],
                "observed_kv_reuse_blocks": observed_kv_reuse_blocks,
                "observed_kv_reuse_tokens": observed_kv_reuse_tokens,
                "observed_kv_reuse_ratio": _safe_ratio(
                    observed_kv_reuse_tokens,
                    text_item.get("token_count"),
                ),
                "kv_block_gap_vs_expected": observed_kv_reuse_blocks - expected_reusable_blocks,
                "block_evidence_complete": _block_evidence_complete(
                    evidence["cache_signal"],
                    observed_kv_reuse_blocks,
                    observed_kv_reuse_tokens,
                ),
                "has_structured_block_evidence": evidence["has_structured_block_evidence"],
                "evidence_source": evidence["evidence_source"],
                "runtime_event_count": len(runtime_raw_by_id.get(request_id, [])),
                "runtime_structured_event_count": len(runtime_structured_by_id.get(request_id, [])),
                "runtime_receipt_count": len(receipts_by_id.get(request_id, [])),
            }
            request_entries.append(request_entry)

            if condition == "cold":
                classification_inputs["cold_expected_miss_count"] += 1
                if request_entry["cache_signal"] == "miss":
                    classification_inputs["cold_observed_miss_count"] += 1
            else:
                classification_inputs["warm_hot_rows"] += 1
                if request_entry["cache_signal"] != "miss":
                    classification_inputs["warm_hot_rows_with_cache_signal"] += 1
                if request_entry["block_evidence_complete"]:
                    classification_inputs["warm_hot_rows_with_block_evidence"] += 1
                elif request_entry["cache_signal"] != "miss":
                    classification_inputs["warm_hot_rows_missing_block_evidence"] += 1
                if request_entry["evidence_source"] == "log_fallback":
                    classification_inputs["fallback_only_rows"] += 1
                monotonic_candidates.append(
                    (
                        request_entry["similarity_bucket"],
                        int(request_entry["observed_kv_reuse_blocks"] or 0),
                    )
                )

        if condition != "cold":
            ordered_hits = [
                hit
                for _, hit in sorted(monotonic_candidates, key=lambda item: _bucket_sort_key(item[0]))
            ]
            if ordered_hits != sorted(ordered_hits, reverse=True):
                classification_inputs["warm_hot_monotonic_ok"] = False

        summary_by_condition[condition] = {
            "warmup_repeats": CONDITION_WARMUP_REPEATS[condition],
            "request_count": len(request_entries),
            "requests": request_entries,
        }

    cold_requests = {
        item["request_id"]: item
        for item in summary_by_condition.get("cold", {}).get("requests", [])
    }
    for condition in ("warm", "hot"):
        for item in summary_by_condition.get(condition, {}).get("requests", []):
            cold = cold_requests.get(item["request_id"])
            if cold is None:
                continue
            item["ttft_delta_vs_cold_ms"] = _delta(item.get("ttft_ms"), cold.get("ttft_ms"))
            item["latency_delta_vs_cold_ms"] = _delta(item.get("latency_ms"), cold.get("latency_ms"))
            if (
                isinstance(item["ttft_delta_vs_cold_ms"], float)
                and item["ttft_delta_vs_cold_ms"] < 0.0
            ):
                classification_inputs["warm_hot_ttft_improved_count"] += 1

    return {
        "context_length": int(spec["context_length"]),
        "context_label": str(spec["context_label"]),
        "block_size_tokens": block_size_tokens,
        "conditions": summary_by_condition,
        "classification": classify_consistency(classification_inputs),
        "consistency_checks": {
            "cold_expected_miss_count": classification_inputs["cold_expected_miss_count"],
            "cold_observed_miss_count": classification_inputs["cold_observed_miss_count"],
            "warm_hot_rows": classification_inputs["warm_hot_rows"],
            "warm_hot_rows_with_cache_signal": classification_inputs["warm_hot_rows_with_cache_signal"],
            "warm_hot_rows_with_block_evidence": classification_inputs["warm_hot_rows_with_block_evidence"],
            "warm_hot_rows_missing_block_evidence": classification_inputs["warm_hot_rows_missing_block_evidence"],
            "warm_hot_monotonic_ok": classification_inputs["warm_hot_monotonic_ok"],
            "warm_hot_ttft_improved_count": classification_inputs["warm_hot_ttft_improved_count"],
            "fallback_only_rows": classification_inputs["fallback_only_rows"],
            "missing_artifacts": classification_inputs["missing_artifacts"],
        },
    }


def summarize_reuse_evidence(
    runtime_raw_events: list[dict[str, Any]],
    runtime_structured_events: list[dict[str, Any]],
    log_events: list[dict[str, Any]],
    *,
    block_size_tokens: int,
) -> dict[str, Any]:
    raw_summary = _summarize_backend_hook_events(runtime_raw_events, block_size_tokens=block_size_tokens)
    structured_summary = _summarize_runtime_structured_events(runtime_structured_events, block_size_tokens=block_size_tokens)
    log_summary = _summarize_log_cache_evidence(log_events, block_size_tokens=block_size_tokens)

    primary_source = "runtime_events_raw"
    primary = raw_summary
    if not raw_summary["has_signal"]:
        if structured_summary["has_signal"]:
            primary_source = "runtime_structured_events"
            primary = structured_summary
        else:
            primary_source = "log_fallback"
            primary = log_summary

    merged = {
        "cache_signal": primary["cache_signal"],
        "cache_hit_tokens": _first_positive(
            primary["structured_cache_hit_tokens"],
            raw_summary["structured_cache_hit_tokens"],
            structured_summary["structured_cache_hit_tokens"],
            log_summary["structured_cache_hit_tokens"],
        ),
        "cache_load_tokens": _first_positive(
            primary["structured_cache_load_tokens"],
            raw_summary["structured_cache_load_tokens"],
            structured_summary["structured_cache_load_tokens"],
            log_summary["structured_cache_load_tokens"],
        ),
        "structured_cache_hit_blocks": _first_positive(
            primary["structured_cache_hit_blocks"],
            raw_summary["structured_cache_hit_blocks"],
            structured_summary["structured_cache_hit_blocks"],
            log_summary["structured_cache_hit_blocks"],
        ),
        "structured_cache_hit_tokens": _first_positive(
            primary["structured_cache_hit_tokens"],
            raw_summary["structured_cache_hit_tokens"],
            structured_summary["structured_cache_hit_tokens"],
            log_summary["structured_cache_hit_tokens"],
        ),
        "structured_cache_load_blocks": _first_positive(
            primary["structured_cache_load_blocks"],
            raw_summary["structured_cache_load_blocks"],
            structured_summary["structured_cache_load_blocks"],
            log_summary["structured_cache_load_blocks"],
        ),
        "structured_cache_load_tokens": _first_positive(
            primary["structured_cache_load_tokens"],
            raw_summary["structured_cache_load_tokens"],
            structured_summary["structured_cache_load_tokens"],
            log_summary["structured_cache_load_tokens"],
        ),
        "structured_store_blocks": _first_positive(
            primary["structured_store_blocks"],
            raw_summary["structured_store_blocks"],
            structured_summary["structured_store_blocks"],
            log_summary["structured_store_blocks"],
        ),
        "evidence_source": primary_source,
        "event_types": sorted(
            set(
                raw_summary["event_types"]
                + structured_summary["event_types"]
                + log_summary["event_types"]
            )
        ),
    }
    if merged["cache_signal"] == "miss":
        if merged["structured_cache_hit_blocks"] > 0 or merged["structured_cache_hit_tokens"] > 0:
            merged["cache_signal"] = "hit"
        elif merged["structured_cache_load_blocks"] > 0 or merged["structured_cache_load_tokens"] > 0:
            merged["cache_signal"] = "load_only"
    merged["has_structured_block_evidence"] = bool(
        merged["structured_cache_hit_blocks"] > 0
        or merged["structured_cache_load_blocks"] > 0
        or merged["structured_cache_hit_tokens"] > 0
        or merged["structured_cache_load_tokens"] > 0
    )
    merged["observed_kv_reuse_blocks"] = _observed_kv_reuse_blocks(merged)
    merged["observed_kv_reuse_tokens"] = _observed_kv_reuse_tokens(merged)
    return merged


def _discover_context_specs(root: Path) -> list[dict[str, Any]]:
    workload_root = root / "workload"
    legacy_analysis = workload_root / "analysis_workload.jsonl"
    if legacy_analysis.is_file():
        rows = load_runtime_workload_jsonl(legacy_analysis)
        context_length = int(rows[0].context_length if rows else 0)
        context_label = _context_label(context_length)
        return [
            {
                "context_length": context_length,
                "context_label": context_label,
                "workload_dir": workload_root,
                "text_observation_dir": root / "text_observation",
                "condition_roots": {
                    condition: root / condition
                    for condition in WARMUP_CONDITIONS
                },
            }
        ]

    specs: list[dict[str, Any]] = []
    for workload_dir in sorted(path for path in workload_root.iterdir() if path.is_dir()):
        analysis_path = workload_dir / "analysis_workload.jsonl"
        if not analysis_path.is_file():
            continue
        rows = load_runtime_workload_jsonl(analysis_path)
        context_length = int(rows[0].context_length if rows else 0)
        context_label = workload_dir.name
        specs.append(
            {
                "context_length": context_length,
                "context_label": context_label,
                "workload_dir": workload_dir,
                "text_observation_dir": root / "text_observation" / context_label,
                "condition_roots": {
                    condition: root / condition / context_label
                    for condition in WARMUP_CONDITIONS
                },
            }
        )
    return specs


def _classify_suite(contexts: dict[str, dict[str, Any]]) -> dict[str, str]:
    if not contexts:
        return {
            "label": "inconsistent",
            "summary": "No text/KV consistency suite artifacts were discovered.",
        }
    labels = [str(item.get("classification", {}).get("label") or "inconsistent") for item in contexts.values()]
    consistent = [key for key, item in contexts.items() if str(item.get("classification", {}).get("label") or "") == "consistent"]
    partial = [
        key
        for key, item in contexts.items()
        if str(item.get("classification", {}).get("label") or "") == "partially_consistent"
    ]
    failed = [key for key, item in contexts.items() if str(item.get("classification", {}).get("label") or "") == "inconsistent"]
    if labels and all(label == "consistent" for label in labels):
        return {
            "label": "consistent",
            "summary": f"All configured context runs ({', '.join(sorted(contexts))}) showed aligned TTFT and KV reuse evidence.",
        }
    if consistent or partial:
        evidence_contexts = sorted(consistent + partial)
        failed_suffix = f", but other contexts were incomplete ({', '.join(sorted(failed))})" if failed else ""
        return {
            "label": "partially_consistent",
            "summary": f"At least one context run produced usable evidence ({', '.join(evidence_contexts)}){failed_suffix}.",
        }
    return {
        "label": "inconsistent",
        "summary": "No context group produced enough aligned evidence for a consistency claim.",
    }


def _workload_row(
    *,
    request_id: str,
    prompt: str,
    prefix_id: str,
    arrival_index: int,
    reuse_ratio: float,
    reuse_bucket: str,
    case: str,
    metadata: dict[str, Any],
    context_length: int,
    expected_output_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_WORKLOAD_SCHEMA,
        "request_id": request_id,
        "prompt": prompt,
        "prefix_id": prefix_id,
        "prefix_hash": prefix_id,
        "cache_key": prefix_id,
        "arrival_index": arrival_index,
        "reuse_ratio": reuse_ratio,
        "reuse_bucket": reuse_bucket,
        "context_length": context_length,
        "expected_output_tokens": expected_output_tokens,
        "batch_size": batch_size,
        "case": case,
        "metadata": metadata,
    }


def _workflow_replay_row(
    *,
    request_id: str,
    arrival_index: int,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_REPLAY_SCHEMA,
        "workflow_id": request_id,
        "parent_request_id": request_id,
        "subtask_index": 0,
        "arrival_index": arrival_index,
        "adapter": "replay_jsonl",
        "messages": messages if messages is not None else [{"role": "user", "content": str(prompt or "")}],
        "dataset_id": "text-kv-consistency",
        "workload_id": "text-kv-consistency-pairwise",
        "model_identifier": "Qwen3-8B",
        "tool_output_sha256": ("a" * 63) + str(arrival_index % 10),
    }


def _build_prompt(
    *,
    bucket: str,
    variant: str,
    target_ratio: float,
    context_length: int,
    block_size_tokens: int,
    block_count: int,
) -> str:
    keep_blocks = _shared_block_count(block_count, target_ratio)
    shared_blocks = [
        _render_prompt_block(bucket=bucket, block_index=index, block_size_tokens=block_size_tokens, mutated=False)
        if variant == "anchor" or index < keep_blocks
        else _render_prompt_block(bucket=bucket, block_index=index, block_size_tokens=block_size_tokens, mutated=True)
        for index in range(block_count)
    ]
    instruction = (
        "Read the shared context and answer in two short sentences about why prefix reuse matters."
    )
    header = (
        "AstraKV text/KV consistency validation. "
        f"Bucket={bucket}. Variant={variant}. "
        f"Target context length={context_length}. "
        f"Block size tokens={block_size_tokens}. "
        "The following block-aligned segments are fixed-width on purpose."
    )
    return f"{header}\n\n" + "\n".join(shared_blocks) + f"\n\n{instruction}"


def _messages_for_prompt(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _chat_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return _coerce_token_ids(result)


def _encode_content_tokens(tokenizer: Any, text: str) -> list[int]:
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        return _coerce_token_ids(encoded)
    if callable(getattr(tokenizer, "encode", None)):
        return [int(item) for item in tokenizer.encode(text, add_special_tokens=False)]
    raise TypeError("tokenizer must support apply_chat_template and plain-text tokenization")


def _coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids", [])
    else:
        input_ids = getattr(value, "input_ids", None)
        if input_ids is not None:
            value = input_ids
    return [int(item) for item in value]


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    decoded = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        decoded = str(decoded)
    return decoded


def _select_token_atoms(tokenizer: Any) -> tuple[str, str, str, str]:
    selected: list[tuple[str, int]] = []
    for candidate in _TOKEN_ATOM_CANDIDATES:
        token_ids = _encode_content_tokens(tokenizer, candidate)
        if len(token_ids) != 1:
            continue
        token_id = token_ids[0]
        decoded = _decode_single_token(tokenizer, token_id)
        if not decoded.strip():
            continue
        if _encode_content_tokens(tokenizer, decoded) != [token_id]:
            continue
        stable = True
        for existing_text, existing_id in selected:
            if _encode_content_tokens(tokenizer, existing_text + decoded) != [existing_id, token_id]:
                stable = False
                break
            if _encode_content_tokens(tokenizer, decoded + existing_text) != [token_id, existing_id]:
                stable = False
                break
        if not stable:
            continue
        if _encode_content_tokens(tokenizer, decoded + decoded) != [token_id, token_id]:
            continue
        selected.append((decoded, token_id))
        if len(selected) >= 4:
            break
    if len(selected) < 4:
        raise ValueError("could not find enough stable tokenizer atoms for token-aligned workload generation")
    return (selected[0][0], selected[1][0], selected[2][0], selected[3][0])


def _render_prompt_block(*, bucket: str, block_index: int, block_size_tokens: int, mutated: bool) -> str:
    code = _BUCKET_CODES[bucket]
    variant_code = "m" if mutated else "s"
    return " ".join(
        f"{code}{variant_code}{block_index:05d}{offset:02d}"
        for offset in range(block_size_tokens)
    )


def _block_count_for_context(context_length: int, block_size_tokens: int) -> int:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")
    return max(1, int(context_length) // int(block_size_tokens))


def _shared_block_count(block_count: int, target_ratio: float) -> int:
    if target_ratio >= 1.0:
        return block_count
    return max(0, min(block_count, int(round(block_count * max(0.0, target_ratio)))))


def _bucket_label(ratio: float) -> str:
    if ratio >= 0.95:
        return "high"
    if ratio >= 0.85:
        return "medium"
    return "low"


def _workload_report(bundle: WorkloadBundle) -> str:
    lines = [
        "# Text/KV Consistency Workload",
        "",
        f"- Context length: `{bundle.context_length}`",
        f"- Block size tokens: `{bundle.block_size_tokens}`",
        "",
        "## Analysis Buckets",
        "",
        "| bucket | target prefix ratio | reusable blocks | mutation start block | analysis request | warmup anchor |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    warmup_by_bucket = {
        str((row.get("metadata") or {}).get("similarity_bucket") or ""): row
        for row in bundle.warmup_rows
    }
    for row in bundle.analysis_rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        bucket = str(metadata.get("similarity_bucket") or "")
        lines.append(
            f"| {bucket} | {float(metadata.get('target_prefix_ratio') or 0.0):.2f} | "
            f"{int(metadata.get('expected_reusable_blocks') or 0)} | "
            f"{int(metadata.get('target_mutation_start_block') or 0)} | "
            f"{row['request_id']} | {warmup_by_bucket[bucket]['request_id']} |"
        )
    lines.extend(
        [
            "",
            "## Warmup Conditions",
            "",
            "- `cold`: no warmup requests before the analysis workload.",
            "- `warm`: one warmup pass using the anchor prompts before the analysis workload.",
            "- `hot`: two warmup passes using the anchor prompts before the analysis workload.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _suite_report_markdown(report: dict[str, Any]) -> str:
    classification = report.get("classification") if isinstance(report.get("classification"), dict) else {}
    lines = [
        "# Text/KV Consistency Report",
        "",
        f"- Classification: `{classification.get('label', 'unknown')}`",
        f"- Summary: {classification.get('summary', '')}",
        f"- Text similarity definition: `{report.get('text_similarity_definition', '')}`",
        "",
    ]
    contexts = report.get("contexts") if isinstance(report.get("contexts"), dict) else {}
    for context_key in sorted(contexts, key=lambda item: int(item)):
        payload = contexts[context_key]
        lines.extend(
            [
                f"## Context {context_key}",
                "",
                f"- Context label: `{payload.get('context_label', '')}`",
                f"- Block size tokens: `{payload.get('block_size_tokens', 0)}`",
                f"- Classification: `{payload.get('classification', {}).get('label', 'unknown')}`",
                f"- Summary: {payload.get('classification', {}).get('summary', '')}",
                "",
            ]
        )
        for condition in WARMUP_CONDITIONS:
            condition_payload = payload.get("conditions", {}).get(condition, {})
            lines.extend(
                [
                    f"### {condition.title()}",
                    "",
                    f"- Warmup repeats: `{condition_payload.get('warmup_repeats', 0)}`",
                    "",
                    "| request | bucket | text ratio | evidence | expected blocks | reuse blocks | hit blocks | load blocks | store blocks | kv reuse ratio | gap | TTFT ms | status |",
                    "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for item in condition_payload.get("requests", []):
                lines.append(
                    f"| {item.get('request_id', '')} | {item.get('similarity_bucket', '')} | "
                    f"{_fmt_float(item.get('observed_text_prefix_ratio'))} | {item.get('evidence_source', '')} | "
                    f"{item.get('expected_reusable_blocks', 0)} | {item.get('observed_kv_reuse_blocks', 0)} | "
                    f"{item.get('observed_kv_hit_blocks', 0)} | {item.get('observed_kv_load_blocks', 0)} | "
                    f"{item.get('observed_kv_store_blocks', 0)} | {_fmt_float(item.get('observed_kv_reuse_ratio'))} | "
                    f"{item.get('kv_block_gap_vs_expected', 0)} | {_fmt_float(item.get('ttft_ms'))} | "
                    f"{item.get('status', '')} |"
                )
            lines.append("")
        checks = payload.get("consistency_checks", {})
        lines.extend(
            [
                "### Checks",
                "",
                f"- Cold requests expected to miss: `{checks.get('cold_expected_miss_count', 0)}`",
                f"- Cold requests observed as miss: `{checks.get('cold_observed_miss_count', 0)}`",
                f"- Warm/hot requests with cache signal: `{checks.get('warm_hot_rows_with_cache_signal', 0)}` / `{checks.get('warm_hot_rows', 0)}`",
                f"- Warm/hot requests with structured block evidence: `{checks.get('warm_hot_rows_with_block_evidence', 0)}` / `{checks.get('warm_hot_rows', 0)}`",
                f"- Warm/hot requests missing structured block evidence: `{checks.get('warm_hot_rows_missing_block_evidence', 0)}`",
                f"- Warm/hot monotonic exact>=90>=80 by observed KV reuse: `{checks.get('warm_hot_monotonic_ok', False)}`",
                f"- Warm/hot TTFT improvements vs cold: `{checks.get('warm_hot_ttft_improved_count', 0)}`",
                f"- Rows using log fallback: `{checks.get('fallback_only_rows', 0)}`",
            ]
        )
        missing = checks.get("missing_artifacts") if isinstance(checks.get("missing_artifacts"), list) else []
        if missing:
            lines.extend(["", "### Missing Artifacts", ""])
            lines.extend(f"- `{item}`" for item in missing)
        lines.append("")
    return "\n".join(lines)


def _summarize_backend_hook_events(
    rows: list[dict[str, Any]],
    *,
    block_size_tokens: int,
) -> dict[str, Any]:
    hit_blocks = 0
    hit_tokens = 0
    load_blocks = 0
    load_tokens = 0
    store_blocks = 0
    event_types: list[str] = []
    signal = "miss"
    has_signal = False
    for row in rows:
        action = str(row.get("action") or "")
        status = str(row.get("status") or "")
        event_types.append(action)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if action == "cache_hit" and status not in {"failed", "not_found"}:
            has_signal = True
            signal = "hit"
            blocks, tokens = _metric_pair(metadata, prefix="hit", block_size_tokens=block_size_tokens)
            hit_blocks = max(hit_blocks, blocks)
            hit_tokens = max(hit_tokens, tokens)
        elif action == "cache_load" and status not in {"failed", "not_found"}:
            has_signal = True
            if signal != "hit":
                signal = "load_only"
            blocks, tokens = _metric_pair(metadata, prefix="load", block_size_tokens=block_size_tokens)
            load_blocks = max(load_blocks, blocks)
            load_tokens = max(load_tokens, tokens)
        elif action == "cache_store":
            blocks, _ = _metric_pair(metadata, prefix="store", block_size_tokens=block_size_tokens)
            store_blocks = max(store_blocks, blocks)
    return {
        "cache_signal": signal,
        "structured_cache_hit_blocks": hit_blocks,
        "structured_cache_hit_tokens": hit_tokens,
        "structured_cache_load_blocks": load_blocks,
        "structured_cache_load_tokens": load_tokens,
        "structured_store_blocks": store_blocks,
        "event_types": sorted(set(event_types)),
        "has_signal": has_signal,
    }


def _summarize_runtime_structured_events(
    rows: list[dict[str, Any]],
    *,
    block_size_tokens: int,
) -> dict[str, Any]:
    hit_blocks = 0
    hit_tokens = 0
    load_blocks = 0
    load_tokens = 0
    store_blocks = 0
    event_types: list[str] = []
    signal = "miss"
    has_signal = False
    for row in rows:
        action = str(row.get("actual_action") or row.get("action") or "")
        status = str(row.get("status") or "")
        event_types.append(action)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if action in {"cache_hit", "hit"} and status not in {"failed", "not_found"}:
            has_signal = True
            signal = "hit"
            blocks, tokens = _metric_pair(metadata, prefix="hit", block_size_tokens=block_size_tokens)
            hit_blocks = max(hit_blocks, blocks)
            hit_tokens = max(hit_tokens, tokens)
        elif action in {"cache_load", "load"} and status not in {"failed", "not_found"}:
            has_signal = True
            if signal != "hit":
                signal = "load_only"
            blocks, tokens = _metric_pair(metadata, prefix="load", block_size_tokens=block_size_tokens)
            load_blocks = max(load_blocks, blocks)
            load_tokens = max(load_tokens, tokens)
        elif action == "cache_store":
            blocks, _ = _metric_pair(metadata, prefix="store", block_size_tokens=block_size_tokens)
            store_blocks = max(store_blocks, blocks)
    return {
        "cache_signal": signal,
        "structured_cache_hit_blocks": hit_blocks,
        "structured_cache_hit_tokens": hit_tokens,
        "structured_cache_load_blocks": load_blocks,
        "structured_cache_load_tokens": load_tokens,
        "structured_store_blocks": store_blocks,
        "event_types": sorted(set(event_types)),
        "has_signal": has_signal,
    }


def _summarize_log_cache_evidence(
    rows: list[dict[str, Any]],
    *,
    block_size_tokens: int,
) -> dict[str, Any]:
    hit_blocks = 0
    hit_tokens = 0
    load_blocks = 0
    load_tokens = 0
    event_types: list[str] = []
    signal = "miss"
    has_signal = False
    for row in rows:
        event_type = str(row.get("event_type") or "")
        event_types.append(event_type)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if event_type == "cache_hit":
            has_signal = True
            signal = "hit"
            tokens = _as_int(metadata.get("hit_tokens"))
            hit_tokens = max(hit_tokens, tokens)
            hit_blocks = max(hit_blocks, _blocks_from_tokens(tokens, block_size_tokens))
            load_hint = _as_int(metadata.get("need_to_load_tokens"))
            if load_hint > 0:
                load_tokens = max(load_tokens, load_hint)
                load_blocks = max(load_blocks, _blocks_from_tokens(load_hint, block_size_tokens))
        elif event_type == "cache_load":
            has_signal = True
            if signal != "hit":
                signal = "load_only"
            tokens = _as_int(metadata.get("tokens"))
            load_tokens = max(load_tokens, tokens)
            load_blocks = max(load_blocks, _blocks_from_tokens(tokens, block_size_tokens))
    return {
        "cache_signal": signal,
        "structured_cache_hit_blocks": hit_blocks,
        "structured_cache_hit_tokens": hit_tokens,
        "structured_cache_load_blocks": load_blocks,
        "structured_cache_load_tokens": load_tokens,
        "structured_store_blocks": 0,
        "event_types": sorted(set(event_types)),
        "has_signal": has_signal,
    }


def _metric_pair(metadata: dict[str, Any], *, prefix: str, block_size_tokens: int) -> tuple[int, int]:
    block_ids = _list_ints(metadata.get(f"block_ids_{prefix}") or metadata.get("block_ids"))
    block_count = _as_int(
        metadata.get(f"block_count_{prefix}")
        or metadata.get("block_count")
        or len(block_ids)
    )
    token_count = _as_int(
        metadata.get(f"token_count_{prefix}")
        or metadata.get(f"{prefix}_tokens")
        or metadata.get("token_count")
        or metadata.get("loaded")
        or metadata.get("hit_tokens")
        or metadata.get("tokens")
    )
    if block_count <= 0 and block_ids:
        block_count = len(block_ids)
    if block_count <= 0 and token_count > 0:
        block_count = _blocks_from_tokens(token_count, block_size_tokens)
    if token_count <= 0 and block_count > 0:
        token_count = block_count * block_size_tokens
    return (block_count, token_count)


def _observed_kv_reuse_blocks(evidence: dict[str, Any]) -> int:
    return _first_positive(
        evidence.get("structured_cache_hit_blocks"),
        evidence.get("structured_cache_load_blocks"),
    )


def _observed_kv_reuse_tokens(evidence: dict[str, Any]) -> int:
    return _first_positive(
        evidence.get("structured_cache_hit_tokens"),
        evidence.get("structured_cache_load_tokens"),
    )


def _block_evidence_complete(cache_signal: str, reuse_blocks: int, reuse_tokens: int) -> bool:
    if cache_signal == "miss":
        return reuse_blocks <= 0 and reuse_tokens <= 0
    return reuse_blocks > 0 or reuse_tokens > 0


def _resolve_block_size_tokens(
    analysis_rows: list[Any],
    text_observations: list[dict[str, Any]],
    workload_dir: Path,
) -> int:
    if analysis_rows:
        metadata = getattr(analysis_rows[0], "metadata", {}) or {}
        value = _as_int(metadata.get("block_size_tokens"))
        if value > 0:
            return value
    if text_observations:
        value = _as_int(text_observations[0].get("block_size_tokens"))
        if value > 0:
            return value
    manifest = _read_json(workload_dir / "text_kv_consistency_workload_manifest.json")
    value = _as_int(manifest.get("target_block_size_tokens"))
    return value if value > 0 else DEFAULT_BLOCK_SIZE_TOKENS


def _context_label(context_length: int) -> str:
    if context_length in CONTEXT_LABEL_OVERRIDES:
        return CONTEXT_LABEL_OVERRIDES[context_length]
    if context_length >= 1024:
        return f"ctx{context_length // 1024}k"
    return f"ctx{context_length}"


def _group_by_request(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        grouped.setdefault(request_id, []).append(row)
    return grouped


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    item = json.loads(path.read_text(encoding="utf-8"))
    return item if isinstance(item, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    left = _as_float(numerator)
    right = _as_float(denominator)
    if right in (None, 0.0) or left is None:
        return 0.0
    return left / right


def _delta(current: Any, baseline: Any) -> float | None:
    left = _as_float(current)
    right = _as_float(baseline)
    if left is None or right is None:
        return None
    return left - right


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if value in (None, "", "None"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt_float(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def _first_positive(*values: int) -> int:
    for value in values:
        if int(value or 0) > 0:
            return int(value)
    return 0


def _blocks_from_tokens(token_count: int, block_size_tokens: int) -> int:
    if token_count <= 0 or block_size_tokens <= 0:
        return 0
    return int(math.ceil(token_count / block_size_tokens))


def _list_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if isinstance(item, (int, float)) and int(item) >= 0]
    return []


def _bucket_sort_key(bucket: str) -> int:
    order = {"exact": 0, "sim90": 1, "sim80": 2}
    return order.get(bucket, 99)
