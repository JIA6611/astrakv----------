"""Competition workload suite generation.

The suite is a fixed set of prompt records for real endpoint and quality
evaluation runs. It does not execute a benchmark by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


WORKLOAD_SCHEMA_VERSION = "astra-workload-suite-v1"


@dataclass(frozen=True, slots=True)
class WorkloadCase:
    sample_id: str
    workload_type: str
    prompt: str
    context_length: int
    expected_output_tokens: int
    repeat_group: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": WORKLOAD_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "workload_type": self.workload_type,
            "prompt": self.prompt,
            "context_length": self.context_length,
            "expected_output_tokens": self.expected_output_tokens,
            "repeat_group": self.repeat_group,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def build_competition_workload_suite(
    *,
    long_context_tokens: int = 4096,
    memory_pressure_tokens: int = 8192,
    repeated_prefix_tokens: int = 2048,
) -> list[WorkloadCase]:
    shared_prefix = synthetic_context("Shared KV prefix for reuse analysis.", repeated_prefix_tokens)
    rag_prefix = "\n\n".join(
        [
            synthetic_context("Document A: GPU memory pressure and KV cache.", 512),
            synthetic_context("Document B: CPU tier stores colder KV blocks.", 512),
            synthetic_context("Document C: SSD tier increases capacity with IO cost.", 512),
        ]
    )
    return [
        WorkloadCase(
            sample_id="short_chat_001",
            workload_type="short_chat",
            prompt=(
                "Explain why KV cache matters for LLM serving in one concise paragraph. "
                "Mention TTFT and TPOT."
            ),
            context_length=256,
            expected_output_tokens=96,
            tags=("short", "latency", "ttft", "tpot"),
        ),
        WorkloadCase(
            sample_id="long_context_qa_001",
            workload_type="long_context_qa",
            prompt=(
                synthetic_context("Long context about memory-constrained LLM inference.", long_context_tokens)
                + "\n\nQuestion: What are the tradeoffs between GPU, CPU, and SSD KV tiers?"
            ),
            context_length=long_context_tokens,
            expected_output_tokens=160,
            tags=("long_context", "qa", "memory_tiering"),
        ),
        WorkloadCase(
            sample_id="prefix_reuse_001_a",
            workload_type="prefix_reuse",
            prompt=shared_prefix + "\n\nTask: Summarize the prefix for a systems audience.",
            context_length=repeated_prefix_tokens,
            expected_output_tokens=128,
            repeat_group="prefix_reuse_001",
            tags=("prefix_reuse", "kv_hit"),
        ),
        WorkloadCase(
            sample_id="prefix_reuse_001_b",
            workload_type="prefix_reuse",
            prompt=shared_prefix + "\n\nTask: Extract three optimization opportunities from the prefix.",
            context_length=repeated_prefix_tokens,
            expected_output_tokens=128,
            repeat_group="prefix_reuse_001",
            tags=("prefix_reuse", "kv_hit", "prefetch"),
        ),
        WorkloadCase(
            sample_id="rag_repeated_prefix_001",
            workload_type="rag_repeated_prefix",
            prompt=rag_prefix + "\n\nQuestion: Which document best supports SSD offload and why?",
            context_length=2048,
            expected_output_tokens=128,
            repeat_group="rag_docs_001",
            tags=("rag", "repeated_prefix", "cache_store"),
        ),
        WorkloadCase(
            sample_id="memory_pressure_001",
            workload_type="memory_pressure",
            prompt=(
                synthetic_context("High memory pressure long prompt for stress testing.", memory_pressure_tokens)
                + "\n\nTask: Give a brief risk analysis for serving this request under limited GPU memory."
            ),
            context_length=memory_pressure_tokens,
            expected_output_tokens=128,
            tags=("memory_pressure", "stress", "oom"),
        ),
    ]


def summarize_workload_cases(cases: list[WorkloadCase]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    repeat_groups: dict[str, int] = {}
    max_context = 0
    total_expected_output = 0
    for case in cases:
        type_counts[case.workload_type] = type_counts.get(case.workload_type, 0) + 1
        if case.repeat_group:
            repeat_groups[case.repeat_group] = repeat_groups.get(case.repeat_group, 0) + 1
        max_context = max(max_context, case.context_length)
        total_expected_output += case.expected_output_tokens
    return {
        "schema": WORKLOAD_SCHEMA_VERSION,
        "case_count": len(cases),
        "type_counts": dict(sorted(type_counts.items())),
        "repeat_group_counts": dict(sorted(repeat_groups.items())),
        "max_context_length": max_context,
        "total_expected_output_tokens": total_expected_output,
    }


def synthetic_context(seed: str, approx_tokens: int) -> str:
    words = [
        "AstraKV",
        "runtime",
        "memory",
        "tier",
        "cache",
        "prefetch",
        "load",
        "recompute",
        "latency",
        "throughput",
    ]
    body = " ".join(words[index % len(words)] for index in range(max(1, approx_tokens - 32)))
    return f"{seed}\n{body}"
