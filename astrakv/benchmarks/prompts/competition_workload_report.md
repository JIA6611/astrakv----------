# Competition Workload Suite Report

Generated: 2026-06-08T17:32:06

## Parameters

- Long context tokens: `4096`
- Memory pressure tokens: `8192`
- Repeated prefix tokens: `2048`

## Summary

- Schema: `astra-workload-suite-v1`
- Case count: `6`
- Max context length: `8192`
- Total expected output tokens: `768`

### Workload Types

| workload type | count |
| --- | ---: |
| long_context_qa | 1 |
| memory_pressure | 1 |
| prefix_reuse | 2 |
| rag_repeated_prefix | 1 |
| short_chat | 1 |

### Cases

| sample | type | context | output | repeat group | tags |
| --- | --- | ---: | ---: | --- | --- |
| short_chat_001 | short_chat | 256 | 96 |  | short, latency, ttft, tpot |
| long_context_qa_001 | long_context_qa | 4096 | 160 |  | long_context, qa, memory_tiering |
| prefix_reuse_001_a | prefix_reuse | 2048 | 128 | prefix_reuse_001 | prefix_reuse, kv_hit |
| prefix_reuse_001_b | prefix_reuse | 2048 | 128 | prefix_reuse_001 | prefix_reuse, kv_hit, prefetch |
| rag_repeated_prefix_001 | rag_repeated_prefix | 2048 | 128 | rag_docs_001 | rag, repeated_prefix, cache_store |
| memory_pressure_001 | memory_pressure | 8192 | 128 |  | memory_pressure, stress, oom |

## Usage

Quality evaluation endpoint mode:

```bash
python scripts/evaluate_quality.py \
  --prompts benchmarks\prompts\competition_workload_suite.jsonl \
  --baseline-base-url http://127.0.0.1:8000 \
  --variant-base-url http://127.0.0.1:8001 \
  --temperature 0.0 \
  --top-p 1.0 \
  --output-dir results/p1_8_quality_suite
```

## Artifacts

- `benchmarks\prompts\competition_workload_suite.jsonl`
- `benchmarks\prompts\competition_workload_manifest.json`
- `competition_workload_report.md`
