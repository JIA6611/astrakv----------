# Real GPU Baseline Result

Status: completed smoke baseline.

Date: 2026-06-03

## Purpose

This document explains the real GPU baseline result stored under
`results/local_smoke_test/`. The baseline is a smoke benchmark against an
already-running vLLM OpenAI-compatible endpoint. It verifies that the remote GPU
inference environment, HTTP serving path, streaming response parser, and basic
measurement pipeline are working.

This result should be used as a real-system reference point before adding
AstraKV-W integration work such as KV cache prefetching, offloading, or
scheduler-aware policies.

## Artifact Location

```text
results/local_smoke_test/
|-- benchmark_report.md
|-- benchmark_results.csv
`-- request_results.jsonl
```

Artifact roles:

- `benchmark_report.md`: human-readable summary of the run.
- `benchmark_results.csv`: case-level aggregate metrics for comparison and
  plotting.
- `request_results.jsonl`: raw per-request records used to build the aggregate
  table.

## Benchmark Source

The result was produced by `scripts/run_real_benchmark.py`.

The script sends streaming chat completion requests to a vLLM
OpenAI-compatible endpoint. It does not import, modify, or activate AstraKV-W
runtime, scheduler, prefetch, offload, LMCache, or third-party runtime code.

Because of that boundary, this result is a real vLLM endpoint baseline, not an
optimization result for AstraKV-W algorithms.

## Configuration

The recorded run used the following configuration:

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Endpoint | `http://127.0.0.1:8000` |
| Backend label | `vllm_openai_endpoint` |
| Context lengths | `512`, `1024` |
| Batch size | `1` |
| Output token target | `64` |
| Repeat | `1` |
| Response mode | streaming chat completions |
| GPU probe | `nvidia-smi` |

The prompt context length is approximate. The benchmark script builds synthetic
text from common words so that the script does not need a tokenizer dependency.
The `output_tokens` value is a maximum generation target, not a guarantee that
the model will generate exactly that many tokens.

## Observed Results

| case | success | observed output tokens | TTFT ms | TPOT ms | latency ms | throughput tok/s | GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bs1_ctx512_out64` | 1/1 | 46 | 95.369 | 16.656 | 844.892 | 54.445 | 22134.0 |
| `bs1_ctx1024_out64` | 1/1 | 42 | 84.822 | 16.861 | 776.123 | 54.115 | 22134.0 |

All requests completed successfully. No HTTP, endpoint, parsing, or runtime
errors were recorded.

## Metric Meaning

| Metric | Meaning |
| --- | --- |
| `TTFT ms` | Time from request start to the first streamed content delta. |
| `TPOT ms` | Time per output token after the first token, computed from the streamed response duration. |
| `latency ms` | End-to-end request time from request start to stream completion. |
| `throughput tok/s` | Observed completion tokens divided by end-to-end request duration. |
| `observed output tokens` | Actual completion token count reported by endpoint usage, or estimated from streamed deltas when usage is unavailable. |
| `GPU MB` | Total visible GPU memory used according to request-time `nvidia-smi` samples. |
| `CPU MB` | RSS of the benchmark client process, not the vLLM server process. |

## Interpretation

This baseline shows that the real GPU serving path is healthy:

- The vLLM endpoint accepted and completed both benchmark requests.
- Streaming response handling worked.
- `nvidia-smi` was available from the benchmark environment.
- The endpoint delivered roughly `54 tok/s` end-to-end throughput for these
  short single-request cases.
- Decode speed was roughly `16.7 ms/token`, or about `59-60 token/s`.
- GPU memory stayed at about `22134 MB` before and after the measured requests.

The `512` and `1024` context cases should not be interpreted as a reliable
context-length scaling comparison. Each case was run only once, and the model
generated different actual output lengths: `46` tokens for the `512` case and
`42` tokens for the `1024` case. The shorter total latency in the `1024` case is
mostly explained by its shorter completion.

## Valid Uses

Use this result for:

- Proving that the remote GPU environment and vLLM endpoint are reachable.
- Proving that real streaming inference measurement works.
- Establishing an initial real-system baseline for single-request Qwen2.5-7B
  serving.
- Checking future experiment runs for obvious regressions in TTFT, TPOT,
  throughput, endpoint reliability, or GPU memory level.
- Documenting that the project has moved beyond purely synthetic benchmark
  infrastructure and has a real endpoint smoke result.

Appropriate report wording:

```text
We first validated the real-GPU measurement pipeline using a vLLM
OpenAI-compatible endpoint serving Qwen/Qwen2.5-7B-Instruct. The smoke test
completed successfully for both 512-token and 1024-token synthetic prompts under
batch size 1, with no request failures. The endpoint achieved approximately
54 output tokens/s end-to-end throughput and about 16.7 ms/token decode TPOT.
GPU memory sampled via nvidia-smi remained stable at about 22.1 GB before and
after the requests. This run is used as an environment and serving baseline, not
as evidence of AstraKV-W prefetch or offload optimization gains.
```

## Invalid Uses

Do not use this result to claim:

- Selective KV prefetch improves performance.
- KV offload reduces GPU memory.
- Scheduler hints improve throughput or latency.
- AstraKV-W changes are active in the real runtime.
- Long-context performance trends are established.
- High-concurrency behavior has been measured.
- Peak GPU memory during the request is fully captured.

Those claims require an explicit baseline-versus-variant experiment, repeated
measurements, and instrumentation that observes the relevant runtime behavior.

## Current Limitations

- Only two requests were measured.
- Each case used `repeat=1`, so there are no p50, p95, or variance estimates.
- Batch size was only `1`.
- Context lengths were only `512` and `1024`.
- The prompt length is approximate rather than tokenizer-verified.
- The model did not generate the full `64` target tokens.
- GPU memory was sampled before and after each request, not continuously.
- CPU memory records the benchmark client, not the vLLM server.
- No AstraKV-W prefetch, offload, scheduler, or runtime adapter path was active.

## Recommended Next Experiments

To turn this smoke baseline into a stronger evaluation, run a larger matrix:

```text
context_lengths: 512, 1024, 2048, 4096, 8192
batch_sizes: 1, 2, 4, 8
output_tokens: 128 or higher
repeat: 5 to 10
```

Improve measurement quality by:

- Using prompts that force a more stable completion length.
- Recording p50 and p95 TTFT, TPOT, latency, and throughput.
- Sampling server-side GPU memory continuously during the request window.
- Recording vLLM server logs or metrics for actual prompt tokens, completion
  tokens, and scheduling behavior.
- Separating client-process memory from server-process memory.
- Keeping the same model, GPU, vLLM version, endpoint configuration, and prompt
  set when comparing baseline and optimized variants.

## Comparison Plan

Future optimization experiments should compare at least two conditions:

| Condition | Meaning |
| --- | --- |
| Baseline | Unmodified vLLM endpoint under the same model, prompt set, and hardware. |
| Variant | vLLM or runtime path with the proposed KV prefetch, offload, or scheduler-aware change active. |

The comparison should report:

- Request success rate and failure modes.
- TTFT p50 and p95.
- TPOT p50 and p95.
- End-to-end latency p50 and p95.
- Output tokens per second.
- Process RSS peak, GPU utilization, and optional SSD read/write metrics when offload is involved.
- Startup-level vLLM KV cache capacity evidence when the platform does not expose case-level GPU memory.
- OOM rate or maximum successful context/concurrency setting.

Only after that comparison should the project claim optimization impact.
