# benchmarks

Synthetic benchmark and workload helpers for AstraKV-W.

The current competition performance path is the real endpoint workflow in
`scripts/run_real_benchmark.py`, `scripts/run_competition_e2e.sh`, and
`scripts/run_competition_extended_evidence.sh`. This package remains useful for
runtime-independent policy simulation, prompt/workload generation, and
multi-model result summarization, but synthetic benchmark numbers should not be
reported as real vLLM/LMCache performance.

## Supported Metrics

- TTFT: time to first token.
- TPOT: time per output token.
- Throughput: generated output tokens per second.
- GPU memory: best-effort `nvidia-smi` probe when available; on DGX Spark this
  can be unavailable and should not be used as a required conclusion metric.
- Estimated GPU KV memory: logical KV block residency from the MVP.
- CPU memory: process RSS peak.
- SSD read/write: temporary scratch-file bandwidth.
- KV cache hit rate.
- Prefetch hit rate.
- Prefetch waste rate.
- TTFT change versus no-prefetch.
- TPOT change versus no-prefetch.
- GPU KV memory reduction versus full GPU residency.

## Layout

- `configs/baseline.yaml`: synthetic baseline benchmark matrix.
- `configs/selective_prefetch_mvp.yaml`: no-prefetch vs Selective KV Prefetch MVP comparison.
- `configs/real_workload_suite.yaml`: fixed real workload suite metadata for
  endpoint quality and memory-behavior validation.
- `configs/multi_model_evaluation_matrix.yaml`: recommended dense, MoE, and
  long-context evaluation matrix for P2-7 official GPU runs.
- `prompts/competition_workload_suite.jsonl`: fixed prompt suite covering short
  chat, long-context QA, prefix reuse, RAG-like repeated prefix, and memory
  pressure cases.
- `workload_suite.py`: generator primitives for fixed competition prompt cases.
- `scripts/benchmark_runner.py`: synthetic benchmark runner.
- `scripts/generate_workload_suite.py`: regenerates the fixed prompt suite,
  manifest, and report.
- `scripts/analyze_multi_model_evaluation.py`: summarizes archived
  multi-model benchmark CSVs and reports per-model baseline deltas.
- `scripts/metrics_collector.py`: synthetic metrics helpers.
- `scripts/plot_benchmarks.py`: chart generator.
- `scripts/run_vm_demo.py`: standalone OS-style file-backed `mmap`
  demonstration for demand loading and software prefetch evidence.
- `results/`: generated CSV, Markdown, and PNG outputs.

## Run

```bash
python scripts/benchmark_runner.py --config astrakv/benchmarks/configs/baseline.yaml --output-dir results/synthetic_baseline
python scripts/benchmark_runner.py --config astrakv/benchmarks/configs/selective_prefetch_mvp.yaml --output-dir results/synthetic_astrakv_mvp
python scripts/generate_workload_suite.py --output-dir astrakv/benchmarks/prompts
python scripts/run_vm_demo.py --output-dir results/vm_demo
```

The runner writes a timestamped directory under `results/` containing:

- `benchmark_results.csv`
- `benchmark_report.md`
- `benchmark_config.json`
- `charts/*.png`

## Selective KV Prefetch MVP

The MVP benchmark compares `synthetic_no_prefetch` with
`selective_prefetch_mvp` on the same synthetic decode trace. It reports
prefetch hit rate, prefetch waste rate, TTFT change, TPOT change, and estimated
GPU KV memory reduction.

## Competition Workload Suite

The fixed workload suite provides real endpoint prompts for:

- Short chat.
- Long-context QA.
- Prefix-reuse KV cache behavior.
- RAG-like repeated prefix behavior.
- Memory-pressure stress behavior.

Use generated prompts with quality evaluation when you have baseline and variant
`request_results.jsonl` files:

```bash
python scripts/evaluate_quality.py \
  --baseline-jsonl results/<baseline_run>/request_results.jsonl \
  --variant-jsonl results/<variant_run>/request_results.jsonl \
  --output-dir results/quality_eval
```

## Current Boundaries

- No scheduler implementation.
- No vLLM core changes.
- No third-party source modifications.
- No CUDA kernel implementation.
- Synthetic results are separate from the real vLLM/LMCache endpoint evidence.
