# AstraKV-W Competition Tasks

Status: task breakdown generated from current project audit.

Scope: this file lists the remaining work for the 2026 National College Student Computer System Capability Competition, OS Design Contest, OS Function Challenge, problem "Runtime Optimization of LLM Inference for the Memory Constraint System".

Current judgement: AstraKV-W is a promising prototype with real vLLM endpoint benchmarking, synthetic Selective KV Prefetch MVP, runtime-agnostic KV metadata, and clear integration boundaries. The main missing piece is a real vLLM/LMCache runtime optimization loop.

## Priority Definition

| Priority | Meaning | Goal |
| --- | --- | --- |
| P0 | Must finish to participate credibly | Produce a real baseline/variant system with measurable memory-tier behavior. |
| P1 | Improves competitiveness | Turn the prototype into a convincing optimization system with ablations and profile-guided decisions. |
| P2 | Award-level bonus work | Add MoE, OS-style virtual memory evidence, richer accuracy analysis, and polished demo assets. |

## P0: Must Finish To Participate

### P0-1 Real vLLM Baseline Matrix

| Field | Content |
| --- | --- |
| Task name | Real vLLM baseline matrix |
| Module | `scripts/run_real_benchmark.py`, `configs/dgx_spark_vllm_qwen7b.yaml`, `results/` |
| Dependencies | Existing vLLM server launch wrapper; model available locally or downloadable; GPU environment ready |
| Estimated code size | 100-200 LOC, mostly config/report refinements |
| Estimated time | 1-2 days |
| Acceptance criteria | Run context lengths 512/1024/2048/4096/8192, batch sizes 1/2/4, repeat >= 5. Produce CSV, JSONL, Markdown report, charts, and per-case samples. Success rate must be visible. |
| Contest requirement | Analyze LLM inference runtime behavior; guarantee inference performance |
| Experiment metrics | TTFT, TPOT, latency p50/p95, throughput, GPU memory, CPU memory, success rate |

### P0-2 LMCache CPU Backend Baseline

| Field | Content |
| --- | --- |
| Task name | vLLM + LMCache CPU-tier baseline |
| Module | `scripts/launch_lmcache_vllm.*`, `configs/dgx_spark_lmcache_cpu.yaml`, LMCache integration config |
| Dependencies | P0-1; verified LMCache version and vLLM integration flags |
| Estimated code size | 150-300 LOC, mostly launcher/config/log parsing |
| Estimated time | 2-3 days |
| Acceptance criteria | Endpoint starts with LMCache CPU backend enabled. Logs prove CPU backend is active. Same benchmark matrix can run against this backend. Report compares vLLM-only vs LMCache CPU. |
| Contest requirement | Memory tiering, swapping, lower physical GPU memory footprint |
| Experiment metrics | GPU memory, CPU memory, TTFT, TPOT, latency p95, throughput, cache load/store count if available |

### P0-3 LMCache Disk Backend Baseline

| Field | Content |
| --- | --- |
| Task name | vLLM + LMCache disk-tier baseline |
| Module | `scripts/launch_lmcache_vllm.*`, `configs/dgx_spark_lmcache_disk.yaml`, LMCache disk path config |
| Dependencies | P0-1; verified LMCache integration; writable SSD cache directory |
| Estimated code size | 150-300 LOC |
| Estimated time | 2-3 days |
| Acceptance criteria | Endpoint starts with disk backend enabled. Logs and disk traffic show disk-tier KV activity. Same benchmark matrix can run. Report compares vLLM-only vs LMCache disk. |
| Contest requirement | Memory tiering, swapping, SSD tier, memory-constrained inference |
| Experiment metrics | SSD read/write MB, SSD read/write MB/s, GPU memory, CPU memory, TTFT, TPOT, throughput |

### P0-4 Real Metrics Sampling Integration

| Field | Content |
| --- | --- |
| Task name | Continuous real metrics sampling for benchmark cases |
| Module | `scripts/run_real_benchmark.py`, `scripts/dgx_metrics_collector.py`, report generator |
| Dependencies | P0-1 |
| Estimated code size | 150-250 LOC |
| Estimated time | 1-2 days |
| Acceptance criteria | Every real benchmark case produces `samples/<case>_samples.csv`. Report includes process RSS peak, GPU utilization peak, disk read/write deltas, sample count, and startup-level KV cache capacity evidence. |
| Contest requirement | Analyze memory behavior; lower runtime physical memory usage |
| Experiment metrics | Process RSS peak, GPU utilization, SSD traffic, sample count, startup KV cache capacity; GPU memory only when exposed by the platform |

### P0-5 Real Cache Event Trace Adapter

| Field | Content |
| --- | --- |
| Task name | Read-only LMCache/vLLM cache event adapter |
| Module | New `runtime/adapters_lmcache.py` or `runtime/adapters_vllm.py`, `kv_cache/metadata.py`, benchmark trace output |
| Dependencies | P0-2 or P0-3; stable source of LMCache/vLLM cache logs, metrics, or connector events |
| Estimated code size | 300-600 LOC |
| Estimated time | 3-5 days |
| Acceptance criteria | Produce `cache_events.jsonl` with request id, cache key or chunk id, event type, tier, bytes, start/end time, status. At least hit/miss/load/store/offload events are represented when available. |
| Contest requirement | KV Cache analysis, on-demand loading, swapping, memory tiering |
| Experiment metrics | KV hit rate, cache miss count, load count, store count, offload bytes, load latency |

### P0-6 Unified Result Comparator

| Field | Content |
| --- | --- |
| Task name | Baseline vs variant comparator |
| Module | `scripts/plot_benchmarks.py`, new comparison/report utility, `results/` |
| Dependencies | P0-1, P0-2, P0-3 |
| Estimated code size | 250-500 LOC |
| Estimated time | 2-3 days |
| Acceptance criteria | Given multiple result directories, generate one comparison Markdown and charts for vLLM-only, LMCache CPU, LMCache disk. Include deltas for memory, latency, throughput, and success rate. |
| Contest requirement | Guarantee inference performance while reducing memory footprint |
| Experiment metrics | TTFT delta, TPOT delta, throughput delta, process RSS change, SSD traffic, success rate, KV capacity boundary |

### P0-7 Memory-Constrained Stress Benchmark

| Field | Content |
| --- | --- |
| Task name | Long-context and constrained-memory stress benchmark |
| Module | Benchmark configs, launch wrappers, result comparator |
| Dependencies | P0-1, P0-2, P0-3, P0-4 |
| Estimated code size | 100-200 LOC |
| Estimated time | 2-4 days |
| Acceptance criteria | Report maximum successful context length and batch size under constrained KV cache headroom. Include OOM/error rate, latency p95, RSS, disk IO, and startup KV cache capacity. |
| Contest requirement | Runtime optimization for memory-constrained systems |
| Experiment metrics | Max context, max batch, OOM rate, success rate, process RSS, disk IO, GPU utilization, latency p95 |

### P0-8 Minimal Real Selective Prefetch Adapter

| Field | Content |
| --- | --- |
| Task name | AstraKV-W selective prefetch on real backend |
| Module | `prefetch/`, `runtime/`, LMCache/vLLM adapter layer |
| Dependencies | P0-5; confirmed public API or safe extension point for triggering cache load/prefetch |
| Estimated code size | 500-900 LOC |
| Estimated time | 5-8 days |
| Acceptance criteria | AstraKV-W policy submits real prefetch/load requests or scheduler hints consumed by the backend. Real run reports prefetch submitted/completed/hit/waste. A no-prefetch vs AstraKV-W-prefetch comparison is available. |
| Contest requirement | Prefetch, async loading, overlap IO and compute, runtime optimization |
| Experiment metrics | Prefetch hit rate, prefetch waste rate, TTFT, TPOT, latency p95, throughput, SSD traffic, GPU memory |

### P0-9 Reproduction Runbook

| Field | Content |
| --- | --- |
| Task name | Competition reproduction runbook |
| Module | `docs/reproduction.md`, `docs/dgx_spark_setup.md`, scripts README |
| Dependencies | P0-1 through P0-8 |
| Estimated code size | 0-100 LOC plus documentation |
| Estimated time | 1 day |
| Acceptance criteria | A clean user can follow commands to launch baseline, launch variants, run benchmarks, generate comparison reports, and locate logs/results. All commands match actual CLI options. |
| Contest requirement | Competition-ready system, reproducibility |
| Experiment metrics | All P0 metrics; command success/failure status |

### P0-10 Basic Unit And Smoke Tests

| Field | Content |
| --- | --- |
| Task name | Core unit tests and smoke tests |
| Module | `kv_cache/`, `prefetch/`, `offload/`, `runtime/`, `scripts/` |
| Dependencies | Existing skeletons and P0 adapter interfaces |
| Estimated code size | 300-600 LOC |
| Estimated time | 2-3 days |
| Acceptance criteria | Tests cover metadata validation, block table operations, placement state transitions, synthetic prefetch metrics, comparator parsing, and CLI smoke paths. |
| Contest requirement | Engineering completeness and stable demo |
| Experiment metrics | Test pass rate, smoke benchmark success |

## P1: Improves Competitiveness

### P1-1 Trace Schema And Trace Store

| Field | Content |
| --- | --- |
| Task name | Unified runtime trace schema |
| Module | New `runtime/trace_schema.py`, `results/traces/`, documentation |
| Dependencies | P0-5 |
| Estimated code size | 300-500 LOC |
| Estimated time | 2-3 days |
| Acceptance criteria | KV, prefetch, placement, memory sample, request, and error events use one schema. Trace validation tool catches missing fields. |
| Contest requirement | Analyze LLM inference memory access behavior |
| Experiment metrics | Trace coverage, event counts, KV hit rate, tier residence time |

### P1-2 ProfileDB

| Field | Content |
| --- | --- |
| Task name | Profile-guided runtime database |
| Module | New `runtime/profile_db.py`, config and report integration |
| Dependencies | P1-1 |
| Estimated code size | 400-700 LOC |
| Estimated time | 3-5 days |
| Acceptance criteria | Store per-workload chunk reuse, load latency, tier latency, memory pressure, and prefetch outcome. Profiles can be reused across runs. |
| Contest requirement | Profile-guided runtime, on-demand loading, prefetch |
| Experiment metrics | Reuse frequency, load latency, prefetch hit rate, tier hit rate |

### P1-3 Chunk Scorer

| Field | Content |
| --- | --- |
| Task name | KV chunk scoring policy |
| Module | New `prefetch/scorer.py`, `runtime/profile_db.py` |
| Dependencies | P1-2 |
| Estimated code size | 250-500 LOC |
| Estimated time | 2-4 days |
| Acceptance criteria | Each chunk gets a score from reuse probability, deadline, size, load latency, and memory pressure. Report explains why chunks were prefetched, kept, offloaded, or dropped. |
| Contest requirement | Selective KV prefetch, memory tiering, runtime optimization |
| Experiment metrics | Prefetch hit rate, prefetch waste rate, GPU memory, latency p95 |

### P1-4 Policy Ablation Framework

| Field | Content |
| --- | --- |
| Task name | Prefetch and placement policy ablation |
| Module | Benchmark configs, policy registry, comparator |
| Dependencies | P0-8, P1-3 |
| Estimated code size | 300-600 LOC |
| Estimated time | 3-5 days |
| Acceptance criteria | Compare no-prefetch, LMCache default, next-N, LRU-aware, reuse-aware, deadline-aware, and AstraKV-W combined policy under same workloads. |
| Contest requirement | Prove runtime optimization and innovation |
| Experiment metrics | TTFT, TPOT, throughput, GPU memory reduction, prefetch hit rate, waste rate, SSD traffic |

### P1-5 Partial KV Load MVP

| Field | Content |
| --- | --- |
| Task name | Partial KV load by chunk/layer/token span |
| Module | `kv_cache/metadata.py`, adapter layer, prefetch/load policy |
| Dependencies | P0-5, P0-8, P1-1 |
| Estimated code size | 600-1200 LOC |
| Estimated time | 1-2 weeks |
| Acceptance criteria | Runtime can request and record partial KV ranges instead of whole request cache objects. Report shows bytes saved and accuracy/performance impact. |
| Contest requirement | Partial KV Load, lower physical memory, on-demand loading |
| Experiment metrics | Loaded bytes, skipped bytes, GPU memory, TTFT, TPOT, PPL or output consistency |

### P1-6 Load-vs-Recompute Decision

| Field | Content |
| --- | --- |
| Task name | Load-vs-recompute scheduler decision |
| Module | New `scheduler/decision.py`, `runtime/profile_db.py`, adapter hints |
| Dependencies | P1-2, P1-5 |
| Estimated code size | 500-900 LOC |
| Estimated time | 1 week |
| Acceptance criteria | For selected chunks, system chooses load, recompute, defer, or drop based on estimated IO latency, compute cost, and memory pressure. Decisions are logged and measurable. |
| Contest requirement | Runtime optimization, virtual memory style demand handling |
| Experiment metrics | Load latency, recompute latency, GPU memory, TPOT, throughput, decision accuracy |

### P1-7 Accuracy And Quality Evaluation

| Field | Content |
| --- | --- |
| Task name | PPL and output consistency evaluation |
| Module | New evaluation script, benchmark report |
| Dependencies | P1-5 or any approximate/partial loading policy |
| Estimated code size | 300-700 LOC |
| Estimated time | 3-5 days |
| Acceptance criteria | Report PPL or task accuracy on fixed eval set where possible. For serving-only endpoints, report deterministic output consistency and token-level divergence. |
| Contest requirement | Guarantee inference quality while optimizing memory |
| Experiment metrics | PPL, output match rate, token divergence, latency, memory |

### P1-8 Benchmark Workload Suite

| Field | Content |
| --- | --- |
| Task name | Realistic workload suite |
| Module | `benchmarks/`, `configs/`, prompt datasets |
| Dependencies | P0-1 |
| Estimated code size | 200-500 LOC |
| Estimated time | 2-4 days |
| Acceptance criteria | Include short chat, long-context QA, prefix-reuse workload, RAG-like repeated prefix workload, and memory-pressure workload. |
| Contest requirement | Analyze LLM inference behavior under different runtime patterns |
| Experiment metrics | TTFT, TPOT, throughput, KV hit rate, prefetch hit rate, GPU memory, SSD traffic |

### P1-9 Unified Object Scheduler MVP

| Field | Content |
| --- | --- |
| Task name | Unified object scheduler for KV objects |
| Module | `runtime/object_manager.py`, `scheduler/`, `offload/`, `prefetch/` |
| Dependencies | P1-2, P1-3, P1-6 |
| Estimated code size | 800-1500 LOC |
| Estimated time | 1-2 weeks |
| Acceptance criteria | One scheduler arbitrates GPU budget across KV chunks using priority, reuse, deadline, and tier latency. Decisions produce placement and prefetch actions. |
| Contest requirement | Unified Object Scheduler, memory tiering, swapping, prefetch |
| Experiment metrics | GPU memory, eviction count, prefetch hit rate, OOM rate, latency p95 |

### P1-10 Report Quality Upgrade

| Field | Content |
| --- | --- |
| Task name | Competition-grade reports and figures |
| Module | Report generator, plotting scripts, docs |
| Dependencies | P0-6, P1-4 |
| Estimated code size | 300-600 LOC |
| Estimated time | 2-4 days |
| Acceptance criteria | Reports include environment, git commit, command lines, p50/p95, error bars, memory curves, cache event summaries, ablation table, and limitations. |
| Contest requirement | Reproducible competition-ready system |
| Experiment metrics | All core metrics plus confidence/error information |

## P2: Award-Level Bonus Work

### P2-1 MoE Expert Activation Trace

| Field | Content |
| --- | --- |
| Task name | MoE expert activation tracing |
| Module | New MoE adapter/profiler, model-specific hooks |
| Dependencies | Stable real benchmark loop; MoE model selected |
| Estimated code size | 600-1200 LOC |
| Estimated time | 1-2 weeks |
| Acceptance criteria | For a MoE model, collect token-to-expert routing, expert frequency, expert hotness, and memory footprint. |
| Contest requirement | MoE Expert access behavior analysis |
| Experiment metrics | Expert hit rate, expert activation frequency, expert memory, latency |

### P2-2 MoE Expert Selective Loader

| Field | Content |
| --- | --- |
| Task name | Selective expert weight loading |
| Module | MoE runtime adapter, expert cache manager |
| Dependencies | P2-1 |
| Estimated code size | 1000-2000 LOC |
| Estimated time | 2-4 weeks |
| Acceptance criteria | Expert weights can be placed across GPU/CPU/SSD or loaded on demand. Report shows memory savings and latency impact. |
| Contest requirement | MoE Expert selective loading, parameter loading, memory tiering |
| Experiment metrics | Expert hit rate, GPU memory, CPU memory, SSD traffic, TTFT, TPOT |

### P2-3 Expert Predictor

| Field | Content |
| --- | --- |
| Task name | Router-aware expert predictor |
| Module | MoE profiler, predictor, prefetch policy |
| Dependencies | P2-1, P2-2 |
| Estimated code size | 500-1000 LOC |
| Estimated time | 1-2 weeks |
| Acceptance criteria | Predictor estimates next experts from routing history or prompt profile. Prefetch hit and waste rates are reported. |
| Contest requirement | MoE Expert prefetch and async loading |
| Experiment metrics | Expert prefetch hit rate, expert waste rate, latency p95, memory |

### P2-4 OS Virtual Memory Demonstration

| Field | Content |
| --- | --- |
| Task name | mmap/UVM/page-fault style memory experiment |
| Module | Experiment scripts, docs, optional small runtime demo |
| Dependencies | P0 real baseline; isolated environment |
| Estimated code size | 300-800 LOC |
| Estimated time | 4-7 days |
| Acceptance criteria | Demonstrate page-like demand loading, page fault or mmap behavior, madvise/UVM-style comparison, or file-backed cache behavior with measurable IO/memory traces. |
| Contest requirement | Virtual memory, on-demand loading, swapping |
| Experiment metrics | Page fault count if available, load latency, RSS, GPU memory, disk traffic |

### P2-5 CKA And Hidden-State Drift

| Field | Content |
| --- | --- |
| Task name | CKA/hidden-state drift evaluation |
| Module | Evaluation scripts, model hooks |
| Dependencies | P1-5 or approximate KV policy |
| Estimated code size | 500-1000 LOC |
| Estimated time | 1-2 weeks |
| Acceptance criteria | Compare baseline vs partial-load/recompute hidden states using CKA or related similarity metrics. |
| Contest requirement | Preserve inference quality under runtime optimization |
| Experiment metrics | CKA, hidden-state drift, PPL, output divergence |

### P2-6 Demo Dashboard

| Field | Content |
| --- | --- |
| Task name | Interactive competition dashboard |
| Module | Optional web dashboard or notebook |
| Dependencies | P0-6, P1-10 |
| Estimated code size | 500-1200 LOC |
| Estimated time | 4-7 days |
| Acceptance criteria | Dashboard shows runs, metric curves, memory tiers, cache events, and baseline/variant deltas. It must use archived results and not be required for correctness. |
| Contest requirement | Competition-ready demonstration |
| Experiment metrics | Visualization of all core metrics |

### P2-7 Multi-Model Evaluation

| Field | Content |
| --- | --- |
| Task name | Multi-model and multi-workload evaluation |
| Module | Configs, benchmark matrix, reports |
| Dependencies | P0 and P1 stable path |
| Estimated code size | 100-300 LOC |
| Estimated time | 3-7 days plus run time |
| Acceptance criteria | Run at least one dense model and one MoE or long-context model. Report when each optimization helps or hurts. |
| Contest requirement | Generality and robustness |
| Experiment metrics | TTFT, TPOT, throughput, GPU memory, KV hit rate, expert hit rate if applicable |

### P2-8 Failure Recovery And Degradation Modes

| Field | Content |
| --- | --- |
| Task name | Runtime failure recovery and fallback modes |
| Module | Runtime adapter, scheduler, launch scripts |
| Dependencies | P0-8, P1-9 |
| Estimated code size | 400-900 LOC |
| Estimated time | 4-7 days |
| Acceptance criteria | If CPU/disk load, prefetch, or profile lookup fails, system falls back to safe baseline mode and logs the cause. |
| Contest requirement | Stable memory-constrained system |
| Experiment metrics | Failure count, fallback count, success rate, latency impact |

## Development Order Roadmap

### Week1

Goal: produce credible real baseline evidence.

1. Finish P0-1 real vLLM baseline matrix.
2. Finish P0-4 continuous metrics sampling integration.
3. Start P0-2 LMCache CPU backend validation.
4. Clean reproduction commands for the baseline path.

Week1 deliverable:

- `vLLM-only` real benchmark report with repeat matrix.
- Per-case GPU/CPU/disk samples.
- Known-good server launch and benchmark commands.

### Week2

Goal: prove multi-tier memory is real.

1. Finish P0-2 LMCache CPU baseline.
2. Finish P0-3 LMCache disk baseline.
3. Finish P0-6 baseline/variant comparator.
4. Start P0-5 real cache event trace adapter.

Week2 deliverable:

- `vLLM-only` vs `LMCache-CPU` vs `LMCache-disk` comparison report.
- Evidence from logs and metrics that CPU/disk tiers are active.
- Initial `cache_events.jsonl` format.

### Week3

Goal: make AstraKV-W affect runtime behavior.

1. Finish P0-5 real cache event trace adapter.
2. Finish P0-8 minimal real selective prefetch adapter.
3. Run P0-7 memory-constrained stress benchmark.
4. Add P0-10 basic unit and smoke tests.

Week3 deliverable:

- `no-prefetch/default` vs `AstraKV-W selective prefetch` real comparison.
- Max context, OOM rate, memory peak, and latency p95 report.
- Passing core tests.

### Week4

Goal: improve competitiveness and polish the submission.

1. Finish P0-9 reproduction runbook.
2. Finish P1-1 trace schema and trace store.
3. Finish P1-3 chunk scorer or a minimal version.
4. Finish P1-4 policy ablation framework.
5. Finish P1-10 report quality upgrade.

Week4 deliverable:

- Competition-ready report with baseline/variant/ablation results.
- Reproducible demo package.
- Clear limitations and next-step story for P2 work.

## Shortest Participation Path

The shortest credible path avoids MoE, CUDA kernels, FlashAttention modifications, and a full scheduler rewrite.

Required tasks:

1. P0-1 Real vLLM Baseline Matrix.
2. P0-2 LMCache CPU Backend Baseline.
3. P0-3 LMCache Disk Backend Baseline.
4. P0-4 Real Metrics Sampling Integration.
5. P0-6 Unified Result Comparator.
6. P0-7 Memory-Constrained Stress Benchmark.
7. P0-9 Reproduction Runbook.

Minimum story:

```text
vLLM-only baseline
-> vLLM + LMCache CPU tier
-> vLLM + LMCache disk tier
-> memory-constrained stress benchmark
-> report GPU memory, CPU memory, SSD traffic, TTFT, TPOT, throughput, OOM rate
```

Minimum acceptable claim:

> AstraKV-W establishes a reproducible memory-constrained LLM serving benchmark and validates real multi-tier KV offload behavior using vLLM and LMCache.

Do not claim real Selective KV Prefetch unless P0-8 is finished.

## Optimal Award Path

The award path needs one original optimization beyond wrapping vLLM and LMCache.

Required tasks:

1. All P0 tasks.
2. P1-1 Trace Schema And Trace Store.
3. P1-2 ProfileDB.
4. P1-3 Chunk Scorer.
5. P1-4 Policy Ablation Framework.
6. P1-5 Partial KV Load MVP, if integration time allows.
7. P1-6 Load-vs-Recompute Decision, if Partial KV Load is stable.
8. P1-7 Accuracy And Quality Evaluation.
9. P1-10 Report Quality Upgrade.
10. One P2 item, preferably P2-4 OS Virtual Memory Demonstration or P2-1 MoE Expert Activation Trace.

Optimal story:

```text
Real vLLM baseline
-> real LMCache CPU/disk tiers
-> real cache event trace
-> profile-guided chunk scoring
-> AstraKV-W selective prefetch and placement
-> partial load or load-vs-recompute
-> ablation and accuracy validation
-> OS-style virtual memory evidence or MoE expert trace
```

Optimal claim:

> AstraKV-W is a profile-guided runtime layer for memory-constrained LLM serving. It observes real KV/cache events, predicts high-value KV objects, selectively prefetches or loads partial KV data across GPU/CPU/SSD tiers, and reports the memory-performance-quality tradeoff with reproducible experiments.
