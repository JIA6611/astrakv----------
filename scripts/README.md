# scripts

Utility entry points for AstraKV-W project automation.

All P0 scripts are designed to stay outside third-party runtime internals. They
use configuration files, the OpenAI-compatible HTTP endpoint, server logs, and
CSV/JSONL artifacts.

## Directory Layout

Scripts are grouped by role. Use the categorized paths directly; the old
top-level `scripts/<name>.py` and `scripts/<name>.sh` entry points have been
removed.

| Directory | Contents |
| --- | --- |
| `entrypoints/` | User-facing workflows and environment bootstrap scripts. |
| `benchmark/` | Real endpoint benchmarks, selective prefetch benchmark, workload generation, cache event extraction, and metrics helpers. |
| `launch/` | vLLM and LMCache server launch wrappers. |
| `reporting/` | Report builders, comparison tools, stress analysis, dashboards, plotting, and report-facing advisors. |
| `policy/` | Trace/ProfileDB/chunk-score/load-vs-recompute/object-scheduler pipeline. |
| `vm/` | DGX Spark VM, mmap KV cache, VM demo, and layer-offload PoC runners. |
| `research/` | MoE, hidden-state, quality, and expert-planning tools outside the main dense-model evidence path. |
| `plotting/` | Paper/demo figure builders and experiment visualization scripts. |
| `archive/` | Older wrappers and optional scripts that are not part of the current Linux/DGX Spark default path. |

## Real Endpoint Benchmarks

| Script | Purpose |
| --- | --- |
| `entrypoints/run_architecture_demo.sh` | Fast demo workflow that reuses archived evidence, runs lightweight checks/VM smoke, and generates an architecture-focused execution analysis report. |
| `entrypoints/run_experiment_figures.sh` | Builds experiment figures, context/batch heatmaps, and a figure report from existing evidence, with optional boundary sensitivity reruns. |
| `entrypoints/run_competition_e2e.sh` | One-command competition workflow for official runs, extreme stress, evidence extraction, policy ablation, and final report generation. |
| `entrypoints/run_competition_extended_evidence.sh` | Extended evidence workflow that wraps the E2E run and adds 32K boundary stress, cache-event extraction, OS VM evidence, quality consistency, policy-chain artifacts, final report, and archive output. |
| `benchmark/run_real_benchmark.py` | Runs a real OpenAI-compatible endpoint benchmark and writes CSV, JSONL, Markdown, charts, and per-case samples. |
| `benchmark/dgx_metrics_collector.py` | Collects continuous GPU/CPU/SSD samples for real benchmark cases. |
| `benchmark/generate_workload_suite.py` | Generates fixed competition prompt workloads for short chat, long-context QA, prefix reuse, RAG-like repeated prefix, and memory pressure. |
| `launch/launch_vllm_server.sh` / `archive/launch_vllm_server.ps1` | Starts vLLM-only baseline with AstraKV-W environment variables. PowerShell wrapper is archived because the current DGX Spark path is Linux/bash-first. |
| `launch/launch_lmcache_vllm.sh` / `archive/launch_lmcache_vllm.ps1` | Starts vLLM + LMCache CPU or disk backend. PowerShell wrapper is archived because the current DGX Spark path is Linux/bash-first. |
| `entrypoints/bootstrap_dgx_spark_env.sh` | Creates a local venv and installs Python, PyTorch, vLLM, and optional LMCache for DGX Spark. |

## Analysis And Reporting

| Script | Purpose |
| --- | --- |
| `benchmark/extract_cache_events.py` | Extracts read-only cache events from server logs, request JSONL, and benchmark CSV files. |
| `reporting/build_architecture_demo_report.py` | Builds the Markdown/JSON/CSV outputs for the fast architecture demo from archived evidence artifacts. |
| `research/extract_moe_expert_events.py` | Extracts read-only MoE expert activation events from router logs or JSONL exports. |
| `research/run_moe_route_trace.py` | Runs a local Hugging Face MoE checkpoint and exports router logits as expert-route JSONL for offline MoE evidence. |
| `research/plan_moe_expert_loading.py` | Plans adapter-facing GPU/CPU/SSD/on-demand placement for MoE experts from activation traces. |
| `research/predict_moe_experts.py` | Predicts next-token MoE experts from route traces and emits passive expert prefetch hints. |
| `vm/run_vm_demo.py` | Runs the file-backed mmap virtual-memory demonstration (`astrakv/runtime/vm_backend.py`) for demand loading and software prefetch evidence. |
| `policy/build_trace_store.py` | Normalizes cache events, prefetch events, and per-case sample CSVs into one `astra-trace-v1` JSONL store. |
| `policy/build_profile_db.py` | Builds a reusable JSON ProfileDB and Markdown report from unified trace events. |
| `reporting/build_competition_report.py` | Builds a competition-grade aggregate report, manifest, and artifact inventory from benchmark, policy, scheduler, workload, and quality artifacts. |
| `reporting/build_demo_dashboard.py` | Builds a static HTML dashboard, data JSON, and manifest from archived competition artifacts. |
| `policy/score_chunks.py` | Scores ProfileDB chunks and recommends `prefetch`, `keep`, `offload`, or `drop` actions. |
| `policy/analyze_policy_ablation.py` | Aggregates benchmark, prefetch, and chunk-score artifacts into a policy ablation CSV and Markdown report. |
| `policy/plan_partial_kv_load.py` | Generates adapter-facing partial KV load plans by layer and token span from chunk metadata. |
| `policy/decide_load_vs_recompute.py` | Produces passive `load`, `recompute`, `defer`, or `drop` scheduler hints from ProfileDB and optional partial KV plans. |
| `policy/run_unified_object_scheduler.py` | Merges ProfileDB, chunk scores, load/recompute decisions, and GPU budget into passive unified object scheduling hints. |
| `reporting/analyze_memory_pressure.py` | Converts benchmark/sample/trace memory artifacts into passive memory-pressure decisions and scheduler hints. |
| `research/evaluate_quality.py` | Compares baseline vs variant outputs for exact/normalized match, token divergence, char divergence, and optional PPL fields. |
| `research/evaluate_hidden_state_drift.py` | Compares exported baseline vs variant hidden states with CKA, cosine similarity, MSE, L2 drift, and max absolute difference. |
| `reporting/compare_real_runs.py` | Compares vLLM-only, LMCache CPU, LMCache disk, or other real benchmark result directories. |
| `reporting/analyze_multi_model_evaluation.py` | Summarizes archived dense/MoE/long-context benchmark runs and reports per-model baseline deltas. |
| `reporting/analyze_failure_recovery.py` | Classifies archived benchmark/prefetch/cache/scheduler failures and emits passive fallback hints. |
| `reporting/analyze_stress_results.py` | Summarizes constrained-memory stress runs into capacity, error, OOM, memory, and disk metrics. |
| `reporting/plot_benchmarks.py` | Creates matplotlib PNG charts from benchmark CSV files. |
| `plotting/build_experiment_figures.py` | Builds paper/demo figures and context × batch heatmaps from E2E, boundary, cache-event, policy, quality, and VM artifacts. |

## Selective Prefetch

| Script | Purpose |
| --- | --- |
| `benchmark/run_selective_prefetch_real.py` | Runs endpoint-level real selective prefetch with no-prefetch vs AstraKV-W-prefetch comparison. |
| `benchmark/benchmark_runner.py` | Runs the runtime-independent synthetic benchmark and Selective KV Prefetch MVP. Keep these results separate from real vLLM/LMCache claims. |

## Local Metrics Helpers

| Script | Purpose |
| --- | --- |
| `benchmark/metrics_collector.py` | Lightweight local process memory and scratch SSD helper used by synthetic benchmarks. |

## Retention / Cleanup Guidance

This section is a practical cleanup map. It does not mean these files should be
deleted automatically; archive first if a future experiment may still need them.

### Keep For Current Competition Path

These files are used by the current DGX Spark real-endpoint workflow, extended
evidence workflow, final report, or policy-chain evidence. Do not remove them
while reproducing the current submission.

| Group | Files |
| --- | --- |
| Main workflow | `entrypoints/run_competition_e2e.sh`, `entrypoints/run_competition_extended_evidence.sh`, `entrypoints/bootstrap_dgx_spark_env.sh` |
| Server launch | `launch/launch_vllm_server.sh`, `launch/launch_lmcache_vllm.sh` |
| Real endpoint benchmark | `benchmark/run_real_benchmark.py`, `benchmark/run_selective_prefetch_real.py`, `benchmark/dgx_metrics_collector.py` |
| Stress/report analysis | `reporting/analyze_stress_results.py`, `reporting/compare_real_runs.py`, `reporting/build_competition_report.py`, `benchmark/extract_cache_events.py`, `research/evaluate_quality.py` |
| VM evidence | `entrypoints/run_dgx_spark_validation.sh`, `vm/run_dgx_spark_vm_evidence.py`, `vm/run_mmap_kv_cache.py` |
| Policy chain | `policy/build_trace_store.py`, `policy/build_profile_db.py`, `policy/score_chunks.py`, `policy/decide_load_vs_recompute.py`, `policy/run_unified_object_scheduler.py`, `policy/analyze_policy_ablation.py` |
| Workload support | `benchmark/generate_workload_suite.py` |
| Figure generation | `entrypoints/run_experiment_figures.sh`, `plotting/build_experiment_figures.py` |

### Archive Candidates If Keeping Only The Current Linux/DGX Spark Path

These files are useful for demos, research branches, or older experiments, but
they are not required by the current one-command DGX Spark evidence path.

| Candidate | Why it can be archived |
| --- | --- |
| `archive/launch_vllm_server.ps1`, `archive/launch_lmcache_vllm.ps1` | Windows/PowerShell launch wrappers. Kept only for Windows reproduction. |
| `benchmark/benchmark_runner.py`, `benchmark/metrics_collector.py` | Synthetic benchmark path. Current claims use real vLLM/LMCache endpoint artifacts instead. |
| `vm/run_vm_demo.py` | Older VM demo wrapper. Current VM evidence uses `vm/run_mmap_kv_cache.py` and `vm/run_dgx_spark_vm_evidence.py`. |
| `reporting/build_demo_dashboard.py` | Optional static dashboard, not required for final evidence generation. |
| `reporting/plot_benchmarks.py` | Optional per-benchmark plotting helper. `plotting/build_experiment_figures.py` now builds paper/demo figures. |
| `archive/run_ablation.sh` | Earlier ablation runner. Current ablation is generated through E2E/extended evidence plus `analyze_policy_ablation.py`. |
| `archive/run_edge_sim_tests.sh`, `archive/setup_edge_sim.sh` | Optional sudo/cgroup edge simulation. Current DGX Spark path does not enable cgroups by default. |

### Research-Only Tools To Keep Separate From Main Claims

These are not dead code, but they target experiments outside the current dense
Qwen2.5-7B DGX Spark submission. Move them to an archive or research folder only
if the repo needs a lean competition-only surface.

| Research area | Files |
| --- | --- |
| MoE expert planning | `research/run_moe_route_trace.py`, `research/extract_moe_expert_events.py`, `research/plan_moe_expert_loading.py`, `research/predict_moe_experts.py` |
| Hidden-state validation | `research/evaluate_hidden_state_drift.py` |
| Multi-model summary | `reporting/analyze_multi_model_evaluation.py` |
| Failure and memory-pressure advisors | `reporting/analyze_failure_recovery.py`, `reporting/analyze_memory_pressure.py` |
| Partial KV planning | `policy/plan_partial_kv_load.py` |
| Layer/weight offload PoC | `vm/run_layer_offload_poc.py` |

## Offline MoE Route Trace Evidence

If the DGX host cannot access the internet, download a small Hugging Face MoE
checkpoint on another machine and copy the full model directory to the DGX host,
for example `models/Qwen1.5-MoE-A2.7B` or another Mixtral-compatible small MoE
checkpoint. Then run the route trace script with local files only:

```bash
python scripts/research/run_moe_route_trace.py \
  --model models/Qwen1.5-MoE-A2.7B \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --output-dir results/moe_route_trace
```

`--local-files-only` is the default behavior; use `--allow-download` only on a
networked machine. The output `moe_route_events.jsonl` can then be fed into the
existing MoE evidence chain:

```bash
python scripts/research/extract_moe_expert_events.py \
  --events-jsonl results/moe_route_trace/moe_route_events.jsonl \
  --output-dir results/moe_expert_evidence/events

python scripts/research/plan_moe_expert_loading.py \
  --expert-summary results/moe_expert_evidence/events/moe_expert_summary.csv \
  --output-dir results/moe_expert_evidence/load_plan

python scripts/research/predict_moe_experts.py \
  --moe-events results/moe_expert_evidence/events/moe_expert_events.jsonl \
  --expert-load-plan results/moe_expert_evidence/load_plan/moe_expert_load_plan.csv \
  --output-dir results/moe_expert_evidence/prediction
```

This path supports expert activation analysis and passive expert-placement /
prefetch hints. It does not prove serving-time expert weight movement unless a
runtime adapter or serving log shows actual expert loading, migration, or hits.

Suggested cleanup flow:

```bash
mkdir -p scripts/archive
git mv <archive-candidate> scripts/archive/
```

After moving scripts, update any references in `scripts/README.md`,
`docs/guides/competition_test_flow_cn.md`, and
`docs/guides/repository_organization_cn.md`, then run the shell and reporting
tests listed below.

## P0 Command Map

Recommended entry points:

```bash
bash scripts/entrypoints/run_architecture_demo.sh --skip-install
bash scripts/entrypoints/run_experiment_figures.sh --skip-install
bash scripts/entrypoints/run_competition_extended_evidence.sh --skip-install --continue-on-failure
bash scripts/entrypoints/run_competition_e2e.sh --skip-install
bash scripts/entrypoints/run_competition_e2e.sh --only smoke --skip-install
bash scripts/entrypoints/run_competition_e2e.sh --only extreme --gpu-util-extreme 0.40 --skip-install
```

The e2e script does not require sudo by default. cgroup memory limiting is optional and should be applied externally or through `--with-cgroup` in environments that provide a cgroup runner. Extreme stress failures are retained as boundary evidence. On DGX Spark, the primary resource metrics are process RSS, GPU utilization, disk IO, sample count, and vLLM startup KV-cache capacity. `gpu_memory_peak_mb` is kept only as a compatibility field and is filled only when `nvidia-smi` or NVML can actually sample it.

The architecture demo is the fastest answer for "show me how the current
architecture works." It reuses archived E2E/boundary evidence, runs lightweight
reporting tests and an mmap smoke, then writes
`results/architecture_demo_<timestamp>/demo_report.md`. Add `--with-live-smoke`
only when the machine has time to start a real vLLM/LMCache smoke run.

Extended evidence workflow examples:

```bash
# Reuse an existing E2E run and build an extended report.
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only report \
  --output-root results/extended_from_existing \
  --existing-e2e-root results/competition_e2e_20260624_012006 \
  --skip-install

# Run only the 32K boundary stress stage.
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only boundary \
  --gpu-util-boundary 0.25 \
  --boundary-max-model-len 32768 \
  --boundary-context-lengths "16384 24576 32768" \
  --boundary-batch-sizes "1 2 4 8" \
  --boundary-output-tokens 128 \
  --boundary-repeat 1 \
  --output-root results/extended_g025_ctx32k_test \
  --skip-install
```

The extended workflow writes staged artifacts under `results/extended_evidence_<timestamp>/`: `01_e2e/`, `02_boundary_32k/`, `03_cache_events/`, `04_os_vm/`, `05_quality/`, `06_policy_chain/`, `07_final_report/`, and `archive/`.

Current boundary reproduction commands:

```bash
# 32K feasible boundary: expected to complete vLLM, LMCache CPU, and LMCache Disk.
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only boundary \
  --gpu-util-boundary 0.16 \
  --boundary-max-model-len 32768 \
  --boundary-context-lengths "24576 32768" \
  --boundary-batch-sizes "4 8 12 16" \
  --boundary-output-tokens 256 \
  --boundary-repeat 1 \
  --boundary-timeout 2400 \
  --output-root results/extended_g016_ctx32k_b16_out256 \
  --skip-install \
  --continue-on-failure

# 32K lower bound: expected to fail startup because KV cache is too small.
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only boundary \
  --gpu-util-boundary 0.15 \
  --boundary-max-model-len 32768 \
  --boundary-context-lengths "24576 32768" \
  --boundary-batch-sizes "4 8 12 16" \
  --boundary-output-tokens 256 \
  --boundary-repeat 1 \
  --boundary-timeout 2400 \
  --output-root results/extended_g015_ctx32k_b16_out256 \
  --skip-install \
  --continue-on-failure
```

Cleanup and file-retention guidance is documented in
`docs/guides/repository_organization_cn.md`.

| P0 task | Primary command |
| --- | --- |
| P0-1 | `python scripts/benchmark/run_real_benchmark.py --config configs/dgx_spark_vllm_qwen7b.yaml --output-dir results/p0_1_vllm` |
| P0-2 | `bash scripts/launch/launch_lmcache_vllm.sh cpu` then `python scripts/benchmark/run_real_benchmark.py --config configs/dgx_spark_lmcache_cpu.yaml --output-dir results/p0_2_lmcache_cpu` |
| P0-3 | `bash scripts/launch/launch_lmcache_vllm.sh disk` then `python scripts/benchmark/run_real_benchmark.py --config configs/dgx_spark_lmcache_disk.yaml --output-dir results/p0_3_lmcache_disk` |
| P0-5 | `python scripts/benchmark/extract_cache_events.py --server-log <server.log> --output-dir results/cache_events` |
| P0-6 | `python scripts/reporting/compare_real_runs.py --run vllm=<dir> --run lmcache_cpu=<dir> --run lmcache_disk=<dir> --output-dir results/comparison` |
| P0-7 | `python scripts/reporting/analyze_stress_results.py --run vllm=<dir> --run lmcache_cpu=<dir> --run lmcache_disk=<dir> --output-dir results/stress_analysis` |
| P0-8 | `python scripts/benchmark/run_selective_prefetch_real.py --config configs/astrakv_real_selective_prefetch.yaml --output-dir results/p0_8_selective_prefetch` |

See `docs/reproduction.md` for the full official run order.
