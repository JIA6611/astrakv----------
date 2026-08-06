# Non-GPU Winner Task Report

Status: completed for local non-GPU scope.

Date: 2026-06-08

## Source Reading

Read and checked the task-relevant source before implementation:

- `COMPETITION_TASKS.md`
- `runtime/profile_db.py`
- `runtime/trace_schema.py`
- `runtime/cache_events.py`
- `runtime/failure_recovery.py`
- `scheduler/decision.py`
- `scheduler/object_scheduler.py`
- `scheduler/hints.py`
- `prefetch/scorer.py`
- `moe/expert_predictor.py`
- `scripts/predict_moe_experts.py`
- `scripts/analyze_stress_results.py`
- `scripts/build_trace_store.py`
- `tests/test_moe_expert_predictor.py`
- `tests/test_profile_db.py`

## Tasks Closed Without Coding

These local tasks were already implemented and were closed after source and
test inspection:

| Task | Status | Reason |
| --- | --- | --- |
| Task-003 Real Cache Event Trace Extraction | Closed | Implemented by `runtime/cache_events.py`, `scripts/extract_cache_events.py`, `runtime/trace_schema.py`, and covered by reporting/trace tests. Real event quality still depends on GPU logs. |
| Task-006 Competition Report From Artifacts | Closed | Implemented by `scripts/build_competition_report.py` and `tests/test_competition_report.py`. Final report must be regenerated from real GPU artifacts. |
| Task-007 Profile-Guided Policy Ablation | Closed | Implemented by `scripts/analyze_policy_ablation.py`, `configs/policy_ablation_matrix.yaml`, and `tests/test_policy_ablation.py`. |
| Task-010 Quality / PPL Evaluation | Closed | Implemented by `evaluation/quality.py`, `scripts/evaluate_quality.py`, and `tests/test_quality_evaluation.py`. PPL requires records with `ppl`, `loss`, or `nll`. |
| Task-011 Hidden-State Drift / CKA | Closed | Implemented by `evaluation/hidden_state.py`, `scripts/evaluate_hidden_state_drift.py`, and `tests/test_hidden_state_drift.py`. Hidden-state exports require GPU/model hooks. |
| Task-012 Demo Package | Closed | Implemented by `scripts/build_demo_dashboard.py` and `tests/test_demo_dashboard.py`. Final dashboard must consume real artifacts. |
| Task-017 OS Virtual Memory Demo | Closed | Implemented by `runtime/vm_backend.py`, compatibility wrapper `experiments/vm_demo.py`, `scripts/run_vm_demo.py`, and `tests/test_vm_demo.py`. |

## Task-A Memory Pressure Controller MVP

### Design Analysis

Existing modules accepted a `memory_pressure` value:

- `prefetch/scorer.py`
- `scheduler/decision.py`
- `scheduler/object_scheduler.py`

But there was no independent controller that could derive pressure from real
benchmark/sample/trace artifacts. The missing piece was an OS-style pressure
classification layer that turns GPU memory, CPU RSS, SSD traffic, OOM, and
error-rate evidence into a normalized pressure score and passive runtime hints.

### Modification Plan

Add a read-only controller:

```text
benchmark_results.csv / *_samples.csv / trace_events.jsonl
-> runtime.memory_pressure
-> memory_pressure_decisions.csv
-> memory_pressure_hints.jsonl
-> memory_pressure_report.md
-> memory_pressure_manifest.json
```

The controller emits passive actions:

- `observe`
- `collect_evidence`
- `reduce_prefetch_budget`
- `offload_more`
- `drop_low_reuse`
- `reduce_batch_or_context`

### Impact Scope

Added:

- `runtime/memory_pressure.py`
- `scripts/analyze_memory_pressure.py`
- `tests/test_memory_pressure.py`
- `GPU_VALIDATION_MEMORY_PRESSURE.md`

Modified:

- `runtime/README.md`
- `scripts/README.md`

Not modified:

- vLLM
- LMCache
- CUDA
- `third_party/`
- existing scheduler decision logic
- existing prefetch execution logic

### Acceptance Status

| Criterion | Status |
| --- | --- |
| Reads benchmark artifacts | Done |
| Reads sample CSV artifacts | Done |
| Reads unified memory trace events | Done |
| Classifies low/medium/high/critical/unknown pressure | Done |
| Emits pressure decisions CSV | Done |
| Emits passive scheduler hints JSONL | Done |
| Provides score usable by chunk scorer/load-vs-recompute/unified scheduler CLIs | Done |
| Claims live vLLM/LMCache control | Not claimed; requires GPU adapter validation |

## Task-B Advanced Expert Predictor Beyond Next-N

### Design Analysis

The existing MoE predictor used previous-token experts, layer hotness, and
optional expert load-plan residency. That was useful but weak for award-level
MoE analysis because it could not distinguish predictor strategies or use
longer local route history.

### Modification Plan

Extend the existing predictor without replacing it:

- Keep `next_token` as the default behavior.
- Add `predictor_name` for ablation reporting.
- Add `history_window` to include recent same-request/same-layer observations.
- Add transition statistics from observed token-to-token expert routes.
- Add output fields: `predictor_name`, `window_size`, `transition_score`.
- Keep passive `expert_prefetch` hints compatible with existing consumers.

### Impact Scope

Modified:

- `moe/expert_predictor.py`
- `scripts/predict_moe_experts.py`
- `tests/test_moe_expert_predictor.py`

Added:

- `GPU_VALIDATION_ADVANCED_EXPERT_PREDICTOR.md`

Not modified:

- `moe/expert_loader.py`
- real model routing
- expert weight loading
- vLLM/LMCache internals

### Acceptance Status

| Criterion | Status |
| --- | --- |
| Default predictor remains compatible | Done |
| Adds `history_window` predictor | Done |
| Adds `profile_guided` policy label | Done |
| Adds transition-score signal | Done |
| Emits strategy fields in CSV/report/manifest/hints | Done |
| Tests show non-previous recent history can improve prediction | Done |
| Claims real expert weight prefetch | Not claimed; requires GPU/runtime adapter validation |

## GPU-Shelved Tasks

These are intentionally not completed locally because they require real GPU
execution or backend logs:

- Task-001 Real vLLM Baseline
- Task-002 LMCache CPU Backend Active Proof
- Task-004 AstraKV-W Policy Against Real Backend
- Task-005 Memory-Constrained Stress Benchmark real experiment
- Task-009 Load-vs-Recompute Real Evidence
- Task-013 MoE Expert Activation Trace real MoE run
- Task-014 MoE Expert Predictor Runtime Evidence
- Task-015 INT8 / Quantized Baseline
- Task-016 Multi-Model Evaluation real run

Validation instructions were added in:

- `GPU_VALIDATION_MEMORY_PRESSURE.md`
- `GPU_VALIDATION_ADVANCED_EXPERT_PREDICTOR.md`

## Test Plan

### Static Compile

```powershell
python -m py_compile runtime\memory_pressure.py scripts\analyze_memory_pressure.py tests\test_memory_pressure.py
python -m py_compile moe\expert_predictor.py scripts\predict_moe_experts.py tests\test_moe_expert_predictor.py
```

### Targeted Tests

```powershell
python -m unittest discover -s tests -p test_memory_pressure.py -v
python -m unittest discover -s tests -p test_moe_expert_predictor.py -v
```

### Full Tests

```powershell
python -m unittest discover -s tests -v
```

## Tests Run

| Command | Result |
| --- | --- |
| `python -m py_compile runtime\memory_pressure.py scripts\analyze_memory_pressure.py tests\test_memory_pressure.py` | Passed |
| `python -m unittest discover -s tests -p test_memory_pressure.py -v` | Passed; 5 tests |
| `python -m unittest discover -s tests -v` after Task-A | Passed; 95 tests |
| `python -m py_compile moe\expert_predictor.py scripts\predict_moe_experts.py tests\test_moe_expert_predictor.py` | Passed |
| `python -m unittest discover -s tests -p test_moe_expert_predictor.py -v` | Passed; 6 tests |
| `python -m unittest discover -s tests -v` final | Passed; 96 tests |

## Final Notes

- No third-party runtime code was modified.
- No GPU metric was fabricated.
- KV hit rate, prefetch hit rate, expert hit rate, PPL, and CKA still require
  real GPU/model artifacts before official claims.
- The local non-GPU toolchain now includes pressure-aware analysis and a more
  useful MoE predictor ablation path.
