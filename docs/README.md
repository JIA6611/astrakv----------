# AstraKV-W Documentation

## License

Project source code is released under the MIT License. Documentation, reports,
presentation materials, and demo-video materials are released under the
Creative Commons Attribution-ShareAlike 4.0 International License
(CC BY-SA 4.0), unless a file states otherwise.

## Structure

| Directory | Purpose |
|-----------|---------|
| `architecture/` | System architecture, module design, interface boundaries |
| `analysis/` | Codebase analysis, third-party analysis, audit reports |
| `guides/` | Reproduction guides, GPU setup, testing workflows |
| `planning/` | Competition tasks, implementation plans, progress reports |

## Architecture (`docs/architecture/`)

| Document | Description |
|----------|-------------|
| `runtime_architecture.md` | Runtime system architecture and adapter design |
| `prefetch_design.md` | Selective KV Prefetch MVP design, data flow, metrics |
| `interface_boundaries.md` | Module interface boundaries and integration contracts |
| `reusable_modules.md` | Reusable module inventory and avoid-modifying list |
| `architecture_layers_cn.md` | Chinese: Architecture layering explanation |
| `current_core_code_adjustments_cn.md` | Chinese: Current core code adjustments |
| `architecture_diagram.svg` | Current architecture diagram: real endpoint orchestration, evidence pipeline, advisory policy chain |

## Analysis (`docs/analysis/`)

| Document | Description |
|----------|-------------|
| `codebase_analysis.md` | Full codebase structural and quality analysis |
| `third_party_analysis.md` | Third-party system analysis (vLLM, LMCache, SGLang, etc.) |
| `project_audit_report.md` | Comprehensive project audit and readiness report |
| `contest_solution_plan.md` | Competition solution design plan |

## Guides (`docs/guides/`)

| Document | Description |
|----------|-------------|
| `reproduction.md` | Authoritative P0 competition reproduction guide |
| `competition_test_flow_cn.md` | Chinese: current competition E2E and extended evidence workflow |
| `repository_organization_cn.md` | Chinese: repository cleanup candidates, file categories, and reproducible run checklist |
| `dgx_spark_setup.md` | DGX/GPU hardware setup and environment guide |
| `real_gpu_baseline.md` | Real GPU baseline measurement and interpretation |
| `GPU_TESTING_WORKFLOW_CN.md` | Chinese: GPU testing and validation workflow |
| `clone_validation.md` | Third-party repository clone validation |
| `license_and_submission_notice.md` | License scope, third-party boundary, AI disclosure, and submission note |

## Planning (`docs/planning/`)

| Document | Description |
|----------|-------------|
| `COMPETITION_TASKS.md` | Competition task breakdown and priority map |
| `IMPLEMENTATION_PLAN.md` | Detailed implementation plan (P0/P1/P2 tasks) |
| `TASK_REPORT.md` | Latest task completion status report |
| `TEACHER_PROGRESS_BRIEF_CN.md` | Chinese: Teacher-facing progress brief |

## Recommended Reading Order

1. `analysis/project_audit_report.md` — understand project scope and readiness
2. `architecture/runtime_architecture.md` — understand system design
3. `architecture/prefetch_design.md` — understand core prefetch mechanism
4. `guides/dgx_spark_setup.md` — hardware environment requirements
5. `guides/competition_test_flow_cn.md` — current competition experiment workflow
6. `guides/repository_organization_cn.md` — cleanup and final evidence packaging

For current competition validation, `guides/competition_test_flow_cn.md` and
`guides/repository_organization_cn.md` are the recommended sources. The older
`guides/reproduction.md` remains useful for manual P0 reproduction.
