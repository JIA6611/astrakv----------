# Three-Route Scope Audit

> Status: Do not merge this branch into `main` as a completed three-route implementation.

## Authoritative Sources

The implementation contract is the approved plan in
`docs/superpowers/plans/2026-07-16-astrakv-three-route-integration.md`.
The committed Task 1 package is immutable input data, not proof that Task 1
execution is complete. The handoff snapshot is only a partial implementation
source.

## Corrected Findings

- Task 1 data is now present at `datasets/task1_qasper/`: 22 files, both
  prompt JSONLs have 200 records, and the transferred SHA-256 values equal the
  local source values.
- The current task-one adapter accepts ZIP input, while the supplied server
  input is a directory package. It must support the immutable directory
  package, emit canonical random/grouped workflow traces, and verify all
  source hashes before this route is complete.
- Missing Task 1-4 deliverables include the observation CLI/config, task-one
  workflow observation test, replay JSONL adapter/fixture/test, pilot matrix,
  reuse-opportunity report, and dataset protocol.
- The current offline modules are not yet verified as a causal replay from
  Task 1 observer output with the required `profile_history_end_index=t-1`
  contract.
- The current endpoint configuration still defaults to synthetic prompts,
  Qwen2.5, and 8K context. It must be replaced by canonical QASPER input,
  Qwen3-8B at 32K with explicit fallback, request identity fields, first
  non-empty-content TTFT, and cold/warm pairing.
- Missing Task 7-8 deliverables include the evidence-class regression test,
  three-route report builder/test, and execution guide. No backend-performance
  or runtime-action claim is currently permitted.

## Required Next Commits

1. Task-one directory adapter, canonical workflow materialization, observer
   CLI/config, and immutable-data tests.
2. Replay adapter and pre-registered reuse pilot/report.
3. Endpoint contract correction and sidecar profiling tests.
4. Causal profile replay, evidence gate, three-route report, and only then
   permitted real endpoint experiments.
