# AstraKV Three-Route Integration Design

## Objective

Integrate the reviewed AstraKV-W handoff snapshot into an isolated server-side
branch and turn the three routes into one auditable evidence pipeline:

1. immutable workload and reuse observation;
2. causal offline profiling and eviction-policy simulation;
3. paired vLLM/LMCache endpoint experiments with honest performance evidence.

No third-party runtime source is modified. Runtime actions remain unsupported
until a version-locked public API supplies a validated structured receipt.

## Architecture

The workload route accepts immutable QASPER rows or externally exported agent
workflow JSONL. It uses the model tokenizer and chat template to derive token
blocks, hashes those blocks, and calculates reuse only from arrivals already
observed. Dataset metadata, heuristic logs, structured runtime evidence, and
VM PoC evidence use separate schemas and report sections.

The policy route stores raw trace events before derived profile state. Every
decision at arrival `t` records that it consumed history ending at `t - 1`.
LRU, FIFO, AstraKV, and the explicitly offline Belady oracle are compared at
the same capacity; modelled latency is labelled as a proxy.

The endpoint route sends the canonical workload unchanged, records request and
workflow metadata, measures TTFT at the first non-empty assistant content
delta, and preserves the first SSE time for diagnosis. Its sidecar is based on
`pidstat`, `iostat`, `sar`, and available GPU probes. `perf` hardware counters
are not required because the server currently has `perf_event_paranoid=4`.
Nsight Systems is limited to a representative diagnostic timeline per route.

## Experimental Gates

- G0: unit tests and immutable QASPER materialization pass before endpoint use.
- G1: raw/composed workload selection is reportable without mixing their rows.
- G2: token/context preflight and three-request random/grouped endpoint smoke
  pass before the paired matrix.
- G3: only completed paired endpoint artifacts may support endpoint claims.
- G4: no-lookahead policy tests pass before reporting simulator results.
- G5/G6: absent a validated public structured event/action API, reports state
  `insufficient_ground_truth` and `apply_hint=unsupported`.

## Acceptance Criteria

- `main` remains untouched; work is committed only on
  `codex/astrakv-three-route-integration` and is not pushed.
- Every new behavioral rule has a red-green regression test.
- The final Markdown report links each conclusion to a manifest, test, or
  experimental artifact and explicitly identifies unavailable evidence.
