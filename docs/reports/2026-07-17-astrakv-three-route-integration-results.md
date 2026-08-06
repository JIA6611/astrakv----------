# AstraKV Three-Route Integration Results

## Scope

This report records the server-side integration performed on branch
`codex/astrakv-three-route-integration`. It combines the reviewed handoff
snapshot with a tokenizer-backed workflow observer. No vLLM, LMCache, CUDA, or
other third-party runtime source was modified.

## Verification

| Check | Result | Evidence |
| --- | --- | --- |
| Isolated baseline | Passed | `135` unit tests before integration |
| Handoff plus observer | Passed | `171` unit tests, including 3 observer regressions |
| Diff hygiene | Passed | `git diff --check` after LF normalization |
| Runtime diagnostic | Completed | `results/codex_three_route/runtime_diagnostic/` |
| vLLM endpoint matrix | Not run | ports `8000`, `8001`, and `8002` returned HTTP `000` |

The four system-Python baseline import failures were environment-only
(`numpy` unavailable outside the project virtual environment). The server's
existing `.venv` was used read-only for all valid results.

## Route 1: Immutable Workload Observation

The integration adds `WorkflowTraceRow` and `ReuseObservation` contracts. The
observer tokenizes the original messages through the configured chat template,
uses SHA-256 of canonical token blocks, and only matches blocks seen before
the current arrival. It rejects duplicate workflow/subtask identities,
duplicate or non-monotonic arrivals, invalid roles, and empty messages.

The regression fixture proves that an identical second request obtains two
historical reused tokens while a later non-identical request receives credit
only for its previously seen generation-prompt block. This is
`modeled_dataset_metadata`, not a runtime cache-hit claim.

## Route 2: Causal Offline Policy

The handoff integration includes immutable trace/profile artifacts, causal
offline eviction simulation, LRU/FIFO/AstraKV/Belady comparisons, and the
three-independent-workload safety gate. Unit coverage verifies no-lookahead
behavior, labels Belady as an offline oracle, separates proxy timing from
backend timing, and keeps unsupported backend actions from executing.

These results support policy diagnosis under declared capacities. They do not
prove that a vLLM block was evicted or that an AstraKV policy controlled a
third-party runtime.

## Route 3: Endpoint and Profiling Evidence

The endpoint benchmark, request/workload contract, structured evidence
normalization, launch configuration, and low-privilege sidecar diagnostics are
integrated. The diagnostic found `pidstat`, `iostat`, `sar`, `perf`,
`bpftrace`, and `nsys` installed; `dcgmi` and `ncu` are unavailable. The host
has `perf_event_paranoid=4`, so hardware counters are not admissible primary
evidence. Use `pidstat`/`iostat`/`sar`/GPU probes for paired runs and reserve
Nsight Systems for one representative timeline per route.

No endpoint was listening on the prescribed local ports. Consequently there
are no measured TTFT, TPOT, throughput, cache counter, quality, or disk-I/O
comparisons in this run. The absence is intentional and must not be replaced
with synthetic values or inferred from logs.

## Conclusion

The code-level integration and offline evidence chain pass their regression
suite. The three-route documentation remains consistent when claims are
bounded as follows: reuse opportunity is modelled, eviction policy outcomes
are offline/proxy evidence, and endpoint performance/runtime behavior remains
`not_available` until paired vLLM and LMCache runs complete with manifests and
sidecar artifacts. Runtime action stays `unsupported` without a verified,
version-locked public structured action receipt.
