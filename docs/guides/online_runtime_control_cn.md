# Online Runtime Control Operator Guide

## Purpose and Claim Boundary

This guide operates the experimental AstraKV control plane for the exact
supported target: vLLM `0.23.0` with LMCache `0.4.7`. It is intentionally
fail-closed and does not authorize a production backend action by default.

Use the following labels precisely:

| Label | Required evidence | Permitted claim |
| --- | --- | --- |
| `endpoint_validated` | Successful endpoint benchmark and request records | The endpoint and measurement path work |
| `hook_observed` | Request context, structured binding/event records, and version preflight | The Hook observed a runtime lifecycle under a linked request |
| `dry_run_action_service` | Protected action-service test, command ledger, and terminal receipt | Command admission and receipt handling were tested |
| `verified_real_action` | Controlled server E2E action plus eligible paired evidence | A real backend action occurred and its comparison is auditable |

`verified_real_action` is permitted only for a run whose archived evidence
passes the paired-evidence validator. The owner-only Unix-domain admission path
creates the reservation, command, proof, and terminal receipt inside the
EngineCore process; callers never receive a reusable reservation lease.

## Non-Negotiable LMCache 0.4.7 Limitation

The supported LMCache `StorageManager.batched_put` path is nonblocking. A
return from `wait_for_save` and the connector callbacks do not prove a durable
terminal store completion. The exact LocalDiskBackend 0.4.7
`batched_submit_put_task(..., on_complete_callback=...)` callback is the sole
version-locked completion signal: it occurs after disk write, index insertion,
and put-task removal. Consequently:

- a store submission is an observation, not a completed/stable cache object;
- logs, a cache hit, an HTTP 2xx result, and a generated command are not action
  receipts;
- `LMCache047ActionEndpoint` remains disabled until that callback completes
  the exact submitted lease and the request release is observed;
- do not report a live-server action until the controlled E2E and paired
  evidence requirements below are met.

The protected `DROP` service is not a public production launch interface. Two
experimental live-dispatch initiators are supported: the owner-only `0600`
Unix-domain socket for a manual controlled action, and the opt-in online policy
worker described below. Both remain inside the EngineCore-owned action service,
and both perform runtime-gate admission, reservation, command creation, proof
issuance, and dispatch.

## Opt-In Online Policy

Set both variables in the EngineCore environment to enable automatic policy
execution; an absent or rejected offline-gate record keeps policy execution
disabled:

```bash
export ASTRAKV_ENABLE_ONLINE_POLICY=true
export ASTRAKV_ONLINE_OFFLINE_GATE_PATH=results/<run_id>/offline_gate.json
```

The host writes raw Hook artifacts and registers bindings on the Hook callback,
then places lifecycle events in a bounded FIFO queue. A daemon worker performs
online profile ingestion, `drop` proposal, and the protected `remove()` call.
Therefore a slow LocalDisk `remove()` cannot block a vLLM request lifecycle.
The default queue capacity is 64 events and a released object has a 30-second
dispatch deadline. A full queue or an expired/blocked decision produces a
structured `runtime_events_raw.jsonl` rejection with `policy_rejection_reason`; it does not
wait for capacity and it does not retry after the deadline.

For an automatic-policy E2E, do not invoke `admit-drop` over the Unix socket.
The resulting `astrakv_runtime_commands.jsonl` entry must have a command ID beginning with
`<run_id>:online-`, followed by one terminal receipt and one `drop` completion
event carrying the same `command_id`. This distinguishes policy execution from
a manual owner-UDS action.

## Preflight and Context Linkage

The DGX environment lock, measured host manifest, and supported backend tuple
are maintained in `constraints/dgx-py312-cu130-20260720.txt`,
`docs/environments/dgx_runtime_manifest_20260720.md`, and
`docs/runtime_backend_compatibility.md`. They are required for reproducing the
validated target, but do not replace the per-run preflight below.

Before a run, create one immutable `backend_capabilities.json` record from observed values.
Execution eligibility requires all of the following:

1. `run_id` and a loopback-only Hook identity.
2. vLLM `0.23.0` and LMCache `0.4.7`.
3. Connector `lmcache-vllm-v1` at version `0.4.7`.
4. Valid installation evidence bound to the endpoint and session.
5. Supported `drop` action, `prefix` object level, and observed binding
   generation.

Publish a loopback-authenticated `RuntimeRequestContext` before the endpoint
request and retain its matching runtime receipt. A lifecycle event is linked
only when it carries the same `run_id`, `request_id`, `object_key`,
`object_level`, opaque backend identity, `binding_id`, and
`binding_generation`. Treat missing, deferred, stale, or reused context as
unlinked observation; it is ineligible for dispatch.

## Dry-Run Validation

The following local tests exercise the version lock, contracts, protected
action service, admission gate, breaker, request handoff, and paired-evidence
validator without claiming a live server action:

```bash
python -m unittest \
  tests.test_backend_capabilities \
  tests.test_backend_binding_registry \
  tests.test_lmcache047_runtime_patch \
  tests.test_lmcache047_action_service \
  tests.test_runtime_execution_gate \
  tests.test_circuit_breaker \
  tests.test_request_context_handoff \
  tests.test_paired_run_validation
```

An accepted dry-run command must be current-generation, inactive, unpinned,
within byte/concurrency/window budgets, not expired, allowed by preflight, and
permitted by the circuit breaker. The action service writes fsync-backed
`astrakv_runtime_commands.jsonl` and terminal `runtime_command_receipts.jsonl`; replaying the same command is
idempotent and a conflicting command ID is rejected.

## Artifact Layout

For every endpoint/Hook experiment, preserve a distinct run directory:

```text
results/<run_id>/
|-- experiment_manifest.json
|-- backend_capabilities.json
|-- backend_binding_events.jsonl
|-- runtime_events_raw.jsonl
|-- astrakv_runtime_commands.jsonl
|-- runtime_command_receipts.jsonl
|-- runtime_structured_events.jsonl
|-- online_profile_checkpoint.json
|-- trace_events.jsonl
|-- request_results.jsonl
|-- benchmark_results.csv
`-- quality_results.csv
```

For `online_control`, each declared artifact must be content-hashed in
`experiment_manifest.json`. Keep baseline and variant inputs identical except
for the explicitly evaluated runtime condition: pair ID, workload hash, matrix
hash, environment hash, model/tokenizer/dtype/quantization, case coverage, and
sample IDs must all match.

Validate the archived pair with:

```bash
python scripts/reporting/compare_real_runs.py \
  --run baseline=results/<baseline_run> \
  --run variant=results/<variant_run> \
  --output-dir results/<comparison>
```

`results/<comparison>/paired_run_manifest.json` must contain `eligible: true`.
For an online-control claim, the validator requires matching binding IDs,
one-to-one command/terminal-receipt IDs, compatible preflight, and complete
request-to-trace coverage. An ineligible manifest blocks the claim.

## Automatic Artifact Export

Pass the EngineCore-owned runtime state directory to the benchmark runner instead
of copying six online artifacts by hand:

```bash
python scripts/benchmark/run_real_benchmark.py \
  --output-dir results/pair/variant \
  --runtime-state-dir results/pair/variant-state \
  --claim-scope online_control \
  --pair-id online-control-pair-1 --pair-role variant
```

The exporter leaves the raw state directory unchanged and writes a validator-ready
snapshot into the benchmark output using the final artifact contract:
`backend_capabilities.json`, `backend_binding_events.jsonl`,
`runtime_events_raw.jsonl`, `astrakv_runtime_commands.jsonl`,
`runtime_command_receipts.jsonl`, `runtime_structured_events.jsonl`, and
`online_profile_checkpoint.json`, plus a derived trace generated from
`request_results.jsonl`. Do not combine
`--runtime-state-dir` with `--online-artifact`; mixed sources are rejected.

After an owner-only UDS `admit-drop` action that occurs after a benchmark has
finished, refresh the same output directory before running the pair validator:

```bash
python scripts/reporting/refresh_runtime_control_artifacts.py \
  --runtime-state-dir results/pair/variant-state \
  --output-dir results/pair/variant
```

## Server E2E Exit Gate

Do not register or execute a live action merely to make this gate pass. A server
E2E is eligible only after the LocalDisk terminal callback completes the exact
submitted lease and the request release is observed.

The server E2E evidence must show, in one controlled run: compatible backend
capabilities;
authenticated request context; current-generation binding; inactive target;
admitted command; exactly one terminal receipt from the action service; and a
post-action server observation. Then run baseline and enabled conditions with
the paired-evidence gate and quality checks. Until all of those artifacts exist,
reports must say `hook_observed` or `dry_run_action_service`, never
`verified_real_action`.
