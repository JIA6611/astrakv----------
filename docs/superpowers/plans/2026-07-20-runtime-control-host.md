# Runtime Control Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task by task.

**Goal:** Connect the existing version-locked LMCache Hook, request-context authority, binding registry, protected LocalDisk action endpoint, and durable artifacts in one explicit runtime-owned host.

**Architecture:** `RuntimeControlHost` owns all state for one `run_id` and exposes only numeric-loopback request-context HTTP plus an owner-only Unix socket action service. The LMCache bootstrap creates this host only when every required environment variable is set; otherwise the current observer-only behavior remains. The host records v2 events, bindings, preflight, commands, and receipts without claiming a real action.

**Tech Stack:** Python 3.12 standard-library HTTP server and Unix sockets; existing `BackendBindingRegistry`, `RuntimeRequestContextReceiver`, `LMCache047ActionEndpoint`, `ProtectedRuntimeActionService`, and `RuntimeExecutionGate`.

## Global Constraints

- Support exactly vLLM `0.23.0`, LMCache `0.4.7`, connector `lmcache-vllm-v1` `0.4.7`.
- Do not modify vLLM or LMCache source code.
- Accept only numeric loopback context endpoints and an owner-only UDS action endpoint.
- Leave action registration disabled until the LocalDisk completion callback and release establish a safe binding.
- Every runtime record is JSONL and keyed by `run_id`; no endpoint response or log line is action proof.

---

### Task 1: Build and test the runtime host

**Files:**
- Create: `astrakv/runtime/runtime_control_host.py`
- Create: `tests/test_runtime_control_host.py`

**Interfaces:**
- `RuntimeControlHost.create_from_environment() -> RuntimeControlHost | None`
- `RuntimeControlHost.install_hooks(installer) -> LMCache047ActionEndpoint`
- `RuntimeControlHost.close() -> None`

- [ ] Write a failing test that creates a host from explicit configuration, publishes an authenticated HTTP request context, installs a fake Hook installer, and asserts `preflight.json` plus JSONL event artifacts.
- [ ] Run `python -m unittest tests.test_runtime_control_host` and confirm the missing module failure.
- [ ] Implement the minimal host: create registry, authority/receiver, event sink, capability preflight, protected action service, UDS server, and context HTTP handler; inject registry/consumer/identity provider into Hook installer.
- [ ] Run the focused test and confirm it passes.

### Task 2: Make bootstrap explicitly install the host

**Files:**
- Modify: `astrakv/runtime/lmcache047_bootstrap.py`
- Modify: `scripts/launch/launch_vllm_server.sh`
- Test: `tests/test_lmcache047_bootstrap.py`

**Interfaces:**
- `ASTRAKV_RUNTIME_CONTROL_RUN_ID` opts into the host.
- Missing host configuration leaves the current event-only observer behavior unchanged.

- [ ] Write a failing bootstrap test that supplies host configuration and verifies the installer receives non-null registry, context consumer, and runtime identity provider.
- [ ] Run the focused bootstrap test and confirm it fails before the wiring exists.
- [ ] Implement environment parsing and launcher validation for run ID, state directory, loopback context port, and 32-byte secret file.
- [ ] Run focused tests and confirm both observer-only and host modes pass.

### Task 3: Verify host artifacts and server readiness

**Files:**
- Modify: `docs/guides/online_runtime_control_cn.md`
- Test: `tests/test_runtime_control_host.py`

- [ ] Add a failing artifact test rejecting a host whose event/receipt record belongs to another run.
- [ ] Implement run-bound writer validation and a close path that closes HTTP and UDS listeners.
- [ ] Run all runtime-host and bootstrap tests, then `python -m unittest discover -s tests`.
- [ ] Document the exact host launch variables and state that a real server E2E is still required for `verified_real_action`.
