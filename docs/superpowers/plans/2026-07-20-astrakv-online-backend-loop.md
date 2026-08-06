# AstraKV Online Backend Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn verified backend object events into safety-gated, receipt-backed online AstraKV actions without modifying vLLM or LMCache source.

**Architecture:** An external backend Hook owns real vLLM/LMCache object identity and action execution. AstraKV accepts only validated object bindings and events, updates an in-memory ProfileDB, produces advisory decisions asynchronously, and dispatches a command only through a loopback Hook bridge after the existing offline safety gate accepts it. A matching structured receipt is the sole proof of execution.

**Tech Stack:** Python 3.12 standard library, existing `ProfileDB`, `OfflineSafetyGate`, `OfflineEvictionDecision`, JSONL evidence artifacts, HTTP loopback Hook transport.

## Global Constraints

- Do not modify vLLM, LMCache, CUDA, or other third-party source code.
- Do not represent a log heuristic, store artifact, or HTTP success alone as a completed backend action.
- A binding must include run, request, logical object, object level, and opaque backend object identity.
- The bridge must reject unknown bindings, inactive requests, unsupported actions, duplicate commands, remote Hook URLs, and mismatched receipts.
- The default controller remains advisory; execution requires an accepted `OfflineSafetyGate` and explicit opt-in.
- Preserve every input event, command, receipt, and generated decision as serializable records.

---

### Task 1: Backend Hook Contract

**Files:** create `astrakv/runtime/backend_hook.py` and `tests/test_backend_hook.py`; modify `astrakv/runtime/__init__.py`.

**Interfaces:** `BackendObjectBinding`, `BackendHookEvent`, `BackendActionCommand`, `BackendActionReceipt`, and `HookAction`; events require matching run, request, logical object, level, and opaque backend object identity.

- [ ] Write a failing valid/mismatched binding contract test; implement frozen records and strict parsing; run the unit test; commit `feat: add verified backend hook contract`.

### Task 2: Safety-Gated Hook Bridge

**Files:** create `astrakv/runtime/backend_bridge.py` and `tests/test_backend_bridge.py`; modify `astrakv/runtime/__init__.py`.

**Interfaces:** `OnlineBackendBridge.dispatch(decision) -> RuntimeActionResult`; only `prefetch`, `offload`, and `drop`; loopback HTTP; only fully matching receipt gives `executed`.

- [ ] Write failing valid/invalid endpoint, duplicate, and mismatched receipt tests; implement injected Hook client and JSON HTTP transport; run unit tests; commit `feat: add safety-gated backend hook bridge`.

### Task 3: Online Profile and Decision Feedback Loop

**Files:** create `astrakv/runtime/online_controller.py` and `tests/test_online_controller.py`; modify `astrakv/runtime/__init__.py`.

**Interfaces:** Hook events update `ProfileDB` via trace records; controller produces advisory `OfflineEvictionDecision` records and receipt feedback; dispatch requires explicit opt-in.

- [ ] Write failing ingestion/advisory/receipt test; implement profile ingestion, action mapping, receipt feedback, record accessors; run test; commit `feat: add online profile and decision feedback loop`.

### Task 4: Runtime Entry Point and Documentation

**Files:** create `scripts/runtime/run_online_policy_loop.py`, `tests/test_online_policy_loop.py`, and `docs/guides/online_backend_hook_cn.md`.

**Interfaces:** CLI writes `events.jsonl`, `decisions.jsonl`, `commands.jsonl`, `receipts.jsonl`, `trace.jsonl`, and `manifest.json`; `--execute` is required for Hook HTTP actions.

- [ ] Write a loopback integration test; implement CLI and operator guide; run target/full tests and `git diff --check`; commit `feat: add auditable online policy loop`.
