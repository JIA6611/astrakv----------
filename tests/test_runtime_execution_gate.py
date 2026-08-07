import tempfile
import threading
import unittest
from pathlib import Path

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_capabilities import build_installation_evidence, normalize_loopback_endpoint, preflight_backend_capabilities
from astrakv.runtime.backend_hook import BackendActionCommand, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.runtime_execution_gate import ExecutionBudget, RuntimeExecutionGate


HOOK_URL = "http://127.0.0.1:7900/actions"


def capability():
    return preflight_backend_capabilities(
        vllm_version="0.23.0", lmcache_version="0.4.7", run_id="run-a", hook_url=HOOK_URL,
        connector_name="lmcache-vllm-v1", connector_version="0.4.7", available_actions=("drop",),
        available_object_levels=(ObjectLevel.PREFIX,), binding_generation_observed=True,
        installation_evidence=build_installation_evidence(
            source="test", method="probe", session_id="session", vllm_version="0.23.0", lmcache_version="0.4.7",
            connector_name="lmcache-vllm-v1", connector_version="0.4.7",
            endpoint_identity=normalize_loopback_endpoint(HOOK_URL),
        ),
    )


def released_binding(*, size_bytes=8):
    registry = BackendBindingRegistry(run_id="run-a", engine_instance_id="engine", worker_id="worker")
    context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
    submitted = registry.observe("key", HookAction.CACHE_STORE, "submitted", context, bytes=size_bytes)
    binding = registry.complete_operation("key", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"], bytes=size_bytes).binding
    registry.observe("key", HookAction.RELEASE, "completed", context)
    return registry, context, binding


def command(binding, *, command_id="command-a", deadline_ns=100, bytes=8, generation=None):
    return BackendActionCommand(
        run_id="run-a", command_id=command_id, decision_id=command_id, request_id="request-a", object_key="prefix-a",
        object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id, backend_object_id=binding.backend_object_id,
        action=HookAction.DROP, issued_at_ns=1, binding_generation=binding.binding_generation if generation is None else generation,
        metadata={"deadline_ns": deadline_ns, "bytes": bytes},
    )


class RuntimeExecutionGateTests(unittest.TestCase):
    def gate(self, registry, **kwargs):
        return RuntimeExecutionGate(run_id="run-a", endpoint=HOOK_URL, capabilities=capability(), binding_registry=registry, **kwargs)

    def test_rejects_incompatible_capability(self):
        registry, _, binding = released_binding(size_bytes=9)
        denied = RuntimeExecutionGate(run_id="run-a", endpoint=HOOK_URL, capabilities=None, binding_registry=registry)
        self.assertEqual(denied.authorize(command(binding), now_ns=2).reason, "capability_preflight")

    def test_rejects_generation_mismatch_as_a_runtime_identity_field(self):
        registry, _, binding = released_binding()
        self.assertEqual(self.gate(registry).authorize(command(binding, generation=2), now_ns=2).reason, "binding_not_found")

    def test_rejects_replaced_bindings(self):
        registry, context, binding = released_binding()
        registry.observe("key", HookAction.CACHE_HIT, "completed", RequestContext("run-a", "request-b", "prefix-b", ObjectLevel.PREFIX))
        self.assertEqual(self.gate(registry).authorize(command(binding), now_ns=2).reason, "binding_replaced")

    def test_rejects_active_pinned_and_pending_binding_states(self):
        registry, context, binding = released_binding()
        registry.observe("key", HookAction.CACHE_HIT, "completed", context)
        self.assertEqual(self.gate(registry).authorize(command(binding), now_ns=2).reason, "active_binding_conflict")

        registry, context, binding = released_binding()
        registry.observe("key", HookAction.CACHE_HIT, "completed", context)
        registry.observe("key", HookAction.RELEASE, "completed", context)
        self.assertEqual(self.gate(registry).authorize(command(binding), now_ns=2).allowed, True)

        registry, context, binding = released_binding()
        binding = registry.observe("key", HookAction.CACHE_STORE, "submitted", context).binding
        self.assertEqual(self.gate(registry).authorize(command(binding), now_ns=2).reason, "pending_action_conflict")

        class PinnedRegistry:
            def snapshot(self, binding_id):
                return {"binding_generation": binding.binding_generation, "active_request_ids": (), "pin_count": 1,
                        "pending_io": 0, "pending_operations": (), "action_reservation": None}
            def current_binding(self, **kwargs):
                return binding
        self.assertEqual(self.gate(PinnedRegistry()).authorize(command(binding), now_ns=2).reason, "pinned_binding")

    def test_rejects_expired_duplicate_byte_rate_and_concurrency_budgets(self):
        registry, _, binding = released_binding(size_bytes=9)
        gate = self.gate(registry, budget=ExecutionBudget(max_bytes_per_command=8, max_bytes_per_window=12, max_commands_per_window=1, window_ns=10, max_concurrent=1))
        self.assertEqual(gate.authorize(command(binding, deadline_ns=1), now_ns=2).reason, "deadline_expired")
        self.assertEqual(gate.authorize(command(binding, command_id="too-big", bytes=9), now_ns=2).reason, "byte_budget")
        registry, _, binding = released_binding(size_bytes=8)
        gate = self.gate(registry, budget=ExecutionBudget(max_bytes_per_command=8, max_bytes_per_window=12, max_commands_per_window=1, window_ns=10, max_concurrent=1))
        self.assertTrue(gate.authorize(command(binding, command_id="first", bytes=8), now_ns=2).allowed)
        self.assertEqual(gate.authorize(command(binding, command_id="first", bytes=8), now_ns=2).reason, "duplicate_command")
        self.assertEqual(gate.authorize(command(binding, command_id="second", bytes=8), now_ns=2).reason, "rate_budget")

    def test_uses_observed_binding_size_not_command_metadata(self):
        registry, _, binding = released_binding(size_bytes=8)
        gate = self.gate(registry, budget=ExecutionBudget(max_bytes_per_command=8))
        self.assertTrue(gate.authorize(command(binding, bytes=1_000_000), now_ns=2).allowed)

    def test_rejects_when_concurrency_budget_is_exhausted(self):
        registry, _, binding = released_binding()
        gate = self.gate(registry, budget=ExecutionBudget(max_commands_per_window=3, max_concurrent=1))
        self.assertTrue(gate.authorize(command(binding, command_id="first"), now_ns=2).allowed)
        self.assertEqual(gate.authorize(command(binding, command_id="second"), now_ns=2).reason, "concurrency_budget")

    def test_concurrent_duplicate_admission_allows_exactly_one(self):
        registry, _, binding = released_binding()
        gate = self.gate(registry)
        results = []
        lock = threading.Lock()
        def submit():
            outcome = gate.authorize(command(binding, command_id="same"), now_ns=2)
            with lock:
                results.append(outcome)
        threads = [threading.Thread(target=submit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(result.allowed for result in results), 1)
        self.assertEqual(sum(result.reason == "duplicate_command" for result in results), 7)

    def test_persists_idempotence_across_restart(self):
        registry, _, binding = released_binding()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            self.assertTrue(self.gate(registry, state_path=path).authorize(command(binding), now_ns=2).allowed)
            self.assertEqual(self.gate(registry, state_path=path).authorize(command(binding), now_ns=2).reason, "duplicate_command")


if __name__ == "__main__":
    unittest.main()
