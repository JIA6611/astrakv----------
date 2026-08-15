import unittest
import tempfile
from dataclasses import replace
from unittest.mock import patch
from urllib.error import HTTPError

from astrakv.runtime.backend_bridge import InMemoryLoopbackHookClient, InMemoryProtectedActionService, JsonHttpHookClient, OnlineBackendBridge
from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_capabilities import RuntimeProbeChallenge, RuntimeProbeProof, build_installation_evidence, normalize_loopback_endpoint, preflight_backend_capabilities
from astrakv.runtime.backend_hook import (
    BackendActionCommand,
    BackendActionReceipt,
    BackendHookEvent,
    BackendObjectBinding,
    HookAction,
)
from astrakv.runtime.eviction import ObjectLevel, OfflineEvictionDecision
from astrakv.runtime.offline_safety import OfflineSafetyGate
from astrakv.runtime.runtime_execution_gate import ExecutionBudget, RuntimeExecutionGate
from astrakv.runtime.lmcache047_action_service import InMemoryRuntimeActionClient, ProtectedRuntimeActionService
from astrakv.runtime.lmcache047_runtime_patch import LMCache047ActionEndpoint


def accepted_gate() -> OfflineSafetyGate:
    manifests = []
    for workload_id in ("a", "b", "c"):
        manifests.append({
            "schema": "astrakv-offline-eviction-v1",
            "simulation_status": "valid",
            "workload_id": workload_id,
            "workload_sha256": "w" + workload_id,
            "trace_sha256": "t",
            "profile_db_sha256": "p",
            "profile_source": "separate_profiling_run",
            "capacities": {"gpu_bytes": 1, "cpu_bytes": 1, "ssd_bytes": 1},
            "policies": [
                {"policy": "astrakv", "request_count": 1, "total_hits": 1, "migration_bytes": 1, "oom_unavoided": 0},
                {"policy": "lru", "request_count": 1, "total_hits": 0, "migration_bytes": 2, "oom_unavoided": 0},
                {"policy": "fifo", "request_count": 1, "total_hits": 0, "migration_bytes": 3, "oom_unavoided": 0},
            ],
        })
    return OfflineSafetyGate.evaluate(manifests)


def binding() -> BackendObjectBinding:
    return BackendObjectBinding(
        run_id="run-1", request_id="request-1", object_key="prefix-1",
        object_level=ObjectLevel.PREFIX, backend_object_id="vllm:block:7", binding_id="binding-1",
    )


def activation_event(bound: BackendObjectBinding | None = None) -> BackendHookEvent:
    bound = binding() if bound is None else bound
    return BackendHookEvent(
        run_id="run-1", event_id="event-1", request_id="request-1", object_key="prefix-1",
        object_level=ObjectLevel.PREFIX, backend_object_id=bound.backend_object_id,
        action=HookAction.CACHE_STORE, status="completed", timestamp_ns=1,
        binding_generation=bound.binding_generation,
    )


def decision() -> OfflineEvictionDecision:
    return OfflineEvictionDecision(
        run_id="run-1", decision_id="decision-1", request_id="request-1",
        object_key="prefix-1", object_level=ObjectLevel.PREFIX, predicted_action="drop",
    )


def released_runtime_binding(binding_cycles: int = 1) -> tuple[BackendBindingRegistry, BackendObjectBinding]:
    registry = BackendBindingRegistry(run_id="run-1", engine_instance_id="engine", worker_id="worker")
    context = RequestContext("run-1", "request-1", "prefix-1", ObjectLevel.PREFIX)
    binding = None
    for _ in range(binding_cycles):
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context, bytes=2)
        binding = registry.complete_operation(
            "key-a", HookAction.CACHE_STORE, "completed", context,
            submitted.event.metadata["operation_lease"], bytes=2,
        ).binding
        registry.observe("key-a", HookAction.RELEASE, "completed", context)
    assert binding is not None
    return registry, binding


def capabilities(hook_url: str = "http://127.0.0.1:7900/actions", available_actions=("drop",)):
    return preflight_backend_capabilities(
        vllm_version="0.23.0", lmcache_version="0.4.7",
        connector_name="lmcache-vllm-v1", connector_version="0.4.7",
        available_actions=available_actions, available_object_levels=(ObjectLevel.PREFIX,),
        binding_generation_observed=True,
        run_id="run-1", hook_url=hook_url,
        installation_evidence=build_installation_evidence(
            source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a", vllm_version="0.23.0",
            lmcache_version="0.4.7", connector_name="lmcache-vllm-v1", connector_version="0.4.7",
            endpoint_identity=normalize_loopback_endpoint(hook_url),
        ),
    )


class BackendBridgeTests(unittest.TestCase):
    def test_bridge_resolves_bindings_by_request_and_binding_id_before_object_key_latest(self) -> None:
        first = BackendObjectBinding(
            "run-1",
            "request-1",
            "prefix-1",
            ObjectLevel.PREFIX,
            "backend-1",
            "binding-1",
        )
        second = BackendObjectBinding(
            "run-1",
            "request-2",
            "prefix-1",
            ObjectLevel.PREFIX,
            "backend-2",
            "binding-2",
        )
        bridge = OnlineBackendBridge(
            run_id="run-1",
            bindings=[first, second],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=accepted_gate(),
        )

        self.assertEqual(
            bridge.binding_for(ObjectLevel.PREFIX, "prefix-1", request_id="request-1"),
            first,
        )
        self.assertEqual(
            bridge.binding_for(ObjectLevel.PREFIX, "prefix-1", request_id="request-2"),
            second,
        )
        self.assertEqual(
            bridge.binding_for(ObjectLevel.PREFIX, "prefix-1", binding_id="binding-1"),
            first,
        )
        self.assertTrue(
            bridge.observe_event(
                BackendHookEvent(
                    run_id="run-1",
                    event_id="event-1",
                    request_id="request-1",
                    object_key="prefix-1",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id="backend-1",
                    action=HookAction.CACHE_STORE,
                    status="completed",
                    timestamp_ns=1,
                    metadata={"binding_id": "binding-1"},
                )
            )
        )

    def test_runtime_execution_gate_blocks_before_hook_submit(self) -> None:
        registry, runtime_binding = released_runtime_binding()
        def responder(command):
            self.fail("runtime gate must stop this command before the Hook")

        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[runtime_binding],
            hook_client=InMemoryLoopbackHookClient("http://127.0.0.1:7900/actions", InMemoryProtectedActionService(responder, source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a")),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=capabilities(), binding_registry=registry,
            execution_gate=RuntimeExecutionGate(
                run_id="run-1", endpoint="http://127.0.0.1:7900/actions", capabilities=capabilities(), binding_registry=registry,
                budget=ExecutionBudget(max_bytes_per_command=1),
            ),
        )
        blocked = replace(decision(), metadata={"deadline_ns": 10**20, "bytes": 2})
        result = bridge.dispatch(blocked)

        self.assertEqual(result.status, "blocked_by_runtime_gate:byte_budget")
        assert result.receipt is not None
        self.assertEqual(result.receipt.status, "rejected")
        self.assertEqual(result.receipt.command_id, "run-1:decision-1")
        self.assertEqual(result.receipt.decision_id, "decision-1")
        self.assertEqual(result.receipt.request_id, "request-1")
        self.assertEqual(result.receipt.rejection_reason, "runtime_execution_gate:byte_budget")
        self.assertIsNone(result.event)

    def test_bridge_transfers_unconsumed_lease_to_runtime_owned_action_service(self) -> None:
        registry, runtime_binding = released_runtime_binding()

        class Manager:
            def __init__(self): self.removed = []
            def remove(self, key, locations=None):
                self.removed.append((key, locations))
                return 1

        manager = Manager()
        endpoint = LMCache047ActionEndpoint(registry, action_registration_enabled=True)
        endpoint.register_binding(runtime_binding, "key-a", manager)
        with tempfile.TemporaryDirectory() as directory:
            service = ProtectedRuntimeActionService(
                action_endpoint=endpoint, state_dir=directory, secret=b"x" * 32,
                source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a",
            )
            bridge = OnlineBackendBridge(
                run_id="run-1", bindings=[runtime_binding],
                hook_client=InMemoryRuntimeActionClient(service),
                hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=capabilities(),
                binding_registry=registry,
            )
            self.assertTrue(bridge.observe_event(activation_event(runtime_binding)))
            self.assertEqual(bridge.dispatch(decision()).status, "executed")
        self.assertEqual(manager.removed, [("key-a", None)])

    def test_matching_hook_receipt_is_the_only_executed_result(self) -> None:
        registry, runtime_binding = released_runtime_binding()
        def responder(command):
            return BackendActionReceipt(
                run_id=command.run_id, command_id=command.command_id, receipt_id="receipt-1",
                binding_id=command.binding_id, backend_object_id=command.backend_object_id,
                action=command.action, status="completed", timestamp_ns=2,
                tier_before="gpu", tier_after="ssd",
            )

        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[runtime_binding],
            hook_client=InMemoryLoopbackHookClient("http://127.0.0.1:7900/actions", InMemoryProtectedActionService(responder, source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a")),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=capabilities(), binding_registry=registry,
        )
        self.assertTrue(bridge.observe_event(activation_event(runtime_binding)))
        result = bridge.dispatch(decision())

        self.assertEqual(result.status, "executed")
        assert result.event is not None
        self.assertTrue(result.event.is_ground_truth)
        self.assertEqual(result.event.tier_after, "ssd")
        self.assertEqual(result.event.metadata["receipt_id"], "receipt-1")
        self.assertEqual(bridge.dispatch(decision()).status, "duplicate_command")

    def test_bridge_rejects_nonloopback_endpoint_and_mismatched_receipt(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            OnlineBackendBridge(
                run_id="run-1", bindings=[binding()], hook_client=object(),
                hook_url="http://10.0.0.2/actions", gate=accepted_gate(),
            )

        def mismatched_responder(command):
            return BackendActionReceipt(
                run_id=command.run_id, command_id=command.command_id, receipt_id="receipt-1",
                binding_id=command.binding_id, backend_object_id="different",
                action=command.action, status="completed", timestamp_ns=2,
            )

        registry, runtime_binding = released_runtime_binding()
        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[runtime_binding],
            hook_client=InMemoryLoopbackHookClient("http://localhost:7900/actions", InMemoryProtectedActionService(mismatched_responder, source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a")),
            hook_url="http://localhost:7900/actions", gate=accepted_gate(),
            capabilities=capabilities("http://localhost:7900/actions"), binding_registry=registry,
        )
        bridge.observe_event(activation_event(runtime_binding))
        self.assertEqual(bridge.dispatch(decision()).status, "receipt_mismatch")

    def test_bridge_refuses_execution_without_compatible_capability_probe(self) -> None:
        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[binding()], hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=None,
        )
        self.assertTrue(bridge.observe_event(activation_event()))
        self.assertEqual(bridge.dispatch(decision()).status, "blocked_by_capability_preflight")

    def test_bridge_rejects_a_stale_receipt_after_binding_reuse(self) -> None:
        registry, first_binding = released_runtime_binding()
        context = RequestContext("run-1", "request-1", "prefix-1", ObjectLevel.PREFIX)
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        reused_binding = registry.complete_operation(
            "key-a",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        ).binding
        registry.observe("key-a", HookAction.RELEASE, "completed", context)

        def stale_responder(command):
            return BackendActionReceipt(
                run_id=command.run_id, command_id=command.command_id, receipt_id="receipt-1",
                binding_id=first_binding.binding_id, backend_object_id=command.backend_object_id,
                action=command.action, status="completed", timestamp_ns=2, binding_generation=1,
            )

        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[reused_binding],
            hook_client=InMemoryLoopbackHookClient("http://127.0.0.1:7900/actions", InMemoryProtectedActionService(stale_responder, source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a")),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=capabilities(), binding_registry=registry,
        )
        self.assertTrue(bridge.observe_event(BackendHookEvent(
            run_id="run-1", event_id="event-1", request_id="request-1", object_key="prefix-1",
            object_level=ObjectLevel.PREFIX, backend_object_id=reused_binding.backend_object_id, action=HookAction.CACHE_STORE,
            status="completed", timestamp_ns=1, binding_generation=1, metadata={"binding_id": reused_binding.binding_id},
        )))
        self.assertEqual(bridge.dispatch(decision()).status, "receipt_mismatch")

    def test_bridge_rejects_preflight_bound_to_a_different_run_or_endpoint(self) -> None:
        for result in (
            preflight_backend_capabilities(
                vllm_version="0.23.0", lmcache_version="0.4.7", connector_name="lmcache-vllm-v1",
                connector_version="0.4.7", available_actions=("drop",),
                available_object_levels=(ObjectLevel.PREFIX,), binding_generation_observed=True,
                run_id="other-run", hook_url="http://127.0.0.1:7900/actions",
                installation_evidence=build_installation_evidence(
                    source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a", vllm_version="0.23.0",
                    lmcache_version="0.4.7", connector_name="lmcache-vllm-v1", connector_version="0.4.7",
                    endpoint_identity="http://127.0.0.1:7900/actions",
                ),
            ),
            preflight_backend_capabilities(
                vllm_version="0.23.0", lmcache_version="0.4.7", connector_name="lmcache-vllm-v1",
                connector_version="0.4.7", available_actions=("drop",),
                available_object_levels=(ObjectLevel.PREFIX,), binding_generation_observed=True,
                run_id="run-1", hook_url="http://127.0.0.1:7901/actions",
                installation_evidence=build_installation_evidence(
                    source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a", vllm_version="0.23.0",
                    lmcache_version="0.4.7", connector_name="lmcache-vllm-v1", connector_version="0.4.7",
                    endpoint_identity="http://127.0.0.1:7901/actions",
                ),
            ),
        ):
            bridge = OnlineBackendBridge(
                run_id="run-1", bindings=[binding()], hook_client=object(),
                hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=result,
            )
            self.assertTrue(bridge.observe_event(activation_event()))
            self.assertEqual(bridge.dispatch(decision()).status, "blocked_by_capability_preflight")

    def test_bridge_rejects_preflight_without_installation_evidence(self) -> None:
        missing_installation = preflight_backend_capabilities(
            vllm_version="0.23.0", lmcache_version="0.4.7", connector_name="lmcache-vllm-v1",
            connector_version="0.4.7", available_actions=("drop",),
            available_object_levels=(ObjectLevel.PREFIX,), binding_generation_observed=True,
            run_id="run-1", hook_url="http://127.0.0.1:7900/actions", installation_evidence=None,
        )
        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[binding()], hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=missing_installation,
        )
        self.assertTrue(bridge.observe_event(activation_event()))
        self.assertEqual(bridge.dispatch(decision()).status, "blocked_by_capability_preflight")

        forged_capability = replace(capabilities(), installation_evidence=None)
        forged_bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[binding()], hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=forged_capability,
        )
        self.assertTrue(forged_bridge.observe_event(activation_event()))
        self.assertEqual(forged_bridge.dispatch(decision()).status, "blocked_by_capability_preflight")

    def test_bridge_rejects_client_endpoint_mismatches_and_json_client_rejects_invalid_url(self) -> None:
        class ExternalClient:
            endpoint_identity = "http://10.0.0.2:7900/actions"

            def submit(self, command):
                raise AssertionError("must not submit")

        class OtherPortClient:
            endpoint_identity = "http://127.0.0.1:7901/actions"

            def submit(self, command):
                raise AssertionError("must not submit")

        class NoIdentityClient:
            def submit(self, command):
                raise AssertionError("must not submit")

        class SelfAttestedClient:
            endpoint_identity = "http://127.0.0.1:7900/actions"

            def submit(self, command):
                raise AssertionError("must not submit")

        for client in (ExternalClient(), OtherPortClient(), NoIdentityClient(), SelfAttestedClient()):
            bridge = OnlineBackendBridge(
                run_id="run-1", bindings=[binding()], hook_client=client,
                hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(), capabilities=capabilities(),
            )
            self.assertTrue(bridge.observe_event(activation_event()))
            self.assertEqual(bridge.dispatch(decision()).status, "blocked_by_transport_endpoint")

        with self.assertRaisesRegex(ValueError, "loopback"):
            JsonHttpHookClient("http://10.0.0.2:7900/actions")
        with self.assertRaisesRegex(ValueError, "literal IP"):
            JsonHttpHookClient("http://localhost:7900/actions")
        with self.assertRaisesRegex(ValueError, "loopback"):
            JsonHttpHookClient("http://127.0.0.1:0/actions")

    def test_json_client_rejects_loopback_to_external_redirect(self) -> None:
        class RedirectingOpener:
            def open(self, request, timeout):
                raise HTTPError(request.full_url, 302, "redirect", {}, None)

        challenge = RuntimeProbeChallenge(
            "nonce", "run-1", "http://127.0.0.1:7900/actions", "0.23.0", "0.4.7",
            "lmcache-vllm-v1", "0.4.7", ("drop",), (ObjectLevel.PREFIX,), True,
        )
        with patch("astrakv.runtime.backend_bridge.build_opener", return_value=RedirectingOpener()) as factory:
            with self.assertRaises(HTTPError):
                JsonHttpHookClient("http://127.0.0.1:7900/actions").issue_runtime_proof(challenge)

        handlers = factory.call_args.args
        self.assertTrue(any(handler.__class__.__name__ == "ProxyHandler" for handler in handlers))
        self.assertTrue(any(handler.__class__.__name__ == "_RejectRedirects" for handler in handlers))

    def test_protected_action_service_rejects_forged_proof_before_invoking_responder(self) -> None:
        service = InMemoryProtectedActionService(
            lambda command: self.fail("forged proof must not reach the runtime action"),
            source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a",
        )
        challenge = RuntimeProbeChallenge(
            "nonce", "run-1", "http://127.0.0.1:7900/actions", "0.23.0", "0.4.7",
            "lmcache-vllm-v1", "0.4.7", ("drop",), (ObjectLevel.PREFIX,), True,
        )
        proof = service.issue_runtime_proof(challenge)
        forged = RuntimeProbeProof(proof.nonce, proof.source, proof.method, proof.session_id, "0" * 64)
        command = BackendActionCommand(
            "run-1", "command-1", "decision-1", "request-1", "prefix-1", ObjectLevel.PREFIX,
            "binding-1", "vllm:block:7", HookAction.DROP, 1,
        )

        self.assertEqual(service.submit(command, challenge, forged).status, "failed")

    def test_prefetch_single_flight_guard_blocks_a_second_promotion(self) -> None:
        class Guard:
            def __init__(self, active: bool = True) -> None:
                self.active = active
                self.completed: list[str] = []

            def try_begin(self, object_key: str, *, request_id: str, deadline_ns: int) -> str | None:
                if self.active:
                    return None
                self.active = True
                return "lease-1"

            def complete(self, lease_id: str) -> None:
                self.completed.append(lease_id)
                self.active = False

            def has_active(self, object_key: str) -> bool:
                return self.active

        registry, runtime_binding = released_runtime_binding()
        guard = Guard(active=True)
        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[runtime_binding],
            hook_client=InMemoryLoopbackHookClient(
                "http://127.0.0.1:7900/actions",
                InMemoryProtectedActionService(
                    lambda command: self.fail("single-flight guard must block before submit"),
                    source="lmcache047_runtime_patch",
                    method="signature_probe",
                    session_id="session-a",
                ),
            ),
            hook_url="http://127.0.0.1:7900/actions",
            gate=accepted_gate(), capabilities=capabilities(available_actions=("drop", "prefetch")),
            binding_registry=registry, prefetch_guard=guard,
        )
        self.assertTrue(bridge.observe_event(activation_event(runtime_binding)))
        prefetch_decision = replace(decision(), decision_id="prefetch-1", predicted_action="prefetch")

        result = bridge.dispatch(prefetch_decision)

        self.assertEqual(result.status, "prefetch_single_flight_conflict")
        self.assertEqual(guard.completed, [])
        self.assertTrue(guard.active)

    def test_prefetch_single_flight_guard_is_released_after_hook_submit(self) -> None:
        class Guard:
            def __init__(self) -> None:
                self.active = False
                self.completed: list[str] = []

            def try_begin(self, object_key: str, *, request_id: str, deadline_ns: int) -> str | None:
                self.active = True
                return "lease-1"

            def complete(self, lease_id: str) -> None:
                self.completed.append(lease_id)
                self.active = False

            def has_active(self, object_key: str) -> bool:
                return self.active

        registry, runtime_binding = released_runtime_binding()

        def responder(command: BackendActionCommand) -> BackendActionReceipt:
            return BackendActionReceipt(
                run_id=command.run_id, command_id=command.command_id, receipt_id="receipt-prefetch",
                binding_id=command.binding_id, backend_object_id=command.backend_object_id,
                action=command.action, status="completed", timestamp_ns=2,
            )

        guard = Guard()
        bridge = OnlineBackendBridge(
            run_id="run-1", bindings=[runtime_binding],
            hook_client=InMemoryLoopbackHookClient(
                "http://127.0.0.1:7900/actions",
                InMemoryProtectedActionService(
                    responder, source="lmcache047_runtime_patch",
                    method="signature_probe", session_id="session-a",
                ),
            ),
            hook_url="http://127.0.0.1:7900/actions", gate=accepted_gate(),
            capabilities=capabilities(available_actions=("drop", "prefetch")),
            binding_registry=registry, prefetch_guard=guard,
        )
        self.assertTrue(bridge.observe_event(activation_event(runtime_binding)))
        prefetch_decision = replace(decision(), decision_id="prefetch-1", predicted_action="prefetch")

        result = bridge.dispatch(prefetch_decision)

        self.assertEqual(result.status, "executed")
        self.assertEqual(guard.completed, ["lease-1"])
        self.assertFalse(guard.active)


if __name__ == "__main__":
    unittest.main()
