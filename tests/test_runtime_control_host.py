import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, build_opener, ProxyHandler

from astrakv.runtime.request_context import AuthenticatedJsonHttpRequestContextClient, RuntimeRequestContext
from astrakv.runtime.runtime_control_host import RuntimeControlHost, RuntimeControlHostConfig, _should_attempt_online_dispatch
from astrakv.runtime.backend_binding_registry import RequestContext
from astrakv.runtime.backend_hook import BackendActionCommand, BackendActionReceipt, BackendHookEvent, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.lmcache047_action_service import command_integrity_digest
from astrakv.runtime.kv_runtime_core import RuntimeMode
from astrakv.runtime.prediction_sidecar import PredictionSidecarIndex, SidecarPrediction


class RuntimeControlHostTests(unittest.TestCase):
    def test_sidecar_authorizes_only_matching_ingress_lead_context(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker",
                online_policy_enabled=True,
                online_prefetch_dispatch_enabled=True,
                online_prefetch_mode="hybrid",
            ))
            prediction = SidecarPrediction(
                run_id="run-a", request_id="request-far",
                candidate_object_id="prefix-a", object_level=ObjectLevel.PREFIX,
                score=0.99, recommended_lead_time_ms=250.0,
                confidence=0.98, reason="exact_next_locality",
                evidence_source="test", predicted_class="exact-next",
                expires_at_ns=9_999_999_999_999_999_999,
            )
            host.online_controller = SimpleNamespace(
                prediction_source=PredictionSidecarIndex([prediction], run_id="run-a"),
                scheduler_hints=None,
            )
            context = RuntimeRequestContext(
                run_id="run-a", request_id="request-far", case="case",
                request_nonce="nonce", request_started_s=time.time(),
                metadata={"cache_key": "prefix-a", "prefix_id": "prefix-a",
                          "prefetch_lead_s": 5.0, "exact_token_ids": [1, 2, 3]},
            )

            authorized = host._authorize_predictive_prefetch_context(context)

            self.assertIsNot(authorized, context)
            self.assertTrue(authorized.metadata["predictive_prefetch_authorized"])
            self.assertEqual(authorized.metadata["prefetch_origin"], "sidecar_b")
            rows = [
                json.loads(line) for line in
                (Path(directory) / "predictive_prefetch_authorizations.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["request_id"], "request-far")
            self.assertEqual(rows[0]["object_key"], "prefix-a")

    def test_duplicate_single_process_bridge_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker",
            ))
            first = SimpleNamespace()
            second = SimpleNamespace()
            host.register_kv_runtime_bridge(first)
            host.register_kv_runtime_bridge(second)
            self.assertIs(host._kv_runtime_bridge, first)

    def test_should_attempt_online_dispatch_respects_release_toggle(self):
        release = BackendHookEvent(
            run_id="run-a",
            event_id="release-a",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="object-a",
            action=HookAction.RELEASE,
            status="completed",
            timestamp_ns=1,
        )
        load_ready = BackendHookEvent(
            run_id="run-a",
            event_id="load-ready-a",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="object-a",
            action=HookAction.CACHE_LOAD,
            status="available",
            timestamp_ns=2,
            metadata={"dispatch_signal": "dynamic_load_target_ready"},
        )
        offload = BackendHookEvent(
            run_id="run-a",
            event_id="offload-a",
            request_id="request-a",
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            backend_object_id="object-a",
            action=HookAction.OFFLOAD,
            status="completed",
            timestamp_ns=3,
        )

        self.assertTrue(_should_attempt_online_dispatch(release))
        self.assertFalse(_should_attempt_online_dispatch(release, dispatch_on_release=False))
        self.assertTrue(_should_attempt_online_dispatch(load_ready, dispatch_on_release=False))
        self.assertTrue(_should_attempt_online_dispatch(offload))

    def test_enabled_online_policy_keep_decision_does_not_write_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
                online_policy_enabled=True, offline_gate_record={"status": "accepted", "reasons": [], "workload_ids": [], "aggregate": {}, "checks": {}}, kv_core_mode=RuntimeMode.ACTIVE,
            ))
            try:
                host.start()
                host.install_hooks(lambda *_args, **_kwargs: object())
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
                completed = host.binding_registry.complete_operation(
                    "key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"],
                )
                assert completed.binding is not None
                for record in submitted.records + completed.records:
                    host._event_sink(record)
                host._event_sink(BackendHookEvent(
                    run_id="run-a",
                    event_id="cache-hit-a",
                    request_id="request-a",
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    backend_object_id=completed.binding.backend_object_id,
                    action=HookAction.CACHE_HIT,
                    status="completed",
                    timestamp_ns=1,
                    tier_after="gpu",
                    binding_generation=completed.binding.binding_generation,
                    metadata={"binding_id": completed.binding.binding_id},
                ).to_record())
                for record in host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context).records:
                    host._event_sink(record)

                deadline = time.monotonic() + 2
                checkpoint_path = Path(directory) / "online_profile_checkpoint.json"
                checkpoint = {}
                while time.monotonic() < deadline:
                    if checkpoint_path.exists():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if checkpoint.get("controller_state", {}).get("dispatch_status_counts", {}).get("no_dispatch_required") == 1:
                            break
                    time.sleep(0.01)
                self.assertTrue(checkpoint_path.exists())
                self.assertEqual(
                    checkpoint.get("controller_state", {}).get("dispatch_status_counts", {}).get("no_dispatch_required"),
                    1,
                )
                receipt_path = Path(directory) / "runtime_command_receipts.jsonl"
                self.assertFalse(receipt_path.exists())
                rows = [json.loads(line) for line in (Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertFalse(any(row["status"] == "rejected" and row.get("metadata", {}).get("policy_rejection_reason") == "no_dispatch_required" for row in rows))
            finally:
                host.close()

    def test_online_policy_release_callback_does_not_wait_for_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
                online_policy_enabled=True, offline_gate_record={"status": "accepted", "reasons": [], "workload_ids": [], "aggregate": {}, "checks": {}}, kv_core_mode=RuntimeMode.ACTIVE,
            ))
            remove_started = threading.Event()
            allow_remove = threading.Event()

            class BlockingManager:
                def remove(self, key, locations=None):
                    remove_started.set()
                    allow_remove.wait(timeout=2)
                    return 1

            try:
                host.start()
                host.install_hooks(lambda *_args, **_kwargs: object())
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context, bytes=4096)
                completed = host.binding_registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"], bytes=4096)
                assert completed.binding is not None
                host.action_endpoint.action_registration_enabled = True
                manager = BlockingManager()
                host.action_endpoint.mark_store_completed(completed.binding, "key-a", manager)
                for record in submitted.records + completed.records:
                    host._event_sink(record)
                host._event_sink(
                    BackendHookEvent(
                        run_id="run-a",
                        event_id="event-gpu",
                        request_id="request-a",
                        object_key="prefix-a",
                        object_level=ObjectLevel.PREFIX,
                        backend_object_id=completed.binding.backend_object_id,
                        action=HookAction.CACHE_MISS,
                        status="completed",
                        timestamp_ns=time.time_ns(),
                        tier_after="gpu",
                        binding_generation=completed.binding.binding_generation,
                        metadata={"binding_id": completed.binding.binding_id},
                    ).to_record()
                )
                released = host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                execution_spec = host.action_endpoint.register_binding(released.binding, "key-a", manager)
                if execution_spec is not None and released.binding_record is not None:
                    released.binding_record["execution_spec"] = execution_spec.to_record()

                timer = threading.Timer(0.5, allow_remove.set)
                timer.start()
                started_at = time.monotonic()
                for record in released.records:
                    host._event_sink(record)
                self.assertLess(time.monotonic() - started_at, 0.25)
                self.assertTrue(remove_started.wait(timeout=1))
                timer.join(timeout=1)

                receipt_path = Path(directory) / "runtime_command_receipts.jsonl"
                deadline = time.monotonic() + 1
                while not receipt_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "completed")
            finally:
                allow_remove.set()
                host.close()

    def test_enabled_online_policy_dispatches_drop_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
                online_policy_enabled=True, offline_gate_record={"status": "accepted", "reasons": [], "workload_ids": [], "aggregate": {}, "checks": {}}, kv_core_mode=RuntimeMode.ACTIVE,
            ))

            class Manager:
                def remove(self, key, locations=None): return 1

            try:
                host.start()
                host.install_hooks(lambda *_args, **_kwargs: object())
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context, bytes=4096)
                completed = host.binding_registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"], bytes=4096)
                assert completed.binding is not None
                host.action_endpoint.action_registration_enabled = True
                manager = Manager()
                host.action_endpoint.mark_store_completed(completed.binding, "key-a", manager)
                for record in submitted.records + completed.records:
                    host._event_sink(record)
                host._event_sink(
                    BackendHookEvent(
                        run_id="run-a",
                        event_id="event-gpu",
                        request_id="request-a",
                        object_key="prefix-a",
                        object_level=ObjectLevel.PREFIX,
                        backend_object_id=completed.binding.backend_object_id,
                        action=HookAction.CACHE_MISS,
                        status="completed",
                        timestamp_ns=time.time_ns(),
                        tier_after="gpu",
                        binding_generation=completed.binding.binding_generation,
                        metadata={"binding_id": completed.binding.binding_id},
                    ).to_record()
                )
                released = host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                execution_spec = host.action_endpoint.register_binding(released.binding, "key-a", manager)
                if execution_spec is not None and released.binding_record is not None:
                    released.binding_record["execution_spec"] = execution_spec.to_record()
                for record in released.records:
                    host._event_sink(record)

                receipt_path = Path(directory) / "runtime_command_receipts.jsonl"
                events_path = Path(directory) / "runtime_events_raw.jsonl"
                deadline = time.monotonic() + 1
                receipt = None
                events = []
                while time.monotonic() < deadline:
                    if receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if events_path.exists():
                        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
                    if receipt is not None and any(
                        row.get("action") == "drop" and row.get("metadata", {}).get("command_id") == receipt["command_id"]
                        for row in events
                    ):
                        break
                    time.sleep(0.01)
                self.assertTrue(receipt_path.exists())
                self.assertIsNotNone(receipt)
                self.assertEqual(receipt["status"], "completed")
                self.assertTrue(any(row.get("action") == "drop" and row.get("metadata", {}).get("command_id") == receipt["command_id"] for row in events))
            finally:
                host.close()

    def test_online_policy_pre_execution_rejection_writes_rejected_command_receipt_and_event(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
                online_policy_enabled=True, offline_gate_record={"status": "rejected", "reasons": ["offline gate blocked"], "workload_ids": [], "aggregate": {}, "checks": {}}, kv_core_mode=RuntimeMode.ACTIVE,
            ))

            class Manager:
                def remove(self, key, locations=None):
                    raise AssertionError("pre-execution rejection must not invoke the backend action")

            try:
                host.start()
                host.install_hooks(lambda *_args, **_kwargs: object())
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
                completed = host.binding_registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"])
                assert completed.binding is not None
                host.action_endpoint.action_registration_enabled = True
                manager = Manager()
                host.action_endpoint.mark_store_completed(completed.binding, "key-a", manager)
                for record in submitted.records + completed.records:
                    host._event_sink(record)
                host._event_sink(
                    BackendHookEvent(
                        run_id="run-a",
                        event_id="event-gpu",
                        request_id="request-a",
                        object_key="prefix-a",
                        object_level=ObjectLevel.PREFIX,
                        backend_object_id=completed.binding.backend_object_id,
                        action=HookAction.CACHE_MISS,
                        status="completed",
                        timestamp_ns=time.time_ns(),
                        tier_after="gpu",
                        binding_generation=completed.binding.binding_generation,
                        metadata={"binding_id": completed.binding.binding_id},
                    ).to_record()
                )
                released = host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                execution_spec = host.action_endpoint.register_binding(released.binding, "key-a", manager)
                if execution_spec is not None and released.binding_record is not None:
                    released.binding_record["execution_spec"] = execution_spec.to_record()
                for record in released.records:
                    host._event_sink(record)

                receipt_path = Path(directory) / "runtime_command_receipts.jsonl"
                deadline = time.monotonic() + 1
                while not receipt_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(receipt_path.exists())
                receipts = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(receipts), 1)
                self.assertEqual(receipts[0]["status"], "rejected")
                self.assertTrue(str(receipts[0]["decision_id"]).startswith("online-0-"))
                self.assertEqual(receipts[0]["request_id"], "request-a")
                self.assertEqual(receipts[0]["rejection_reason"], "offline_gate_rejected")

                command_path = Path(directory) / "astrakv_runtime_commands.jsonl"
                self.assertTrue(command_path.exists())
                commands = [json.loads(line) for line in command_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(commands), 1)
                self.assertEqual(commands[0]["request_id"], "request-a")

                structured_path = Path(directory) / "runtime_structured_events.jsonl"
                self.assertTrue(structured_path.exists())
                structured = [json.loads(line) for line in structured_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(structured), 1)
                self.assertEqual(structured[0]["status"], "rejected")
                self.assertEqual(structured[0]["command_id"], receipts[0]["command_id"])
                events = [json.loads(line) for line in (Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(any(row.get("status") == "rejected" and row.get("metadata", {}).get("policy_rejection_reason") == "blocked_by_offline_gate" for row in events))
            finally:
                host.close()

    def test_event_artifact_flattens_binding_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            try:
                host.start()
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                observation = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
                for record in observation.records:
                    host._event_sink(record)
                event = json.loads((Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8"))
                self.assertTrue(event["binding_id"])
                self.assertEqual(event["binding_generation"], 1)
            finally:
                host.close()

    def test_owner_only_uds_admits_and_executes_verified_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))

            class Manager:
                def __init__(self): self.removed = []
                def remove(self, key, locations=None):
                    self.removed.append((key, locations))
                    return 1

            try:
                host.start()
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context, bytes=4096)
                binding = host.binding_registry.complete_operation(
                    "key-a", HookAction.CACHE_STORE, "completed", context,
                    submitted.event.metadata["operation_lease"], bytes=4096,
                ).binding
                host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                assert binding is not None
                manager = Manager()
                host.action_endpoint.action_registration_enabled = True
                host.action_endpoint.register_binding(binding, "key-a", manager)

                receipt = host.admit_drop(binding)

                self.assertEqual(receipt.status, "completed")
                self.assertEqual(manager.removed, [("key-a", None)])
                commands = (Path(directory) / "astrakv_runtime_commands.jsonl").read_text(encoding="utf-8").splitlines()
                receipts = (Path(directory) / "runtime_command_receipts.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(commands), 1)
                self.assertEqual(len(receipts), 1)
                command_row = json.loads(commands[0])
                structured = [
                    json.loads(line)
                    for line in (Path(directory) / "runtime_structured_events.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(structured), 1)
                self.assertEqual(
                    {
                        structured[0]["run_id"],
                        structured[0]["command_id"],
                        structured[0]["decision_id"],
                        structured[0]["request_id"],
                        structured[0]["object_key"],
                        structured[0]["object_level"],
                        structured[0]["actual_action"],
                        structured[0]["status"],
                    },
                    {
                        "run-a",
                        command_row["command_id"],
                        command_row["decision_id"],
                        "request-a",
                        "prefix-a",
                        "prefix",
                        "drop",
                        "completed",
                    },
                )
            finally:
                host.close()

    def test_authenticated_context_client_publishes_to_runtime_host(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", session_id="session-a", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            self.assertLess(len(str(host.action_socket_path).encode("utf-8")), 100)
            try:
                host.start()
                client = AuthenticatedJsonHttpRequestContextClient(
                    host.context_url, run_id="run-a", session_id="session-a", secret=b"a" * 32,
                )
                receipt = client.publish(RuntimeRequestContext("run-a", "request-a", "case-a", "nonce-a", 1.0))
                self.assertEqual(receipt.status, "recorded")
            finally:
                host.close()

    def test_host_authenticates_context_injects_hook_dependencies_and_writes_run_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            captured = {}
            try:
                host.start()
                context = RuntimeRequestContext("run-a", "request-a", "case-a", "nonce-a", 1.0)
                request = Request(
                    host.context_url, data=json.dumps(context.to_record()).encode("utf-8"),
                    headers={"Content-Type": "application/json", **host.context_headers(context)}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
                    receipt = json.loads(response.read().decode("utf-8"))
                self.assertEqual(receipt["status"], "recorded")
                self.assertEqual(
                    host.runtime_identity_for("request-a"),
                    host.runtime_identity_for("request-a").__class__("run-a", "request-a", "nonce-a"),
                )
                self.assertEqual(
                    host.runtime_identity_for("request-a-vllm-suffix"),
                    host.runtime_identity_for("request-a"),
                )
                self.assertEqual(
                    host.runtime_identity_for("chatcmpl-request-a-vllm-suffix"),
                    host.runtime_identity_for("request-a"),
                )

                host.install_hooks(lambda sink, **kwargs: captured.update(sink=sink, **kwargs) or object())
                self.assertIs(captured["binding_registry"], host.binding_registry)
                self.assertIsNotNone(captured["request_context_consumer"])
                self.assertTrue(callable(captured["runtime_request_identity_provider"]))
                captured["sink"]({"record_type": "event", "run_id": "run-a", "event_id": "event-a"})

                self.assertTrue((Path(directory) / "backend_capabilities.json").exists())
                self.assertEqual(
                    json.loads((Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8"))["event_id"], "event-a",
                )
            finally:
                host.close()

    def test_host_action_endpoint_returns_terminal_receipt_for_a_reserved_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            class Manager:
                def __init__(self): self.removed = []
                def remove(self, key, locations=None):
                    self.removed.append((key, locations))
                    return 1
            try:
                host.start()
                context = RequestContext("run-a", "request-a", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
                binding = host.binding_registry.complete_operation(
                    "key-a", HookAction.CACHE_STORE, "completed", context,
                    submitted.event.metadata["operation_lease"],
                ).binding
                host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                assert binding is not None
                manager = Manager()
                host.action_endpoint.action_registration_enabled = True
                host.action_endpoint.register_binding(binding, "key-a", manager)
                lease = host.binding_registry.reserve_action(
                    binding_id=binding.binding_id, binding_generation=binding.binding_generation,
                    backend_object_id=binding.backend_object_id, request_id="request-a",
                    object_key="prefix-a", object_level=ObjectLevel.PREFIX,
                )
                command = BackendActionCommand(
                    run_id="run-a", command_id="command-a", decision_id="decision-a",
                    request_id="request-a", object_key="prefix-a", object_level=ObjectLevel.PREFIX,
                    binding_id=binding.binding_id, backend_object_id=binding.backend_object_id,
                    action=HookAction.DROP, issued_at_ns=time.time_ns(), binding_generation=binding.binding_generation,
                    metadata={"reservation_lease": lease, "deadline_ns": time.time_ns() + 1_000_000_000},
                )
                command = BackendActionCommand(
                    run_id=command.run_id, command_id=command.command_id, decision_id=command.decision_id,
                    request_id=command.request_id, object_key=command.object_key, object_level=command.object_level,
                    binding_id=command.binding_id, backend_object_id=command.backend_object_id,
                    action=command.action, issued_at_ns=command.issued_at_ns, binding_generation=command.binding_generation,
                    metadata={**command.metadata, "command_sha256": command_integrity_digest(command)},
                )
                assert host.action_service is not None
                challenge = host.action_service.new_challenge_for(command)
                proof = host.action_service.issue_runtime_proof(challenge)
                request = Request(
                    host.action_url,
                    data=json.dumps({"command": command.to_record(), "challenge": challenge.to_record(), "proof": proof.__dict__ if hasattr(proof, "__dict__") else {
                        "nonce": proof.nonce, "source": proof.source, "method": proof.method,
                        "session_id": proof.session_id, "mac": proof.mac,
                    }}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
                    receipt = json.loads(response.read().decode("utf-8"))
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(manager.removed, [("key-a", None)])
                command_row = json.loads((Path(directory) / "astrakv_runtime_commands.jsonl").read_text(encoding="utf-8"))
                receipt_row = json.loads((Path(directory) / "runtime_command_receipts.jsonl").read_text(encoding="utf-8"))
                self.assertEqual((command_row["record_type"], command_row["run_id"]), ("command", "run-a"))
                self.assertEqual((receipt_row["record_type"], receipt_row["command_id"]), ("receipt", "command-a"))
            finally:
                host.close()

    def test_write_receipt_preserves_non_drop_action_in_structured_and_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            try:
                host.start()
                command = BackendActionCommand(
                    run_id="run-a", command_id="command-offload", decision_id="decision-offload",
                    request_id="request-a", object_key="prefix-a", object_level=ObjectLevel.PREFIX,
                    binding_id="binding-a", backend_object_id="backend-a", action=HookAction.OFFLOAD,
                    issued_at_ns=1, target_tier="ssd", metadata={}, binding_generation=1,
                )
                host._write_command(command)
                host._write_receipt(type("Receipt", (), {
                    "run_id": "run-a",
                    "command_id": "command-offload",
                    "receipt_id": "receipt-offload",
                    "binding_id": "binding-a",
                    "backend_object_id": "backend-a",
                    "action": HookAction.OFFLOAD,
                    "status": "completed",
                    "timestamp_ns": 2,
                    "tier_before": "cpu",
                    "tier_after": "ssd",
                    "bytes": 1234,
                    "binding_generation": 1,
                    "matches_command": lambda self, other: True,
                    "to_record": lambda self: {
                        "schema": "astrakv-backend-hook-v2",
                        "record_type": "receipt",
                        "run_id": self.run_id,
                        "command_id": self.command_id,
                        "receipt_id": self.receipt_id,
                        "binding_id": self.binding_id,
                        "backend_object_id": self.backend_object_id,
                        "action": self.action.value,
                        "status": self.status,
                        "timestamp_ns": self.timestamp_ns,
                        "tier_before": self.tier_before,
                        "tier_after": self.tier_after,
                        "bytes": self.bytes,
                        "binding_generation": self.binding_generation,
                        "metadata": {},
                        "decision_id": "",
                        "request_id": "",
                        "rejection_reason": "",
                    },
                })())

                structured = [json.loads(line) for line in (Path(directory) / "runtime_structured_events.jsonl").read_text(encoding="utf-8").splitlines()]
                raw_events = [json.loads(line) for line in (Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(structured[0]["actual_action"], "offload")
                self.assertTrue(any(row["action"] == "offload" and row["event_id"] == "command-offload:offload" for row in raw_events))
            finally:
                host.close()

    def test_rejected_receipt_synthesizes_command_and_rejection_event_keeps_binding_id(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            try:
                host.start()
                host._write_receipt(
                    BackendActionReceipt(
                        run_id="run-a",
                        command_id="command-rejected",
                        receipt_id="receipt-rejected",
                        binding_id="binding-a",
                        backend_object_id="backend-a",
                        action=HookAction.PREFETCH,
                        status="rejected",
                        timestamp_ns=2,
                        binding_generation=1,
                        decision_id="decision-rejected",
                        request_id="request-a",
                        rejection_reason="runtime_execution_gate:active_binding",
                        metadata={
                            "pre_execution_rejected": True,
                            "target_tier": "cpu",
                            "object_key": "prefix-a",
                            "object_level": "prefix",
                        },
                    )
                )
                command_rows = [
                    json.loads(line)
                    for line in (Path(directory) / "astrakv_runtime_commands.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                receipt_rows = [
                    json.loads(line)
                    for line in (Path(directory) / "runtime_command_receipts.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(command_rows[0]["command_id"], "command-rejected")
                self.assertEqual(command_rows[0]["object_key"], "prefix-a")
                self.assertEqual(receipt_rows[0]["command_id"], "command-rejected")

                host._write_online_policy_rejection(
                    BackendHookEvent(
                        "run-a",
                        "event-release",
                        "request-a",
                        "prefix-a",
                        ObjectLevel.PREFIX,
                        "backend-a",
                        HookAction.RELEASE,
                        "completed",
                        3,
                        tier_before="ssd",
                        tier_after="ssd",
                        metadata={"binding_id": "binding-a"},
                        binding_generation=1,
                    ),
                    "blocked_by_runtime_gate:active_binding",
                    deadline_ns=4,
                )
                raw_events = [
                    json.loads(line)
                    for line in (Path(directory) / "runtime_events_raw.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                rejected = next(row for row in raw_events if row["status"] == "rejected")
                self.assertEqual(rejected["binding_id"], "binding-a")
                self.assertEqual(rejected["binding_generation"], 1)
            finally:
                host.close()

    def test_backend_capabilities_artifact_advertises_runtime_action_set_after_start(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            ))
            try:
                host.start()
                capabilities = json.loads((Path(directory) / "backend_capabilities.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    tuple(capabilities["allowed_actions"]),
                    ("drop", "offload", "load", "prefetch", "evict"),
                )
                self.assertEqual(capabilities["action_status"]["load"]["status"], "allowed")
                self.assertEqual(capabilities["action_status"]["prefetch"]["status"], "allowed")
                self.assertEqual(capabilities["action_status"]["evict"]["status"], "allowed")
            finally:
                host.close()

    def test_online_controller_never_dispatches_generic_load_after_dynamic_load_target_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id="run-a", state_dir=Path(directory), secret=b"a" * 32,
                engine_instance_id="engine", worker_id="worker", observed_versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
                online_policy_enabled=False, offline_gate_record={"status": "accepted", "reasons": [], "workload_ids": [], "aggregate": {}, "checks": {}}, kv_core_mode=RuntimeMode.ACTIVE,
            ))

            class FakeTensor:
                def element_size(self): return 256
                def numel(self): return 16

            class Engine:
                def __init__(self):
                    self.load_calls = []
                def retrieve(self, tokens, token_mask, **kwargs):
                    self.load_calls.append({
                        "tokens": list(tokens),
                        "token_mask": list(token_mask),
                        "kwargs": dict(kwargs),
                    })
                    return [False, False, True, True]

            class Manager:
                def __init__(self):
                    self.engine = Engine()
                    self.storage_backends = {"LocalDiskBackend": object()}
                    self.lmcache_engine = self.engine
                def batched_contains(self, keys, search_range=None, pin=False):
                    return len(keys), {"LocalDiskBackend": list(keys)}
                def batched_get(self, keys, location=None):
                    return [FakeTensor() for _ in keys]

            manager = Manager()

            try:
                host.start()
                host.install_hooks(lambda *_args, **_kwargs: object())
                context = RequestContext("run-a", "stored-request", "prefix-a", ObjectLevel.PREFIX)
                submitted = host.binding_registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
                completed = host.binding_registry.complete_operation(
                    "key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"],
                )
                released = host.binding_registry.observe("key-a", HookAction.RELEASE, "completed", context)
                assert released.binding is not None
                host.action_endpoint.action_registration_enabled = True
                host.action_endpoint.mark_store_completed(completed.binding, "key-a", manager)
                host.action_endpoint.register_binding(released.binding, "key-a", manager)
                _, updated_binding = host.action_endpoint.register_dynamic_load_target(
                    object_key="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    target_id="load-target-1",
                    runtime_reqmeta_id="reqmeta-1",
                    token_ids=[101, 102, 103, 104],
                    slot_mapping=[0, 1, 2, 3],
                    vllm_cached_tokens=2,
                    lmcache_cached_tokens=4,
                    request_configs={"scenario": "hot_load"},
                    kvcaches=[FakeTensor()],
                    target_tier="gpu",
                )
                assert updated_binding is not None

                assert host.online_bridge is not None
                assert host.online_controller is not None
                host.online_bridge.register_binding(updated_binding)

                cache_hit = BackendHookEvent(
                    run_id="run-a",
                    event_id="cache-hit-hot",
                    request_id=updated_binding.request_id,
                    object_key=updated_binding.object_key,
                    object_level=updated_binding.object_level,
                    backend_object_id=updated_binding.backend_object_id,
                    action=HookAction.CACHE_HIT,
                    status="completed",
                    timestamp_ns=time.time_ns(),
                    tier_before="ssd",
                    tier_after="ssd",
                    bytes=1024,
                    binding_generation=updated_binding.binding_generation,
                    metadata={"binding_id": updated_binding.binding_id},
                )
                load_ready = BackendHookEvent(
                    run_id="run-a",
                    event_id="load-target-ready",
                    request_id=updated_binding.request_id,
                    object_key=updated_binding.object_key,
                    object_level=updated_binding.object_level,
                    backend_object_id=updated_binding.backend_object_id,
                    action=HookAction.CACHE_LOAD,
                    status="available",
                    timestamp_ns=time.time_ns(),
                    tier_before="ssd",
                    tier_after="ssd",
                    bytes=2048,
                    binding_generation=updated_binding.binding_generation,
                    metadata={
                        "binding_id": updated_binding.binding_id,
                        "dispatch_signal": "dynamic_load_target_ready",
                        "load_target_id": "load-target-1",
                        "runtime_reqmeta_id": "reqmeta-1",
                    },
                )
                self.assertTrue(host.online_controller.ingest(cache_hit))
                self.assertTrue(host.online_controller.ingest(load_ready))
                host.online_controller.execution_enabled = True
                decision = host.online_controller.propose_for("prefix-a", ObjectLevel.PREFIX)
                self.assertEqual(decision.predicted_action, "load")
                result = host.online_controller.dispatch(decision)
                self.assertEqual(result.status, "native_connector_required")
                self.assertFalse(host.online_bridge.commands)
                self.assertFalse(manager.engine.load_calls)
            finally:
                host.close()


if __name__ == "__main__":
    unittest.main()
