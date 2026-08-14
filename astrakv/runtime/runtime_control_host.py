"""Runtime-owned assembly for the version-locked LMCache control plane."""

from __future__ import annotations

import json
import importlib.metadata
import os
import queue
import secrets
import hashlib
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry
from astrakv.runtime.backend_capabilities import (
    build_installation_evidence,
    preflight_backend_capabilities,
)
from astrakv.runtime.circuit_breaker import CircuitBreaker
from astrakv.runtime.lmcache047_action_service import (
    ProtectedRuntimeActionService,
    UnixDomainSocketActionServer,
    _challenge_from_record,
    _proof_from_record,
    command_integrity_digest,
)
from astrakv.runtime.lmcache047_runtime_patch import (
    LMCache047ActionEndpoint,
    LMCache047RequestContextConsumer,
)
from astrakv.runtime.request_context import (
    RequestContextAssociationJsonlArtifact,
    RequestContextReceipt,
    RuntimeRequestContext,
    RuntimeRequestContextAuthority,
    RuntimeRequestContextReceiver,
    RuntimeRequestIdentity,
)
from astrakv.runtime.runtime_execution_gate import RuntimeExecutionGate
from astrakv.runtime.backend_hook import BackendActionCommand, BackendActionReceipt, BackendHookEvent, BackendObjectBinding, HookAction
from astrakv.runtime.backend_bridge import OnlineBackendBridge
from astrakv.runtime.artifact_contract import canonical_artifact_path
from astrakv.runtime.lmcache047_action_service import InMemoryRuntimeActionClient
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.offline_safety import OfflineSafetyGate, OfflineSafetyGateResult
from astrakv.runtime.online_controller import OnlinePolicyController, OnlinePolicyControllerConfig
from astrakv.runtime.online_profile import OnlineProfileStore
from astrakv.runtime.prediction_sidecar import PredictionSidecarIndex
from astrakv.runtime.profile_db import ProfileDB
from astrakv.runtime.scheduler_hints import SchedulerHintIndex
from astrakv.runtime.kv_runtime_core import RuntimeMode


@dataclass(frozen=True, slots=True)
class RuntimeControlHostConfig:
    run_id: str
    state_dir: Path
    secret: bytes
    engine_instance_id: str
    worker_id: str
    context_port: int = 0
    session_id: str = ""
    observed_versions: dict[str, str] | None = None
    online_policy_enabled: bool = False
    online_policy_dispatch_on_release: bool = True
    offline_gate_record: dict[str, Any] | None = None
    online_policy_queue_size: int = 64
    online_policy_dispatch_deadline_s: float = 30.0
    prediction_sidecar_path: Path | None = None
    profile_db_path: Path | None = None
    scheduler_hints_path: Path | None = None
    online_prefetch_dispatch_enabled: bool = True
    online_prefetch_mode: str = "disabled"
    online_evict_dispatch_enabled: bool = True
    evict_pressure_gate_enabled: bool = True
    evict_pressure_trigger: float = 0.8
    evict_cpu_capacity_bytes: int = 0
    evict_ssd_capacity_bytes: int = 0
    evict_cold_score_threshold: float = 0.35
    global_evict_scan_enabled: bool = True
    global_evict_scan_min_interval_s: float = 5.0
    global_evict_scan_max_victims: int = 4
    evict_periodic_scan_enabled: bool = False
    evict_periodic_scan_interval_s: float = 1.0
    kv_core_mode: RuntimeMode = RuntimeMode.OFF

    def __post_init__(self) -> None:
        if not all((self.run_id, self.engine_instance_id, self.worker_id)):
            raise ValueError("run_id, engine_instance_id, and worker_id are required")
        if len(self.secret) < 32:
            raise ValueError("runtime control host secret must be at least 32 bytes")
        if not 0 <= self.context_port <= 65535:
            raise ValueError("context_port must be between 0 and 65535")
        if self.online_policy_queue_size <= 0:
            raise ValueError("online_policy_queue_size must be positive")
        if not self.online_policy_dispatch_deadline_s > 0:
            raise ValueError("online_policy_dispatch_deadline_s must be positive")
        if not isinstance(self.kv_core_mode, RuntimeMode):
            raise ValueError("kv_core_mode must be a RuntimeMode")


@dataclass(frozen=True, slots=True)
class _OnlinePolicyTask:
    event: BackendHookEvent
    deadline_ns: int | None


def _measured_runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("vllm", "lmcache"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = ""
    return result


class RuntimeControlHost:
    """Own one run's context ingress, Hook dependencies, and action authority."""

    def __init__(self, config: RuntimeControlHostConfig) -> None:
        self.config = config
        self.state_dir = config.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = config.session_id or secrets.token_urlsafe(24)
        self.binding_registry = BackendBindingRegistry(
            run_id=config.run_id,
            engine_instance_id=config.engine_instance_id,
            worker_id=config.worker_id,
        )
        self.authority = RuntimeRequestContextAuthority.install(
            run_id=config.run_id, session_id=self.session_id, secret=config.secret,
        )
        self.context_receiver: RuntimeRequestContextReceiver | None = None
        self.context_consumer: LMCache047RequestContextConsumer | None = None
        self._association_artifact: RequestContextAssociationJsonlArtifact | None = None
        self.action_endpoint = LMCache047ActionEndpoint(binding_registry=self.binding_registry)
        self.action_service: ProtectedRuntimeActionService | None = None
        self.action_server: UnixDomainSocketActionServer | None = None
        self.execution_gate: RuntimeExecutionGate | None = None
        self._runtime_identities: dict[str, RuntimeRequestIdentity] = {}
        self._kv_runtime_bridge: Any | None = None
        self._identity_lock = threading.RLock()
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._event_lock = threading.Lock()
        self._artifact_command_ids: set[str] = set()
        self._artifact_receipt_command_ids: set[str] = set()
        self._commands_by_id: dict[str, BackendActionCommand] = {}
        self.online_bridge: OnlineBackendBridge | None = None
        self.online_controller: OnlinePolicyController | None = None
        self._online_policy_queue: queue.Queue[_OnlinePolicyTask] = queue.Queue(config.online_policy_queue_size)
        self._online_policy_stop = threading.Event()
        self._online_policy_thread: threading.Thread | None = None
        self._evict_scan_stop = threading.Event()
        self._evict_scan_thread: threading.Thread | None = None
        self._evict_scan_lock = threading.Lock()
        socket_id = hashlib.sha256(f"{config.run_id}:{self.session_id}".encode("utf-8")).hexdigest()[:24]
        self.action_socket_path = Path(tempfile.gettempdir()) / f"astrakv-{socket_id}.sock"

    @property
    def context_url(self) -> str:
        if self._http_server is None:
            raise RuntimeError("runtime control host is not started")
        return f"http://127.0.0.1:{self._http_server.server_port}/request-context"

    @property
    def action_url(self) -> str:
        if self._http_server is None:
            raise RuntimeError("runtime control host is not started")
        return f"http://127.0.0.1:{self._http_server.server_port}/actions"

    def start(self) -> None:
        if self._http_server is not None:
            return
        host = self

        class ContextHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/request-context", "/actions", "/actions/runtime-proof"}:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1_000_000:
                        raise ValueError("invalid request context payload length")
                    record = json.loads(self.rfile.read(length).decode("utf-8"))
                    if self.path == "/request-context":
                        receiver = host.context_receiver
                        if receiver is None:
                            raise RuntimeError("receiver is not ready")
                        context = RuntimeRequestContext.from_record(record)
                        payload = receiver.receive(record, self.headers).to_record()
                        host.register_runtime_identity(
                            context.request_id,
                            RuntimeRequestIdentity(context.run_id, context.request_id, context.request_nonce),
                        )
                        bridge = host._kv_runtime_bridge
                        if bridge is not None:
                            bridge.ingress_request(context)
                    else:
                        service = host.action_service
                        if service is None:
                            raise RuntimeError("action service is not ready")
                        challenge = _challenge_from_record(record["challenge"])
                        if self.path == "/actions/runtime-proof":
                            proof = service.issue_runtime_proof(challenge)
                            payload = {
                                "nonce": proof.nonce, "source": proof.source, "method": proof.method,
                                "session_id": proof.session_id, "mac": proof.mac,
                            }
                        else:
                            command = BackendActionCommand.from_record(record["command"])
                            host._write_command(command)
                            receipt = service.submit(
                                command, challenge,
                                _proof_from_record(record["proof"]),
                            )
                            host._write_receipt(receipt)
                            payload = receipt.to_record()
                    self.send_response(200)
                except (TypeError, ValueError) as exc:
                    payload = {"error": type(exc).__name__}
                    self.send_response(400)
                encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._http_server = ThreadingHTTPServer(("127.0.0.1", self.config.context_port), ContextHandler)
        self.context_receiver = RuntimeRequestContextReceiver(self.context_url, self.authority)
        self._association_artifact = RequestContextAssociationJsonlArtifact(
            self.state_dir / "request_context_associations.jsonl"
        )
        self.context_consumer = LMCache047RequestContextConsumer(
            self.context_receiver,
            association_sink=self._association_artifact.append,
        )
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, name="astrakv-request-context", daemon=True)
        self._http_thread.start()

        versions = self.config.observed_versions or _measured_runtime_versions()
        installation_evidence = build_installation_evidence(
            source="astrakv-runtime-control-host", method="lmcache047-localdisk", session_id=self.session_id,
            vllm_version=versions.get("vllm"), lmcache_version=versions.get("lmcache"), connector_name="lmcache-vllm-v1",
            connector_version="0.4.7", endpoint_identity=self.action_url,
        )
        preflight = preflight_backend_capabilities(
            vllm_version=versions.get("vllm"), lmcache_version=versions.get("lmcache"), run_id=self.config.run_id,
            hook_url=self.action_url, installation_evidence=installation_evidence,
            connector_name="lmcache-vllm-v1", connector_version="0.4.7", available_actions=("drop", "offload", "load", "prefetch", "evict"),
            available_object_levels=("prefix",), binding_generation_observed=True,
        )
        canonical_artifact_path(self.state_dir, "backend_capabilities").write_text(
            json.dumps(preflight.to_record(), ensure_ascii=False, sort_keys=True), encoding="utf-8",
        )
        self.execution_gate = RuntimeExecutionGate(
            run_id=self.config.run_id, endpoint=self.action_url, capabilities=preflight,
            binding_registry=self.binding_registry, breaker=CircuitBreaker(state_path=self.state_dir / "circuit_breaker.json"),
            state_path=self.state_dir / "execution_gate.json",
        )
        self.action_service = ProtectedRuntimeActionService(
            action_endpoint=self.action_endpoint, state_dir=self.state_dir / "action_ledger", secret=self.config.secret,
            source="astrakv-runtime-control-host", method="lmcache047-localdisk", session_id=self.session_id,
        )
        self.action_server = UnixDomainSocketActionServer(
            self.action_service, self.action_socket_path, admit_drop=self.admit_drop,
        )
        if os.name == "posix":
            self.action_server.start()

    def close(self) -> None:
        self._online_policy_stop.set()
        if self._online_policy_thread is not None:
            self._online_policy_thread.join(timeout=1)
            self._online_policy_thread = None
        self._evict_scan_stop.set()
        if self._evict_scan_thread is not None:
            self._evict_scan_thread.join(timeout=1)
            self._evict_scan_thread = None
        if self.action_server is not None:
            self.action_server.close()
            self.action_server = None
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
            self._http_thread = None

    def admit_drop(self, binding: BackendObjectBinding) -> BackendActionReceipt:
        """Atomically admit and dispatch one owner-only LocalDisk DROP."""
        service = self.action_service
        gate = self.execution_gate
        if service is None or gate is None:
            raise RuntimeError("runtime action service is not ready")
        if binding.run_id != self.config.run_id or not binding.verified:
            raise ValueError("binding is not eligible for this runtime host")
        current = self.binding_registry.current_binding(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            request_id=binding.request_id, object_key=binding.object_key, object_level=binding.object_level,
        )
        if current is None or current.backend_object_id != binding.backend_object_id:
            raise ValueError("binding is not current")

        issued_at_ns = time.time_ns()
        decision_id = f"owner-uds-drop:{secrets.token_hex(16)}"
        command_id = f"{self.config.run_id}:{decision_id}"
        metadata = {"deadline_ns": issued_at_ns + 30_000_000_000}
        admission_command = BackendActionCommand(
            run_id=self.config.run_id, command_id=command_id, decision_id=decision_id,
            request_id=binding.request_id, object_key=binding.object_key, object_level=binding.object_level,
            binding_id=binding.binding_id, backend_object_id=binding.backend_object_id,
            action=HookAction.DROP, issued_at_ns=issued_at_ns, binding_generation=binding.binding_generation,
            metadata=metadata,
        )
        admission = gate.authorize(admission_command, now_ns=issued_at_ns)
        if not admission.allowed:
            raise ValueError(f"runtime admission rejected: {admission.reason}")
        reservation_lease = self.binding_registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id=binding.request_id,
            object_key=binding.object_key, object_level=binding.object_level,
            deadline_ns=metadata["deadline_ns"],
        )
        if reservation_lease is None:
            gate.complete(command_id)
            raise ValueError("binding reservation rejected")
        command = BackendActionCommand(
            run_id=admission_command.run_id, command_id=command_id, decision_id=decision_id,
            request_id=binding.request_id, object_key=binding.object_key, object_level=binding.object_level,
            binding_id=binding.binding_id, backend_object_id=binding.backend_object_id,
            action=HookAction.DROP, issued_at_ns=issued_at_ns, binding_generation=binding.binding_generation,
            metadata={**metadata, "reservation_lease": reservation_lease},
        )
        command = BackendActionCommand(
            run_id=command.run_id, command_id=command.command_id, decision_id=command.decision_id,
            request_id=command.request_id, object_key=command.object_key, object_level=command.object_level,
            binding_id=command.binding_id, backend_object_id=command.backend_object_id,
            action=command.action, issued_at_ns=command.issued_at_ns,
            binding_generation=command.binding_generation,
            metadata={**command.metadata, "command_sha256": command_integrity_digest(command)},
        )
        self._write_command(command)
        try:
            challenge = service.new_challenge_for(command)
            receipt = service.submit(command, challenge, service.issue_runtime_proof(challenge))
            self._write_receipt(receipt)
            return receipt
        finally:
            gate.complete(command_id)

    def context_headers(self, context: RuntimeRequestContext) -> dict[str, str]:
        return self.authority.context_headers(context, self.context_url)

    def register_runtime_identity(self, runtime_request_id: str, identity: RuntimeRequestIdentity) -> None:
        if identity.run_id != self.config.run_id:
            raise ValueError("runtime identity run_id does not match host")
        with self._identity_lock:
            prior = self._runtime_identities.get(runtime_request_id)
            if prior is not None and prior != identity:
                raise ValueError("runtime request identity conflict")
            self._runtime_identities[runtime_request_id] = identity

    def associate_runtime_request(self, runtime_request_id: str) -> RequestContextReceipt | None:
        """Associate a concrete native ReqMeta ID through the host-owned consumer.

        The vendor connector is the only component allowed to supply the native
        ID.  A missing or ambiguous identity is deliberately a no-op so a
        connector cannot manufacture a logical binding from benchmark data.
        """
        runtime_request_id = str(runtime_request_id or "")
        consumer = self.context_consumer
        if not runtime_request_id or consumer is None:
            return None
        identity = self.runtime_identity_for(runtime_request_id)
        if identity is None:
            return None
        if consumer.associate(runtime_request_id, identity) is None:
            return None
        return consumer.receipt_for(runtime_request_id)

    def runtime_identity_for(self, runtime_request_id: str) -> RuntimeRequestIdentity | None:
        with self._identity_lock:
            exact = self._runtime_identities.get(runtime_request_id)
            if exact is not None:
                return exact
            matches = [
                identity for request_id, identity in self._runtime_identities.items()
                if (
                    runtime_request_id.startswith(request_id + "-")
                    and runtime_request_id[len(request_id) + 1:]
                )
                or (
                    runtime_request_id.startswith("chatcmpl-" + request_id + "-")
                    and runtime_request_id[len("chatcmpl-" + request_id) + 1:]
                )
            ]
            return matches[0] if len(matches) == 1 else None

    def register_kv_runtime_bridge(self, bridge: Any) -> None:
        """Attach the process-local vendor bridge to authenticated ingress.

        vLLM 0.23 may construct scheduler-side and worker-side LMCache
        connectors in one single-worker EngineCore.  They share this host;
        only the first bridge owns ingress, while each connector keeps its own
        native callback state.  This is idempotent within one process and does
        not permit sharing a host across processes.
        """
        if bridge is None:
            raise ValueError("KV runtime bridge is required")
        with self._identity_lock:
            if self._kv_runtime_bridge is None:
                self._kv_runtime_bridge = bridge

    def install_hooks(self, installer: Callable[..., LMCache047ActionEndpoint]) -> LMCache047ActionEndpoint:
        if self.context_consumer is None:
            raise RuntimeError("runtime control host must be started before installing hooks")
        endpoint = installer(
            self._event_sink,
            binding_registry=self.binding_registry,
            request_context_consumer=self.context_consumer,
            runtime_request_identity_provider=self.runtime_identity_for,
        )
        if isinstance(endpoint, LMCache047ActionEndpoint):
            self.action_endpoint = endpoint
            if self.action_service is not None:
                self.action_service.action_endpoint = endpoint
        self._install_online_policy()
        return endpoint

    def _install_online_policy(self) -> None:
        if self.online_controller is not None or self.action_service is None or self.execution_gate is None:
            return
        gate_record = self.config.offline_gate_record or {}
        gate = OfflineSafetyGate.from_record(gate_record) if gate_record else OfflineSafetyGate(
            OfflineSafetyGateResult("rejected", ("offline_gate_record_required",), (), {}, {})
        )
        self.online_bridge = OnlineBackendBridge(
            run_id=self.config.run_id, bindings=[],
            hook_client=InMemoryRuntimeActionClient(self.action_service, endpoint_identity=self.action_url),
            hook_url=self.action_url, gate=gate, capabilities=self.execution_gate.capabilities,
            binding_registry=self.binding_registry, execution_gate=self.execution_gate,
        )
        prediction_source = None
        if self.config.prediction_sidecar_path is not None:
            prediction_source = PredictionSidecarIndex.from_jsonl(
                self.config.prediction_sidecar_path,
                run_id=self.config.run_id,
            )
        profile_db = None
        if self.config.profile_db_path is not None and self.config.profile_db_path.exists():
            profile_db = ProfileDB.load(self.config.profile_db_path)
        scheduler_hints = None
        if self.config.scheduler_hints_path is not None and self.config.scheduler_hints_path.exists():
            scheduler_hints = SchedulerHintIndex.from_jsonl(self.config.scheduler_hints_path)
        self.online_controller = OnlinePolicyController(
            run_id=self.config.run_id, workload_id=self.config.run_id, bridge=self.online_bridge,
            profile_store=OnlineProfileStore(
                run_id=self.config.run_id,
                checkpoint_path=canonical_artifact_path(self.state_dir, "online_profile_checkpoint"),
            ),
            config=OnlinePolicyControllerConfig(
                enable_prefetch_dispatch=self.config.online_prefetch_dispatch_enabled,
                online_prefetch_mode=self.config.online_prefetch_mode,
                evict_dispatch_enabled=self.config.online_evict_dispatch_enabled,
                evict_pressure_gate_enabled=self.config.evict_pressure_gate_enabled,
                evict_pressure_trigger=self.config.evict_pressure_trigger,
                evict_cpu_capacity_bytes=self.config.evict_cpu_capacity_bytes,
                evict_ssd_capacity_bytes=self.config.evict_ssd_capacity_bytes,
                evict_cold_score_threshold=self.config.evict_cold_score_threshold,
                global_evict_scan_enabled=self.config.global_evict_scan_enabled,
                global_evict_scan_min_interval_s=self.config.global_evict_scan_min_interval_s,
                global_evict_scan_max_victims=self.config.global_evict_scan_max_victims,
                kv_core_mode=self.config.kv_core_mode,
            ),
            prediction_source=prediction_source,
            profile_db=profile_db,
            scheduler_hints=scheduler_hints,
        )
        self.online_controller.execution_enabled = self.config.online_policy_enabled
        self._online_policy_stop.clear()
        self._online_policy_thread = threading.Thread(
            target=self._run_online_policy_worker,
            name="astrakv-online-policy",
            daemon=True,
        )
        self._online_policy_thread.start()
        if self.config.evict_periodic_scan_enabled:
            self._evict_scan_stop.clear()
            self._evict_scan_thread = threading.Thread(
                target=self._run_evict_scan_loop,
                name="astrakv-evict-scan",
                daemon=True,
            )
            self._evict_scan_thread.start()

    def _event_sink(self, record: dict[str, Any]) -> None:
        if str(record.get("run_id") or "") != self.config.run_id:
            return
        record = dict(record)
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            for field in ("binding_id", "binding_generation"):
                if field not in record and field in metadata:
                    record[field] = metadata[field]
        record_type = str(record.get("record_type") or "observation")
        filename = (
            canonical_artifact_path(self.state_dir, "backend_binding_events").name
            if record_type == "binding"
            else canonical_artifact_path(self.state_dir, "runtime_events_raw").name
        )
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._event_lock:
            with (self.state_dir / filename).open("a", encoding="utf-8") as handle:
                handle.write(encoded)
        if self.online_bridge is not None and record_type == "binding":
            try:
                self.online_bridge.register_binding(BackendObjectBinding.from_record(record))
            except ValueError:
                return
        if self.online_controller is not None and record_type == "event":
            try:
                event = BackendHookEvent.from_record(record)
            except ValueError:
                return
            self._enqueue_online_policy_event(event)

    def _enqueue_online_policy_event(self, event: BackendHookEvent) -> None:
        deadline_ns = _online_policy_deadline_ns(
            event,
            dispatch_deadline_s=self.config.online_policy_dispatch_deadline_s,
            dispatch_on_release=self.config.online_policy_dispatch_on_release,
        )
        try:
            self._online_policy_queue.put_nowait(_OnlinePolicyTask(event=event, deadline_ns=deadline_ns))
        except queue.Full:
            self._write_online_policy_rejection(event, "policy_queue_full", action=event.action)

    def _run_online_policy_worker(self) -> None:
        while not self._online_policy_stop.is_set():
            try:
                task = self._online_policy_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._online_policy_stop.is_set():
                    continue
                controller = self.online_controller
                bridge = self.online_bridge
                if controller is None or bridge is None or not controller.ingest(task.event):
                    continue
                if not _should_attempt_online_dispatch(
                    task.event,
                    dispatch_on_release=self.config.online_policy_dispatch_on_release,
                ):
                    continue
                if not controller.execution_enabled:
                    continue
                if task.deadline_ns is not None and time.time_ns() >= task.deadline_ns:
                    self._write_online_policy_rejection(task.event, "policy_dispatch_deadline_expired", deadline_ns=task.deadline_ns)
                    continue
                try:
                    decision = controller.propose_for(
                        task.event.object_key,
                        task.event.object_level,
                        request_id=task.event.request_id,
                        binding_id=str(task.event.metadata.get("binding_id") or ""),
                    )
                    decision = replace(decision, metadata={**decision.metadata, "deadline_ns": task.deadline_ns})
                    result = controller.dispatch(decision)
                except Exception as exc:
                    self._write_online_policy_rejection(task.event, f"policy_dispatch_exception:{type(exc).__name__}", deadline_ns=task.deadline_ns)
                    continue
                if result.receipt is not None:
                    self._write_receipt(result.receipt)
                if result.event is None:
                    if result.status in {"advisory_only", "no_dispatch_required"}:
                        continue
                    self._write_online_policy_rejection(task.event, result.status, deadline_ns=task.deadline_ns)
                    continue
                with self._evict_scan_lock:
                    if task.event.action is HookAction.RELEASE and task.event.status == "completed":
                        for _decision, _result in controller.global_evict_scan():
                            if _result.receipt is not None:
                                self._write_receipt(_result.receipt)
                    for command in bridge.commands:
                        self._write_command(command)
                    for receipt in bridge.receipts:
                        self._write_receipt(receipt)
            finally:
                self._online_policy_queue.task_done()

    def _run_evict_scan_loop(self) -> None:
        """Periodic watermark-style eviction scan (mirrors LMCache's loop).

        When enabled, checks pressure on a fixed interval and dispatches
        evict-B decisions through the normal receipt-backed chain.
        """
        interval = max(0.1, float(self.config.evict_periodic_scan_interval_s))
        while not self._evict_scan_stop.wait(interval):
            controller = self.online_controller
            bridge = self.online_bridge
            if controller is None or bridge is None or not controller.execution_enabled:
                continue
            with self._evict_scan_lock:
                try:
                    results = controller.global_evict_scan()
                except Exception:
                    continue
                for _decision, _result in results:
                    if _result.receipt is not None:
                        self._write_receipt(_result.receipt)
                for command in bridge.commands:
                    self._write_command(command)
                for receipt in bridge.receipts:
                    self._write_receipt(receipt)

    def _write_online_policy_rejection(
        self, event: BackendHookEvent, reason: str, *, action: HookAction = HookAction.DROP, deadline_ns: int | None = None,
    ) -> None:
        metadata = {
            **event.metadata,
            "binding_id": event.metadata.get("binding_id", ""),
            "policy_rejection_reason": reason,
        }
        if deadline_ns is not None:
            metadata["deadline_ns"] = deadline_ns
        record = BackendHookEvent(
            run_id=event.run_id, event_id=f"{event.event_id}:online-policy-rejected:{time.time_ns()}",
            request_id=event.request_id, object_key=event.object_key, object_level=event.object_level,
            backend_object_id=event.backend_object_id, action=action, status="rejected", timestamp_ns=time.time_ns(),
            tier_before=event.tier_before, tier_after=event.tier_after, bytes=event.bytes,
            binding_generation=event.binding_generation, metadata=metadata,
        ).to_record()
        record["binding_id"] = str(metadata.get("binding_id") or "")
        with self._event_lock:
            self._append_artifact(
                canonical_artifact_path(self.state_dir, "runtime_events_raw").name,
                record,
            )

    def _write_command(self, command: BackendActionCommand) -> None:
        if command.run_id != self.config.run_id:
            return
        with self._event_lock:
            if command.command_id in self._artifact_command_ids:
                return
            self._artifact_command_ids.add(command.command_id)
            self._commands_by_id[command.command_id] = command
            self._append_artifact(
                canonical_artifact_path(self.state_dir, "astrakv_runtime_commands").name,
                command.to_record(),
            )

    def _write_receipt(self, receipt: Any) -> None:
        if receipt.run_id != self.config.run_id:
            return
        if receipt.command_id not in self._commands_by_id:
            synthetic = self._command_from_receipt(receipt)
            if synthetic is not None:
                self._write_command(synthetic)
        with self._event_lock:
            if receipt.command_id in self._artifact_receipt_command_ids:
                return
            self._artifact_receipt_command_ids.add(receipt.command_id)
            self._append_artifact(
                canonical_artifact_path(self.state_dir, "runtime_command_receipts").name,
                receipt.to_record(),
            )
        command = self._commands_by_id.get(receipt.command_id)
        if command is not None and receipt.matches_command(command):
            receipt_metadata = dict(getattr(receipt, "metadata", {}) or {})
            self._write_runtime_structured_event(command, receipt)
            self._event_sink(BackendHookEvent(
                run_id=command.run_id, event_id=f"{receipt.command_id}:{receipt.action.value}", request_id=command.request_id,
                object_key=command.object_key, object_level=command.object_level,
                backend_object_id=command.backend_object_id, action=receipt.action, status=receipt.status,
                timestamp_ns=receipt.timestamp_ns, tier_before=receipt.tier_before, tier_after=receipt.tier_after,
                bytes=receipt.bytes, binding_generation=command.binding_generation,
                metadata={
                    "binding_id": command.binding_id,
                    "command_id": command.command_id,
                    "receipt_id": receipt.receipt_id,
                    **receipt_metadata,
                },
            ).to_record())

    def _write_runtime_structured_event(
        self, command: BackendActionCommand, receipt: BackendActionReceipt,
    ) -> None:
        capabilities = self.execution_gate.capabilities if self.execution_gate is not None else None
        backend_versions = {
            "vllm": None if capabilities is None else capabilities.vllm_version,
            "lmcache": None if capabilities is None else capabilities.lmcache_version,
        }
        actual_action = "load" if receipt.action is HookAction.CACHE_LOAD else receipt.action.value
        receipt_metadata = dict(getattr(receipt, "metadata", {}) or {})
        record = {
            "schema": "astrakv-runtime-structured-event-v1",
            "run_id": command.run_id,
            "runtime_event_id": f"{receipt.command_id}:structured",
            "command_id": command.command_id,
            "decision_id": command.decision_id,
            "request_id": command.request_id,
            "object_key": command.object_key,
            "object_level": command.object_level.value,
            "actual_action": actual_action,
            "tier_before": receipt.tier_before,
            "tier_after": receipt.tier_after,
            "bytes": receipt.bytes,
            "status": receipt.status,
            "timestamp_ns": receipt.timestamp_ns,
            "backend_version": backend_versions,
            "connector_version": None if capabilities is None else capabilities.connector_version,
            "provenance": "runtime_structured",
            "binding_id": command.binding_id,
            "backend_object_id": command.backend_object_id,
            "binding_generation": command.binding_generation,
            "receipt_id": receipt.receipt_id,
            "metadata": receipt_metadata,
        }
        with self._event_lock:
            self._append_artifact(
                canonical_artifact_path(self.state_dir, "runtime_structured_events").name,
                record,
            )

    def _command_from_receipt(self, receipt: Any) -> BackendActionCommand | None:
        metadata = dict(getattr(receipt, "metadata", {}) or {})
        if not metadata.get("pre_execution_rejected"):
            return None
        object_key = str(metadata.get("object_key") or "")
        object_level = str(metadata.get("object_level") or "")
        if not object_key or not object_level:
            return None
        try:
            action = receipt.action if isinstance(receipt.action, HookAction) else HookAction(str(receipt.action))
            object_level_enum = ObjectLevel(object_level)
            issued_at_ns = int(getattr(receipt, "timestamp_ns", time.time_ns()))
        except (TypeError, ValueError):
            return None
        try:
            return BackendActionCommand(
                run_id=str(receipt.run_id),
                command_id=str(receipt.command_id),
                decision_id=str(getattr(receipt, "decision_id", "") or ""),
                request_id=str(getattr(receipt, "request_id", "") or ""),
                object_key=object_key,
                object_level=object_level_enum,
                binding_id=str(receipt.binding_id),
                backend_object_id=str(receipt.backend_object_id),
                action=action,
                issued_at_ns=issued_at_ns,
                target_tier=str(metadata.get("target_tier") or "unknown"),
                metadata=metadata,
                binding_generation=int(getattr(receipt, "binding_generation", 1) or 1),
            )
        except (TypeError, ValueError):
            return None

    def _append_artifact(self, filename: str, record: dict[str, Any]) -> None:
        with (self.state_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _should_attempt_online_dispatch(
    event: BackendHookEvent,
    *,
    dispatch_on_release: bool = True,
) -> bool:
    if dispatch_on_release and event.action is HookAction.RELEASE and event.status == "completed":
        return True
    if event.action is HookAction.OFFLOAD and event.status in {"completed", "ok", "executed"}:
        return True
    return (
        event.action is HookAction.CACHE_LOAD
        and event.status == "available"
        and str(event.metadata.get("dispatch_signal") or "") == "dynamic_load_target_ready"
    )


def _online_policy_deadline_ns(
    event: BackendHookEvent,
    *,
    dispatch_deadline_s: float,
    dispatch_on_release: bool = True,
) -> int | None:
    if not _should_attempt_online_dispatch(event, dispatch_on_release=dispatch_on_release):
        return None
    return time.time_ns() + int(dispatch_deadline_s * 1_000_000_000)
