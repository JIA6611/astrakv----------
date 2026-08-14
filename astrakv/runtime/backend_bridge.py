"""Safety-gated transport from AstraKV decisions to a public backend Hook."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from astrakv.runtime.backend_capabilities import (
    BackendCapabilityPreflight,
    RuntimeProbeProof,
    RuntimeProbeChallenge,
    normalize_loopback_endpoint,
)
from astrakv.runtime.backend_binding_registry import BackendBindingRegistry
from astrakv.runtime.backend_hook import (
    BackendActionCommand,
    BackendActionReceipt,
    BackendHookEvent,
    BackendObjectBinding,
    HookAction,
)
from astrakv.runtime.lmcache047_action_service import InMemoryRuntimeActionClient, command_integrity_digest
from astrakv.runtime.eviction import OfflineEvictionDecision, RuntimeActionResult, RuntimeEvictionEvent
from astrakv.runtime.offline_safety import OfflineSafetyGate
from astrakv.runtime.runtime_execution_gate import RuntimeExecutionGate


EXECUTION_ACTIONS = {
    "prefetch": HookAction.PREFETCH,
    "offload": HookAction.OFFLOAD,
    "load": HookAction.CACHE_LOAD,
    "evict": HookAction.EVICT,
    "drop": HookAction.DROP,
}
ACTIVE_ACTIONS = {HookAction.CACHE_CREATE, HookAction.CACHE_STORE, HookAction.CACHE_HIT, HookAction.CACHE_LOAD}


class BackendHookClient(Protocol):
    @property
    def endpoint_identity(self) -> str:
        """Canonical loopback endpoint used by this transport."""

    def submit(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof,
    ) -> BackendActionReceipt:
        """Issue a command to the backend Hook and return its structured receipt."""

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        """Ask the installed runtime/action service to sign a fresh challenge."""


@dataclass(frozen=True, slots=True)
class JsonHttpHookClient:
    hook_url: str
    timeout_s: float = 10.0
    proof_url: str | None = None
    endpoint_identity: str = field(init=False)

    def __post_init__(self) -> None:
        endpoint = _literal_ip_loopback_endpoint(self.hook_url)
        if endpoint is None:
            raise ValueError("production backend Hook URL must use a literal IP loopback host")
        object.__setattr__(self, "endpoint_identity", endpoint)

    def submit(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof,
    ) -> BackendActionReceipt:
        request = Request(
            self.hook_url,
            data=json.dumps({
                "command": command.to_record(), "challenge": challenge.to_record(), "proof": _proof_record(proof),
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        with opener.open(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("backend Hook response must be a JSON object")
        return BackendActionReceipt.from_record(payload)

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        url = self.proof_url or f"{self.hook_url.rstrip('/')}/runtime-proof"
        if not _same_literal_ip_loopback_peer(url, self.hook_url):
            raise ValueError("runtime proof URL must use the same literal IP loopback endpoint")
        request = Request(
            url,
            data=json.dumps(challenge.to_record()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        with opener.open(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("runtime proof response must be a JSON object")
        try:
            return RuntimeProbeProof(
                nonce=str(payload["nonce"]), source=str(payload["source"]), method=str(payload["method"]),
                session_id=str(payload["session_id"]), mac=str(payload["mac"]),
            )
        except KeyError as exc:
            raise ValueError("runtime proof response is incomplete") from exc


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None

    def http_error_302(self, request: Any, fp: Any, code: int, msg: str, headers: Any) -> None:
        raise HTTPError(request.full_url, code, "backend Hook redirects are forbidden", headers, fp)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


@dataclass(frozen=True, slots=True)
class InMemoryLoopbackHookClient:
    """Trusted in-process loopback transport for tests and the future action service."""

    hook_url: str
    action_service: "InMemoryProtectedActionService"
    endpoint_identity: str = field(init=False)

    def __post_init__(self) -> None:
        endpoint = normalize_loopback_endpoint(self.hook_url)
        if endpoint is None:
            raise ValueError("backend Hook URL must use a loopback host")
        object.__setattr__(self, "endpoint_identity", endpoint)

    def submit(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof,
    ) -> BackendActionReceipt:
        return self.action_service.submit(command, challenge, proof)

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        return self.action_service.issue_runtime_proof(challenge)


class InMemoryProtectedActionService:
    """Test-only protected runtime action service owning its deployment secret."""

    def __init__(self, responder: Callable[[BackendActionCommand], BackendActionReceipt], *, source: str, method: str, session_id: str) -> None:
        self._responder = responder
        self._source = source
        self._method = method
        self._session_id = session_id
        self._secret = secrets.token_bytes(32)

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        return RuntimeProbeProof(
            nonce=challenge.nonce, source=self._source, method=self._method, session_id=self._session_id,
            mac=hmac.new(self._secret, _challenge_payload(challenge, self._source, self._method, self._session_id), hashlib.sha256).hexdigest(),
        )

    def submit(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof,
    ) -> BackendActionReceipt:
        if not self._valid_proof(challenge, proof) or not _command_matches_challenge(command, challenge):
            return BackendActionReceipt(
                run_id=command.run_id, command_id=command.command_id, receipt_id=f"rejected-{command.command_id}",
                binding_id=command.binding_id, backend_object_id=command.backend_object_id, action=command.action,
                status="failed", timestamp_ns=time.time_ns(), binding_generation=command.binding_generation,
            )
        return self._responder(command)

    def _valid_proof(self, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof) -> bool:
        expected = self.issue_runtime_proof(challenge)
        return (
            proof.nonce == expected.nonce and proof.source == expected.source and proof.method == expected.method
            and proof.session_id == expected.session_id and hmac.compare_digest(proof.mac, expected.mac)
        )


class OnlineBackendBridge:
    """Dispatch only verified, active objects through one loopback Hook."""

    def __init__(
        self,
        *,
        run_id: str,
        bindings: list[BackendObjectBinding],
        hook_client: BackendHookClient,
        hook_url: str,
        gate: OfflineSafetyGate,
        capabilities: BackendCapabilityPreflight | None = None,
        binding_registry: BackendBindingRegistry | None = None,
        execution_gate: RuntimeExecutionGate | None = None,
    ) -> None:
        if not is_loopback_url(hook_url):
            raise ValueError("backend Hook URL must use a loopback host")
        self.run_id = run_id
        self.hook_client = hook_client
        self.hook_url = hook_url
        self.endpoint_identity = normalize_loopback_endpoint(hook_url)
        self.gate = gate
        self.capabilities = capabilities
        self.binding_registry = binding_registry
        self.execution_gate = execution_gate
        self._bindings_by_object: dict[tuple[Any, str], BackendObjectBinding] = {}
        self._bindings_by_request_object: dict[tuple[str, Any, str], BackendObjectBinding] = {}
        self._bindings_by_id: dict[str, BackendObjectBinding] = {}
        for item in bindings:
            if item.run_id == run_id and item.verified:
                self.register_binding(item)
        self._active_binding_ids: set[str] = set()
        self._issued_decision_ids: set[str] = set()
        self.commands: list[BackendActionCommand] = []
        self.receipts: list[BackendActionReceipt] = []

    def binding_for(
        self,
        object_level: Any,
        object_key: str,
        *,
        request_id: str | None = None,
        binding_id: str | None = None,
    ) -> BackendObjectBinding | None:
        if binding_id:
            binding = self._bindings_by_id.get(binding_id)
            if binding is not None and binding.object_level == object_level and binding.object_key == object_key:
                if request_id in (None, "", binding.request_id):
                    return binding
        if request_id:
            binding = self._bindings_by_request_object.get((str(request_id), object_level, object_key))
            if binding is not None:
                return binding
        return self._bindings_by_object.get((object_level, object_key))

    def register_binding(self, binding: BackendObjectBinding) -> bool:
        """Accept only a current verified binding from the runtime owner."""
        if binding.run_id != self.run_id or not binding.verified:
            return False
        self._bindings_by_object[(binding.object_level, binding.object_key)] = binding
        self._bindings_by_request_object[(binding.request_id, binding.object_level, binding.object_key)] = binding
        self._bindings_by_id[binding.binding_id] = binding
        return True

    def binding_snapshot(self, binding_id: str) -> dict[str, Any] | None:
        if self.binding_registry is None:
            return None
        try:
            return self.binding_registry.snapshot(binding_id)
        except ValueError:
            return None

    def binding_is_active(self, binding_id: str) -> bool:
        return binding_id in self._active_binding_ids

    def observe_event(self, event: BackendHookEvent) -> bool:
        event_binding_id = str(event.metadata.get("binding_id") or "")
        binding = self.binding_for(
            event.object_level,
            event.object_key,
            request_id=event.request_id,
            binding_id=event_binding_id,
        )
        if binding is None or not binding.matches_event(event):
            return False
        if self.binding_registry is not None:
            current = self.binding_registry.current_binding(
                binding_id=binding.binding_id,
                binding_generation=binding.binding_generation,
                request_id=event.request_id,
                object_key=event.object_key,
                object_level=event.object_level,
            )
            if current is None or current.backend_object_id != event.backend_object_id:
                return False
        if event.action == HookAction.RELEASE:
            self._active_binding_ids.discard(binding.binding_id)
        elif event.action in ACTIVE_ACTIONS and event.status in {"completed", "ok", "executed"}:
            self._active_binding_ids.add(binding.binding_id)
        return True

    def dispatch(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        if decision.run_id != self.run_id:
            return RuntimeActionResult("run_mismatch", "decision run_id does not match bridge run_id")
        action = EXECUTION_ACTIONS.get(decision.predicted_action)
        binding = self.binding_for(
            decision.object_level,
            decision.object_key,
            request_id=decision.request_id,
            binding_id=str(decision.metadata.get("binding_id") or ""),
        )

        def rejected(result_status: str, message: str, rejection_reason: str) -> RuntimeActionResult:
            return RuntimeActionResult(
                result_status,
                message,
                receipt=self._pre_execution_rejection_receipt(
                    decision,
                    action=action,
                    binding=binding,
                    rejection_reason=rejection_reason,
                    result_status=result_status,
                ),
            )

        if not self.gate.result.allowed:
            return rejected("blocked_by_offline_gate", "offline safety gate rejected runtime action", "offline_gate_rejected")
        if self.capabilities is None:
            return rejected(
                "blocked_by_capability_preflight",
                "runtime capability preflight rejected execution",
                "capability_preflight_missing",
            )
        if not self.capabilities.validate_for_execution(self.run_id, self.hook_url):
            return rejected(
                "blocked_by_capability_preflight",
                "runtime capability preflight rejected execution",
                "capability_preflight_rejected",
            )
        transport_endpoint = _trusted_client_endpoint_identity(self.hook_client)
        if transport_endpoint != self.endpoint_identity:
            return rejected(
                "blocked_by_transport_endpoint",
                "backend Hook client endpoint does not match bridge endpoint",
                "transport_endpoint_mismatch",
            )
        nonce = secrets.token_urlsafe(24)
        try:
            challenge = RuntimeProbeChallenge(
                nonce=nonce,
                run_id=self.run_id,
                endpoint_identity=self.endpoint_identity or "",
                vllm_version=self.capabilities.vllm_version,
                lmcache_version=self.capabilities.lmcache_version,
                connector_name=self.capabilities.connector_name,
                connector_version=self.capabilities.connector_version,
                allowed_actions=self.capabilities.allowed_actions,
                object_levels=self.capabilities.object_levels,
                binding_generation_observed=self.capabilities.binding_generation_observed,
            )
            proof = self.hook_client.issue_runtime_proof(challenge)
        except Exception as exc:
            return rejected(
                "blocked_by_capability_preflight",
                f"runtime authority failed to issue proof: {exc}",
                f"runtime_authority_failed:{type(exc).__name__}",
            )
        if action is None:
            return RuntimeActionResult("unsupported", f"online backend bridge does not support {decision.predicted_action}")
        if _execution_action_name(action) not in self.capabilities.allowed_actions or decision.object_level not in self.capabilities.object_levels:
            return rejected(
                "blocked_by_capability_preflight",
                "runtime capability preflight does not permit this action/object level",
                "action_or_object_level_not_permitted",
            )
        if binding is None or binding.request_id != decision.request_id:
            return rejected("unbound_object", "decision has no verified request/object binding", "verified_binding_missing")
        if self.binding_registry is None:
            return rejected(
                "unbound_object",
                "runtime dispatch requires a request-aware binding registry",
                "binding_registry_missing",
            )
        current = self.binding_registry.current_binding(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
        )
        if current is None or current.backend_object_id != binding.backend_object_id:
            binding_status = self.binding_registry.binding_status(binding.binding_id)
            rejection_reason = (
                "binding_registry_replaced_binding"
                if binding_status == "replaced"
                else "binding_registry_rejected_request_object_association"
            )
            return rejected(
                "unbound_object",
                "binding registry rejected this request/object association",
                rejection_reason,
            )
        if decision.decision_id in self._issued_decision_ids:
            return rejected("duplicate_command", "decision was already dispatched", "duplicate_command")
        admission_metadata = {
            "reason": decision.reason,
            # A caller may set a shorter deadline but cannot omit one.
            "deadline_ns": time.time_ns() + 30_000_000_000,
            "binding_identity_mode": str(binding.metadata.get("binding_identity_mode") or "unique_binding_id"),
            **dict(decision.metadata),
        }
        for field in ("previous_binding_id", "binding_replacement_reason"):
            value = str(binding.metadata.get(field) or "")
            if value:
                admission_metadata[field] = value
        if binding.execution_spec is not None:
            admission_metadata["execution_spec_id"] = binding.execution_spec.spec_id
        admission_command = BackendActionCommand(
            run_id=self.run_id, command_id=f"{self.run_id}:{decision.decision_id}", decision_id=decision.decision_id,
            request_id=decision.request_id, object_key=decision.object_key, object_level=decision.object_level,
            binding_id=binding.binding_id, backend_object_id=binding.backend_object_id, action=action,
            issued_at_ns=time.time_ns(), target_tier=decision.target_tier, metadata=admission_metadata,
            binding_generation=binding.binding_generation,
        )
        if self.execution_gate is not None:
            admission = self.execution_gate.authorize(admission_command, now_ns=time.time_ns())
            if not admission.allowed:
                return rejected(
                    f"blocked_by_runtime_gate:{admission.reason}",
                    "runtime execution gate rejected command",
                    f"runtime_execution_gate:{admission.reason}",
                )
        reservation_lease = self.binding_registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
        )
        if reservation_lease is None:
            if self.execution_gate is not None:
                self.execution_gate.complete(admission_command.command_id)
            return rejected(
                "unsafe_binding_state",
                "binding registry rejected this command association",
                "binding_registry_rejected_action_reservation",
            )

        command_metadata = {
            "reservation_lease": reservation_lease,
            **admission_metadata,
        }
        command = BackendActionCommand(
            run_id=self.run_id,
            command_id=f"{self.run_id}:{decision.decision_id}",
            decision_id=decision.decision_id,
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=action,
            issued_at_ns=time.time_ns(),
            target_tier=decision.target_tier,
            metadata=command_metadata,
            binding_generation=binding.binding_generation,
        )
        runtime_owned_action_service = isinstance(self.hook_client, InMemoryRuntimeActionClient)
        if runtime_owned_action_service:
            command = BackendActionCommand(
                run_id=command.run_id, command_id=command.command_id, decision_id=command.decision_id,
                request_id=command.request_id, object_key=command.object_key, object_level=command.object_level,
                binding_id=command.binding_id, backend_object_id=command.backend_object_id, action=command.action,
                issued_at_ns=command.issued_at_ns, target_tier=command.target_tier,
                metadata={**command.metadata, "command_sha256": command_integrity_digest(command)},
                binding_generation=command.binding_generation,
            )
        elif not self.binding_registry.consume_action_reservation(
            reservation_lease,
            command_id=command.command_id,
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
        ):
            if self.execution_gate is not None:
                self.execution_gate.complete(command.command_id)
            return rejected(
                "unsafe_binding_state",
                "binding registry could not bind reservation to command",
                "binding_registry_could_not_bind_reservation",
            )
        self._issued_decision_ids.add(decision.decision_id)
        self.commands.append(command)
        try:
            receipt = self.hook_client.submit(command, challenge, proof)
        except Exception as exc:
            if not runtime_owned_action_service:
                self.binding_registry.complete_action(reservation_lease, command_id=command.command_id, status="failed")
            if self.execution_gate is not None:
                self.execution_gate.complete(command.command_id)
                if self.execution_gate.breaker is not None:
                    self.execution_gate.breaker.record_failure(now_ns=time.time_ns())
            return RuntimeActionResult("hook_transport_failed", f"backend Hook request failed: {exc}")
        self.receipts.append(receipt)
        if not receipt.matches_command(command):
            if not runtime_owned_action_service:
                self.binding_registry.complete_action(reservation_lease, command_id=command.command_id, status="failed")
            if self.execution_gate is not None:
                self.execution_gate.complete(command.command_id)
                if self.execution_gate.breaker is not None:
                    self.execution_gate.breaker.record_failure(now_ns=time.time_ns())
            return RuntimeActionResult("receipt_mismatch", "backend Hook receipt does not match the issued command")

        if not runtime_owned_action_service:
            self.binding_registry.complete_action(reservation_lease, command_id=command.command_id, status=receipt.status)
        if self.execution_gate is not None:
            self.execution_gate.complete(command.command_id)
            # not_found is a benign race (the object was already evicted by the
            # backend before the command ran); only genuine execution failures
            # should feed the circuit breaker.
            if (
                receipt.status not in {"completed", "ok", "executed", "not_found"}
                and self.execution_gate.breaker is not None
            ):
                self.execution_gate.breaker.record_failure(now_ns=time.time_ns())

        event = RuntimeEvictionEvent(
            run_id=self.run_id,
            runtime_event_id=f"hook-{receipt.receipt_id}",
            request_id=decision.request_id,
            object_key=decision.object_key,
            object_level=decision.object_level,
            actual_action=action.value,
            tier_before=receipt.tier_before,
            tier_after=receipt.tier_after,
            bytes=receipt.bytes,
            timestamp_ns=receipt.timestamp_ns,
            arrival_index=decision.decision_index,
            status="completed" if receipt.status in {"completed", "ok", "executed"} else "failed",
            provenance="runtime_structured",
            metadata={
                "hook_url": self.hook_url,
                "binding_id": binding.binding_id,
                "binding_generation": binding.binding_generation,
                "backend_object_id": binding.backend_object_id,
                "command_id": command.command_id,
                "receipt_id": receipt.receipt_id,
                "receipt_status": receipt.status,
                **dict(receipt.metadata),
            },
        )
        if event.status == "completed":
            return RuntimeActionResult("executed", "backend Hook acknowledged action", event)
        return RuntimeActionResult("backend_failed", "backend Hook reported action failure", event)

    def _pre_execution_rejection_receipt(
        self,
        decision: OfflineEvictionDecision,
        *,
        action: HookAction | None,
        binding: BackendObjectBinding | None,
        rejection_reason: str,
        result_status: str,
    ) -> BackendActionReceipt | None:
        if action is None or binding is None:
            return None
        return BackendActionReceipt(
            run_id=self.run_id,
            command_id=f"{self.run_id}:{decision.decision_id}",
            receipt_id=f"{self.run_id}:{decision.decision_id}:rejected",
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=action,
            status="rejected",
            timestamp_ns=time.time_ns(),
            binding_generation=binding.binding_generation,
            decision_id=decision.decision_id,
            request_id=decision.request_id,
            rejection_reason=rejection_reason,
            metadata={
                "pre_execution_rejected": True,
                "result_status": result_status,
                "target_tier": decision.target_tier,
                "object_key": decision.object_key,
                "object_level": decision.object_level.value,
            },
        )


def is_loopback_url(value: str) -> bool:
    return normalize_loopback_endpoint(value) is not None


def _trusted_client_endpoint_identity(client: BackendHookClient) -> str | None:
    if not isinstance(client, (JsonHttpHookClient, InMemoryLoopbackHookClient, InMemoryRuntimeActionClient)):
        return None
    return client.endpoint_identity


def _literal_ip_loopback_endpoint(value: str) -> str | None:
    endpoint = normalize_loopback_endpoint(value)
    if endpoint is None:
        return None
    return endpoint if urlparse(value).hostname in {"127.0.0.1", "::1"} else None


def _same_literal_ip_loopback_peer(left: str, right: str) -> bool:
    left_endpoint = _literal_ip_loopback_endpoint(left)
    right_endpoint = _literal_ip_loopback_endpoint(right)
    if left_endpoint is None or right_endpoint is None:
        return False
    left_parsed = urlparse(left_endpoint)
    right_parsed = urlparse(right_endpoint)
    return left_parsed.hostname == right_parsed.hostname and left_parsed.port == right_parsed.port


def _challenge_payload(challenge: RuntimeProbeChallenge, source: str, method: str, session_id: str) -> bytes:
    return json.dumps(
        {"schema": "astrakv-runtime-probe-proof-v1", "source": source, "method": method,
         "session_id": session_id, **challenge.to_record()},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _proof_record(proof: RuntimeProbeProof) -> dict[str, str]:
    return {
        "nonce": proof.nonce, "source": proof.source, "method": proof.method,
        "session_id": proof.session_id, "mac": proof.mac,
    }


def _command_matches_challenge(command: BackendActionCommand, challenge: RuntimeProbeChallenge) -> bool:
    return (
        command.run_id == challenge.run_id
        and _execution_action_name(command.action) in challenge.allowed_actions
        and command.object_level in challenge.object_levels
    )


def _execution_action_name(action: HookAction) -> str:
    return "load" if action is HookAction.CACHE_LOAD else action.value
