"""Runtime-owned, fail-closed action service for LMCache 0.4.7.

The bridge can request a destructive action, but this service owns the
deployment secret, validates the runtime proof, and records every terminal
outcome.  The endpoint remains observational unless its lifecycle contract has
explicitly enabled action registration.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from astrakv.runtime.backend_capabilities import RuntimeProbeChallenge, RuntimeProbeProof
from astrakv.runtime.backend_hook import BackendActionCommand, BackendActionReceipt, BackendObjectBinding, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.lmcache047_runtime_patch import LMCache047ActionEndpoint


_LEDGER_SCHEMA = "astrakv-runtime-action-ledger-v1"
_PROOF_SCHEMA = "astrakv-runtime-action-proof-v1"


def command_integrity_digest(command: BackendActionCommand) -> str:
    """Canonical digest excluding its self-referential integrity field."""
    record = command.to_record()
    metadata = dict(record["metadata"])
    metadata.pop("command_sha256", None)
    record["metadata"] = metadata
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proof_payload(challenge: RuntimeProbeChallenge, source: str, method: str, session_id: str) -> bytes:
    return json.dumps(
        {"schema": _PROOF_SCHEMA, "source": source, "method": method, "session_id": session_id,
         **challenge.to_record()},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


class ProtectedRuntimeActionService:
    """Version-scoped action authority with durable command idempotency."""

    def __init__(
        self,
        *,
        action_endpoint: LMCache047ActionEndpoint,
        state_dir: str | Path,
        secret: bytes,
        source: str,
        method: str,
        session_id: str,
        now_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("runtime action service secret must be at least 32 bytes")
        if not source or not method or not session_id:
            raise ValueError("runtime action service identity is required")
        self.action_endpoint = action_endpoint
        self.source = source
        self.method = method
        self.session_id = session_id
        self._secret = bytes(secret)
        self._now_ns = now_ns
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._commands_path = self.state_dir / "commands.jsonl"
        self._receipts_path = self.state_dir / "receipts.jsonl"
        self._lock = threading.RLock()
        self._command_digests: dict[str, str] = {}
        self._commands: dict[str, BackendActionCommand] = {}
        self._receipts: dict[str, BackendActionReceipt] = {}
        self._load_ledger()

    def new_challenge_for(self, command: BackendActionCommand) -> RuntimeProbeChallenge:
        """Convenience constructor for trusted local clients and integration tests."""
        return RuntimeProbeChallenge(
            nonce=hashlib.sha256(f"{command.command_id}:{command_integrity_digest(command)}".encode()).hexdigest(),
            run_id=command.run_id,
            endpoint_identity=f"unix://{self.session_id}",
            vllm_version="0.23.0",
            lmcache_version="0.4.7",
            connector_name="lmcache-vllm-v1",
            connector_version="0.4.7",
            allowed_actions=("drop", "offload", "load", "prefetch", "evict"),
            object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True,
        )

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        return RuntimeProbeProof(
            nonce=challenge.nonce, source=self.source, method=self.method, session_id=self.session_id,
            mac=hmac.new(self._secret, _proof_payload(challenge, self.source, self.method, self.session_id), hashlib.sha256).hexdigest(),
        )

    def submit(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof,
    ) -> BackendActionReceipt:
        digest = command_integrity_digest(command)
        with self._lock:
            seen_digest = self._command_digests.get(command.command_id)
            if seen_digest is not None:
                if seen_digest == digest and command.command_id in self._receipts:
                    return self._receipts[command.command_id]
                return self._terminal(command, "command_id_conflict", {"expected_command_sha256": seen_digest})

            self._command_digests[command.command_id] = digest
            self._commands[command.command_id] = command
            self._append(self._commands_path, {
                "schema": _LEDGER_SCHEMA, "record_type": "command", "command_sha256": digest,
                "command": command.to_record(),
            })

            status, details = self._validate(command, challenge, proof, digest)
            if status is not None:
                return self._remember_terminal(self._terminal(command, status, details))
            response = self._invoke_endpoint(command)
            raw_bytes = response.get("bytes")
            try:
                bytes_value = None if raw_bytes in (None, "") else int(raw_bytes)
            except (TypeError, ValueError):
                bytes_value = None
            metadata = {
                "endpoint": "lmcache047",
                "command_sha256": digest,
                "reservation_lease": command.metadata.get("reservation_lease"),
            }
            for key in (
                "blocked_reason", "failure_reason", "error", "error_type",
                "source_location", "target_location", "load_target_id",
                "load_target_state", "load_target_created_at_ns",
                "load_target_consumed_at_ns", "disk_present_before",
                "cpu_present_before", "cpu_present_after",
                "missing_memory_obj_count", "memory_obj_count",
                "expected_tokens", "loaded_tokens",
                "cpu_used_bytes", "cpu_capacity_bytes",
                "cpu_prefetch_budget_bytes", "memory_pressure",
            ):
                if key in response and response.get(key) is not None:
                    metadata[key] = response[key]
            if "removed" in response:
                metadata["removed"] = int(response.get("removed") or 0)
            if "offloaded" in response:
                metadata["offloaded"] = int(response.get("offloaded") or 0)
            if "loaded" in response:
                metadata["loaded"] = int(response.get("loaded") or 0)
            if "prefetched" in response:
                metadata["prefetched"] = int(response.get("prefetched") or 0)
            if "evicted" in response:
                metadata["evicted"] = int(response.get("evicted") or 0)
            if response.get("load_target_id"):
                metadata["load_target_id"] = str(response.get("load_target_id"))
            if response.get("runtime_reqmeta_id"):
                metadata["runtime_reqmeta_id"] = str(response.get("runtime_reqmeta_id"))
            if "native_request_load" in response:
                metadata["native_request_load"] = bool(response.get("native_request_load"))
            if "locations" in response:
                metadata["locations"] = list(response.get("locations") or [])
            receipt = self._terminal(
                command,
                str(response.get("status") or "failed"),
                metadata,
                tier_before=str(response.get("tier_before") or "unknown"),
                tier_after=str(response.get("tier_after") or "unknown"),
                bytes=bytes_value,
            )
            return self._remember_terminal(receipt)

    def _validate(
        self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof, digest: str,
    ) -> tuple[str | None, dict[str, Any]]:
        if command.metadata.get("command_sha256") != digest:
            return "invalid_command_integrity", {"command_sha256": digest}
        deadline = command.metadata.get("deadline_ns")
        try:
            deadline_ns = int(deadline)
        except (TypeError, ValueError):
            return "deadline_required", {}
        if deadline_ns <= self._now_ns():
            return "expired", {"deadline_ns": deadline_ns}
        if not _command_matches_challenge(command, challenge) or not self._valid_proof(challenge, proof):
            return "invalid_proof", {}
        if command.action not in {HookAction.DROP, HookAction.OFFLOAD, HookAction.CACHE_LOAD, HookAction.PREFETCH, HookAction.EVICT}:
            return "unsupported_action", {}
        if not command.metadata.get("reservation_lease"):
            return "reservation_required", {}
        return None, {}

    def _valid_proof(self, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof) -> bool:
        expected = self.issue_runtime_proof(challenge)
        return (
            proof.nonce == expected.nonce and proof.source == expected.source and proof.method == expected.method
            and proof.session_id == expected.session_id and hmac.compare_digest(proof.mac, expected.mac)
        )

    def _invoke_endpoint(self, command: BackendActionCommand) -> dict[str, Any]:
        try:
            return self.action_endpoint.execute_action(command)
        except Exception as exc:
            return {"status": "failed", "error": type(exc).__name__}

    def _terminal(
        self,
        command: BackendActionCommand,
        status: str,
        metadata: dict[str, Any],
        *,
        tier_before: str = "unknown",
        tier_after: str = "unknown",
        bytes: int | None = None,
    ) -> BackendActionReceipt:
        receipt = BackendActionReceipt(
            run_id=command.run_id, command_id=command.command_id,
            receipt_id=f"{command.command_id}:terminal", binding_id=command.binding_id,
            backend_object_id=command.backend_object_id, action=command.action, status=status,
            timestamp_ns=self._now_ns(), binding_generation=command.binding_generation,
            tier_before=tier_before, tier_after=tier_after, bytes=bytes, metadata=dict(metadata),
            decision_id=command.decision_id, request_id=command.request_id,
        )
        signature = hmac.new(self._secret, json.dumps(receipt.to_record(), sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
        return replace(receipt, metadata={**receipt.metadata, "receipt_signature": signature})

    def _remember_terminal(self, receipt: BackendActionReceipt) -> BackendActionReceipt:
        self._receipts[receipt.command_id] = receipt
        self._append(self._receipts_path, {
            "schema": _LEDGER_SCHEMA, "record_type": "terminal_receipt", "receipt": receipt.to_record(),
        })
        return receipt

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def _load_ledger(self) -> None:
        if self._commands_path.exists():
            for line in self._commands_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                command = BackendActionCommand.from_record(record["command"])
                digest = str(record["command_sha256"])
                if command_integrity_digest(command) != digest:
                    raise ValueError("action command ledger integrity check failed")
                self._command_digests[command.command_id] = digest
                self._commands[command.command_id] = command
        if self._receipts_path.exists():
            for line in self._receipts_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                receipt = BackendActionReceipt.from_record(json.loads(line)["receipt"])
                self._receipts[receipt.command_id] = receipt
        # A crash between the two fsync-backed appends is never replayed as an
        # action.  Close it with an auditable terminal outcome on restart.
        for command_id, command in tuple(self._commands.items()):
            if command_id not in self._receipts:
                self._remember_terminal(self._terminal(
                    command, "interrupted_before_terminal_receipt", {"recovered_on_startup": True},
                ))


class InMemoryRuntimeActionClient:
    """Injectable protected transport for unit tests and local integration tests."""

    def __init__(self, service: ProtectedRuntimeActionService, endpoint_identity: str = "http://127.0.0.1:7900/actions") -> None:
        self.service = service
        self.endpoint_identity = endpoint_identity

    def submit(self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof) -> BackendActionReceipt:
        return self.service.submit(command, challenge, proof)

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        return self.service.issue_runtime_proof(challenge)


class UnixDomainSocketActionServer:
    """Owner-only local Unix socket server for the action service."""

    def __init__(
        self, service: ProtectedRuntimeActionService, socket_path: str | Path,
        *, admit_drop: Callable[[BackendObjectBinding], BackendActionReceipt] | None = None,
    ) -> None:
        self.service = service
        self.socket_path = Path(socket_path)
        self._admit_drop = admit_drop
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if os.name != "posix":
            raise RuntimeError("Unix-domain socket action service requires POSIX")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise FileExistsError(f"refusing to replace existing socket: {self.socket_path}")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(os.fspath(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server.listen(8)
        server.settimeout(0.2)
        self._socket = server
        self._thread = threading.Thread(target=self._serve, name="astrakv-runtime-action", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.socket_path.exists():
            self.socket_path.unlink()

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except (TimeoutError, socket.timeout, OSError):
                continue
            with connection:
                try:
                    raw = _recv_line(connection)
                    payload = json.loads(raw)
                    if payload.get("kind") == "runtime-proof":
                        proof = self.service.issue_runtime_proof(_challenge_from_record(payload["challenge"]))
                        connection.sendall((json.dumps({
                            "nonce": proof.nonce, "source": proof.source, "method": proof.method,
                            "session_id": proof.session_id, "mac": proof.mac,
                        }, sort_keys=True) + "\n").encode("utf-8"))
                        continue
                    if payload.get("kind") == "admit-drop":
                        if self._admit_drop is None:
                            raise ValueError("admission is not configured")
                        receipt = self._admit_drop(BackendObjectBinding.from_record(payload["binding"]))
                        connection.sendall((json.dumps(receipt.to_record(), sort_keys=True) + "\n").encode("utf-8"))
                        continue
                    receipt = self.service.submit(
                        BackendActionCommand.from_record(payload["command"]), _challenge_from_record(payload["challenge"]),
                        _proof_from_record(payload["proof"]),
                    )
                    connection.sendall((json.dumps(receipt.to_record(), sort_keys=True) + "\n").encode("utf-8"))
                except Exception:
                    connection.sendall(b'{"error":"invalid_request"}\n')


class UnixDomainSocketActionClient:
    """Small request/response client matching the backend Hook submit protocol."""

    def __init__(self, socket_path: str | Path, *, timeout_s: float = 5.0) -> None:
        self.socket_path = os.fspath(socket_path)
        self.timeout_s = timeout_s
        self.endpoint_identity = f"unix://{self.socket_path}"

    def submit(self, command: BackendActionCommand, challenge: RuntimeProbeChallenge, proof: RuntimeProbeProof) -> BackendActionReceipt:
        request = {"command": command.to_record(), "challenge": challenge.to_record(), "proof": proof.__dict__ if hasattr(proof, "__dict__") else {
            "nonce": proof.nonce, "source": proof.source, "method": proof.method, "session_id": proof.session_id, "mac": proof.mac,
        }}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_s)
            client.connect(self.socket_path)
            client.sendall((json.dumps(request, sort_keys=True) + "\n").encode("utf-8"))
            response = json.loads(_recv_line(client))
        if "error" in response:
            raise ValueError(response["error"])
        return BackendActionReceipt.from_record(response)

    def issue_runtime_proof(self, challenge: RuntimeProbeChallenge) -> RuntimeProbeProof:
        response = self._request({"kind": "runtime-proof", "challenge": challenge.to_record()})
        if "error" in response:
            raise ValueError(response["error"])
        return _proof_from_record(response)

    def admit_drop(self, binding: BackendObjectBinding) -> BackendActionReceipt:
        response = self._request({"kind": "admit-drop", "binding": binding.to_record()})
        if "error" in response:
            raise ValueError(response["error"])
        return BackendActionReceipt.from_record(response)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_s)
            client.connect(self.socket_path)
            client.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            return json.loads(_recv_line(client))


def _recv_line(connection: socket.socket) -> str:
    chunks = bytearray()
    while len(chunks) <= 1_000_000:
        data = connection.recv(4096)
        if not data:
            break
        chunks.extend(data)
        if b"\n" in data:
            return bytes(chunks).split(b"\n", 1)[0].decode("utf-8")
    raise ValueError("invalid socket request")


def _challenge_from_record(record: dict[str, Any]) -> RuntimeProbeChallenge:
    return RuntimeProbeChallenge(
        nonce=str(record["nonce"]), run_id=str(record["run_id"]), endpoint_identity=str(record["endpoint_identity"]),
        vllm_version=record.get("vllm_version"), lmcache_version=record.get("lmcache_version"),
        connector_name=record.get("connector_name"), connector_version=record.get("connector_version"),
        allowed_actions=tuple(str(item) for item in record.get("allowed_actions") or ()),
        object_levels=tuple(ObjectLevel(item) for item in record.get("object_levels") or ()),
        binding_generation_observed=bool(record.get("binding_generation_observed")),
    )


def _proof_from_record(record: dict[str, Any]) -> RuntimeProbeProof:
    return RuntimeProbeProof(
        nonce=str(record["nonce"]), source=str(record["source"]), method=str(record["method"]),
        session_id=str(record["session_id"]), mac=str(record["mac"]),
    )


def _command_matches_challenge(command: BackendActionCommand, challenge: RuntimeProbeChallenge) -> bool:
    action_name = "load" if command.action is HookAction.CACHE_LOAD else command.action.value
    return (
        command.run_id == challenge.run_id and action_name in challenge.allowed_actions
        and command.object_level in challenge.object_levels
    )
