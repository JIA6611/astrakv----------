"""Versioned request-context handoff for a runtime Hook.

The endpoint benchmark publishes this artifact to a loopback-only consumer. A
runtime Hook may consume it to associate its own request identity, but the
benchmark never assumes that association happened without a matching receipt.
"""

from __future__ import annotations

import json
import ipaddress
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from astrakv.runtime.backend_capabilities import normalize_loopback_endpoint


REQUEST_CONTEXT_SCHEMA = "astrakv-request-context-v1"
REQUEST_CONTEXT_SESSION_HEADER = "X-AstraKV-Context-Session"
REQUEST_CONTEXT_EXPIRY_HEADER = "X-AstraKV-Context-Expires-At-Ns"
REQUEST_CONTEXT_MAC_HEADER = "X-AstraKV-Context-Mac"


def _numeric_loopback_endpoint(url: str) -> str | None:
    parsed = urlsplit(url)
    host = parsed.hostname
    if not host:
        return None
    try:
        numeric_host = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not numeric_host.is_loopback:
        return None
    return normalize_loopback_endpoint(url)


def _required(value: Any, field_name: str) -> str:
    text = str(value or "")
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


@dataclass(frozen=True, slots=True)
class RuntimeRequestContext:
    run_id: str
    request_id: str
    case: str
    request_nonce: str
    request_started_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = REQUEST_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        _required(self.run_id, "run_id")
        _required(self.request_id, "request_id")
        _required(self.case, "case")
        _required(self.request_nonce, "request_nonce")
        if self.schema != REQUEST_CONTEXT_SCHEMA:
            raise ValueError("unsupported request context schema")
        object.__setattr__(self, "metadata", _context_metadata(self.metadata))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RuntimeRequestContext":
        return cls(
            run_id=_required(record.get("run_id"), "run_id"),
            request_id=_required(record.get("request_id"), "request_id"),
            case=_required(record.get("case"), "case"),
            request_nonce=_required(record.get("request_nonce"), "request_nonce"),
            request_started_s=float(record.get("request_started_s")),
            metadata=_context_metadata(record.get("metadata")),
            schema=str(record.get("schema") or ""),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": "request_context",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "case": self.case,
            "request_nonce": self.request_nonce,
            "request_started_s": self.request_started_s,
            "metadata": dict(self.metadata),
        }


def _context_metadata(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("request context metadata must be a mapping")
    return {str(name): item for name, item in value.items()}


@dataclass(frozen=True, slots=True)
class RuntimeRequestIdentity:
    """Runtime-side identity that must match a recorded endpoint context."""

    run_id: str
    request_id: str
    request_nonce: str

    def __post_init__(self) -> None:
        _required(self.run_id, "run_id")
        _required(self.request_id, "request_id")
        _required(self.request_nonce, "request_nonce")


def _context_mac_payload(
    record: Mapping[str, Any], *, endpoint_identity: str, session_id: str, expires_at_ns: int,
) -> bytes:
    payload = {
        "schema": REQUEST_CONTEXT_SCHEMA,
        "endpoint_identity": endpoint_identity,
        "session_id": session_id,
        "expires_at_ns": expires_at_ns,
        "record": dict(record),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RuntimeRequestContextAuthority:
    """Runtime-owned, session-scoped MAC authority for loopback context handoff."""

    _key: bytes
    run_id: str
    session_id: str
    ttl_ns: int

    @classmethod
    def install(
        cls, *, run_id: str, session_id: str, secret: bytes | None = None, ttl_s: float = 30.0,
    ) -> "RuntimeRequestContextAuthority":
        _required(run_id, "run_id")
        _required(session_id, "session_id")
        ttl_ns = int(ttl_s * 1_000_000_000)
        if ttl_ns <= 0:
            raise ValueError("ttl_s must be positive")
        key = secrets.token_bytes(32) if secret is None else bytes(secret)
        if len(key) < 32:
            raise ValueError("request context authority secret must have at least 32 bytes")
        return cls(key, run_id, session_id, ttl_ns)

    def context_headers(
        self, context: RuntimeRequestContext, endpoint_identity: str, *, now_ns: int | None = None,
    ) -> dict[str, str]:
        if context.run_id != self.run_id:
            raise ValueError("request context run_id does not match authority")
        endpoint = _numeric_loopback_endpoint(endpoint_identity)
        if endpoint is None:
            raise ValueError("request context endpoint must use a numeric loopback host")
        issued = time.time_ns() if now_ns is None else int(now_ns)
        expires_at_ns = issued + self.ttl_ns
        mac = hmac.new(
            self._key,
            _context_mac_payload(context.to_record(), endpoint_identity=endpoint, session_id=self.session_id, expires_at_ns=expires_at_ns),
            hashlib.sha256,
        ).hexdigest()
        return {
            REQUEST_CONTEXT_SESSION_HEADER: self.session_id,
            REQUEST_CONTEXT_EXPIRY_HEADER: str(expires_at_ns),
            REQUEST_CONTEXT_MAC_HEADER: mac,
        }

    def verify_context(
        self, record: Mapping[str, Any], headers: Mapping[str, str], endpoint_identity: str, *, now_ns: int | None = None,
    ) -> int:
        endpoint = _numeric_loopback_endpoint(endpoint_identity)
        if endpoint is None:
            raise ValueError("request context endpoint must use a numeric loopback host")
        normalized_headers = {str(name).lower(): str(value) for name, value in headers.items()}
        session_id = normalized_headers.get(REQUEST_CONTEXT_SESSION_HEADER.lower(), "")
        mac = normalized_headers.get(REQUEST_CONTEXT_MAC_HEADER.lower(), "")
        try:
            expires_at_ns = int(normalized_headers.get(REQUEST_CONTEXT_EXPIRY_HEADER.lower(), ""))
        except ValueError as exc:
            raise ValueError("request context authentication expiry is invalid") from exc
        current = time.time_ns() if now_ns is None else int(now_ns)
        if session_id != self.session_id or not mac:
            raise ValueError("request context authentication failed")
        if expires_at_ns <= current or expires_at_ns > current + self.ttl_ns:
            raise ValueError("request context authentication expired")
        expected = hmac.new(
            self._key,
            _context_mac_payload(record, endpoint_identity=endpoint, session_id=session_id, expires_at_ns=expires_at_ns),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(mac, expected):
            raise ValueError("request context authentication failed")
        return expires_at_ns

    def _receipt_mac(self, receipt: "RequestContextReceipt", endpoint_identity: str) -> str:
        endpoint = _numeric_loopback_endpoint(endpoint_identity)
        if endpoint is None:
            raise ValueError("request context endpoint must use a numeric loopback host")
        payload = _context_mac_payload(
            receipt.to_record(include_mac=False), endpoint_identity=endpoint,
            session_id=self.session_id, expires_at_ns=receipt.expires_at_ns,
        )
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def sign_receipt(self, receipt: "RequestContextReceipt", endpoint_identity: str) -> "RequestContextReceipt":
        if receipt.run_id != self.run_id:
            raise ValueError("receipt run_id does not match authority")
        unsigned = RequestContextReceipt(
            run_id=receipt.run_id, request_id=receipt.request_id, request_nonce=receipt.request_nonce,
            runtime_request_id=receipt.runtime_request_id, runtime_event_id=receipt.runtime_event_id,
            status=receipt.status, session_id=self.session_id, expires_at_ns=receipt.expires_at_ns,
        )
        return RequestContextReceipt(
            run_id=unsigned.run_id, request_id=unsigned.request_id, request_nonce=unsigned.request_nonce,
            runtime_request_id=unsigned.runtime_request_id, runtime_event_id=unsigned.runtime_event_id,
            status=unsigned.status, session_id=unsigned.session_id, expires_at_ns=unsigned.expires_at_ns,
            mac=self._receipt_mac(unsigned, endpoint_identity),
        )

    def verify_receipt(self, receipt: "RequestContextReceipt", endpoint_identity: str) -> bool:
        if receipt.run_id != self.run_id or receipt.session_id != self.session_id or not receipt.mac:
            return False
        unsigned = RequestContextReceipt(
            run_id=receipt.run_id, request_id=receipt.request_id, request_nonce=receipt.request_nonce,
            runtime_request_id=receipt.runtime_request_id, runtime_event_id=receipt.runtime_event_id,
            status=receipt.status, session_id=receipt.session_id, expires_at_ns=receipt.expires_at_ns,
        )
        return hmac.compare_digest(receipt.mac, self._receipt_mac(unsigned, endpoint_identity))


class RuntimeRequestContextReceiver:
    """Runtime-owned receiver: record first, associate only from ReqMeta lifecycle."""

    def __init__(self, endpoint: str, authority: RuntimeRequestContextAuthority) -> None:
        endpoint_identity = _numeric_loopback_endpoint(endpoint)
        if endpoint_identity is None:
            raise ValueError("request context receiver must use a numeric loopback endpoint")
        self.endpoint_identity = endpoint_identity
        self.authority = authority
        self._contexts: dict[str, tuple[RuntimeRequestContext, str, int]] = {}
        self._associations: dict[str, RuntimeRequestContext] = {}
        self._lock = threading.RLock()

    def receive(
        self, record: Mapping[str, Any], headers: Mapping[str, str], *, now_ns: int | None = None,
    ) -> "RequestContextReceipt":
        if record.get("record_type") != "request_context":
            raise ValueError("request context record_type is invalid")
        context = RuntimeRequestContext.from_record(dict(record))
        if context.run_id != self.authority.run_id:
            raise ValueError("request context run_id does not match receiver")
        expires_at_ns = self.authority.verify_context(record, headers, self.endpoint_identity, now_ns=now_ns)
        digest = hashlib.sha256(json.dumps(dict(record), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        with self._lock:
            prior = self._contexts.get(context.request_nonce)
            if prior is not None and prior[1] != digest:
                raise ValueError("request context nonce replay conflict")
            if prior is None:
                self._contexts[context.request_nonce] = (context, digest, expires_at_ns)
            return self.authority.sign_receipt(
                RequestContextReceipt(
                    run_id=context.run_id, request_id=context.request_id, request_nonce=context.request_nonce,
                    status="recorded", expires_at_ns=expires_at_ns,
                ), self.endpoint_identity,
            )

    def associate_runtime_request(
        self, runtime_request_id: str, identity: RuntimeRequestIdentity, *, now_ns: int | None = None,
    ) -> "RequestContextReceipt":
        runtime_request_id = _required(runtime_request_id, "runtime_request_id")
        current = time.time_ns() if now_ns is None else int(now_ns)
        with self._lock:
            entry = self._contexts.get(identity.request_nonce)
            if entry is None:
                raise ValueError("request context has not been recorded")
            context, _, expires_at_ns = entry
            if expires_at_ns <= current:
                raise ValueError("request context authentication expired")
            if (context.run_id, context.request_id, context.request_nonce) != (
                identity.run_id, identity.request_id, identity.request_nonce,
            ):
                raise ValueError("runtime request identity does not match recorded context")
            prior = self._associations.get(runtime_request_id)
            if prior is not None and prior != context:
                raise ValueError("runtime request association conflict")
            self._associations[runtime_request_id] = context
            return self.authority.sign_receipt(
                RequestContextReceipt(
                    run_id=context.run_id, request_id=context.request_id, request_nonce=context.request_nonce,
                    runtime_request_id=runtime_request_id,
                    runtime_event_id=f"runtime-context:{runtime_request_id}",
                    status="associated", expires_at_ns=expires_at_ns,
                ), self.endpoint_identity,
            )

    def associated_context(self, runtime_request_id: str) -> RuntimeRequestContext | None:
        with self._lock:
            return self._associations.get(runtime_request_id)


@dataclass(frozen=True, slots=True)
class RequestContextReceipt:
    run_id: str
    request_id: str
    request_nonce: str
    runtime_request_id: str = ""
    runtime_event_id: str = ""
    status: str = "recorded"
    session_id: str = ""
    expires_at_ns: int = 0
    mac: str = ""
    schema: str = REQUEST_CONTEXT_SCHEMA

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RequestContextReceipt":
        if record.get("record_type") != "request_context_receipt":
            raise ValueError("record_type must be request_context_receipt")
        return cls(
            run_id=_required(record.get("run_id"), "run_id"),
            request_id=_required(record.get("request_id"), "request_id"),
            request_nonce=_required(record.get("request_nonce"), "request_nonce"),
            runtime_request_id=str(record.get("runtime_request_id") or ""),
            runtime_event_id=str(record.get("runtime_event_id") or ""),
            status=str(record.get("status") or "recorded"),
            session_id=str(record.get("session_id") or ""),
            expires_at_ns=int(record.get("expires_at_ns") or 0),
            mac=str(record.get("mac") or ""),
            schema=str(record.get("schema") or ""),
        )

    def matches(self, context: RuntimeRequestContext) -> bool:
        return (
            self.schema == REQUEST_CONTEXT_SCHEMA
            and self.status == "associated"
            and bool(self.runtime_request_id)
            and bool(self.runtime_event_id)
            and self.run_id == context.run_id
            and self.request_id == context.request_id
            and self.request_nonce == context.request_nonce
        )

    def to_record(self, *, include_mac: bool = True) -> dict[str, Any]:
        record = {
            "schema": self.schema,
            "record_type": "request_context_receipt",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "request_nonce": self.request_nonce,
            "runtime_request_id": self.runtime_request_id,
            "runtime_event_id": self.runtime_event_id,
            "status": self.status,
            "session_id": self.session_id,
            "expires_at_ns": self.expires_at_ns,
        }
        if include_mac:
            record["mac"] = self.mac
        return record


class RequestContextClient(Protocol):
    @property
    def endpoint_identity(self) -> str:
        """Canonical loopback endpoint used for context handoff."""

    def publish(self, context: RuntimeRequestContext) -> RequestContextReceipt:
        """Send a request context to a runtime Hook-compatible consumer."""


class RequestContextArtifact(Protocol):
    def append(self, context: RuntimeRequestContext) -> None:
        """Persist the exact context handed to the runtime consumer."""


@dataclass(slots=True)
class RequestContextJsonlArtifact:
    path: Path
    _lock: threading.Lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, context: RuntimeRequestContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(context.to_record(), ensure_ascii=False) + "\n")


@dataclass(frozen=True, slots=True)
class JsonHttpRequestContextClient:
    context_url: str
    timeout_s: float = 2.0
    endpoint_identity: str = ""

    def __post_init__(self) -> None:
        endpoint = _numeric_loopback_endpoint(self.context_url)
        if endpoint is None:
            raise ValueError("request context URL must use a numeric loopback host")
        object.__setattr__(self, "endpoint_identity", endpoint)

    def publish(self, context: RuntimeRequestContext) -> RequestContextReceipt:
        request = Request(
            self.context_url,
            data=json.dumps(context.to_record()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        with opener.open(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request context response must be a JSON object")
        return RequestContextReceipt.from_record(payload)


@dataclass(frozen=True, slots=True)
class AuthenticatedJsonHttpRequestContextClient:
    """Loopback context publisher carrying the runtime-owned session MAC."""

    context_url: str
    run_id: str
    session_id: str
    secret: bytes
    timeout_s: float = 2.0
    endpoint_identity: str = ""
    _authority: RuntimeRequestContextAuthority | None = None

    def __post_init__(self) -> None:
        endpoint = _numeric_loopback_endpoint(self.context_url)
        if endpoint is None:
            raise ValueError("request context URL must use a numeric loopback host")
        authority = RuntimeRequestContextAuthority.install(
            run_id=self.run_id, session_id=self.session_id, secret=self.secret,
        )
        object.__setattr__(self, "endpoint_identity", endpoint)
        object.__setattr__(self, "_authority", authority)

    def publish(self, context: RuntimeRequestContext) -> RequestContextReceipt:
        authority = self._authority
        if authority is None:
            raise RuntimeError("request context authority is not initialized")
        request = Request(
            self.context_url,
            data=json.dumps(context.to_record()).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **authority.context_headers(context, self.endpoint_identity),
            },
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        with opener.open(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request context response must be a JSON object")
        return RequestContextReceipt.from_record(payload)


@dataclass(frozen=True, slots=True)
class InMemoryLoopbackRequestContextClient:
    context_url: str
    responder: Callable[[RuntimeRequestContext], RequestContextReceipt]
    endpoint_identity: str = ""

    def __post_init__(self) -> None:
        endpoint = _numeric_loopback_endpoint(self.context_url)
        if endpoint is None:
            raise ValueError("request context URL must use a numeric loopback host")
        object.__setattr__(self, "endpoint_identity", endpoint)

    def publish(self, context: RuntimeRequestContext) -> RequestContextReceipt:
        return self.responder(context)


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None

    def http_error_302(self, request: Any, fp: Any, code: int, msg: str, headers: Any) -> None:
        raise HTTPError(request.full_url, code, "request context redirects are forbidden", headers, fp)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302
