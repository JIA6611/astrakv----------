"""Version-locked capability preflight for the supported vLLM/LMCache backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from astrakv.runtime.eviction import ObjectLevel


BACKEND_CAPABILITIES_SCHEMA = "astrakv-backend-capabilities-v1"
INSTALLATION_EVIDENCE_SCHEMA = "astrakv-installation-evidence-v1"
SUPPORTED_VLLM_VERSION = "0.23.0"
SUPPORTED_LMCACHE_VERSION = "0.4.7"
SUPPORTED_CONNECTOR_NAME = "lmcache-vllm-v1"
SUPPORTED_ACTIONS = ("drop", "offload", "load", "prefetch", "evict")
SUPPORTED_OBJECT_LEVELS = (ObjectLevel.PREFIX,)
AUDIT_ACTIONS = ("drop", "offload", "load", "prefetch", "evict")
BLOCKED_ACTION_REASONS = {
    "offload": "no_public_object_level_api_lmcache047",
    "load": "no_runtime_load_target_lmcache047",
    "prefetch": "no_stable_object_level_api_lmcache047",
    "evict": "no_verified_distinct_backend_entrypoint_lmcache047",
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _normalized_versions(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalized_actions(actions: Iterable[str] | None) -> tuple[str, ...]:
    values = () if actions is None else tuple(str(item) for item in actions)
    return tuple(item for item in SUPPORTED_ACTIONS if item in values)


def _normalized_object_levels(levels: Iterable[ObjectLevel | str] | None) -> tuple[ObjectLevel, ...]:
    values: list[ObjectLevel] = []
    for item in levels or ():
        try:
            values.append(ObjectLevel(item))
        except ValueError:
            continue
    return tuple(item for item in SUPPORTED_OBJECT_LEVELS if item in values)


def normalize_loopback_endpoint(value: str | None) -> str | None:
    """Return a stable identity for an HTTP loopback Hook endpoint."""

    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme.lower() != "http" or parsed.hostname not in LOOPBACK_HOSTS:
        return None
    try:
        parsed_port = parsed.port
    except ValueError:
        return None
    if parsed_port is not None and parsed_port <= 0:
        return None
    port = parsed_port if parsed_port is not None else 80
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    host = parsed.hostname.lower()
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{port}{parsed.path or '/'}"


def _probe_digest(
    *, source: str, method: str, session_id: str, vllm_version: str | None, lmcache_version: str | None,
    connector_name: str | None, connector_version: str | None, endpoint_identity: str | None,
) -> str:
    payload = {
        "schema": INSTALLATION_EVIDENCE_SCHEMA,
        "source": source,
        "method": method,
        "session_id": session_id,
        "vllm_version": vllm_version,
        "lmcache_version": lmcache_version,
        "connector_name": connector_name,
        "connector_version": connector_version,
        "endpoint_identity": endpoint_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class InstallationEvidence:
    """Versioned digest of the concrete runtime installation probe."""

    source: str
    method: str
    session_id: str
    probe_digest: str
    schema: str = INSTALLATION_EVIDENCE_SCHEMA

    def is_valid(
        self,
        *,
        vllm_version: str | None,
        lmcache_version: str | None,
        connector_name: str | None,
        connector_version: str | None,
        endpoint_identity: str | None,
    ) -> bool:
        return (
            self.schema == INSTALLATION_EVIDENCE_SCHEMA
            and bool(self.source)
            and bool(self.method)
            and bool(self.session_id)
            and self.probe_digest == _probe_digest(
                source=self.source,
                method=self.method,
                session_id=self.session_id,
                vllm_version=vllm_version,
                lmcache_version=lmcache_version,
                connector_name=connector_name,
                connector_version=connector_version,
                endpoint_identity=endpoint_identity,
            )
        )

    def to_record(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "source": self.source,
            "method": self.method,
            "session_id": self.session_id,
            "probe_digest": self.probe_digest,
        }


@dataclass(frozen=True, slots=True)
class RuntimeProbeProof:
    """Fresh runtime-authority MAC over a concrete probe challenge."""

    nonce: str
    source: str
    method: str
    session_id: str
    mac: str


@dataclass(frozen=True, slots=True)
class RuntimeProbeChallenge:
    nonce: str
    run_id: str
    endpoint_identity: str
    vllm_version: str | None
    lmcache_version: str | None
    connector_name: str | None
    connector_version: str | None
    allowed_actions: tuple[str, ...]
    object_levels: tuple[ObjectLevel, ...]
    binding_generation_observed: bool

    def to_record(self) -> dict[str, object]:
        return {
            "nonce": self.nonce,
            "run_id": self.run_id,
            "endpoint_identity": self.endpoint_identity,
            "vllm_version": self.vllm_version,
            "lmcache_version": self.lmcache_version,
            "connector_name": self.connector_name,
            "connector_version": self.connector_version,
            "allowed_actions": list(self.allowed_actions),
            "object_levels": [item.value for item in self.object_levels],
            "binding_generation_observed": self.binding_generation_observed,
        }


def build_installation_evidence(
    *,
    source: str,
    method: str,
    session_id: str,
    vllm_version: str | None,
    lmcache_version: str | None,
    connector_name: str | None,
    connector_version: str | None,
    endpoint_identity: str | None,
) -> InstallationEvidence:
    """Build evidence from observed runtime and connector probe values."""

    source = _normalized_versions(source) or ""
    method = _normalized_versions(method) or ""
    endpoint = normalize_loopback_endpoint(endpoint_identity)
    session_id = _normalized_versions(session_id) or ""
    if not source or not method or not session_id or endpoint is None:
        raise ValueError("installation evidence requires source, method, session_id, and a loopback endpoint")
    vllm = _normalized_versions(vllm_version)
    lmcache = _normalized_versions(lmcache_version)
    connector = _normalized_versions(connector_name)
    connector_version = _normalized_versions(connector_version)
    return InstallationEvidence(
        source=source,
        method=method,
        probe_digest=_probe_digest(
            source=source,
            method=method,
            session_id=session_id,
            vllm_version=vllm,
            lmcache_version=lmcache,
            connector_name=connector,
            connector_version=connector_version,
            endpoint_identity=endpoint,
        ),
        session_id=session_id,
    )


@dataclass(frozen=True, slots=True)
class BackendCapabilityPreflight:
    """Serializable statement of exactly what a runtime installation may expose."""

    vllm_version: str | None
    lmcache_version: str | None
    run_id: str | None
    endpoint_identity: str | None
    installation_evidence: InstallationEvidence | None
    connector_name: str | None
    connector_version: str | None
    allowed_actions: tuple[str, ...]
    object_levels: tuple[ObjectLevel, ...]
    binding_generation_observed: bool
    capability_flags: dict[str, bool]
    compatible: bool
    blocked_reason: str | None

    @property
    def execution_eligible(self) -> bool:
        """Whether every independently observed prerequisite permits execution."""

        return self.validate_for_execution(self.run_id or "", self.endpoint_identity or "")

    @property
    def action_status(self) -> dict[str, dict[str, str | None]]:
        """Expose an explicit audit status for every planned action."""

        return {
            action: (
                {"status": "allowed", "reason": None}
                if action in self.allowed_actions
                else {
                    "status": "blocked",
                    "reason": BLOCKED_ACTION_REASONS.get(action, "not_declared_by_preflight"),
                }
            )
            for action in AUDIT_ACTIONS
        }

    def validate_for_execution(self, run_id: str, endpoint: str) -> bool:
        """Recompute and validate every execution prerequisite from raw evidence."""

        flags, compatible, blocked_reason = _evaluate_preflight(
            vllm_version=self.vllm_version,
            lmcache_version=self.lmcache_version,
            run_id=self.run_id,
            endpoint_identity=self.endpoint_identity,
            installation_evidence=self.installation_evidence,
            connector_name=self.connector_name,
            connector_version=self.connector_version,
            allowed_actions=self.allowed_actions,
            object_levels=self.object_levels,
            binding_generation_observed=self.binding_generation_observed,
        )
        structural_valid = (
            compatible
            and self.compatible == compatible
            and self.blocked_reason == blocked_reason
            and self.capability_flags == flags
            and _normalized_versions(run_id) == self.run_id
            and normalize_loopback_endpoint(endpoint) == self.endpoint_identity
        )
        return structural_valid

    def to_record(self) -> dict[str, object]:
        return {
            "schema": BACKEND_CAPABILITIES_SCHEMA,
            "backend": "vllm-lmcache",
            "backend_versions": {
                "vllm": self.vllm_version,
                "lmcache": self.lmcache_version,
            },
            "run_id": self.run_id,
            "endpoint_identity": self.endpoint_identity,
            "installation_evidence": None if self.installation_evidence is None else self.installation_evidence.to_record(),
            "connector": {
                "name": self.connector_name,
                "version": self.connector_version,
            },
            "allowed_actions": list(self.allowed_actions),
            "action_status": self.action_status,
            "object_levels": [item.value for item in self.object_levels],
            "capability_flags": dict(self.capability_flags),
            "compatible": self.compatible,
            "execution_eligible": self.execution_eligible,
            "blocked_reason": self.blocked_reason,
        }


def preflight_backend_capabilities(
    *,
    vllm_version: str | None,
    lmcache_version: str | None,
    run_id: str | None,
    hook_url: str | None,
    installation_evidence: InstallationEvidence | None,
    connector_name: str | None,
    connector_version: str | None,
    available_actions: Iterable[str] | None = None,
    available_object_levels: Iterable[ObjectLevel | str] | None = None,
    binding_generation_observed: bool = False,
) -> BackendCapabilityPreflight:
    """Return a deny-by-default capability record for one supported version tuple.

    Callers pass observed versions explicitly so the emitted artifact records the
    exact values used for an experiment rather than deriving a claim from imports.
    """

    vllm = _normalized_versions(vllm_version)
    lmcache = _normalized_versions(lmcache_version)
    normalized_run_id = _normalized_versions(run_id)
    endpoint_identity = normalize_loopback_endpoint(hook_url)
    connector_name = _normalized_versions(connector_name)
    connector = _normalized_versions(connector_version)
    allowed_actions = _normalized_actions(available_actions)
    object_levels = _normalized_object_levels(available_object_levels)

    flags, compatible, blocked_reason = _evaluate_preflight(
        vllm_version=vllm,
        lmcache_version=lmcache,
        run_id=normalized_run_id,
        endpoint_identity=endpoint_identity,
        installation_evidence=installation_evidence,
        connector_name=connector_name,
        connector_version=connector,
        allowed_actions=allowed_actions,
        object_levels=object_levels,
        binding_generation_observed=binding_generation_observed,
    )
    return BackendCapabilityPreflight(
        vllm_version=vllm,
        lmcache_version=lmcache,
        run_id=normalized_run_id,
        endpoint_identity=endpoint_identity,
        installation_evidence=installation_evidence,
        connector_name=connector_name,
        connector_version=connector,
        allowed_actions=allowed_actions,
        object_levels=object_levels,
        binding_generation_observed=bool(binding_generation_observed),
        capability_flags=flags,
        compatible=compatible,
        blocked_reason=blocked_reason,
    )


def _evaluate_preflight(
    *,
    vllm_version: str | None,
    lmcache_version: str | None,
    run_id: str | None,
    endpoint_identity: str | None,
    installation_evidence: InstallationEvidence | None,
    connector_name: str | None,
    connector_version: str | None,
    allowed_actions: tuple[str, ...],
    object_levels: tuple[ObjectLevel, ...],
    binding_generation_observed: bool,
) -> tuple[dict[str, bool], bool, str | None]:
    evidence_valid = installation_evidence is not None and installation_evidence.is_valid(
        vllm_version=vllm_version,
        lmcache_version=lmcache_version,
        connector_name=connector_name,
        connector_version=connector_version,
        endpoint_identity=endpoint_identity,
    )
    flags = {
        "run_identity": run_id is not None,
        "endpoint_identity": endpoint_identity is not None,
        "installation_evidence": evidence_valid,
        "vllm_version": vllm_version == SUPPORTED_VLLM_VERSION,
        "lmcache_version": lmcache_version == SUPPORTED_LMCACHE_VERSION,
        "connector_identity": connector_name == SUPPORTED_CONNECTOR_NAME,
        "connector_version": connector_version == SUPPORTED_LMCACHE_VERSION,
        "drop_action": "drop" in allowed_actions,
        "prefix_object_level": ObjectLevel.PREFIX in object_levels,
        "binding_generation": bool(binding_generation_observed),
    }
    compatible = all(flags.values())
    blocked_reason = _blocked_reason(
        vllm_version,
        lmcache_version,
        run_id,
        endpoint_identity,
        installation_evidence is not None,
        evidence_valid,
        connector_name,
        connector_version,
        allowed_actions,
        object_levels,
        binding_generation_observed,
    )
    return flags, compatible, blocked_reason


def _blocked_reason(
    vllm_version: str | None,
    lmcache_version: str | None,
    run_id: str | None,
    endpoint_identity: str | None,
    installation_evidence_present: bool,
    installation_evidence_valid: bool,
    connector_name: str | None,
    connector_version: str | None,
    allowed_actions: tuple[str, ...],
    object_levels: tuple[ObjectLevel, ...],
    binding_generation_observed: bool,
) -> str | None:
    if run_id is None:
        return "unknown_run_id"
    if endpoint_identity is None:
        return "invalid_loopback_endpoint"
    if not installation_evidence_present:
        return "missing_installation_evidence"
    if not installation_evidence_valid:
        return "invalid_installation_evidence"
    if vllm_version is None:
        return "unknown_vllm_version"
    if vllm_version != SUPPORTED_VLLM_VERSION:
        return f"unsupported_vllm_version:{vllm_version}"
    if lmcache_version is None:
        return "unknown_lmcache_version"
    if lmcache_version != SUPPORTED_LMCACHE_VERSION:
        return f"unsupported_lmcache_version:{lmcache_version}"
    if connector_name is None:
        return "unknown_connector_name"
    if connector_name != SUPPORTED_CONNECTOR_NAME:
        return f"unsupported_connector_name:{connector_name}"
    if connector_version is None:
        return "unknown_connector_version"
    if connector_version != SUPPORTED_LMCACHE_VERSION:
        return f"unsupported_connector_version:{connector_version}"
    if not allowed_actions:
        return "missing_supported_action"
    if not object_levels:
        return "missing_supported_object_level"
    if not binding_generation_observed:
        return "binding_generation_not_observed"
    return None
