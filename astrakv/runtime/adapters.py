"""Runtime adapter interfaces.

Adapters isolate AstraKV-W from third-party runtime internals. A vLLM, SGLang,
or TensorRT-LLM integration should implement this interface outside upstream
core code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from astrakv.kv_cache.metadata import KVChunkMeta
from astrakv.runtime.eviction import OfflineEvictionDecision, RuntimeActionResult, RuntimeCapabilities, RuntimeEvictionEvent


@dataclass(slots=True)
class RuntimeRequest:
    request_id: str
    token_ids: tuple[int, ...]
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(Protocol):
    """Boundary for runtime observation and optional policy execution.

    Third-party runtime adapters must advertise unsupported capabilities rather
    than letting a policy report pretend that a hint was executed.
    """

    name: str

    def describe(self) -> dict[str, Any]:
        """Return adapter capabilities and static metadata."""

    def discover_chunks(self, request: RuntimeRequest) -> list[KVChunkMeta]:
        """Map a runtime request to known or planned KV chunks."""

    def capabilities(self) -> RuntimeCapabilities:
        """Return object identity and execution capabilities."""

    def collect_runtime_events(self, *args: Any, **kwargs: Any) -> list[RuntimeEvictionEvent]:
        """Return normalized runtime events without modifying the backend."""

    def apply_hint(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        """Execute a supported action or return an explicit unsupported result."""
