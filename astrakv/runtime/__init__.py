"""Runtime skeleton and adapter boundaries for AstraKV-W."""

from .adapters import RuntimeAdapter, RuntimeRequest
from .backend_bridge import InMemoryLoopbackHookClient, InMemoryProtectedActionService, JsonHttpHookClient, OnlineBackendBridge
from .backend_capabilities import (
    BackendCapabilityPreflight,
    RuntimeProbeChallenge,
    RuntimeProbeProof,
    preflight_backend_capabilities,
)
from .backend_hook import (
    BackendActionCommand,
    BackendActionReceipt,
    BackendHookEvent,
    BackendObjectBinding,
    HookAction,
)
from .backend_binding_registry import BackendBindingRegistry, RequestContext
from .eviction import (
    MMapEvictionAdapter,
    ObjectLevel,
    OfflineEvictionDecision,
    RuntimeActionResult,
    RuntimeCapabilities,
    RuntimeEvictionEvent,
    VllmLmCacheArtifactAdapter,
)
from .object_manager import RuntimeObjectManager, RuntimeObjectSnapshot
from .online_controller import OnlinePolicyController
from .profile_db import LayerSensitivityRecord, ProfileDB, QualityGuardRecord
from .runtime_plan import RuntimeActionPlan, RuntimeProfileGuard
from .vm_backend import VMDemoConfig, VirtualMemoryDemoRunner

__all__ = [
    "RuntimeAdapter",
    "RuntimeActionResult",
    "RuntimeCapabilities",
    "RuntimeEvictionEvent",
    "RuntimeObjectManager",
    "RuntimeObjectSnapshot",
    "RuntimeRequest",
    "BackendHookEvent",
    "BackendObjectBinding",
    "HookAction",
    "BackendActionCommand",
    "BackendActionReceipt",
    "BackendBindingRegistry",
    "RequestContext",
    "JsonHttpHookClient",
    "InMemoryLoopbackHookClient",
    "InMemoryProtectedActionService",
    "OnlineBackendBridge",
    "BackendCapabilityPreflight",
    "RuntimeProbeChallenge",
    "RuntimeProbeProof",
    "preflight_backend_capabilities",
    "OnlinePolicyController",
    "ProfileDB",
    "LayerSensitivityRecord",
    "QualityGuardRecord",
    "RuntimeActionPlan",
    "RuntimeProfileGuard",
    "MMapEvictionAdapter",
    "ObjectLevel",
    "OfflineEvictionDecision",
    "VMDemoConfig",
    "VllmLmCacheArtifactAdapter",
    "VirtualMemoryDemoRunner",
]
