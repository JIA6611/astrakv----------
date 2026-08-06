"""Version-locked runtime instrumentation for LMCache 0.4.7."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import platform
import types
import contextvars
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from astrakv.runtime.backend_capabilities import BLOCKED_ACTION_REASONS
from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext, _canonical_key
from astrakv.runtime.backend_hook import BackendExecutionSpec, BackendHookEvent, BackendObjectBinding
from astrakv.runtime.backend_hook import HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.runtime_action_executor import RuntimeActionExecutor
from astrakv.runtime.request_context import (
    RuntimeRequestContextReceiver,
    RuntimeRequestIdentity,
)


SUPPORTED_VERSIONS = {"vllm": "0.23.0", "lmcache": "0.4.7"}
EventSink = Callable[[dict[str, Any]], None]
_LOCAL_CPU_BACKEND = "LocalCPUBackend"
_LOCAL_DISK_BACKEND = "LocalDiskBackend"
_RUNTIME_OWNER_ACTION_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "astrakv_lmcache047_runtime_owner_action_active",
    default=False,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _offload_ready(manager: Any) -> tuple[bool, str]:
    if manager is None:
        return False, BLOCKED_ACTION_REASONS["offload"]
    if not all(callable(getattr(manager, name, None)) for name in ("batched_contains", "batched_get", "batched_remove")):
        return False, BLOCKED_ACTION_REASONS["offload"]
    storage_backends = getattr(manager, "storage_backends", None)
    if not isinstance(storage_backends, dict):
        return False, BLOCKED_ACTION_REASONS["offload"]
    if _LOCAL_CPU_BACKEND not in storage_backends:
        return False, "offload_source_backend_missing:LocalCPUBackend"
    if _LOCAL_DISK_BACKEND not in storage_backends:
        return False, "offload_target_backend_missing:LocalDiskBackend"
    return True, ""


def _prefetch_ready(manager: Any) -> tuple[bool, str]:
    if manager is None:
        return False, BLOCKED_ACTION_REASONS["prefetch"]
    if not all(callable(getattr(manager, name, None)) for name in ("batched_contains", "batched_get")):
        return False, BLOCKED_ACTION_REASONS["prefetch"]
    storage_backends = getattr(manager, "storage_backends", None)
    if not isinstance(storage_backends, dict):
        return False, BLOCKED_ACTION_REASONS["prefetch"]
    if _LOCAL_CPU_BACKEND not in storage_backends:
        return False, "prefetch_target_backend_missing:LocalCPUBackend"
    if _LOCAL_DISK_BACKEND not in storage_backends:
        return False, "prefetch_source_backend_missing:LocalDiskBackend"
    return True, ""


def _load_ready(manager: Any) -> tuple[bool, str]:
    if manager is None:
        return False, BLOCKED_ACTION_REASONS["load"]
    storage_backends = getattr(manager, "storage_backends", None)
    if not isinstance(storage_backends, dict) or _LOCAL_DISK_BACKEND not in storage_backends:
        return False, "load_source_backend_missing:LocalDiskBackend"
    engine = getattr(manager, "lmcache_engine", None) or getattr(manager, "_lmcache_engine", None)
    if engine is None or not callable(getattr(engine, "retrieve", None)):
        return False, BLOCKED_ACTION_REASONS["load"]
    return True, ""


def _evict_ready(manager: Any) -> tuple[bool, str]:
    if manager is None:
        return False, BLOCKED_ACTION_REASONS["evict"]
    if not all(callable(getattr(manager, name, None)) for name in ("batched_contains", "batched_get", "batched_remove")):
        return False, BLOCKED_ACTION_REASONS["evict"]
    storage_backends = getattr(manager, "storage_backends", None)
    if not isinstance(storage_backends, dict) or _LOCAL_DISK_BACKEND not in storage_backends:
        return False, "evict_source_backend_missing:LocalDiskBackend"
    return True, ""


def _manager_contains(manager: Any, key: Any, *, location: str) -> bool:
    hit_chunks, _ = manager.batched_contains([key], [location], False)
    return int(hit_chunks) > 0


def _safe_manager_contains(manager: Any, key: Any, *, location: str) -> bool:
    try:
        return _manager_contains(manager, key, location=location)
    except Exception:
        return False


def _observed_resident_tier(manager: Any, key: Any) -> str:
    if manager is None:
        return "unknown"
    if _safe_manager_contains(manager, key, location=_LOCAL_CPU_BACKEND):
        return "cpu"
    if _safe_manager_contains(manager, key, location=_LOCAL_DISK_BACKEND):
        return "ssd"
    return "unknown"


def _event_metadata_from_storage_key(key: Any, *, action: str) -> dict[str, Any]:
    block_ids = _extract_named_int_sequence(key, ("block_ids", "blocks"))
    block_count = len(block_ids) if block_ids else _extract_named_int_hint(key, ("block_count", "num_blocks", "block_len"))
    block_size_tokens = _extract_named_int_hint(key, ("block_size_tokens", "chunk_size", "tokens_per_block", "block_size"))
    token_count = _extract_named_int_hint(key, ("token_count", "tokens", "num_tokens"))
    if token_count is None and block_count is not None and block_size_tokens is not None:
        token_count = block_count * block_size_tokens
    result: dict[str, Any] = {}
    if block_ids:
        result[f"block_ids_{action}"] = list(block_ids)
    if block_count is not None and block_count > 0:
        result[f"block_count_{action}"] = int(block_count)
    if token_count is not None and token_count > 0:
        result[f"token_count_{action}"] = int(token_count)
    if block_size_tokens is not None and block_size_tokens > 0:
        result["block_size_tokens"] = int(block_size_tokens)
    return result


def _extract_named_int_hint(value: Any, names: tuple[str, ...]) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            hinted = _coerce_positive_int(value.get(name))
            if hinted is not None:
                return hinted
        for item in value.values():
            hinted = _extract_named_int_hint(item, names)
            if hinted is not None:
                return hinted
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            hinted = _extract_named_int_hint(item, names)
            if hinted is not None:
                return hinted
        return None
    if hasattr(value, "__dict__"):
        return _extract_named_int_hint(vars(value), names)
    return None


def _extract_named_int_sequence(value: Any, names: tuple[str, ...]) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        for name in names:
            hinted = _sequence_of_ints(value.get(name))
            if hinted:
                return hinted
        for item in value.values():
            hinted = _extract_named_int_sequence(item, names)
            if hinted:
                return hinted
        return ()
    if isinstance(value, (list, tuple)):
        for item in value:
            hinted = _extract_named_int_sequence(item, names)
            if hinted:
                return hinted
        return ()
    if hasattr(value, "__dict__"):
        return _extract_named_int_sequence(vars(value), names)
    return ()


def _sequence_of_ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            return ()
        if number < 0:
            return ()
        items.append(number)
    return tuple(items)


def _owner_action_batched_get(manager: Any, key: Any, *, location: str) -> Any:
    """Read through the owner path without exposing the binding to other I/O."""
    token = _RUNTIME_OWNER_ACTION_ACTIVE.set(True)
    try:
        return manager.batched_get([key], location=location)
    finally:
        _RUNTIME_OWNER_ACTION_ACTIVE.reset(token)


def _owner_action_batched_put(manager: Any, key: Any, memory_objs: Any, *, location: str) -> Any:
    """Write through the owner path without exposing the binding to other I/O."""
    token = _RUNTIME_OWNER_ACTION_ACTIVE.set(True)
    try:
        return manager.batched_put([key], memory_objs, location=location)
    finally:
        _RUNTIME_OWNER_ACTION_ACTIVE.reset(token)


def _memory_obj_size_bytes(memory_obj: Any) -> int:
    for name in ("get_physical_size", "get_size"):
        callback = getattr(memory_obj, name, None)
        if callable(callback):
            try:
                return max(0, int(callback()))
            except (TypeError, ValueError):
                return 0
    return 0


def _slice_prefix(value: Any, count: int) -> Any:
    if count < 0:
        count = 0
    try:
        return value[:count]
    except Exception:
        return value


def _slot_count(slot_mapping: Any) -> int:
    try:
        return max(0, int(len(slot_mapping)))
    except Exception:
        return 0


def _tensor_or_buffer_nbytes(value: Any) -> int:
    if value is None:
        return 0
    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        try:
            return max(0, int(element_size()) * int(numel()))
        except Exception:
            return 0
    if isinstance(value, dict):
        return sum(_tensor_or_buffer_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_or_buffer_nbytes(item) for item in value)
    return 0


def _estimate_runtime_load_bytes(kvcaches: Any, slot_mapping: Any, loaded_tokens: int, estimated_bytes: int | None = None) -> int:
    if estimated_bytes is not None:
        try:
            return max(0, int(estimated_bytes))
        except (TypeError, ValueError):
            return 0
    total_bytes = _tensor_or_buffer_nbytes(kvcaches)
    slot_count = _slot_count(slot_mapping)
    if total_bytes <= 0 or slot_count <= 0 or loaded_tokens <= 0:
        return 0
    return max(0, int((total_bytes / slot_count) * loaded_tokens))


def _load_target_metric_metadata(load_target: "_RuntimeLoadTarget", token_count: int) -> dict[str, int]:
    normalized_tokens = max(0, int(token_count or 0))
    block_size_tokens = max(1, int(load_target.chunk_size or 1))
    if normalized_tokens <= 0:
        return {}
    block_count = (normalized_tokens + block_size_tokens - 1) // block_size_tokens
    return {
        "block_count_load": block_count,
        "token_count_load": normalized_tokens,
        "block_size_tokens": block_size_tokens,
    }


def _binding_load_metadata(binding: BackendObjectBinding | None) -> tuple[str, str, bool]:
    if binding is None:
        return ("", "", False)
    load_action = {}
    if binding.execution_spec is not None:
        load_action = dict(binding.execution_spec.actions.get("load") or {})
    load_metadata = dict(load_action.get("metadata") or {}) if isinstance(load_action.get("metadata"), dict) else {}
    load_target_id = str(
        load_action.get("load_target_id")
        or load_metadata.get("load_target_id")
        or binding.metadata.get("load_target_id")
        or ""
    )
    runtime_reqmeta_id = str(
        load_action.get("runtime_reqmeta_id")
        or load_metadata.get("runtime_reqmeta_id")
        or binding.metadata.get("runtime_reqmeta_id")
        or ""
    )
    native_request_load = bool(
        load_action.get("native_request_load")
        or load_metadata.get("native_request_load")
    )
    return (load_target_id, runtime_reqmeta_id, native_request_load)


def _truthy_token_count(mask: Any) -> int:
    if mask is None:
        return 0
    sum_attr = getattr(mask, "sum", None)
    if callable(sum_attr):
        try:
            summed = sum_attr()
            item = getattr(summed, "item", None)
            return max(0, int(item() if callable(item) else summed))
        except Exception:
            pass
    try:
        return max(0, sum(1 for item in mask if item))
    except Exception:
        return 0


def _build_load_token_mask(token_count: int, cached_tokens: int, chunk_size: int, *, preset: Any = None, slot_mapping: Any = None) -> Any:
    if preset is not None:
        mask = preset.clone() if callable(getattr(preset, "clone", None)) else preset[:]
    else:
        try:
            import torch  # type: ignore

            kwargs: dict[str, Any] = {"dtype": torch.bool}
            device = getattr(slot_mapping, "device", None)
            if device is not None:
                kwargs["device"] = device
            mask = torch.ones(token_count, **kwargs)
        except Exception:
            mask = [True] * token_count
    effective_chunk = max(1, int(chunk_size or 1))
    masked_token_count = max(0, int(cached_tokens) // effective_chunk * effective_chunk)
    if masked_token_count <= 0:
        return mask
    try:
        mask[:masked_token_count] = False
    except Exception:
        for index in range(min(masked_token_count, token_count)):
            mask[index] = False
    return mask


def _now_ns() -> int:
    return time.time_ns()


def _coerce_non_negative_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _normalize_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        for field in ("token_ids", "input_ids", "prompt_token_ids"):
            normalized = _normalize_token_ids(value.get(field))
            if normalized:
                return normalized
        return ()
    if hasattr(value, "tolist") and callable(getattr(value, "tolist", None)):
        try:
            return _normalize_token_ids(value.tolist())
        except Exception:
            return ()
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, bool)) for item in value):
            return tuple(int(item) for item in value)
        for item in value:
            normalized = _normalize_token_ids(item)
            if normalized:
                return normalized
    for field in ("token_ids", "input_ids", "prompt_token_ids"):
        candidate = getattr(value, field, None)
        normalized = _normalize_token_ids(candidate)
        if normalized:
            return normalized
    return ()


def _iter_candidate_values(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = [kwargs]
    values.extend(kwargs.values())
    values.extend(args)
    values.append(connector)
    parent = getattr(connector, "_parent", None)
    if parent is not None:
        values.append(parent)
        getter = getattr(parent, "_get_connector_metadata", None)
        if callable(getter):
            try:
                metadata = getter()
            except Exception:
                metadata = None
            if metadata is not None:
                values.append(metadata)
                requests = getattr(metadata, "requests", None)
                if isinstance(requests, (list, tuple)):
                    values.extend(requests)
    return tuple(values)


def _extract_runtime_reqmeta_id(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    for field in ("req_id", "runtime_reqmeta_id", "request_id"):
        value = kwargs.get(field)
        if value not in (None, ""):
            return str(value)
    for value in _iter_candidate_values(connector, args, kwargs):
        if isinstance(value, dict):
            for field in ("req_id", "runtime_reqmeta_id", "request_id"):
                candidate = value.get(field)
                if candidate not in (None, ""):
                    return str(candidate)
        for field in ("req_id", "runtime_reqmeta_id", "request_id"):
            candidate = getattr(value, field, None)
            if candidate not in (None, ""):
                return str(candidate)
    return ""


def _extract_slot_mapping(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in _iter_candidate_values(connector, args, kwargs):
        if isinstance(value, dict):
            for field in ("slot_mapping", "slots"):
                if value.get(field) not in (None, ""):
                    return value.get(field)
        for field in ("slot_mapping", "slots"):
            candidate = getattr(value, field, None)
            if candidate not in (None, ""):
                return candidate
    return ()


def _extract_token_ids(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
    explicit = _normalize_token_ids(kwargs.get("token_ids"))
    if explicit:
        return explicit
    for value in _iter_candidate_values(connector, args, kwargs):
        normalized = _normalize_token_ids(value)
        if normalized:
            return normalized
    return ()


def _extract_numeric_hint(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *field_names: str) -> int | None:
    for field in field_names:
        value = kwargs.get(field)
        coerced = _coerce_non_negative_int(value)
        if coerced is not None:
            return coerced
    for value in _iter_candidate_values(connector, args, kwargs):
        if isinstance(value, dict):
            for field in field_names:
                coerced = _coerce_non_negative_int(value.get(field))
                if coerced is not None:
                    return coerced
        for field in field_names:
            coerced = _coerce_non_negative_int(getattr(value, field, None))
            if coerced is not None:
                return coerced
    return None


def _extract_request_configs(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    for value in _iter_candidate_values(connector, args, kwargs):
        if isinstance(value, dict):
            for field in ("request_configs", "request_config", "configs", "config"):
                candidate = value.get(field)
                if isinstance(candidate, dict):
                    return dict(candidate)
        for field in ("request_configs", "request_config", "configs", "config"):
            candidate = getattr(value, field, None)
            if isinstance(candidate, dict):
                return dict(candidate)
    return {}


def _extract_kvcaches(connector: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    for value in _iter_candidate_values(connector, args, kwargs):
        if isinstance(value, dict):
            for field in ("kvcaches", "kv_caches", "kv_cache"):
                candidate = value.get(field)
                if isinstance(candidate, list):
                    return list(candidate)
                if isinstance(candidate, tuple):
                    return list(candidate)
        for field in ("kvcaches", "kv_caches", "kv_cache"):
            candidate = getattr(value, field, None)
            if isinstance(candidate, list):
                return list(candidate)
            if isinstance(candidate, tuple):
                return list(candidate)
    return []


@dataclass(slots=True)
class _RuntimeLoadTarget:
    target_id: str
    runtime_reqmeta_id: str
    token_ids: tuple[int, ...]
    slot_mapping: Any
    vllm_cached_tokens: int
    lmcache_cached_tokens: int
    chunk_size: int = 1
    can_load: bool = True
    request_configs: dict[str, Any] = field(default_factory=dict)
    kvcaches: list[Any] = field(default_factory=list)
    estimated_bytes: int | None = None
    target_tier: str = "gpu"
    token_mask: Any = None
    native_request_load: bool = False
    partial_load_target: dict[str, Any] | None = None
    state: str = "ready"
    created_at_ns: int = field(default_factory=time.time_ns)
    consumed_at_ns: int | None = None


def _supports_local_disk_completion_callback(backend: Any) -> bool:
    """Recognize only the exact LMCache 0.4.7 LocalDisk callback contract."""
    backend_cls = type(backend)
    if (
        backend_cls.__name__ != "LocalDiskBackend"
        or backend_cls.__module__ != "lmcache.v1.storage_backend.local_disk_backend"
    ):
        return False
    callback = getattr(backend, "batched_submit_put_task", None)
    if not callable(callback):
        return False
    return tuple(inspect.signature(callback).parameters) == (
        "keys", "memory_objs", "transfer_spec", "on_complete_callback",
    )


def probe_lmcache047_storage_contract(
    storage_manager_cls: type[Any], local_disk_backend: Any | None = None,
) -> dict[str, Any]:
    """Fail closed unless the installed class exposes verified completion/release callbacks.

    LMCache 0.4.7's StorageManager.batched_put only submits asynchronous backend
    work.  Its return is not a durable-store completion, and the class has no
    request release callback.  Keeping this explicit prevents observations from
    being promoted into executable bindings.
    """
    required = ("batched_put", "get", "batched_get", "remove")
    missing = tuple(name for name in required if not callable(getattr(storage_manager_cls, name, None)))
    compatible = not missing
    signatures = {
        name: tuple(inspect.signature(getattr(storage_manager_cls, name)).parameters)
        for name in required if name not in missing
    }
    local_disk_callback_verified = (
        local_disk_backend is not None and _supports_local_disk_completion_callback(local_disk_backend)
    )
    return {
        "schema": "astrakv-lmcache047-storage-contract-v2",
        "compatible": compatible,
        "missing_methods": list(missing),
        "method_parameters": {name: list(values) for name, values in signatures.items()},
        "local_disk_completion_callback_verified": local_disk_callback_verified,
        "action_registration_enabled": False,
        "blocked_reason": (
            None if not compatible else (
                "awaiting_completed_store_and_release" if local_disk_callback_verified
                else "no_verified_terminal_store_and_release_callbacks"
            )
        ),
    }


def installed_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in SUPPORTED_VERSIONS}


def patch_usage_context_cpu_info(usage_context_cls: type[Any] | None = None) -> None:
    """Keep LMCache telemetry failures from aborting cache-engine creation."""
    if usage_context_cls is None:
        from lmcache.usage_context import UsageContext
        usage_context_cls = UsageContext
    original = usage_context_cls._get_cpu_info
    if getattr(original, "__astrakv_cpuinfo_fallback__", False):
        return

    def safe_cpu_info(self: Any) -> tuple[int, str, str]:
        try:
            return original(self)
        except Exception as exc:
            print(f"AstraKV LMCache UsageContext cpuinfo fallback: {type(exc).__name__}", flush=True)
            return (os.cpu_count() or 1, platform.processor() or "unknown", "")

    safe_cpu_info.__astrakv_cpuinfo_fallback__ = True
    usage_context_cls._get_cpu_info = safe_cpu_info


@dataclass(slots=True)
class LMCache047ActionEndpoint:
    binding_registry: BackendBindingRegistry | None = None
    action_registration_enabled: bool = False
    _keys: dict[tuple[str, str], tuple[Any, Any]] = field(default_factory=dict)
    _completed_stores: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    _execution_specs: dict[tuple[str, str], BackendExecutionSpec] = field(default_factory=dict)
    _bindings_by_object: dict[tuple[str, str], BackendObjectBinding] = field(default_factory=dict)
    _load_targets: dict[str, _RuntimeLoadTarget] = field(default_factory=dict)
    _require_verified_local_disk_completion: bool = False
    _runtime_action_executor: RuntimeActionExecutor | None = field(default=None, init=False, repr=False)

    def remember(self, key: Any, manager: Any) -> str:
        """Legacy observation helper. It intentionally does not authorize actions."""
        return str(key)

    def register_binding(self, binding: BackendObjectBinding, key: Any, manager: Any) -> BackendExecutionSpec | None:
        """Register only a registry-issued backend identity, never an arbitrary key."""
        if self.binding_registry is None:
            return None
        inherited_binding = self._bindings_by_object.get((binding.object_level.value, binding.object_key))
        inherited_load_target_id, inherited_runtime_reqmeta_id, inherited_native_request_load = _binding_load_metadata(
            inherited_binding
        )
        known = self.binding_registry.current_binding(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            request_id=binding.request_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
        )
        if known is None or known.backend_object_id != binding.backend_object_id:
            return None
        if self._require_verified_local_disk_completion:
            completed = self._completed_stores.get(binding.binding_id)
            if completed is None:
                return None
            key, manager = completed
            if not self.binding_registry.authorize_action(
                binding_id=binding.binding_id, binding_generation=binding.binding_generation,
                backend_object_id=binding.backend_object_id, request_id=binding.request_id,
                object_key=binding.object_key, object_level=binding.object_level,
            ):
                return None
            self.action_registration_enabled = True
        elif not self.action_registration_enabled:
            return None
        self._keys[(binding.backend_object_id, binding.binding_id)] = (key, manager)
        spec = self.build_execution_spec(binding, key=key, manager=manager)
        registered_binding = BackendObjectBinding(
            run_id=binding.run_id,
            request_id=binding.request_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
            backend_object_id=binding.backend_object_id,
            binding_id=binding.binding_id,
            verified=binding.verified,
            metadata=dict(binding.metadata),
            binding_generation=binding.binding_generation,
            execution_spec=spec,
        )
        if inherited_load_target_id and inherited_runtime_reqmeta_id:
            registered_binding = self._update_binding_load_metadata(
                registered_binding,
                load_target_id=inherited_load_target_id,
                runtime_reqmeta_id=inherited_runtime_reqmeta_id,
                native_request_load=inherited_native_request_load,
            )
            spec = registered_binding.execution_spec or spec
        else:
            self._execution_specs[(binding.backend_object_id, binding.binding_id)] = spec
            self._bindings_by_object[(binding.object_level.value, binding.object_key)] = registered_binding
        return spec

    def mark_store_completed(self, binding: BackendObjectBinding | None, key: Any, manager: Any) -> None:
        """Remember a LocalDisk key only after its terminal callback is accepted."""
        if self.binding_registry is None or binding is None:
            return
        known = self.binding_registry.current_binding(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            request_id=binding.request_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
        )
        if known is not None and known.backend_object_id == binding.backend_object_id:
            self._completed_stores[binding.binding_id] = (key, manager)

    def register_load_target(
        self,
        *,
        target_id: str,
        runtime_reqmeta_id: str,
        token_ids: list[int] | tuple[int, ...],
        slot_mapping: Any,
        vllm_cached_tokens: int,
        lmcache_cached_tokens: int,
        chunk_size: int = 1,
        can_load: bool = True,
        request_configs: dict[str, Any] | None = None,
        kvcaches: list[Any] | None = None,
        estimated_bytes: int | None = None,
        target_tier: str = "gpu",
        token_mask: Any = None,
        native_request_load: bool = False,
        partial_load_target: dict[str, Any] | None = None,
    ) -> str:
        load_target = _RuntimeLoadTarget(
            target_id=str(target_id),
            runtime_reqmeta_id=str(runtime_reqmeta_id),
            token_ids=tuple(int(item) for item in token_ids),
            slot_mapping=slot_mapping,
            vllm_cached_tokens=max(0, int(vllm_cached_tokens)),
            lmcache_cached_tokens=max(0, int(lmcache_cached_tokens)),
            chunk_size=max(1, int(chunk_size or 1)),
            can_load=bool(can_load),
            request_configs=dict(request_configs or {}),
            kvcaches=list(kvcaches or []),
            estimated_bytes=estimated_bytes,
            target_tier=str(target_tier or "gpu"),
            token_mask=token_mask,
            native_request_load=bool(native_request_load),
            partial_load_target=(dict(partial_load_target) if isinstance(partial_load_target, dict) else None),
        )
        self._load_targets[load_target.target_id] = load_target
        return load_target.target_id

    def binding_for_object(self, object_key: str, object_level: ObjectLevel | str) -> BackendObjectBinding | None:
        try:
            normalized_level = object_level if isinstance(object_level, ObjectLevel) else ObjectLevel(str(object_level))
        except ValueError:
            return None
        return self._bindings_by_object.get((normalized_level.value, str(object_key or "")))

    def register_dynamic_load_target(
        self,
        *,
        object_key: str,
        object_level: ObjectLevel | str,
        target_id: str,
        runtime_reqmeta_id: str,
        token_ids: list[int] | tuple[int, ...],
        slot_mapping: Any,
        vllm_cached_tokens: int,
        lmcache_cached_tokens: int,
        chunk_size: int = 1,
        can_load: bool = True,
        request_configs: dict[str, Any] | None = None,
        kvcaches: list[Any] | None = None,
        estimated_bytes: int | None = None,
        target_tier: str = "gpu",
        token_mask: Any = None,
        native_request_load: bool = False,
        partial_load_target: dict[str, Any] | None = None,
    ) -> tuple[str, BackendObjectBinding | None]:
        load_target_id = self.register_load_target(
            target_id=target_id,
            runtime_reqmeta_id=runtime_reqmeta_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            vllm_cached_tokens=vllm_cached_tokens,
            lmcache_cached_tokens=lmcache_cached_tokens,
            chunk_size=chunk_size,
            can_load=can_load,
            request_configs=request_configs,
            kvcaches=kvcaches,
            estimated_bytes=estimated_bytes,
            target_tier=target_tier,
            token_mask=token_mask,
            native_request_load=native_request_load,
            partial_load_target=partial_load_target,
        )
        binding = self.binding_for_object(object_key, object_level)
        if binding is None:
            return load_target_id, None
        updated = self._update_binding_load_metadata(
            binding,
            load_target_id=load_target_id,
            runtime_reqmeta_id=runtime_reqmeta_id,
            native_request_load=native_request_load,
            partial_load_target=partial_load_target,
        )
        return load_target_id, updated

    def _update_binding_load_metadata(
        self,
        binding: BackendObjectBinding,
        *,
        load_target_id: str,
        runtime_reqmeta_id: str,
        native_request_load: bool = False,
        partial_load_target: dict[str, Any] | None = None,
    ) -> BackendObjectBinding:
        spec_key = (binding.backend_object_id, binding.binding_id)
        existing_spec = self._execution_specs.get(spec_key)
        key_entry = self._keys.get(spec_key)
        if key_entry is None or key_entry[1] is None:
            completed_entry = self._completed_stores.get(binding.binding_id)
            if completed_entry is not None:
                key_entry = completed_entry
                self._keys[spec_key] = completed_entry
        if key_entry is not None and (
            existing_spec is None
            or str((existing_spec.actions.get("load") or {}).get("status") or "") != "ready"
        ):
            existing_spec = self.build_execution_spec(binding, key=key_entry[0], manager=key_entry[1])
        base_spec = existing_spec or BackendExecutionSpec(
            spec_id=f"execspec:{binding.binding_id}:dynamic-load",
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
            runtime_owner="lmcache047-runtime-owner",
            owner_channel="lmcache047-owner-local",
            key_identity="dynamic-load-target",
            lifecycle=str(binding.metadata.get("lifecycle") or "released"),
            actions={},
            metadata={},
        )
        load_action = dict(base_spec.actions.get("load") or {})
        load_action_metadata = (
            dict(load_action.get("metadata"))
            if isinstance(load_action.get("metadata"), dict)
            else {}
        )
        load_action_metadata.update({
            "load_target_id": load_target_id,
            "runtime_reqmeta_id": runtime_reqmeta_id,
            "native_request_load": bool(native_request_load),
        })
        if isinstance(partial_load_target, dict):
            load_action_metadata["partial_load_target"] = dict(partial_load_target)
        load_action.update({
            "load_target_id": load_target_id,
            "runtime_reqmeta_id": runtime_reqmeta_id,
            "native_request_load": bool(native_request_load),
            "metadata": load_action_metadata,
        })
        actions = dict(base_spec.actions)
        actions["load"] = load_action
        spec = BackendExecutionSpec(
            spec_id=base_spec.spec_id,
            binding_id=base_spec.binding_id,
            binding_generation=base_spec.binding_generation,
            backend_object_id=base_spec.backend_object_id,
            object_key=base_spec.object_key,
            object_level=base_spec.object_level,
            runtime_owner=base_spec.runtime_owner,
            owner_channel=base_spec.owner_channel,
            key_identity=base_spec.key_identity,
            lifecycle=base_spec.lifecycle,
            actions=actions,
            metadata={
                **dict(base_spec.metadata),
                "load_target_id": load_target_id,
                "runtime_reqmeta_id": runtime_reqmeta_id,
                "native_request_load": bool(native_request_load),
                **({"partial_load_target": dict(partial_load_target)} if isinstance(partial_load_target, dict) else {}),
            },
        )
        self._execution_specs[spec_key] = spec
        updated_binding = BackendObjectBinding(
            run_id=binding.run_id,
            request_id=binding.request_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
            backend_object_id=binding.backend_object_id,
            binding_id=binding.binding_id,
            verified=binding.verified,
            metadata={
                **dict(binding.metadata),
                "load_target_id": load_target_id,
                "runtime_reqmeta_id": runtime_reqmeta_id,
                **({"partial_load_target": dict(partial_load_target)} if isinstance(partial_load_target, dict) else {}),
            },
            binding_generation=binding.binding_generation,
            execution_spec=spec,
        )
        self._bindings_by_object[(binding.object_level.value, binding.object_key)] = updated_binding
        return updated_binding

    def build_execution_spec(self, binding: BackendObjectBinding, *, key: Any, manager: Any) -> BackendExecutionSpec:
        key_identity = _canonical_key(key)
        manager_type = "" if manager is None else f"{type(manager).__module__}.{type(manager).__qualname__}"
        drop_ready = bool(
            self.binding_registry is not None
            and binding.binding_id in self._completed_stores
            and self.binding_registry.authorize_action(
                binding_id=binding.binding_id,
                binding_generation=binding.binding_generation,
                backend_object_id=binding.backend_object_id,
                request_id=binding.request_id,
                object_key=binding.object_key,
                object_level=binding.object_level,
            )
        )
        actions = {
            "drop": {
                "status": "ready" if drop_ready else "blocked",
                "executor": "DropExecutor",
                "owner_channel": "lmcache047-owner-local",
                "requires_reservation": True,
                "blocked_reason": None if drop_ready else "binding_not_yet_ready_for_owner_dispatch",
            },
        }
        offload_supported, offload_blocked_reason = (
            _offload_ready(manager) if drop_ready else (False, "binding_not_yet_ready_for_owner_dispatch")
        )
        actions["offload"] = {
            "status": "ready" if offload_supported else "blocked",
            "executor": "OffloadExecutor",
            "owner_channel": "lmcache047-owner-local",
            "requires_reservation": True,
            "blocked_reason": None if offload_supported else offload_blocked_reason,
        }
        prefetch_supported, prefetch_blocked_reason = (
            _prefetch_ready(manager) if drop_ready else (False, "binding_not_yet_ready_for_owner_dispatch")
        )
        actions["prefetch"] = {
            "status": "ready" if prefetch_supported else "blocked",
            "executor": "PrefetchExecutor",
            "owner_channel": "lmcache047-owner-local",
            "requires_reservation": True,
            "blocked_reason": None if prefetch_supported else prefetch_blocked_reason,
        }
        load_supported, load_blocked_reason = (
            _load_ready(manager) if drop_ready else (False, "binding_not_yet_ready_for_owner_dispatch")
        )
        actions["load"] = {
            "status": "ready" if load_supported else "blocked",
            "executor": "LoadExecutor",
            "owner_channel": "lmcache047-owner-local",
            "requires_reservation": True,
            "requires_dynamic_target": "load_target_id",
            "blocked_reason": None if load_supported else load_blocked_reason,
        }
        evict_supported, evict_blocked_reason = (
            _evict_ready(manager) if drop_ready else (False, "binding_not_yet_ready_for_owner_dispatch")
        )
        actions["evict"] = {
            "status": "ready" if evict_supported else "blocked",
            "executor": "EvictExecutor",
            "owner_channel": "lmcache047-owner-local",
            "requires_reservation": True,
            "blocked_reason": None if evict_supported else evict_blocked_reason,
        }
        digest = hashlib.sha256(
            json.dumps(
                {
                    "binding_id": binding.binding_id,
                    "backend_object_id": binding.backend_object_id,
                    "key_identity": key_identity,
                    "actions": actions,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        return BackendExecutionSpec(
            spec_id=f"execspec:{binding.binding_id}:{digest}",
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            object_key=binding.object_key,
            object_level=binding.object_level,
            runtime_owner="lmcache047-runtime-owner",
            owner_channel="lmcache047-owner-local",
            key_identity=key_identity,
            lifecycle=str(binding.metadata.get("lifecycle") or "unknown"),
            actions=actions,
            metadata={
                "connector_name": "lmcache-vllm-v1",
                "connector_version": "0.4.7",
                "manager_type": manager_type,
                "verified_terminal_store": binding.binding_id in self._completed_stores,
                "observed_resident_tier": _observed_resident_tier(manager, key),
            },
        )

    def execute_action(self, command: Any) -> dict[str, Any]:
        if self._runtime_action_executor is None:
            self._runtime_action_executor = RuntimeActionExecutor(self)
        return self._runtime_action_executor.execute(command)

    def _execute_drop_command(self, command: Any) -> dict[str, Any]:
        return self._drop_impl(
            command.backend_object_id,
            binding_id=command.binding_id,
            binding_generation=command.binding_generation,
            request_id=command.request_id,
            object_key=command.object_key,
            object_level=command.object_level,
            reservation_lease=command.metadata.get("reservation_lease"),
            command_id=command.command_id,
        )

    def _execute_offload_command(self, command: Any) -> dict[str, Any]:
        return self._offload_impl(
            command.backend_object_id,
            binding_id=command.binding_id,
            binding_generation=command.binding_generation,
            request_id=command.request_id,
            object_key=command.object_key,
            object_level=command.object_level,
            reservation_lease=command.metadata.get("reservation_lease"),
            command_id=command.command_id,
            target_tier=command.target_tier,
        )

    def _execute_prefetch_command(self, command: Any) -> dict[str, Any]:
        return self._prefetch_impl(
            command.backend_object_id,
            binding_id=command.binding_id,
            binding_generation=command.binding_generation,
            request_id=command.request_id,
            object_key=command.object_key,
            object_level=command.object_level,
            reservation_lease=command.metadata.get("reservation_lease"),
            command_id=command.command_id,
            target_tier=command.target_tier,
        )

    def _execute_load_command(self, command: Any) -> dict[str, Any]:
        return self._load_impl(
            command.backend_object_id,
            binding_id=command.binding_id,
            binding_generation=command.binding_generation,
            request_id=command.request_id,
            object_key=command.object_key,
            object_level=command.object_level,
            reservation_lease=command.metadata.get("reservation_lease"),
            command_id=command.command_id,
            target_tier=command.target_tier,
            load_target_id=command.metadata.get("load_target_id") or command.metadata.get("runtime_reqmeta_id"),
            partial_load_target=command.metadata.get("partial_load_target"),
        )

    def _execute_evict_command(self, command: Any) -> dict[str, Any]:
        return self._evict_impl(
            command.backend_object_id,
            binding_id=command.binding_id,
            binding_generation=command.binding_generation,
            request_id=command.request_id,
            object_key=command.object_key,
            object_level=command.object_level,
            reservation_lease=command.metadata.get("reservation_lease"),
            command_id=command.command_id,
            target_tier=command.target_tier,
        )

    def drop(
        self, backend_object_id: str, *, binding_id: str | None = None, binding_generation: int | None = None,
        request_id: str | None = None, object_key: str | None = None, object_level: Any = None,
        reservation_lease: str | None = None, command_id: str | None = None,
        locations: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._drop_impl(
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
            locations=locations,
        )

    def _drop_impl(
        self, backend_object_id: str, *, binding_id: str | None = None, binding_generation: int | None = None,
        request_id: str | None = None, object_key: str | None = None, object_level: Any = None,
        reservation_lease: str | None = None, command_id: str | None = None,
        locations: list[str] | None = None,
    ) -> dict[str, Any]:
        authorized = self._authorize_bound_action(
            "drop",
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
        )
        if isinstance(authorized, dict):
            return authorized
        key, manager = authorized
        try:
            removed = manager.remove(key, locations)
        except Exception:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            raise
        status = "completed" if removed else "not_found"
        assert reservation_lease is not None and command_id is not None
        self.binding_registry.complete_action(reservation_lease, command_id=command_id, status=status)
        return {"action": "drop", "backend_object_id": backend_object_id, "status": status, "removed": int(removed), "locations": list(locations or []), "reservation_lease": reservation_lease}

    def _offload_impl(
        self,
        backend_object_id: str,
        *,
        binding_id: str | None = None,
        binding_generation: int | None = None,
        request_id: str | None = None,
        object_key: str | None = None,
        object_level: Any = None,
        reservation_lease: str | None = None,
        command_id: str | None = None,
        target_tier: str = "unknown",
    ) -> dict[str, Any]:
        authorized = self._authorize_bound_action(
            "offload",
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
        )
        if isinstance(authorized, dict):
            return authorized
        key, manager = authorized
        offload_supported, blocked_reason = _offload_ready(manager)
        if not offload_supported:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "offload",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": blocked_reason,
            }
        if target_tier not in {"unknown", "", "ssd"}:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "offload",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": f"unsupported_offload_target_tier:{target_tier}",
            }
        if not _manager_contains(manager, key, location=_LOCAL_DISK_BACKEND):
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "offload",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": "offload_target_backend_miss:LocalDiskBackend",
                "source_location": _LOCAL_CPU_BACKEND,
                "target_location": _LOCAL_DISK_BACKEND,
            }
        if not _manager_contains(manager, key, location=_LOCAL_CPU_BACKEND):
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease, command_id=command_id, status="not_found", preserve_lifecycle=True,
            )
            return {
                "action": "offload",
                "backend_object_id": backend_object_id,
                "status": "not_found",
                "tier_before": "ssd",
                "tier_after": "ssd",
                "bytes": 0,
                "offloaded": 0,
                "locations": [_LOCAL_CPU_BACKEND],
                "source_location": _LOCAL_CPU_BACKEND,
                "target_location": _LOCAL_DISK_BACKEND,
                "reservation_lease": reservation_lease,
            }
        memory_objs = _owner_action_batched_get(manager, key, location=_LOCAL_CPU_BACKEND)
        bytes_offloaded = sum(_memory_obj_size_bytes(item) for item in memory_objs if item is not None)
        try:
            removed = int(manager.batched_remove([key], locations=[_LOCAL_CPU_BACKEND]) or 0)
        except Exception:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            raise
        status = "completed" if removed > 0 else "not_found"
        assert reservation_lease is not None and command_id is not None
        self.binding_registry.complete_action(
            reservation_lease, command_id=command_id, status=status, preserve_lifecycle=True,
        )
        return {
            "action": "offload",
            "backend_object_id": backend_object_id,
            "status": status,
            "tier_before": "cpu",
            "tier_after": "ssd",
            "bytes": bytes_offloaded if status == "completed" else 0,
            "offloaded": removed,
            "locations": [_LOCAL_CPU_BACKEND],
            "source_location": _LOCAL_CPU_BACKEND,
            "target_location": _LOCAL_DISK_BACKEND,
            "reservation_lease": reservation_lease,
        }

    def _load_impl(
        self,
        backend_object_id: str,
        *,
        binding_id: str | None = None,
        binding_generation: int | None = None,
        request_id: str | None = None,
        object_key: str | None = None,
        object_level: Any = None,
        reservation_lease: str | None = None,
        command_id: str | None = None,
        target_tier: str = "unknown",
        load_target_id: Any = None,
        partial_load_target: Any = None,
    ) -> dict[str, Any]:
        authorized = self._authorize_bound_action(
            "load",
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
        )
        if isinstance(authorized, dict):
            return authorized
        key, manager = authorized
        load_supported, blocked_reason = _load_ready(manager)
        if not load_supported:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": blocked_reason,
            }
        if target_tier not in {"unknown", "", "gpu", "runtime"}:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": f"unsupported_load_target_tier:{target_tier}",
            }
        target_key = "" if load_target_id is None else str(load_target_id)
        load_target = self._load_targets.get(target_key) if target_key else None
        if load_target is None:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": "load_target_not_available",
                "failure_reason": "load_target_missing_or_already_consumed",
                "load_target_id": target_key,
            }
        if load_target.state != "ready":
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": "load_target_not_available",
                "failure_reason": f"load_target_state:{load_target.state}",
                "load_target_id": load_target.target_id,
                "load_target_state": load_target.state,
                "load_target_created_at_ns": load_target.created_at_ns,
                "load_target_consumed_at_ns": load_target.consumed_at_ns,
            }
        load_target.state = "consumed"
        load_target.consumed_at_ns = time.time_ns()
        self._load_targets.pop(target_key, None)
        effective_partial_target = (
            dict(partial_load_target)
            if isinstance(partial_load_target, dict)
            else (
                dict(load_target.partial_load_target)
                if isinstance(load_target.partial_load_target, dict)
                else None
            )
        )
        partial_load_applied = False
        partial_load_reason = ""
        if not load_target.can_load:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease, command_id=command_id, status="completed", preserve_lifecycle=True,
            )
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "completed",
                "tier_before": load_target.target_tier,
                "tier_after": load_target.target_tier,
                "bytes": 0,
                "loaded": 0,
                "load_target_id": load_target.target_id,
                "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
                "partial_load_applied": False,
                "partial_load_reason": "can_load_disabled",
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "vllm_paged_kv",
                "reservation_lease": reservation_lease,
                **_load_target_metric_metadata(load_target, 0),
            }
        if not _manager_contains(manager, key, location=_LOCAL_DISK_BACKEND):
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(reservation_lease, command_id=command_id, status="not_found")
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "not_found",
                "failure_reason": "ssd_object_missing_at_load_execute",
                "disk_present_before": False,
                "tier_before": "unknown",
                "tier_after": "unknown",
                "bytes": 0,
                "loaded": 0,
                "load_target_id": load_target.target_id,
                "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
                "partial_load_applied": False,
                "partial_load_reason": "load_target_missing_or_already_consumed",
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "vllm_paged_kv",
                "reservation_lease": reservation_lease,
            }
        total_tokens = len(load_target.token_ids)
        lmcache_cached_tokens = min(total_tokens, max(0, int(load_target.lmcache_cached_tokens or total_tokens)))
        vllm_cached_tokens = min(lmcache_cached_tokens, max(0, int(load_target.vllm_cached_tokens or 0)))
        if effective_partial_target is not None:
            token_span = effective_partial_target.get("token_span")
            if isinstance(token_span, dict) and bool(effective_partial_target.get("allow_partial", True)):
                start_token = _coerce_non_negative_int(token_span.get("start_token")) or 0
                end_token = _coerce_non_negative_int(token_span.get("end_token")) or 0
                if start_token == 0 and end_token > 0:
                    lmcache_cached_tokens = min(lmcache_cached_tokens, end_token)
                    vllm_cached_tokens = min(vllm_cached_tokens, lmcache_cached_tokens)
                    partial_load_applied = lmcache_cached_tokens < total_tokens
                    partial_load_reason = "prefix_aligned_contiguous_partial_load"
                else:
                    partial_load_reason = "invalid_partial_token_span"
            else:
                partial_load_reason = "partial_load_not_allowed"
        expected_tokens = max(0, lmcache_cached_tokens - vllm_cached_tokens)
        if load_target.native_request_load:
            bytes_loaded = _estimate_runtime_load_bytes(
                load_target.kvcaches,
                load_target.slot_mapping,
                expected_tokens,
                estimated_bytes=load_target.estimated_bytes,
            )
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease, command_id=command_id, status="completed", preserve_lifecycle=True,
            )
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "completed",
                "tier_before": "ssd",
                "tier_after": load_target.target_tier,
                "bytes": bytes_loaded,
                "loaded": expected_tokens,
                "load_target_id": load_target.target_id,
                "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
                "native_request_load": True,
                "partial_load_applied": partial_load_applied,
                "partial_load_reason": partial_load_reason,
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "vllm_paged_kv",
                "reservation_lease": reservation_lease,
                **_load_target_metric_metadata(load_target, expected_tokens),
            }
        if expected_tokens == 0:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease, command_id=command_id, status="completed", preserve_lifecycle=True,
            )
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "completed",
                "tier_before": load_target.target_tier,
                "tier_after": load_target.target_tier,
                "bytes": 0,
                "loaded": 0,
                "load_target_id": load_target.target_id,
                "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
                "partial_load_applied": partial_load_applied,
                "partial_load_reason": partial_load_reason,
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "vllm_paged_kv",
                "reservation_lease": reservation_lease,
                **_load_target_metric_metadata(load_target, 0),
            }
        engine = getattr(manager, "lmcache_engine", None) or getattr(manager, "_lmcache_engine", None)
        token_mask = _build_load_token_mask(
            total_tokens,
            vllm_cached_tokens,
            load_target.chunk_size,
            preset=load_target.token_mask,
            slot_mapping=load_target.slot_mapping,
        )
        token_slice = list(load_target.token_ids[:lmcache_cached_tokens])
        token_mask_slice = _slice_prefix(token_mask, lmcache_cached_tokens)
        slot_mapping_slice = _slice_prefix(load_target.slot_mapping, lmcache_cached_tokens)
        try:
            ret_token_mask = engine.retrieve(
                token_slice,
                token_mask_slice,
                kvcaches=list(load_target.kvcaches),
                slot_mapping=slot_mapping_slice,
                vllm_cached_tokens=vllm_cached_tokens,
                request_configs=dict(load_target.request_configs),
                req_id=load_target.runtime_reqmeta_id,
            )
        except Exception as exc:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "load",
                "backend_object_id": backend_object_id,
                "status": "failed",
                "failure_reason": "runtime_retrieve_exception",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "load_target_id": load_target.target_id,
                "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
                "partial_load_applied": partial_load_applied,
                "partial_load_reason": partial_load_reason,
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "vllm_paged_kv",
                "reservation_lease": reservation_lease,
            }
        loaded_tokens = _truthy_token_count(ret_token_mask)
        status = "completed" if loaded_tokens >= expected_tokens else "failed"
        bytes_loaded = _estimate_runtime_load_bytes(
            load_target.kvcaches,
            load_target.slot_mapping,
            loaded_tokens,
            estimated_bytes=load_target.estimated_bytes,
        )
        assert reservation_lease is not None and command_id is not None
        self.binding_registry.complete_action(
            reservation_lease, command_id=command_id, status=status, preserve_lifecycle=True,
        )
        return {
            "action": "load",
            "backend_object_id": backend_object_id,
            "status": status,
            "failure_reason": None if status == "completed" else "loaded_token_count_below_expected",
            "expected_tokens": expected_tokens,
            "loaded_tokens": loaded_tokens,
            "tier_before": "ssd",
            "tier_after": load_target.target_tier,
            "bytes": bytes_loaded if status == "completed" else 0,
            "loaded": loaded_tokens if status == "completed" else 0,
            "load_target_id": load_target.target_id,
            "runtime_reqmeta_id": load_target.runtime_reqmeta_id,
            "partial_load_applied": partial_load_applied,
            "partial_load_reason": partial_load_reason,
            "source_location": _LOCAL_DISK_BACKEND,
            "target_location": "vllm_paged_kv",
            "reservation_lease": reservation_lease,
            **_load_target_metric_metadata(load_target, loaded_tokens if status == "completed" else 0),
        }

    def _evict_impl(
        self,
        backend_object_id: str,
        *,
        binding_id: str | None = None,
        binding_generation: int | None = None,
        request_id: str | None = None,
        object_key: str | None = None,
        object_level: Any = None,
        reservation_lease: str | None = None,
        command_id: str | None = None,
        target_tier: str = "unknown",
    ) -> dict[str, Any]:
        authorized = self._authorize_bound_action(
            "evict",
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
        )
        if isinstance(authorized, dict):
            return authorized
        key, manager = authorized
        evict_supported, blocked_reason = _evict_ready(manager)
        if not evict_supported:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "evict",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": blocked_reason,
            }
        if target_tier not in {"unknown", "", "ssd"}:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "evict",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": f"unsupported_evict_target_tier:{target_tier}",
            }
        cpu_present_before = _manager_contains(manager, key, location=_LOCAL_CPU_BACKEND)
        if not _manager_contains(manager, key, location=_LOCAL_DISK_BACKEND):
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease,
                command_id=command_id,
                status="not_found",
                preserve_lifecycle=cpu_present_before,
            )
            return {
                "action": "evict",
                "backend_object_id": backend_object_id,
                "status": "not_found",
                "tier_before": "cpu" if cpu_present_before else "unknown",
                "tier_after": "cpu" if cpu_present_before else "unknown",
                "bytes": 0,
                "evicted": 0,
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": "none",
                "locations": [_LOCAL_DISK_BACKEND],
                "reservation_lease": reservation_lease,
            }
        try:
            memory_objs = _owner_action_batched_get(manager, key, location=_LOCAL_DISK_BACKEND)
            bytes_evicted = sum(_memory_obj_size_bytes(item) for item in memory_objs if item is not None)
            cpu_present_after_read = _manager_contains(manager, key, location=_LOCAL_CPU_BACKEND)
            removed = int(manager.batched_remove([key], locations=[_LOCAL_DISK_BACKEND]) or 0)
            if not cpu_present_before and cpu_present_after_read:
                manager.batched_remove([key], locations=[_LOCAL_CPU_BACKEND])
        except Exception:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            raise
        status = "completed" if removed > 0 else "not_found"
        assert reservation_lease is not None and command_id is not None
        self.binding_registry.complete_action(
            reservation_lease,
            command_id=command_id,
            status=status,
            preserve_lifecycle=cpu_present_before,
        )
        return {
            "action": "evict",
            "backend_object_id": backend_object_id,
            "status": status,
            "tier_before": "ssd",
            "tier_after": "cpu" if cpu_present_before else "none",
            "bytes": bytes_evicted if status == "completed" else 0,
            "evicted": removed,
            "source_location": _LOCAL_DISK_BACKEND,
            "target_location": "none",
            "locations": [_LOCAL_DISK_BACKEND],
            "reservation_lease": reservation_lease,
        }

    def _prefetch_impl(
        self,
        backend_object_id: str,
        *,
        binding_id: str | None = None,
        binding_generation: int | None = None,
        request_id: str | None = None,
        object_key: str | None = None,
        object_level: Any = None,
        reservation_lease: str | None = None,
        command_id: str | None = None,
        target_tier: str = "unknown",
    ) -> dict[str, Any]:
        authorized = self._authorize_bound_action(
            "prefetch",
            backend_object_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            request_id=request_id,
            object_key=object_key,
            object_level=object_level,
            reservation_lease=reservation_lease,
            command_id=command_id,
        )
        if isinstance(authorized, dict):
            return authorized
        key, manager = authorized
        prefetch_supported, blocked_reason = _prefetch_ready(manager)
        if not prefetch_supported:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "prefetch",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": blocked_reason,
            }
        if target_tier not in {"unknown", "", "cpu"}:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "prefetch",
                "backend_object_id": backend_object_id,
                "status": "blocked",
                "blocked_reason": f"unsupported_prefetch_target_tier:{target_tier}",
            }
        cpu_present_before = _safe_manager_contains(manager, key, location=_LOCAL_CPU_BACKEND)
        disk_present_before = _safe_manager_contains(manager, key, location=_LOCAL_DISK_BACKEND)
        if cpu_present_before:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(
                reservation_lease, command_id=command_id, status="completed", preserve_lifecycle=True,
            )
            return {
                "action": "prefetch",
                "backend_object_id": backend_object_id,
                "status": "completed",
                "tier_before": "cpu",
                "tier_after": "cpu",
                "bytes": 0,
                "prefetched": 0,
                "failure_reason": "already_cpu_resident",
                "disk_present_before": disk_present_before,
                "cpu_present_before": cpu_present_before,
                "cpu_present_after": True,
                "locations": [_LOCAL_DISK_BACKEND],
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": _LOCAL_CPU_BACKEND,
                "reservation_lease": reservation_lease,
            }
        if not disk_present_before:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.complete_action(reservation_lease, command_id=command_id, status="not_found")
            return {
                "action": "prefetch",
                "backend_object_id": backend_object_id,
                "status": "not_found",
                "tier_before": "unknown",
                "tier_after": "unknown",
                "bytes": 0,
                "prefetched": 0,
                "failure_reason": "ssd_object_missing_at_prefetch_execute",
                "disk_present_before": False,
                "cpu_present_before": cpu_present_before,
                "cpu_present_after": False,
                "locations": [_LOCAL_DISK_BACKEND],
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": _LOCAL_CPU_BACKEND,
                "reservation_lease": reservation_lease,
            }
        try:
            memory_objs = _owner_action_batched_get(manager, key, location=_LOCAL_DISK_BACKEND)
            missing_memory_obj_count = sum(1 for item in memory_objs if item is None)
            bytes_prefetched = sum(_memory_obj_size_bytes(item) for item in memory_objs if item is not None)
            cpu_present = _safe_manager_contains(manager, key, location=_LOCAL_CPU_BACKEND)
            if not cpu_present:
                batched_put = getattr(manager, "batched_put", None)
                if callable(batched_put):
                    _owner_action_batched_put(manager, key, memory_objs, location=_LOCAL_CPU_BACKEND)
                    cpu_present = _safe_manager_contains(manager, key, location=_LOCAL_CPU_BACKEND)
        except Exception as exc:
            assert reservation_lease is not None and command_id is not None
            self.binding_registry.cancel_action_reservation(reservation_lease, command_id=command_id)
            return {
                "action": "prefetch",
                "backend_object_id": backend_object_id,
                "status": "failed",
                "failure_reason": "prefetch_backend_exception",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "disk_present_before": disk_present_before,
                "cpu_present_before": cpu_present_before,
                "cpu_present_after": False,
                "prefetched": 0,
                "source_location": _LOCAL_DISK_BACKEND,
                "target_location": _LOCAL_CPU_BACKEND,
                "reservation_lease": reservation_lease,
            }
        status = "completed" if cpu_present and missing_memory_obj_count == 0 else "failed"
        assert reservation_lease is not None and command_id is not None
        self.binding_registry.complete_action(
            reservation_lease, command_id=command_id, status=status, preserve_lifecycle=True,
        )
        return {
            "action": "prefetch",
            "backend_object_id": backend_object_id,
            "status": status,
            "failure_reason": None if status == "completed" else (
                "missing_memory_objects" if missing_memory_obj_count else "cpu_residency_not_observed_after_prefetch"
            ),
            "disk_present_before": disk_present_before,
            "cpu_present_before": cpu_present_before,
            "cpu_present_after": cpu_present,
            "missing_memory_obj_count": missing_memory_obj_count,
            "memory_obj_count": len(memory_objs),
            "tier_before": "ssd",
            "tier_after": "cpu" if cpu_present else "ssd",
            "bytes": bytes_prefetched if status == "completed" else 0,
            "prefetched": 1 if status == "completed" else 0,
            "locations": [_LOCAL_DISK_BACKEND],
            "source_location": _LOCAL_DISK_BACKEND,
            "target_location": _LOCAL_CPU_BACKEND,
            "reservation_lease": reservation_lease,
        }

    def _authorize_bound_action(
        self,
        action_name: str,
        backend_object_id: str,
        *,
        binding_id: str | None,
        binding_generation: int | None,
        request_id: str | None,
        object_key: str | None,
        object_level: Any,
        reservation_lease: str | None,
        command_id: str | None,
    ) -> tuple[Any, Any] | dict[str, Any]:
        if not self.action_registration_enabled:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "observational_only"}
        if self.binding_registry is None or not binding_id or binding_generation is None or not request_id or not object_key:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "unbound"}
        try:
            normalized_level = ObjectLevel(object_level)
        except ValueError:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "unbound"}
        entry = self._keys.get((backend_object_id, binding_id))
        if entry is None:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "unbound"}
        if not self.binding_registry.authorize_action(
            binding_id=binding_id, binding_generation=binding_generation, backend_object_id=backend_object_id,
            request_id=request_id, object_key=object_key, object_level=normalized_level,
        ) and reservation_lease is None:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "not_eligible"}
        if not reservation_lease or not command_id:
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "reservation_required"}
        if not self.binding_registry.consume_action_reservation(
            reservation_lease, command_id=command_id, binding_id=binding_id,
            binding_generation=binding_generation, backend_object_id=backend_object_id,
        ):
            return {"action": action_name, "backend_object_id": backend_object_id, "status": "reservation_not_available"}
        return entry


class LMCache047RequestContextConsumer:
    """Convert authenticated runtime associations into registry request context.

    A context publication alone deliberately has no binding effect.  The
    receiver records it as ``recorded``; this consumer resolves it only once a
    connector has observed a concrete LMCache ``ReqMeta.req_id`` and supplied
    the matching logical identity.
    """

    def __init__(self, receiver: RuntimeRequestContextReceiver) -> None:
        self._receiver = receiver
        self._contexts: dict[str, RequestContext] = {}

    def associate(self, runtime_reqmeta_id: str, identity: RuntimeRequestIdentity) -> RequestContext | None:
        try:
            self._receiver.associate_runtime_request(runtime_reqmeta_id, identity)
        except ValueError:
            return None
        context = self._receiver.associated_context(runtime_reqmeta_id)
        if context is None:
            return None
        binding_context = RequestContext(
            run_id=context.run_id,
            request_id=context.request_id,
            object_key=_request_context_object_key(context, runtime_reqmeta_id),
            object_level=ObjectLevel.PREFIX,
            metadata=dict(context.metadata),
        )
        self._contexts[runtime_reqmeta_id] = binding_context
        return binding_context

    def context_for(self, runtime_reqmeta_id: str) -> RequestContext | None:
        return self._contexts.get(runtime_reqmeta_id)


def _request_context_object_key(context: Any, runtime_reqmeta_id: str) -> str:
    metadata = dict(getattr(context, "metadata", {}) or {})
    for field in ("object_key", "cache_key", "prefix_id", "workflow_id", "request_id"):
        value = str(metadata.get(field) or "")
        if value:
            return value
    return f"lmcache:reqmeta:{runtime_reqmeta_id}"


@dataclass(slots=True)
class _TrackedStore:
    key: Any
    manager: Any
    context: RequestContext
    operation_lease: str
    runtime_reqmeta_id: str
    store_completed: bool = False
    unpin_observed: bool = False
    release_source: str = ""
    released: bool = False


@dataclass(slots=True)
class _TrackedAccess:
    key: Any
    manager: Any
    context: RequestContext
    runtime_reqmeta_id: str


@dataclass(slots=True)
class _PendingLoadRegistration:
    runtime_reqmeta_id: str
    object_key: str
    object_level: ObjectLevel
    token_ids: tuple[int, ...] = ()
    slot_mapping: Any = ()
    vllm_cached_tokens: int = 0
    lmcache_cached_tokens: int = 0
    chunk_size: int = 1
    can_load: bool = True
    request_configs: dict[str, Any] = field(default_factory=dict)
    kvcaches: list[Any] = field(default_factory=list)
    estimated_bytes: int | None = None
    target_tier: str = "gpu"
    token_mask: Any = None
    emitted: bool = False
    native_request_load: bool = True


@dataclass(slots=True)
class _ActionCapableManagerProxy:
    storage_manager: Any
    lmcache_engine: Any
    storage_backends: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.storage_manager, name)


def _action_capable_manager(
    storage_manager: Any,
    *,
    action_manager: Any = None,
    lmcache_engine: Any = None,
) -> Any:
    manager = action_manager if action_manager is not None else storage_manager
    manager_backends = getattr(manager, "storage_backends", None)
    storage_backends = getattr(storage_manager, "storage_backends", None)
    engine = (
        getattr(manager, "lmcache_engine", None)
        or getattr(manager, "_lmcache_engine", None)
        or lmcache_engine
    )
    if (
        engine is None
        or (
            callable(getattr(engine, "retrieve", None))
            and isinstance(manager_backends, dict)
            and _LOCAL_DISK_BACKEND in manager_backends
        )
    ):
        return manager
    return _ActionCapableManagerProxy(
        storage_manager=storage_manager,
        lmcache_engine=engine,
        storage_backends=(
            storage_backends
            if isinstance(storage_backends, dict)
            else manager_backends if isinstance(manager_backends, dict)
            else {}
        ),
    )


class _ConnectorLifecycle:
    """Track store and cache-hit request lifetimes while keeping completion fail-closed."""

    def __init__(
        self, event_sink: EventSink, binding_registry: BackendBindingRegistry | None,
        consumer: LMCache047RequestContextConsumer | None,
        identity_provider: Callable[[str], RuntimeRequestIdentity | None] | None,
        endpoint: LMCache047ActionEndpoint,
    ) -> None:
        self._event_sink = event_sink
        self._binding_registry = binding_registry
        self._consumer = consumer
        self._identity_provider = identity_provider
        self._endpoint = endpoint
        self._active_reqmeta_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "astrakv_lmcache047_active_reqmeta_id", default=None,
        )
        self._active_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
            "astrakv_lmcache047_active_request_context", default=None,
        )
        self._stores: dict[str, list[_TrackedStore]] = {}
        self._accesses: dict[str, list[_TrackedAccess]] = {}
        self._pending_loads: dict[str, _PendingLoadRegistration] = {}
        # Some controlled validation scenarios need to exercise SSD prefetch
        # without turning the same native request into a direct load demand.
        self._disable_native_request_load = _env_flag(
            "ASTRAKV_RUNTIME_DISABLE_NATIVE_REQUEST_LOAD",
        )

    def enter_runtime_request(self, runtime_reqmeta_id: str) -> tuple[contextvars.Token[str | None], contextvars.Token[RequestContext | None]]:
        context = self._request_context_for(runtime_reqmeta_id)
        return (
            self._active_reqmeta_id.set(runtime_reqmeta_id),
            self._active_context.set(context),
        )

    def enter_store(self, runtime_reqmeta_id: str) -> tuple[contextvars.Token[str | None], contextvars.Token[RequestContext | None]]:
        return self.enter_runtime_request(runtime_reqmeta_id)

    def exit_runtime_request(self, tokens: tuple[contextvars.Token[str | None], contextvars.Token[RequestContext | None]]) -> None:
        self._active_context.reset(tokens[1])
        self._active_reqmeta_id.reset(tokens[0])

    def exit_store(self, tokens: tuple[contextvars.Token[str | None], contextvars.Token[RequestContext | None]]) -> None:
        self.exit_runtime_request(tokens)

    def active_context(self) -> RequestContext | None:
        return self._active_context.get()

    def _request_context_for(self, runtime_reqmeta_id: str) -> RequestContext | None:
        if self._consumer is None or self._identity_provider is None:
            return None
        identity = self._identity_provider(runtime_reqmeta_id)
        if identity is None:
            return self._consumer.context_for(runtime_reqmeta_id)
        associated = self._consumer.associate(runtime_reqmeta_id, identity)
        return associated or self._consumer.context_for(runtime_reqmeta_id)

    def track_submission(self, key: Any, observation: Any, context: RequestContext | None, manager: Any = None) -> None:
        runtime_reqmeta_id = self._active_reqmeta_id.get()
        if runtime_reqmeta_id is None or context is None or observation.event is None:
            return
        event = observation.event
        event.metadata.setdefault("runtime_reqmeta_id", runtime_reqmeta_id)
        if event.action in {HookAction.CACHE_HIT, HookAction.CACHE_LOAD} and event.status in {"completed", "ok", "executed"}:
            self._accesses.setdefault(runtime_reqmeta_id, []).append(
                _TrackedAccess(key, manager, context, runtime_reqmeta_id)
            )
            return
        if event.action is not HookAction.CACHE_STORE:
            return
        event.metadata.update({
            "store_completion": "unproven_batched_put_is_non_blocking",
            "observational_only": True,
        })
        operation_lease = str(event.metadata.get("operation_lease") or "")
        if not operation_lease:
            return
        self._stores.setdefault(runtime_reqmeta_id, []).append(
            _TrackedStore(key, manager, context, operation_lease, runtime_reqmeta_id)
        )

    def fail_store(self, runtime_reqmeta_id: str) -> None:
        if self._binding_registry is None:
            return
        for tracked in self._stores.get(runtime_reqmeta_id, []):
            if tracked.released:
                continue
            observation = self._binding_registry.complete_operation(
                tracked.key, HookAction.CACHE_STORE, "failed", tracked.context, tracked.operation_lease,
                metadata={
                    "runtime_reqmeta_id": runtime_reqmeta_id,
                    "terminal_condition": "lmcache_engine_store_exception",
                    "observational_only": True,
                },
            )
            for record in observation.records:
                self._event_sink(record)

    def release_after_unpin(self, runtime_reqmeta_id: str, *, source: str) -> None:
        if self._binding_registry is None:
            return
        for tracked in self._stores.get(runtime_reqmeta_id, []):
            if tracked.released:
                continue
            tracked.unpin_observed = True
            tracked.release_source = source
            self._release_if_ready(tracked)
        accesses = self._accesses.pop(runtime_reqmeta_id, [])
        for tracked in accesses:
            self._release_access(tracked, source=source)

    def complete_store(self, key: Any, operation_lease: str) -> None:
        for stores in self._stores.values():
            for tracked in stores:
                if (
                    not tracked.released
                    and tracked.operation_lease == operation_lease
                    and _canonical_key(tracked.key) == _canonical_key(key)
                ):
                    tracked.store_completed = True
                    self._release_if_ready(tracked)
                    return

    def note_load_target(
        self,
        runtime_reqmeta_id: str,
        *,
        token_ids: Any = None,
        slot_mapping: Any = None,
        vllm_cached_tokens: Any = None,
        lmcache_cached_tokens: Any = None,
        chunk_size: Any = None,
        can_load: Any = None,
        request_configs: dict[str, Any] | None = None,
        kvcaches: list[Any] | None = None,
        estimated_bytes: Any = None,
        target_tier: str | None = None,
        token_mask: Any = None,
        native_request_load: Any = None,
    ) -> BackendObjectBinding | None:
        if self._disable_native_request_load:
            return None
        reqmeta_id = str(runtime_reqmeta_id or "")
        if not reqmeta_id:
            return None
        context = self._request_context_for(reqmeta_id)
        if context is None:
            return None
        pending = self._pending_loads.get(reqmeta_id)
        if pending is None:
            pending = _PendingLoadRegistration(
                runtime_reqmeta_id=reqmeta_id,
                object_key=context.object_key,
                object_level=context.object_level,
            )
            self._pending_loads[reqmeta_id] = pending
        pending.object_key = context.object_key
        pending.object_level = context.object_level
        normalized_tokens = _normalize_token_ids(token_ids)
        if normalized_tokens:
            pending.token_ids = normalized_tokens
        if slot_mapping not in (None, ""):
            pending.slot_mapping = slot_mapping
        if pending.slot_mapping in (None, "", ()) and pending.token_ids:
            pending.slot_mapping = list(range(len(pending.token_ids)))
        normalized_vllm = _coerce_non_negative_int(vllm_cached_tokens)
        if normalized_vllm is not None:
            pending.vllm_cached_tokens = normalized_vllm
        normalized_lmcache = _coerce_non_negative_int(lmcache_cached_tokens)
        if normalized_lmcache is not None:
            pending.lmcache_cached_tokens = normalized_lmcache
        elif pending.token_ids:
            pending.lmcache_cached_tokens = max(pending.lmcache_cached_tokens, len(pending.token_ids))
        normalized_chunk = _coerce_positive_int(chunk_size)
        if normalized_chunk is not None:
            pending.chunk_size = normalized_chunk
        if can_load is not None:
            pending.can_load = bool(can_load)
        if request_configs:
            pending.request_configs = dict(request_configs)
        if kvcaches:
            pending.kvcaches = list(kvcaches)
        normalized_bytes = _coerce_non_negative_int(estimated_bytes)
        if normalized_bytes is not None:
            pending.estimated_bytes = normalized_bytes
        if token_mask is not None:
            pending.token_mask = token_mask
        if target_tier:
            pending.target_tier = str(target_tier)
        if native_request_load is not None:
            pending.native_request_load = bool(native_request_load)
        partial_load_target = None
        if isinstance(request_configs, dict):
            maybe_partial = request_configs.get("partial_load_target")
            if isinstance(maybe_partial, dict):
                partial_load_target = dict(maybe_partial)
        if not pending.token_ids:
            return None
        pending.lmcache_cached_tokens = min(
            len(pending.token_ids),
            max(pending.vllm_cached_tokens, pending.lmcache_cached_tokens or len(pending.token_ids)),
        )
        load_target_id, binding = self._endpoint.register_dynamic_load_target(
            object_key=pending.object_key,
            object_level=pending.object_level,
            target_id=f"load-target:{reqmeta_id}",
            runtime_reqmeta_id=reqmeta_id,
            token_ids=list(pending.token_ids),
            slot_mapping=pending.slot_mapping,
            vllm_cached_tokens=min(pending.vllm_cached_tokens, pending.lmcache_cached_tokens),
            lmcache_cached_tokens=pending.lmcache_cached_tokens,
            chunk_size=pending.chunk_size,
            can_load=pending.can_load,
            request_configs=dict(pending.request_configs),
            kvcaches=list(pending.kvcaches),
            estimated_bytes=pending.estimated_bytes,
            target_tier=pending.target_tier,
            token_mask=pending.token_mask,
            native_request_load=pending.native_request_load,
            partial_load_target=partial_load_target,
        )
        if binding is not None:
            self._event_sink(binding.to_record())
            if not pending.emitted:
                self._event_sink(BackendHookEvent(
                    run_id=binding.run_id,
                    event_id=f"{binding.binding_id}:load-target:{reqmeta_id}",
                    request_id=binding.request_id,
                    object_key=binding.object_key,
                    object_level=binding.object_level,
                    backend_object_id=binding.backend_object_id,
                    action=HookAction.CACHE_LOAD,
                    status="available",
                    timestamp_ns=_now_ns(),
                    tier_before="ssd",
                    tier_after="ssd",
                    bytes=pending.estimated_bytes,
                    binding_generation=binding.binding_generation,
                    metadata={
                        "binding_id": binding.binding_id,
                        "load_target_id": load_target_id,
                        "runtime_reqmeta_id": reqmeta_id,
                        "load_target_ready": True,
                        "dispatch_signal": "dynamic_load_target_ready",
                        "vllm_cached_tokens": pending.vllm_cached_tokens,
                        "lmcache_cached_tokens": pending.lmcache_cached_tokens,
                        "target_tier": pending.target_tier,
                        "native_request_load": pending.native_request_load,
                        **_load_target_metric_metadata(
                            self._endpoint._load_targets[load_target_id],
                            max(0, pending.lmcache_cached_tokens - pending.vllm_cached_tokens),
                        ),
                    },
                ).to_record())
                pending.emitted = True
        return binding

    def _release_if_ready(self, tracked: _TrackedStore) -> None:
        if self._binding_registry is None or not tracked.unpin_observed:
            return
        if self._endpoint._require_verified_local_disk_completion and not tracked.store_completed:
            return
        resident_tier = _observed_resident_tier(tracked.manager, tracked.key)
        observation = self._binding_registry.observe(
            tracked.key, HookAction.RELEASE, "completed", tracked.context,
            tier_before=resident_tier,
            tier_after=resident_tier,
            metadata={
                "runtime_reqmeta_id": tracked.runtime_reqmeta_id,
                "release_condition": tracked.release_source,
                "store_completion": (
                    "lmcache_local_disk_write_and_index_complete"
                    if tracked.store_completed else "unproven_batched_put_is_non_blocking"
                ),
                "observational_only": not tracked.store_completed,
            },
        )
        tracked.released = True
        if observation.binding is not None:
            execution_spec = self._endpoint.register_binding(observation.binding, tracked.key, tracked.manager)
            if execution_spec is not None and observation.binding_record is not None:
                observation.binding_record["execution_spec"] = execution_spec.to_record()
        for record in observation.records:
            self._event_sink(record)

    def _release_access(self, tracked: _TrackedAccess, *, source: str) -> None:
        if self._binding_registry is None:
            return
        resident_tier = _observed_resident_tier(tracked.manager, tracked.key)
        observation = self._binding_registry.observe(
            tracked.key,
            HookAction.RELEASE,
            "completed",
            tracked.context,
            tier_before=resident_tier,
            tier_after=resident_tier,
            metadata={
                "runtime_reqmeta_id": tracked.runtime_reqmeta_id,
                "release_condition": source,
                "access_completion": "request_lifecycle_finished",
                "observational_only": True,
            },
        )
        if observation.binding is not None:
            execution_spec = self._endpoint.register_binding(observation.binding, tracked.key, tracked.manager)
            if execution_spec is not None and observation.binding_record is not None:
                observation.binding_record["execution_spec"] = execution_spec.to_record()
        for record in observation.records:
            self._event_sink(record)


def install_lmcache047_hooks(
    event_sink: EventSink,
    *,
    factory_cls: type[Any] | None = None,
    manager_cls: type[Any] | None = None,
    connector_cls: type[Any] | None = None,
    versions: dict[str, str] | None = None,
    binding_registry: BackendBindingRegistry | None = None,
    request_context_provider: Callable[[], RequestContext | None] | None = None,
    request_context_consumer: LMCache047RequestContextConsumer | None = None,
    runtime_request_identity_provider: Callable[[str], RuntimeRequestIdentity | None] | None = None,
) -> LMCache047ActionEndpoint:
    observed_versions = versions or installed_versions()
    if observed_versions != SUPPORTED_VERSIONS:
        raise RuntimeError("unsupported runtime versions: " + repr(observed_versions))
    # Unit fixtures intentionally do not install LMCache.  In a real target
    # environment this import exists and the version preflight has already
    # established that the instrumentation is operating on LMCache 0.4.7.
    try:
        patch_usage_context_cpu_info()
    except ModuleNotFoundError as exc:
        if exc.name != "lmcache":
            raise
    if factory_cls is None:
        from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
        factory_cls = VllmServiceFactory
    if manager_cls is None:
        try:
            from lmcache.v1.manager import LMCacheManager
            manager_cls = LMCacheManager
        except ModuleNotFoundError as exc:
            if exc.name != "lmcache":
                raise
    endpoint = LMCache047ActionEndpoint(binding_registry=binding_registry, action_registration_enabled=False)
    lifecycle = _ConnectorLifecycle(
        event_sink, binding_registry, request_context_consumer, runtime_request_identity_provider, endpoint,
    )

    def resolved_context() -> RequestContext | None:
        return lifecycle.active_context() or (request_context_provider() if request_context_provider is not None else None)

    original = factory_cls.get_or_create_lmcache_engine
    if getattr(original, "__astrakv_lmcache047_patch__", False):
        raise RuntimeError("LMCache 0.4.7 hooks are already installed")

    def factory_wrapper(factory: Any, *args: Any, **kwargs: Any) -> Any:
        engine = original(factory, *args, **kwargs)
        if engine is not None and getattr(engine, "storage_manager", None) is not None:
            _patch_storage_manager(
                engine.storage_manager, endpoint, event_sink, binding_registry, resolved_context,
                lifecycle.track_submission, lifecycle.complete_store,
                action_manager=_action_capable_manager(engine.storage_manager, lmcache_engine=engine),
            )
        return engine

    factory_wrapper.__astrakv_lmcache047_patch__ = True
    factory_cls.get_or_create_lmcache_engine = factory_wrapper
    if manager_cls is not None:
        _patch_manager_post_init(
            manager_cls, endpoint, event_sink, binding_registry, resolved_context,
            lifecycle.track_submission, lifecycle.complete_store,
        )
    if connector_cls is None:
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
            connector_cls = LMCacheConnectorV1Impl
        except ModuleNotFoundError as exc:
            if exc.name != "lmcache":
                raise
    if connector_cls is not None:
        _patch_connector_lifecycle(connector_cls, lifecycle)
    return endpoint


def _patch_manager_post_init(
    manager_cls: type[Any], endpoint: LMCache047ActionEndpoint, event_sink: EventSink,
    binding_registry: BackendBindingRegistry | None, request_context_provider: Callable[[], RequestContext | None] | None,
    binding_observer: Callable[[Any, Any, RequestContext | None, Any], None] | None = None,
    completion_observer: Callable[[Any, str], None] | None = None,
) -> None:
    original = manager_cls.post_init
    if getattr(original, "__astrakv_lmcache047_patch__", False):
        return
    def post_init_wrapper(manager: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(manager, *args, **kwargs)
        engine = getattr(manager, "lmcache_engine", None) or getattr(manager, "_lmcache_engine", None)
        if engine is not None and getattr(engine, "storage_manager", None) is not None:
            _patch_storage_manager(
                engine.storage_manager, endpoint, event_sink, binding_registry, request_context_provider,
                binding_observer, completion_observer, action_manager=manager,
            )
        return result
    post_init_wrapper.__astrakv_lmcache047_patch__ = True
    manager_cls.post_init = post_init_wrapper


def _patch_storage_manager(
    manager: Any, endpoint: LMCache047ActionEndpoint, event_sink: EventSink,
    binding_registry: BackendBindingRegistry | None = None,
    request_context_provider: Callable[[], RequestContext | None] | None = None,
    binding_observer: Callable[[Any, Any, RequestContext | None, Any], None] | None = None,
    completion_observer: Callable[[Any, str], None] | None = None,
    action_manager: Any = None,
) -> None:
    if getattr(manager, "__astrakv_lmcache047_patch__", False):
        return
    action_manager = _action_capable_manager(
        manager,
        action_manager=action_manager,
        lmcache_engine=getattr(action_manager, "lmcache_engine", None) if action_manager is not None else None,
    )
    original_put, original_get, original_remove = manager.batched_put, manager.get, manager.remove
    original_batched_get = getattr(manager, "batched_get", None)
    pending_store_callbacks: dict[str, list[tuple[Any, RequestContext, str]]] = {}

    def emit(key: Any, action: HookAction, status: str, **metadata: Any) -> Any:
        if binding_registry is None:
            legacy_action = "cache_store_submitted" if action == HookAction.CACHE_STORE and status == "submitted" else action.value
            event_sink({"backend_object_id": endpoint.remember(key, manager), "action": legacy_action, "status": status, **metadata})
            return None
        context = request_context_provider() if request_context_provider is not None else None
        if context is not None and action in {HookAction.CACHE_HIT, HookAction.CACHE_MISS, HookAction.CACHE_LOAD}:
            metadata.setdefault("allow_reserved_io", True)
        observation = binding_registry.observe(key, action, status, context, metadata=metadata)
        if binding_observer is not None:
            binding_observer(key, observation, context, action_manager)
        if observation.binding is not None:
            endpoint.register_binding(observation.binding, key, action_manager)
        for record in observation.records:
            event_sink(record)
        return observation

    def emit_callback_failure(key: Any, detail: str) -> None:
        if binding_registry is None:
            event_sink({
                "backend_object_id": endpoint.remember(key, manager), "action": "cache_store",
                "status": "failed", "observational_only": True, "callback_detail": detail,
            })
            return
        observation = binding_registry.observe(
            key, HookAction.CACHE_STORE, "failed", None,
            metadata={
                "terminal_condition": "lmcache_local_disk_completion_callback",
                "callback_detail": detail,
                "observational_only": True,
            },
        )
        for record in observation.records:
            event_sink(record)

    def complete_local_disk_store(callback_key: Any) -> None:
        tracked = pending_store_callbacks.get(_canonical_key(callback_key))
        if not tracked:
            emit_callback_failure(callback_key, "no_matching_submitted_operation_lease")
            return
        key, context, operation_lease = tracked.pop(0)
        if not tracked:
            del pending_store_callbacks[_canonical_key(callback_key)]
        if binding_registry is None:
            emit_callback_failure(callback_key, "registry_not_configured")
            return
        try:
            tier_after = _observed_resident_tier(action_manager, key)
            observation = binding_registry.complete_operation(
                key, HookAction.CACHE_STORE, "completed", context, operation_lease,
                tier_before="unknown",
                tier_after=tier_after,
                metadata={
                    "terminal_condition": "lmcache_local_disk_write_and_index_complete",
                    "storage_backend": "LocalDiskBackend",
                },
            )
        except (TypeError, ValueError) as exc:
            emit_callback_failure(callback_key, f"lease_completion_rejected:{type(exc).__name__}")
            return
        endpoint.mark_store_completed(observation.binding, key, action_manager)
        for record in observation.records:
            event_sink(record)
        if completion_observer is not None:
            completion_observer(key, operation_lease)

    def patch_local_disk_completion_backends() -> bool:
        backends = getattr(manager, "storage_backends", {})
        if not isinstance(backends, dict):
            return False
        patched = False
        for backend_name, backend in backends.items():
            if backend_name != "LocalDiskBackend" or not _supports_local_disk_completion_callback(backend):
                continue
            original_submit = backend.batched_submit_put_task
            if getattr(original_submit, "__astrakv_lmcache047_completion_patch__", False):
                patched = True
                continue

            def submit_wrapper(
                backend_self: Any, keys: Any, memory_objs: Any, transfer_spec: Any = None,
                on_complete_callback: Callable[[Any], None] | None = None,
                *, _original: Callable[..., Any] = original_submit,
            ) -> Any:
                def complete(key: Any) -> None:
                    complete_local_disk_store(key)
                    if on_complete_callback is not None:
                        on_complete_callback(key)

                return _original(keys, memory_objs, transfer_spec=transfer_spec, on_complete_callback=complete)

            submit_wrapper.__astrakv_lmcache047_completion_patch__ = True
            backend.batched_submit_put_task = types.MethodType(submit_wrapper, backend)
            patched = True
        return patched

    endpoint._require_verified_local_disk_completion = patch_local_disk_completion_backends()

    def put_wrapper(self: Any, keys: Any, memory_objs: Any, *args: Any, **kwargs: Any) -> Any:
        if binding_registry is not None and (request_context_provider is None or request_context_provider() is None):
            if any(binding_registry.is_key_reserved(key) for key in keys):
                for key in keys:
                    emit(key, HookAction.CACHE_STORE, "deferred")
                return None
        for key in keys:
            endpoint.remember(key, self)
            observation = emit(
                key,
                HookAction.CACHE_STORE,
                "submitted",
                **_event_metadata_from_storage_key(key, action="store"),
            )
            if (
                endpoint._require_verified_local_disk_completion
                and observation is not None
                and observation.event is not None
                and request_context_provider is not None
            ):
                context = request_context_provider()
                operation_lease = str(observation.event.metadata.get("operation_lease") or "")
                if context is not None and operation_lease:
                    pending_store_callbacks.setdefault(_canonical_key(key), []).append((key, context, operation_lease))
        result = original_put(keys, memory_objs, *args, **kwargs)
        return result

    def get_wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        if binding_registry is not None and (request_context_provider is None or request_context_provider() is None) and binding_registry.is_key_reserved(key):
            emit(key, HookAction.CACHE_HIT, "deferred")
            return None
        result = original_get(key, *args, **kwargs)
        endpoint.remember(key, self)
        emit(
            key,
            HookAction.CACHE_HIT if result is not None else HookAction.CACHE_MISS,
            "completed",
            **(_event_metadata_from_storage_key(key, action="hit") if result is not None else {}),
        )
        return result

    def batched_get_wrapper(self: Any, keys: Any, *args: Any, **kwargs: Any) -> Any:
        assert original_batched_get is not None
        if (
            binding_registry is not None
            and not _RUNTIME_OWNER_ACTION_ACTIVE.get()
            and (request_context_provider is None or request_context_provider() is None)
        ):
            if any(binding_registry.is_key_reserved(key) for key in keys):
                for key in keys:
                    emit(key, HookAction.CACHE_HIT, "deferred")
                return [None] * len(keys)
        results = original_batched_get(keys, *args, **kwargs)
        for key, result in zip(keys, results):
            endpoint.remember(key, self)
            emit(
                key,
                HookAction.CACHE_HIT if result is not None else HookAction.CACHE_MISS,
                "completed",
                **(_event_metadata_from_storage_key(key, action="hit") if result is not None else {}),
            )
        return results

    def batched_put_wrapper(self: Any, keys: Any, memory_objs: Any, *args: Any, **kwargs: Any) -> Any:
        if _RUNTIME_OWNER_ACTION_ACTIVE.get():
            return original_put(keys, memory_objs, *args, **kwargs)
        if binding_registry is not None and (request_context_provider is None or request_context_provider() is None):
            if any(binding_registry.is_key_reserved(key) for key in keys):
                for key in keys:
                    emit(key, HookAction.CACHE_STORE, "deferred")
                return None
        for key in keys:
            endpoint.remember(key, self)
            observation = emit(
                key,
                HookAction.CACHE_STORE,
                "submitted",
                **_event_metadata_from_storage_key(key, action="store"),
            )
            if (
                endpoint._require_verified_local_disk_completion
                and observation is not None
                and observation.event is not None
                and request_context_provider is not None
            ):
                context = request_context_provider()
                operation_lease = str(observation.event.metadata.get("operation_lease") or "")
                if context is not None and operation_lease:
                    pending_store_callbacks.setdefault(_canonical_key(key), []).append((key, context, operation_lease))
        return original_put(keys, memory_objs, *args, **kwargs)

    def remove_wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_remove(key, *args, **kwargs)
        endpoint.remember(key, self)
        emit(key, HookAction.DROP, "completed" if result else "not_found", removed=int(result))
        return result

    manager.batched_put = types.MethodType(batched_put_wrapper, manager)
    manager.get = types.MethodType(get_wrapper, manager)
    if original_batched_get is not None:
        manager.batched_get = types.MethodType(batched_get_wrapper, manager)
    manager.remove = types.MethodType(remove_wrapper, manager)
    manager.__astrakv_lmcache047_patch__ = True


def _patch_connector_lifecycle(connector_cls: type[Any], lifecycle: _ConnectorLifecycle) -> None:
    """Patch the 0.4.7 connector methods whose source defines request lifetime.

    ``StorageManager.batched_put`` is non-blocking in the supported source, so
    neither ``wait_for_save`` nor ``get_finished`` can be promoted to a durable
    store completion.  A returned ``wait_for_save`` is only used to record the
    observed ``lookup_unpin`` release lifecycle.
    """
    original_save = connector_cls.save_kv_layer
    if getattr(original_save, "__astrakv_lmcache047_connector_patch__", False):
        return
    original_wait = connector_cls.wait_for_save
    original_finished = connector_cls.request_finished
    original_get_finished = connector_cls.get_finished
    original_start_load = getattr(connector_cls, "start_load_kv", None)
    original_wait_load = getattr(connector_cls, "wait_for_layer_load", None)
    original_get_matched = getattr(connector_cls, "get_num_new_matched_tokens", None)
    original_update_state = getattr(connector_cls, "update_state_after_alloc", None)

    def patch_engine(connector: Any) -> None:
        engine = getattr(connector, "lmcache_engine", None)
        if engine is None:
            return
        original_store = getattr(engine, "store", None)
        original_retrieve = getattr(engine, "retrieve", None)

        if callable(original_store) and not getattr(original_store, "__astrakv_lmcache047_store_patch__", False):
            def store_wrapper(*args: Any, **kwargs: Any) -> Any:
                reqmeta_id = str(kwargs.get("req_id") or "")
                if not reqmeta_id:
                    return original_store(*args, **kwargs)
                tokens = lifecycle.enter_store(reqmeta_id)
                try:
                    return original_store(*args, **kwargs)
                except Exception:
                    lifecycle.fail_store(reqmeta_id)
                    raise
                finally:
                    lifecycle.exit_store(tokens)

            store_wrapper.__astrakv_lmcache047_store_patch__ = True
            engine.store = store_wrapper
        if callable(original_retrieve) and not getattr(original_retrieve, "__astrakv_lmcache047_retrieve_patch__", False):
            def retrieve_wrapper(*args: Any, **kwargs: Any) -> Any:
                reqmeta_id = str(kwargs.get("req_id") or "")
                if not reqmeta_id:
                    return original_retrieve(*args, **kwargs)
                tokens = lifecycle.enter_runtime_request(reqmeta_id)
                try:
                    return original_retrieve(*args, **kwargs)
                finally:
                    lifecycle.exit_runtime_request(tokens)

            retrieve_wrapper.__astrakv_lmcache047_retrieve_patch__ = True
            engine.retrieve = retrieve_wrapper

    def request_ids(connector: Any) -> tuple[str, ...]:
        parent = getattr(connector, "_parent", None)
        getter = getattr(parent, "_get_connector_metadata", None)
        if not callable(getter):
            return ()
        metadata = getter()
        return tuple(str(getattr(request, "req_id", "") or "") for request in getattr(metadata, "requests", ()) if getattr(request, "req_id", None))

    def note_dynamic_load(
        connector: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        vllm_cached_tokens: Any = None,
        lmcache_cached_tokens: Any = None,
        can_load: Any = None,
    ) -> None:
        reqmeta_id = _extract_runtime_reqmeta_id(connector, args, kwargs)
        if not reqmeta_id:
            request_id_candidates = request_ids(connector)
            reqmeta_id = request_id_candidates[0] if request_id_candidates else ""
        if not reqmeta_id:
            return
        token_ids = _extract_token_ids(connector, args, kwargs)
        lifecycle.note_load_target(
            reqmeta_id,
            token_ids=token_ids,
            slot_mapping=_extract_slot_mapping(connector, args, kwargs),
            vllm_cached_tokens=(
                vllm_cached_tokens
                if vllm_cached_tokens is not None
                else _extract_numeric_hint(connector, args, kwargs, "vllm_cached_tokens", "num_computed_tokens")
            ),
            lmcache_cached_tokens=(
                lmcache_cached_tokens
                if lmcache_cached_tokens is not None
                else _extract_numeric_hint(connector, args, kwargs, "lmcache_cached_tokens", "num_matched_tokens", "num_new_matched_tokens")
            ),
            chunk_size=_extract_numeric_hint(connector, args, kwargs, "chunk_size"),
            can_load=can_load,
            request_configs=_extract_request_configs(connector, args, kwargs),
            kvcaches=_extract_kvcaches(connector, args, kwargs),
            estimated_bytes=_extract_numeric_hint(connector, args, kwargs, "estimated_bytes"),
            target_tier="gpu",
        )

    def save_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
        patch_engine(connector)
        return original_save(connector, *args, **kwargs)

    def wait_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_wait(connector, *args, **kwargs)
        for reqmeta_id in request_ids(connector):
            lifecycle.release_after_unpin(
                reqmeta_id, source="lmcache_connector_wait_for_save_returned_after_lookup_unpin",
            )
        return result

    def request_finished_wrapper(connector: Any, request: Any, block_ids: Any) -> Any:
        reqmeta_id = str(getattr(request, "request_id", "") or getattr(request, "req_id", ""))
        if getattr(connector, "lmcache_engine", None) is None:
            if reqmeta_id:
                lifecycle.release_after_unpin(
                    reqmeta_id, source="lmcache_connector_request_finished_teardown_safe"
                )
            return False, None
        try:
            result = original_finished(connector, request, block_ids)
        except Exception:
            if getattr(connector, "lmcache_engine", None) is None:
                if reqmeta_id:
                    lifecycle.release_after_unpin(
                        reqmeta_id, source="lmcache_connector_request_finished_teardown_safe"
                    )
                return False, None
            raise
        if reqmeta_id:
            lifecycle.release_after_unpin(reqmeta_id, source="lmcache_connector_request_finished_returned")
        return result

    def get_finished_wrapper(connector: Any, finished_req_ids: Any) -> Any:
        # The installed 0.4.7 implementation returns (None, None), so this is
        # deliberately observation-only and cannot establish store completion.
        return original_get_finished(connector, finished_req_ids)

    if callable(original_start_load):
        def start_load_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
            patch_engine(connector)
            note_dynamic_load(connector, args, kwargs)
            return original_start_load(connector, *args, **kwargs)

        connector_cls.start_load_kv = start_load_wrapper
    if callable(original_wait_load):
        def wait_load_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_wait_load(connector, *args, **kwargs)
            note_dynamic_load(connector, args, kwargs, can_load=True)
            return result

        connector_cls.wait_for_layer_load = wait_load_wrapper
    if callable(original_get_matched):
        def get_matched_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_get_matched(connector, *args, **kwargs)
            note_dynamic_load(connector, args, kwargs, lmcache_cached_tokens=result)
            return result

        connector_cls.get_num_new_matched_tokens = get_matched_wrapper
    if callable(original_update_state):
        def update_state_wrapper(connector: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_update_state(connector, *args, **kwargs)
            note_dynamic_load(connector, args, kwargs)
            return result

        connector_cls.update_state_after_alloc = update_state_wrapper

    save_wrapper.__astrakv_lmcache047_connector_patch__ = True
    connector_cls.save_kv_layer = save_wrapper
    connector_cls.wait_for_save = wait_wrapper
    connector_cls.request_finished = request_finished_wrapper
    connector_cls.get_finished = get_finished_wrapper
