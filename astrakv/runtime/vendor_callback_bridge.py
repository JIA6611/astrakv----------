"""Version-locked control bridge for the native LMCache connector.

The bridge may bound scheduler-visible external tokens and promote immutable
LMCache chunks from LocalDiskBackend to LocalCPUBackend.  It never receives a
slot mapping, calls ``engine.retrieve``, or writes vLLM paged KV.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey,
    NativeKVLoadReceipt,
    NativeKVObjectRegistry,
    PhysicalKVObject,
    PrefetchStatus,
    PrefetchTicket,
    RequestKVIntent,
    RuntimeMode,
    TierCapabilitySnapshot,
    TierTopology,
    choose_load_vs_recompute,
    exact_token_prefix_hash,
)
from astrakv.runtime.lmcache047_bootstrap import (
    install_from_environment,
    installed_kv_core_callbacks,
    installed_runtime_control_host,
)
from astrakv.runtime.offline_kv_profile import OfflineKVProfileIndex, PrefixRuntimeHint
from astrakv.runtime.third_party_patch import PATCH_ID, REQUIRED_CALLBACKS


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "true" if default else "false").lower() == "true"


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _owns_runtime_control_host(connector: Any) -> bool:
    """Return whether this connector process may bind the control-plane port.

    vLLM 0.23 creates the LMCache connector as ``WORKER`` even for a
    single-process ``kv_both`` deployment.  That process invokes both the
    scheduler and worker callback paths, so it must own the authenticated
    context ingress.  A distributed worker never receives this exception:
    it must not race another process for the control-plane port.
    """
    role = getattr(connector, "_role", None)
    role_name = str(getattr(role, "name", role) or "").lower()
    if "scheduler" in role_name:
        return True
    kv_role = str(getattr(connector, "kv_role", "") or "").lower()
    worker_count = _env_int("ASTRAKV_TENSOR_PARALLEL_SIZE", 0)
    if worker_count <= 0:
        worker_count = int(getattr(connector, "worker_count", 0) or 0)
    return "worker" in role_name and kv_role == "kv_both" and worker_count == 1


class VendorCallbackBridge:
    """Control and record only version-locked native connector lifecycles."""

    def __init__(self, connector: Any) -> None:
        self._connector = connector
        self._lock = threading.RLock()
        self._registry = NativeKVObjectRegistry()
        self._physical_by_request: dict[str, PhysicalKVObject] = {}
        self._keys_by_request: dict[str, tuple[Any, ...]] = {}
        self._prefetch_by_request: dict[str, str] = {}
        self._ingress_started_ns: dict[str, int] = {}
        self._lookup_started_ns: dict[str, int] = {}
        self._seen_callbacks: set[str] = set()
        self._prefetch_request_seen: set[str] = set()
        self._terminal_request_seen: set[str] = set()
        self._prefetch_keys: dict[str, tuple[Any, ...]] = {}
        self._prefetch_futures: dict[str, Any] = {}
        self._prefetch_watcher: threading.Thread | None = None
        self._bootstrap_loads = 0
        self._ssd_bytes_per_ms_ema = 0.0
        self._prefill_ms_per_token_ema = _env_float("ASTRAKV_KV_CORE_PREFILL_MS_PER_TOKEN", 0.0)
        state = os.environ.get("ASTRAKV_RUNTIME_CONTROL_STATE_DIR", "")
        self._state_dir = Path(state) if state else None
        self._profile_index: OfflineKVProfileIndex | None = None
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            (self._state_dir / "PATCH_MARKER").write_text(PATCH_ID + "\n", encoding="utf-8")
        profile_path = os.environ.get("ASTRAKV_KV_CORE_OFFLINE_PROFILE", "")
        if profile_path:
            try:
                self._profile_index = OfflineKVProfileIndex.load(profile_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError("invalid compatibility-aware KV-Core offline profile") from exc
        self._start_prefetch_watcher_if_worker()

    @classmethod
    def from_environment(cls, connector: Any = None) -> "VendorCallbackBridge | None":
        if os.environ.get("ASTRAKV_KV_CORE_VENDOR_PATCH", "false") != "true":
            return None
        # ``connector is None`` is the legacy/bootstrap call made before the
        # vendor connector exists.  For vLLM 0.23 single-worker kv_both, the
        # EngineCore exposes scheduler callbacks through a WORKER connector;
        # _owns_runtime_control_host handles that version-specific topology.
        scheduler_role = connector is None or _owns_runtime_control_host(connector)
        install_from_environment(
            vendor_engine_child=True,
            start_runtime_host=scheduler_role,
        )
        if installed_kv_core_callbacks() is None or connector is None:
            return None
        bridge = cls(connector)
        host = installed_runtime_control_host()
        if host is not None and scheduler_role:
            host.register_kv_runtime_bridge(bridge)
        return bridge

    def ingress_request(self, context: Any) -> None:
        """Accept an authenticated exact-token intent before HTTP submission."""
        request_id = str(getattr(context, "request_id", "") or "")
        metadata = dict(getattr(context, "metadata", {}) or {})
        try:
            tokens = tuple(int(token) for token in metadata.get("exact_token_ids", ()))
        except (TypeError, ValueError):
            tokens = ()
        if not request_id:
            return
        self._ingress_started_ns[request_id] = time.time_ns()
        if not tokens:
            return
        if self._callbacks() is None or self._callbacks().mode is not RuntimeMode.ACTIVE:
            return
        request_configs = metadata.get("request_configs")
        normalized_configs = request_configs if isinstance(request_configs, dict) else None
        if self._storage_manager() is None:
            # The EngineCore scheduler owns authenticated ingress, while the
            # worker owns LocalDiskBackend/LocalCPUBackend.  Publish the exact
            # request intent so the worker can overlap promotion with queueing.
            self._write_prefetch_request(
                request_id,
                tokens,
                normalized_configs,
                promote=_env_flag("ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED"),
            )
        else:
            self._publish_tier_observation(request_id, tokens, normalized_configs)
            if _env_flag("ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED"):
                self._schedule_cpu_promotion(
                    request_id=request_id,
                    token_ids=tokens,
                    request_configs=normalized_configs,
                )

    def scheduler_exact_lookup(
        self,
        connector: Any,
        *,
        request_id: str,
        token_ids: Iterable[int],
        request_configs: dict[str, Any] | None,
        lookup_hit_tokens: int,
        num_computed_tokens: int = 0,
        priority: int = 0,
    ) -> int | None:
        """Record exact lookup and return an active-mode external-token cap."""
        tokens = tuple(int(token) for token in token_ids)
        requested = len(tokens)
        hit = min(requested, max(0, int(lookup_hit_tokens)))
        local_tokens = min(requested, max(0, int(num_computed_tokens)))
        available_hit = max(0, hit - local_tokens)
        if hit == requested and available_hit > 0:
            available_hit -= 1
        candidate_tokens = tokens[:hit] if hit else tokens
        logical_request_id = self._logical_request_id(request_id)
        physical, native_keys = self._physical(connector, candidate_tokens, request_configs)
        if physical is not None:
            physical = self._apply_tier_observation(logical_request_id, physical)
        if physical is None:
            return 0 if self._active_admission() else None
        self._write_run_metadata(connector)
        callbacks = self._callbacks()
        if callbacks is None:
            return None
        try:
            with self._lock:
                existing = self._physical_by_request.get(request_id)
                if existing is None:
                    cap = self._external_token_cap(
                        physical=physical,
                        requested_tokens=requested,
                        available_external_tokens=available_hit,
                        priority=int(priority),
                        logical_request_id=logical_request_id,
                    )
                    callbacks.submit_intent(RequestKVIntent(
                        request_id=request_id,
                        compatibility_key=physical.compatibility_key,
                        physical_object=physical,
                        max_external_tokens=cap,
                        requested_prefix_tokens=requested,
                        deadline_ns=self._ingress_started_ns.get(
                            logical_request_id, time.time_ns(),
                        ) + _env_int(
                            "ASTRAKV_KV_CORE_LOAD_DEADLINE_NS", 60_000_000_000,
                        ),
                        priority=int(priority),
                        cancellation_token=logical_request_id,
                    ))
                    self._physical_by_request[request_id] = physical
                    self._keys_by_request[request_id] = native_keys
                else:
                    physical = existing
                callbacks.record_scheduler_lookup(
                    request_id=request_id,
                    physical=physical,
                    locally_cached_tokens=local_tokens,
                    lookup_hit_tokens=available_hit,
                    native_request_id=request_id,
                )
                self._consume_matching_prefetch(
                    logical_request_id, request_id, physical, native_keys, available_hit,
                )
                intent = callbacks.intent_for(request_id)
                if intent is None:
                    raise ValueError("scheduler lookup did not create a request intent")
                self._write_native_intent(
                    request_id=request_id,
                    physical=physical,
                    requested_prefix_tokens=requested,
                    locally_cached_tokens=local_tokens,
                    lookup_hit_tokens=available_hit,
                    max_external_tokens=intent.max_external_tokens,
                    prefetch_id=self._prefetch_by_request.get(request_id, ""),
                )
                self._lookup_started_ns[request_id] = time.time_ns()
                self._record("scheduler_exact_lookup", {
                    "request_id": request_id,
                    "logical_request_id": logical_request_id,
                    "physical_object_id": physical.physical_object_id,
                    "binding_generation": physical.binding_generation,
                    "native_key": physical.native_key,
                    "compatibility_identity": physical.compatibility_key.identity,
                    "lookup_hit_tokens": available_hit,
                    "native_lookup_prefix_tokens": hit,
                    "locally_cached_tokens": local_tokens,
                    "max_external_tokens": intent.max_external_tokens,
                    "source_tier": physical.source_tier,
                })
                self._append_request_accounting(request_id, physical)
                return intent.max_external_tokens if self._active_admission() else None
        except (TypeError, ValueError) as exc:
            self._record("scheduler_exact_lookup", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
            })
            return 0 if self._active_admission() else None

    def scheduler_external_admission(self, *, request_id: str, allocated_external_tokens: int) -> None:
        callbacks, physical = self._callbacks(), self._physical_by_request.get(request_id)
        if callbacks is None or physical is None:
            return
        try:
            admission = callbacks.record_scheduler_admission(
                request_id=request_id,
                physical=physical,
                allocated_external_tokens=max(0, int(allocated_external_tokens)),
            )
            self._update_native_intent(
                request_id,
                allocated_external_tokens=admission.allocated_external_tokens,
            )
            self._record("scheduler_external_admission", asdict(admission))
            self._append_request_accounting(request_id, physical)
        except (TypeError, ValueError) as exc:
            self._record("scheduler_external_admission", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
            })

    def scheduler_compute_progress(self, *, request_id: str, scheduled_tokens: int) -> None:
        callbacks = self._callbacks()
        if callbacks is None:
            return
        try:
            callbacks.record_scheduler_compute(
                request_id=request_id, scheduled_tokens=max(0, int(scheduled_tokens)),
            )
            self._record("scheduler_compute_progress", {
                "request_id": request_id, "scheduled_tokens": max(0, int(scheduled_tokens)),
            })
        except ValueError as exc:
            self._record("scheduler_compute_progress", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
            })

    def connector_metadata(self, *, request_id: str, metadata_present: bool, can_load: bool) -> None:
        self._record("connector_metadata", {
            "request_id": request_id,
            "metadata_present": bool(metadata_present),
            "can_load": bool(can_load),
        })

    def native_load_completion(
        self,
        *,
        request_id: str,
        connector: Any | None = None,
        token_ids: Iterable[int] = (),
        request_configs: dict[str, Any] | None = None,
        compatibility_prefix_tokens: int = 0,
        requested_prefix_tokens: int = 0,
        locally_cached_tokens: int = 0,
        lookup_hit_tokens: int = 0,
        allocated_external_tokens: int = 0,
        native_retrieved_tokens: int,
        native_bytes: int,
        load_latency_ns: int,
        status: str,
    ) -> bool:
        callback_tokens = tuple(int(token) for token in token_ids)
        physical = self._physical_by_request.get(request_id)
        compatibility_count = min(
            len(callback_tokens), max(0, int(compatibility_prefix_tokens)),
        )
        if physical is None and compatibility_count > 0:
            physical, _ = self._physical(
                connector or self._connector,
                callback_tokens[:compatibility_count],
                request_configs,
            )
        intent_record = self._read_native_intent(request_id)
        if physical is None and intent_record is None:
            return self._active_admission()
        try:
            native_tokens = max(0, int(native_retrieved_tokens))
            allocated = max(0, int(allocated_external_tokens))
            actual_tokens = min(native_tokens, allocated)
            reported_native_bytes = max(0, int(native_bytes))
            actual_bytes = (
                reported_native_bytes * actual_tokens // native_tokens
                if native_tokens > 0
                else 0
            )
            requested_tokens = int(
                (intent_record or {}).get("requested_prefix_tokens", 0)
            ) or max(0, int(requested_prefix_tokens)) or len(callback_tokens)
            local_tokens = min(
                requested_tokens,
                int((intent_record or {}).get("locally_cached_tokens", 0))
                or max(0, int(locally_cached_tokens)),
            )
            hit_tokens = max(
                allocated,
                int(lookup_hit_tokens),
                int((intent_record or {}).get("lookup_hit_tokens", 0)),
            )
            prefetch_id = (
                self._prefetch_by_request.get(request_id, "")
                or str((intent_record or {}).get("prefetch_id", ""))
            )
            physical_object_id = (
                physical.physical_object_id
                if physical is not None
                else str(intent_record["physical_object_id"])
            )
            binding_generation = (
                physical.binding_generation
                if physical is not None
                else int(intent_record["binding_generation"])
            )
            native_key = (
                physical.native_key if physical is not None else str(intent_record["native_key"])
            )
            compatibility_identity = (
                physical.compatibility_key.identity
                if physical is not None
                else str(intent_record["compatibility_identity"])
            )
            prefix_hash = (
                physical.compatibility_key.prefix_hash
                if physical is not None
                else str(intent_record["prefix_hash"])
            )
            receipt = NativeKVLoadReceipt(
                request_id=request_id,
                physical_object_id=physical_object_id,
                binding_generation=binding_generation,
                native_key=native_key,
                compatibility_identity=compatibility_identity,
                prefix_hash=prefix_hash,
                requested_prefix_tokens=requested_tokens,
                locally_cached_tokens=local_tokens,
                lookup_hit_tokens=hit_tokens,
                allocated_external_tokens=allocated,
                actual_loaded_tokens=actual_tokens,
                native_retrieved_tokens=native_tokens,
                missing_tokens=max(0, requested_tokens - local_tokens - actual_tokens),
                unallocated_recompute_tokens=max(
                    0, requested_tokens - local_tokens - allocated,
                ),
                load_shortfall_tokens=allocated - actual_tokens,
                bytes_loaded=actual_bytes,
                load_latency_ns=max(0, int(load_latency_ns)),
                native_request_id=request_id,
                status=status,
                prefetch_id=prefetch_id,
            )
            self._write_native_receipt(receipt)
            source_tier = (
                physical.source_tier
                if physical is not None
                else str((intent_record or {}).get("source_tier", "unknown"))
            )
            if receipt.load_latency_ns > 0 and receipt.bytes_loaded > 0 and source_tier == "ssd":
                observed = receipt.bytes_loaded / (receipt.load_latency_ns / 1_000_000.0)
                self._ssd_bytes_per_ms_ema = observed if self._ssd_bytes_per_ms_ema <= 0 else 0.8 * self._ssd_bytes_per_ms_ema + 0.2 * observed
            record = asdict(receipt)
            self._append("kv_core_native_receipts.jsonl", record)
            self._record("native_load_completion", record)
            return self._active_admission() and receipt.load_shortfall_tokens > 0
        except (TypeError, ValueError) as exc:
            self._record("native_load_completion", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
                "native_retrieved_tokens": max(0, int(native_retrieved_tokens)),
            })
            return self._active_admission()

    def native_load_start(
        self,
        *,
        request_id: str,
        connector: Any | None = None,
        token_ids: Iterable[int] = (),
        request_configs: dict[str, Any] | None = None,
        compatibility_prefix_tokens: int = 0,
        allocated_external_tokens: int = 0,
    ) -> None:
        """Observe the request-owned native load before LMCache retrieval.

        This callback never waits for prefetch and never writes GPU KV.  Its
        only active responsibility is to attribute a completed SSD->CPU
        promotion to the real request that is about to consume the same exact
        native keys.
        """
        tokens = tuple(int(token) for token in token_ids)
        prefix_tokens = min(len(tokens), max(0, int(compatibility_prefix_tokens)))
        intent_record = self._read_native_intent(request_id)
        physical = self._physical_by_request.get(request_id)
        native_keys: tuple[Any, ...] = ()
        if physical is None and prefix_tokens > 0:
            physical, native_keys = self._physical(
                connector or self._connector,
                tokens[:prefix_tokens],
                request_configs,
            )
        if physical is None:
            self._record("native_load_start", {
                "request_id": request_id,
                "status": "unbound",
                "allocated_external_tokens": max(0, int(allocated_external_tokens)),
            })
            return
        if intent_record is not None:
            expected = (
                str(intent_record.get("physical_object_id") or ""),
                int(intent_record.get("binding_generation") or 0),
                str(intent_record.get("native_key") or ""),
                str(intent_record.get("compatibility_identity") or ""),
            )
            observed = (
                physical.physical_object_id,
                physical.binding_generation,
                physical.native_key,
                physical.compatibility_key.identity,
            )
            if expected != observed:
                self._record("native_load_start", {
                    "request_id": request_id,
                    "status": "rejected",
                    "reason": "native_intent_identity_mismatch",
                })
                return
        logical_request_id = str(
            (intent_record or {}).get("logical_request_id") or request_id
        )
        self._consume_matching_prefetch(
            logical_request_id,
            request_id,
            physical,
            native_keys,
            max(0, int(allocated_external_tokens)),
        )
        self._record("native_load_start", {
            "request_id": request_id,
            "logical_request_id": logical_request_id,
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "native_key": physical.native_key,
            "compatibility_identity": physical.compatibility_key.identity,
            "allocated_external_tokens": max(0, int(allocated_external_tokens)),
            "prefetch_id": self._prefetch_by_request.get(request_id, ""),
        })

    def request_finished(
        self,
        *,
        request_id: str,
        finish_status: str,
        num_computed_tokens: int,
        num_tokens: int,
    ) -> None:
        callbacks, physical = self._callbacks(), self._physical_by_request.get(request_id)
        if callbacks is None or physical is None:
            return
        completed = "ABORT" not in finish_status.upper() and "ERROR" not in finish_status.upper()
        try:
            receipt = self._read_native_receipt(request_id)
            if receipt is not None:
                callbacks.import_native_load_receipt(receipt, physical=physical)
            accounting = callbacks.finalize_request(
                request_id=request_id,
                physical=physical,
                finish_status=finish_status,
                completed=completed,
                native_num_computed_tokens=max(0, int(num_computed_tokens)),
            )
            record = asdict(accounting)
            record.update({
                "native_num_computed_tokens": max(0, int(num_computed_tokens)),
                "native_num_tokens": max(0, int(num_tokens)),
                "native_key": physical.native_key,
                "compatibility_identity": physical.compatibility_key.identity,
                "timestamp_ns": time.time_ns(),
            })
            self._append("kv_core_request_accounting.jsonl", record)
            self._record("request_finished", record)
            self._write_request_terminal(
                logical_request_id=self._logical_request_id(request_id),
                finish_status=finish_status,
                completed=completed,
            )
        except (TypeError, ValueError) as exc:
            self._record("request_finished", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
            })

    def _active_admission(self) -> bool:
        callbacks = self._callbacks()
        return bool(
            callbacks is not None
            and callbacks.mode is RuntimeMode.ACTIVE
            and _env_flag("ASTRAKV_KV_CORE_ADMISSION_ENABLED")
        )

    def _external_token_cap(
        self,
        *,
        physical: PhysicalKVObject,
        requested_tokens: int,
        available_external_tokens: int,
        priority: int,
        logical_request_id: str,
    ) -> int:
        available = max(0, int(available_external_tokens))
        if not self._active_admission() or available == 0:
            return available
        chunk = physical.compatibility_key.chunk_size_tokens
        profile_hint = self._profile_hint(physical)
        configured_cap = _env_int("ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP", available)
        cap = min(available, configured_cap if configured_cap > 0 else available)
        if _env_flag("ASTRAKV_KV_CORE_PARTIAL_PREFIX_UPPER_BOUND_ENABLED"):
            partial_cap = _env_int("ASTRAKV_KV_CORE_PARTIAL_PREFIX_TOKENS", chunk)
            if profile_hint is not None:
                partial_cap = max(
                    chunk,
                    int(partial_cap * profile_hint.partial_load_fraction),
                )
            cap = min(cap, max(chunk, partial_cap))
        cap = cap // chunk * chunk
        if cap <= 0:
            return 0
        if physical.source_tier not in {"cpu", "ssd", "mixed"}:
            self._append("kv_core_policy_decisions.jsonl", {
                "request_id": logical_request_id,
                "physical_object_id": physical.physical_object_id,
                "binding_generation": physical.binding_generation,
                "action": "recompute",
                "reason": "native_tier_observation_missing",
                "candidate_external_tokens": cap,
                "timestamp_ns": time.time_ns(),
            })
            return 0
        capability = self._runtime_capability(external_token_cap=cap)
        now_ns = time.time_ns()
        ingress_ns = self._ingress_started_ns.get(logical_request_id, now_ns)
        intent = RequestKVIntent(
            request_id=logical_request_id or "policy-request",
            compatibility_key=physical.compatibility_key,
            physical_object=physical,
            max_external_tokens=cap,
            requested_prefix_tokens=requested_tokens,
            deadline_ns=ingress_ns + _env_int(
                "ASTRAKV_KV_CORE_LOAD_DEADLINE_NS", 60_000_000_000,
            ),
            priority=priority + (
                profile_hint.admission_priority_boost if profile_hint is not None else 0
            ),
        )
        queue_delay_ms = 0.0
        ingress = self._ingress_started_ns.get(logical_request_id)
        if ingress is not None:
            queue_delay_ms = max(0.0, (time.time_ns() - ingress) / 1_000_000.0)
        bandwidth = self._ssd_bytes_per_ms_ema
        if bandwidth <= 0:
            seed_gbps = _env_float("ASTRAKV_KV_CORE_SSD_READ_GBPS", 0.0)
            bandwidth = seed_gbps * 1_000_000.0 if seed_gbps > 0 else 0.0
        prefill = self._prefill_ms_per_token_ema
        bootstrap_limit = max(0, _env_int("ASTRAKV_KV_CORE_BOOTSTRAP_LOADS", 1))
        if bandwidth <= 0 or prefill <= 0:
            if self._bootstrap_loads < bootstrap_limit:
                self._bootstrap_loads += 1
                return cap
            return 0
        candidate_bytes = (
            physical.size_bytes * cap // max(1, available)
            if physical.source_tier in {"ssd", "mixed"}
            else 0
        )
        read_ms = candidate_bytes / bandwidth if bandwidth > 0 else 0.0
        decision = choose_load_vs_recompute(
            intent=intent,
            capability=capability,
            queue_delay_ms=queue_delay_ms,
            tier_read_ms=read_ms,
            transfer_ms=_env_float("ASTRAKV_KV_CORE_TRANSFER_MS", 0.0),
            materialization_ms=_env_float("ASTRAKV_KV_CORE_MATERIALIZATION_MS", 0.0),
            contention_ms=_env_float("ASTRAKV_KV_CORE_CONTENTION_MS", 0.0),
            prefill_ms_per_token=prefill,
        )
        self._append("kv_core_policy_decisions.jsonl", {
            "request_id": logical_request_id,
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "action": decision.action,
            "load_cost_ms": decision.load_cost_ms,
            "recompute_cost_ms": decision.recompute_cost_ms,
            "reason": decision.reason,
            "candidate_external_tokens": cap,
            "candidate_ssd_read_bytes": candidate_bytes,
            "queue_delay_ms": queue_delay_ms,
            "observed_prefill_ms_per_token": prefill,
            "observed_ssd_bytes_per_ms": bandwidth,
            "profile_matched": profile_hint is not None,
            "profile_sensitivity_rank": (
                profile_hint.sensitivity_rank if profile_hint is not None else None
            ),
            "profile_admission_priority_boost": (
                profile_hint.admission_priority_boost if profile_hint is not None else 0
            ),
            "timestamp_ns": time.time_ns(),
        })
        return cap if decision.action == "admit_external_prefix" else 0

    def _schedule_cpu_promotion(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
    ) -> None:
        manager = self._storage_manager()
        if manager is None:
            return
        backends = getattr(manager, "storage_backends", {})
        cpu, disk = backends.get("LocalCPUBackend"), backends.get("LocalDiskBackend")
        if cpu is None or disk is None or not bool(getattr(cpu, "use_hot", False)):
            return
        chunks = self._token_chunks(self._connector, token_ids, request_configs)
        disk_chunks: list[tuple[int, int, Any]] = []
        for start, end, key in chunks:
            if bool(cpu.contains(key, False)):
                continue
            if not bool(disk.contains(key, False)):
                break
            disk_chunks.append((start, end, key))
        if not disk_chunks:
            return
        max_tokens = _env_int("ASTRAKV_KV_CORE_PREFETCH_MAX_TOKENS", 0)
        if max_tokens > 0:
            disk_chunks = [row for row in disk_chunks if row[1] <= max_tokens]
        if not disk_chunks:
            return
        prefix_end = disk_chunks[-1][1]
        prefix_tokens = token_ids[:prefix_end]
        physical, native_keys = self._physical(self._connector, prefix_tokens, request_configs)
        if physical is None or not self._registry.is_current(*physical.generation_key):
            return
        selected_keys = tuple(row[2] for row in disk_chunks)
        requested_bytes = self._disk_bytes(disk, selected_keys)
        if requested_bytes <= 0:
            return
        callbacks = self._callbacks()
        if callbacks is None:
            return
        capability = self._runtime_capability(external_token_cap=0)
        callbacks.update_capability(capability)
        now = time.time_ns()
        ticket = PrefetchTicket(
            prefetch_id=f"prefetch:{request_id}:{physical.binding_generation}:{now}",
            physical_object_id=physical.physical_object_id,
            binding_generation=physical.binding_generation,
            prefix_hash=physical.compatibility_key.prefix_hash,
            source_tier="ssd",
            target_tier="cpu",
            requested_bytes=requested_bytes,
            deadline_ns=now + _env_int("ASTRAKV_KV_CORE_PREFETCH_DEADLINE_NS", 5_000_000_000),
            expires_at_ns=now + _env_int("ASTRAKV_KV_CORE_PREFETCH_TTL_NS", 30_000_000_000),
            target_request_id=request_id,
            native_key=physical.native_key,
            compatibility_identity=physical.compatibility_key.identity,
        )
        reason = callbacks.begin_cpu_prefetch(ticket, physical, now_ns=now)
        profile_hint = self._profile_hint(physical)
        if reason is None and profile_hint is not None and capability.cpu_prefetch_budget_bytes > 0:
            budget_utilization = (
                capability.cpu_prefetch_used_budget_bytes
                / capability.cpu_prefetch_budget_bytes
            )
            if budget_utilization > profile_hint.prefetch_priority:
                reason = "profile_prefetch_priority"
        if reason is not None:
            self._append_ticket(replace(ticket, status=PrefetchStatus.CANCELLED, failure_reason=reason))
            return
        self._append_ticket(ticket)
        self._prefetch_keys[ticket.prefetch_id] = selected_keys

        async def promote() -> int:
            memory_objs = await disk.batched_get_non_blocking(ticket.prefetch_id, list(selected_keys))
            completed = 0
            try:
                if len(memory_objs) != len(selected_keys):
                    return 0
                cpu.batched_submit_put_task(list(selected_keys), list(memory_objs))
                completed = requested_bytes if all(cpu.contains(key, False) for key in selected_keys) else 0
                return completed
            finally:
                # LocalDiskBackend 0.4.7 unpins disk metadata after each
                # async read.  Only release the staging MemoryObj ownership
                # acquired by batched_get_non_blocking here.
                if len(memory_objs) != len(selected_keys):
                    # The backend returns early before scheduling its disk
                    # worker when staging allocation is partial; those keys
                    # remain pinned and must be released here.
                    for key in selected_keys[: len(memory_objs)]:
                        try:
                            disk.unpin(key)
                        except Exception:
                            pass
                for _key, memory_obj in zip(selected_keys, memory_objs, strict=False):
                    try:
                        if getattr(memory_obj, "is_pinned", False):
                            memory_obj.unpin()
                        memory_obj.ref_count_down()
                    except Exception:
                        pass

        future = asyncio.run_coroutine_threadsafe(promote(), manager.loop)
        self._prefetch_futures[ticket.prefetch_id] = future

        def completed(done: Any) -> None:
            try:
                completed_bytes = int(done.result())
                updated = callbacks.complete_cpu_prefetch(
                    ticket.prefetch_id, completed_bytes=completed_bytes,
                )
            except Exception as exc:
                try:
                    updated = callbacks.tickets.cancel(
                        ticket.prefetch_id, reason=f"promotion_failed:{type(exc).__name__}",
                    )
                except ValueError:
                    current = callbacks.tickets.get(ticket.prefetch_id)
                    if current is not None and current.status in {
                        PrefetchStatus.CANCELLED, PrefetchStatus.WASTED,
                        PrefetchStatus.EXPIRED,
                    }:
                        self._demote_prefetch_keys(ticket.prefetch_id)
                    return
            finally:
                self._prefetch_futures.pop(ticket.prefetch_id, None)
            self._append_ticket(updated)

        future.add_done_callback(completed)

    def _profile_hint(self, physical: PhysicalKVObject) -> PrefixRuntimeHint | None:
        if self._profile_index is None:
            return None
        return self._profile_index.hint_for(physical.compatibility_key)

    def _start_prefetch_watcher_if_worker(self) -> None:
        if (
            self._state_dir is None
            or self._storage_manager() is None
        ):
            return
        callbacks = self._callbacks()
        if callbacks is None or callbacks.mode is not RuntimeMode.ACTIVE:
            return

        def watch() -> None:
            directory = self._state_dir / "prefetch_requests"
            while True:
                try:
                    paths = tuple(directory.glob("*.json")) if directory.is_dir() else ()
                    for path in paths:
                        key = path.name
                        if key in self._prefetch_request_seen:
                            continue
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            request_id = str(payload["request_id"])
                            token_ids = tuple(int(token) for token in payload["exact_token_ids"])
                            expires_at_ns = int(payload["expires_at_ns"])
                            request_configs = payload.get("request_configs")
                            promote = payload.get("promote") is True
                            if not isinstance(request_configs, dict):
                                request_configs = None
                        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                            self._prefetch_request_seen.add(key)
                            continue
                        self._prefetch_request_seen.add(key)
                        if expires_at_ns <= time.time_ns():
                            continue
                        self._publish_tier_observation(
                            request_id, token_ids, request_configs,
                        )
                        if promote:
                            self._schedule_cpu_promotion(
                                request_id=request_id,
                                token_ids=token_ids,
                                request_configs=request_configs,
                            )
                    for expired in callbacks.tickets.expire():
                        self._append_ticket(expired)
                        self._demote_prefetch_keys(expired.prefetch_id)
                    self._reap_terminal_prefetches(callbacks)
                except Exception as exc:
                    self._append("kv_core_prefetch_watcher_errors.jsonl", {
                        "error": type(exc).__name__, "timestamp_ns": time.time_ns(),
                    })
                time.sleep(max(0.002, _env_float("ASTRAKV_KV_CORE_PREFETCH_POLL_S", 0.005)))

        self._prefetch_watcher = threading.Thread(
            target=watch,
            name="astrakv-kv-prefetch-worker",
            daemon=True,
        )
        self._prefetch_watcher.start()

    def _write_request_terminal(
        self,
        *,
        logical_request_id: str,
        finish_status: str,
        completed: bool,
    ) -> None:
        if self._state_dir is None:
            return
        directory = self._state_dir / "request_terminals"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(logical_request_id.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "request_id": logical_request_id,
            "finish_status": finish_status,
            "completed": bool(completed),
            "finished_at_ns": time.time_ns(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _reap_terminal_prefetches(self, callbacks: KVCoreConnectorCallbacks) -> None:
        if self._state_dir is None:
            return
        directory = self._state_dir / "request_terminals"
        paths = tuple(directory.glob("*.json")) if directory.is_dir() else ()
        for path in paths:
            if path.name in self._terminal_request_seen:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                request_id = str(record["request_id"])
                completed = record.get("completed") is True
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                self._terminal_request_seen.add(path.name)
                continue
            self._terminal_request_seen.add(path.name)
            for ticket in callbacks.tickets.snapshot(
                statuses=(PrefetchStatus.SUBMITTED, PrefetchStatus.COMPLETED),
            ):
                if ticket.target_request_id != request_id:
                    continue
                try:
                    updated = (
                        callbacks.tickets.mark_wasted(
                            ticket.prefetch_id, reason="target_finished_without_consumption",
                        )
                        if completed
                        else callbacks.tickets.cancel(
                            ticket.prefetch_id, reason="target_cancelled_or_failed",
                        )
                    )
                except ValueError:
                    continue
                future = self._prefetch_futures.get(ticket.prefetch_id)
                if future is not None:
                    future.cancel()
                self._demote_prefetch_keys(ticket.prefetch_id)
                self._append_ticket(updated)

    def _demote_prefetch_keys(self, prefetch_id: str) -> None:
        keys = self._prefetch_keys.pop(prefetch_id, ())
        # A prefetched chunk may already be pinned by another exact-prefix
        # consumer.  Direct LocalCPUBackend.remove(force=True) would violate
        # LMCache ref-count ownership, so reclamation is delegated to its
        # native capacity/eviction policy after the ticket reservation closes.
        self._append("kv_core_policy_decisions.jsonl", {
            "prefetch_id": prefetch_id,
            "action": "demote_cpu_copy",
            "status": "delegated_to_lmcache_eviction",
            "key_count": len(keys),
            "timestamp_ns": time.time_ns(),
        })

    def _write_prefetch_request(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
        *,
        promote: bool,
    ) -> None:
        if self._state_dir is None:
            return
        directory = self._state_dir / "prefetch_requests"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.json"
        now = time.time_ns()
        payload = {
            "schema": "astrakv-prefetch-request-v1",
            "request_id": request_id,
            "exact_token_ids": list(token_ids),
            "request_configs": request_configs,
            "promote": bool(promote),
            "created_at_ns": now,
            "expires_at_ns": now + _env_int(
                "ASTRAKV_KV_CORE_PREFETCH_TTL_NS", 30_000_000_000,
            ),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _publish_tier_observation(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
    ) -> None:
        if self._state_dir is None:
            return
        manager = self._storage_manager()
        if manager is None:
            return
        backends = getattr(manager, "storage_backends", {})
        cpu, disk = backends.get("LocalCPUBackend"), backends.get("LocalDiskBackend")
        prefix_end = 0
        for _start, end, key in self._token_chunks(
            self._connector, token_ids, request_configs,
        ):
            if bool(cpu is not None and cpu.contains(key, False)) or bool(
                disk is not None and disk.contains(key, False)
            ):
                prefix_end = end
            else:
                break
        physical = None
        if prefix_end > 0:
            physical, _keys = self._physical(
                self._connector, token_ids[:prefix_end], request_configs,
            )
        directory = self._state_dir / "tier_observations"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.json"
        record = {
            "schema": "astrakv-native-tier-observation-v1",
            "request_id": request_id,
            "observed_prefix_tokens": prefix_end,
            "physical_object_id": physical.physical_object_id if physical else "",
            "binding_generation": physical.binding_generation if physical else 0,
            "native_key": physical.native_key if physical else "",
            "compatibility_identity": (
                physical.compatibility_key.identity if physical else ""
            ),
            "source_tier": physical.source_tier if physical else "unknown",
            "size_bytes": physical.size_bytes if physical else 0,
            "observed_at_ns": time.time_ns(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _apply_tier_observation(
        self,
        logical_request_id: str,
        physical: PhysicalKVObject,
    ) -> PhysicalKVObject:
        if self._state_dir is None:
            return physical
        digest = hashlib.sha256(logical_request_id.encode("utf-8")).hexdigest()
        path = self._state_dir / "tier_observations" / f"{digest}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return physical
        identity = (
            str(record.get("physical_object_id") or ""),
            int(record.get("binding_generation") or 0),
            str(record.get("native_key") or ""),
            str(record.get("compatibility_identity") or ""),
        )
        expected = (
            physical.physical_object_id,
            physical.binding_generation,
            physical.native_key,
            physical.compatibility_key.identity,
        )
        if identity != expected:
            return physical
        source_tier = str(record.get("source_tier") or "unknown")
        if source_tier not in {"cpu", "ssd", "mixed"}:
            return physical
        return replace(
            physical,
            source_tier=source_tier,
            size_bytes=max(0, int(record.get("size_bytes") or 0)),
        )

    def _consume_matching_prefetch(
        self,
        logical_request_id: str,
        native_request_id: str,
        physical: PhysicalKVObject,
        native_keys: tuple[Any, ...],
        lookup_hit_tokens: int,
    ) -> None:
        callbacks = self._callbacks()
        manager = self._storage_manager()
        if callbacks is None or manager is None or lookup_hit_tokens <= 0:
            return
        cpu = getattr(manager, "storage_backends", {}).get("LocalCPUBackend")
        if cpu is None or not native_keys or not all(cpu.contains(key, False) for key in native_keys):
            return
        for ticket in callbacks.tickets.snapshot(statuses=(
            PrefetchStatus.SUBMITTED, PrefetchStatus.COMPLETED,
        )):
            if (
                ticket.target_request_id == logical_request_id
                and ticket.generation_key == physical.generation_key
                and ticket.prefix_hash == physical.compatibility_key.prefix_hash
                and ticket.native_key == physical.native_key
            ):
                if ticket.status is PrefetchStatus.SUBMITTED:
                    ticket = callbacks.complete_cpu_prefetch(
                        ticket.prefetch_id,
                        completed_bytes=ticket.requested_bytes,
                    )
                    self._append_ticket(ticket)
                consumed = callbacks.consume_cpu_prefetch(
                    ticket.prefetch_id,
                    request_id=logical_request_id,
                    physical=physical,
                )
                self._prefetch_by_request[native_request_id] = ticket.prefetch_id
                self._append_ticket(consumed)
                return

    def _physical(
        self,
        connector: Any,
        token_ids: Iterable[int],
        request_configs: dict[str, Any] | None,
    ) -> tuple[PhysicalKVObject | None, tuple[Any, ...]]:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return None, ()
        chunks = self._token_chunks(connector, tokens, request_configs)
        keys = tuple(key for _start, _end, key in chunks)
        if not keys:
            return None, ()
        native_key = json.dumps([key.to_string() for key in keys], separators=(",", ":"), ensure_ascii=True)
        config = getattr(connector, "_vllm_config", None)
        model_config = getattr(config, "model_config", None)
        model_id = str(
            os.environ.get("ASTRAKV_MODEL_ID", "")
            or getattr(model_config, "served_model_name", "")
            or getattr(model_config, "model", "")
            or "Qwen3-8B"
        )
        compatibility = KVCompatibilityKey(
            model_id=model_id,
            model_revision=os.environ.get("ASTRAKV_MODEL_REVISION", "local-qwen3-8b"),
            tokenizer_revision=os.environ.get("ASTRAKV_TOKENIZER_REVISION", "local-qwen3-8b"),
            chat_template_revision=os.environ.get("ASTRAKV_CHAT_TEMPLATE_REVISION", "qwen3-default"),
            dtype=str(getattr(config, "dtype", "bfloat16")),
            rope_config=json.dumps(getattr(model_config, "rope_scaling", None) or {}, sort_keys=True),
            adapter_namespace=os.environ.get("ASTRAKV_ADAPTER_NAMESPACE", "base"),
            kv_layout=os.environ.get("ASTRAKV_KV_LAYOUT", "vllm-paged-kv-v1"),
            block_size_tokens=max(1, int(getattr(connector, "_block_size", 1))),
            chunk_size_tokens=max(1, int(getattr(connector, "_lmcache_chunk_size", 1))),
            layer_group="all-kv-layers",
            prefix_hash=exact_token_prefix_hash(tokens),
        )
        physical_id = hashlib.sha256(
            f"{compatibility.identity}:{native_key}".encode("ascii")
        ).hexdigest()
        manager = self._storage_manager()
        source_tier = "unknown"
        size_bytes = 0
        if manager is not None:
            backends = getattr(manager, "storage_backends", {})
            cpu, disk = backends.get("LocalCPUBackend"), backends.get("LocalDiskBackend")
            cpu_hits = 0 if cpu is None else sum(bool(cpu.contains(key, False)) for key in keys)
            disk_hits = 0 if disk is None else sum(bool(disk.contains(key, False)) for key in keys)
            source_tier = "cpu" if cpu_hits == len(keys) else "ssd" if disk_hits == len(keys) else "mixed" if cpu_hits + disk_hits else "unknown"
            disk_read_keys = tuple(
                key for key in keys
                if not bool(cpu is not None and cpu.contains(key, False))
                and bool(disk is not None and disk.contains(key, False))
            )
            size_bytes = self._disk_bytes(disk, disk_read_keys) if disk is not None else 0
        generation = self._registry.generation(physical_id)
        return PhysicalKVObject(
            native_key=native_key,
            physical_object_id=physical_id,
            binding_generation=generation,
            compatibility_key=compatibility,
            source_tier=source_tier,
            size_bytes=size_bytes,
        ), keys

    @staticmethod
    def _token_chunks(connector: Any, tokens: tuple[int, ...], request_configs: dict[str, Any] | None) -> tuple[tuple[int, int, Any], ...]:
        # LMCache 0.4.7 keeps the scheduler connector's lookup-side
        # TokenDatabase on ``lookup_client``; its ``lmcache_engine`` is
        # intentionally absent (the engine is worker-owned).  Use that
        # native database as the scheduler-side identity source so callback
        # records and admission decisions use the same keys as the worker.
        engine = getattr(connector, "lmcache_engine", None)
        database = getattr(engine, "token_database", None)
        if database is None:
            lookup_client = getattr(connector, "lookup_client", None)
            database = getattr(lookup_client, "token_database", None)
        if database is None:
            return ()
        return tuple(database.process_tokens(tokens=list(tokens), request_configs=request_configs))

    def _storage_manager(self) -> Any:
        engine = getattr(self._connector, "lmcache_engine", None)
        return getattr(engine, "storage_manager", None)

    @staticmethod
    def _disk_bytes(disk: Any, keys: Iterable[Any]) -> int:
        if disk is None:
            return 0
        total = 0
        lock = getattr(disk, "disk_lock", None)
        context = lock if lock is not None else _NullContext()
        with context:
            mapping = getattr(disk, "dict", {})
            for key in keys:
                metadata = mapping.get(key)
                total += max(0, int(getattr(metadata, "size", 0))) if metadata is not None else 0
        return total

    def _runtime_capability(self, *, external_token_cap: int) -> TierCapabilitySnapshot:
        callbacks = self._callbacks()
        base = callbacks.capability if callbacks is not None else TierCapabilitySnapshot(
            topology=TierTopology.GPU_SSD, local_cpu_enabled=False, local_disk_enabled=True,
        )
        manager = self._storage_manager()
        backends = {} if manager is None else getattr(manager, "storage_backends", {})
        cpu, disk = backends.get("LocalCPUBackend"), backends.get("LocalDiskBackend")
        cpu_capacity = int(float(getattr(getattr(self._connector, "config", None), "max_local_cpu_size", 0.0)) * 1024**3)
        cpu_used = 0
        if cpu is not None:
            cpu_lock = getattr(cpu, "cpu_lock", None)
            cpu_context = cpu_lock if cpu_lock is not None else _NullContext()
            with cpu_context:
                cpu_objects = tuple(getattr(cpu, "hot_cache", {}).values())
            cpu_used = sum(
                max(0, int(getattr(obj, "get_physical_size", lambda: 0)()))
                for obj in cpu_objects
            )
        ssd_capacity = max(0, int(getattr(disk, "max_cache_size", 0))) if disk is not None else 0
        ssd_used = max(0, int(getattr(disk, "current_cache_size", 0))) if disk is not None else 0
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
            available_kib = next(int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
            uma_available = available_kib * 1024
        except (OSError, StopIteration, ValueError):
            uma_available = base.uma_available_bytes
        pressure = base.memory_pressure
        try:
            current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            limit_text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if limit_text != "max":
                limit = int(limit_text)
                if limit > 0:
                    pressure = max(pressure, min(1.0, current / limit))
        except (OSError, ValueError):
            pass
        cache_config = getattr(getattr(self._connector, "_vllm_config", None), "cache_config", None)
        available_blocks = max(0, int(getattr(cache_config, "num_gpu_blocks", 0) or 0))
        unfinished = getattr(self._connector, "_unfinished_requests", None)
        queue_depth = len(unfinished) if isinstance(unfinished, dict) else base.queue_depth
        topology = TierTopology(os.environ.get("ASTRAKV_KV_CORE_TOPOLOGY", base.topology.value))
        return TierCapabilitySnapshot(
            topology=topology,
            local_cpu_enabled=bool(cpu is not None and getattr(cpu, "use_hot", False) and topology is TierTopology.GPU_CPU_SSD),
            local_disk_enabled=disk is not None,
            cpu_capacity_bytes=cpu_capacity,
            cpu_used_bytes=cpu_used,
            ssd_capacity_bytes=ssd_capacity,
            ssd_used_bytes=ssd_used,
            available_kv_blocks=available_blocks,
            external_token_cap=max(0, external_token_cap),
            uma_available_bytes=max(0, uma_available),
            memory_pressure=pressure,
            queue_depth=max(0, queue_depth),
            cpu_prefetch_budget_fraction=base.cpu_prefetch_budget_fraction,
        )

    def _logical_request_id(self, native_request_id: str) -> str:
        host = installed_runtime_control_host()
        identity = None if host is None else host.runtime_identity_for(native_request_id)
        return native_request_id if identity is None else identity.request_id

    @staticmethod
    def native_bytes_per_token(kvcaches: Iterable[Any], block_size: int) -> int:
        total = 0
        for cache in kvcaches:
            shape = getattr(cache, "shape", ())
            if not shape or int(shape[0]) <= 0:
                continue
            total += int(cache.numel()) * int(cache.element_size()) // (int(shape[0]) * max(1, block_size))
        return total

    def _callbacks(self) -> KVCoreConnectorCallbacks | None:
        return installed_kv_core_callbacks()

    def _record(self, callback: str, record: dict[str, Any]) -> None:
        self._seen_callbacks.add(callback)
        self._append("kv_core_native_callbacks.jsonl", {
            "callback": callback, "timestamp_ns": time.time_ns(), **record,
        })
        self._append_smoke()
        self._append_uma_sample(callback)

    def _append_uma_sample(self, callback: str) -> None:
        try:
            cgroup = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        except (OSError, ValueError):
            cgroup = 0
        rss = 0
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
        capability = self._runtime_capability(external_token_cap=0)
        cache_config = getattr(
            getattr(self._connector, "_vllm_config", None), "cache_config", None,
        )
        self._append("uma_resource_samples.jsonl", {
            "timestamp_ns": time.time_ns(), "callback": callback,
            "cgroup_memory_current_bytes": cgroup, "process_rss_bytes": rss,
            "lmcache_cpu_occupancy_bytes": capability.cpu_used_bytes,
            "lmcache_cpu_capacity_bytes": capability.cpu_capacity_bytes,
            "lmcache_ssd_occupancy_bytes": capability.ssd_used_bytes,
            "lmcache_ssd_capacity_bytes": capability.ssd_capacity_bytes,
            "vllm_kv_block_budget": max(
                0, int(getattr(cache_config, "num_gpu_blocks", 0) or 0),
            ),
            "uma_memory_pressure": capability.memory_pressure,
            "topology": capability.topology.value,
        })

    def _append_request_accounting(self, request_id: str, physical: PhysicalKVObject) -> None:
        callbacks = self._callbacks()
        if callbacks is None:
            return
        final = callbacks.final_accounting_for(request_id)
        if final is not None:
            record = asdict(final)
            terminal = True
        else:
            intent = callbacks.intent_for(request_id)
            lookup = callbacks.lookup_for(request_id)
            if intent is None or lookup is None:
                return
            admission = callbacks.admission_for(request_id)
            receipt = callbacks.receipt_for(request_id)
            allocated = 0 if admission is None else admission.allocated_external_tokens
            loaded = 0 if receipt is None else receipt.actual_loaded_tokens
            record = {
                "request_id": request_id,
                "physical_object_id": physical.physical_object_id,
                "binding_generation": physical.binding_generation,
                "native_key": physical.native_key,
                "compatibility_identity": physical.compatibility_key.identity,
                "prefix_hash": physical.compatibility_key.prefix_hash,
                "requested_prefix_tokens": intent.requested_prefix_tokens,
                "locally_cached_tokens": lookup.locally_cached_tokens,
                "lookup_hit_tokens": lookup.lookup_hit_tokens,
                "allocated_external_tokens": allocated,
                "actual_loaded_tokens": loaded,
                "missing_tokens": max(
                    0,
                    intent.requested_prefix_tokens - lookup.locally_cached_tokens - loaded,
                ),
                "unallocated_recompute_tokens": max(
                    0,
                    intent.requested_prefix_tokens
                    - lookup.locally_cached_tokens
                    - allocated,
                ),
                "load_shortfall_tokens": max(0, allocated - loaded),
                "scheduled_prefill_tokens": callbacks.scheduled_prefill_for(request_id),
                "recomputed_tokens": 0,
                "recompute_confirmed": False,
                "finish_status": "pending",
                "terminal_reason": "native_request_pending",
            }
            terminal = False
        record.update({
            "native_key": physical.native_key,
            "compatibility_identity": physical.compatibility_key.identity,
            "prefix_hash": physical.compatibility_key.prefix_hash,
            "terminal": terminal,
            "timestamp_ns": time.time_ns(),
        })
        self._append("kv_core_request_accounting.jsonl", record)

    def _append_ticket(self, ticket: PrefetchTicket) -> None:
        self._append("kv_core_prefetch_tickets.jsonl", asdict(ticket))

    def _native_receipt_path(self, request_id: str) -> Path | None:
        if self._state_dir is None:
            return None
        digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
        directory = self._state_dir / "native_receipts"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}.json"

    def _native_intent_path(self, request_id: str) -> Path | None:
        if self._state_dir is None:
            return None
        digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
        directory = self._state_dir / "native_intents"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}.json"

    def _write_native_intent(
        self,
        *,
        request_id: str,
        physical: PhysicalKVObject,
        requested_prefix_tokens: int,
        locally_cached_tokens: int,
        lookup_hit_tokens: int,
        max_external_tokens: int,
        prefetch_id: str,
    ) -> None:
        path = self._native_intent_path(request_id)
        if path is None:
            return
        record = {
            "request_id": request_id,
            "logical_request_id": self._logical_request_id(request_id),
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "native_key": physical.native_key,
            "compatibility_identity": physical.compatibility_key.identity,
            "prefix_hash": physical.compatibility_key.prefix_hash,
            "source_tier": physical.source_tier,
            "requested_prefix_tokens": int(requested_prefix_tokens),
            "locally_cached_tokens": int(locally_cached_tokens),
            "lookup_hit_tokens": int(lookup_hit_tokens),
            "max_external_tokens": int(max_external_tokens),
            "allocated_external_tokens": 0,
            "prefetch_id": prefetch_id,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _read_native_intent(self, request_id: str) -> dict[str, Any] | None:
        path = self._native_intent_path(request_id)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _update_native_intent(self, request_id: str, **changes: Any) -> None:
        path = self._native_intent_path(request_id)
        record = self._read_native_intent(request_id)
        if path is None or record is None:
            return
        record.update(changes)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _write_native_receipt(self, receipt: NativeKVLoadReceipt) -> None:
        """Publish a worker receipt atomically for a scheduler-side finisher."""
        path = self._native_receipt_path(receipt.request_id)
        if path is None:
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(receipt), sort_keys=True) + "\n", encoding="utf-8",
        )
        os.replace(temporary, path)

    def _read_native_receipt(self, request_id: str) -> NativeKVLoadReceipt | None:
        path = self._native_receipt_path(request_id)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return NativeKVLoadReceipt(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_run_metadata(self, connector: Any) -> None:
        if self._state_dir is None:
            return
        config = getattr(connector, "config", None)
        cache_config = getattr(getattr(connector, "_vllm_config", None), "cache_config", None)
        payload = {
            "schema": "astrakv-kv-core-runtime-metadata-v2",
            "topology": os.environ.get("ASTRAKV_KV_CORE_TOPOLOGY", "unknown"),
            "lmcache_local_cpu_enabled": bool(getattr(config, "local_cpu", False)),
            "lmcache_local_disk_enabled": bool(getattr(config, "local_disk", False)),
            "vllm_kv_block_budget": getattr(cache_config, "num_gpu_blocks", None),
            "vllm_block_size_tokens": getattr(connector, "_block_size", None),
            "lmcache_chunk_size_tokens": getattr(connector, "_lmcache_chunk_size", None),
            "vendor_patch": True,
            "legacy_owner_load_enabled": False,
            "observed_at_ns": time.time_ns(),
        }
        (self._state_dir / "kv_core_run_metadata.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8",
        )

    def _append_smoke(self) -> None:
        if self._state_dir is None:
            return
        target = self._state_dir / "callback-smoke.json"
        lock_path = self._state_dir / "callback-smoke.lock"
        with lock_path.open("a+", encoding="ascii") as lock_handle:
            try:
                import fcntl
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                fcntl = None  # type: ignore[assignment]
            observed = set(self._seen_callbacks)
            try:
                prior = json.loads(target.read_text(encoding="utf-8"))
                observed.update(str(item) for item in prior.get("observed_callbacks", ()))
            except (OSError, AttributeError, json.JSONDecodeError):
                pass
            payload = {
                "patch_id": PATCH_ID,
                "callbacks": list(REQUIRED_CALLBACKS),
                "observed_callbacks": sorted(observed),
                "passed": set(REQUIRED_CALLBACKS).issubset(observed),
                "updated_at_ns": time.time_ns(),
            }
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8",
            )
            os.replace(temporary, target)
            if fcntl is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    def _append(self, filename: str, record: dict[str, Any]) -> None:
        if self._state_dir is None:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with (self._state_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


__all__ = ["VendorCallbackBridge"]
