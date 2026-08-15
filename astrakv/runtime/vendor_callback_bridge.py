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
from astrakv.runtime.kv_core_connector import native_key_prefix_ok
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
from astrakv.runtime.lmcache047_runtime_patch import patch_local_disk_remove_race_class
from astrakv.runtime.offline_kv_profile import OfflineKVProfileIndex, PrefixRuntimeHint
from astrakv.runtime.third_party_patch import PATCH_ID, REQUIRED_CALLBACKS
from astrakv.runtime.uma_metrics import current_cgroup_memory_evidence


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


def _exact_token_sequence_hash(tokens: Iterable[int]) -> str:
    """Hash a complete token sequence without conflating it with KV identity."""
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


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
        self._ingress_object_key: dict[str, str] = {}
        self._ingress_prefetch_origin: dict[str, str] = {}
        # Test-only per-request decision overrides (memory pressure, deadline,
        # IO bandwidth, forced recompute) published through the authenticated
        # request context.  Populated only when the equivalence test gate is
        # enabled; the production path never reads this dict.
        self._decision_probe_by_request: dict[str, dict[str, Any]] = {}
        # In-memory fallback for the same-process equivalence probe. The
        # authoritative cross-process handoff is an atomic state-dir record.
        self._equivalence_force_recompute_token_hashes: set[str] = set()
        self._lookup_started_ns: dict[str, int] = {}
        self._seen_callbacks: set[str] = set()
        self._prefetch_request_seen: set[str] = set()
        self._terminal_request_seen: set[str] = set()
        self._consumed_prefetch_seen: set[str] = set()
        self._association_receipts: dict[str, Any] = {}
        self._prefetch_keys: dict[str, tuple[Any, ...]] = {}
        self._prefetch_lease_by_id: dict[str, str] = {}
        self._prefetch_futures: dict[str, Any] = {}
        self._prefetch_watcher: threading.Thread | None = None
        # Cold external-copy reaper state, keyed by physical object id.  The
        # reaper only ever removes LocalCPU/LocalDisk copies through the
        # manager's ref-count-aware ``remove`` API; it never touches GPU paged
        # KV and never bypasses LMCache ownership.
        self._reap_state: dict[str, dict[str, Any]] = {}
        self._bootstrap_loads = 0
        self._ssd_bytes_per_ms_ema = 0.0
        self._prefill_ms_per_token_ema = _env_float("ASTRAKV_KV_CORE_PREFILL_MS_PER_TOKEN", 0.0)
        # ``scheduler_compute_progress`` is emitted immediately before each
        # native execution step.  The following callback closes the previous
        # step, so its elapsed time is an online, scheduler-owned prefill
        # observation.  This avoids guessing a load-vs-recompute cost from
        # prompt text or HTTP timing.
        self._pending_prefill_steps: dict[str, tuple[int, int]] = {}
        self._scheduled_prefix_tokens: dict[str, int] = {}
        self._prefill_observation_count = 0
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
        # Crash-safety guard: LocalDiskBackend.remove races a concurrent
        # eviction (missing backing file) and would otherwise kill EngineCore.
        # Applied on the class so every LMCache engine instance in this
        # process is covered, including LMCache-only servers launched with
        # AstraKV hooks disabled (e.g. the E11 warmup server).  Idempotent.
        patch_local_disk_remove_race_class()
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
        # ``connector=None`` may have created the process-local control host
        # before LMCache constructs the real connector.  In that startup
        # order the connector's role metadata is not a reliable ownership
        # signal (vLLM 0.23 can expose WORKER with no worker_count yet), but a
        # process-local host is authoritative proof that this bridge must be
        # attached to it.  A bridge in another process observes ``host is
        # None`` and therefore still cannot register across process bounds.
        if host is not None:
            host.register_kv_runtime_bridge(bridge)
        return bridge

    def ingress_request(self, context: Any) -> None:
        """Accept an authenticated exact-token intent before HTTP submission."""
        request_id = str(getattr(context, "request_id", "") or "")
        metadata = dict(getattr(context, "metadata", {}) or {})
        object_key = str(
            metadata.get("object_key")
            or metadata.get("cache_key")
            or request_id
        )
        self._ingress_object_key[request_id] = object_key
        predictive_prefetch_authorized = (
            metadata.get("predictive_prefetch_authorized") is True
            and str(metadata.get("prefetch_origin") or "") in {
                "sidecar_b", "profile_b",
            }
        )
        prefetch_origin = (
            str(metadata.get("prefetch_origin") or "")
            if predictive_prefetch_authorized
            else "prefetch_a"
        )
        self._ingress_prefetch_origin[request_id] = prefetch_origin
        try:
            tokens = tuple(int(token) for token in metadata.get("exact_token_ids", ()))
        except (TypeError, ValueError):
            tokens = ()
        if not request_id:
            return
        self._ingress_started_ns[request_id] = time.time_ns()
        if _env_flag("ASTRAKV_KV_CORE_EQUIVALENCE_TEST"):
            probe = metadata.get("kv_core_decision_probe")
            if isinstance(probe, dict) and probe:
                self._decision_probe_by_request[request_id] = dict(probe)
        if (
            _env_flag("ASTRAKV_KV_CORE_EQUIVALENCE_TEST")
            and str(metadata.get("kv_core_equivalence_mode") or "") == "force_recompute"
            and tokens
        ):
            token_sequence_hash = _exact_token_sequence_hash(tokens)
            self._equivalence_force_recompute_token_hashes.add(token_sequence_hash)
            self._write_equivalence_recompute_intent(
                token_sequence_hash=token_sequence_hash,
                logical_request_id=request_id,
            )
        if not tokens:
            return
        request_configs = metadata.get("request_configs")
        normalized_configs = request_configs if isinstance(request_configs, dict) else None
        callbacks = self._callbacks()
        active = callbacks is not None and callbacks.mode is RuntimeMode.ACTIVE
        try:
            prefetch_lead_s = float(metadata.get("prefetch_lead_s") or 0.0)
        except (TypeError, ValueError):
            prefetch_lead_s = 0.0
        invalidate_disk_backed_cpu = (
            _env_flag("ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD")
            and prefetch_lead_s > 0.0
        )
        promote = active and (
            _env_flag("ASTRAKV_KV_CORE_CPU_PREFETCH_ENABLED")
            or predictive_prefetch_authorized
        )
        if self._storage_manager() is None:
            # The EngineCore scheduler owns authenticated ingress, while the
            # worker owns LocalDiskBackend/LocalCPUBackend.  Publish the exact
            # request intent so the worker can reset a stale hot copy and,
            # only for the active variant, overlap promotion with queueing.
            if invalidate_disk_backed_cpu or active:
                self._write_prefetch_request(
                    request_id,
                    tokens,
                    normalized_configs,
                    promote=promote,
                    invalidate_disk_backed_cpu=invalidate_disk_backed_cpu,
                    prefetch_origin=prefetch_origin,
                )
            return
        if invalidate_disk_backed_cpu:
            self._invalidate_disk_backed_cpu_prefix(
                request_id=request_id,
                token_ids=tokens,
                request_configs=normalized_configs,
                reason="symmetric_cpu_cold_before_prefetch_lead",
            )
        if not active:
            return
        self._publish_tier_observation(request_id, tokens, normalized_configs)
        if promote:
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
                        token_sequence_hash=_exact_token_sequence_hash(tokens),
                    )
                    callbacks.submit_intent(RequestKVIntent(
                        request_id=request_id,
                        compatibility_key=physical.compatibility_key,
                        physical_object=physical,
                        max_external_tokens=cap,
                        requested_prefix_tokens=requested,
                        deadline_ns=self._ingress_started_ns.get(
                            logical_request_id, time.time_ns(),
                        ) + self._probe_deadline_offset_ns(logical_request_id),
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
                    token_prefix=candidate_tokens, request_configs=request_configs,
                )
                self._note_reap_reference(
                    physical=physical,
                    native_keys=native_keys,
                    hit_tokens=available_hit,
                    now_ns=time.time_ns(),
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
        now_ns = time.time_ns()
        try:
            self._observe_completed_prefill_step(request_id, now_ns=now_ns)
            scheduled = max(0, int(scheduled_tokens))
            callbacks.record_scheduler_compute(
                request_id=request_id, scheduled_tokens=scheduled,
            )
            intent = callbacks.intent_for(request_id)
            admitted = callbacks.admission_for(request_id)
            prefix_seen = self._scheduled_prefix_tokens.get(
                request_id,
                0 if admitted is None else admitted.allocated_external_tokens,
            )
            requested_prefix = 0 if intent is None else intent.requested_prefix_tokens
            prefill_tokens = min(scheduled, max(0, requested_prefix - prefix_seen))
            self._scheduled_prefix_tokens[request_id] = prefix_seen + prefill_tokens
            if prefill_tokens > 0:
                self._pending_prefill_steps[request_id] = (now_ns, prefill_tokens)
            self._record("scheduler_compute_progress", {
                "request_id": request_id,
                "scheduled_tokens": scheduled,
                "prefill_tokens": prefill_tokens,
            })
        except ValueError as exc:
            self._record("scheduler_compute_progress", {
                "request_id": request_id, "status": "rejected", "reason": str(exc),
            })

    def _observe_completed_prefill_step(self, request_id: str, *, now_ns: int) -> None:
        """Update the prefill-cost EMA from a completed native scheduler step.

        The vendor callback has no direct model-runner timing API in the
        supported vLLM 0.23.0/LMCache 0.4.7 integration.  Consecutive
        scheduler progress callbacks are the narrowest native boundary that
        surrounds an executed batch.  We retain only chunk-sized prompt work,
        reject impossible/outlier samples, and write every accepted or
        rejected observation to an independent audit artifact.
        """
        pending = self._pending_prefill_steps.pop(request_id, None)
        if pending is None or not _env_flag("ASTRAKV_KV_CORE_PREFILL_ONLINE_CALIBRATION"):
            return
        started_ns, tokens = pending
        elapsed_ns = max(0, int(now_ns) - started_ns)
        elapsed_ms = elapsed_ns / 1_000_000.0
        minimum_tokens = max(1, _env_int("ASTRAKV_KV_CORE_PREFILL_SAMPLE_MIN_TOKENS", 32))
        maximum_ms_per_token = _env_float(
            "ASTRAKV_KV_CORE_PREFILL_SAMPLE_MAX_MS_PER_TOKEN", 5.0,
        )
        sample = elapsed_ms / tokens if tokens > 0 else 0.0
        accepted = (
            tokens >= minimum_tokens
            and elapsed_ns > 0
            and sample > 0.0
            and maximum_ms_per_token > 0.0
            and sample <= maximum_ms_per_token
        )
        previous = self._prefill_ms_per_token_ema
        if accepted:
            alpha = min(1.0, max(0.0, _env_float("ASTRAKV_KV_CORE_PREFILL_EMA_ALPHA", 0.25)))
            self._prefill_ms_per_token_ema = (
                sample if previous <= 0.0 else (1.0 - alpha) * previous + alpha * sample
            )
            self._prefill_observation_count += 1
        self._append("kv_core_cost_observations.jsonl", {
            "schema": "astrakv-kv-core-cost-observation-v1",
            "request_id": request_id,
            "source": "scheduler_compute_progress",
            "prefill_tokens": tokens,
            "elapsed_ns": elapsed_ns,
            "sample_ms_per_token": sample,
            "accepted": accepted,
            "previous_prefill_ms_per_token": previous,
            "observed_prefill_ms_per_token": self._prefill_ms_per_token_ema,
            "observation_count": self._prefill_observation_count,
            "timestamp_ns": int(now_ns),
        })

    def connector_metadata(self, *, request_id: str, metadata_present: bool, can_load: bool) -> None:
        receipt = self._associate_runtime_request(request_id)
        logical_request_id = self._logical_request_id(request_id)
        self._record("connector_metadata", {
            "request_id": request_id,
            "native_request_id": request_id,
            "logical_request_id": logical_request_id,
            "runtime_event_id": "" if receipt is None else str(getattr(receipt, "runtime_event_id", "") or ""),
            "association_receipt_reference": "" if receipt is None else str(getattr(receipt, "runtime_event_id", "") or ""),
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
        receipt_identity = self._associate_runtime_request(request_id)
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
            record.update(self._association_evidence(request_id, receipt_identity))
            self._append("kv_core_native_receipts.jsonl", record)
            self._record("native_load_completion", record)
            return self._active_admission() and receipt.load_shortfall_tokens > 0
        except (TypeError, ValueError) as exc:
            self._record("native_load_completion", {
                "request_id": request_id,
                **self._association_evidence(request_id, receipt_identity),
                "status": "rejected", "reason": str(exc),
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
        association_receipt = self._associate_runtime_request(request_id)
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
                **self._association_evidence(request_id, association_receipt),
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
                # Churn can evict tail chunks between scheduler lookup and the
                # native load.  A shorter block-aligned prefix of the same
                # binding is still the scheduler-declared object; the evicted
                # tail is reconciled by missing_tokens recompute.
                partial_prefix_ok = (
                    int(expected[1]) == int(observed[1])
                    and native_key_prefix_ok(str(expected[2]), str(observed[2]))
                )
                if not partial_prefix_ok:
                    self._record("native_load_start", {
                        "request_id": request_id,
                        **self._association_evidence(request_id, association_receipt),
                        "status": "rejected",
                        "reason": "native_intent_identity_mismatch",
                    })
                    return
                self._record("native_load_start", {
                    "request_id": request_id,
                    **self._association_evidence(request_id, association_receipt),
                    "status": "accepted_partial_prefix",
                    "reason": "observed_prefix_subset_of_intent",
                    "physical_object_id": physical.physical_object_id,
                    "binding_generation": physical.binding_generation,
                })
        logical_request_id = str(
            (intent_record or {}).get("logical_request_id") or request_id
        )
        self._consume_matching_prefetch(
            logical_request_id,
            request_id,
            physical,
            native_keys,
            max(0, int(allocated_external_tokens)),
            token_prefix=tokens[:prefix_tokens],
            request_configs=request_configs,
        )
        self._record("native_load_start", {
            "request_id": request_id,
            **self._association_evidence(request_id, association_receipt),
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
        association_receipt = self._associate_runtime_request(request_id)
        callbacks, physical = self._callbacks(), self._physical_by_request.get(request_id)
        if callbacks is None or physical is None:
            return
        completed = "ABORT" not in finish_status.upper() and "ERROR" not in finish_status.upper()
        try:
            # A final request callback also closes the last prefill step when
            # no later scheduler callback exists (for example, zero-token
            # generation).  Decode-only steps are filtered by the same
            # minimum-token guard used by the online calibration path.
            self._observe_completed_prefill_step(request_id, now_ns=time.time_ns())
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
                **self._association_evidence(request_id, association_receipt),
                "native_num_computed_tokens": max(0, int(num_computed_tokens)),
                "native_num_tokens": max(0, int(num_tokens)),
                "native_key": physical.native_key,
                "compatibility_identity": physical.compatibility_key.identity,
                # This is the final native request lifecycle callback. The
                # accounting value is no longer provisional once it returns.
                "terminal": True,
                "timestamp_ns": time.time_ns(),
            })
            self._append("kv_core_request_accounting.jsonl", record)
            self._record("request_finished", record)
            self._write_request_terminal(
                logical_request_id=self._logical_request_id(request_id),
                finish_status=finish_status,
                completed=completed,
            )
            self._finalize_local_prefetch_for_terminal(
                logical_request_id=self._logical_request_id(request_id),
                completed=completed,
            )
        except (TypeError, ValueError) as exc:
            self._record("request_finished", {
                "request_id": request_id, "native_request_id": request_id,
                "logical_request_id": self._logical_request_id(request_id),
                "status": "rejected", "reason": str(exc),
            })
        finally:
            self._note_reap_release(physical, time.time_ns())
            self._release_cpu_staging_after_consume(request_id)
            self._pending_prefill_steps.pop(request_id, None)
            self._scheduled_prefix_tokens.pop(request_id, None)

    def _finalize_local_prefetch_for_terminal(
        self,
        *,
        logical_request_id: str,
        completed: bool,
    ) -> None:
        """Settle target tickets immediately when this bridge owns CPU tier state.

        The state-dir watcher remains necessary for split scheduler/worker
        processes.  In the supported single-worker ``kv_both`` topology the
        native request-finished callback executes in the same process as
        LocalCPUBackend, so waiting for a later poll can leave a completed
        ticket unaccounted when the benchmark stops the service immediately.
        """
        callbacks = self._callbacks()
        if callbacks is None or self._storage_manager() is None:
            return
        for ticket in callbacks.tickets.snapshot(
            statuses=(PrefetchStatus.SUBMITTED, PrefetchStatus.COMPLETED),
        ):
            if ticket.target_request_id != logical_request_id:
                continue
            try:
                settled = (
                    callbacks.tickets.mark_wasted(
                        ticket.prefetch_id,
                        reason="target_finished_without_consumption",
                    )
                    if completed
                    else callbacks.tickets.cancel(
                        ticket.prefetch_id,
                        reason="target_cancelled_or_failed",
                    )
                )
            except ValueError:
                continue
            future = self._prefetch_futures.get(ticket.prefetch_id)
            if future is not None:
                future.cancel()
            self._demote_prefetch_keys(ticket.prefetch_id)
            self._append_ticket(settled)

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
        token_sequence_hash: str,
    ) -> int:
        available = max(0, int(available_external_tokens))
        if not self._active_admission() or available == 0:
            return available
        probe = self._decision_probe_by_request.get(logical_request_id)
        if probe is not None:
            probe = dict(probe)
        if probe is not None and probe.get("force_recompute") is True:
            self._append("kv_core_policy_decisions.jsonl", {
                "request_id": logical_request_id,
                "physical_object_id": physical.physical_object_id,
                "binding_generation": physical.binding_generation,
                "action": "recompute",
                "reason": "equivalence_probe_force_recompute",
                "requested_prefix_tokens": int(requested_tokens),
                "exact_token_sequence_hash": token_sequence_hash,
                "candidate_external_tokens": 0,
                "test_only": True,
                "decision_probe": probe,
                "timestamp_ns": time.time_ns(),
            })
            return 0
        if self._consume_equivalence_force_recompute(
            logical_request_id, requested_tokens, token_sequence_hash,
        ):
            return 0
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
            deadline_ns=ingress_ns + self._probe_deadline_offset_ns(logical_request_id),
            priority=priority + (
                profile_hint.admission_priority_boost if profile_hint is not None else 0
            ),
        )
        queue_delay_ms = 0.0
        ingress = self._ingress_started_ns.get(logical_request_id)
        if ingress is not None:
            queue_delay_ms = max(0.0, (time.time_ns() - ingress) / 1_000_000.0)
        probe_ssd_gbps = None
        if probe is not None:
            raw_gbps = probe.get("ssd_gbps")
            if raw_gbps not in (None, ""):
                try:
                    probe_ssd_gbps = float(raw_gbps)
                except (TypeError, ValueError):
                    probe_ssd_gbps = None
        if probe_ssd_gbps is not None:
            bandwidth = probe_ssd_gbps * 1_000_000.0
        else:
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
        transfer_ms = _env_float("ASTRAKV_KV_CORE_TRANSFER_MS", 0.0)
        materialization_ms = _env_float("ASTRAKV_KV_CORE_MATERIALIZATION_MS", 0.0)
        contention_ms = _env_float("ASTRAKV_KV_CORE_CONTENTION_MS", 0.0)
        if probe is not None:
            for key in ("transfer_ms", "materialization_ms", "contention_ms"):
                value = probe.get(key)
                if value not in (None, ""):
                    try:
                        if key == "transfer_ms":
                            transfer_ms = float(value)
                        elif key == "materialization_ms":
                            materialization_ms = float(value)
                        else:
                            contention_ms = float(value)
                    except (TypeError, ValueError):
                        pass
        pressure_override = None
        if probe is not None:
            raw_pressure = probe.get("memory_pressure")
            if raw_pressure not in (None, ""):
                try:
                    pressure_override = min(1.0, max(0.0, float(raw_pressure)))
                except (TypeError, ValueError):
                    pressure_override = None
        if pressure_override is not None:
            capability = replace(capability, memory_pressure=pressure_override)
        decision = choose_load_vs_recompute(
            intent=intent,
            capability=capability,
            queue_delay_ms=queue_delay_ms,
            tier_read_ms=read_ms,
            transfer_ms=transfer_ms,
            materialization_ms=materialization_ms,
            contention_ms=contention_ms,
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
            "decision_probe": probe,
            "timestamp_ns": time.time_ns(),
        })
        return cap if decision.action == "admit_external_prefix" else 0

    def _probe_deadline_offset_ns(self, logical_request_id: str) -> int:
        """Test-only per-request load deadline; falls back to the env knob."""

        probe = self._decision_probe_by_request.get(logical_request_id)
        if probe is not None:
            value = probe.get("deadline_ns")
            if value not in (None, ""):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return _env_int("ASTRAKV_KV_CORE_LOAD_DEADLINE_NS", 60_000_000_000)

    def _consume_equivalence_force_recompute(
        self,
        logical_request_id: str,
        requested_tokens: int,
        token_sequence_hash: str,
    ) -> bool:
        """One-shot, test-only recompute control keyed by exact full tokens.

        Scheduler lookup intentionally occurs before ReqMeta association. The
        authenticated ingress context therefore authorizes this probe through
        an exact complete-token digest, never by guessing a native request ID.
        """
        if not _env_flag("ASTRAKV_KV_CORE_EQUIVALENCE_TEST"):
            return False
        intent_path = self._consume_equivalence_recompute_intent(token_sequence_hash)
        if intent_path is None:
            return False
        self._equivalence_force_recompute_token_hashes.discard(token_sequence_hash)
        self._append("kv_core_policy_decisions.jsonl", {
            "request_id": logical_request_id,
            "action": "recompute_missing_suffix",
            "reason": "equivalence_probe_force_recompute",
            "requested_prefix_tokens": int(requested_tokens),
            "exact_token_sequence_hash": token_sequence_hash,
            "equivalence_intent_path": str(intent_path),
            "test_only": True,
            "timestamp_ns": time.time_ns(),
        })
        return True

    def _equivalence_recompute_intent_path(
        self, token_sequence_hash: str, *, consumed: bool = False,
    ) -> Path | None:
        if self._state_dir is None or len(token_sequence_hash) != 64:
            return None
        directory = self._state_dir / "equivalence_recompute_intents"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = ".consumed.json" if consumed else ".json"
        return directory / f"{token_sequence_hash}{suffix}"

    def _write_equivalence_recompute_intent(
        self, *, token_sequence_hash: str, logical_request_id: str,
    ) -> None:
        """Publish test-only intent for the separate EngineCore process."""
        path = self._equivalence_recompute_intent_path(token_sequence_hash)
        if path is None:
            return
        now_ns = time.time_ns()
        payload = {
            "schema": "astrakv-kv-equivalence-intent-v1",
            "exact_token_sequence_hash": token_sequence_hash,
            "logical_request_id": logical_request_id,
            "created_at_ns": now_ns,
            "expires_at_ns": now_ns + _env_int(
                "ASTRAKV_KV_CORE_EQUIVALENCE_TEST_TTL_NS", 30_000_000_000,
            ),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _consume_equivalence_recompute_intent(self, token_sequence_hash: str) -> Path | None:
        """Atomically claim an unexpired exact-token test intent once."""
        path = self._equivalence_recompute_intent_path(token_sequence_hash)
        consumed_path = self._equivalence_recompute_intent_path(token_sequence_hash, consumed=True)
        if path is None or consumed_path is None:
            # Unit-level bridge tests may not use a state directory. Runtime
            # execution never takes this branch because active mode requires it.
            return Path("<in-memory>") if token_sequence_hash in self._equivalence_force_recompute_token_hashes else None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expires_at_ns = int(payload.get("expires_at_ns") or 0)
            if (
                payload.get("schema") != "astrakv-kv-equivalence-intent-v1"
                or payload.get("exact_token_sequence_hash") != token_sequence_hash
                or expires_at_ns <= time.time_ns()
            ):
                return None
            os.replace(path, consumed_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return consumed_path

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
        profile_hint = self._profile_hint(physical)
        in_flight_reserved_bytes = callbacks.tickets.in_flight_reserved_bytes()
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
        reason: str | None = None
        prefetch_lease: str | None = None
        if callbacks.tickets.has_open_for_physical(
            physical.physical_object_id, physical.binding_generation,
        ):
            reason = "prefetch_duplicate_for_physical_object"
        object_key = self._ingress_object_key.get(request_id, request_id)
        if reason is None:
            host = installed_runtime_control_host()
            guard = getattr(host, "prefetch_guard", None) if host is not None else None
            if guard is not None:
                prefetch_lease = guard.try_begin(
                    object_key,
                    request_id=request_id,
                    deadline_ns=ticket.deadline_ns,
                )
                if prefetch_lease is None:
                    reason = "prefetch_single_flight_conflict"
        if profile_hint is not None and capability.cpu_prefetch_budget_bytes > 0:
            budget_utilization = (
                capability.cpu_prefetch_used_budget_bytes + in_flight_reserved_bytes
            ) / capability.cpu_prefetch_budget_bytes
            if budget_utilization > profile_hint.prefetch_priority:
                reason = "profile_prefetch_priority"
        if reason is None:
            reason = callbacks.begin_cpu_prefetch(ticket, physical, now_ns=now)
        self._append("kv_core_policy_decisions.jsonl", {
            "action": "prefetch_ssd_to_cpu",
            "request_id": request_id,
            "prefetch_id": ticket.prefetch_id,
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "requested_bytes": ticket.requested_bytes,
            "cpu_used_bytes": capability.cpu_used_bytes,
            "cpu_capacity_bytes": capability.cpu_capacity_bytes,
            "cpu_prefetch_budget_bytes": capability.cpu_prefetch_budget_bytes,
            "in_flight_reserved_bytes": in_flight_reserved_bytes,
            "memory_pressure": capability.memory_pressure,
            "status": "rejected" if reason is not None else "submitted",
            "reason": reason or "",
            "prefetch_origin": self._ingress_prefetch_origin.get(
                request_id, "prefetch_a",
            ),
            "timestamp_ns": time.time_ns(),
        })
        if reason is not None:
            if prefetch_lease is not None:
                host = installed_runtime_control_host()
                guard = getattr(host, "prefetch_guard", None) if host is not None else None
                if guard is not None:
                    guard.complete(prefetch_lease)
            self._append_ticket(replace(ticket, status=PrefetchStatus.CANCELLED, failure_reason=reason))
            return
        self._append_ticket(ticket)
        self._prefetch_keys[ticket.prefetch_id] = selected_keys
        self._prefetch_lease_by_id[ticket.prefetch_id] = prefetch_lease

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
                self._release_prefetch_lease(ticket.prefetch_id)
            self._append_ticket(updated)

        future.add_done_callback(completed)

    def _profile_hint(self, physical: PhysicalKVObject) -> PrefixRuntimeHint | None:
        if self._profile_index is None:
            return None
        return self._profile_index.hint_for(physical.compatibility_key)

    def _process_prefetch_request_file(self, path: Path) -> None:
        """Process one worker-side prefetch-request handoff file.

        Extracted from the watcher loop so the handoff path is directly
        testable: parse, mark seen, and dispatch invalidation/observation/
        promotion for a single request file.
        """
        key = path.name
        if key in self._prefetch_request_seen:
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            request_id = str(payload["request_id"])
            token_ids = tuple(int(token) for token in payload["exact_token_ids"])
            expires_at_ns = int(payload["expires_at_ns"])
            request_configs = payload.get("request_configs")
            promote = payload.get("promote") is True
            invalidate_disk_backed_cpu = payload.get("invalidate_disk_backed_cpu") is True
            prefetch_origin = str(payload.get("prefetch_origin") or "prefetch_a")
            if not isinstance(request_configs, dict):
                request_configs = None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._prefetch_request_seen.add(key)
            return
        self._prefetch_request_seen.add(key)
        if expires_at_ns <= time.time_ns():
            return
        callbacks = self._callbacks()
        self._ingress_prefetch_origin[request_id] = prefetch_origin
        if invalidate_disk_backed_cpu:
            self._invalidate_disk_backed_cpu_prefix(
                request_id=request_id,
                token_ids=token_ids,
                request_configs=request_configs,
                reason="symmetric_cpu_cold_before_prefetch_lead",
            )
        self._publish_tier_observation(
            request_id, token_ids, request_configs,
        )
        if promote and callbacks is not None and callbacks.mode is RuntimeMode.ACTIVE:
            self._schedule_cpu_promotion(
                request_id=request_id,
                token_ids=token_ids,
                request_configs=request_configs,
            )

    def _start_prefetch_watcher_if_worker(self) -> None:
        if (
            self._state_dir is None
            or self._storage_manager() is None
        ):
            return
        callbacks = self._callbacks()
        if callbacks is None or (
            callbacks.mode is not RuntimeMode.ACTIVE
            and not _env_flag("ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD")
        ):
            return

        def watch() -> None:
            directory = self._state_dir / "prefetch_requests"
            while True:
                try:
                    paths = tuple(directory.glob("*.json")) if directory.is_dir() else ()
                    for path in paths:
                        self._process_prefetch_request_file(path)
                    for expired in callbacks.tickets.expire():
                        self._append_ticket(expired)
                        self._demote_prefetch_keys(expired.prefetch_id)
                    self._reap_terminal_prefetches(callbacks)
                    self._cold_reap_pass(callbacks)
                    self._release_consumed_prefetches()
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
        self._release_prefetch_lease(prefetch_id)
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

    def _release_prefetch_lease(self, prefetch_id: str) -> None:
        lease_id = self._prefetch_lease_by_id.pop(prefetch_id, None)
        if not lease_id:
            return
        host = installed_runtime_control_host()
        guard = getattr(host, "prefetch_guard", None) if host is not None else None
        if guard is not None:
            guard.complete(lease_id)

    def _note_reap_reference(
        self,
        *,
        physical: PhysicalKVObject,
        native_keys: Iterable[Any],
        hit_tokens: int,
        now_ns: int,
    ) -> None:
        state = self._reap_state.setdefault(physical.physical_object_id, {
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "prefix_hash": physical.compatibility_key.prefix_hash,
            "request_count": 0,
            "reuse_count": 0,
            "ref_count": 0,
            "last_reference_ns": 0,
            "last_release_ns": None,
            "native_keys": [],
        })
        state["request_count"] = int(state.get("request_count") or 0) + 1
        if int(hit_tokens) > 0:
            state["reuse_count"] = int(state.get("reuse_count") or 0) + 1
        state["ref_count"] = int(state.get("ref_count") or 0) + 1
        state["last_reference_ns"] = int(now_ns)
        state["last_release_ns"] = None
        state["native_keys"] = [key for key in native_keys if key is not None]

    def _note_reap_release(self, physical: PhysicalKVObject | None, now_ns: int) -> None:
        if physical is None:
            return
        state = self._reap_state.get(physical.physical_object_id)
        if state is None:
            return
        state["ref_count"] = max(0, int(state.get("ref_count") or 0) - 1)
        if int(state.get("ref_count") or 0) == 0:
            state["last_release_ns"] = int(now_ns)

    def _release_cpu_staging_after_consume(self, native_request_id: str) -> None:
        """Demote a consumed CPU staging copy back out of LocalCPUBackend.

        A prefetched CPU copy exists only to stage the load; once the native
        request has consumed it, keeping it resident accumulates unified-memory
        peak and exhausts the CPU prefetch budget for later prefixes.  Release
        respects LMCache ref-count ownership via non-force ``remove`` and skips
        pinned/missing chunks.
        """
        if not _env_flag("ASTRAKV_KV_CORE_RELEASE_CPU_STAGING_ON_CONSUME", True):
            return
        prefetch_id = self._prefetch_by_request.pop(native_request_id, "")
        if not prefetch_id:
            return
        keys = self._prefetch_keys.pop(prefetch_id, ())
        if not keys:
            return
        manager = self._storage_manager()
        if manager is None:
            return
        cpu = getattr(manager, "storage_backends", {}).get("LocalCPUBackend")
        if cpu is None:
            return
        removed = 0
        for key in keys:
            try:
                deleted = bool(cpu.remove(key, force=False))
            except (AttributeError, TypeError, ValueError):
                continue
            if deleted:
                policy = getattr(cpu, "cache_policy", None)
                update = getattr(policy, "update_on_force_evict", None)
                if callable(update):
                    update(key)
                removed += 1
        self._append("kv_core_policy_decisions.jsonl", {
            "request_id": native_request_id,
            "prefetch_id": prefetch_id,
            "action": "release_cpu_staging_after_consume",
            "tier": "cpu",
            "key_count": len(keys),
            "removed_count": removed,
            "status": "completed" if removed else "pinned_or_missing",
            "timestamp_ns": time.time_ns(),
        })

    def _write_consumed_prefetch_marker(
        self,
        *,
        prefetch_id: str,
        target_request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
    ) -> None:
        if self._state_dir is None:
            return
        if not _env_flag("ASTRAKV_KV_CORE_RELEASE_CPU_STAGING_ON_CONSUME", True):
            return
        if not token_ids:
            return
        directory = self._state_dir / "consumed_prefetches"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(prefetch_id.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "schema": "astrakv-consumed-prefetch-v1",
            "prefetch_id": prefetch_id,
            "target_request_id": target_request_id,
            "token_ids": list(token_ids),
            "request_configs": request_configs,
            "consumed_at_ns": time.time_ns(),
        }, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _release_consumed_prefetches(self) -> None:
        """Cross-process release of consumed CPU staging copies.

        Consumption is recorded in the scheduler-side bridge while the CPU
        backend lives in the worker-side bridge.  A state-dir marker carrying
        the exact token prefix bridges the two; the worker re-derives the
        native keys from its own connector, so release does not depend on any
        in-memory ``_prefetch_keys`` state.  Release only runs after the target
        request's terminal marker exists, so the native load has already read
        the CPU copy.
        """
        if self._state_dir is None:
            return
        if not _env_flag("ASTRAKV_KV_CORE_RELEASE_CPU_STAGING_ON_CONSUME", True):
            return
        manager = self._storage_manager()
        if manager is None:
            return
        cpu = getattr(manager, "storage_backends", {}).get("LocalCPUBackend")
        if cpu is None:
            return
        directory = self._state_dir / "consumed_prefetches"
        terminals = self._state_dir / "request_terminals"
        paths = tuple(directory.glob("*.json")) if directory.is_dir() else ()
        for path in paths:
            if path.name in self._consumed_prefetch_seen:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                prefetch_id = str(payload["prefetch_id"])
                target_request_id = str(payload["target_request_id"] or "")
                token_ids = tuple(int(token) for token in payload.get("token_ids") or ())
                request_configs = payload.get("request_configs")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                self._consumed_prefetch_seen.add(path.name)
                continue
            if not target_request_id or not token_ids:
                self._consumed_prefetch_seen.add(path.name)
                continue
            terminal_digest = hashlib.sha256(target_request_id.encode("utf-8")).hexdigest()
            if not (terminals / f"{terminal_digest}.json").exists():
                continue  # target still running; wait for the terminal marker
            self._consumed_prefetch_seen.add(path.name)
            try:
                chunks = self._token_chunks(self._connector, token_ids, request_configs)
            except (TypeError, ValueError):
                continue
            keys = tuple(key for _start, _end, key in chunks)
            if not keys:
                continue
            removed = 0
            for key in keys:
                try:
                    deleted = bool(cpu.remove(key, force=False))
                except (AttributeError, TypeError, ValueError):
                    continue
                if deleted:
                    policy = getattr(cpu, "cache_policy", None)
                    update = getattr(policy, "update_on_force_evict", None)
                    if callable(update):
                        update(key)
                    removed += 1
            self._append("kv_core_policy_decisions.jsonl", {
                "request_id": target_request_id,
                "prefetch_id": prefetch_id,
                "action": "release_cpu_staging_after_consume",
                "tier": "cpu",
                "key_count": len(keys),
                "removed_count": removed,
                "status": "completed" if removed else "pinned_or_missing",
                "timestamp_ns": time.time_ns(),
            })

    @staticmethod
    def _cpu_bytes(cpu: Any, keys: Iterable[Any]) -> int:
        if cpu is None:
            return 0
        total = 0
        lock = getattr(cpu, "cpu_lock", None)
        context = lock if lock is not None else _NullContext()
        with context:
            hot = getattr(cpu, "hot_cache", {}) or {}
            for key in keys:
                obj = hot.get(key)
                if obj is None:
                    continue
                size = getattr(obj, "get_physical_size", None)
                total += max(0, int(size())) if callable(size) else 0
        return total

    def _cold_reap_pass(self, callbacks: KVCoreConnectorCallbacks) -> None:
        """Reap cold external copies (E5).  Fail-closed and owner-safe."""
        if callbacks is None or callbacks.mode is not RuntimeMode.ACTIVE:
            return
        if not _env_flag("ASTRAKV_KV_CORE_COLD_REAP_ENABLED"):
            return
        manager = self._storage_manager()
        if manager is None:
            return
        backends = getattr(manager, "storage_backends", {}) or {}
        cpu = backends.get("LocalCPUBackend")
        disk = backends.get("LocalDiskBackend")
        if cpu is None and disk is None:
            return
        threshold = _env_float("ASTRAKV_KV_CORE_COLD_REAP_REUSE_THRESHOLD", 0.2)
        idle_ms = _env_int("ASTRAKV_KV_CORE_COLD_REAP_IDLE_MS", 30000)
        now = time.time_ns()
        pending = tuple(
            callbacks.tickets.snapshot(statuses=(PrefetchStatus.SUBMITTED, PrefetchStatus.COMPLETED))
        )
        pending_object_ids = {ticket.physical_object_id for ticket in pending}
        run_id = os.environ.get("ASTRAKV_RUNTIME_CONTROL_RUN_ID", "")
        for object_id, state in list(self._reap_state.items()):
            if int(state.get("ref_count") or 0) > 0:
                continue
            if object_id in pending_object_ids:
                continue
            last_release_ns = state.get("last_release_ns")
            if last_release_ns is None:
                continue
            if (int(now) - int(last_release_ns)) / 1_000_000.0 < idle_ms:
                continue
            request_count = max(1, int(state.get("request_count") or 0))
            if (int(state.get("reuse_count") or 0) / request_count) >= threshold:
                continue
            keys = tuple(state.get("native_keys") or ())
            if not keys:
                continue
            if not callable(getattr(manager, "remove", None)):
                self._append("kv_core_external_reaps.jsonl", {
                    "schema": "astrakv-kv-core-external-reap-v1",
                    "run_id": run_id,
                    "physical_object_id": object_id,
                    "binding_generation": int(state.get("binding_generation") or 0),
                    "prefix_hash": str(state.get("prefix_hash") or ""),
                    "source_tier": "cpu",
                    "target_tier": "none",
                    "freed_bytes": 0,
                    "reason": "cold_reuse_below_threshold",
                    "status": "delegated_to_lmcache_eviction",
                    "demoted_keys": 0,
                    "invalidated_keys": 0,
                    "failed_keys": 0,
                    "timestamp_ns": now,
                })
                state["request_count"] = 0
                state["reuse_count"] = 0
                continue
            freed_cpu = self._cpu_bytes(cpu, keys)
            freed_disk = self._disk_bytes(disk, keys)
            demoted = 0
            invalidated = 0
            failed = 0
            with self._lock:
                for key in keys:
                    if cpu is not None and bool(getattr(cpu, "contains", lambda *_: False)(key, False)):
                        try:
                            removed = bool(manager.remove(key, ["LocalCPUBackend"]))
                        except Exception:
                            failed += 1
                        else:
                            if removed:
                                demoted += 1
                    if disk is not None and bool(getattr(disk, "contains", lambda *_: False)(key, False)):
                        try:
                            removed = bool(manager.remove(key, ["LocalDiskBackend"]))
                        except Exception:
                            failed += 1
                        else:
                            if removed:
                                invalidated += 1
            status = (
                "invalidated"
                if invalidated > 0
                else ("demoted" if demoted > 0 else ("failed" if failed > 0 else "not_found"))
            )
            self._append("kv_core_external_reaps.jsonl", {
                "schema": "astrakv-kv-core-external-reap-v1",
                "run_id": run_id,
                "physical_object_id": object_id,
                "binding_generation": int(state.get("binding_generation") or 0),
                "prefix_hash": str(state.get("prefix_hash") or ""),
                "source_tier": "cpu",
                "target_tier": "none" if invalidated > 0 else "ssd",
                "freed_bytes": freed_cpu + freed_disk,
                "reason": "cold_reuse_below_threshold",
                "status": status,
                "demoted_keys": demoted,
                "invalidated_keys": invalidated,
                "failed_keys": failed,
                "timestamp_ns": now,
            })
            state["request_count"] = 0
            state["reuse_count"] = 0

    def _write_prefetch_request(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
        *,
        promote: bool,
        invalidate_disk_backed_cpu: bool = False,
        prefetch_origin: str = "prefetch_a",
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
            "invalidate_disk_backed_cpu": bool(invalidate_disk_backed_cpu),
            "prefetch_origin": str(prefetch_origin or "prefetch_a"),
            "created_at_ns": now,
            "expires_at_ns": now + _env_int(
                "ASTRAKV_KV_CORE_PREFETCH_TTL_NS", 30_000_000_000,
            ),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _invalidate_disk_backed_cpu_prefix(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        request_configs: dict[str, Any] | None,
        reason: str,
    ) -> None:
        """Remove only unpinned target chunks with a confirmed SSD copy.

        ``LocalCPUBackend.clear()`` is intentionally not used: it can erase
        unrelated or CPU-only objects.  This version-locked operation first
        verifies each exact native chunk in ``LocalDiskBackend``, then removes
        only its matching, unpinned CPU-hot copy under LMCache's CPU lock.
        CPU-only, absent, and pinned chunks remain resident and cannot be
        misrepresented as SSD-prefetch candidates.
        """
        manager = self._storage_manager()
        backends = {} if manager is None else getattr(manager, "storage_backends", {})
        cpu, disk = backends.get("LocalCPUBackend"), backends.get("LocalDiskBackend")
        chunks = self._token_chunks(self._connector, token_ids, request_configs)
        physical, _native_keys = self._physical(self._connector, token_ids, request_configs)
        callbacks = self._callbacks()
        if (
            physical is not None
            and callbacks is not None
            and callbacks.tickets.has_open_for_physical(
                physical.physical_object_id, physical.binding_generation,
            )
        ):
            self._append("kv_core_policy_decisions.jsonl", {
                "request_id": request_id,
                "action": "invalidate_external_copy",
                "tier": "cpu",
                "reason": reason,
                "status": "skipped_open_prefetch_ticket",
                "physical_object_id": physical.physical_object_id,
                "binding_generation": physical.binding_generation,
                "timestamp_ns": time.time_ns(),
            })
            return
        disk_backed_keys: list[Any] = []
        cpu_only_chunks = 0
        for _start, _end, key in chunks:
            try:
                disk_backed = bool(disk is not None and disk.contains(key, False))
            except Exception:
                disk_backed = False
            if disk_backed:
                disk_backed_keys.append(key)
            else:
                try:
                    cpu_only_chunks += int(cpu is not None and cpu.contains(key, False))
                except Exception:
                    pass
        removed = 0
        pinned = 0
        cpu_missing = 0
        lock = getattr(cpu, "cpu_lock", None)
        hot_cache = getattr(cpu, "hot_cache", None)
        if (
            cpu is not None and bool(getattr(cpu, "use_hot", False))
            and lock is not None and callable(getattr(hot_cache, "get", None))
        ):
            with lock:
                for key in disk_backed_keys:
                    memory_obj = hot_cache.get(key)
                    if memory_obj is None:
                        cpu_missing += 1
                        continue
                    if bool(getattr(memory_obj, "is_pinned", False)):
                        pinned += 1
                        continue
                    try:
                        deleted = bool(cpu.remove(key, force=False))
                        if deleted:
                            policy = getattr(cpu, "cache_policy", None)
                            update = getattr(policy, "update_on_force_evict", None)
                            if callable(update):
                                update(key)
                            removed += 1
                    except (AttributeError, TypeError, ValueError):
                        continue
        self._append("kv_core_policy_decisions.jsonl", {
            "request_id": request_id,
            "action": "invalidate_external_copy",
            "tier": "cpu",
            "reason": reason,
            "physical_object_id": "" if physical is None else physical.physical_object_id,
            "binding_generation": 0 if physical is None else physical.binding_generation,
            "native_key": "" if physical is None else physical.native_key,
            "compatibility_identity": "" if physical is None else physical.compatibility_key.identity,
            "prefix_hash": "" if physical is None else physical.compatibility_key.prefix_hash,
            "requested_chunk_count": len(chunks),
            "disk_backed_chunk_count": len(disk_backed_keys),
            "cpu_only_chunk_count": cpu_only_chunks,
            "cpu_removed_chunk_count": removed,
            "pinned_cpu_chunk_count": pinned,
            "cpu_missing_chunk_count": cpu_missing,
            "timestamp_ns": time.time_ns(),
        })

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
        token_prefix: tuple[int, ...] = (),
        request_configs: dict[str, Any] | None = None,
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
                self._write_consumed_prefetch_marker(
                    prefetch_id=ticket.prefetch_id,
                    target_request_id=logical_request_id,
                    token_ids=tuple(int(token) for token in token_prefix),
                    request_configs=request_configs,
                )
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
            cpu_prefetch_budget_fraction=_env_float(
                "ASTRAKV_KV_CORE_CPU_PREFETCH_BUDGET_FRACTION",
                float(base.cpu_prefetch_budget_fraction),
            ),
        )

    def _logical_request_id(self, native_request_id: str) -> str:
        host = installed_runtime_control_host()
        identity = None if host is None else host.runtime_identity_for(native_request_id)
        return native_request_id if identity is None else identity.request_id

    def _associate_runtime_request(self, native_request_id: str) -> Any | None:
        """Bind a real LMCache ReqMeta ID through the runtime-owned host."""
        native_request_id = str(native_request_id or "")
        if not native_request_id:
            return None
        cached = self._association_receipts.get(native_request_id)
        if cached is not None:
            return cached
        host = installed_runtime_control_host()
        if host is None:
            return None
        associate = getattr(host, "associate_runtime_request", None)
        if not callable(associate):
            return None
        try:
            receipt = associate(native_request_id)
        except (TypeError, ValueError):
            return None
        if receipt is not None and str(getattr(receipt, "status", "") or "") == "associated":
            self._association_receipts[native_request_id] = receipt
            return receipt
        return None

    def _association_evidence(self, native_request_id: str, receipt: Any | None = None) -> dict[str, str]:
        receipt = receipt or self._association_receipts.get(str(native_request_id))
        runtime_event_id = "" if receipt is None else str(getattr(receipt, "runtime_event_id", "") or "")
        return {
            "native_request_id": str(native_request_id),
            "logical_request_id": self._logical_request_id(native_request_id),
            "runtime_event_id": runtime_event_id,
            "association_receipt_reference": runtime_event_id,
        }

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
        cgroup, cgroup_status, cgroup_path = current_cgroup_memory_evidence()
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
            "cgroup_memory_status": cgroup_status,
            "cgroup_memory_current_path": cgroup_path,
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
        identity = self._association_evidence(request_id)
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
                **identity,
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
            **identity,
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
            "disk_backed_cpu_invalidation_on_prefetch_lead": _env_flag(
                "ASTRAKV_KV_CORE_INVALIDATE_DISK_BACKED_CPU_ON_PREFETCH_LEAD"
            ),
            "vllm_kv_block_budget": getattr(cache_config, "num_gpu_blocks", None),
            "vllm_block_size_tokens": getattr(connector, "_block_size", None),
            "lmcache_chunk_size_tokens": getattr(connector, "_lmcache_chunk_size", None),
            "vendor_patch": True,
            "legacy_owner_load_enabled": False,
            "equivalence_test_enabled": _env_flag("ASTRAKV_KV_CORE_EQUIVALENCE_TEST"),
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
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Concurrent async callbacks and teardown can race on the leaf
            # directory; the append below still surfaces any real problem.
            pass
        with (self._state_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


__all__ = ["VendorCallbackBridge"]
