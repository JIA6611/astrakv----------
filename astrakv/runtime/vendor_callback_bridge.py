"""Native lifecycle recorder used only by the version-locked LMCache patch.

This module is intentionally tensor-agnostic except for observing the result
returned by LMCache's own ``retrieve`` call.  It never calls ``retrieve``,
constructs slot mappings, or changes a scheduler result.  The vendor patch
invokes it after each native lifecycle point.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import (
    KVCompatibilityKey,
    PhysicalKVObject,
    RequestKVIntent,
    exact_token_prefix_hash,
)
from astrakv.runtime.lmcache047_bootstrap import (
    install_from_environment,
    installed_kv_core_callbacks,
)
from astrakv.runtime.third_party_patch import PATCH_ID, REQUIRED_CALLBACKS


class VendorCallbackBridge:
    """Records only events emitted by the patched native LMCache connector."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._physical_by_request: dict[str, PhysicalKVObject] = {}
        self._seen_callbacks: set[str] = set()
        state = os.environ.get("ASTRAKV_RUNTIME_CONTROL_STATE_DIR", "")
        self._state_dir = Path(state) if state else None

    @classmethod
    def from_environment(cls) -> "VendorCallbackBridge | None":
        if os.environ.get("ASTRAKV_KV_CORE_VENDOR_PATCH", "false") != "true":
            return None
        # This function is called from the version-locked LMCache adapter,
        # which is constructed only inside EngineCore.  vLLM 0.23 does not
        # consistently expose the older --multiprocessing-fork marker.
        install_from_environment(vendor_engine_child=True)
        if installed_kv_core_callbacks() is None:
            return None
        return cls()

    def scheduler_exact_lookup(
        self,
        connector: Any,
        *,
        request_id: str,
        token_ids: Iterable[int],
        request_configs: dict[str, Any] | None,
        lookup_hit_tokens: int,
        priority: int = 0,
    ) -> None:
        physical = self._physical(connector, request_id, token_ids, request_configs)
        if physical is None:
            return
        self._write_run_metadata(connector)
        callbacks = self._callbacks()
        if callbacks is None:
            return
        try:
            with self._lock:
                if request_id not in self._physical_by_request:
                    requested = len(tuple(token_ids))
                    cap = callbacks.capability.external_token_cap
                    max_external = requested if cap <= 0 else min(requested, cap)
                    callbacks.submit_intent(RequestKVIntent(
                        request_id=request_id,
                        compatibility_key=physical.compatibility_key,
                        physical_object=physical,
                        max_external_tokens=max_external,
                        requested_prefix_tokens=requested,
                        deadline_ns=time.time_ns() + 60_000_000_000,
                        priority=int(priority),
                    ))
                    self._physical_by_request[request_id] = physical
                callbacks.record_scheduler_lookup(
                    request_id=request_id,
                    physical=physical,
                    lookup_hit_tokens=max(0, int(lookup_hit_tokens)),
                    native_request_id=request_id,
                )
                self._record("scheduler_exact_lookup", {
                    "request_id": request_id,
                    "physical_object_id": physical.physical_object_id,
                    "binding_generation": physical.binding_generation,
                    "native_key": physical.native_key,
                    "lookup_hit_tokens": int(lookup_hit_tokens),
                })
                self._append_request_accounting(request_id, physical)
        except (TypeError, ValueError) as exc:
            self._record("scheduler_exact_lookup", {"request_id": request_id, "status": "rejected", "reason": str(exc)})

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
            self._record("scheduler_external_admission", asdict(admission))
            self._append_request_accounting(request_id, physical)
        except (TypeError, ValueError) as exc:
            self._record("scheduler_external_admission", {"request_id": request_id, "status": "rejected", "reason": str(exc)})

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
        native_retrieved_tokens: int,
        native_bytes: int,
        load_latency_ns: int,
        status: str,
    ) -> None:
        callbacks, physical = self._callbacks(), self._physical_by_request.get(request_id)
        if callbacks is None or physical is None:
            return
        try:
            admission = callbacks._admissions.get(request_id)  # Native admission was already observed.
            allocated = 0 if admission is None else admission.allocated_external_tokens
            receipt = callbacks.record_native_load_completion(
                request_id=request_id,
                physical=physical,
                actual_loaded_tokens=min(max(0, int(native_retrieved_tokens)), allocated),
                bytes_loaded=max(0, int(native_bytes)),
                load_latency_ns=max(0, int(load_latency_ns)),
                native_request_id=request_id,
                status=status,
            )
            record = asdict(receipt)
            # Chunk alignment can overwrite vLLM-local overlap.  These fields
            # retain the physical transfer without corrupting scheduler token
            # accounting required by the receipt invariant.
            record["native_retrieved_tokens"] = max(0, int(native_retrieved_tokens))
            record["native_bytes"] = max(0, int(native_bytes))
            self._append("kv_core_native_receipts.jsonl", record)
            self._record("native_load_completion", record)
            self._append_request_accounting(request_id, physical)
        except (TypeError, ValueError) as exc:
            self._record("native_load_completion", {"request_id": request_id, "status": "rejected", "reason": str(exc)})

    def _physical(
        self,
        connector: Any,
        request_id: str,
        token_ids: Iterable[int],
        request_configs: dict[str, Any] | None,
    ) -> PhysicalKVObject | None:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return None
        existing = self._physical_by_request.get(request_id)
        if existing is not None:
            return existing
        engine = getattr(connector, "lmcache_engine", None)
        token_database = getattr(engine, "token_database", None)
        if token_database is None:
            return None
        try:
            keys = [key.to_string() for _start, _end, key in token_database.process_tokens(
                tokens=list(tokens), request_configs=request_configs,
            )]
        except Exception as exc:
            self._record("scheduler_exact_lookup", {"request_id": request_id, "status": "rejected", "reason": f"native_key_error:{type(exc).__name__}"})
            return None
        if not keys:
            return None
        native_key = json.dumps(keys, separators=(",", ":"), ensure_ascii=True)
        first_key = keys[0].split("@", 1)[0]
        config = getattr(connector, "_vllm_config", None)
        transfer = getattr(config, "kv_transfer_config", None)
        engine_id = str(getattr(transfer, "engine_id", "unknown-engine") or "unknown-engine")
        worker_id = str(getattr(getattr(engine, "metadata", None), "worker_id", "worker-0"))
        compatibility = KVCompatibilityKey(
            model_id=first_key,
            model_revision=os.environ.get("ASTRAKV_MODEL_REVISION", "local-qwen3-8b"),
            tokenizer_revision=os.environ.get("ASTRAKV_TOKENIZER_REVISION", "local-qwen3-8b"),
            chat_template_revision=os.environ.get("ASTRAKV_CHAT_TEMPLATE_REVISION", "qwen3-default"),
            dtype=str(getattr(config, "dtype", "bfloat16")),
            rope_config=str(getattr(getattr(config, "model_config", None), "rope_scaling", "default")),
            adapter_namespace=os.environ.get("ASTRAKV_ADAPTER_NAMESPACE", "base"),
            kv_layout=str(getattr(connector, "_block_size", "unknown")),
            block_size_tokens=max(1, int(getattr(connector, "_block_size", 1))),
            chunk_size_tokens=max(1, int(getattr(connector, "_lmcache_chunk_size", 1))),
            layer_group="all-kv-layers",
            prefix_hash=exact_token_prefix_hash(tokens),
            engine_id=engine_id,
            worker_id=worker_id,
        )
        return PhysicalKVObject(
            native_key=native_key,
            physical_object_id=hashlib.sha256(native_key.encode("ascii")).hexdigest(),
            binding_generation=1,
            compatibility_key=compatibility,
            source_tier="ssd" if getattr(getattr(connector, "config", None), "local_disk", None) else "unknown",
        )

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
        self._append("kv_core_native_callbacks.jsonl", {"callback": callback, "timestamp_ns": time.time_ns(), **record})
        self._append_smoke()
        self._append_uma_sample(callback)

    def _append_uma_sample(self, callback: str) -> None:
        """Capture real host-memory evidence at native lifecycle boundaries."""
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
        self._append("uma_resource_samples.jsonl", {
            "timestamp_ns": time.time_ns(),
            "callback": callback,
            "cgroup_memory_current_bytes": cgroup,
            "process_rss_bytes": rss,
        })

    def _append_request_accounting(self, request_id: str, physical: PhysicalKVObject) -> None:
        """Persist request-level truth without manufacturing a load receipt.

        A scheduler decline is already a terminal recompute decision.  An
        admitted request remains non-terminal until the native connector
        reports ``start_load_kv`` completion.  The validator selects the last
        record per request and rejects incomplete admissions.
        """
        callbacks = self._callbacks()
        if callbacks is None:
            return
        intent = callbacks._intents.get(request_id)
        lookup = callbacks._lookups.get(request_id)
        if intent is None or lookup is None:
            return
        admission = callbacks._admissions.get(request_id)
        receipt = callbacks.receipt_for(request_id)
        allocated = 0 if admission is None else admission.allocated_external_tokens
        loaded = 0 if receipt is None else receipt.actual_loaded_tokens
        if receipt is not None:
            terminal_reason = "native_load_completed"
            terminal = True
        elif admission is not None and allocated == 0:
            terminal_reason = "scheduler_declined_recompute"
            terminal = True
        elif admission is not None:
            terminal_reason = "native_load_pending"
            terminal = False
        else:
            terminal_reason = "scheduler_admission_pending"
            terminal = False
        self._append("kv_core_request_accounting.jsonl", {
            "request_id": request_id,
            "physical_object_id": physical.physical_object_id,
            "binding_generation": physical.binding_generation,
            "native_key": physical.native_key,
            "requested_prefix_tokens": intent.requested_prefix_tokens,
            "lookup_hit_tokens": lookup.lookup_hit_tokens,
            "allocated_external_tokens": allocated,
            "actual_loaded_tokens": loaded,
            "recomputed_tokens": max(0, intent.requested_prefix_tokens - loaded),
            "native_retrieved_tokens": 0 if receipt is None else receipt.actual_loaded_tokens,
            "native_bytes": 0 if receipt is None else receipt.bytes_loaded,
            "load_latency_ns": 0 if receipt is None else receipt.load_latency_ns,
            "terminal": terminal,
            "terminal_reason": terminal_reason,
            "timestamp_ns": time.time_ns(),
        })

    def _write_run_metadata(self, connector: Any) -> None:
        """Record runtime-observed topology and vLLM allocation, never a plan.

        The fields are intentionally absent when the installed connector does
        not expose them.  Capacity acceptance then fails closed instead of
        turning a configured budget into claimed physical evidence.
        """
        if self._state_dir is None:
            return
        config = getattr(connector, "config", None)
        vllm_config = getattr(connector, "_vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
        try:
            budget = int(num_gpu_blocks)
        except (TypeError, ValueError):
            budget = 0
        payload = {
            "schema": "astrakv-kv-core-runtime-metadata-v1",
            "topology": os.environ.get("ASTRAKV_KV_CORE_TOPOLOGY", "unknown"),
            "lmcache_local_cpu_enabled": bool(getattr(config, "local_cpu", False)),
            "lmcache_local_disk_enabled": bool(getattr(config, "local_disk", False)),
            "vllm_kv_block_budget": budget if budget > 0 else None,
            "vllm_block_size_tokens": getattr(connector, "_block_size", None),
            "lmcache_chunk_size_tokens": getattr(connector, "_lmcache_chunk_size", None),
            "observed_at_ns": time.time_ns(),
        }
        (self._state_dir / "kv_core_run_metadata.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _append_smoke(self) -> None:
        if self._state_dir is None:
            return
        payload = {
            "patch_id": PATCH_ID,
            "callbacks": list(REQUIRED_CALLBACKS),
            "observed_callbacks": sorted(self._seen_callbacks),
            "passed": set(REQUIRED_CALLBACKS).issubset(self._seen_callbacks),
            "updated_at_ns": time.time_ns(),
        }
        (self._state_dir / "callback-smoke.json").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def _append(self, filename: str, record: dict[str, Any]) -> None:
        if self._state_dir is None:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with (self._state_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["VendorCallbackBridge"]
