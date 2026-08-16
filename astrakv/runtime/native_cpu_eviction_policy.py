"""LMCache 0.4.7 CPU-tier victim selection for the E11 native-policy A/B.

The wrapper deliberately leaves LMCache's cache mapping and physical removal
path intact.  In ``lru`` mode it delegates victim selection byte-for-byte to
the installed policy.  In ``astrakv`` mode it ranks *the same currently
evictable CPU entries* with request-scoped reuse evidence, then falls back to
the delegate for candidates without usable evidence.

No file I/O or controller call runs while LMCache holds ``cpu_lock``.  Native
selection/removal evidence is handed to a daemon emitter through an unbounded
in-memory queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
import types
from typing import Any, Callable, MutableMapping

from astrakv.runtime.backend_binding_registry import _canonical_key


EventSink = Callable[[dict[str, Any]], None]
SUPPORTED_NATIVE_CPU_POLICIES = frozenset({"disabled", "lru", "astrakv"})


def normalize_native_cpu_policy(value: str | None) -> str:
    policy = str(value or "disabled").strip().lower()
    if policy not in SUPPORTED_NATIVE_CPU_POLICIES:
        choices = ", ".join(sorted(SUPPORTED_NATIVE_CPU_POLICIES))
        raise RuntimeError(f"unsupported ASTRAKV_E11_CPU_EVICTION_POLICY={policy!r}; expected one of {choices}")
    return policy


def _bounded_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return min(1.0, max(0.0, number))


def _non_negative_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


class _AsyncEventEmitter:
    """Best-effort evidence export without I/O in LMCache's capacity lock."""

    def __init__(self, event_sink: EventSink):
        self._event_sink = event_sink
        self._records: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run,
            name="astrakv-native-eviction-evidence",
            daemon=True,
        )
        self._thread.start()

    def emit(self, record: dict[str, Any]) -> None:
        self._records.put(record)

    def _run(self) -> None:
        while True:
            record = self._records.get()
            try:
                self._event_sink(record)
            except Exception:
                # Evidence export must never change cache correctness or the
                # victim chosen by the native capacity path.
                continue


@dataclass(slots=True)
class NativeKeyScore:
    logical_object_key: str = ""
    request_id: str = ""
    reuse_ratio: float | None = None
    runtime_confidence: float | None = None
    prefetch_waste_count: int = 0
    store_count: int = 0
    hit_count: int = 0
    last_observed_ns: int = 0

    def score(self) -> tuple[float, dict[str, Any]] | None:
        # Workload/runtime reuse hints are the primary MVP signal.  A physical
        # hit is authoritative positive evidence and can only make an object
        # hotter, never colder.
        empirical_reuse = self.hit_count / max(1, self.hit_count + self.store_count)
        if self.reuse_ratio is None and self.hit_count == 0:
            return None
        policy_reuse = max(self.reuse_ratio or 0.0, empirical_reuse)
        confidence = self.runtime_confidence
        if confidence is None:
            confidence = min(1.0, (self.store_count + self.hit_count) / 3.0)
        waste_signal = min(1.0, float(self.prefetch_waste_count))
        cold_score = round(
            (0.70 * (1.0 - policy_reuse))
            + (0.20 * (1.0 - confidence))
            + (0.10 * waste_signal),
            6,
        )
        return cold_score, {
            "policy_reuse": round(policy_reuse, 6),
            "reuse_ratio_hint": self.reuse_ratio,
            "empirical_reuse": round(empirical_reuse, 6),
            "runtime_confidence": round(confidence, 6),
            "prefetch_waste_count": self.prefetch_waste_count,
            "store_count": self.store_count,
            "hit_count": self.hit_count,
            "logical_object_key": self.logical_object_key,
            "request_id": self.request_id,
        }


@dataclass(slots=True)
class NativeCPUScoreRegistry:
    _scores: dict[str, NativeKeyScore] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(
        self,
        key: Any,
        *,
        metadata: dict[str, Any] | None = None,
        action: str = "",
        status: str = "",
        logical_object_key: str = "",
        request_id: str = "",
    ) -> None:
        values = dict(metadata or {})
        identity = _canonical_key(key)
        with self._lock:
            score = self._scores.setdefault(identity, NativeKeyScore())
            score.logical_object_key = str(logical_object_key or score.logical_object_key)
            score.request_id = str(request_id or score.request_id)
            reuse_ratio = _bounded_float(
                values.get("e11_policy_reuse_ratio", values.get("reuse_ratio"))
            )
            if reuse_ratio is not None:
                score.reuse_ratio = reuse_ratio
            confidence = _bounded_float(values.get("runtime_confidence"))
            if confidence is not None:
                score.runtime_confidence = confidence
            waste = _non_negative_int(
                values.get("prefetch_waste_count", values.get("prefetch_waste"))
            )
            if waste is not None:
                score.prefetch_waste_count = waste
            if action in {"cache_store", "store"} and status in {
                "submitted", "completed", "ok", "executed",
            }:
                score.store_count += 1
            if action in {"cache_hit", "hit"} and status in {
                "completed", "ok", "executed",
            }:
                score.hit_count += 1
            score.last_observed_ns = time.time_ns()

    def record_store(self, key: Any) -> None:
        self.observe(key, action="store", status="completed")

    def record_hit(self, key: Any) -> None:
        self.observe(key, action="hit", status="completed")

    def score_for(self, key: Any) -> tuple[float, dict[str, Any]] | None:
        identity = _canonical_key(key)
        with self._lock:
            score = self._scores.get(identity)
            return None if score is None else score.score()

    def forget(self, key: Any) -> None:
        with self._lock:
            self._scores.pop(_canonical_key(key), None)


@dataclass(slots=True)
class _PendingSelection:
    selection_id: str
    record: dict[str, Any]


class NativeCPUCachePolicy:
    """Tier-aware wrapper around the already-constructed LMCache CPU policy."""

    def __init__(
        self,
        delegate: Any,
        *,
        mode: str,
        run_id: str,
        event_sink: EventSink,
        score_registry: NativeCPUScoreRegistry | None = None,
    ) -> None:
        self.delegate = delegate
        self.mode = normalize_native_cpu_policy(mode)
        if self.mode == "disabled":
            raise ValueError("disabled policy must not be installed")
        self.run_id = str(run_id)
        self.score_registry = score_registry or NativeCPUScoreRegistry()
        self._emitter = _AsyncEventEmitter(event_sink)
        self._selection_index = 0
        self._pending: dict[str, deque[_PendingSelection]] = {}

    def init_mutable_mapping(self) -> MutableMapping[Any, Any]:
        return self.delegate.init_mutable_mapping()

    def update_on_hit(self, key: Any, cache_dict: MutableMapping[Any, Any]) -> None:
        self.delegate.update_on_hit(key, cache_dict)
        self.score_registry.record_hit(key)

    def update_on_put(self, key: Any) -> None:
        self.delegate.update_on_put(key)
        self.score_registry.record_store(key)

    def update_on_force_evict(self, key: Any) -> None:
        self.delegate.update_on_force_evict(key)
        # LMCache 0.4.7's batched CPU allocator invokes this immediately
        # before popping the key.  The supported-version lock makes that a
        # terminal native capacity-removal callback for this path.
        self.record_removed(key, source="update_on_force_evict")

    def get_evict_candidates(
        self,
        cache_dict: MutableMapping[Any, Any],
        num_candidates: int = 1,
    ) -> list[Any]:
        selection_started_ns = time.perf_counter_ns()
        requested = max(0, int(num_candidates))
        if requested == 0:
            return []
        delegate_duration_ns = 0
        policy_scoring_duration_ns = 0
        candidate_scan_count = 0
        scored_candidate_count = 0
        fallback_count = 0

        delegate_started_ns = time.perf_counter_ns()
        lru_candidates = list(
            self.delegate.get_evict_candidates(cache_dict, num_candidates=requested)
        )
        delegate_duration_ns += time.perf_counter_ns() - delegate_started_ns
        if self.mode == "lru":
            selected = lru_candidates
            # Scores are observational in the baseline arm: recording them
            # enables apples-to-apples quality analysis but cannot influence
            # the delegate's exact victim order.
            scored = {}
            scoring_started_ns = time.perf_counter_ns()
            for key in selected:
                candidate_scan_count += 1
                score = self.score_registry.score_for(key)
                if score is not None:
                    scored[_canonical_key(key)] = score
            policy_scoring_duration_ns = time.perf_counter_ns() - scoring_started_ns
            scored_candidate_count = len(scored)
        else:
            scoring_started_ns = time.perf_counter_ns()
            ranked: list[tuple[float, int, str, Any, dict[str, Any]]] = []
            for lru_index, (key, cache) in enumerate(cache_dict.items()):
                if not bool(getattr(cache, "can_evict", False)):
                    continue
                candidate_scan_count += 1
                score = self.score_registry.score_for(key)
                if score is None:
                    continue
                cold_score, signals = score
                ranked.append(
                    (-cold_score, lru_index, _canonical_key(key), key, signals)
                )
            ranked.sort(key=lambda row: (row[0], row[1], row[2]))
            selected = [row[3] for row in ranked[:requested]]
            scored = {
                _canonical_key(row[3]): (-row[0], row[4]) for row in ranked[:requested]
            }
            scored_candidate_count = len(ranked)
            policy_scoring_duration_ns = time.perf_counter_ns() - scoring_started_ns
            # Unknown/unscored entries retain native LRU semantics and fill
            # any remaining slots.  This also guarantees forward progress.
            selected_ids = {_canonical_key(key) for key in selected}
            if len(selected) < requested:
                delegate_started_ns = time.perf_counter_ns()
                fallback = list(
                    self.delegate.get_evict_candidates(
                        cache_dict, num_candidates=max(requested, len(cache_dict))
                    )
                )
                delegate_duration_ns += time.perf_counter_ns() - delegate_started_ns
                for key in fallback:
                    identity = _canonical_key(key)
                    if identity in selected_ids:
                        continue
                    selected.append(key)
                    selected_ids.add(identity)
                    fallback_count += 1
                    if len(selected) == requested:
                        break

        selection_duration_ns = time.perf_counter_ns() - selection_started_ns
        timestamp_ns = time.time_ns()
        for key in selected:
            self._selection_index += 1
            identity = _canonical_key(key)
            scored_value = scored.get(identity)
            record = {
                "schema": "astrakv-native-cache-policy-eviction-v1",
                "record_type": "native_cache_policy_eviction",
                "run_id": self.run_id,
                "selection_id": f"native-cpu-{timestamp_ns}-{self._selection_index}",
                "timestamp_ns": timestamp_ns,
                "status": "selected",
                "tier_before": "cpu",
                "tier_after": "unknown",
                "requested_policy": self.mode,
                "effective_policy": "lmcache_lru" if self.mode == "lru" else "astrakv_native_cpu",
                "delegate_policy_class": type(self.delegate).__name__,
                "backend_key_identity": identity,
                "requested_candidate_count": requested,
                "returned_candidate_count": len(selected),
                "candidate_scan_count": candidate_scan_count,
                "scored_candidate_count": scored_candidate_count,
                "fallback_candidate_count": fallback_count,
                "delegate_duration_ns": delegate_duration_ns,
                "policy_scoring_duration_ns": policy_scoring_duration_ns,
                "selection_duration_ns": selection_duration_ns,
                "score_source": "native_lru" if self.mode == "lru" else (
                    "astrakv_score" if scored_value is not None else "native_lru_fallback"
                ),
                "cold_score": None if scored_value is None else round(scored_value[0], 6),
                "signals": {} if scored_value is None else dict(scored_value[1]),
            }
            self._pending.setdefault(identity, deque()).append(
                _PendingSelection(record["selection_id"], record)
            )
            self._emitter.emit(record)
        return selected

    def record_removed(self, key: Any, *, source: str) -> bool:
        identity = _canonical_key(key)
        pending = self._pending.get(identity)
        if not pending:
            return False
        selection = pending.popleft()
        if not pending:
            self._pending.pop(identity, None)
        self._emitter.emit({
            **selection.record,
            "timestamp_ns": time.time_ns(),
            "status": "completed",
            "tier_after": "unknown",
            "terminal_condition": source,
            "removed": 1,
        })
        self.score_registry.forget(key)
        return True


def install_native_cpu_eviction_policy(
    manager: Any,
    *,
    mode: str,
    run_id: str,
    event_sink: EventSink,
) -> NativeCPUCachePolicy | None:
    """Install the E11 selector on CPU only and emit an auditable manifest."""
    normalized = normalize_native_cpu_policy(mode)
    if normalized == "disabled":
        return None
    backends = getattr(manager, "storage_backends", None)
    if not isinstance(backends, dict):
        raise RuntimeError("native CPU eviction policy requires storage_backends")
    cpu_backend = backends.get("LocalCPUBackend")
    if cpu_backend is None:
        raise RuntimeError("native CPU eviction policy requires LocalCPUBackend")
    if not bool(getattr(cpu_backend, "use_hot", False)):
        raise RuntimeError("native CPU eviction policy requires LocalCPUBackend.use_hot=true")
    delegate = getattr(cpu_backend, "cache_policy", None)
    required = (
        "init_mutable_mapping", "update_on_hit", "update_on_put",
        "update_on_force_evict", "get_evict_candidates",
    )
    missing = [name for name in required if not callable(getattr(delegate, name, None))]
    if missing:
        raise RuntimeError("native CPU eviction delegate is incompatible: " + ",".join(missing))
    if isinstance(delegate, NativeCPUCachePolicy):
        if delegate.mode != normalized:
            raise RuntimeError(
                f"native CPU eviction policy already installed as {delegate.mode}, requested {normalized}"
            )
        return delegate

    wrapper = NativeCPUCachePolicy(
        delegate,
        mode=normalized,
        run_id=run_id,
        event_sink=event_sink,
    )
    cpu_backend.cache_policy = wrapper

    original_remove = cpu_backend.remove
    if getattr(original_remove, "__astrakv_native_cpu_remove_patch__", False):
        raise RuntimeError("native CPU remove completion hook is already installed")

    def remove_wrapper(
        backend_self: Any,
        key: Any,
        force: bool = True,
        *,
        _original: Callable[..., Any] = original_remove,
    ) -> bool:
        removed = bool(_original(key, force=force))
        if removed:
            wrapper.record_removed(
                key,
                source="local_cpu_backend.remove:force=" + str(bool(force)).lower(),
            )
        return removed

    remove_wrapper.__astrakv_native_cpu_remove_patch__ = True
    cpu_backend.remove = types.MethodType(remove_wrapper, cpu_backend)

    disk_backend = backends.get("LocalDiskBackend")
    disk_policy = getattr(disk_backend, "cache_policy", None) if disk_backend is not None else None
    wrapper._emitter.emit({
        "schema": "astrakv-native-policy-installation-v1",
        "record_type": "native_policy_installation",
        "run_id": str(run_id),
        "timestamp_ns": time.time_ns(),
        "status": "installed",
        "cpu_requested_policy": normalized,
        "cpu_effective_policy": "lmcache_lru" if normalized == "lru" else "astrakv_native_cpu",
        "cpu_wrapper_class": type(wrapper).__name__,
        "cpu_delegate_policy_class": type(delegate).__name__,
        "cpu_backend_class": type(cpu_backend).__name__,
        "cpu_same_native_capacity_path": True,
        "ssd_policy_unchanged": True,
        "ssd_effective_policy_class": type(disk_policy).__name__ if disk_policy is not None else None,
        "ssd_backend_class": type(disk_backend).__name__ if disk_backend is not None else None,
    })
    return wrapper
