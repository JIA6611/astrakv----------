"""Persisted circuit breaker for execution-plane fault containment."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


CIRCUIT_BREAKER_SCHEMA = "astrakv-circuit-breaker-v1"


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    failure_threshold: int = 3
    timeout_threshold: int = 2
    pressure_threshold: int = 2
    cooldown_ns: int = 30_000_000_000

    def __post_init__(self) -> None:
        if min(self.failure_threshold, self.timeout_threshold, self.pressure_threshold, self.cooldown_ns) <= 0:
            raise ValueError("circuit breaker thresholds and cooldown must be positive")


class CircuitBreaker:
    """Open on unsafe conditions; closing always requires an explicit health proof."""

    def __init__(self, policy: CircuitBreakerPolicy | None = None, *, state_path: Path | str | None = None) -> None:
        self.policy = policy or CircuitBreakerPolicy()
        self.state_path = None if state_path is None else Path(state_path)
        self._lock = threading.RLock()
        self.state = "closed"
        self.opened_at_ns: int | None = None
        self.failures = 0
        self.timeouts = 0
        self.pressures = 0
        self.health_restored = False
        if self.state_path is not None and self.state_path.exists():
            self._restore()

    def allow_dispatch(self, *, now_ns: int) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open" and self.opened_at_ns is not None and now_ns - self.opened_at_ns >= self.policy.cooldown_ns:
                self.state = "half_open"
                self._persist()
            return self.state == "closed"

    def record_failure(self, *, now_ns: int) -> None:
        self._record("failures", self.policy.failure_threshold, now_ns)

    def record_timeout(self, *, now_ns: int) -> None:
        self._record("timeouts", self.policy.timeout_threshold, now_ns)

    def record_pressure(self, *, now_ns: int) -> None:
        self._record("pressures", self.policy.pressure_threshold, now_ns)

    def restore_health(self, *, now_ns: int) -> None:
        with self._lock:
            if self.state == "open" and (self.opened_at_ns is None or now_ns - self.opened_at_ns < self.policy.cooldown_ns):
                raise ValueError("circuit breaker cooldown has not elapsed")
            self.state = "closed"
            self.opened_at_ns = None
            self.failures = self.timeouts = self.pressures = 0
            self.health_restored = True
            self._persist()

    def _record(self, counter: str, threshold: int, now_ns: int) -> None:
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)
            if getattr(self, counter) >= threshold:
                self.state = "open"
                self.opened_at_ns = now_ns
                self.health_restored = False
            self._persist()

    def _restore(self) -> None:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != CIRCUIT_BREAKER_SCHEMA or payload.get("policy") != asdict(self.policy):
            raise ValueError("circuit breaker checkpoint does not match policy")
        self.state = payload["state"]
        self.opened_at_ns = payload.get("opened_at_ns")
        self.failures = int(payload.get("failures", 0))
        self.timeouts = int(payload.get("timeouts", 0))
        self.pressures = int(payload.get("pressures", 0))
        self.health_restored = bool(payload.get("health_restored", False))

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": CIRCUIT_BREAKER_SCHEMA, "policy": asdict(self.policy), "state": self.state,
                   "opened_at_ns": self.opened_at_ns, "failures": self.failures, "timeouts": self.timeouts,
                   "pressures": self.pressures, "health_restored": self.health_restored}
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.state_path)
