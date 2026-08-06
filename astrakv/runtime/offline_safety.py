"""Offline-policy admission gate for runtime action adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrakv.runtime.eviction import OfflineEvictionDecision, RuntimeActionResult


@dataclass(frozen=True, slots=True)
class OfflineSafetyGateResult:
    status: str
    reasons: tuple[str, ...]
    workload_ids: tuple[str, ...]
    aggregate: dict[str, float]
    checks: dict[str, bool]
    evidence: tuple[dict[str, Any], ...] = ()

    @property
    def allowed(self) -> bool:
        return self.status == "accepted"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "astrakv-offline-safety-gate-v1",
            "status": self.status,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "workload_ids": list(self.workload_ids),
            "aggregate": dict(self.aggregate),
            "checks": dict(self.checks),
            "evidence": [dict(item) for item in self.evidence],
        }


class OfflineSafetyGate:
    """Require reproducible offline evidence before dispatching an action."""

    def __init__(self, result: OfflineSafetyGateResult) -> None:
        self.result = result

    @classmethod
    def evaluate(cls, manifests: list[dict[str, Any]]) -> "OfflineSafetyGate":
        reasons: list[str] = []
        evidence: list[dict[str, Any]] = []
        workload_ids: set[str] = set()
        valid_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for manifest in manifests:
            workload_id = str(manifest.get("workload_id") or "")
            evidence.append({
                "workload_id": workload_id,
                "schema": manifest.get("schema"),
                "simulation_status": manifest.get("simulation_status"),
                "profile_source": manifest.get("profile_source"),
                "workload_sha256": manifest.get("workload_sha256"),
            })
            if manifest.get("schema") != "astrakv-offline-eviction-v1":
                reasons.append(f"{workload_id or 'unknown'}: unsupported offline manifest schema")
                continue
            if manifest.get("simulation_status") != "valid":
                reasons.append(f"{workload_id or 'unknown'}: simulation status is not valid")
                continue
            if not workload_id or not manifest.get("workload_sha256") or not manifest.get("trace_sha256") or not manifest.get("profile_db_sha256"):
                reasons.append(f"{workload_id or 'unknown'}: required input hashes are missing")
                continue
            capacities = manifest.get("capacities")
            if not isinstance(capacities, dict) or any(name not in capacities for name in ("gpu_bytes", "cpu_bytes", "ssd_bytes")):
                reasons.append(f"{workload_id}: explicit tier capacities are missing")
                continue
            if manifest.get("self_profile_leakage") or manifest.get("profile_source") != "separate_profiling_run":
                reasons.append(f"{workload_id}: profile source is not an independent profiling run")
                continue
            if manifest.get("legacy_unlinked_in_denominator"):
                reasons.append(f"{workload_id}: legacy-unlinked objects entered the comparison denominator")
                continue
            policies = manifest.get("policies") if isinstance(manifest.get("policies"), list) else []
            by_policy = {str(row.get("policy")): row for row in policies if isinstance(row, dict)}
            if not all(name in by_policy for name in ("astrakv", "lru", "fifo")):
                reasons.append(f"{workload_id}: missing astrakv/lru/fifo policy summaries")
                continue
            workload_ids.add(workload_id)
            valid_rows.append((by_policy["astrakv"], by_policy["lru"], by_policy["fifo"]))
        aggregate = aggregate_rows(valid_rows)
        checks = {
            "minimum_three_distinct_workloads": len(workload_ids) >= 3,
            "astrakv_hit_rate_not_lower_than_lru": aggregate["astrakv_hit_rate"] >= aggregate["lru_hit_rate"],
            "astrakv_hit_rate_not_lower_than_fifo": aggregate["astrakv_hit_rate"] >= aggregate["fifo_hit_rate"],
            "astrakv_migration_not_higher_than_best_baseline": aggregate["astrakv_migration_bytes"] <= min(aggregate["lru_migration_bytes"], aggregate["fifo_migration_bytes"]),
            "no_astrakv_unavoided_oom": aggregate["astrakv_oom_unavoided"] == 0,
            "all_input_manifests_valid": not reasons and len(valid_rows) == len(manifests),
        }
        for name, passed in checks.items():
            if not passed:
                reasons.append(f"gate check failed: {name}")
        result = OfflineSafetyGateResult(
            status="accepted" if not reasons else "rejected",
            reasons=tuple(dict.fromkeys(reasons)), workload_ids=tuple(sorted(workload_ids)),
            aggregate=aggregate, checks=checks, evidence=tuple(evidence),
        )
        return cls(result)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "OfflineSafetyGate":
        return cls(OfflineSafetyGateResult(
            status=str(record.get("status") or "rejected"),
            reasons=tuple(str(item) for item in record.get("reasons", [])),
            workload_ids=tuple(str(item) for item in record.get("workload_ids", [])),
            aggregate=record.get("aggregate") if isinstance(record.get("aggregate"), dict) else {},
            checks=record.get("checks") if isinstance(record.get("checks"), dict) else {},
            evidence=tuple(item for item in record.get("evidence", []) if isinstance(item, dict)),
        ))


class GatedRuntimeAdapter:
    """Composition wrapper that blocks execution while leaving observation intact."""

    def __init__(self, adapter: Any, gate: OfflineSafetyGate) -> None:
        self.adapter = adapter
        self.gate = gate
        self.name = f"gated:{getattr(adapter, 'name', type(adapter).__name__)}"

    def describe(self) -> dict[str, Any]:
        payload = self.adapter.describe() if hasattr(self.adapter, "describe") else {}
        return {**payload, "offline_safety_gate": self.gate.result.to_record()}

    def capabilities(self) -> Any:
        return self.adapter.capabilities()

    def collect_runtime_events(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.collect_runtime_events(*args, **kwargs)

    def apply_hint(self, decision: OfflineEvictionDecision) -> RuntimeActionResult:
        if not self.gate.result.allowed:
            return RuntimeActionResult(
                status="blocked_by_offline_gate",
                message="offline safety gate rejected runtime action: " + "; ".join(self.gate.result.reasons),
            )
        return self.adapter.apply_hint(decision)


def aggregate_rows(rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, float]:
    names = ("astrakv", "lru", "fifo")
    total_requests = {name: 0.0 for name in names}
    total_hits = {name: 0.0 for name in names}
    migration = {name: 0.0 for name in names}
    oom = {name: 0.0 for name in names}
    for astro, lru, fifo in rows:
        for name, row in (("astrakv", astro), ("lru", lru), ("fifo", fifo)):
            total_requests[name] += number(row.get("request_count"))
            total_hits[name] += number(row.get("total_hits"))
            migration[name] += number(row.get("migration_bytes"))
            oom[name] += number(row.get("oom_unavoided"))
    return {
        "workload_count": float(len(rows)),
        "astrakv_hit_rate": total_hits["astrakv"] / max(1.0, total_requests["astrakv"]),
        "lru_hit_rate": total_hits["lru"] / max(1.0, total_requests["lru"]),
        "fifo_hit_rate": total_hits["fifo"] / max(1.0, total_requests["fifo"]),
        "astrakv_migration_bytes": migration["astrakv"],
        "lru_migration_bytes": migration["lru"],
        "fifo_migration_bytes": migration["fifo"],
        "astrakv_oom_unavoided": oom["astrakv"],
    }


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_gate(path: str | Path) -> OfflineSafetyGate:
    return OfflineSafetyGate.from_record(json.loads(Path(path).read_text(encoding="utf-8")))


def write_gate(path: str | Path, gate: OfflineSafetyGate) -> None:
    Path(path).write_text(json.dumps(gate.result.to_record(), indent=2, ensure_ascii=False), encoding="utf-8")
