"""Metadata-only MoE expert selective loading planner.

The planner consumes expert activation profiles and optional expert weight
catalogs, then emits adapter-facing placement/load decisions. It does not load
weights, allocate tensors, or modify a serving runtime.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from astrakv.scheduler.hints import SchedulerHint


class ExpertLoadAction(str, Enum):
    LOAD_GPU = "load_gpu"
    KEEP_GPU = "keep_gpu"
    KEEP_CPU = "keep_cpu"
    OFFLOAD_SSD = "offload_ssd"
    ON_DEMAND = "on_demand"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class ExpertProfile:
    layer_id: int | None
    expert_id: str
    activation_count: int = 0
    token_count: int = 0
    avg_score: float = 0.0
    bytes_observed: int = 0
    latency_ms: float = 0.0
    hotness_share: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return expert_key(self.layer_id, self.expert_id)


@dataclass(frozen=True, slots=True)
class ExpertCatalogEntry:
    layer_id: int | None
    expert_id: str
    size_bytes: int
    current_tier: str = "unknown"
    weight_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return expert_key(self.layer_id, self.expert_id)


@dataclass(frozen=True, slots=True)
class ExpertLoadPlannerConfig:
    gpu_budget_bytes: int = 2 * 1024 * 1024 * 1024
    cpu_budget_bytes: int = 16 * 1024 * 1024 * 1024
    default_expert_bytes: int = 256 * 1024 * 1024
    hotness_weight: float = 0.55
    activation_weight: float = 0.25
    score_weight: float = 0.10
    latency_weight: float = 0.10
    hot_threshold: float = 0.45
    warm_threshold: float = 0.18
    drop_threshold: float = 0.02
    ssd_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ExpertLoadDecision:
    layer_id: int | None
    expert_id: str
    action: ExpertLoadAction
    target_tier: str
    priority: float
    size_bytes: int
    gpu_budget_bytes: int
    gpu_bytes_after: int
    cpu_budget_bytes: int
    cpu_bytes_after: int
    reason: str
    current_tier: str = "unknown"
    activation_count: int = 0
    token_count: int = 0
    hotness_share: float = 0.0
    avg_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return expert_key(self.layer_id, self.expert_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "layer_id": "" if self.layer_id is None else self.layer_id,
            "expert_id": self.expert_id,
            "action": self.action.value,
            "target_tier": self.target_tier,
            "priority": round(self.priority, 6),
            "size_bytes": self.size_bytes,
            "current_tier": self.current_tier,
            "gpu_budget_bytes": self.gpu_budget_bytes,
            "gpu_bytes_after": self.gpu_bytes_after,
            "cpu_budget_bytes": self.cpu_budget_bytes,
            "cpu_bytes_after": self.cpu_bytes_after,
            "activation_count": self.activation_count,
            "token_count": self.token_count,
            "hotness_share": round(self.hotness_share, 6),
            "avg_score": round(self.avg_score, 6),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_hint(self) -> SchedulerHint:
        return SchedulerHint(
            request_id=self.key,
            action=self.action.value,
            reason=self.reason,
            priority=int(round(self.priority * 100)),
            metadata={
                "object_type": "moe_expert",
                "layer_id": self.layer_id,
                "expert_id": self.expert_id,
                "target_tier": self.target_tier,
                "current_tier": self.current_tier,
                "size_bytes": self.size_bytes,
                "gpu_budget_bytes": self.gpu_budget_bytes,
                "gpu_bytes_after": self.gpu_bytes_after,
                "cpu_budget_bytes": self.cpu_budget_bytes,
                "cpu_bytes_after": self.cpu_bytes_after,
                "activation_count": self.activation_count,
                "token_count": self.token_count,
                "hotness_share": self.hotness_share,
                "avg_score": self.avg_score,
                **dict(self.metadata),
            },
        )


class SelectiveExpertLoaderPlanner:
    def __init__(self, config: ExpertLoadPlannerConfig | None = None) -> None:
        self.config = config or ExpertLoadPlannerConfig()

    def plan(
        self,
        profiles: Iterable[ExpertProfile],
        *,
        catalog: dict[str, ExpertCatalogEntry] | None = None,
    ) -> list[ExpertLoadDecision]:
        profile_list = list(profiles)
        catalog_by_key = catalog or {}
        max_activation = max((profile.activation_count for profile in profile_list), default=0)
        ordered = sorted(
            profile_list,
            key=lambda profile: (
                self._priority(profile, max_activation),
                profile.activation_count,
                profile.key,
            ),
            reverse=True,
        )
        gpu_used = 0
        cpu_used = 0
        decisions: list[ExpertLoadDecision] = []
        for profile in ordered:
            catalog_entry = catalog_by_key.get(profile.key)
            size = _resolve_size(profile, catalog_entry, self.config.default_expert_bytes)
            current_tier = catalog_entry.current_tier if catalog_entry is not None else "unknown"
            priority = self._priority(profile, max_activation)
            action, target_tier, reason, gpu_used, cpu_used = self._choose_action(
                priority=priority,
                size=size,
                current_tier=current_tier,
                gpu_used=gpu_used,
                cpu_used=cpu_used,
            )
            decisions.append(
                ExpertLoadDecision(
                    layer_id=profile.layer_id,
                    expert_id=profile.expert_id,
                    action=action,
                    target_tier=target_tier,
                    priority=priority,
                    size_bytes=size,
                    current_tier=current_tier,
                    gpu_budget_bytes=self.config.gpu_budget_bytes,
                    gpu_bytes_after=gpu_used,
                    cpu_budget_bytes=self.config.cpu_budget_bytes,
                    cpu_bytes_after=cpu_used,
                    activation_count=profile.activation_count,
                    token_count=profile.token_count,
                    hotness_share=profile.hotness_share,
                    avg_score=profile.avg_score,
                    reason=reason,
                    metadata={
                        "bytes_observed": profile.bytes_observed,
                        "latency_ms": profile.latency_ms,
                        "weight_path": catalog_entry.weight_path if catalog_entry is not None else "",
                        **dict(profile.metadata),
                        **(dict(catalog_entry.metadata) if catalog_entry is not None else {}),
                    },
                )
            )
        return decisions

    def _priority(self, profile: ExpertProfile, max_activation: int) -> float:
        activation_score = profile.activation_count / max(1, max_activation)
        latency_score = min(1.0, profile.latency_ms / 100.0) if profile.latency_ms else 0.0
        priority = (
            profile.hotness_share * self.config.hotness_weight
            + activation_score * self.config.activation_weight
            + profile.avg_score * self.config.score_weight
            + latency_score * self.config.latency_weight
        )
        return clamp(priority)

    def _choose_action(
        self,
        *,
        priority: float,
        size: int,
        current_tier: str,
        gpu_used: int,
        cpu_used: int,
    ) -> tuple[ExpertLoadAction, str, str, int, int]:
        cfg = self.config
        if priority <= cfg.drop_threshold:
            return ExpertLoadAction.DROP, "none", "priority below drop threshold", gpu_used, cpu_used

        if priority >= cfg.hot_threshold and gpu_used + size <= cfg.gpu_budget_bytes:
            action = ExpertLoadAction.KEEP_GPU if current_tier == "gpu" else ExpertLoadAction.LOAD_GPU
            return action, "gpu", "hot expert scheduled within GPU budget", gpu_used + size, cpu_used

        if priority >= cfg.warm_threshold and cpu_used + size <= cfg.cpu_budget_bytes:
            return ExpertLoadAction.KEEP_CPU, "cpu", "warm expert kept in CPU tier", gpu_used, cpu_used + size

        if cfg.ssd_enabled:
            return ExpertLoadAction.OFFLOAD_SSD, "ssd", "cold expert assigned to SSD tier", gpu_used, cpu_used

        return ExpertLoadAction.ON_DEMAND, "unknown", "cold expert left for on-demand load", gpu_used, cpu_used


def load_expert_profiles_from_summary(path: str | Path) -> list[ExpertProfile]:
    summary_path = Path(path)
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return [profile_from_summary_row(row) for row in csv.DictReader(handle) if row.get("expert_id")]


def profile_from_summary_row(row: dict[str, Any]) -> ExpertProfile:
    return ExpertProfile(
        layer_id=_optional_int(row.get("layer_id")),
        expert_id=str(row.get("expert_id", "")),
        activation_count=as_int(row.get("activation_count")),
        token_count=as_int(row.get("token_count")),
        avg_score=as_float(row.get("avg_score")),
        bytes_observed=as_int(row.get("bytes") or row.get("bytes_observed")),
        latency_ms=as_float(row.get("latency_ms")),
        hotness_share=as_float(row.get("hotness_share")),
    )


def load_expert_catalog(path: str | Path) -> dict[str, ExpertCatalogEntry]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    if catalog_path.suffix.lower() == ".csv":
        return _load_catalog_csv(catalog_path)
    if catalog_path.suffix.lower() == ".jsonl":
        return _load_catalog_jsonl(catalog_path)
    return _load_catalog_json(catalog_path)


def write_decisions_csv(path: str | Path, decisions: Iterable[ExpertLoadDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [decision.to_record() for decision in decisions]
    fieldnames = [
        "layer_id",
        "expert_id",
        "action",
        "target_tier",
        "priority",
        "size_bytes",
        "current_tier",
        "gpu_budget_bytes",
        "gpu_bytes_after",
        "cpu_budget_bytes",
        "cpu_bytes_after",
        "activation_count",
        "token_count",
        "hotness_share",
        "avg_score",
        "reason",
        "metadata",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["metadata"] = json.dumps(output.get("metadata", {}), ensure_ascii=False)
            writer.writerow(output)


def write_hints_jsonl(path: str | Path, decisions: Iterable[ExpertLoadDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            hint = decision.to_hint()
            handle.write(
                json.dumps(
                    {
                        "schema": "astra-scheduler-hint-v1",
                        "request_id": hint.request_id,
                        "action": hint.action,
                        "reason": hint.reason,
                        "priority": hint.priority,
                        "metadata": hint.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def summarize_decisions(decisions: Iterable[ExpertLoadDecision]) -> dict[str, Any]:
    decision_list = list(decisions)
    action_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    total_bytes = 0
    gpu_bytes = 0
    cpu_bytes = 0
    ssd_bytes = 0
    skipped_gpu_bytes = 0
    for decision in decision_list:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
        tier_counts[decision.target_tier] = tier_counts.get(decision.target_tier, 0) + 1
        total_bytes += decision.size_bytes
        if decision.target_tier == "gpu":
            gpu_bytes += decision.size_bytes
        elif decision.target_tier == "cpu":
            cpu_bytes += decision.size_bytes
            skipped_gpu_bytes += decision.size_bytes
        elif decision.target_tier == "ssd":
            ssd_bytes += decision.size_bytes
            skipped_gpu_bytes += decision.size_bytes
        else:
            skipped_gpu_bytes += decision.size_bytes
    return {
        "total_experts": len(decision_list),
        "action_counts": dict(sorted(action_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "total_expert_bytes": total_bytes,
        "planned_gpu_bytes": gpu_bytes,
        "planned_cpu_bytes": cpu_bytes,
        "planned_ssd_bytes": ssd_bytes,
        "estimated_gpu_bytes_saved": skipped_gpu_bytes,
        "estimated_gpu_saving_rate": skipped_gpu_bytes / max(1, total_bytes),
    }


def expert_key(layer_id: int | None, expert_id: str) -> str:
    layer = "*" if layer_id is None else str(layer_id)
    return f"layer{layer}:expert{expert_id}"


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def as_int(value: Any) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def as_float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _resolve_size(
    profile: ExpertProfile,
    catalog_entry: ExpertCatalogEntry | None,
    default_size: int,
) -> int:
    if catalog_entry is not None and catalog_entry.size_bytes > 0:
        return catalog_entry.size_bytes
    if profile.bytes_observed > 0:
        return profile.bytes_observed
    return default_size


def _load_catalog_csv(path: Path) -> dict[str, ExpertCatalogEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        entries = [catalog_entry_from_record(row) for row in csv.DictReader(handle) if row.get("expert_id")]
    return {entry.key: entry for entry in entries}


def _load_catalog_jsonl(path: Path) -> dict[str, ExpertCatalogEntry]:
    entries: list[ExpertCatalogEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                entries.append(catalog_entry_from_record(record))
    return {entry.key: entry for entry in entries}


def _load_catalog_json(path: Path) -> dict[str, ExpertCatalogEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("experts"), list):
        records = data["experts"]
    elif isinstance(data, list):
        records = data
    else:
        records = []
    entries = [catalog_entry_from_record(record) for record in records if isinstance(record, dict)]
    return {entry.key: entry for entry in entries}


def catalog_entry_from_record(record: dict[str, Any]) -> ExpertCatalogEntry:
    return ExpertCatalogEntry(
        layer_id=_optional_int(_first_present(record, "layer_id", "layer")),
        expert_id=str(_first_present(record, "expert_id", "expert") or ""),
        size_bytes=as_int(_first_present(record, "size_bytes", "bytes", "weight_bytes")),
        current_tier=str(record.get("current_tier", "unknown") or "unknown"),
        weight_path=str(record.get("weight_path", "") or record.get("path", "")),
        metadata={
            key: value
            for key, value in record.items()
            if key
            not in {
                "layer_id",
                "layer",
                "expert_id",
                "expert",
                "size_bytes",
                "bytes",
                "weight_bytes",
                "current_tier",
                "weight_path",
                "path",
            }
        },
    )


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
