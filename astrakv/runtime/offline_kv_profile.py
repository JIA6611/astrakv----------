"""Compatibility-aware offline profile records for KV-Core scheduling.

The profile is evidence for prefix admission priority only.  It does not
authorize skipping arbitrary layer KV or moving weights online.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from astrakv.runtime.kv_runtime_core import KVCompatibilityKey


OFFLINE_PROFILE_SCHEMA = "astrakv-kv-core-offline-profile-v2"


def validate_qwen3_8b_target(*, model_id: str, quantization: str) -> None:
    if model_id.strip() != "Qwen3-8B":
        raise ValueError("KV-Core offline profiles are pinned to non-quantized Qwen3-8B")
    if quantization.strip().lower() != "unquantized":
        raise ValueError("AWQ and other quantized artifacts cannot enter the Qwen3-8B KV-Core profile")


def bootstrap_mean_ci(values: Iterable[float], *, samples: int = 1000, seed: int = 0) -> tuple[float | None, float | None, float | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return None, None, None
    point = sum(data) / len(data)
    generator = random.Random(seed)
    means = sorted(sum(data[generator.randrange(len(data))] for _ in data) / len(data) for _ in range(samples))
    return point, means[max(0, math.ceil(0.025 * samples) - 1)], means[max(0, math.ceil(0.975 * samples) - 1)]


@dataclass(frozen=True, slots=True)
class OfflinePrefixProfile:
    sample_id: str
    compatibility_key: KVCompatibilityKey
    native_key: str
    prefix_tokens: int
    teacher_forced_loss: float
    teacher_forced_ppl: float
    layer_activity_l2: dict[str, float]
    layer_cka: dict[str, float]
    layer_cosine: dict[str, float]
    layer_l2: dict[str, float]
    layer_max_abs: dict[str, float]
    path_quality: dict[str, dict[str, float]] | None = None
    layer_sensitivity: dict[str, dict[str, float]] | None = None
    correctness_passed: bool = True

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "compatibility_key": self.compatibility_key.to_record(),
            "native_key": self.native_key,
            "prefix_tokens": self.prefix_tokens,
            "teacher_forced_loss": self.teacher_forced_loss,
            "teacher_forced_ppl": self.teacher_forced_ppl,
            "layer_activity_l2": dict(self.layer_activity_l2),
            "layer_cka": dict(self.layer_cka),
            "layer_cosine": dict(self.layer_cosine),
            "layer_l2": dict(self.layer_l2),
            "layer_max_abs": dict(self.layer_max_abs),
            "path_quality": dict(self.path_quality or {}),
            "layer_sensitivity": dict(self.layer_sensitivity or {}),
            "correctness_passed": bool(self.correctness_passed),
        }


@dataclass(frozen=True, slots=True)
class PrefixRuntimeHint:
    compatibility_identity: str
    sensitivity_rank: float
    admission_priority_boost: int
    partial_load_fraction: float
    prefetch_priority: float


class OfflineKVProfileIndex:
    """Fail-closed exact-prefix hints exported by the Qwen3 KV profiler."""

    def __init__(self, *, target: dict[str, Any], hints: dict[str, PrefixRuntimeHint]) -> None:
        self.target = dict(target)
        self._hints = dict(hints)

    def hint_for(self, key: KVCompatibilityKey) -> PrefixRuntimeHint | None:
        static = {
            "model_id": key.model_id,
            "model_revision": key.model_revision,
            "tokenizer_revision": key.tokenizer_revision,
            "chat_template_revision": key.chat_template_revision,
            "dtype": key.dtype,
            "block_size_tokens": key.block_size_tokens,
            "chunk_size_tokens": key.chunk_size_tokens,
        }
        for name, value in static.items():
            if str(self.target.get(name)) != str(value):
                return None
        return self._hints.get(key.identity)

    @classmethod
    def load(cls, path: str | Path) -> "OfflineKVProfileIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != OFFLINE_PROFILE_SCHEMA:
            raise ValueError("offline KV profile schema mismatch")
        if (payload.get("correctness") or {}).get("passed") is not True:
            raise ValueError("offline KV profile failed correctness validation")
        target = payload.get("target")
        profiles = payload.get("profiles")
        if not isinstance(target, dict) or not isinstance(profiles, list) or not profiles:
            raise ValueError("offline KV profile is incomplete")
        scores_by_identity: dict[str, float] = {}
        for profile in profiles:
            if not isinstance(profile, dict) or profile.get("correctness_passed") is not True:
                raise ValueError("offline KV profile contains an unverified prefix")
            key_record = profile.get("compatibility_key")
            if not isinstance(key_record, dict):
                raise ValueError("offline KV profile compatibility key missing")
            key = KVCompatibilityKey(**{
                name: key_record[name]
                for name in (
                    "model_id", "model_revision", "tokenizer_revision",
                    "chat_template_revision", "dtype", "rope_config",
                    "adapter_namespace", "kv_layout", "block_size_tokens",
                    "chunk_size_tokens", "layer_group", "prefix_hash",
                )
            })
            sensitivity = profile.get("layer_sensitivity") or {}
            deltas = [
                abs(float(metrics.get("teacher_forced_loss_delta", 0.0)))
                for metrics in sensitivity.values()
                if isinstance(metrics, dict)
            ]
            score = max(deltas, default=0.0)
            scores_by_identity[key.identity] = max(
                score, scores_by_identity.get(key.identity, 0.0),
            )
        ordered = sorted(set(scores_by_identity.values()))
        rank_by_score = (
            {ordered[0]: 0.5}
            if len(ordered) == 1
            else {
                score: index / (len(ordered) - 1)
                for index, score in enumerate(ordered)
            }
        )
        hints = {}
        for identity, score in scores_by_identity.items():
            rank = rank_by_score[score]
            hints[identity] = PrefixRuntimeHint(
                compatibility_identity=identity,
                sensitivity_rank=rank,
                admission_priority_boost=int(round(100 * rank)),
                partial_load_fraction=0.5 + 0.5 * rank,
                prefetch_priority=rank,
            )
        return cls(target=target, hints=hints)


def build_manifest(*, target: dict[str, Any], hardware: dict[str, Any], workload_sha256: str, profiles: Iterable[OfflinePrefixProfile]) -> dict[str, Any]:
    entries = [profile.to_record() for profile in profiles]
    mean_ppl, low, high = bootstrap_mean_ci((item["teacher_forced_ppl"] for item in entries))
    layer_values: dict[str, list[float]] = {}
    for item in entries:
        for layer, metrics in item.get("layer_sensitivity", {}).items():
            if isinstance(metrics, dict) and isinstance(metrics.get("teacher_forced_loss_delta"), (int, float)):
                layer_values.setdefault(str(layer), []).append(float(metrics["teacher_forced_loss_delta"]))
    layer_summary = {}
    for layer, values in sorted(layer_values.items(), key=lambda row: int(row[0])):
        point, layer_low, layer_high = bootstrap_mean_ci(values)
        layer_summary[layer] = {
            "sample_count": len(values),
            "teacher_forced_loss_delta_mean": point,
            "bootstrap_ci95": [layer_low, layer_high],
        }
    return {
        "schema": OFFLINE_PROFILE_SCHEMA,
        "target": dict(target),
        "hardware": dict(hardware),
        "workload_sha256": workload_sha256,
        "sample_count": len(entries),
        "teacher_forced_ppl": {"mean": mean_ppl, "bootstrap_ci95": [low, high]},
        "profiles": entries,
        "correctness": {
            "passed": bool(entries) and all(bool(item.get("correctness_passed")) for item in entries),
            "failed_samples": [item["sample_id"] for item in entries if not item.get("correctness_passed")],
        },
        "layer_sensitivity": layer_summary,
        "online_scope": {
            "may_influence": ["kv_prefix_admission_priority", "cpu_prefetch_budget", "partial_load_upper_bound"],
            "must_not_influence": ["layer_kv_skipping", "online_weight_movement", "moe_expert_residency"],
        },
    }


__all__ = [
    "OFFLINE_PROFILE_SCHEMA", "OfflineKVProfileIndex", "OfflinePrefixProfile",
    "PrefixRuntimeHint", "bootstrap_mean_ci", "build_manifest",
    "validate_qwen3_8b_target",
]
