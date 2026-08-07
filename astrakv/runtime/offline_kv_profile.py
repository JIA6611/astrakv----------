"""Compatibility-aware offline profile records for KV-Core scheduling.

The profile is evidence for prefix admission priority only.  It does not
authorize skipping arbitrary layer KV or moving weights online.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from astrakv.runtime.kv_runtime_core import KVCompatibilityKey, exact_token_prefix_hash


OFFLINE_PROFILE_SCHEMA = "astrakv-kv-core-offline-profile-v1"


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
        }


def build_manifest(*, target: dict[str, Any], hardware: dict[str, Any], workload_sha256: str, profiles: Iterable[OfflinePrefixProfile]) -> dict[str, Any]:
    entries = [profile.to_record() for profile in profiles]
    mean_ppl, low, high = bootstrap_mean_ci((item["teacher_forced_ppl"] for item in entries))
    return {
        "schema": OFFLINE_PROFILE_SCHEMA,
        "target": dict(target),
        "hardware": dict(hardware),
        "workload_sha256": workload_sha256,
        "sample_count": len(entries),
        "teacher_forced_ppl": {"mean": mean_ppl, "bootstrap_ci95": [low, high]},
        "profiles": entries,
        "online_scope": {
            "may_influence": ["kv_prefix_admission_priority", "cpu_prefetch_budget", "partial_load_upper_bound"],
            "must_not_influence": ["layer_kv_skipping", "online_weight_movement", "moe_expert_residency"],
        },
    }


__all__ = ["OFFLINE_PROFILE_SCHEMA", "OfflinePrefixProfile", "bootstrap_mean_ci", "build_manifest", "validate_qwen3_8b_target"]
