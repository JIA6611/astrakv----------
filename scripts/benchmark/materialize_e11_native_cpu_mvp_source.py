"""Build deterministic workload regimes for E11 native CPU policy A/B.

The selector keeps real grouped prompts and materializes one of three
pre-registered access/profile regimes:

``recency_aligned``
    cold-C, cold-D, hot-A first/revisit/revisit,
    hot-B first/revisit/revisit

``scan_pollution_past_observed``
    hot-A training visits, hot-B training visits, cold-C, cold-D,
    hot-B delayed revisit

``profile_shift_or_stale``
    stale-old, new-hot first, filler-C, filler-D,
    new-hot revisit/revisit

All cells set the policy-visible reuse hint to zero.  AstraKV-W can therefore
learn only from hits/stores that already happened earlier in the schedule;
the measured schedule's future group size remains report-only metadata.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


REGIMES = (
    "recency_aligned",
    "scan_pollution_past_observed",
    "profile_shift_or_stale",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("grouped source must contain JSON objects")
    return sorted(rows, key=lambda row: int(row.get("order") or 0))


def context_tokens(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    try:
        return max(1, int(metadata.get("context_token_estimate") or 0))
    except (TypeError, ValueError):
        return max(1, len(str(row.get("prompt") or "").split()))


def select_schedule(
    rows: list[dict[str, Any]], *, hot_groups: int = 2, cold_groups: int = 2,
    regime: str = "scan_pollution_past_observed",
) -> list[dict[str, Any]]:
    if regime not in REGIMES:
        raise ValueError(f"unsupported E11 regime {regime!r}; expected one of {', '.join(REGIMES)}")
    if hot_groups < 2 or cold_groups < 1:
        raise ValueError("E11 MVP requires at least two hot groups and one cold group")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        group = str(row.get("reuse_group") or "")
        if not group:
            raise ValueError("every source row must have reuse_group")
        if group not in buckets:
            order.append(group)
        buckets[group].append(row)
    eligible = [buckets[group] for group in order if len(buckets[group]) >= 3]
    singleton = [buckets[group] for group in order if len(buckets[group]) == 1]
    if len(eligible) < hot_groups:
        raise ValueError(f"source has only {len(eligible)} groups with at least three visits")
    if len(singleton) < cold_groups:
        raise ValueError(f"source has only {len(singleton)} singleton groups")

    # Keep the first real high-reuse groups (stable source order).  Pick the
    # shortest singleton contexts so the tiny validation does not spend time
    # decoding unrelated long cold prompts.
    hot = eligible[:hot_groups]
    cold = sorted(singleton, key=lambda bucket: (context_tokens(bucket[0]), str(bucket[0].get("reuse_group"))))[
        :cold_groups
    ]
    if regime == "recency_aligned":
        scheduled = [bucket[0] for bucket in cold]
        for bucket in hot:
            scheduled.extend(bucket[:3])
        profile_quality = "past_observed_only"
    elif regime == "scan_pollution_past_observed":
        scheduled = [*hot[0][:3], *hot[1][:3]]
        scheduled.extend(bucket[0] for bucket in cold)
        delayed = dict(hot[1][2])
        delayed["request_id"] = f"{hot[1][2].get('request_id')}-e11-delayed"
        scheduled.append(delayed)
        profile_quality = "past_observed_only"
    else:
        # A becomes hot from observed hits, then the access phase shifts to B.
        # LRU can adapt immediately to recency; AstraKV-W may over-protect A.
        scheduled = [*hot[0][:3], *hot[1][:3]]
        profile_quality = "past_observed_phase_shift"

    result: list[dict[str, Any]] = []
    for index, source in enumerate(scheduled):
        row = dict(source)
        metadata = dict(row.get("metadata") or {})
        group = str(row.get("reuse_group") or "")
        is_hot = any(group == str(bucket[0].get("reuse_group") or "") for bucket in hot)
        phase_index = sum(
            str(item.get("reuse_group") or "") == group
            for item in scheduled[:index]
        )
        metadata.update({
            "e11_mvp_schedule": True,
            "e11_regime": regime,
            "e11_profile_quality": profile_quality,
            "e11_temperature": "hot" if is_hot else "cold",
            "e11_phase": "first" if phase_index == 0 else "revisit",
            "e11_source_order": source.get("order"),
        })
        # Do not expose group size/future visits to the native policy.  The
        # canonical row still carries its actual reuse_ratio for reporting.
        row["_e11_policy_reuse_ratio"] = 0.0
        row["metadata"] = metadata
        row["order"] = index
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hot-groups", type=int, default=2)
    parser.add_argument("--cold-groups", type=int, default=2)
    parser.add_argument("--regime", choices=REGIMES, default="scan_pollution_past_observed")
    args = parser.parse_args()
    selected = select_schedule(
        load_rows(args.input), hot_groups=args.hot_groups, cold_groups=args.cold_groups,
        regime=args.regime,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": "astrakv-e11-native-cpu-regime-source-v1",
        "regime": args.regime,
        "output": str(args.output),
        "request_count": len(selected),
        "reuse_groups": len({str(row.get("reuse_group")) for row in selected}),
        "schedule": [
            {
                "order": row["order"],
                "request_id": row.get("request_id"),
                "reuse_group": row.get("reuse_group"),
                "temperature": row["metadata"]["e11_temperature"],
                "phase": row["metadata"]["e11_phase"],
                "context_token_estimate": context_tokens(row),
            }
            for row in selected
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
