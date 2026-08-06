"""Strategy-level AstraKV ON/OFF TTFT validation helpers.

This module builds a controlled repeated-prefix workload that forces KV churn
before revisiting an anchor prompt, then summarizes request/cache evidence from
paired real-endpoint runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow, load_runtime_workload_jsonl


WORKLOAD_MANIFEST_SCHEMA = "astrakv-policy-ab-workload-v1"
REPORT_SCHEMA = "astrakv-policy-ab-ttft-report-v1"
TIMELINE_SCHEMA = "astrakv-policy-ab-event-v1"
SAMPLE_SCHEMA = "astrakv-policy-ab-sample-v1"
DEFAULT_ANCHOR_COUNT = 6
DEFAULT_CHURN_VARIANTS = 12
DEFAULT_PROMPT_TOKENS = 8192
DEFAULT_WARMUP_CYCLES = 5
DEFAULT_SAMPLE_CYCLES = 30
DEFAULT_IDLE_SECONDS = 1.5
DEFAULT_EXPECTED_OUTPUT_TOKENS = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_CONTEXT_LENGTH = 8192
DEFAULT_KV_CACHE_MEMORY_BYTES = "2G"
ROLES = ("baseline", "variant")
COMPLETED_STATUSES = frozenset({"completed", "ok", "executed", "available"})

_ANCHOR_THEMES = (
    "astronomy orbit nebula telescope spectrum vacuum stellar lunar comet eclipse gravity plasma",
    "botany forest pollen canopy fern moss seedling petal orchard vine meadow wetland",
    "geology basalt quartz strata granite magma fault sediment ridge canyon mineral bedrock",
    "oceanography current tide salinity reef trench estuary plankton seagrass monsoon gyre",
    "music rhythm cadence timbre chorus harmony octave resonance overtone sonata tempo coda",
    "architecture atrium masonry column lintel facade terrace arcade vault courtyard parapet",
)
_CHURN_PRIMARY_THEMES = (
    "finance ledger coupon invoice tariff bond futures margin clearing hedging premium",
    "cuisine saffron skillet broth citrus barley paprika simmer garnish roasting herbal",
    "aviation runway fuselage turbine cockpit waypoint glide beacon hangar throttle radar",
    "medicine triage dosage biopsy stethoscope plasma neuron tendon marrow enzyme therapy",
    "literature stanza allegory fable sonnet epilogue metaphor cadence dialogue motif prose",
    "logistics pallet freight customs depot forklift manifest routing container berth cargo",
    "mathematics tensor manifold eigenvalue lattice theorem corollary lemma integral scalar",
    "robotics actuator gripper lidar servo kinematics torque chassis waypoint encoder sensor",
    "agriculture harvest furrow irrigation compost tiller pasture orchard mildew barley",
    "textiles spindle warp weft loom filament dyeing satin corduroy quilting canvas fiber",
    "energy turbine capacitor inverter reactor cathode anode gridline storage thermal voltage",
    "history empire treaty census dynasty archive relic maritime colony chronicle frontier",
)
_CHURN_SECONDARY_THEMES = (
    "meteorology cyclone drizzle anticyclone cumulus isobar dewpoint hailstorm squall monsoon",
    "sports relay sprint dribble kickoff bullpen offense defense medal bracket qualifier",
    "cinema storyboard montage closeup screenplay dolly retake subtitle premiere soundtrack",
    "cybersecurity firewall sandbox ciphertext nonce kernel exploit audit patch signing",
    "education seminar practicum rubric syllabus lecture tutorial cohort archive thesis",
    "manufacturing foundry lathe billet gasket tolerances milling casting forging fixture",
    "wildlife habitat migration burrow antler roost clutch molting estivation tundra",
    "photonics diffraction waveguide aperture prism scintillation lensing radiance chromatic",
    "transport metro viaduct junction axle convoy tollway carriage depot switchyard route",
    "ceramics kiln glaze porcelain stoneware slip casting terracotta grog annealing",
    "gaming questline speedrun checkpoint inventory crafting dungeon encounter scoreboard",
    "law jurisprudence affidavit docket verdict tribunal injunction charter statute appeal",
)


@dataclass(frozen=True, slots=True)
class PolicyAbWorkloadBundle:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def build_workload_bundle(
    *,
    anchor_count: int = DEFAULT_ANCHOR_COUNT,
    churn_variants: int = DEFAULT_CHURN_VARIANTS,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    warmup_cycles: int = DEFAULT_WARMUP_CYCLES,
    sample_cycles: int = DEFAULT_SAMPLE_CYCLES,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> PolicyAbWorkloadBundle:
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive")
    if churn_variants <= 0:
        raise ValueError("churn_variants must be positive")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if warmup_cycles < 0 or sample_cycles <= 0:
        raise ValueError("warmup_cycles must be >= 0 and sample_cycles must be positive")
    if idle_seconds < 0:
        raise ValueError("idle_seconds must be >= 0")

    anchors = [
        _prompt_spec("A", index + 1, prompt_tokens, _ANCHOR_THEMES[index % len(_ANCHOR_THEMES)])
        for index in range(anchor_count)
    ]
    churn_primary = [
        _prompt_spec("B", index + 1, prompt_tokens, _CHURN_PRIMARY_THEMES[index % len(_CHURN_PRIMARY_THEMES)])
        for index in range(churn_variants)
    ]
    churn_secondary = [
        _prompt_spec("C", index + 1, prompt_tokens, _CHURN_SECONDARY_THEMES[index % len(_CHURN_SECONDARY_THEMES)])
        for index in range(churn_variants)
    ]

    rows: list[dict[str, Any]] = []
    arrival_index = 0
    cycle_specs = [("warmup", warmup_cycles), ("sample", sample_cycles)]
    for cycle_kind, cycle_count in cycle_specs:
        for cycle_index in range(cycle_count):
            anchor = anchors[cycle_index % len(anchors)]
            primary = churn_primary[cycle_index % len(churn_primary)]
            secondary = churn_secondary[(cycle_index * 5 + 3) % len(churn_secondary)]
            cycle_label = f"{cycle_kind}-{cycle_index:03d}"
            seed_request_id = f"{cycle_label}-{anchor['label'].lower()}-seed"
            revisit_request_id = f"{cycle_label}-{anchor['label'].lower()}-revisit"
            rows.append(
                _workload_row(
                    request_id=seed_request_id,
                    prompt=anchor["prompt"],
                    prefix_id=anchor["prefix_id"],
                    arrival_index=arrival_index,
                    reuse_ratio=0.0,
                    reuse_bucket="none",
                    case=f"{cycle_kind}_anchor_seed",
                    context_length=context_length,
                    expected_output_tokens=expected_output_tokens,
                    batch_size=batch_size,
                    metadata={
                        "scenario": "policy_ab_ttft",
                        "workload_type": "policy_ab_ttft",
                        "cycle_kind": cycle_kind,
                        "cycle_index": cycle_index,
                        "phase": "anchor_seed",
                        "anchor_id": anchor["label"],
                        "churn_primary_id": primary["label"],
                        "churn_secondary_id": secondary["label"],
                        "reuse_group": anchor["prefix_id"],
                        "shared_context": True,
                        "expected_reuse": 0,
                        "reference_request_id": "",
                        "cache_pressure_role": "target",
                    },
                )
            )
            arrival_index += 1
            rows.append(
                _workload_row(
                    request_id=f"{cycle_label}-{primary['label'].lower()}-churn",
                    prompt=primary["prompt"],
                    prefix_id=primary["prefix_id"],
                    arrival_index=arrival_index,
                    reuse_ratio=0.0,
                    reuse_bucket="none",
                    case=f"{cycle_kind}_churn_primary",
                    context_length=context_length,
                    expected_output_tokens=expected_output_tokens,
                    batch_size=batch_size,
                    metadata={
                        "scenario": "policy_ab_ttft",
                        "workload_type": "policy_ab_ttft",
                        "cycle_kind": cycle_kind,
                        "cycle_index": cycle_index,
                        "phase": "churn_primary",
                        "anchor_id": anchor["label"],
                        "churn_primary_id": primary["label"],
                        "churn_secondary_id": secondary["label"],
                        "reuse_group": primary["prefix_id"],
                        "shared_context": False,
                        "expected_reuse": 0,
                        "reference_request_id": seed_request_id,
                        "cache_pressure_role": "churn",
                    },
                )
            )
            arrival_index += 1
            rows.append(
                _workload_row(
                    request_id=f"{cycle_label}-{secondary['label'].lower()}-churn",
                    prompt=secondary["prompt"],
                    prefix_id=secondary["prefix_id"],
                    arrival_index=arrival_index,
                    reuse_ratio=0.0,
                    reuse_bucket="none",
                    case=f"{cycle_kind}_churn_secondary",
                    context_length=context_length,
                    expected_output_tokens=expected_output_tokens,
                    batch_size=batch_size,
                    metadata={
                        "scenario": "policy_ab_ttft",
                        "workload_type": "policy_ab_ttft",
                        "cycle_kind": cycle_kind,
                        "cycle_index": cycle_index,
                        "phase": "churn_secondary",
                        "anchor_id": anchor["label"],
                        "churn_primary_id": primary["label"],
                        "churn_secondary_id": secondary["label"],
                        "reuse_group": secondary["prefix_id"],
                        "shared_context": False,
                        "expected_reuse": 0,
                        "reference_request_id": seed_request_id,
                        "cache_pressure_role": "churn",
                    },
                )
            )
            arrival_index += 1
            rows.append(
                _workload_row(
                    request_id=revisit_request_id,
                    prompt=anchor["prompt"],
                    prefix_id=anchor["prefix_id"],
                    arrival_index=arrival_index,
                    reuse_ratio=1.0,
                    reuse_bucket="high",
                    case=f"{cycle_kind}_anchor_revisit",
                    context_length=context_length,
                    expected_output_tokens=expected_output_tokens,
                    batch_size=batch_size,
                    sleep_before_s=idle_seconds,
                    metadata={
                        "scenario": "policy_ab_ttft",
                        "workload_type": "policy_ab_ttft",
                        "cycle_kind": cycle_kind,
                        "cycle_index": cycle_index,
                        "phase": "anchor_revisit",
                        "anchor_id": anchor["label"],
                        "churn_primary_id": primary["label"],
                        "churn_secondary_id": secondary["label"],
                        "reuse_group": anchor["prefix_id"],
                        "shared_context": True,
                        "expected_reuse": 1,
                        "reference_request_id": seed_request_id,
                        "cache_pressure_role": "target",
                        "idle_seconds_before_revisit": idle_seconds,
                    },
                )
            )
            arrival_index += 1

    manifest = {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "anchor_count": anchor_count,
        "churn_primary_count": churn_variants,
        "churn_secondary_count": churn_variants,
        "prompt_tokens": prompt_tokens,
        "context_length": context_length,
        "warmup_cycles": warmup_cycles,
        "sample_cycles": sample_cycles,
        "idle_seconds": idle_seconds,
        "expected_output_tokens": expected_output_tokens,
        "batch_size": batch_size,
        "row_count": len(rows),
        "phase_counts": _count_by(rows, lambda item: str(item.get("metadata", {}).get("phase") or "")),
        "cycle_kind_counts": _count_by(rows, lambda item: str(item.get("metadata", {}).get("cycle_kind") or "")),
        "request_order": [str(item["request_id"]) for item in rows],
        "notes": [
            "Each cycle follows anchor seed -> churn primary -> churn secondary -> idle -> anchor revisit.",
            "Warm-up cycles remain in the workload for cache shaping but are excluded from revisit summary statistics.",
            "Revisit requests use max_tokens=1 via expected_output_tokens to emphasize TTFT and prefill behavior.",
        ],
    }
    return PolicyAbWorkloadBundle(rows=rows, manifest=manifest)


def write_workload_bundle(output_dir: str | Path, bundle: PolicyAbWorkloadBundle) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    workload_path = root / "policy_ab_ttft_workload.jsonl"
    manifest_path = root / "policy_ab_ttft_workload_manifest.json"
    report_path = root / "policy_ab_ttft_workload_report.md"
    _write_jsonl(workload_path, bundle.rows)
    manifest_path.write_text(json.dumps(bundle.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(_workload_report_markdown(bundle), encoding="utf-8")
    return {
        "workload": workload_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def build_suite_report(suite_dir: str | Path) -> dict[str, Any]:
    root = Path(suite_dir)
    workload_path = _discover_workload_path(root)
    workload_rows = load_runtime_workload_jsonl(workload_path)
    workload_by_request = {row.request_id: row for row in workload_rows}
    role_reports = {role: _build_role_report(root, role, workload_by_request) for role in ROLES}
    comparison = _build_comparison(role_reports)
    event_timeline = _build_event_timeline(role_reports)
    revisit_samples = [
        {**sample, "role": role}
        for role, report in role_reports.items()
        for sample in report["revisit_samples"]
    ]
    return {
        "schema": REPORT_SCHEMA,
        "suite_dir": str(root),
        "workload_path": str(workload_path),
        "workload_summary": {
            "row_count": len(workload_rows),
            "warmup_cycles": len([
                row for row in workload_rows
                if str((row.metadata or {}).get("cycle_kind") or "") == "warmup"
                and str((row.metadata or {}).get("phase") or "") == "anchor_revisit"
            ]),
            "sample_cycles": len([
                row for row in workload_rows
                if str((row.metadata or {}).get("cycle_kind") or "") == "sample"
                and str((row.metadata or {}).get("phase") or "") == "anchor_revisit"
            ]),
        },
        "roles": role_reports,
        "comparison": comparison,
        "revisit_samples": revisit_samples,
        "event_timeline": event_timeline,
    }


def write_suite_report(output_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "policy_ab_ttft_report.json"
    markdown_path = root / "policy_ab_ttft_report.md"
    timeline_path = root / "policy_ab_event_timeline.jsonl"
    sample_path = root / "policy_ab_revisit_samples.jsonl"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_suite_report_markdown(report), encoding="utf-8")
    _write_jsonl(timeline_path, report.get("event_timeline", []))
    _write_jsonl(sample_path, report.get("revisit_samples", []))
    return {
        "json": json_path,
        "markdown": markdown_path,
        "timeline": timeline_path,
        "samples": sample_path,
    }


def _build_role_report(
    root: Path,
    role: str,
    workload_by_request: dict[str, RuntimeWorkloadRow],
) -> dict[str, Any]:
    run_dir = root / role
    request_rows = _read_jsonl(run_dir / "request_results.jsonl")
    runtime_events = _read_jsonl(run_dir / "runtime_events_raw.jsonl")
    runtime_structured = _read_jsonl(run_dir / "runtime_structured_events.jsonl")
    commands = _read_jsonl(run_dir / "astrakv_runtime_commands.jsonl")
    receipts = _read_jsonl(run_dir / "runtime_command_receipts.jsonl")
    request_by_id = {str(item.get("request_id") or ""): item for item in request_rows}
    revisit_samples: list[dict[str, Any]] = []
    for request_id, workload_row in workload_by_request.items():
        if str((workload_row.metadata or {}).get("phase") or "") != "anchor_revisit":
            continue
        if str((workload_row.metadata or {}).get("cycle_kind") or "") != "sample":
            continue
        request_result = request_by_id.get(request_id, {})
        revisit_start_ms = _request_start_ms(request_result)
        cache_id = workload_row.cache_key or workload_row.prefix_id
        related_events = [
            event for event in runtime_events
            if str(event.get("object_key") or "") == cache_id
        ]
        prior_events = [
            event for event in related_events
            if revisit_start_ms is None or _event_ts_ms(event) < revisit_start_ms
        ]
        valid_sample = (
            _as_float(request_result.get("ttft_ms")) is not None
            and _request_succeeded(request_result)
            and _has_gpu_departure(prior_events)
            and (role == "baseline" or _has_prefetch(prior_events))
        )
        revisit_samples.append(
            {
                "schema": SAMPLE_SCHEMA,
                "request_id": request_id,
                "cache_id": cache_id,
                "role": role,
                "phase": "anchor_revisit",
                "cycle_index": _as_int((workload_row.metadata or {}).get("cycle_index")),
                "anchor_id": str((workload_row.metadata or {}).get("anchor_id") or ""),
                "reference_request_id": str((workload_row.metadata or {}).get("reference_request_id") or ""),
                "ttft_ms": _as_float(request_result.get("ttft_ms")),
                "latency_ms": _as_float(request_result.get("latency_ms")),
                "request_started_ms": revisit_start_ms,
                "valid_sample": valid_sample,
                "request_status": str(request_result.get("status") or "missing"),
                "evicted_before_revisit": _has_gpu_departure(prior_events),
                "prefetch_before_revisit": _has_prefetch(prior_events),
                "prior_event_count": len(prior_events),
                "prior_events": [
                    {
                        "event": str(event.get("action") or ""),
                        "status": str(event.get("status") or ""),
                        "tier_before": str(event.get("tier_before") or ""),
                        "tier_after": str(event.get("tier_after") or ""),
                        "ts_ms": _event_ts_ms(event),
                    }
                    for event in prior_events
                ],
            }
        )

    valid_ttfts = [sample["ttft_ms"] for sample in revisit_samples if sample["valid_sample"] and sample["ttft_ms"] is not None]
    return {
        "run_dir": str(run_dir),
        "request_count": len(request_rows),
        "runtime_event_count": len(runtime_events),
        "runtime_structured_event_count": len(runtime_structured),
        "runtime_command_count": len(commands),
        "runtime_receipt_count": len(receipts),
        "revisit_samples": revisit_samples,
        "valid_revisit_sample_count": len(valid_ttfts),
        "ttft_ms": {
            "mean": _mean(valid_ttfts),
            "p50": _percentile(valid_ttfts, 50),
            "p90": _percentile(valid_ttfts, 90),
        },
        "notes": _role_notes(role, revisit_samples, request_rows, runtime_events),
        "request_rows": request_rows,
        "runtime_events": runtime_events,
        "runtime_structured_events": runtime_structured,
        "runtime_commands": commands,
        "runtime_receipts": receipts,
    }


def _build_comparison(role_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = role_reports["baseline"]["ttft_ms"]
    variant = role_reports["variant"]["ttft_ms"]
    mean_delta = _delta(baseline.get("mean"), variant.get("mean"))
    p50_delta = _delta(baseline.get("p50"), variant.get("p50"))
    p90_delta = _delta(baseline.get("p90"), variant.get("p90"))
    verdict = "inconclusive"
    if (
        role_reports["baseline"]["valid_revisit_sample_count"] > 0
        and role_reports["variant"]["valid_revisit_sample_count"] > 0
        and all(
            isinstance(value, float) and value > 0.0
            for value in (mean_delta, p50_delta, p90_delta)
        )
    ):
        verdict = "variant_better"
    elif (
        role_reports["baseline"]["valid_revisit_sample_count"] == 0
        or role_reports["variant"]["valid_revisit_sample_count"] == 0
    ):
        verdict = "missing_valid_samples"
    return {
        "verdict": verdict,
        "ttft_delta_ms": {
            "mean": mean_delta,
            "p50": p50_delta,
            "p90": p90_delta,
        },
        "summary": _comparison_summary(verdict, role_reports),
    }


def _build_event_timeline(role_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for role, report in role_reports.items():
        request_phase_map = {
            str(row.get("request_id") or ""): _timeline_phase_from_request_row(row)
            for row in report["request_rows"]
        }
        for row in report["request_rows"]:
            request_id = str(row.get("request_id") or "")
            timeline.append(
                {
                    "schema": TIMELINE_SCHEMA,
                    "role": role,
                    "source": "request",
                    "request_id": request_id,
                    "cache_id": str(row.get("cache_key") or row.get("prefix_id") or ""),
                    "phase": request_phase_map.get(request_id, ""),
                    "event": "request_completed",
                    "status": str(row.get("status") or ""),
                    "ttft_ms": _as_float(row.get("ttft_ms")),
                    "ts_ms": _request_start_ms(row),
                    "tier_before": "",
                    "tier_after": "",
                    "metadata": {
                        "latency_ms": _as_float(row.get("latency_ms")),
                        "case": row.get("case", ""),
                    },
                }
            )
        for event in report["runtime_events"]:
            request_id = str(event.get("request_id") or "")
            timeline.append(
                {
                    "schema": TIMELINE_SCHEMA,
                    "role": role,
                    "source": "cache",
                    "request_id": request_id,
                    "cache_id": str(event.get("object_key") or ""),
                    "phase": request_phase_map.get(request_id, ""),
                    "event": str(event.get("action") or ""),
                    "status": str(event.get("status") or ""),
                    "ttft_ms": None,
                    "ts_ms": _event_ts_ms(event),
                    "tier_before": str(event.get("tier_before") or ""),
                    "tier_after": str(event.get("tier_after") or ""),
                    "metadata": dict(event.get("metadata") or {}),
                }
            )
    timeline.sort(key=lambda item: (float(item.get("ts_ms") or 0.0), str(item.get("role") or ""), str(item.get("source") or "")))
    return timeline


def _role_notes(
    role: str,
    revisit_samples: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    if not request_rows:
        notes.append("missing request_results.jsonl")
    if not runtime_events:
        notes.append("missing runtime_events_raw.jsonl")
    if role == "variant" and not any(sample["prefetch_before_revisit"] for sample in revisit_samples):
        notes.append("no sample showed a prefetch event before revisit")
    if not any(sample["evicted_before_revisit"] for sample in revisit_samples):
        notes.append("no sample showed GPU departure evidence before revisit")
    if not any(sample["valid_sample"] for sample in revisit_samples):
        notes.append("no revisit sample satisfied the controlled-evidence filter")
    return notes


def _comparison_summary(verdict: str, role_reports: dict[str, dict[str, Any]]) -> str:
    if verdict == "variant_better":
        return "AstraKV ON beat the OFF baseline on mean, p50, and p90 TTFT across valid revisit samples."
    if verdict == "missing_valid_samples":
        return (
            "At least one role produced zero valid revisit samples. Check whether anchors actually left GPU and, "
            "for the ON run, whether prefetch events were emitted before revisit."
        )
    return (
        "The paired runs completed, but the valid revisit subset did not show a clean ON-over-OFF improvement "
        "on mean, p50, and p90 TTFT at the same time."
    )


def _timeline_phase_from_request_row(row: dict[str, Any]) -> str:
    case = str(row.get("case") or "")
    if "_anchor_seed_" in case or case.endswith("anchor_seed"):
        return "anchor_seed"
    if "_anchor_revisit_" in case or case.endswith("anchor_revisit"):
        return "anchor_revisit"
    if "_churn_primary_" in case or case.endswith("churn_primary"):
        return "churn_primary"
    if "_churn_secondary_" in case or case.endswith("churn_secondary"):
        return "churn_secondary"
    return case


def _request_start_ms(row: dict[str, Any]) -> float | None:
    started_s = _as_float(row.get("request_started_s"))
    if started_s is None:
        return None
    return started_s * 1000.0


def _event_ts_ms(event: dict[str, Any]) -> float | None:
    timestamp_ns = _as_int(event.get("timestamp_ns"))
    if timestamp_ns is None:
        return None
    return timestamp_ns / 1_000_000.0


def _request_succeeded(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "") == "ok"


def _has_gpu_departure(events: list[dict[str, Any]]) -> bool:
    for event in events:
        action = str(event.get("action") or "")
        status = str(event.get("status") or "")
        if status not in COMPLETED_STATUSES:
            continue
        tier_before = str(event.get("tier_before") or "")
        tier_after = str(event.get("tier_after") or "")
        if action in {"offload", "evict", "drop"}:
            return True
        if tier_before == "gpu" and tier_after not in {"", "gpu", "unknown"}:
            return True
    return False


def _has_prefetch(events: list[dict[str, Any]]) -> bool:
    for event in events:
        if str(event.get("action") or "") == "prefetch" and str(event.get("status") or "") in COMPLETED_STATUSES:
            return True
    return False


def _discover_workload_path(root: Path) -> Path:
    candidates = (
        root / "workload" / "policy_ab_ttft_workload.jsonl",
        root / "policy_ab_ttft_workload.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"policy_ab_ttft_workload.jsonl not found under {root}")


def _prompt_spec(prefix: str, index: int, prompt_tokens: int, theme: str) -> dict[str, str]:
    label = f"{prefix}{index}"
    prefix_id = f"policy-ab-{prefix.lower()}-{index:02d}"
    return {
        "label": label,
        "prefix_id": prefix_id,
        "prompt": _build_long_prompt(label=label, theme=theme, prompt_tokens=prompt_tokens),
    }


def _build_long_prompt(*, label: str, theme: str, prompt_tokens: int) -> str:
    theme_words = [item.strip().lower() for item in theme.split() if item.strip()]
    header_words = [
        "astrakv",
        "policy",
        "ab",
        "validation",
        label.lower(),
        "dedicated",
        "prefix",
        "memory",
        "context",
    ]
    body_words = list(header_words)
    segment_index = 0
    while len(body_words) < prompt_tokens:
        chosen = theme_words[segment_index % len(theme_words)]
        next_word = theme_words[(segment_index + 3) % len(theme_words)]
        body_words.extend(
            [
                label.lower(),
                chosen,
                "context",
                next_word,
                "sequence",
                chosen,
                "analysis",
                label.lower(),
                next_word,
                "detail",
            ]
        )
        segment_index += 1
    prompt_body = " ".join(body_words[:prompt_tokens])
    return (
        f"{prompt_body}\n\n"
        "Reply with exactly one short acknowledgement token."
    )


def _workload_row(
    *,
    request_id: str,
    prompt: str,
    prefix_id: str,
    arrival_index: int,
    reuse_ratio: float,
    reuse_bucket: str,
    case: str,
    context_length: int,
    expected_output_tokens: int,
    batch_size: int,
    metadata: dict[str, Any],
    sleep_before_s: float | None = None,
) -> dict[str, Any]:
    row = RuntimeWorkloadRow(
        request_id=request_id,
        prompt=prompt,
        prefix_id=prefix_id,
        prefix_hash=prefix_id,
        cache_key=prefix_id,
        arrival_index=arrival_index,
        reuse_ratio=reuse_ratio,
        reuse_bucket=reuse_bucket,
        context_length=context_length,
        expected_output_tokens=expected_output_tokens,
        batch_size=batch_size,
        sleep_before_s=sleep_before_s,
        case=case,
        metadata=dict(metadata),
    )
    return row.to_record()


def _count_by(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = key_fn(row)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _workload_report_markdown(bundle: PolicyAbWorkloadBundle) -> str:
    manifest = bundle.manifest
    phase_counts = manifest.get("phase_counts") or {}
    lines = [
        "# Policy A/B TTFT Workload",
        "",
        "## Parameters",
        "",
        f"- Anchors: `{manifest['anchor_count']}`",
        f"- Churn primary prompts: `{manifest['churn_primary_count']}`",
        f"- Churn secondary prompts: `{manifest['churn_secondary_count']}`",
        f"- Warm-up cycles: `{manifest['warmup_cycles']}`",
        f"- Sample cycles: `{manifest['sample_cycles']}`",
        f"- Prompt length target: `{manifest['prompt_tokens']}` tokens",
        f"- Idle before revisit: `{manifest['idle_seconds']}` seconds",
        f"- Output tokens: `{manifest['expected_output_tokens']}`",
        "",
        "## Phase Counts",
        "",
        "| phase | rows |",
        "| --- | ---: |",
    ]
    for phase, count in sorted(phase_counts.items()):
        lines.append(f"| {phase} | {count} |")
    lines.extend(
        [
            "",
            "## Cycle Shape",
            "",
            "Each cycle is `anchor_seed -> churn_primary -> churn_secondary -> idle -> anchor_revisit`.",
            "Only sample-cycle revisits count toward the final TTFT comparison.",
        ]
    )
    return "\n".join(lines)


def _suite_report_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison") or {}
    roles = report.get("roles") or {}
    lines = [
        "# Policy A/B TTFT Report",
        "",
        f"- Verdict: `{comparison.get('verdict', 'inconclusive')}`",
        f"- Summary: {comparison.get('summary', '')}",
        "",
        "## Valid Revisit TTFT",
        "",
        "| role | valid samples | mean ms | p50 ms | p90 ms | notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for role in ROLES:
        item = roles.get(role) or {}
        ttft = item.get("ttft_ms") or {}
        notes = "; ".join(item.get("notes") or [])
        lines.append(
            f"| {role} | {item.get('valid_revisit_sample_count', 0)} | "
            f"{_fmt(ttft.get('mean'))} | {_fmt(ttft.get('p50'))} | {_fmt(ttft.get('p90'))} | {notes} |"
        )
    delta = comparison.get("ttft_delta_ms") or {}
    lines.extend(
        [
            "",
            "## ON minus OFF Improvement",
            "",
            "| metric | OFF - ON delta ms |",
            "| --- | ---: |",
            f"| mean | {_fmt(delta.get('mean'))} |",
            f"| p50 | {_fmt(delta.get('p50'))} |",
            f"| p90 | {_fmt(delta.get('p90'))} |",
            "",
            "## Evidence Filter",
            "",
            "- A revisit sample is valid only if TTFT is present and the anchor showed pre-revisit GPU departure evidence.",
            "- The ON role additionally requires a prefetch event before the revisit request starts.",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(item) for item in values if item is not None]
    if not filtered:
        return None
    return mean(filtered)


def _percentile(values: list[float | None], pct: int) -> float | None:
    filtered = sorted(float(item) for item in values if item is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    rank = max(0, math.ceil((pct / 100.0) * len(filtered)) - 1)
    return filtered[min(rank, len(filtered) - 1)]


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _as_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.3f}"
