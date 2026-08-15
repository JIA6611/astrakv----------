"""Acceptance summary for a Prefetch-B experiment run.

Consumes one experiment directory produced by run_prefetch_B_experiments.sh
(``sidecar/`` or ``profile-test/``) and emits the acceptance table:

  - workload structure (fire-consume confirmed via visit gaps)
  - TTFT P50/P95: baseline vs variant (overall and per visit bucket)
  - Prefetch-B functional: receipts completed / prefetched=1 / bytes, failures
  - consumption evidence: LMCache external hits, vLLM local tokens
  - disk read deltas per role
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    i = max(0, min(len(values) - 1, int(round(len(values) * p / 100.0 - 0.5))))
    return values[i]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def visit_buckets(canonical: Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    """Classify each request index as first/far/near for fire-consume schedules."""
    rows = [json.loads(l) for l in canonical.read_text(encoding="utf-8").splitlines() if l.strip()]
    visits: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        visits[str(r.get("cache_key") or r.get("prefix_id") or "")].append(i)
    bucket: dict[int, str] = {}
    for indices in visits.values():
        indices = sorted(indices)
        if len(indices) == 1:
            bucket[indices[0]] = "first"
            continue
        bucket[indices[0]] = "first"
        prev = indices[0]
        for idx in indices[1:]:
            if idx - prev <= 1:
                bucket[prev] = "far"
                bucket[idx] = "near"
            else:
                bucket[idx] = "far"
            prev = idx
    return bucket, dict(visits)


def _benchmark_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(root: Path, exp: str) -> dict[str, Any]:
    exp_root = root / exp / "qasper"
    measured = exp_root / "materialized/qasper_grouped_exact_next_measured_workload.jsonl"
    canonical = (
        measured if measured.is_file()
        else exp_root / "materialized/qasper_grouped_exact_next_canonical_workload.jsonl"
    )
    buckets, visits = (visit_buckets(canonical) if canonical.is_file() else ({}, {}))
    workload_rows = _load_jsonl(canonical)
    far_request_ids = {
        str(row.get("request_id") or "")
        for row in workload_rows
        if str(
            row.get("prefetch_phase")
            or (row.get("metadata") or {}).get("prefetch_phase")
            or ""
        ) == "far"
    }

    def ttft_by_role() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for role in ("baseline", "variant"):
            rows = _benchmark_rows(exp_root / role / "benchmark_results.csv")
            overall: list[float] = []
            by_bucket: dict[str, list[float]] = defaultdict(list)
            disk: list[float] = []
            for row in rows:
                if str(row.get("status") or "") != "ok":
                    continue
                import re
                m = re.search(r"_(\d+)$", row.get("case") or "")
                idx = int(m.group(1)) if m else None
                try:
                    tt = float(row.get("ttft_ms"))
                except (TypeError, ValueError):
                    continue
                overall.append(tt)
                if idx is not None:
                    by_bucket[buckets.get(idx, "other")].append(tt)
                try:
                    disk.append(float(row.get("disk_read_delta_mb") or 0))
                except (TypeError, ValueError):
                    pass
            out[role] = {
                "n": len(overall),
                "ttft_p50_ms": _percentile(overall, 50),
                "ttft_p95_ms": _percentile(overall, 95),
                "disk_read_delta_sum_mb": round(sum(disk), 1),
                "by_bucket": {
                    k: {"p50_ms": _percentile(v, 50), "p95_ms": _percentile(v, 95), "n": len(v)}
                    for k, v in sorted(by_bucket.items())
                },
            }
        return out

    ttft = ttft_by_role()
    base = ttft.get("baseline", {})
    var = ttft.get("variant", {})
    delta_p50 = (
        (var["ttft_p50_ms"] / base["ttft_p50_ms"] - 1) * 100
        if base.get("ttft_p50_ms") and var.get("ttft_p50_ms")
        else None
    )
    delta_p95 = (
        (var["ttft_p95_ms"] / base["ttft_p95_ms"] - 1) * 100
        if base.get("ttft_p95_ms") and var.get("ttft_p95_ms")
        else None
    )

    state = exp_root / "variant-state"
    legacy_rows = _load_jsonl(state / "runtime_command_receipts.jsonl")
    if not legacy_rows:
        legacy_rows = _load_jsonl(exp_root / "variant/runtime_command_receipts.jsonl")
    receipts = [
        row for row in legacy_rows
        if str(row.get("action") or "") == "prefetch"
    ]
    completed = [r for r in receipts if str(r.get("status") or "") == "completed"]
    prefetched1 = [r for r in completed if int((r.get("metadata") or {}).get("prefetched") or 0) > 0]
    failures = Counter(
        str((r.get("metadata") or {}).get("failure_reason") or "unknown")
        for r in receipts
        if str(r.get("status") or "") != "completed"
    )
    authorizations = _load_jsonl(
        state / "predictive_prefetch_authorizations.jsonl"
    )
    authorized_requests = {
        str(row.get("request_id") or ""): str(row.get("prefetch_origin") or "")
        for row in authorizations
        if str(row.get("request_id") or "")
        and str(row.get("prefetch_origin") or "") in {"sidecar_b", "profile_b"}
    }
    decisions = [
        row for row in _load_jsonl(state / "kv_core_policy_decisions.jsonl")
        if str(row.get("action") or "") == "prefetch_ssd_to_cpu"
        and str(row.get("request_id") or "") in authorized_requests
        and str(row.get("prefetch_origin") or "") == authorized_requests.get(
            str(row.get("request_id") or ""),
        )
    ]
    latest_tickets: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(state / "kv_core_prefetch_tickets.jsonl"):
        if str(row.get("target_request_id") or "") not in authorized_requests:
            continue
        prefetch_id = str(row.get("prefetch_id") or "")
        if prefetch_id:
            latest_tickets[prefetch_id] = row
    predictive_tickets = list(latest_tickets.values())
    completed_tickets = [
        row for row in predictive_tickets
        if str(row.get("status") or "") in {"completed", "consumed"}
        and int(row.get("completed_bytes") or 0) > 0
    ]
    consumed_tickets = [
        row for row in completed_tickets
        if str(row.get("status") or "") == "consumed"
    ]
    for row in predictive_tickets:
        if str(row.get("status") or "") in {
            "cancelled", "failed", "expired", "wasted",
        }:
            failures[str(row.get("failure_reason") or row.get("status") or "unknown")] += 1

    lookups: list[dict[str, Any]] = []
    nc = state / "kv_core_native_callbacks.jsonl"
    if nc.is_file():
        lookups = [
            r for r in _load_jsonl(nc)
            if str(r.get("callback") or "") == "scheduler_exact_lookup"
        ]
    lookup_hit = sum(1 for r in lookups if int(r.get("lookup_hit_tokens") or 0) > 0)
    locally_cached = sum(1 for r in lookups if int(r.get("locally_cached_tokens") or 0) > 0)

    native_load_by_role: dict[str, dict[str, float]] = {}
    native_load_summary: dict[str, dict[str, Any]] = {}
    for role in ("baseline", "variant"):
        rows = [
            row for row in _load_jsonl(
                exp_root / f"{role}-state/kv_core_native_receipts.jsonl"
            )
            if str(row.get("logical_request_id") or "") in far_request_ids
            and str(row.get("status") or "") == "completed"
            and int(row.get("load_latency_ns") or 0) > 0
        ]
        by_request = {
            str(row.get("logical_request_id") or ""): (
                float(row.get("load_latency_ns") or 0) / 1_000_000.0
            )
            for row in rows
        }
        native_load_by_role[role] = by_request
        values = list(by_request.values())
        native_load_summary[role] = {
            "n": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "mean_ms": (sum(values) / len(values)) if values else None,
            "bytes_loaded": sum(int(row.get("bytes_loaded") or 0) for row in rows),
            "prefetch_id_count": sum(bool(row.get("prefetch_id")) for row in rows),
        }
    common_native_ids = sorted(
        set(native_load_by_role["baseline"]) & set(native_load_by_role["variant"])
    )
    native_pair_deltas = [
        (
            native_load_by_role["variant"][request_id]
            / native_load_by_role["baseline"][request_id]
            - 1.0
        ) * 100.0
        for request_id in common_native_ids
        if native_load_by_role["baseline"][request_id] > 0
    ]

    baseline_state = exp_root / "baseline-state"
    baseline_authorization_count = len(_load_jsonl(
        baseline_state / "predictive_prefetch_authorizations.jsonl"
    ))
    baseline_ticket_count = len({
        str(row.get("prefetch_id") or "")
        for row in _load_jsonl(baseline_state / "kv_core_prefetch_tickets.jsonl")
        if str(row.get("prefetch_id") or "")
    })
    submitted_decision_count = sum(
        1 for row in decisions if str(row.get("status") or "") == "submitted"
    )
    predictive_authorization_count = len(authorized_requests)
    functional_checks = {
        "predictive_authorization_present": predictive_authorization_count > 0,
        "one_submitted_decision_per_authorization": (
            submitted_decision_count == predictive_authorization_count
        ),
        "one_ticket_per_authorization": (
            len(predictive_tickets) == predictive_authorization_count
        ),
        "all_tickets_completed": (
            len(completed_tickets) == predictive_authorization_count
        ),
        "all_tickets_consumed": (
            len(consumed_tickets) == predictive_authorization_count
        ),
        "completed_bytes_positive": (
            sum(int(row.get("completed_bytes") or 0) for row in completed_tickets) > 0
        ),
        "no_prefetch_failures": not failures,
        "baseline_has_no_predictive_authorization": baseline_authorization_count == 0,
        "baseline_has_no_predictive_ticket": baseline_ticket_count == 0,
    }

    return {
        "schema": "astrakv-prefetch-B-acceptance-v2",
        "experiment": exp,
        "root": str(root),
        "workload": {
            "rows": sum(len(v) for v in visits.values()),
            "unique_objects": len(visits),
            "multi_visit_objects": sum(1 for v in visits.values() if len(v) > 1),
            "fire_consume_near_gaps": sum(
                1 for v in visits.values() if any(b - a <= 1 for a, b in zip(sorted(v), sorted(v)[1:]))
            ),
        },
        "ttft": {"baseline": base, "variant": var, "delta_p50_percent": delta_p50, "delta_p95_percent": delta_p95},
        "prefetch_b": {
            "receipt_count": len(receipts) + len(predictive_tickets),
            "completed": len(completed) + len(completed_tickets),
            "completed_prefetched_1": len(prefetched1) + len(completed_tickets),
            "completed_bytes": sum(
                int((row.get("metadata") or {}).get("bytes") or 0)
                for row in prefetched1
            ) + sum(int(row.get("completed_bytes") or 0) for row in completed_tickets),
            "authorization_count": len(authorizations),
            "authorization_origins": dict(Counter(authorized_requests.values())),
            "submitted_decision_count": submitted_decision_count,
            "ticket_count": len(predictive_tickets),
            "consumed_ticket_count": len(consumed_tickets),
            "failure_reasons": dict(failures),
        },
        "functional_acceptance": {
            "passed": all(functional_checks.values()),
            "checks": functional_checks,
            "baseline_authorization_count": baseline_authorization_count,
            "baseline_ticket_count": baseline_ticket_count,
        },
        "consumption": {
            "lookups": len(lookups),
            "lmcache_external_hit_count": lookup_hit,
            "vllm_local_cached_count": locally_cached,
        },
        "native_load": {
            "phase": "far",
            "baseline": native_load_summary["baseline"],
            "variant": native_load_summary["variant"],
            "paired_count": len(native_pair_deltas),
            "variant_wins": sum(delta < 0 for delta in native_pair_deltas),
            "paired_median_delta_percent": _percentile(native_pair_deltas, 50),
            "paired_deltas_percent": native_pair_deltas,
        },
    }


def _markdown(data: dict[str, Any]) -> str:
    t = data["ttft"]
    b, v = t["baseline"], t["variant"]
    lines = [
        f"# Prefetch-B acceptance — {data['experiment']}",
        "",
        f"workload: {data['workload']['rows']} requests, {data['workload']['unique_objects']} objects, "
        f"multi-visit {data['workload']['multi_visit_objects']}",
        "",
        "| role | n | TTFT P50 (ms) | TTFT P95 (ms) | disk read sum (MB) |",
        "|---|---|---:|---:|---:|",
        f"| baseline | {b.get('n')} | {_fmt(b.get('ttft_p50_ms'))} | {_fmt(b.get('ttft_p95_ms'))} | {b.get('disk_read_delta_sum_mb')} |",
        f"| variant | {v.get('n')} | {_fmt(v.get('ttft_p50_ms'))} | {_fmt(v.get('ttft_p95_ms'))} | {v.get('disk_read_delta_sum_mb')} |",
        "",
        f"delta: P50 {_fmt(t.get('delta_p50_percent'))}% , P95 {_fmt(t.get('delta_p95_percent'))}%",
        "",
        "## Per visit bucket (P50 ms)",
        "| bucket | baseline | variant |",
        "|---|---:|---:|",
    ]
    buckets = set((b.get("by_bucket") or {})) | set((v.get("by_bucket") or {}))
    for k in sorted(buckets):
        bb = (b.get("by_bucket") or {}).get(k, {})
        vv = (v.get("by_bucket") or {}).get(k, {})
        lines.append(f"| {k} | {_fmt(bb.get('p50_ms'))} (n={bb.get('n')}) | {_fmt(vv.get('p50_ms'))} (n={vv.get('n')}) |")
    p = data["prefetch_b"]
    functional = data["functional_acceptance"]
    native = data["native_load"]
    lines += [
        "",
        "## Prefetch-B functional",
        f"- receipts: {p['receipt_count']}, completed: {p['completed']}, completed&prefetched=1: {p['completed_prefetched_1']}",
        f"- authorizations: {p.get('authorization_count', 0)} {p.get('authorization_origins', {})}; "
        f"submitted decisions: {p.get('submitted_decision_count', 0)}; consumed tickets: {p.get('consumed_ticket_count', 0)}",
        f"- failures: {p['failure_reasons']}",
        f"- functional acceptance: {'PASS' if functional['passed'] else 'FAIL'}",
        "",
        "## Native far-request KV load",
        f"- baseline P50: {_fmt(native['baseline']['p50_ms'])} ms; "
        f"variant P50: {_fmt(native['variant']['p50_ms'])} ms",
        f"- paired median delta: {_fmt(native['paired_median_delta_percent'])}%; "
        f"variant wins: {native['variant_wins']}/{native['paired_count']}",
        "",
        "## Consumption",
        f"- lookups: {data['consumption']['lookups']}, LMCache external hits: {data['consumption']['lmcache_external_hit_count']}, "
        f"vLLM local: {data['consumption']['vllm_local_cached_count']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="B experiment output root (prefetch-B-*).")
    parser.add_argument("--exp", choices=("sidecar", "profile-test"), required=True)
    parser.add_argument("--output", default="", help="Markdown output path (default stdout).")
    parser.add_argument(
        "--require-functional-pass", action="store_true",
        help="Exit non-zero unless the complete B authorization/decision/ticket/consumption chain passes.",
    )
    args = parser.parse_args()

    data = analyze(Path(args.root), args.exp)
    report = _markdown(data)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        (out.with_suffix(".json")).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    print(json.dumps(data, ensure_ascii=False, sort_keys=True))
    if args.require_functional_pass and not data["functional_acceptance"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
