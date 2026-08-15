"""Validate Profile-B transfer provenance, anti-leakage, and execution evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-profile-transfer-acceptance-v1"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def analyze(root: Path) -> dict[str, Any]:
    subset = _load_json(root / "subset/transfer_subset_manifest.json")
    hints = _load_json(root / "hints/prefix_prefetch_hint_report.json")
    profile_acceptance = _load_json(root / "profile_acceptance.json")
    ttft = _load_json(
        root / "profile-test/qasper/predicted_far_ttft_summary.json"
    )
    profile_path = root / "profile/profile_db.json"
    variant_state = root / "profile-test/qasper/variant-state"
    baseline_state = root / "profile-test/qasper/baseline-state"
    variant_auth = _load_jsonl(
        variant_state / "predictive_prefetch_authorizations.jsonl"
    )
    baseline_auth = _load_jsonl(
        baseline_state / "predictive_prefetch_authorizations.jsonl"
    )
    variant_origins = sorted({
        str(row.get("prefetch_origin") or "") for row in variant_auth
        if str(row.get("prefetch_origin") or "")
    })
    runtime_env_path = variant_state / "runtime.env"
    runtime_env = (
        runtime_env_path.read_text(encoding="utf-8", errors="replace")
        if runtime_env_path.is_file() else ""
    )
    nonempty_sidecar_lines = [
        line for line in runtime_env.splitlines()
        if line.startswith("ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH=")
        and line.partition("=")[2].strip()
    ]
    hint_source = str(hints.get("workload_jsonl") or "")
    anti_leakage = subset.get("anti_leakage") or {}
    functional = profile_acceptance.get("functional_acceptance") or {}
    prefetch = profile_acceptance.get("prefetch_b") or {}
    checks = {
        "subset_anti_leakage_passed": anti_leakage.get("passed") is True,
        "request_id_overlap_zero": not anti_leakage.get(
            "train_test_original_request_id_overlap"
        ),
        "prompt_hash_overlap_zero": not anti_leakage.get(
            "train_test_prompt_hash_overlap"
        ),
        "shared_prefix_count_matches_limit": (
            int(anti_leakage.get("shared_prefix_count") or 0)
            == int(subset.get("limit") or 0)
            > 0
        ),
        "hints_are_train_derived": (
            bool(hint_source)
            and "/train-run/" in hint_source.replace("\\", "/")
            and "/profile-test/" not in hint_source.replace("\\", "/")
            and "/subset/test/" not in hint_source.replace("\\", "/")
        ),
        "hint_count_covers_selected_prefixes": (
            int(hints.get("hint_count") or 0)
            >= int(anti_leakage.get("shared_prefix_count") or 0)
            > 0
        ),
        "profile_db_exists": profile_path.is_file() and profile_path.stat().st_size > 0,
        "test_sidecar_absent": not nonempty_sidecar_lines,
        "baseline_authorizations_zero": not baseline_auth,
        "variant_authorizations_present": bool(variant_auth),
        "variant_origins_profile_only": variant_origins == ["profile_b"],
        "functional_acceptance_passed": functional.get("passed") is True,
        "completed_and_consumed_present": (
            int(prefetch.get("completed") or 0) > 0
            and int(prefetch.get("consumed_ticket_count") or 0) > 0
        ),
    }
    return {
        "schema": SCHEMA,
        "root": str(root),
        "passed": all(checks.values()),
        "checks": checks,
        "anti_leakage": anti_leakage,
        "provenance": {
            "profile_db": str(profile_path),
            "hint_source": hint_source,
            "hint_count": int(hints.get("hint_count") or 0),
            "test_sidecar_lines": nonempty_sidecar_lines,
        },
        "execution": {
            "baseline_authorization_count": len(baseline_auth),
            "variant_authorization_count": len(variant_auth),
            "variant_origins": variant_origins,
            "completed": int(prefetch.get("completed") or 0),
            "consumed_ticket_count": int(prefetch.get("consumed_ticket_count") or 0),
            "completed_bytes": int(prefetch.get("completed_bytes") or 0),
        },
        "native_load": profile_acceptance.get("native_load") or {},
        "ttft": ttft,
    }


def _markdown(data: dict[str, Any]) -> str:
    anti = data["anti_leakage"]
    execution = data["execution"]
    native = data["native_load"]
    native_baseline = native.get("baseline") or {}
    native_variant = native.get("variant") or {}
    ttft = data["ttft"]
    lines = [
        "# Profile-B train/test transfer acceptance",
        "",
        f"overall: **{'PASS' if data['passed'] else 'FAIL'}**",
        "",
        "## Anti-leakage and provenance",
        "",
        f"- selected shared prefixes: {anti.get('shared_prefix_count', 0)}",
        f"- train/test request-id overlap: {len(anti.get('train_test_original_request_id_overlap') or [])}",
        f"- train/test prompt-hash overlap: {len(anti.get('train_test_prompt_hash_overlap') or [])}",
        f"- hint source: `{data['provenance']['hint_source']}`",
        f"- test sidecar present: {'yes' if data['provenance']['test_sidecar_lines'] else 'no'}",
        "",
        "## Functional execution",
        "",
        f"- baseline authorizations: {execution['baseline_authorization_count']}",
        f"- variant authorizations: {execution['variant_authorization_count']} {execution['variant_origins']}",
        f"- completed: {execution['completed']}",
        f"- consumed tickets: {execution['consumed_ticket_count']}",
        f"- completed bytes: {execution['completed_bytes']}",
        "",
        "## Far-request performance",
        "",
        f"- TTFT P50: {ttft.get('baseline_p50_ms')} ms -> {ttft.get('variant_p50_ms')} ms "
        f"({ttft.get('p50_delta_percent')}%)",
        f"- TTFT P95: {ttft.get('baseline_p95_ms')} ms -> {ttft.get('variant_p95_ms')} ms "
        f"({ttft.get('p95_delta_percent')}%)",
        f"- TTFT paired wins: {ttft.get('variant_wins')}/{ttft.get('paired_count')}",
        f"- TTFT P50 bootstrap 95% CI: {ttft.get('p50_delta_bootstrap_ci_percent')}",
        f"- native load P50: {native_baseline.get('p50_ms')} ms -> {native_variant.get('p50_ms')} ms",
        f"- native paired median delta: {native.get('paired_median_delta_percent')}%",
        f"- native paired wins: {native.get('variant_wins')}/{native.get('paired_count')}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in data["checks"].items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    data = analyze(root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(data), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(_markdown(data))
    if args.require_pass and not data["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
