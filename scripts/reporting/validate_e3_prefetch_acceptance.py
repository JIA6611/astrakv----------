"""Aggregate Prefetch-A/B action evidence from an E3 prefetch suite run.

The E3 suite copies Prefetch-A artifacts (``kv_core_policy_decisions.jsonl``,
``kv_core_prefetch_tickets.jsonl``) into each ``repeat-*/<role>/`` run dir,
but Prefetch-B receipts stay in the runtime state dir because the E3 copy
list does not include ``runtime_command_receipts.jsonl``.  This checker reads
both locations and reports:

- A: ``prefetch_ssd_to_cpu`` decisions (submitted/rejected) and ticket statuses
  (completed/consumed/wasted/...);
- B: ``action=prefetch`` receipts (completed with bytes, no-op, failures) with
  their ``failure_reason`` and CPU capacity/pressure diagnostics.

Strict per-cell acceptance (B completed receipt in B-only cell, A consumed
ticket in A-only cell) is enforced by ``validate_prefetch_2x2_ablation.py``;
this checker is the E3-side evidence aggregator and failure diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-e3-prefetch-acceptance-v1"
ROLES = ("baseline", "variant")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _aggregate_role(role_dir: Path, state_dir: Path, role: str) -> dict[str, Any]:
    decisions_path = _first_existing(
        role_dir / "kv_core_policy_decisions.jsonl",
        state_dir / "kv_core_policy_decisions.jsonl",
    )
    tickets_path = _first_existing(
        role_dir / "kv_core_prefetch_tickets.jsonl",
        state_dir / "kv_core_prefetch_tickets.jsonl",
    )
    receipts_path = _first_existing(
        state_dir / "runtime_command_receipts.jsonl",
        role_dir / "runtime_command_receipts.jsonl",
    )
    decisions = _read_jsonl(decisions_path) if decisions_path is not None else []
    tickets = _read_jsonl(tickets_path) if tickets_path is not None else []
    receipts = _read_jsonl(receipts_path) if receipts_path is not None else []

    a_decisions = [
        row for row in decisions
        if str(row.get("action") or "") == "prefetch_ssd_to_cpu"
    ]
    a_submitted = sum(1 for row in a_decisions if str(row.get("status") or "") == "submitted")
    a_rejected = Counter(
        str(row.get("reason") or "unknown")
        for row in a_decisions
        if str(row.get("status") or "") == "rejected"
    )
    ticket_statuses = Counter(str(row.get("status") or "unknown") for row in tickets)
    consumed = ticket_statuses.get("consumed", 0)

    b_receipts = [row for row in receipts if str(row.get("action") or "") == "prefetch"]
    b_completed = [
        row for row in b_receipts
        if str(row.get("status") or "") == "completed"
        and int(row.get("metadata", {}).get("prefetched") or 0) > 0
    ]
    b_failed = [
        row for row in b_receipts
        if str(row.get("status") or "") != "completed"
    ]
    failure_reasons: Counter[str] = Counter()
    diagnostics_seen = 0
    for row in b_failed:
        metadata = row.get("metadata") or {}
        failure_reasons[str(metadata.get("failure_reason") or "unknown")] += 1
        if any(
            key in metadata
            for key in ("cpu_used_bytes", "cpu_capacity_bytes", "cpu_prefetch_budget_bytes", "memory_pressure")
        ):
            diagnostics_seen += 1

    return {
        "role": role,
        "prefetch_a": {
            "decision_count": len(a_decisions),
            "submitted": a_submitted,
            "rejected_reasons": a_rejected,
            "ticket_statuses": ticket_statuses,
            "tickets_consumed": consumed,
        },
        "prefetch_b": {
            "receipt_count": len(b_receipts),
            "completed_with_bytes": len(b_completed),
            "failed_count": len(b_failed),
            "failure_reasons": failure_reasons,
            "failed_with_diagnostics": diagnostics_seen,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-root", required=True, help="E3 suite output directory.")
    parser.add_argument("--output", default="", help="Summary JSON path (default stdout).")
    args = parser.parse_args()

    root = Path(args.e3_root)
    if not root.is_dir():
        raise SystemExit(f"E3 root is not a directory: {root}")

    repeats = sorted(
        path for path in root.glob("repeat-*")
        if path.is_dir() and any((path / role).is_dir() for role in ROLES)
    )
    rows: list[dict[str, Any]] = []
    for repeat in repeats:
        for role in ROLES:
            role_dir = repeat / role
            state_dir = repeat / role / "state"
            if role_dir.is_dir():
                rows.append(_aggregate_role(role_dir, state_dir, role))

    variant_rows = [row for row in rows if row["role"] == "variant"]
    summary = {
        "schema": SCHEMA,
        "e3_root": str(root),
        "repeat_count": len(repeats),
        "roles": rows,
        "acceptance": {
            "a_consumed_ticket_found": any(
                row["prefetch_a"]["tickets_consumed"] > 0 for row in variant_rows
            ),
            "b_completed_receipt_found": any(
                row["prefetch_b"]["completed_with_bytes"] > 0 for row in variant_rows
            ),
            "b_failures_have_diagnostics": all(
                row["prefetch_b"]["failed_count"] == 0
                or row["prefetch_b"]["failed_with_diagnostics"] == row["prefetch_b"]["failed_count"]
                for row in variant_rows
            ),
        },
        "note": (
            "B strict acceptance is also enforced cell-wise by "
            "validate_prefetch_2x2_ablation.py; the E3 cold single-prefix "
            "workload may legitimately show zero B receipts when no sidecar/"
            "profile evidence is present."
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
