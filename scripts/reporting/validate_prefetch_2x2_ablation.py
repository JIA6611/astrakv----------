"""Aggregate the Prefetch-A / Prefetch-B 2x2 ablation evidence.

The grouped exact-next ablation produces, per dataset and per role
(baseline = B off, variant = B on), two artifact families:

- run dir (``<root>/<dataset>/<role>/``): ``runtime_command_receipts.jsonl``
  (Prefetch-B receipts, exported by the benchmark runner);
- state dir (``<root>/<dataset>/<role>-state/``): ``kv_core_policy_decisions.jsonl``
  and ``kv_core_prefetch_tickets.jsonl`` (Prefetch-A decisions/tickets).

This validator folds the two wrapper runs (A off / A on) into the four cells,
counts B receipts, A decisions and tickets, and emits the conflict counters
that feed the Phase-2 mitigation decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "astrakv-prefetch-2x2-validation-v1"
DATASETS = ("qasper", "multifieldqa_en")
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


def _receipts(run_dir: Path, state_dir: Path) -> list[dict[str, Any]]:
    path = _first_existing(
        run_dir / "runtime_command_receipts.jsonl",
        state_dir / "runtime_command_receipts.jsonl",
    )
    return _read_jsonl(path) if path is not None else []


def _decisions(run_dir: Path, state_dir: Path) -> list[dict[str, Any]]:
    path = _first_existing(
        state_dir / "kv_core_policy_decisions.jsonl",
        run_dir / "kv_core_policy_decisions.jsonl",
    )
    return _read_jsonl(path) if path is not None else []


def _tickets(run_dir: Path, state_dir: Path) -> list[dict[str, Any]]:
    path = _first_existing(
        state_dir / "kv_core_prefetch_tickets.jsonl",
        run_dir / "kv_core_prefetch_tickets.jsonl",
    )
    return _read_jsonl(path) if path is not None else []


def _aggregate_cell(root: Path, dataset: str, role: str) -> dict[str, Any]:
    run_dir = root / dataset / role
    state_dir = root / dataset / f"{role}-state"
    receipts = _receipts(run_dir, state_dir)
    decisions = _decisions(run_dir, state_dir)
    tickets = _tickets(run_dir, state_dir)

    b_receipts = [row for row in receipts if str(row.get("action") or "") == "prefetch"]
    b_completed = [
        row for row in b_receipts
        if str(row.get("status") or "") == "completed" and int(row.get("metadata", {}).get("prefetched") or 0) > 0
    ]
    b_failures = [
        str(row.get("metadata", {}).get("failure_reason") or "unknown")
        for row in b_receipts
        if str(row.get("status") or "") != "completed"
    ]
    b_already_cpu = sum(
        1 for row in b_receipts
        if str(row.get("status") or "") == "completed"
        and int(row.get("metadata", {}).get("prefetched") or 0) == 0
    )

    a_decisions = [
        row for row in decisions
        if str(row.get("action") or "") == "prefetch_ssd_to_cpu"
    ]
    a_submitted = sum(1 for row in a_decisions if str(row.get("status") or "") == "submitted")
    a_rejected = Counter(str(row.get("reason") or "unknown") for row in a_decisions if str(row.get("status") or "") == "rejected")
    invalidations = [
        row for row in decisions
        if str(row.get("action") or "") == "invalidate_external_copy"
    ]
    invalidate_removed = sum(int(row.get("cpu_removed_chunk_count") or 0) for row in invalidations)

    ticket_statuses = Counter(str(row.get("status") or "unknown") for row in tickets)
    consumed = ticket_statuses.get("consumed", 0)
    completed = ticket_statuses.get("completed", 0)
    wasted = ticket_statuses.get("wasted", 0)

    return {
        "dataset": dataset,
        "role": role,
        "prefetch_b": {
            "receipt_count": len(b_receipts),
            "completed_with_bytes": len(b_completed),
            "already_cpu_noop": b_already_cpu,
            "failure_reasons": Counter(b_failures),
        },
        "prefetch_a": {
            "decision_count": len(a_decisions),
            "submitted": a_submitted,
            "rejected_reasons": a_rejected,
            "ticket_statuses": ticket_statuses,
            "tickets_consumed": consumed,
            "tickets_completed": completed,
            "tickets_wasted": wasted,
        },
        "conflict_signals": {
            "invalidate_external_copy_count": len(invalidations),
            "invalidate_removed_chunk_count": invalidate_removed,
            "b_noop_when_a_resident": b_already_cpu,
            "dual_accounting_ticket_consumed_and_b_completed": (
                consumed > 0 and len(b_completed) > 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-off", required=True, help="Output dir of the A-off ablation run.")
    parser.add_argument("--a-on", required=True, help="Output dir of the A-on ablation run.")
    parser.add_argument("--output", default="", help="Summary JSON path (default stdout).")
    args = parser.parse_args()

    a_off = Path(args.a_off)
    a_on = Path(args.a_on)
    if not a_off.is_dir() or not a_on.is_dir():
        raise SystemExit("both --a-off and --a-on must be existing directories")

    cells: list[dict[str, Any]] = []
    for a_enabled, root in ((False, a_off), (True, a_on)):
        for dataset in DATASETS:
            for role in ROLES:
                cell = _aggregate_cell(root, dataset, role)
                cell["a_enabled"] = a_enabled
                cell["b_enabled"] = role == "variant"
                cell["cell"] = f"A{int(a_enabled)}B{int(role == 'variant')}"
                cells.append(cell)

    b_only_cells = [cell for cell in cells if cell["cell"] == "A0B1"]
    a_only_cells = [cell for cell in cells if cell["cell"] == "A1B0"]
    both_cells = [cell for cell in cells if cell["cell"] == "A1B1"]
    conflict_totals = {
        "invalidate_removed_chunk_count": sum(
            cell["conflict_signals"]["invalidate_removed_chunk_count"]
            for cell in both_cells
        ),
        "b_noop_when_a_resident": sum(
            cell["conflict_signals"]["b_noop_when_a_resident"]
            for cell in both_cells
        ),
        "dual_accounting_cells": sum(
            1 for cell in both_cells
            if cell["conflict_signals"]["dual_accounting_ticket_consumed_and_b_completed"]
        ),
    }

    summary = {
        "schema": SCHEMA,
        "a_off_root": str(a_off),
        "a_on_root": str(a_on),
        "cells": cells,
        "acceptance": {
            "cells_present": len(cells) == 8,
            "b_only_completed_receipt": any(
                cell["prefetch_b"]["completed_with_bytes"] > 0
                for cell in b_only_cells
            ),
            "a_only_consumed_ticket": any(
                cell["prefetch_a"]["tickets_consumed"] > 0
                for cell in a_only_cells
            ),
        },
        "both_cell_conflict_totals": conflict_totals,
        "note": (
            "both-cell conflict totals are Phase-2 input evidence; nonzero "
            "values are expected until the unified accounting/invalidate "
            "mitigation lands."
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
