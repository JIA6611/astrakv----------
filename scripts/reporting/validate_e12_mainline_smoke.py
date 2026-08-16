"""Validate one real-machine E12 end-to-end control-chain audit smoke.

E12 is a functional audit, not a performance experiment.  A passing chain is:

    Prefetch-A submitted and consumed
      -> explicit load/recompute lookup decision
      -> completed RELEASE lifecycle event
      -> Prefetch-B command with a completed, positive-byte receipt

The first three stages describe one anchor request.  Prefetch-B is attributed
to that request as the source of a gap-time action; it may benefit a later
request and therefore is not described as consumption by the anchor request.
Native CPU capacity eviction is reported as an independent branch because the
current E11 implementation runs at LMCache's native capacity-reclaim point,
not at the request RELEASE callback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LOOKUP_ACTIONS = frozenset({"admit_external_prefix", "recompute"})
TERMINAL_SUCCESS = frozenset({"completed", "consumed", "executed", "ok", "success"})


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_str(value: Any) -> str:
    return "" if value is None else str(value)


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def artifact_path(run_dir: Path, state_dir: Path, name: str) -> Path | None:
    for root in (run_dir, state_dir):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_prefetch_bytes(receipt: dict[str, Any]) -> int:
    # BackendActionReceipt.to_record() exposes the actual transfer size at the
    # top level. metadata.prefetched is only a 0/1 outcome flag and must never
    # be interpreted as a byte count in an audit.
    return as_int(receipt.get("bytes"))


def _latest_by_id(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = as_str(row.get(key))
        if identity:
            result[identity] = row
    return result


def _receipt_matches_command(command: dict[str, Any], receipt: dict[str, Any]) -> bool:
    if not receipt or as_str(command.get("command_id")) != as_str(receipt.get("command_id")):
        return False
    for key in ("run_id", "action", "binding_id", "backend_object_id"):
        left = as_str(command.get(key))
        right = as_str(receipt.get(key))
        if not left or not right or left != right:
            return False
    for key in ("decision_id", "request_id"):
        left = as_str(command.get(key))
        right = as_str(receipt.get(key))
        if right and left != right:
            return False
    command_generation = as_int(command.get("binding_generation"))
    receipt_generation = as_int(receipt.get("binding_generation"))
    return not (command_generation and receipt_generation and command_generation != receipt_generation)


def _compact_a(decision: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": as_str(decision.get("request_id")),
        "target_request_id": as_str(ticket.get("target_request_id")),
        "prefetch_id": as_str(decision.get("prefetch_id") or ticket.get("prefetch_id")),
        "physical_object_id": as_str(
            decision.get("physical_object_id") or ticket.get("physical_object_id")
        ),
        "binding_generation": as_int(
            decision.get("binding_generation") or ticket.get("binding_generation")
        ),
        "decision_status": as_str(decision.get("status")),
        "ticket_status": as_str(ticket.get("status")),
        "completed_bytes": as_int(ticket.get("completed_bytes")),
        "consumer_request_id": as_str(ticket.get("consumer_request_id")),
        "timestamp_ns": as_int(decision.get("timestamp_ns")),
    }


def _compact_lookup(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": as_str(row.get("request_id")),
        "physical_object_id": as_str(row.get("physical_object_id")),
        "binding_generation": as_int(row.get("binding_generation")),
        "action": as_str(row.get("action")),
        "reason": as_str(row.get("reason")),
        "candidate_external_tokens": as_int(row.get("candidate_external_tokens")),
        "candidate_evidence_present": any(
            key in row
            for key in ("candidate_external_tokens", "load_cost_ms", "recompute_cost_ms")
        ),
        "timestamp_ns": as_int(row.get("timestamp_ns")),
    }


def _compact_accounting_lookup(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize the terminal native lookup ledger into lookup evidence.

    The scheduler callback writes the authoritative load/recompute outcome to
    request accounting.  It is intentionally separate from policy decisions:
    a native lookup can allocate and complete an external load even when the
    policy process did not emit a duplicate ``admit_external_prefix`` row.
    Only terminal rows with positive load or confirmed recompute are admitted.
    """
    allocated = as_int(row.get("allocated_external_tokens"))
    loaded = as_int(row.get("actual_loaded_tokens"))
    recomputed = as_int(row.get("recomputed_tokens"))
    confirmed = row.get("recompute_confirmed") is True
    terminal = row.get("terminal") is True
    action = ""
    if terminal and allocated > 0 and loaded > 0:
        action = "admit_external_prefix"
    elif terminal and confirmed and recomputed > 0:
        action = "recompute"
    request_id = as_str(row.get("logical_request_id")) or as_str(row.get("request_id"))
    return {
        "request_id": request_id,
        "physical_object_id": as_str(row.get("physical_object_id")),
        "binding_generation": as_int(row.get("binding_generation")),
        "action": action,
        "reason": as_str(row.get("terminal_reason")) or "native_lookup_terminal",
        "candidate_external_tokens": max(
            allocated, recomputed, as_int(row.get("lookup_hit_tokens"))
        ),
        "candidate_evidence_present": bool(
            as_int(row.get("lookup_hit_tokens")) > 0
            and ((allocated > 0 and loaded > 0) or (confirmed and recomputed > 0))
        ),
        "timestamp_ns": as_int(row.get("timestamp_ns")),
    }


def _compact_release(row: dict[str, Any]) -> dict[str, Any]:
    meta = metadata(row)
    return {
        "request_id": as_str(row.get("request_id")),
        "event_id": as_str(row.get("event_id")),
        "object_key": as_str(row.get("object_key")),
        "backend_object_id": as_str(row.get("backend_object_id")),
        "binding_id": as_str(row.get("binding_id") or meta.get("binding_id")),
        "binding_generation": as_int(row.get("binding_generation")),
        "status": as_str(row.get("status")),
        "bridge_eligible": meta.get("bridge_eligible") is True,
        "timestamp_ns": as_int(row.get("timestamp_ns")),
    }


def _compact_b(command: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    meta = metadata(command)
    return {
        "source_request_id": as_str(command.get("request_id")),
        "command_id": as_str(command.get("command_id")),
        "receipt_id": as_str(receipt.get("receipt_id")),
        "object_key": as_str(command.get("object_key")),
        "backend_object_id": as_str(command.get("backend_object_id")),
        "binding_id": as_str(command.get("binding_id")),
        "binding_generation": as_int(command.get("binding_generation")),
        "prefetch_kind": as_str(meta.get("prefetch_kind")),
        "dispatch_origin": as_str(meta.get("dispatch_origin")),
        "status": as_str(receipt.get("status")),
        "moved_bytes": positive_prefetch_bytes(receipt),
        "command_timestamp_ns": as_int(command.get("issued_at_ns") or command.get("timestamp_ns")),
        "receipt_timestamp_ns": as_int(receipt.get("timestamp_ns")),
        "receipt_matches_command": _receipt_matches_command(command, receipt),
        "command_record_type": as_str(command.get("record_type")),
        "receipt_record_type": as_str(receipt.get("record_type")),
    }


def _is_valid_a(row: dict[str, Any]) -> bool:
    return bool(
        row["prefetch_id"]
        and row["physical_object_id"]
        and row["target_request_id"] == row["request_id"]
        and row["decision_status"] == "submitted"
        and row["ticket_status"] == "consumed"
        and row["completed_bytes"] > 0
        and row["consumer_request_id"] == row["request_id"]
        and row["timestamp_ns"] > 0
    )


def _is_valid_lookup(row: dict[str, Any]) -> bool:
    return bool(
        row["action"] in LOOKUP_ACTIONS
        and row["physical_object_id"]
        and row["binding_generation"] > 0
        and row["reason"]
        and row["candidate_evidence_present"]
        and row["timestamp_ns"] > 0
    )


def _is_valid_release(row: dict[str, Any]) -> bool:
    return bool(
        row["status"] in TERMINAL_SUCCESS
        and row["event_id"]
        and row["binding_id"]
        and row["binding_generation"] > 0
        and row["bridge_eligible"]
        and row["timestamp_ns"] > 0
    )


def _is_valid_b(row: dict[str, Any]) -> bool:
    return bool(
        row["command_id"]
        and row["receipt_id"]
        and row["command_record_type"] == "command"
        and row["receipt_record_type"] == "receipt"
        and row["binding_id"]
        and row["binding_generation"] > 0
        and row["prefetch_kind"] in {"next_use", "prefix", "heuristic"}
        and row["dispatch_origin"] in {"release_completed", "offload_completed"}
        and row["status"] in TERMINAL_SUCCESS
        and row["moved_bytes"] > 0
        and row["receipt_matches_command"]
        and row["command_timestamp_ns"] > 0
        and row["receipt_timestamp_ns"] >= row["command_timestamp_ns"]
    )


def _find_complete_chain(chain: dict[str, Any]) -> dict[str, Any] | None:
    request_result = chain["request_result"]
    if as_str(request_result.get("status")) != "ok":
        return None
    valid_a = [row for row in chain["prefetch_a"] if _is_valid_a(row)]
    valid_lookup = [row for row in chain["lookup"] if _is_valid_lookup(row)]
    valid_release = [row for row in chain["release"] if _is_valid_release(row)]
    valid_b = [row for row in chain["prefetch_b"] if _is_valid_b(row)]
    for a_row in sorted(valid_a, key=lambda row: row["timestamp_ns"]):
        for lookup_row in sorted(valid_lookup, key=lambda row: row["timestamp_ns"]):
            if a_row["physical_object_id"] != lookup_row["physical_object_id"]:
                continue
            if a_row["binding_generation"] != lookup_row["binding_generation"]:
                continue
            if a_row["timestamp_ns"] > lookup_row["timestamp_ns"]:
                continue
            for release_row in sorted(valid_release, key=lambda row: row["timestamp_ns"]):
                if lookup_row["timestamp_ns"] > release_row["timestamp_ns"]:
                    continue
                for b_row in sorted(valid_b, key=lambda row: row["command_timestamp_ns"]):
                    if release_row["timestamp_ns"] > b_row["command_timestamp_ns"]:
                        continue
                    if release_row["object_key"] and b_row["object_key"] != release_row["object_key"]:
                        continue
                    return {
                        "request_id": chain["request_id"],
                        "request_result": {
                            "status": as_str(request_result.get("status")),
                        },
                        "prefetch_a": a_row,
                        "lookup": lookup_row,
                        "release": release_row,
                        "prefetch_b": b_row,
                    }
    return None


def validate(run_dir: Path, state_dir: Path) -> dict[str, Any]:
    names = {
        "tickets": "kv_core_prefetch_tickets.jsonl",
        "decisions": "kv_core_policy_decisions.jsonl",
        "lookup_accounting": "kv_core_request_accounting.jsonl",
        "runtime_events": "runtime_events_raw.jsonl",
        "commands": "astrakv_runtime_commands.jsonl",
        "receipts": "runtime_command_receipts.jsonl",
        "native_installation": "native_policy_installation.jsonl",
        "native_evictions": "native_cache_policy_evictions.jsonl",
        "request_results": "request_results.jsonl",
    }
    paths = {key: artifact_path(run_dir, state_dir, name) for key, name in names.items()}
    rows = {key: read_jsonl(path) for key, path in paths.items()}
    ticket_by_id = _latest_by_id(rows["tickets"], "prefetch_id")
    receipt_by_command = _latest_by_id(rows["receipts"], "command_id")

    chains: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "request_id": "",
            "request_result": {},
            "prefetch_a": [],
            "lookup": [],
            "release": [],
            "prefetch_b": [],
        }
    )
    for request_result in rows["request_results"]:
        request_id = as_str(request_result.get("request_id"))
        if request_id:
            chains[request_id]["request_id"] = request_id
            chains[request_id]["request_result"] = request_result
    for decision in rows["decisions"]:
        request_id = as_str(decision.get("request_id"))
        if not request_id:
            continue
        chain = chains[request_id]
        chain["request_id"] = request_id
        action = as_str(decision.get("action"))
        if action == "prefetch_ssd_to_cpu":
            ticket = ticket_by_id.get(as_str(decision.get("prefetch_id")), {})
            chain["prefetch_a"].append(_compact_a(decision, ticket))
        elif action in LOOKUP_ACTIONS:
            chain["lookup"].append(_compact_lookup(decision))

    # Native scheduler lookup accounting is the terminal source of truth for
    # external bytes actually loaded or recompute explicitly confirmed.
    for accounting in rows["lookup_accounting"]:
        lookup = _compact_accounting_lookup(accounting)
        if lookup["request_id"] and lookup["action"]:
            chain = chains[lookup["request_id"]]
            chain["request_id"] = lookup["request_id"]
            chain["lookup"].append(lookup)

    for event in rows["runtime_events"]:
        if as_str(event.get("record_type")) != "event" or as_str(event.get("action")) != "release":
            continue
        request_id = as_str(event.get("request_id"))
        if not request_id:
            continue
        chain = chains[request_id]
        chain["request_id"] = request_id
        chain["release"].append(_compact_release(event))

    for command in rows["commands"]:
        if as_str(command.get("action")) != "prefetch":
            continue
        request_id = as_str(command.get("request_id"))
        if not request_id:
            continue
        chain = chains[request_id]
        chain["request_id"] = request_id
        chain["prefetch_b"].append(
            _compact_b(command, receipt_by_command.get(as_str(command.get("command_id")), {}))
        )

    complete_chains = [
        match
        for request_id in sorted(chains)
        if (match := _find_complete_chain(chains[request_id])) is not None
    ]
    required_missing = [
        names[key]
        for key in (
            "tickets", "decisions", "runtime_events", "commands", "receipts", "request_results"
        )
        if paths[key] is None
    ]
    observed_run_ids = sorted({
        as_str(row.get("run_id"))
        for key in ("runtime_events", "commands", "receipts", "request_results")
        for row in rows[key]
        if as_str(row.get("run_id"))
    })
    errors: list[str] = []
    invalid_errors: list[str] = []
    if required_missing:
        invalid_errors.append("missing_required_artifacts:" + ",".join(required_missing))
    if len(observed_run_ids) != 1:
        invalid_errors.append("run_id_not_unique")
    runtime_row_groups = ("runtime_events", "commands", "receipts", "request_results")
    if any(not as_str(row.get("run_id")) for key in runtime_row_groups for row in rows[key]):
        invalid_errors.append("run_id_missing_from_runtime_row")
    errors.extend(invalid_errors)
    if not complete_chains:
        errors.append("no_complete_control_chain")
    validation_status = "INVALID" if invalid_errors else ("PASS" if not errors else "INCOMPLETE")

    native_selected = sum(
        1 for row in rows["native_evictions"] if as_str(row.get("status")) == "selected"
    )
    native_completed = sum(
        1 for row in rows["native_evictions"] if as_str(row.get("status")) == "completed"
    )
    native_installed = any(
        as_str(row.get("status")) == "installed" for row in rows["native_installation"]
    )
    artifacts = {
        key: {
            "path": "" if path is None else str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows[key]),
        }
        for key, path in paths.items()
    }
    return {
        "schema": "astrakv-e12-mainline-audit-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_scope": "functional_control_chain_only",
        "eligible": not errors,
        "validation_status": validation_status,
        "errors": errors,
        "run_id": observed_run_ids[0] if len(observed_run_ids) == 1 else "",
        "run_id_values": observed_run_ids,
        "complete_chain_count": len(complete_chains),
        "complete_chains": complete_chains,
        "request_chain_count": len(chains),
        "stage_request_counts": {
            "request_ok": sum(
                as_str(chain["request_result"].get("status")) == "ok"
                for chain in chains.values()
            ),
            "prefetch_a": sum(bool(chain["prefetch_a"]) for chain in chains.values()),
            "lookup": sum(bool(chain["lookup"]) for chain in chains.values()),
            "release": sum(bool(chain["release"]) for chain in chains.values()),
            "prefetch_b": sum(bool(chain["prefetch_b"]) for chain in chains.values()),
        },
        "native_capacity_branch": {
            "in_pass_criteria": False,
            "installation_observed": native_installed,
            "selected_count": native_selected,
            "completed_count": native_completed,
            "reason": (
                "E11 native CPU capacity eviction is asynchronous to request RELEASE and is reported separately"
            ),
        },
        "artifacts": artifacts,
    }


def render_markdown(result: dict[str, Any]) -> str:
    status = result["validation_status"]
    lines = [
        "# E12 端到端控制链审计",
        "",
        f"- 判定: **{status}**",
        f"- run_id: `{result['run_id']}`",
        f"- 完整链数量: **{result['complete_chain_count']}**",
        "- 证据范围: 仅证明控制链可串接，不证明 TTFT、吞吐或容量收益",
        "",
        "## 严格链路",
        "",
        "`Prefetch-A submitted+consumed -> lookup load/recompute -> RELEASE completed -> Prefetch-B command+completed receipt`",
        "",
        "Prefetch-B 的 request_id 表示触发 gap 动作的源请求；它可能服务后续请求，不解释成源请求消费。",
        "",
        "## 分段覆盖",
        "",
        "| 阶段 | 有记录的 request 数 |",
        "| --- | ---: |",
    ]
    for key, value in result["stage_request_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 完整链", ""])
    if not result["complete_chains"]:
        lines.append("没有 request 同时满足四段、身份和时间顺序门禁。")
    else:
        lines.extend([
            "| request_id | A ticket | lookup | release event | B command/receipt |",
            "| --- | --- | --- | --- | --- |",
        ])
        for chain in result["complete_chains"]:
            lines.append(
                f"| `{chain['request_id']}` | `{chain['prefetch_a']['prefetch_id']}` | "
                f"`{chain['lookup']['action']}` | `{chain['release']['event_id']}` | "
                f"`{chain['prefetch_b']['command_id']}` / `{chain['prefetch_b']['receipt_id']}` |"
            )
    native = result["native_capacity_branch"]
    lines.extend([
        "",
        "## Native CPU 容量回收旁支",
        "",
        f"- selector 安装事件: `{native['installation_observed']}`",
        f"- selected/completed: `{native['selected_count']} / {native['completed_count']}`",
        "- 此旁支不进入 E12 PASS；其性能与适用边界由 E11 判定。",
        "",
        "## 拒绝原因",
        "",
    ])
    lines.extend([f"- `{error}`" for error in result["errors"]] or ["- 无"])
    lines.extend(["", "## Artifact", "", "| 名称 | 行数 | SHA-256 | 路径 |", "| --- | ---: | --- | --- |"])
    for key, item in result["artifacts"].items():
        lines.append(f"| {key} | {item['row_count']} | `{item['sha256']}` | `{item['path']}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()
    result = validate(args.run_dir, args.state_dir)
    markdown_path = args.output or args.run_dir / "e12_mainline_audit.md"
    json_path = args.json_output or args.run_dir / "e12_mainline_audit.json"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"E12 audit report written to {markdown_path}")
    print(f"E12 audit manifest written to {json_path}")
    return 2 if args.require_eligible and not result["eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
