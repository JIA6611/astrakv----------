"""Build a per-request mainline evidence report for the evict-B story chain.

The mainline is:
    ingress(prefetch-A) -> lookup(load/recompute) -> release(evict) -> gap(prefetch-B)

Every segment is read from run artifacts in a state directory:
  - kv_core_prefetch_tickets.jsonl        (prefetch-A tickets)
  - kv_core_policy_decisions.jsonl        (lookup load-vs-recompute decisions)
  - astrakv_runtime_commands.jsonl        (online policy commands, owner of request_id)
  - runtime_command_receipts.jsonl        (action receipts, joined to commands)
  - online_profile_checkpoint.json        (optional controller/profile context)

Outputs a markdown story report plus a JSONL chain for downstream aggregation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_str(value: Any) -> str:
    return "" if value is None else str(value)


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def compact_action(record: dict[str, Any]) -> dict[str, Any]:
    meta = metadata(record)
    return {
        "action": as_str(record.get("action")),
        "status": as_str(record.get("status")),
        "request_id": as_str(record.get("request_id")),
        "object_key": as_str(record.get("object_key")),
        "backend_object_id": as_str(record.get("backend_object_id")),
        "decision_id": as_str(record.get("decision_id")),
        "command_id": as_str(record.get("command_id")),
        "timestamp_ns": as_int(record.get("timestamp_ns") or record.get("issued_at_ns")),
        "reason": as_str(meta.get("reason")),
        "decision_source": as_str(meta.get("decision_source")),
        "evict_ready": bool(meta.get("evict_ready")),
        "evict_pressure_over": bool(meta.get("evict_pressure_over")),
        "evict_cold_score": as_float(meta.get("evict_cold_score")),
        "runtime_confidence": as_float(meta.get("runtime_confidence")),
        "policy_reuse_frequency": as_float(meta.get("policy_reuse_frequency")),
        "prefetch_waste_count": as_int(meta.get("prefetch_waste_count")),
        "evict_pressure_snapshot": dict(meta.get("evict_pressure_snapshot") or {}),
    }


def build_chain(
    *,
    commands: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    tickets: list[dict[str, Any]],
    policy_decisions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt_by_command = {
        as_str(row.get("command_id")): row
        for row in receipts
        if as_str(row.get("command_id"))
    }
    command_by_id = {as_str(row.get("command_id")): row for row in commands if as_str(row.get("command_id"))}

    def merged_actions() -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for command in commands:
            cid = as_str(command.get("command_id"))
            if cid in seen:
                continue
            seen.add(cid)
            row = compact_action(command)
            receipt = receipt_by_command.get(cid)
            if receipt is not None:
                row["status"] = as_str(receipt.get("status") or row["status"])
                row["receipt_id"] = as_str(receipt.get("receipt_id"))
                row["timestamp_ns"] = as_int(receipt.get("timestamp_ns") or row["timestamp_ns"])
            merged.append(row)
        for cid, receipt in receipt_by_command.items():
            if cid not in seen:
                row = compact_action(receipt)
                command = command_by_id.get(cid)
                if command is not None:
                    row["request_id"] = as_str(command.get("request_id") or row["request_id"])
                    row["object_key"] = as_str(command.get("object_key") or row["object_key"])
                    row["reason"] = as_str(metadata(command).get("reason") or row["reason"])
                merged.append(row)
        return merged

    actions = merged_actions()
    chains: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "request_id": "",
        "prefetch_a": [],
        "lookup": [],
        "release": [],
        "prefetch_b": [],
        "earliest_ns": 0,
        "latest_ns": 0,
    })
    for row in actions:
        rid = as_str(row.get("request_id"))
        if not rid:
            continue
        chain = chains[rid]
        chain["request_id"] = rid
        action = row["action"]
        ts = row["timestamp_ns"]
        if action in {"evict", "drop", "offload"}:
            chain["release"].append(row)
        elif action == "prefetch":
            chain["prefetch_b"].append(row)
        if ts:
            chain["earliest_ns"] = min(chain["earliest_ns"], ts) if chain["earliest_ns"] else ts
            chain["latest_ns"] = max(chain["latest_ns"], ts)

    for row in tickets:
        rid = as_str(row.get("target_request_id") or row.get("consumer_request_id") or "")
        if not rid:
            continue
        chain = chains[rid]
        chain["request_id"] = rid
        chain["prefetch_a"].append({
            "prefetch_id": as_str(row.get("prefetch_id")),
            "physical_object_id": as_str(row.get("physical_object_id")),
            "status": as_str(row.get("status")),
            "requested_bytes": as_int(row.get("requested_bytes")),
            "completed_bytes": as_int(row.get("completed_bytes")),
            "consumer_request_id": as_str(row.get("consumer_request_id")),
            "failure_reason": as_str(row.get("failure_reason")),
        })
        ts = as_int(row.get("deadline_ns"))
        if ts:
            chain["earliest_ns"] = min(chain["earliest_ns"], ts) if chain["earliest_ns"] else ts
            chain["latest_ns"] = max(chain["latest_ns"], ts)

    for row in policy_decisions:
        rid = as_str(row.get("request_id"))
        if not rid:
            continue
        chain = chains[rid]
        chain["request_id"] = rid
        chain["lookup"].append({
            "action": as_str(row.get("action")),
            "reason": as_str(row.get("reason")),
            "candidate_ssd_read_bytes": as_int(row.get("candidate_ssd_read_bytes")),
            "load_cost_ms": as_float(row.get("load_cost_ms")),
            "recompute_cost_ms": as_float(row.get("recompute_cost_ms")),
            "queue_delay_ms": as_float(row.get("queue_delay_ms")),
            "profile_matched": bool(row.get("profile_matched")),
            "timestamp_ns": as_int(row.get("timestamp_ns")),
        })
        ts = as_int(row.get("timestamp_ns"))
        if ts:
            chain["earliest_ns"] = min(chain["earliest_ns"], ts) if chain["earliest_ns"] else ts
            chain["latest_ns"] = max(chain["latest_ns"], ts)

    ordered = {
        rid: chains[rid]
        for rid in sorted(chains, key=lambda item: chains[item]["earliest_ns"] or 0)
    }
    summary = summarize(ordered)
    return ordered, summary


def summarize(chains: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prefetch_a_status: Counter[str] = Counter()
    lookup_action: Counter[str] = Counter()
    release_action: Counter[str] = Counter()
    release_status: Counter[str] = Counter()
    prefetch_b_action: Counter[str] = Counter()
    prefetch_b_status: Counter[str] = Counter()
    evict_completed: list[dict[str, Any]] = []
    evict_decisions_source: Counter[str] = Counter()
    cold_scores: list[float] = []
    pressure_fractions: list[float] = []
    for chain in chains.values():
        for item in chain["prefetch_a"]:
            prefetch_a_status[as_str(item.get("status"))] += 1
        for item in chain["lookup"]:
            lookup_action[as_str(item.get("action"))] += 1
        for item in chain["release"]:
            release_action[as_str(item.get("action"))] += 1
            release_status[f"{item['action']}:{item['status']}"] += 1
            if item["action"] == "evict":
                evict_decisions_source[as_str(item.get("decision_source") or "online_profile")] += 1
                cold_scores.append(item["evict_cold_score"])
                snapshot = item.get("evict_pressure_snapshot") or {}
                pressure_fractions.extend([
                    as_float(snapshot.get("cpu_usage_fraction")),
                    as_float(snapshot.get("ssd_usage_fraction")),
                ])
                if item["status"] == "completed":
                    evict_completed.append(item)
        for item in chain["prefetch_b"]:
            prefetch_b_action[as_str(item.get("action"))] += 1
            prefetch_b_status[f"{item['action']}:{item['status']}"] += 1
    return {
        "request_count": len(chains),
        "requests_with_prefetch_a": sum(1 for c in chains.values() if c["prefetch_a"]),
        "requests_with_lookup": sum(1 for c in chains.values() if c["lookup"]),
        "requests_with_release": sum(1 for c in chains.values() if c["release"]),
        "requests_with_prefetch_b": sum(1 for c in chains.values() if c["prefetch_b"]),
        "prefetch_a_ticket_status": dict(prefetch_a_status),
        "lookup_action_counts": dict(lookup_action),
        "release_action_status": dict(release_status),
        "prefetch_b_action_status": dict(prefetch_b_status),
        "evict_completed_receipts": len(evict_completed),
        "evict_decision_sources": dict(evict_decisions_source),
        "evict_cold_score_mean": round(sum(cold_scores) / len(cold_scores), 4) if cold_scores else None,
        "evict_cold_score_min": round(min(cold_scores), 4) if cold_scores else None,
        "evict_cold_score_max": round(max(cold_scores), 4) if cold_scores else None,
        "evict_pressure_fraction_mean": (
            round(sum(pressure_fractions) / len(pressure_fractions), 4) if pressure_fractions else None
        ),
    }


def render_markdown(
    chains: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    *,
    run_id: str,
    state_dir: Path,
) -> str:
    lines = [
        "# evict-B 主线证据链报告",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- run_id: `{run_id}`",
        f"- state_dir: `{state_dir}`",
        "",
        "## 证据边界",
        "",
        "本报告只把每个环节的**观测与收据**串成链，不声称端到端时延因果；"
        "Belady 只在离线模拟层作为理论上界。",
        "",
        "## 汇总",
        "",
        "| 环节 | 有证据的请求数 | 说明 |",
        "| --- | ---: | --- |",
        f"| ingress(prefetch-A) | {summary['requests_with_prefetch_a']} | kv_core_prefetch_tickets |",
        f"| lookup(load/recompute) | {summary['requests_with_lookup']} | kv_core_policy_decisions |",
        f"| release(evict/drop/offload) | {summary['requests_with_release']} | commands+receipts |",
        f"| gap(prefetch-B) | {summary['requests_with_prefetch_b']} | prefetch receipts |",
        "",
        f"- evict completed receipts: **{summary['evict_completed_receipts']}**",
        f"- evict 决策来源: `{summary['evict_decision_sources']}`",
        f"- evict 冷度评分 mean/min/max: "
        f"`{summary['evict_cold_score_mean']} / {summary['evict_cold_score_min']} / {summary['evict_cold_score_max']}`",
        f"- evict 压力占比 mean: `{summary['evict_pressure_fraction_mean']}`",
        "",
        "## 逐请求主线",
        "",
        "| request_id | prefetch-A | lookup | release | prefetch-B |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rid, chain in chains.items():
        lines.append(
            f"| `{rid}` | "
            f"{len(chain['prefetch_a'])} ({', '.join(sorted(set(i['status'] for i in chain['prefetch_a'])))}) | "
            f"{len(chain['lookup'])} ({', '.join(sorted(set(i['action'] for i in chain['lookup'])))}) | "
            f"{len(chain['release'])} ({', '.join(sorted(set(i['action'] + ':' + i['status'] for i in chain['release'])))}) | "
            f"{len(chain['prefetch_b'])} ({', '.join(sorted(set(i['action'] + ':' + i['status'] for i in chain['prefetch_b'])))}) |"
        )
    lines.extend(["", "## evict-B 明细（completed receipts）", ""])
    evict_rows = [
        item
        for chain in chains.values()
        for item in chain["release"]
        if item["action"] == "evict" and item["status"] == "completed"
    ]
    if not evict_rows:
        lines.append("（当前 state_dir 中没有 evict completed receipts——这是需要真机跑通的目标。）")
    else:
        lines.append("| request_id | object_key | cold_score | confidence | pressure_over | reason |")
        lines.append("| --- | --- | ---: | ---: | --- | --- |")
        for item in evict_rows:
            lines.append(
                f"| `{item['request_id']}` | `{item['object_key']}` | {item['evict_cold_score']} | "
                f"{item['runtime_confidence']} | {item['evict_pressure_over']} | {item['reason']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path, help="Runtime artifact state directory")
    parser.add_argument("--output", type=Path, help="Markdown report path (default state_dir/evict_mainline_report.md)")
    parser.add_argument("--jsonl-output", type=Path, help="JSONL chain path (default state_dir/evict_mainline_chain.jsonl)")
    args = parser.parse_args()

    state_dir = args.state_dir
    commands = load_jsonl(state_dir / "astrakv_runtime_commands.jsonl")
    receipts = load_jsonl(state_dir / "runtime_command_receipts.jsonl")
    tickets = load_jsonl(state_dir / "kv_core_prefetch_tickets.jsonl")
    policy_decisions = load_jsonl(state_dir / "kv_core_policy_decisions.jsonl")
    checkpoint = load_checkpoint(state_dir / "online_profile_checkpoint.json")

    run_id = as_str(checkpoint.get("run_id") or (commands[0].get("run_id") if commands else ""))
    chains, summary = build_chain(
        commands=commands,
        receipts=receipts,
        tickets=tickets,
        policy_decisions=policy_decisions,
    )

    markdown = render_markdown(chains, summary, run_id=run_id, state_dir=state_dir)
    output = args.output or (state_dir / "evict_mainline_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    jsonl_output = args.jsonl_output or (state_dir / "evict_mainline_chain.jsonl")
    with jsonl_output.open("w", encoding="utf-8") as handle:
        for rid, chain in chains.items():
            handle.write(json.dumps({
                "schema": "astrakv-evict-mainline-chain-v1",
                "run_id": run_id,
                "request_id": rid,
                "prefetch_a": chain["prefetch_a"],
                "lookup": chain["lookup"],
                "release": chain["release"],
                "prefetch_b": chain["prefetch_b"],
                "earliest_ns": chain["earliest_ns"],
                "latest_ns": chain["latest_ns"],
            }, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"evict mainline report written to {output}")
    print(f"evict mainline chain written to {jsonl_output}")
    print(f"summary: {json.dumps(summary, ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
