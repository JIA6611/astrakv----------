"""Immutable, tokenizer-backed workflow reuse observation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
REPLAY_SCHEMA = "astrakv-workflow-trace-v1"
REPLAY_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "workflow_id",
        "parent_request_id",
        "subtask_index",
        "arrival_index",
        "adapter",
        "messages",
        "dataset_id",
        "workload_id",
        "model_identifier",
        "tool_output_sha256",
    }
)


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self, messages: Sequence[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class WorkflowTraceRow:
    workflow_id: str
    parent_request_id: str
    subtask_index: int
    arrival_index: int
    messages: tuple[dict[str, str], ...]
    dataset_id: str
    workload_id: str
    adapter: str
    model_identifier: str | None = None
    tool_output_sha256: str | None = None


@dataclass(frozen=True)
class ReuseObservation:
    workflow_id: str
    parent_request_id: str
    subtask_index: int
    arrival_index: int
    dataset_id: str
    workload_id: str
    adapter: str
    token_count: int
    block_size_tokens: int
    block_hashes: tuple[str, ...]
    historical_reused_tokens: int
    historical_reuse_count: int
    kv_bytes_per_token: int
    potential_kv_bytes: int
    model_identifier: str | None = None
    tool_output_sha256: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "evidence_class": "modeled_dataset_metadata",
            "workflow_id": self.workflow_id,
            "parent_request_id": self.parent_request_id,
            "subtask_index": self.subtask_index,
            "arrival_index": self.arrival_index,
            "dataset_id": self.dataset_id,
            "workload_id": self.workload_id,
            "adapter": self.adapter,
            "token_count": self.token_count,
            "block_size_tokens": self.block_size_tokens,
            "block_hashes": list(self.block_hashes),
            "historical_reused_tokens": self.historical_reused_tokens,
            "historical_reuse_count": self.historical_reuse_count,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "potential_kv_bytes": self.potential_kv_bytes,
        }
        if self.model_identifier is not None:
            record["model_identifier"] = self.model_identifier
        if self.tool_output_sha256 is not None:
            record["tool_output_sha256"] = self.tool_output_sha256
        return record


def observe_workflow_rows(
    rows: Iterable[WorkflowTraceRow],
    *,
    tokenizer: ChatTokenizer,
    block_size_tokens: int,
    kv_bytes_per_token: int,
) -> list[ReuseObservation]:
    if block_size_tokens <= 0 or kv_bytes_per_token <= 0:
        raise ValueError("block_size_tokens and kv_bytes_per_token must be positive")

    rows = list(rows)
    _validate_workflow_trace_rows(rows)
    historical_blocks: set[str] = set()
    observations: list[ReuseObservation] = []

    for row in rows:
        tokens = _token_ids(
            tokenizer.apply_chat_template(
                list(row.messages), tokenize=True, add_generation_prompt=True
            )
        )
        if not all(isinstance(token, int) for token in tokens):
            raise ValueError("tokenizer must return integer token IDs")
        blocks = _block_hashes(tokens, block_size_tokens)
        reused = [block for block in blocks if block in historical_blocks]
        reused_tokens = sum(_block_token_count(tokens, index, block_size_tokens) for index, block in enumerate(blocks) if block in historical_blocks)
        observations.append(
            ReuseObservation(
                workflow_id=row.workflow_id,
                parent_request_id=row.parent_request_id,
                subtask_index=row.subtask_index,
                arrival_index=row.arrival_index,
                dataset_id=row.dataset_id,
                workload_id=row.workload_id,
                adapter=row.adapter,
                token_count=len(tokens),
                block_size_tokens=block_size_tokens,
                block_hashes=tuple(blocks),
                historical_reused_tokens=reused_tokens,
                historical_reuse_count=len(reused),
                kv_bytes_per_token=kv_bytes_per_token,
                potential_kv_bytes=reused_tokens * kv_bytes_per_token,
                model_identifier=row.model_identifier,
                tool_output_sha256=row.tool_output_sha256,
            )
        )
        historical_blocks.update(blocks)
    return observations


def parse_workflow_row(record: dict[str, Any]) -> WorkflowTraceRow:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("every message must be an object")
    normalized_messages = tuple(
        {"role": message.get("role"), "content": message.get("content")}
        for message in messages
    )
    return WorkflowTraceRow(
        workflow_id=str(record["workflow_id"]),
        parent_request_id=str(record["parent_request_id"]),
        subtask_index=int(record["subtask_index"]),
        arrival_index=int(record["arrival_index"]),
        messages=normalized_messages,
        dataset_id=str(record["dataset_id"]),
        workload_id=str(record["workload_id"]),
        adapter=str(record.get("adapter", "replay_jsonl")),
        model_identifier=_optional_string(record, "model_identifier"),
        tool_output_sha256=_optional_string(record, "tool_output_sha256"),
    )


def load_replay_workflow_rows(path: str | Path) -> list[WorkflowTraceRow]:
    """Load a recorded callback export without inferring agent-internal payloads."""
    replay_path = Path(path)
    if not replay_path.is_file():
        raise ValueError(f"replay JSONL file not found: {replay_path}")
    rows: list[WorkflowTraceRow] = []
    for line_number, line in enumerate(replay_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid replay JSONL at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"replay record at line {line_number} must be an object")
        rows.append(parse_replay_workflow_record(record))
    _validate_workflow_trace_rows(rows)
    return rows


def parse_replay_workflow_record(record: dict[str, Any]) -> WorkflowTraceRow:
    fields = set(record)
    if fields != REPLAY_REQUIRED_FIELDS:
        raise ValueError("unsupported replay fields")
    if record["schema"] != REPLAY_SCHEMA:
        raise ValueError("unsupported replay schema")
    if record["adapter"] != "replay_jsonl":
        raise ValueError("replay records must use replay_jsonl adapter")
    row = parse_workflow_row(record)
    if not row.model_identifier:
        raise ValueError("replay model_identifier must be non-empty")
    if not row.tool_output_sha256 or not _is_sha256(row.tool_output_sha256):
        raise ValueError("replay tool_output_sha256 must be a SHA-256 digest")
    return row


def task1_prompt_records_to_workflow_rows(
    records: Iterable[dict[str, Any]], *, workload_type: str
) -> list[WorkflowTraceRow]:
    """Map immutable Task 1 prompt rows to the single-request workflow fallback."""
    if workload_type not in {"random", "grouped"}:
        raise ValueError("workload_type must be random or grouped")
    rows: list[WorkflowTraceRow] = []
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Task 1 record messages must be a list")
        request_id = str(record.get("request_id", ""))
        rows.append(
            WorkflowTraceRow(
                workflow_id=request_id,
                parent_request_id=request_id,
                subtask_index=0,
                arrival_index=int(record.get("order")),
                messages=tuple(dict(item) for item in messages if isinstance(item, dict)),
                dataset_id="qasper",
                workload_id=f"qasper-{workload_type}",
                adapter="single_request",
            )
        )
    return rows


def load_task1_prompt_records(
    directory_path: str | Path, *, workload_type: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Read Task 1 prompts verbatim in their published random/grouped order."""
    if workload_type not in {"random", "grouped"}:
        raise ValueError("workload_type must be random or grouped")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    path = Path(directory_path) / "prompts" / f"qasper_{workload_type}_prompts.jsonl"
    if not path.is_file():
        raise ValueError(f"Task 1 prompt file not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records if limit is None else records[:limit]


def _validate_row(row: WorkflowTraceRow) -> None:
    if not row.workflow_id or not row.parent_request_id or not row.dataset_id or not row.workload_id:
        raise ValueError("workflow identifiers must be non-empty")
    if row.subtask_index < 0 or row.arrival_index < 0:
        raise ValueError("indices must be non-negative")
    if not row.messages:
        raise ValueError("messages must be non-empty")
    for message in row.messages:
        if set(message) != {"role", "content"}:
            raise ValueError("messages must contain exactly role and content")
        if message["role"] not in ALLOWED_MESSAGE_ROLES:
            raise ValueError("message role is not supported")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("message content must be non-empty")


def _validate_workflow_trace_rows(rows: Sequence[WorkflowTraceRow]) -> None:
    seen_identities: set[tuple[str, int]] = set()
    seen_arrivals: set[int] = set()
    previous_arrival = -1
    for row in rows:
        _validate_row(row)
        identity = (row.workflow_id, row.subtask_index)
        if identity in seen_identities:
            raise ValueError("duplicate workflow/subtask identity")
        if row.arrival_index in seen_arrivals:
            raise ValueError("duplicate arrival index")
        if row.arrival_index <= previous_arrival:
            raise ValueError("arrival_index must be strictly increasing")
        seen_identities.add(identity)
        seen_arrivals.add(row.arrival_index)
        previous_arrival = row.arrival_index


def _optional_string(record: dict[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when provided")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _token_ids(tokenized: Any) -> list[int]:
    """Normalize tokenizer BatchEncoding output without accepting ambiguous payloads."""
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if isinstance(tokenized, (str, bytes)) or not isinstance(tokenized, Sequence):
        raise ValueError("tokenizer must return integer token IDs")
    return list(tokenized)


def _block_hashes(tokens: Sequence[int], block_size_tokens: int) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(tokens[start : start + block_size_tokens], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for start in range(0, len(tokens), block_size_tokens)
    ]


def _block_token_count(tokens: Sequence[int], block_index: int, block_size_tokens: int) -> int:
    start = block_index * block_size_tokens
    return min(block_size_tokens, len(tokens) - start)
