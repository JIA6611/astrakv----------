"""Losslessly adapt task-one QASPER ZIP workloads to the canonical contract."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .runtime_workload import RuntimeWorkloadRow, RUNTIME_WORKLOAD_SCHEMA_VERSION


TASK1_SCHEMA = "astrakv-task1-qasper-adapter-v1"
_PROMPT_PATH = "prompts/qasper_{workload}_prompts.jsonl"
_METADATA_PATH = "metadata/qasper_metadata.jsonl"
_REQUIRED = ("request_id", "sample_id", "prompt", "order", "reuse_group", "ground_truth")


class Task1QasperAdapterError(ValueError):
    """Raised when the supplied task-one archive cannot be adapted safely."""


@dataclass(frozen=True, slots=True)
class Task1QasperWorkload:
    workload_type: str
    rows: tuple[RuntimeWorkloadRow, ...]
    audit: dict[str, Any]


def load_task1_qasper_directory(
    directory_path: str | Path, workload_type: str
) -> Task1QasperWorkload:
    """Load the supplied immutable Task 1 directory package without rewriting it."""
    if workload_type not in {"random", "grouped"}:
        raise Task1QasperAdapterError("workload_type must be random or grouped")
    root = Path(directory_path)
    prompt_path = root / _PROMPT_PATH.format(workload=workload_type)
    required = (
        prompt_path,
        root / _METADATA_PATH,
        root / "metadata/qasper_manifest.json",
        root / "metadata/qasper_sha256.json",
        root / f"validation/{workload_type}_prompt_validation.json",
        root / f"validation/{workload_type}_prompt_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Task1QasperAdapterError(f"task-one directory missing: {', '.join(missing)}")
    validation = json.loads(
        (root / f"validation/{workload_type}_prompt_validation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / f"validation/{workload_type}_prompt_manifest.json").read_text(encoding="utf-8")
    )
    prompt_sha256 = sha256_file(prompt_path)
    expected_sha256 = str(manifest.get("prompts_sha256", "")).removeprefix("sha256:")
    if not validation.get("passed") or validation.get("checks", {}).get("prompt_count") != 200:
        raise Task1QasperAdapterError("task-one directory validation is not passed for 200 prompts")
    if not expected_sha256 or prompt_sha256 != expected_sha256:
        raise Task1QasperAdapterError("task-one prompt SHA-256 does not match its validation manifest")

    # The established parser operates on the published package layout. Build a
    # temporary archive from validated input files only; never mutate the source.
    with tempfile.TemporaryDirectory() as raw_tmp:
        archive_path = Path(raw_tmp) / "task1-input.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in required[:-1]:
                archive.write(path, path.relative_to(root).as_posix())
        workload = load_task1_qasper_workload(archive_path, workload_type)
    audit = dict(workload.audit)
    audit.update(
        {
            "input_kind": "directory_package",
            "source_directory": str(root),
            "prompt_sha256": prompt_sha256,
            "validation_manifest_sha256": sha256_file(
                root / f"validation/{workload_type}_prompt_manifest.json"
            ),
            "source_zip": "",
            "source_zip_sha256": "",
            "immutability": "validated Task 1 directory input is read only; no request content or order is rewritten",
        }
    )
    return Task1QasperWorkload(workload.workload_type, workload.rows, audit)


def load_task1_qasper_workload(zip_path: str | Path, workload_type: str) -> Task1QasperWorkload:
    if workload_type not in {"random", "grouped"}:
        raise Task1QasperAdapterError("workload_type must be random or grouped")
    archive_path = Path(zip_path)
    if not archive_path.exists():
        raise Task1QasperAdapterError(f"task-one ZIP not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        prompts = _read_jsonl(archive, _PROMPT_PATH.format(workload=workload_type))
        metadata = {str(item.get("sample_id") or ""): item for item in _read_jsonl(archive, _METADATA_PATH)}
        source_manifest = _read_json(archive, "metadata/qasper_manifest.json")
        source_hashes = _read_json(archive, "metadata/qasper_sha256.json")
        validation = _read_json(archive, f"validation/{workload_type}_prompt_validation.json")
    if len(prompts) != 200:
        raise Task1QasperAdapterError(f"expected 200 {workload_type} prompts, found {len(prompts)}")
    rows: list[RuntimeWorkloadRow] = []
    for line_number, prompt in enumerate(prompts, start=1):
        missing = [name for name in _REQUIRED if prompt.get(name) in (None, "")]
        if missing:
            raise Task1QasperAdapterError(f"{workload_type}:{line_number} missing {', '.join(missing)}")
        sample_id = str(prompt["sample_id"])
        detail = metadata.get(sample_id)
        if detail is None:
            raise Task1QasperAdapterError(f"{workload_type}:{line_number} missing metadata for {sample_id}")
        group_size = _int(detail.get("reuse_group_size"), default=1)
        if group_size < 1:
            raise Task1QasperAdapterError(f"{workload_type}:{line_number} invalid reuse_group_size")
        ratio = (group_size - 1) / group_size
        bucket = "none" if group_size == 1 else ("medium" if group_size == 2 else "high")
        task_metadata = {
            "sample_id": sample_id,
            "dataset": prompt.get("dataset", "longbench"),
            "task": prompt.get("task", "qasper"),
            "workload_type": workload_type,
            "reuse_group": prompt["reuse_group"],
            "reuse_group_size": group_size,
            "shared_context": bool(prompt.get("shared_context", False)),
            "expected_reuse": detail.get("expected_reuse"),
            "estimated_kv_tokens": detail.get("estimated_kv_tokens"),
            "estimated_reusable_tokens": detail.get("estimated_reusable_tokens"),
            "ground_truth": prompt.get("ground_truth", ""),
            "answer": prompt.get("answer", ""),
            "messages": prompt.get("messages") if isinstance(prompt.get("messages"), list) else None,
            "task1_audit": prompt.get("audit") if isinstance(prompt.get("audit"), dict) else {},
            "question_hash": prompt.get("question_hash", ""),
            "answer_hash": prompt.get("answer_hash", ""),
            "ground_truth_hash": prompt.get("ground_truth_hash", ""),
            "context_hash": prompt.get("context_hash", ""),
            "metadata_ref": prompt.get("metadata_ref", ""),
            "reuse_ratio_definition": "(reuse_group_size - 1) / reuse_group_size",
        }
        rows.append(RuntimeWorkloadRow(
            request_id=str(prompt["request_id"]), prompt=str(prompt["prompt"]),
            prefix_id=str(prompt["reuse_group"]), prefix_hash=str(prompt["reuse_group"]),
            arrival_index=_int(prompt.get("order"), default=line_number - 1), reuse_ratio=ratio,
            reuse_bucket=bucket, context_length=_int(detail.get("estimated_context_tokens"), default=0),
            expected_output_tokens=_int(prompt.get("max_tokens"), default=128), batch_size=1,
            case=f"qasper_{workload_type}", metadata={key: value for key, value in task_metadata.items() if value is not None},
        ))
    rows.sort(key=lambda item: item.arrival_index)
    if len({item.request_id for item in rows}) != len(rows) or len({item.arrival_index for item in rows}) != len(rows):
        raise Task1QasperAdapterError(f"{workload_type} has duplicate request_id or order")
    group_sizes: dict[int, int] = {}
    for row in rows:
        size = _int(row.metadata.get("reuse_group_size"), default=1)
        group_sizes[size] = group_sizes.get(size, 0) + 1
    kv_tokens = sum(_int(item.metadata.get("estimated_kv_tokens"), default=0) for item in rows)
    reusable_tokens = sum(_int(item.metadata.get("estimated_reusable_tokens"), default=0) for item in rows)
    shared_groups = {row.prefix_id for row in rows if _int(row.metadata.get("reuse_group_size"), default=1) > 1}
    return Task1QasperWorkload(workload_type, tuple(rows), {
        "schema": TASK1_SCHEMA, "source_zip": str(archive_path), "source_zip_sha256": sha256_file(archive_path),
        "source_manifest": source_manifest, "source_hashes": source_hashes, "validation": validation,
        "prompt_entry": _PROMPT_PATH.format(workload=workload_type), "metadata_entry": _METADATA_PATH,
        "request_count": len(rows), "reuse_group_count": len({item.prefix_id for item in rows}),
        "shared_request_count": sum(1 for item in rows if item.metadata.get("shared_context")),
        "shared_reuse_group_count": len(shared_groups),
        "reuse_group_size_request_histogram": {str(key): value for key, value in sorted(group_sizes.items())},
        "estimated_kv_tokens": kv_tokens, "estimated_reusable_tokens": reusable_tokens,
        "estimated_reusable_token_ratio": reusable_tokens / kv_tokens if kv_tokens else 0.0,
        "immutability": "adapter copies existing task-one requests; it does not add, remove, repeat, or reorder requests",
    })


def write_task1_qasper_artifacts(workload: Task1QasperWorkload, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"qasper_{workload.workload_type}_canonical_workload.jsonl"
    audit_path = output / f"qasper_{workload.workload_type}_adapter_audit.json"
    distance_path = output / f"qasper_{workload.workload_type}_reuse_distance_report.csv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in workload.rows:
            handle.write(json.dumps(row.to_record(), ensure_ascii=False) + "\n")
    audit = {**workload.audit, "canonical_workload": str(manifest_path), "canonical_sha256": sha256_file(manifest_path)}
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reuse_distance(distance_path, workload.rows)
    return {"workload": manifest_path, "audit": audit_path, "reuse_distance": distance_path}


def write_reuse_distance(path: str | Path, rows: Iterable[RuntimeWorkloadRow]) -> None:
    previous: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.arrival_index):
        prior = previous.get(row.prefix_id)
        metadata = row.metadata
        records.append({
            "request_id": row.request_id, "prefix_id": row.prefix_id, "arrival_index": row.arrival_index,
            "first_access": prior is None, "previous_same_prefix_arrival_index": "" if prior is None else prior,
            "reuse_distance": "" if prior is None else row.arrival_index - prior,
            "reuse_group_size": metadata.get("reuse_group_size", ""),
            "estimated_reusable_tokens": metadata.get("estimated_reusable_tokens", ""),
        })
        previous[row.prefix_id] = row.arrival_index
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["request_id"])
        writer.writeheader()
        writer.writerows(records)


def _read_jsonl(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    try:
        with archive.open(name) as handle:
            return [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line.strip()]
    except KeyError as exc:
        raise Task1QasperAdapterError(f"missing archive entry: {name}") from exc


def _read_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        with archive.open(name) as handle:
            value = json.loads(handle.read().decode("utf-8"))
            return value if isinstance(value, dict) else {}
    except KeyError as exc:
        raise Task1QasperAdapterError(f"missing archive entry: {name}") from exc


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
