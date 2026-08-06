"""Runtime-agnostic KV block table skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .metadata import KVChunkMeta


@dataclass(frozen=True, slots=True)
class KVBlockEntry:
    """Mapping between a logical chunk and runtime-owned block ids."""

    chunk_id: str
    request_id: str
    layer_id: int
    start_token: int
    end_token: int
    block_ids: tuple[int, ...]

    @classmethod
    def from_meta(cls, meta: KVChunkMeta) -> "KVBlockEntry":
        return cls(
            chunk_id=meta.chunk_id,
            request_id=meta.request_id,
            layer_id=meta.layer_id,
            start_token=meta.start_token,
            end_token=meta.end_token,
            block_ids=meta.block_ids,
        )

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token


class KVBlockTable:
    """Bookkeeping table for KV chunks and their block ids.

    This class does not own tensor memory. It stores metadata that adapters can
    translate to vLLM, SGLang, LMCache, or TensorRT-LLM structures later.
    """

    def __init__(self) -> None:
        self._entries: dict[str, KVBlockEntry] = {}
        self._by_request: dict[str, set[str]] = {}

    def add_chunk(self, meta: KVChunkMeta) -> KVBlockEntry:
        entry = KVBlockEntry.from_meta(meta)
        self._entries[entry.chunk_id] = entry
        self._by_request.setdefault(entry.request_id, set()).add(entry.chunk_id)
        return entry

    def remove_chunk(self, chunk_id: str) -> KVBlockEntry | None:
        entry = self._entries.pop(chunk_id, None)
        if entry is None:
            return None
        chunk_ids = self._by_request.get(entry.request_id)
        if chunk_ids is not None:
            chunk_ids.discard(chunk_id)
            if not chunk_ids:
                self._by_request.pop(entry.request_id, None)
        return entry

    def get_chunk(self, chunk_id: str) -> KVBlockEntry | None:
        return self._entries.get(chunk_id)

    def list_request_chunks(self, request_id: str) -> list[KVBlockEntry]:
        chunk_ids = self._by_request.get(request_id, set())
        entries = [self._entries[item] for item in chunk_ids if item in self._entries]
        return sorted(entries, key=lambda item: (item.layer_id, item.start_token, item.end_token))

    def iter_entries(self) -> Iterable[KVBlockEntry]:
        return iter(self._entries.values())

    def clear_request(self, request_id: str) -> list[KVBlockEntry]:
        removed: list[KVBlockEntry] = []
        for chunk_id in list(self._by_request.get(request_id, set())):
            entry = self.remove_chunk(chunk_id)
            if entry is not None:
                removed.append(entry)
        return removed

    def to_records(self) -> list[dict]:
        return [
            {
                "chunk_id": entry.chunk_id,
                "request_id": entry.request_id,
                "layer_id": entry.layer_id,
                "start_token": entry.start_token,
                "end_token": entry.end_token,
                "token_count": entry.token_count,
                "block_ids": list(entry.block_ids),
            }
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (item.request_id, item.layer_id, item.start_token),
            )
        ]
