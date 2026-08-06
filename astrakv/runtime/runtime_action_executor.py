"""Unified runtime-owner action executors for version-locked backend control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrakv.runtime.backend_capabilities import BLOCKED_ACTION_REASONS
from astrakv.runtime.backend_hook import BackendActionCommand, HookAction


@dataclass(slots=True)
class _BaseExecutor:
    endpoint: Any


class DropExecutor(_BaseExecutor):
    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        return self.endpoint._execute_drop_command(command)


class OffloadExecutor(_BaseExecutor):
    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        return self.endpoint._execute_offload_command(command)


class LoadExecutor(_BaseExecutor):
    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        return self.endpoint._execute_load_command(command)


class PrefetchExecutor(_BaseExecutor):
    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        return self.endpoint._execute_prefetch_command(command)


class EvictExecutor(_BaseExecutor):
    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        return self.endpoint._execute_evict_command(command)


class RuntimeActionExecutor:
    """Dispatch version-locked owner-only commands to concrete action executors."""

    def __init__(self, endpoint: Any) -> None:
        self.drop_executor = DropExecutor(endpoint)
        self.offload_executor = OffloadExecutor(endpoint)
        self.load_executor = LoadExecutor(endpoint)
        self.prefetch_executor = PrefetchExecutor(endpoint)
        self.evict_executor = EvictExecutor(endpoint)
        self._by_hook_action = {
            HookAction.DROP: self.drop_executor,
            HookAction.OFFLOAD: self.offload_executor,
            HookAction.PREFETCH: self.prefetch_executor,
            HookAction.CACHE_LOAD: self.load_executor,
            HookAction.EVICT: self.evict_executor,
        }

    def execute(self, command: BackendActionCommand) -> dict[str, Any]:
        executor = self._by_hook_action.get(command.action)
        if executor is None:
            return _blocked_response(command, command.action.value, "no_runtime_executor_for_action")
        return executor.execute(command)


def _blocked_response(command: BackendActionCommand, action_name: str, blocked_reason: str) -> dict[str, Any]:
    return {
        "action": action_name,
        "backend_object_id": command.backend_object_id,
        "status": "blocked",
        "blocked_reason": blocked_reason,
        "command_id": command.command_id,
    }
