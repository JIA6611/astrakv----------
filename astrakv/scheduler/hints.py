"""Scheduler hint data objects.

These hints are passive metadata. They are not a scheduler implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SchedulerHint:
    request_id: str
    action: str
    reason: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
