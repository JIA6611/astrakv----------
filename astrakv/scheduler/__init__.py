"""Scheduler adapter boundary placeholders.

No scheduler algorithm is implemented in this package. It only defines a small
hint object that future runtime adapters can exchange with AstraKV-W managers.
"""

from .hints import SchedulerHint
from .decision import (
    LoadRecomputeAction,
    LoadRecomputeConfig,
    LoadRecomputeDecision,
    LoadRecomputePlanner,
    PartialPlanStats,
)
from .object_scheduler import (
    ObjectScheduleAction,
    ObjectScheduleCandidate,
    ObjectScheduleDecision,
    ObjectSchedulerConfig,
    UnifiedObjectScheduler,
)

__all__ = [
    "LoadRecomputeAction",
    "LoadRecomputeConfig",
    "LoadRecomputeDecision",
    "LoadRecomputePlanner",
    "ObjectScheduleAction",
    "ObjectScheduleCandidate",
    "ObjectScheduleDecision",
    "ObjectSchedulerConfig",
    "PartialPlanStats",
    "SchedulerHint",
    "UnifiedObjectScheduler",
]
