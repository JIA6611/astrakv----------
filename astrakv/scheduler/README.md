# scheduler

Scheduler boundary placeholders.

## Files

- `hints.py`: passive `SchedulerHint` metadata object.
- `decision.py`: advisory load-vs-recompute planner that turns ProfileDB and
  optional partial KV load plans into passive scheduler hints.
- `object_scheduler.py`: unified object scheduler MVP that merges chunk scores,
  load-vs-recompute decisions, object size, reuse, deadline, and GPU budget into
  passive placement/prefetch/recompute hints.

## Scope

This package intentionally does not implement scheduling. It exists so future
runtime adapters and cache managers have a stable place to exchange passive
metadata such as `prefetch`, `wait`, or `evict` hints.

No request admission, queueing, preemption, or backend-owned tensor movement is
implemented here.

The load-vs-recompute planner estimates whether a chunk should be loaded,
recomputed, deferred, or dropped from profile statistics, IO estimates, compute
cost estimates, deadline, memory pressure, and optional P1-5 partial-load
records. The output remains passive metadata for future adapters.

The unified object scheduler arbitrates a GPU byte budget across KV objects.
Its decisions are advisory: `prefetch` and `keep` consume budget, while
`offload`, `recompute`, `defer`, and `drop` express what a future backend
adapter should do. It does not move memory by itself.
