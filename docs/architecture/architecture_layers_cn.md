# AstraKV-W 代码逻辑层次整理

日期：2026-06-09

这份文档描述当前仓库的推荐代码布局。目标不是大规模改目录，而是把“对象是什么、证据从哪里来、策略如何决策、动作由谁执行、实验怎么跑”分清楚。

## 1. 总体分层

当前代码建议按四层理解：

| 层次 | 目录/文件 | 职责 | 不负责 |
| --- | --- | --- | --- |
| 对象层 | `kv_cache/`, `moe/`, `offload/tier_placement.py` | 定义 KV chunk、token span、MoE expert、tier placement 等对象和状态。 | 不移动 tensor，不调用 vLLM/LMCache。 |
| 证据层 | `runtime/trace_schema.py`, `runtime/cache_events.py`, `runtime/profile_db.py`, `runtime/memory_pressure.py` | 从日志、benchmark、trace、sample 中提取事件和 profile，形成可复用证据。 | 不直接影响正在运行的 backend。 |
| 策略层 | `prefetch/scorer.py`, `scheduler/decision.py`, `scheduler/object_scheduler.py` | 根据 profile、pressure、budget 产生 prefetch/keep/offload/drop/load/recompute 等决策。 | 不保证动作已经被 backend 执行。 |
| 执行/实验层 | `runtime/endpoint_prefetch.py`, `runtime/vm_backend.py`, `experiments/`, `scripts/` | 提供 endpoint warmup adapter、OS VM PoC backend、实验 runner 和 CLI glue。 | 不把实验结果伪装成真实 vLLM 内部实现。 |

这四层可以对应成一条数据流：

```text
benchmark/log/sample
  -> trace/cache events
  -> ProfileDB / memory pressure
  -> scorer / scheduler decisions
  -> endpoint warmup or VM PoC actions
  -> new artifacts / reports
```

## 2. 目录职责

### `kv_cache/`

KV 对象层。这里描述 KV cache 的逻辑形态：

- `metadata.py`：`KVChunkMeta`、`MemoryTier`。
- `block_table.py`：request 到 chunk/block 的关系。
- `partial_load.py`：按 layer/token span 规划 partial KV load。

推荐边界：这里可以定义“要加载哪些 KV”，但不要实现“怎么从 vLLM/LMCache 取 tensor”。

### `runtime/`

运行时证据和 adapter-facing 能力层。这里既有证据解析，也有不侵入 backend 的执行入口：

- `trace_schema.py`：统一 trace schema。
- `cache_events.py`：只读解析 vLLM/LMCache cache 相关日志。
- `profile_db.py`：从 trace 聚合 workload/chunk profile。
- `memory_pressure.py`：从 artifact 判断 pressure，并输出 passive hints。
- `endpoint_prefetch.py`：通过 OpenAI-compatible endpoint 发 warmup 请求。
- `vm_backend.py`：file-backed `mmap` 虚拟内存 backend helper，用于 OS VM PoC 和后续 KV chunk/page 映射。
- `object_manager.py`：组合 KV metadata、placement、prefetch lifecycle state。

推荐边界：`runtime/` 可以放“真实系统交互的 adapter 壳”和“OS VM 机制 helper”，但不要把策略算法塞进这里。

### `prefetch/`

预取策略和预取生命周期层：

- `scorer.py`：根据 profile 计算 chunk score 和推荐动作。
- `async_engine.py`：管理 prefetch request/result 状态，真实工作由 adapter callback 完成。
- `selective_kv.py`：runtime-independent selective KV prefetch MVP。

推荐边界：策略和 lifecycle 可以在这里，endpoint HTTP、LMCache log parsing、mmap 系统调用不要放这里。

### `scheduler/`

统一调度决策层：

- `decision.py`：load-vs-recompute advisory decision。
- `object_scheduler.py`：在 GPU budget 下合并 chunk score 和 load/recompute decision。
- `hints.py`：统一 scheduler hint 格式。

推荐边界：这里输出“应该做什么”，不直接“怎么执行”。

### `offload/`

放 tier placement 状态管理：

- `tier_placement.py`：记录 current/target tier、planned/resident/failed 等状态。

推荐边界：这里可以跟踪 placement intent，但真实迁移应交给未来 adapter。

### `experiments/`

实验 runner 层。当前 `experiments/vm_demo.py` 是兼容 wrapper，核心 VM 机制已经放到 `runtime/vm_backend.py`。

推荐边界：这里负责实验流程、access pattern、结果汇总，不放可复用 runtime backend 逻辑。

### `scripts/`

CLI glue 层。脚本应尽量薄：

- parse args；
- 调用 `runtime/`、`scheduler/`、`prefetch/`、`evaluation/` 等核心模块；
- 写 CSV/JSON/JSONL/Markdown artifact；
- 打印输出路径。

推荐边界：如果某段逻辑要被测试、复用或报告解释，应优先放回核心模块。

## 3. 虚拟内存主线

虚拟内存相关代码建议按下面的逻辑组织：

```text
runtime/vm_backend.py
  -> file-backed mmap / page-like access / future madvise+mincore

experiments/vm_demo.py
  -> VM demo compatibility wrapper and experiment-facing import path

scripts/run_vm_demo.py
  -> CLI, artifact writing, Markdown report

docs/current_core_code_adjustments_cn.md
  -> 当前缺口和下一步建议
```

后续如果继续增强虚拟内存，可以按这个顺序加：

1. 在 `runtime/vm_backend.py` 补 `madvise(WILLNEED/DONTNEED)`、`mincore`、cold/hot read helper。
2. 新增 `runtime/vm_page_table.py` 或扩展 `vm_backend.py`，建立 `KVChunkMeta -> mmap block/page` 映射。
3. `experiments/vm_demo.py` 只负责调用这些 helper 跑实验。
4. `scripts/run_vm_demo.py` 只负责 CLI 和 artifact。

## 4. 低风险整理原则

当前阶段不建议大规模重命名所有目录。更稳妥的整理原则是：

1. 保留现有包名，避免 import 大面积断裂。
2. 把可复用核心逻辑从 `experiments/` 和 `scripts/` 抽回 `runtime/`、`scheduler/`、`kv_cache/`。
3. 对旧 import 路径保留薄 wrapper，例如 `experiments/vm_demo.py`。
4. 每次移动后先跑目标测试，再动文档。
5. 文档里明确区分 planned/advisory 和 executed/action evidence。

## 5. 推荐阅读路径

理解系统时建议按这个顺序看：

1. `kv_cache/metadata.py`
2. `runtime/trace_schema.py`
3. `runtime/cache_events.py`
4. `runtime/profile_db.py`
5. `prefetch/scorer.py`
6. `scheduler/decision.py`
7. `scheduler/object_scheduler.py`
8. `runtime/endpoint_prefetch.py`
9. `runtime/vm_backend.py`
10. `scripts/run_vm_demo.py`

这样会先看到对象和证据，再看到策略和执行入口，比较符合系统实际层次。
