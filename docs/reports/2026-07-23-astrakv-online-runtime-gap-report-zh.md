# AstraKV 真实在线运行时目标差距分析报告

> 日期：2026-07-23
>
> 用途：本报告用于汇报 AstraKV 当前代码实现与“真实在线 KV 内存管理最终目标”之间的对应关系、已完成能力、未完成能力、实施优先级与最终验收目标。

## 1. 结论摘要

当前仓库已经不再是“只有离线分析和日志观察”的状态，而是已经具备了较完整的在线控制面骨架，包括：

- 版本锁定的后端 capability preflight
- 结构化 Hook 契约
- 请求感知的对象绑定表与 `binding_generation` 防陈旧机制
- 运行时命令准入门、预算门和熔断保护
- LMCache 0.4.7 的运行时 patch、动作服务和控制宿主
- 在线控制器、在线画像存储以及受控测试闭环

但是，当前代码仍然不能证明“已经在真实 DGX 生产运行中稳定在线控制 vLLM/LMCache 的 KV tensor”。更准确地说，当前状态应定义为：

- 已完成：在线控制面的核心工程骨架
- 已完成：受控 `drop` 执行路径与大量回归测试
- 未完成：面向真实 DGX 生产环境的完整在线闭环证据
- 未完成：`prefetch/load/offload/evict` 等目标动作的真实后端开放
- 未完成：baseline 与 AstraKV-enabled 的真实对照实验与收益证明

因此，当前最合理、最准确的汇报口径是：

> AstraKV 已经完成“真实在线控制面”的大部分基础设施建设，并在受控环境中验证了关键控制链路；但尚不能宣称已经完成对真实 vLLM/LMCache KV 内存的全面在线管理，当前仍处于“受控单动作验证 + 向真实在线闭环推进”的阶段。

## 2. 本次分析依据

本报告综合了以下两类依据：

### 2.1 目标文档

- `E:/os59/project3136859-384917/docs/guides/astrakv_online_runtime_execution_target_cn.md`

该文档定义了“真实在线 KV 内存管理”的目标边界、阶段路线和最终产物要求。

### 2.2 当前仓库实际代码

重点核查了以下模块：

- `astrakv/runtime/adapters.py`
- `astrakv/runtime/eviction.py`
- `astrakv/runtime/backend_hook.py`
- `astrakv/runtime/backend_capabilities.py`
- `astrakv/runtime/backend_binding_registry.py`
- `astrakv/runtime/backend_bridge.py`
- `astrakv/runtime/runtime_execution_gate.py`
- `astrakv/runtime/runtime_control_host.py`
- `astrakv/runtime/lmcache047_runtime_patch.py`
- `astrakv/runtime/lmcache047_action_service.py`
- `astrakv/runtime/online_controller.py`
- `astrakv/runtime/online_profile.py`
- `astrakv/benchmarks/experiment_manifest.py`
- `astrakv/benchmarks/runtime_artifacts.py`
- `scripts/benchmark/dgx_metrics_collector.py`
- `scripts/benchmark/verify_structured_eviction_hook.py`
- `scripts/reporting/normalize_runtime_eviction_events.py`
- `astrakv/runtime/eviction_agreement.py`

### 2.3 受控验证结果

本次额外运行了以下单元测试集合：

```text
python -m unittest \
  tests.test_backend_capabilities \
  tests.test_backend_hook \
  tests.test_backend_bridge \
  tests.test_backend_binding_registry \
  tests.test_runtime_control_host \
  tests.test_online_controller \
  tests.test_lmcache047_action_service \
  tests.test_lmcache047_runtime_patch
```

结果：

- 运行测试数：59
- 结果：通过
- 跳过：1

这说明当前“在线控制面骨架”不是停留在文档或接口层，而是在仓库内的受控场景中已经具备可验证行为。

## 3. 目标系统的正确定位

目标系统不是“用 AstraKV 替换 vLLM/LMCache”，而是让 AstraKV 成为 vLLM/LMCache 的 KV 内存控制面：

```text
vLLM / LMCache
  持有真实 KV block
  执行模型推理
  执行真实内存动作

AstraKV
  识别复用机会
  维护在线画像
  生成动作建议
  验证动作效果

Backend Hook / Bridge
  提供稳定对象身份
  接收受控命令
  调用真实后端 API
  返回结构化执行回执
```

因此，“是否已经完成真实在线能力”的关键判断标准不是：

- 是否有离线策略 CSV
- 是否有 HTTP warmup
- 是否能解析日志
- 是否有 mmap VM PoC

而是以下四件事是否同时成立：

1. 后端拥有真实 KV 对象的进程能够导出稳定对象绑定。
2. AstraKV 发出的动作命令经过版本锁定和安全门校验。
3. 真实后端执行了该动作。
4. 后端返回了带对象身份、动作、状态、时间的结构化回执。

## 4. 当前实现全景结论

### 4.1 目标文档与当前代码之间存在时间差

目标文档把很多能力描述为“尚未实现”或“不能证明”，但当前仓库实际已经比该文档推进得更远，尤其是在以下方面：

- 已经有 `backend_hook` 结构化契约
- 已经有 `backend_binding_registry`
- 已经有 `backend_bridge`
- 已经有 `runtime_execution_gate`
- 已经有 `runtime_control_host`
- 已经有 `lmcache047_runtime_patch`
- 已经有 `lmcache047_action_service`
- 已经有 `online_controller` 和 `online_profile`

因此，后续汇报时需要明确：

> 目标文档仍然有参考价值，但它低估了当前代码在“控制面工程化”方面的进展。

### 4.2 当前系统的最准确定义

当前系统最适合被定义为：

- 离线策略层：成熟
- 在线控制面基础设施：已基本完成
- 真实生产动作面：仅实现了受控、版本锁定、单动作优先的第一阶段
- 真实在线收益证明：尚未完成

## 5. 已实现能力清单

本节按“目标能力”而不是按文件列出当前已完成内容。

### 5.1 运行时适配契约与能力边界已完成

当前代码已经明确区分：

- 只读观察适配器
- 可执行 PoC 适配器
- 受控后端桥接路径

核心证据：

- `RuntimeAdapter` 为统一运行时接口
- `RuntimeCapabilities` 默认是 deny-by-default
- `VllmLmCacheArtifactAdapter` 明确声明自己是 observational
- `MMapEvictionAdapter` 明确声明自己是 `vm_poc_execution`

这意味着当前仓库在架构边界上是清楚的，没有把“日志观察”“PoC 执行”“真实后端执行”混为一谈。

### 5.2 结构化 Hook 契约已完成

`backend_hook.py` 已定义四类关键结构：

- `BackendObjectBinding`
- `BackendHookEvent`
- `BackendActionCommand`
- `BackendActionReceipt`

这些结构已经包含：

- `run_id`
- `request_id`
- `object_key`
- `object_level`
- `backend_object_id`
- `binding_id`
- `binding_generation`
- `command_id`
- `receipt_id`
- `timestamp_ns`

这说明“结构化身份”和“结构化命令/回执契约”已经在代码层完成，不再是概念设计。

### 5.3 对象绑定表与防陈旧机制已完成

`backend_binding_registry.py` 已实现：

- 逻辑对象到物理后端对象的关联
- 物理对象生命周期管理
- `binding_generation` 递增
- 活跃请求、pin、pending I/O 状态跟踪
- 动作 reservation
- 对象是否允许桥接执行的判断

这是整个在线系统中最关键的基础设施之一。它解决的不是“能不能发命令”，而是“能不能确保命令不会命中一个已重用、已过期、仍活跃或状态不安全的对象”。

### 5.4 版本锁定与 capability preflight 已完成

`backend_capabilities.py` 已具备：

- 精确的版本锁定
- endpoint 身份校验
- connector 身份校验
- action 白名单
- object level 白名单
- binding generation 观察能力要求

当前允许的目标范围非常保守：

- 支持的 vLLM 版本：`0.23.0`
- 支持的 LMCache 版本：`0.4.7`
- 支持的 connector：`lmcache-vllm-v1`
- 支持动作：`drop`
- 支持对象层级：`prefix`

这是一种典型的“先缩小可证明范围，再逐步扩展动作集”的工程策略。

### 5.5 运行时执行门与熔断保护已完成

`runtime_execution_gate.py` 已经做了真正的运行时准入控制，检查内容包括：

- `run_id` 是否匹配
- capability preflight 是否通过
- circuit breaker 是否允许调度
- `deadline_ns` 是否有效
- binding 是否仍然是当前 binding
- `binding_generation` 是否匹配
- 对象是否有 pending I/O 或 pending operation
- 对象是否仍被活跃请求引用
- 对象是否被 pin
- 命令是否重复
- 单命令 bytes 预算
- 时间窗速率预算
- 最大并发预算

这说明“执行安全门”已经不是文档要求中的待设计概念，而是落地代码。

### 5.6 LMCache 0.4.7 patch 与动作服务已完成骨架

`lmcache047_runtime_patch.py` 和 `lmcache047_action_service.py` 已经实现了：

- 对 LMCache factory / manager / connector 的 patch 安装
- request context 关联
- storage manager `batched_put/get/remove` 事件捕获
- LocalDisk completion callback 识别
- runtime proof 签发与校验
- 运行时动作服务
- 受控 `drop` 执行
- 命令 ledger 与 receipt ledger

这是当前最接近“真实在线执行”的代码部分。

### 5.7 在线控制宿主已完成

`runtime_control_host.py` 已经具备：

- request context 接收 HTTP 入口
- action HTTP 入口
- runtime-proof 入口
- binding registry 挂载
- action service 挂载
- execution gate 挂载
- online bridge 挂载
- online controller 挂载
- 后台 policy worker
- `bindings.jsonl / events.jsonl / commands.jsonl / receipts.jsonl` 落盘

也就是说，控制宿主已经具备“一个 run 内的运行时在线控制装配能力”。

### 5.8 在线控制器与在线画像已完成第一版

`online_controller.py` 与 `online_profile.py` 已经实现：

- 消费 verified backend hook event
- 在线对象状态 materialization
- checkpoint / replay
- 决策与执行分离
- 基于 `release` 事件触发简化版策略
- 受控 dispatch 记录 command/receipt/event

虽然它离最终的“完整在线滚动调度器”还有距离，但第一版闭环已经存在。

### 5.9 结构化事件验证与运行时对账工具已完成

已完成的外围验证工具包括：

- `verify_structured_eviction_hook.py`
- `normalize_runtime_eviction_events.py`
- `eviction_agreement.py`
- `experiment_manifest.py`
- `runtime_artifacts.py`

它们已经可以支撑：

- 结构化事件格式验证
- 已验证事件归一化
- 运行时事件与离线决策的一致性比较
- run 级 manifest 和 artifact hash 追踪

这意味着“证据链框架”是存在的，当前缺的是“真实 DGX 运行得到的最终证据”。

### 5.10 请求级资源采样能力比目标文档更前

目标文档中对 `DgxMetricsCollector` 的描述已经落后于当前代码。

当前 `dgx_metrics_collector.py` 已经支持：

- `request_id`
- request 生命周期窗口
- shared batch attribution
- 独占请求与共享批次区分

因此，资源归因这部分并不是完全空白，而是已经进入“可继续工程化收口”的状态。

## 6. 当前尚未完成或尚不能宣称完成的能力

### 6.1 传统 `VllmLmCacheArtifactAdapter` 仍然是观察模式

`eviction.py` 中的 `VllmLmCacheArtifactAdapter.apply_hint()` 仍然明确返回：

- `unsupported`

这意味着：

- 旧的 “离线策略 -> 直接调用 vLLM/LMCache 适配器” 路线没有打通
- 当前真实执行不是靠这个适配器完成的
- 对外汇报时不能把它说成“已经支持真实 runtime action”

### 6.2 真实开放动作仍然非常窄

当前可被 preflight 接受的动作和层级非常有限：

- 动作：`drop`
- 层级：`prefix`

因此下列能力目前仍未真正开放为生产级真实动作：

- `prefetch`
- `load`
- `offload`
- `evict`
- `recompute`

即使代码中有这些概念，也不能把它们都算作“当前已完成的真实在线动作集”。

### 6.3 默认仍然是 fail-closed，而不是“生产已开放”

LMCache 0.4.7 的真实动作路径当前仍然是非常保守的。

关键事实：

- `action_registration_enabled` 默认不是自由开启
- 缺少经验证的 terminal callback 时，系统应保持 observational-only
- 多处测试明确验证了“默认 endpoint 为 observational_only”

这意味着当前状态是：

- 已具备“安全放行动作”的工程结构
- 尚不能直接宣称“真实生产路径已经默认开启”

### 6.4 在线策略引擎仍是第一版，不是最终版

当前 online controller 的策略能力仍然很窄，主要表现为：

- 基于 `release` 事件生成简化 `drop` 建议
- 没有完整在线化 `ChunkScorer + Load-vs-Recompute + UnifiedObjectScheduler`
- 还没有形成完整动作空间的滚动优化

目标文档要求的是完整动作集合：

- `keep`
- `prefetch/load`
- `offload`
- `evict/drop`
- `recompute`
- `defer`

当前距离这个目标还有明显差距。

### 6.5 在线画像指标还不够完整

当前 `OnlineProfileStore` 更像是一个事件物化器和基础对象统计容器，而不是最终目标中的完整策略画像。

尚待补齐的典型指标包括：

- `load_latency_ema`
- `prefetch_success_count`
- `prefetch_waste_count`
- `eviction_reaccess_count`
- 动作成功率统计
- 动作后延迟变化统计
- 压力变化反馈

### 6.6 最终 artifact 合同尚未收口

当前仓库内已有的输出文件更偏工程内部命名，例如：

- `bindings.jsonl`
- `events.jsonl`
- `commands.jsonl`
- `receipts.jsonl`
- `online_bindings.jsonl`
- `online_events.jsonl`
- `online_commands.jsonl`
- `online_receipts.jsonl`

但目标文档要求面向汇报和验收的最终产物命名为：

- `backend_capabilities.json`
- `backend_binding_events.jsonl`
- `runtime_events_raw.jsonl`
- `astrakv_runtime_commands.jsonl`
- `runtime_command_receipts.jsonl`
- `runtime_structured_events.jsonl`
- `online_profile_checkpoint.json`

这说明最终交付的 artifact contract 还需要统一。

### 6.7 真实 DGX 证据链尚未补齐

当前工作区内没有足够的成对真实实验产物去支撑以下结论：

- TTFT 改善
- TPOT 改善
- 吞吐提升
- QASPER 质量不退化
- `precision / recall / F1` 达标
- 动作成功率稳定
- 资源压力变化受控

换句话说：

> 验证工具已经实现，但真实“收益证明”还没有完成。

## 7. 当前最适合对外汇报的实现状态

为避免过度承诺，建议使用以下汇报分层。

### 7.1 可以明确宣称已经完成的

- AstraKV 已完成在线控制面的核心工程骨架设计与实现。
- 系统已具备版本锁定、结构化 Hook 契约、对象绑定、命令回执、安全门、熔断和在线宿主能力。
- 系统已具备 LMCache 0.4.7 方向的运行时 patch 和动作服务原型。
- 关键在线控制链路已通过 59 项受控单元测试验证。

### 7.2 可以宣称“已初步打通”的

- 受控 `drop` 动作路径
- 在线事件 ingest -> 在线画像 -> 策略触发 -> command/receipt 落盘
- 结构化事件验证和运行时一致性对账框架

### 7.3 不能宣称已经完成的

- 已全面在线管理真实 vLLM/LMCache KV tensor
- 已稳定支持真实 `prefetch/load/offload/evict`
- 已在真实 DGX 上证明收益
- 已完成普适生产化闭环

## 8. 详细差距矩阵

| 能力域 | 当前状态 | 结论 |
| --- | --- | --- |
| 离线工作负载与策略链 | 已完成 | 可作为在线策略输入基础 |
| 结构化 Hook 契约 | 已完成 | 当前代码已具备 |
| 版本锁定 capability preflight | 已完成 | 当前代码已具备 |
| 对象绑定表与 `binding_generation` | 已完成 | 当前代码已具备 |
| 运行时准入门与预算/熔断 | 已完成 | 当前代码已具备 |
| LMCache 0.4.7 patch | 已完成第一版 | 已有真实接入骨架 |
| 运行时动作服务 | 已完成第一版 | 已有受控动作服务 |
| 在线控制宿主 | 已完成第一版 | 已能装配在线控制链路 |
| 在线画像 | 已完成第一版 | 还需扩展指标与反馈 |
| 在线策略调度器 | 部分完成 | 当前主要是简化 `drop` 触发 |
| 真实 `drop` 动作 | 受控可行 | 但默认仍应 fail-closed |
| 真实 `prefetch/load/offload/evict` | 未完成 | 还未进入真实动作白名单 |
| 最终 artifact contract | 部分完成 | 需要统一命名与交付口径 |
| baseline / AstraKV-enabled 对照实验 | 未完成 | 没有最终收益证据 |
| 汇报级最终结论 | 尚不可下最终结论 | 目前只能汇报阶段性完成 |

## 9. 建议的实施路线

本节给出建议的落地顺序，目标是最快形成一条“可证明、可汇报、风险可控”的真实在线路径。

### 阶段 0：先校正文档与口径

目标：

- 把“目标文档中的旧状态描述”更新为“当前代码真实状态”
- 明确三条路径边界

需要明确区分的三条路径：

- `artifact-observational`
- `lmcache047-controlled-drop`
- `vm_poc_execution`

交付物：

- 一份更新后的目标说明文档
- 一份统一术语表

价值：

- 避免汇报时混淆 PoC、日志观察和真实动作执行

### 阶段 1：锁定第一条可证明真实路径

建议锁定范围：

- vLLM `0.23.0`
- LMCache `0.4.7`
- connector `lmcache-vllm-v1`
- object level `prefix`
- action `drop`

原因：

- 这条路径与当前代码最一致
- capability preflight 已经围绕它设计
- 风险最可控

交付物：

- 真实 DGX 环境下可运行的 `drop-only` 路径
- 可验证的结构化回执

### 阶段 2：统一最终 artifact contract

目标：

- 让当前内部状态文件与目标文档中的最终交付清单对齐

建议输出的最终结果目录：

```text
results/<run_id>/
  experiment_manifest.json
  backend_capabilities.json
  workload.jsonl
  request_results.jsonl
  benchmark_results.csv
  samples/<case>_samples.csv
  backend_binding_events.jsonl
  runtime_events_raw.jsonl
  astrakv_runtime_commands.jsonl
  runtime_command_receipts.jsonl
  runtime_structured_events.jsonl
  online_profile_checkpoint.json
  runtime_agreement/
  quality/
  final_report.md
```

### 阶段 3：补齐真实 DGX Hook 证据

目标：

- 在真实 DGX 环境确认 Hook 安装有效
- 验证 request context、binding、release、callback completion 全链路

关键检查：

- 是否能得到同 run 的结构化 binding/event
- 是否能证明 `binding_generation` 真正参与动作校验
- 是否能证明 `action_registration_enabled` 何时被安全放开

### 阶段 4：扩展在线策略引擎

第一步建议：

- 先做 `keep / drop / defer`

第二步建议：

- 再接入 `load vs recompute`

第三步建议：

- 最后开放 `prefetch / offload / load`

原因：

- 这样能始终保持“动作开放范围 <= 可验证能力范围”

### 阶段 5：扩展在线画像与反馈闭环

需要补齐：

- `load_latency_ema`
- `prefetch_hit_rate`
- `prefetch_waste_rate`
- `reaccess_after_drop`
- 动作失败率
- 预算消耗
- 熔断状态

目标：

- 不仅能发命令
- 还能根据真实执行效果修正下一轮决策

### 阶段 6：跑 baseline / AstraKV-enabled 对照实验

实验必须满足：

- 同模型
- 同硬件
- 同 workload
- 同 run 条件
- 同质量评估流程

最终要回答的问题：

- correctness 是否成立
- `precision / recall / F1` 是否达标
- TTFT / TPOT / 吞吐是否改善
- 质量是否退化
- 资源压力是否受控

## 10. 最终实现目标清单

本节给出“真正达到目标状态时应满足什么”的最终清单，可直接作为项目收尾验收表。

### 10.1 环境与版本

- 固定 vLLM、LMCache、connector、CUDA、模型和桥接器版本
- 每个 run 都有可信 manifest
- 每个 run 都有可信 capability preflight

### 10.2 对象身份与绑定

- 后端能稳定导出真实对象 binding
- binding 带 `binding_generation`
- 重用、释放、重新分配时 generation 正确递增
- 任何动作都不能命中 stale binding

### 10.3 运行时动作控制

- 所有命令带 `command_id`
- 所有命令带 `deadline_ns`
- 所有命令都经过 run、version、binding、busy 状态、预算和熔断检查
- 所有动作都有结构化 receipt

### 10.4 动作开放范围

第一阶段必须完成：

- `drop`

第二阶段逐步扩展：

- `offload`
- `load`
- `prefetch`
- `evict`

任何动作开放都必须满足：

- 有真实后端 API
- 有结构化回执
- 有安全门
- 有实验验证

### 10.5 在线控制闭环

- 常驻在线画像存在
- 异步策略线程存在
- 策略不阻塞 vLLM 推理线程
- 动作结果能回写画像
- 动作失败和拒绝会影响后续调度优先级

### 10.6 证据链与报告

- 结构化 Hook 事件可验证
- 运行时事件可归一化
- 离线决策与真实动作可对账
- baseline / variant 对照报告完整
- 所有结论都有同 run 原始产物支撑

### 10.7 最终“可宣称完成”的判断标准

只有在以下条件同时成立时，才可以对外宣称“AstraKV 已完成真实在线 KV 内存管理能力”：

1. 已有真实后端对象 binding 证据。
2. 已有真实后端结构化 action receipt 证据。
3. 已完成至少一个真实动作的安全在线执行验证。
4. 已完成 baseline / AstraKV-enabled 对照实验。
5. 已证明质量不退化且收益可重复。

## 11. 建议的汇报话术

如果你一会儿要做口头汇报，建议用下面这段作为主线：

> 我们目前不是停留在离线策略或日志观察阶段，而是已经完成了 AstraKV 在线控制面的核心工程骨架，包括版本锁定、结构化 Hook、对象绑定、运行时安全门、动作服务、在线控制宿主和第一版在线控制器。当前最接近真实在线执行的路径，是面向 LMCache 0.4.7 的受控 `drop` 动作链路，并且这条链路已经通过受控单元测试闭环验证。下一步的核心工作不是再补基础框架，而是把这套控制面在真实 DGX 环境中跑通、统一最终 artifact 合同、逐步开放更多真实动作，并通过 baseline 与 AstraKV-enabled 的配对实验给出收益证明。

如果需要一句更短的总结：

> 当前我们已经完成“在线控制面”，正在补齐“真实动作面”和“真实收益证据”。

## 12. 本报告的最终判断

当前最准确的项目判断如下：

- 可以判断：AstraKV 已经完成真实在线 KV 控制面的核心工程建设。
- 可以判断：项目已经进入“真实后端动作验证”阶段，而不是概念验证阶段。
- 不能判断：AstraKV 已经完成真实生产级在线 KV 内存管理。
- 不能判断：AstraKV 已经在真实 DGX 上证明完整收益。

因此，当前最合适的项目状态标签是：

```text
阶段状态：在线控制面已完成，真实动作面部分完成，真实收益证明未完成
```
