# AstraKV 真实在线 KV 控制面对外任务计划

## 1. 项目目标

本计划面向对外沟通与阶段汇报，目标是把 AstraKV 从“具备离线分析和受控验证能力”推进到“具备真实在线控制、真实动作执行、真实收益验证能力”的可验收状态。

本轮实施范围锁定在单一版本主线：
- `vLLM 0.23.0`
- `LMCache 0.4.7`
- `lmcache-vllm-v1`

我们不再以演示型闭环为目标，而是以真实在线路径打通为目标，最终形成一条可以被复核、可以被对照实验验证、可以用于正式汇报的完整能力链路。

## 2. 推进原则

1. 单版本推进。先把锁定版本做实，再考虑横向兼容，不在当前主线上引入多版本复杂度。
2. 分阶段放开。动作能力按安全门逐步开放，不把尚未被真实证据证明的能力写成“已完成”。
3. 真实证据优先。所有关键结论以真实 DGX 环境中的结构化证据为准，不以日志猜测、接口返回成功或演示效果替代。
4. 验收口径统一。所有运行结果、证据文件、对照实验和最终报告使用统一命名和统一解释口径。
5. 结果导向交付。每一阶段都必须沉淀为可复核的产物，而不仅是代码改动。

## 3. 任务计划

### 3.1 任务一：打通真实在线控制基础

任务目标：
完成真实在线控制面的基础能力建设，形成稳定、可追溯、可验收的运行底座。

重点工作：
1. 统一对外口径和结果目录规范，明确哪些文件才是“真实在线证据”。
2. 固化 `RuntimeControlHost`、Hook、binding registry 等核心链路，确保真实环境下可以稳定记录对象绑定与运行事件。
3. 明确 observer-only 与 execution-enabled 两种模式的切换条件，避免控制面在条件不完备时误宣称具备执行能力。
4. 在真实 DGX 环境中完成基础采证，验证同一 `run_id` 下 command、receipt、event 可以一致回溯。
5. 先把 `drop` 打造成第一条真实可验收动作链路，形成后续扩展的基线。

阶段交付物：
- 统一命名的真实在线 artifact 目录
- `backend_capabilities.json`
- `backend_binding_events.jsonl`
- `runtime_events_raw.jsonl`
- `astrakv_runtime_commands.jsonl`
- `runtime_command_receipts.jsonl`
- `runtime_structured_events.jsonl`
- `online_profile_checkpoint.json`

阶段验收标志：
1. 真实在线 run 可以稳定输出统一命名的证据文件。
2. 对象绑定、运行事件、命令和回执之间存在完整追溯关系。
3. `drop` 动作具备真实 accepted、rejected、expired 等多路径样本，而不只是单一路径成功演示。

### 3.2 任务二：完成动作能力扩展与在线策略闭环

任务目标：
在已打通的真实控制底座上，逐步扩展完整动作能力，并将控制器从“被动触发”升级为“可根据真实回执持续调整”的在线策略器。

重点工作：
1. 在 `drop` 基线稳定后，按顺序扩展 `offload`、`load`、`prefetch`、`evict` 四类动作。
2. 按 `prefix -> cache_key -> block` 的顺序逐级开放对象层级，前提是后端 identity 可以被真实稳定导出。
3. 固化 command、receipt、structured event 三段式链路，确保每个动作都具备可验证的结构化执行证据。
4. 将 controller 从“收到 release 触发单一动作”升级为统一异步调度器，支持 `keep / prefetch / load / offload / evict / drop / recompute / defer` 等动作建议。
5. 建立在线对象画像能力，使 controller 可以根据真实回执更新对象状态、优先级和动作白名单。
6. 保持安全边界，若底层 API 或受控扩展能力不足，则动作明确标注为 blocked，不做伪实现。

阶段交付物：
- 完整动作链路的结构化运行证据
- 增强后的 `OnlineProfileStore`
- 支持多动作决策的 `OnlinePolicyController`
- 分层对象建模与增量状态更新能力
- 动作成功、失败、拒绝、超时原因的回流机制

阶段验收标志：
1. `drop / offload / load / prefetch / evict` 五类动作均存在真实结构化样本。
2. controller 不再只输出单一 `drop` 建议，而是能基于能力白名单与真实状态产出完整动作建议。
3. execution disabled 与 execution enabled 两种模式边界清晰、可测试、可回放。

### 3.3 任务三：完成基准验证、收益证明与最终汇报包

任务目标：
把工程实现转化为可以正式汇报的结果包，回答“是否真的执行了动作、是否真正带来收益、是否没有损害质量”这三个核心问题。

重点工作：
1. 固定评估路径，围绕现有 `Task1/QASPER` 与 benchmark runner 开展 baseline 和 AstraKV-enabled 成对实验。
2. 保证模型、硬件、容量、connector、启动参数、数据顺序与质量评估方法完全一致，确保对照结果可采信。
3. 以 verified structured events 作为 correctness 与动作统计的唯一可信来源，日志与启发式结果仅作为辅助诊断。
4. 输出统一格式的运行结果、质量结果、配对比较结果和汇总结论，形成完整的对外交付包。
5. 对未被证据支持的能力或收益结论自动降级为 advisory 或 blocked，避免对外口径过度承诺。

阶段交付物：
- baseline 与 AstraKV-enabled 的成对实验结果
- `request_results.jsonl`
- `benchmark_results.csv`
- `quality/`
- `samples/<case>_samples.csv`
- runtime artifact 全量证据包
- paired comparison 与 suite summary
- 最终对外汇报文档

阶段验收标志：
1. correctness、动作一致性、run/hash 一致性验证全部通过。
2. 能够完整报告 `precision / recall / F1`、TTFT、TPOT、吞吐、质量和资源开销。
3. 最终报告可以明确回答动作真实性、指标收益和质量边界三个问题。

## 4. 里程碑安排

为便于对外管理，本计划将原有多阶段技术任务收束为三段式里程碑：

1. 里程碑一：完成真实在线底座与 `drop` 基线，证明控制面已经从“可观察”走向“可验证执行”。
2. 里程碑二：完成动作扩展与在线策略闭环，证明系统已经从“单动作演示”走向“多动作在线调度”。
3. 里程碑三：完成对照实验与最终汇报包，证明系统已经从“工程实现”走向“可验收成果”。

## 5. 验收口径

1. AstraKV 已完成真实在线控制面的主线建设，具备从对象绑定、命令下发、结构化回执到策略反馈的完整链路。
2. 动作能力按照安全边界分阶段开放，只有经过真实后端验证的动作才计入正式能力说明。
3. 最终收益结论以 baseline 对照实验为准，重点报告时延、吞吐、质量保持和资源开销之间的综合表现。

## 6. 风险与约束说明

1. 当前主线只覆盖锁定版本组合，不承诺同步支持多版本后端。
2. 若真实后端缺少公开 API，需要通过受控 fork 或 connector 扩展补齐；如无法补齐，则相应动作保持 blocked。
3. 若真实 DGX 证据与当前受控测试结果冲突，以真实环境证据为准，并同步调整对外能力声明。
4. 对于 `cache_key` 和 `block` 等更细粒度对象层级，只有在 identity 稳定可靠时才会正式开放。

## 7. 预期结果

计划完成后，AstraKV 对外将不再停留在“具备潜力的控制策略原型”，而是可以明确表述为：

“AstraKV 已在锁定版本技术栈上打通真实在线 KV 控制主线，具备分阶段开放的真实动作能力、可回放的结构化证据体系，以及基于对照实验的性能与质量评估能力。”
