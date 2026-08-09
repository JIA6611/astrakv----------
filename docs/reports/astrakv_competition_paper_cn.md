# AstraKV：面向内存受限大语言模型推理的可观测分层 KV 管理系统

> 全国大学生计算机系统能力大赛操作系统功能挑战赛道项目报告  
> 项目方向：大语言模型推理访存分析、分层缓存、按需加载与预取  
> 文档状态：论文正文初稿，系统与实现章节已展开，实验数值与插图位置暂以占位符保留  
> 目标平台：DGX Spark GB10、Qwen3-8B、vLLM 0.23.0、LMCache 0.4.7、单 worker

## 摘要

大语言模型部署到资源受限设备时，首先遇到的往往不是算力不足，而是模型参数、KV cache 和运行时工作集共同造成的内存压力。模型参数决定了服务启动时的基础占用；KV cache 随上下文长度和并发请求数持续增长；在混合专家模型中，专家参数总量很大，但每个 token 实际激活的专家数量又相对有限。三类对象的共同特点是容量大，差异在于访问规律不同：模型层按前向顺序访问，KV 前缀可能跨请求复用，专家对象则由路由器稀疏选择。固定或可预测的访问顺序，使按需加载、分层驻留、对象换出和提前预取具备了应用空间。

AstraKV 围绕这一问题构建了一套面向 vLLM 与 LMCache 的分层 KV 控制系统。项目没有重新实现推理引擎，也不直接改写 vLLM 的 paged KV，而是在原生调度器和缓存连接器之外增加对象身份、行为观测、策略决策、安全执行和证据验收几层能力。系统以精确 token IDs、模型兼容条件、LMCache 原生 key 和绑定代际共同确定 KV 对象；以 Cache Event、原生回调、命令回执和请求核算描述对象从查找、准入、加载到消费的过程；以 ProfileDB、ChunkScorer、LoadRecomputePlanner 和 UnifiedObjectScheduler 评估复用价值、加载成本、重计算成本和内存压力；以 RuntimeControlHost、Execution Gate、Circuit Breaker 和 LMCache 0.4.7 动作端点约束 drop、offload、prefetch、evict 与 request-owned load 的执行边界。

项目路线经历了两个阶段。初赛阶段首先通过真实 vLLM/LMCache endpoint、外部预热、压力测试、32K 边界实验、离线策略和 mmap 虚拟内存 PoC，确认了分层存储与前缀复用问题具有可观测性。但这一阶段的预取主要发生在 endpoint warmup 层面，策略输出也没有进入 vLLM 内部调度器。当前版本把重点转向真实运行时闭环：预取被严格定义为目标请求出现后、KV 消费前的 SSD 到 CPU 迁移；GPU KV 的写入仍归原生 connector 所有；任何请求身份、对象绑定、版本能力或回执证据不完整的运行都按故障关闭处理。这样做增加了工程复杂度，却避免把日志推断、HTTP 预热或估算完成量误写成原生 KV 动作。

除 KV 主线外，仓库还实现了基于稀疏文件、`mmap`、`madvise` 和 `mincore` 的操作系统虚拟内存机制原型，用于验证页级按需访问、软件预取、回收提示与驻留率观测；实现了 Qwen3-8B 的离线 KV 敏感性分析、层传输测量骨架，以及 MoE 路由解析、专家预测和驻留规划工具。上述模块共同覆盖了赛题提出的参数、KV 和专家访问分析方向，但当前完成度并不相同：KV 控制面已经进入版本锁定的真实运行时接入阶段，OS VM、参数和 MoE 仍以机制验证或离线分析为主。

本文从项目问题、理论依据、路线演进、系统架构、核心模块和创新点展开，重点说明 AstraKV 为什么从“外部可观测”转向“身份与证据驱动的运行时控制”，以及每层模块如何在不破坏原生推理正确性的前提下管理 KV 对象。实验章节暂保留实验设计、指标、验收规则和图表位置，待目标平台结果完成后补入数据，不在当前稿件中预设性能结论。

**关键词：** 大语言模型推理；KV cache；分层存储；虚拟内存；按需加载；预取；vLLM；LMCache；统一内存；MoE

## Abstract

Deploying large language models on memory-constrained systems is limited not only by computation, but also by the combined footprint of model parameters, runtime KV cache, and temporary working sets. The KV cache grows with context length and concurrency, while expert parameters in mixture-of-experts models exhibit sparse activation. These access patterns create opportunities for demand loading, tiered residency, eviction, and prefetching.

AstraKV is a tiered KV management system built around vLLM and LMCache. Instead of replacing the serving engine or directly writing paged GPU KV tensors, AstraKV introduces a control plane for exact object identity, runtime observation, policy planning, guarded execution, and evidence-based validation. KV objects are identified by exact token IDs, compatibility metadata, LMCache native keys, and binding generations. Runtime decisions combine reuse profiles, estimated I/O cost, recomputation cost, deadlines, and memory pressure. Backend actions are constrained by version-locked capabilities, request ownership, execution budgets, circuit breakers, and native receipts.

The project evolved from an endpoint-oriented prototype into a native runtime integration effort. The preliminary version demonstrated real vLLM/LMCache endpoints, stress and long-context workloads, offline policy analysis, and an independent mmap-based virtual-memory prototype. The current version further distinguishes endpoint warmup from native prefetch, advisory decisions from executed actions, and lookup observations from completed KV loads. It defines prefetch as an SSD-to-CPU transfer associated with a known target request, while preserving the native connector as the only writer of paged GPU KV.

This report presents the motivation, theoretical foundation, route selection, system architecture, core modules, and implemented innovations of AstraKV. Experimental sections retain the methodology and acceptance criteria but leave measurements and figures as explicit placeholders until target-platform validation is complete.

**Keywords:** large language model inference; KV cache; tiered storage; virtual memory; demand loading; prefetch; vLLM; LMCache; unified memory; mixture of experts

---

## 文档占位符约定

当前稿件使用以下三类占位符。占位符不是缺失说明，而是后续实验和制图的接口。

- **图占位符：** 给出图号、图题、需要绘制的对象、数据来源和预期表达关系；后续用架构图、流程图、时序图或实验图替换。
- **表占位符：** 给出字段、分组方式和验收口径；待结果产生后填入数值，不在当前阶段构造示例数字。
- **结论占位符：** 只写该实验允许回答的问题，不提前写“提升”“降低”或“通过”等结论。

所有待补结果统一使用 `【待实验】` 标记，所有待制图位置统一使用 `【待制图】` 标记。正式提交前应全文搜索这两个标记，逐项替换或删除。

---

## 目录

1. 项目介绍与目标描述  
2. 研究背景、理论基础与问题建模  
3. 相关工作分析与项目路线选择  
4. AstraKV 系统框架设计  
5. 关键技术与创新点  
6. 核心模块实现  
7. 系统运行流程与安全机制  
8. 实验设计、验证方法与结果占位  
9. 工程组织、使用方法与复现路径  
10. 当前边界、风险与后续工作  
11. 总结  
附录 A. 图表清单  
附录 B. 代码与能力索引  
附录 C. 实验结果填充检查表

---

# 1 项目介绍与目标描述

## 1.1 项目面对的直接问题

当前项目面对的直接问题是：在物理内存和显存预算有限的设备上，如何让大语言模型继续支持较长上下文和一定并发，而不是在服务启动或请求增长阶段因为 KV cache 容量不足而提前触碰上限。

这个问题容易被简化成“把缓存放到磁盘”，但真正落地时至少有四个难点。第一，推理引擎内部的 KV cache 不是普通文件缓存，而是由调度器、block allocator 和 connector 共同管理的运行时状态；第二，一个 KV 对象是否可以复用，取决于模型、tokenizer、chat template、数据类型、位置编码、KV 布局和精确 token 前缀，不能只看文本是否相似；第三，把对象移到更慢的层级虽然能够释放高层容量，却可能把原本的内存瓶颈转化成 I/O 阻塞；第四，端到端 TTFT 发生变化并不能自动证明发生了 KV 命中、预取或换出，实验必须能够追踪对象的真实生命周期。

因此，AstraKV 解决的不是单一缓存算法问题，而是一个跨越推理引擎、缓存后端、操作系统和实验验证的系统问题。项目需要同时回答四个问题：对象是什么、何时需要它、由谁移动它、如何证明动作真实发生。

> **【待制图：图 1-1 内存受限 LLM 推理问题示意图】**  
> 绘图内容：左侧展示模型参数、KV cache、激活和 staging buffer 随模型规模、上下文和并发增长；中间展示 GPU/UMA、CPU/UMA、SSD 三个逻辑层级；右侧展示 OOM、TTFT 增长和吞吐下降三类后果。  
> 图中必须把“逻辑 tier”与“独立物理内存”区分开，DGX Spark 部分标注 UMA。

## 1.2 推理内存压力的组成

### 1.2.1 模型参数

模型参数决定了推理服务的基础驻留量。对一个包含 $P$ 个参数、每个参数使用 $s_w$ 字节表示的模型，未考虑框架开销时，权重存储量可近似写为：

$$
M_{weight} \approx P \times s_w.
$$

实际运行还会增加张量元数据、算子 workspace、图捕获缓冲和临时激活。对于稠密模型，前向计算通常按层依次读取参数；对于 MoE 模型，公共层仍然参与每次前向，但专家层只访问路由器选中的少数专家。这种顺序性和稀疏性是层级加载和专家预取的理论依据。

AstraKV 当前没有把权重卸载作为主线执行对象。仓库中的 `layer_offload.py` 用于层传输和显存占用测量，Qwen3 离线 profile 用于观察 KV 与层敏感性；它们为后续参数对象扩展提供接口，但不应被描述为已经完成真实权重分层推理。

### 1.2.2 KV cache

KV cache 是当前版本的主线对象。自回归模型在 prefill 阶段为输入 token 计算 key 和 value，在 decode 阶段持续读取历史 KV 并追加新 token 的 KV。其容量随序列长度和并发数增长，不会像单次激活那样在一层计算结束后立即释放。

KV 还有一个区别于模型权重的特点：不同请求可能共享完全相同的 token 前缀。固定 system prompt、共享检索文档、模板化任务和多轮会话都可能产生重复前缀。如果这些前缀已经存入外部缓存，重新计算不是唯一选择；系统可以在兼容性满足时加载已有 KV，或者在预计将被使用时提前把它放到更靠近计算的位置。

### 1.2.3 专家参数与稀疏对象

MoE 模型包含多个专家，但 router 通常只为每个 token 选择 top-k 专家。若路由行为在相邻 token、相似上下文或特定层上具有局部性，系统可以保留热点专家、换出冷专家，或者根据路由历史产生预取提示。

不过，专家对象比 KV chunk 更大，错误预取的代价也更高。当前 AstraKV 只完成专家路由事件归一化、预测和驻留规划，没有把 expert tensor 迁移接入真实 serving。因此本文把 MoE 作为统一对象模型的扩展验证，而不是与 KV-Core 同等完成度的执行路径。

## 1.3 为什么访存行为可以被利用

大语言模型推理不是任意访存程序。它的执行结构为系统优化提供了三个前提。

第一，层访问顺序相对固定。Transformer 前向沿网络层推进，参数和中间状态的访问范围具有明显阶段性。第二，KV 的时间局部性来自请求历史和前缀复用，虽然不能保证每次出现，但可以通过 trace 和 ProfileDB 统计。第三，MoE 专家访问由 router 决定，激活数量稀疏，并可能呈现转移概率和热点分布。

这些规律并不意味着预取一定有效。只有对象身份准确、预测提前量大于加载时延、目标层有足够预算，并且预取对象最终被目标请求消费时，I/O 才可能被隐藏。AstraKV 的设计重点正是把这些必要条件逐项显式化，而不是把“可预测”直接写成“已获得收益”。

## 1.4 项目路线的演进

### 1.4.1 第一阶段：先建立可运行、可观测的外部路径

项目初期优先完成真实 vLLM、LMCache CPU、LMCache Disk 三类 endpoint，围绕 baseline、prefetch、stress、32K boundary、Cache Event、OS VM 和 MoE trace 组织实验。这样做的原因很实际：在无法稳定修改第三方 runtime 的阶段，先用 endpoint、日志和独立 PoC 回答“分层路径能否运行”“前缀预热是否影响 TTFT”“磁盘后备是否能支撑长上下文”“页缓存机制是否可测”等基础问题。

这一阶段形成了 ProfileDB、ChunkScorer、PartialKVLoadPlanner、LoadRecomputePlanner 和 UnifiedObjectScheduler 等模块，但策略主要输出 advisory record。所谓 Selective Prefetch 也以 endpoint-level warmup 为主，无法控制内部 PagedAttention block。初赛版本已经主动写明这一边界，只是从系统演进角度看，它说明“观察”和“控制”之间仍缺少一层真实连接。

### 1.4.2 第二阶段：从外部现象转向原生生命周期

当项目尝试把策略用于真实请求时，问题发生了变化。此时不能只知道“某个前缀似乎命中”，还要知道：这个请求的精确 token IDs 是什么；LMCache 使用了哪个 native key；当前物理对象属于哪个层；调度器准入了多少 external tokens；原生 connector 实际加载了多少；剩余 token 是否由 prefill 重计算；对象在命令到达前是否已经被释放并复用。

当前版本因此增加了 compatibility key、request context、physical binding、binding generation、native callback、load receipt、request accounting、execution gate 和 action service。策略仍然重要，但它不再是系统的中心。系统的中心变成了一条可以复核的对象生命周期：查找、绑定、决策、执行、回执、消费和终止。

> **【待制图：图 1-2 AstraKV 两阶段演进路线图】**  
> 左半部分为初赛阶段：endpoint、日志、offline policy、warmup、VM PoC；右半部分为当前阶段：exact identity、native callback、binding、gate、action、receipt、accounting。  
> 中间用“观察与控制之间的断层”连接，突出路线升级而不是简单功能堆叠。

## 1.5 项目目标

AstraKV 当前目标分为五层，层与层之间存在明确依赖。

### 1.5.1 建立真实访存证据

系统需要从真实 benchmark 输出、server log、LMCache callback、对象绑定和资源采样中提取结构化事件，覆盖 request、token prefix、chunk、block、tier、bytes、latency 和 terminal state。事件不仅用于画图，还要能够回溯某个策略动作的输入和结果。

### 1.5.2 建立统一 KV 对象模型

系统需要把逻辑请求、精确 token 前缀、LMCache native key、物理 KV 对象和层级状态连接起来。兼容性条件不完整时，默认不复用；binding generation 不匹配时，默认不执行动作。

### 1.5.3 建立分层策略

系统根据复用历史、加载时延、deadline、对象大小、内存压力和预取浪费率，输出 keep、load、recompute、prefetch、offload、evict、drop 或 defer。策略需要同时支持离线分析、shadow 观测和 active 执行，而不是为每种实验维护一套不一致的逻辑。

### 1.5.4 建立受约束的真实动作路径

GPU KV 继续由 vLLM/LMCache 原生 connector 管理；AstraKV 的 prefetch 只做 SSD 到 CPU；offload、drop 和 evict 通过 LMCache 0.4.7 manager/backend 执行；load 只能注册 request-owned target，并由原生请求生命周期完成 GPU 写入。动作必须通过版本、能力、身份、预算、时限和断路器检查。

### 1.5.5 建立可判定的实验验收

每组 baseline/variant 需要固定模型、版本、随机种子、workload、请求顺序、输出长度和 cache state，并对环境、输入、命令、回执和 binding 计算或核对哈希。实验只有在 `eligible=true` 时才能用于性能结论。失败实验保留诊断价值，但不进入收益统计。

## 1.6 核心研究问题

本文围绕以下研究问题展开。

**RQ1：如何定义能够安全复用的 KV 对象？** 需要确定哪些模型和运行时条件必须进入 compatibility key，以及如何把 exact token prefix 与 LMCache native key 对齐。

**RQ2：何时应加载，何时应重计算？** 需要比较 I/O 时间、prefill 时间、deadline 和资源压力，并考虑 partial prefix 对加载量和重计算量的影响。

**RQ3：预取如何与普通 warmup、同步加载区分？** 需要限定目标请求出现时间、source/destination tier、完成时间和真实 consumer，并对 waste、late、cancel 单独核算。

**RQ4：如何在不破坏 vLLM 对 GPU KV 的所有权前提下执行动作？** 需要利用原生 connector，把 AstraKV 限定在控制、后端层级迁移和 request-owned load target 范围内。

**RQ5：如何证明内存真的下降？** 在 DGX Spark UMA 上，需要区分逻辑 tier occupancy、vLLM KV capacity 和系统总物理内存，不能把 GPU budget 变化直接当作物理内存节省。

## 1.7 已实现的主要工作

围绕上述问题，当前仓库已实现以下工作。

1. KV chunk、block、tier、compatibility key、request binding、physical object、prefetch ticket 和 native receipt 等对象结构。
2. Cache Event、trace schema、ProfileDB、online checkpoint、质量敏感度和分层统计。
3. ChunkScorer、PartialKVLoadPlanner、LoadRecomputePlanner、UnifiedObjectScheduler 和 OfflineEvictionSimulator。
4. off、shadow、active 三种运行模式，以及 Offline Safety Gate、Runtime Execution Gate 和 Circuit Breaker。
5. 认证 request context、binding registry、RuntimeControlHost、backend command/receipt 契约。
6. 面向 vLLM 0.23.0、LMCache 0.4.7 的版本锁定 hook 和 drop/offload/prefetch/evict/load target 适配。
7. 命令、回执、binding、request accounting、resource samples 和 paired-run acceptance 产物。
8. 基于 mmap/madvise/mincore 的 OS VM PoC，以及参数层传输、Qwen3 KV 敏感性和 MoE 路由分析工具。

## 1.8 系统边界

本文中的“已实现”表示仓库存在可执行代码和相应测试或脚本，不自动表示目标硬件验收已经通过。当前边界如下。

- KV-Core 在线控制代码已经形成，但真实 DGX 验收仍需实验结果补齐。
- 预取严格指 SSD 到 CPU，不指 HTTP warmup，也不指提前写入 GPU。
- partial load 只支持从 token 0 开始、连续且 block/chunk 对齐的 external-prefix 上界。
- mmap VM 是独立机制 PoC，没有替换 vLLM 内部 paged KV allocator。
- layer offload 没有执行完整 Transformer forward，也没有实现计算与传输重叠。
- MoE 模块生成 route、prediction 和 load plan，不移动真实 expert weight。
- 本文实验章节保留设计和验收标准，待结果完成后再填入数据和结论。

---

# 2 研究背景、理论基础与问题建模

## 2.1 大语言模型推理过程

一次自回归推理通常分为 prefill 和 decode 两个阶段。prefill 接收完整输入序列，批量计算每个位置的隐藏状态并建立 KV cache；decode 每次生成一个或少量 token，读取已有 KV，计算新 token，并把新 KV 追加到缓存。

prefill 具有更高并行度，但其计算量随输入长度增长。decode 单步计算规模较小，却需要反复读取历史 KV，容易受到内存带宽和缓存容量影响。长上下文服务中，TTFT 很大程度上受 prefill、外部 KV 恢复、调度排队和首次输出准备影响；TPOT 则更接近连续 decode 的稳定成本。

可以把首 token 时间近似分解为：

$$
T_{TTFT}=T_{queue}+T_{lookup}+T_{load}+T_{prefill}+T_{prepare}+T_{emit}.
$$

若外部 KV 命中，$T_{prefill}$ 可能因复用前缀而下降，但会增加 $T_{lookup}$ 和 $T_{load}$。预取的目标不是消除 I/O，而是把一部分 $T_{load}$ 与排队或其他准备时间重叠。

> **【待制图：图 2-1 Prefill、Decode 与 KV 生命周期时序图】**  
> 绘图内容：请求入队、exact lookup、external admission、SSD/CPU load、prefill remainder、decode、request finished。  
> 图中区分同步 demand load 与提前 prefetch 两条路径。

## 2.2 KV cache 容量模型

设 Transformer 层数为 $L$，每层 KV head 数为 $H_{kv}$，head dimension 为 $D_h$，每个元素占 $s$ 字节，则每个 token 的 KV 大小近似为：

$$
B_{token}=2LH_{kv}D_hs,
$$

其中系数 2 对应 key 和 value。对并发序列集合 $\mathcal{R}$，第 $r$ 个请求已缓存 token 数为 $T_r$，则总 KV 数据量近似为：

$$
B_{KV}=\sum_{r\in\mathcal{R}}T_rB_{token}.
$$

实际系统不会按单 token 任意分配，而是以 block 为单位管理。若 block size 为 $b$，实际分配 token 数为 $\lceil T_r/b\rceil b$，因此存在对齐开销。外部缓存又可能按更大的 chunk 组织对象，load、prefetch、offload 和 evict 的粒度由 block、chunk 和后端 API 共同决定。

当前目标配置使用 16-token block。partial prefix 也必须遵守 block/chunk 对齐，避免控制器生成原生 connector 无法安全执行的任意 tensor slice。

## 2.3 工作集、局部性与复用距离

把一段时间内可能再次访问的对象集合记为工作集 $W(t,\Delta)$。若高层容量能够覆盖该工作集，系统可以保留热点对象并减少重复加载；若工作集超过容量，就需要选择 victim。

KV 的复用不只由访问次数决定。两个对象访问次数相同，但一个刚被访问且很快再次出现，另一个已经长时间不再出现，它们的保留价值不同。AstraKV 因此在 ProfileDB 中记录 reuse count、recent access、cache hit、load latency、prefetch history、tier count 和 bytes 等信息，再由 scorer 把历史事实转换为当前建议。

离线策略模块提供 LRU、FIFO、Belady 和 AstraKV victim 规则。Belady 需要未来信息，只能作为理论上界或离线参照；LRU 和 FIFO 提供简单基线；AstraKV 策略则把复用、加载、大小、压力和预取历史结合起来。离线 simulator 输出的 latency、I/O 和 OOM 是 proxy，不能替代真实端点测量。

## 2.4 分层存储模型

AstraKV 使用 GPU、CPU 和 SSD 三个逻辑层级。

| 层级 | 管理者 | 主要作用 | 主要约束 |
| --- | --- | --- | --- |
| GPU paged KV | vLLM/LMCache 原生 connector | 被注意力计算直接消费 | 容量有限，所有权必须由 scheduler 保持 |
| LocalCPUBackend | LMCache | demand load staging、CPU-hot KV、SSD prefetch 目标 | 占用主存或 UMA，错误预取会增加压力 |
| LocalDiskBackend | LMCache | 大容量后备对象 | 时延高，需要真实 native key 和 I/O 预算 |

在传统独立显存设备上，GPU 和 CPU 往往对应不同物理内存池；在 DGX Spark GB10 上，它们共享 UMA。逻辑层级仍然重要，因为 vLLM block budget、LMCache CPU occupancy 和 SSD I/O 分属不同管理路径，但物理内存结论必须基于系统总峰值，而不是逻辑名称。

## 2.5 虚拟内存与按需调页

文件映射允许进程创建大于当前物理驻留量的虚拟地址空间。页面首次访问时，内核通过缺页把后备文件内容载入页缓存；暂时不再使用的页可以被回收，而文件仍保存数据。对 KV 对象而言，这一机制提供了三个基础能力：逻辑空间与物理驻留分离、first-touch demand paging、以及操作系统参与的页缓存回收。

当前 VM PoC 使用稀疏文件和 `mmap` 表示 KV block，通过 `MADV_WILLNEED` 发出预取提示，通过 `MADV_DONTNEED` 发出近期不再需要的提示，并用 `mincore` 查询页面驻留。需要强调，`madvise` 是提示而不是强制命令；内核是否预读、回收多少页，受内存压力、文件系统和内核策略影响。因此 VM 章节应同时报告 resident ratio、RSS、cold/warm latency 和实际读写事件。

## 2.6 预取窗口与收益条件

设目标对象加载时间为 $T_{io}$，从预取发起到对象被请求消费之间的提前量为 $T_{lead}$。只有当：

$$
T_{lead}\ge T_{io}
$$

时，I/O 才可能完全隐藏。若 $0<T_{lead}<T_{io}$，仍可能减少同步等待；若预取在消费之后完成，则属于 late prefetch。

预取收益还必须扣除错误预取的代价。设命中收益为 $G_{hit}$，浪费 I/O 成本为 $C_{waste}$，额外驻留成本为 $C_{resident}$，调度开销为 $C_{control}$，则预期净收益可以表示为：

$$
E[G]=P(hit)G_{hit}-P(waste)C_{waste}-C_{resident}-C_{control}.
$$

这解释了为什么预取不能只看 hit rate。一个体积很大的对象即使最终被消费，如果它提前占用过多内存或挤出更有价值的对象，也可能得不偿失。AstraKV 的 scorer 因而把对象大小、内存压力、历史 waste 和 deadline 同时纳入决策。

## 2.7 加载与重计算模型

外部 KV 存在时，系统仍需要判断加载是否值得。加载估计时间可写为：

$$
T_{load}=T_{fixed\_io}+\frac{B_{load}}{BW_{effective}},
$$

重计算估计时间可写为：

$$
T_{recompute}=T_{fixed\_compute}+N_{missing}\cdot C_{prefill/token}.
$$

若 $T_{load}<T_{recompute}$，并且加载能够在 deadline 前完成，load 才具有时间上的合理性。实际 planner 还要考虑后端是否可用、对象是否仍驻留、请求是否取消、I/O 是否拥塞、是否存在 bootstrap 测量，以及估计值是否来自当前 workload。

系统必须保留 recompute fallback。外部对象缺失、过期、超时、绑定失效或 receipt 不完整时，正确行为不是伪造加载完成，而是让 vLLM 计算缺失前缀。

## 2.8 部分前缀加载

完整加载命中前缀并不总是最优。如果对象很大、deadline 较近，或者只加载前半段就足以减少大部分 prefill，可以选择 external partial prefix。设 lookup 命中 $N_{hit}$ 个 token，系统准入 $N_{external}$，原生实际加载 $N_{loaded}$，则必须满足：

$$
N_{hit}\ge N_{external}\ge N_{loaded}.
$$

剩余前缀由原生 prefill 重计算：

$$
N_{prefix}=N_{loaded}+N_{recomputed},
$$

若请求取消或失败，则需要 terminal reason 解释差额。

当前实现只允许从 token 0 开始的连续前缀，并要求 block/chunk 对齐。这是一种保守设计。它放弃任意 layer/token slice 的表达能力，换取与 scheduler admission、slot mapping 和原生 load token mask 的一致性。

## 2.9 MoE 稀疏激活模型

设某 MoE 层包含 $E$ 个专家，router 为 token $t$ 选择集合 $A_t$，且 $|A_t|=k\ll E$。若预测器给出集合 $\hat A_t$，可定义：

$$
Precision_t=\frac{|\hat A_t\cap A_t|}{|\hat A_t|},\qquad
Coverage_t=\frac{|\hat A_t\cap A_t|}{|A_t|}.
$$

只追求 coverage 会倾向于预取更多专家，增加 waste；只追求 precision 又可能漏掉关键专家，造成同步加载。因此专家 planner 还需要专家大小、层级驻留、加载时延和 GPU/CPU budget。

AstraKV 当前的 RouterAwareExpertPredictor 支持 previous token、history window、transition model、layer hotness 和 load-plan residency 等输入，SelectiveExpertLoaderPlanner 输出 keep/load GPU、keep CPU、offload SSD、drop 或 on-demand 建议。由于没有真实 expert tensor movement，这些指标属于离线分析，不进入 KV-Core 性能结论。

## 2.10 DGX Spark UMA 下的内存解释

DGX Spark GB10 使用统一内存。传统 dGPU 场景中常见的“GPU 显存下降、CPU 内存上升”在 UMA 上可能只是同一物理内存池内的逻辑归属变化。AstraKV 因此区分三种结论。

第一种是 **KV capacity 结论**：在相同 SLO 下减少 vLLM KV block budget，或支持更长上下文、更高并发。第二种是 **逻辑层级结论**：对象从 GPU paged KV 转到 LMCache CPU 或 SSD，改变了调度器可用容量。第三种是 **物理内存结论**：cgroup memory、进程 RSS 和 UMA 总峰值真实下降。

只有第三类指标得到一致证据时，报告才能写“物理内存降低”。如果只是 vLLM block budget 下降，而 LMCache CPU occupancy 同步上升，应写“KV 容量重新分配”，不能写成总内存节省。

## 2.11 性能与正确性指标

系统实验需要同时覆盖性能、资源、缓存行为、动作正确性和生成质量。

| 维度 | 指标 | 解释 |
| --- | --- | --- |
| 首 token | TTFT、TTFT p50/p95/p99 | 反映排队、lookup、load 与 prefill |
| 连续生成 | TPOT、inter-token latency | 反映 decode 稳态性能 |
| 整体能力 | throughput、success rate | 防止只优化单请求延迟 |
| 内存 | RSS、cgroup memory、KV blocks、CPU occupancy、disk occupancy | 区分物理内存与逻辑层级 |
| I/O | read/write bytes、latency、queue depth | 判断迁移成本与预取重叠 |
| 缓存 | hit/miss、loaded tokens、recomputed tokens | 核对实际复用量 |
| 预取 | issued、hit、waste、late、cancel、lead time | 判断预取是否真实且有效 |
| 正确性 | output token IDs、finish reason、loss/logprob、质量指标 | 防止性能来自错误跳算 |
| 证据资格 | callback、binding、receipt、hash、accounting | 判断实验能否用于结论 |

## 2.12 问题形式化

在时间窗口 $[0,T]$ 内，设对象集合为 $O$，层级集合为 $\{G,C,S\}$，分别表示 GPU、CPU、SSD。对象 $o$ 在层级 $l$ 的驻留变量为 $x_{o,l}(t)\in\{0,1\}$，层级容量为 $Cap_l$，对象大小为 $Size_o$，则有：

$$
\sum_{o\in O}x_{o,l}(t)Size_o\le Cap_l.
$$

系统希望在正确性和 SLO 约束下，最小化同步加载、重计算、迁移和内存压力的组合成本：

$$
\min \sum_t \left(
\alpha T_{stall}(t)+
\beta B_{migration}(t)+
\gamma C_{recompute}(t)+
\delta W_{prefetch}(t)+
\eta P_{memory}(t)
\right).
$$

约束包括对象兼容、binding generation、request ownership、tier capability、deadline、I/O budget 和 native receipt。AstraKV 没有在线求解全局最优，而是用 ProfileDB、scorer、planner、gate 和 breaker 把问题分解为可执行的局部决策。

---

# 3 相关工作分析与项目路线选择

## 3.1 相关技术路线分类

与赛题相关的工作大致可以分为五类：推理引擎内部 KV 管理、外部分层 KV 缓存、权重卸载和虚拟内存、KV 压缩与选择性计算、MoE 专家管理。不同路线解决的对象和所有权边界不同，不能简单用同一指标比较。

推理引擎内部方案最接近 scheduler 和 block allocator，能够直接影响 GPU KV，但版本耦合和正确性风险较高。外部 KV 缓存通过 connector 把对象保存到 CPU 或 SSD，工程接入成本较低，但需要解决请求身份、native key 和恢复时机。权重卸载和虚拟内存利用层访问顺序或页缓存降低驻留量，重点在参数对象和 OS 机制。KV 压缩减少单对象大小，却可能引入精度损失。MoE 管理利用稀疏路由，但对象更大、预测错误代价更高。

## 3.2 vLLM 与 LMCache 的基础角色

vLLM 提供模型 serving、请求调度、paged KV block 管理和 OpenAI-compatible endpoint。LMCache 通过 connector 为 KV 提供 CPU 和磁盘后端，并参与外部 KV 的 store/retrieve。AstraKV 不取代这两个系统：vLLM 继续负责推理正确性和 GPU paged KV，LMCache 继续负责 native key 与后端对象，AstraKV 负责在两者之外建立对象抽象、策略、安全门禁和证据链。

初赛路线选择 endpoint 和日志作为切入点，避免在比赛早期直接修改 runtime。当前路线则维护 vLLM 0.23.0 与 LMCache 0.4.7 的版本锁定外置补丁，补充七类 callback 和动作适配。它仍然不 vendor 整棵第三方源码，但不能再描述为“完全不修改第三方代码”。

## 3.3 分层缓存与替换策略

传统 LRU、FIFO 和 LFU 只使用历史访问信息，优点是简单稳定，缺点是难以表达加载成本和对象大小。Belady 使用未来访问位置，可作为离线最优参照，但无法直接在线部署。面向 KV 的策略还要考虑前缀结构、token 数、block/chunk 对齐和 recompute 成本。

AstraKV 没有提出一个孤立的替换算法，而是把策略拆成三步：ChunkScorer 判断对象价值，LoadRecomputePlanner 判断缺失状态如何恢复，UnifiedObjectScheduler 在预算下输出最终动作。拆分后，同一对象可能“复用价值较高但加载来不及”，也可能“复用次数不高但重计算极贵”，策略能够保留这些差异。

## 3.4 权重卸载、KV 卸载与虚拟内存路线

相关工作中的权重卸载通常利用层序执行，把暂时不参与计算的层放在 CPU 或 SSD；KV 卸载则面对请求级和 token 级动态增长；虚拟内存路线进一步利用页表、缺页和后备文件分离虚拟地址与物理驻留。

初赛报告参考了 FlexInfer、NEO、IMPRESS、Cake、Bidaw、ShadowKV、RocketKV、Jenga、KTransformers、Klotski 等工作，用于理解分层、预取、选择性 KV 和专家管理思路。AstraKV 当前没有复制这些系统的内核或算法实现。项目的实际贡献集中在 vLLM/LMCache 运行时控制和证据组织，OS VM 则通过独立 PoC 验证机制可行性。

## 3.5 为什么没有直接重写推理引擎

直接修改 scheduler、block allocator 或 CUDA kernel 能获得更强控制能力，但会同时扩大正确性和版本维护风险。比赛环境还要求在真实模型、真实 serving 和限定硬件上复现，一旦底层 patch 过大，问题很难区分是策略错误、对象身份错误还是引擎兼容问题。

项目因此采用“保持原生所有权、增加外置控制”的路线。GPU KV 的最终写入仍由原生 connector 完成；AstraKV 通过 callback 观察 scheduler 和 connector，通过 LMCache backend 执行 CPU/SSD 动作，通过 request-owned target 表达 load 意图。这个选择不追求对 runtime 的无限控制，而是优先保留可回退性和可审计性。

## 3.6 初赛路线带来的认识

初赛版本验证了三件重要事实。第一，vLLM 和 LMCache CPU/Disk 可以形成真实端点和分层路径；第二，外部预热能够改变重复前缀场景的 TTFT；第三，32K 边界和 mmap 页缓存都可以被脚本化、指标化。

它也暴露了三个断层。第一，endpoint warmup 不是原生 prefetch，额外请求本身已经完成了 prefill。第二，Cache Event 和策略 CSV 能说明控制器“建议了什么”，却不能证明 scheduler 采用了建议。第三，日志中的 cache hit 不能自动回答“查到多少、准入多少、实际加载多少、剩余多少由计算完成”。

当前架构正是围绕这三个断层展开。换句话说，AstraKV 的系统故事不是从一个预先设计完整的框架开始，而是从可运行的外围实验出发，逐步发现缺少对象身份、原生所有权和因果证据，最终形成当前 KV-Core。

> **【待制图：图 3-1 从初赛外围路径到当前 KV-Core 的问题驱动演进图】**  
> 建议采用“现象 - 暴露问题 - 新增机制”三列结构：warmup TTFT - 无法证明内部预取 - PrefetchTicket；日志 hit - 无法证明真实加载 - NativeReceipt；策略 CSV - 无法证明动作执行 - Command/Receipt；对象 id - 可能复用 - BindingGeneration。

## 3.7 路线选择原则

当前路线遵循六条原则。

1. **精确身份优先。** 无 exact token IDs 和 native key，不进入 active。
2. **原生所有权优先。** 不直接写 GPU paged KV，不伪造 slot mapping。
3. **重计算始终可回退。** load 不值得或证据不足时，允许原生 prefill。
4. **执行与观测分离。** 同一策略支持 offline、shadow 和 active。
5. **结果必须可归因。** 每个 prefetch、load、offload 或 evict 都能追到 binding、generation、command 和 receipt。
6. **实验资格独立于性能数值。** 即使 TTFT 更低，只要 pair、环境或回执不合格，就不能形成收益结论。

## 3.8 AstraKV 与基础系统的关系

| 系统或组件 | 主要职责 | AstraKV 的使用方式 | AstraKV 不承担的部分 |
| --- | --- | --- | --- |
| vLLM | serving、scheduler、paged KV | 真实推理基座、native request 生命周期 | 不重写注意力 kernel 和 block allocator |
| LMCache | native key、CPU/Disk KV backend、connector | 真实分层对象和动作落点 | 不用自定义文本 hash 替代 native key |
| Linux VM | 页表、页缓存、mmap、madvise、mincore | 独立 KV block 机制 PoC | 尚未替换 vLLM paged KV |
| AstraKV Profile | 访问、时延、压力和质量统计 | 为策略提供历史证据 | 不直接移动 tensor |
| AstraKV Policy | score、load/recompute、对象调度 | 输出建议与受控动作计划 | 不能越过 execution gate |
| AstraKV Runtime | context、binding、gate、action、receipt | 连接策略与原生 backend | 不伪造原生完成状态 |

## 3.9 本项目路线的取舍

这条路线的优点是边界清楚。任何动作失败时，系统可以定位在 identity、binding、capability、gate、backend 或 receipt 中的具体环节；active 关闭后，vLLM/LMCache 原生路径仍可工作。缺点是版本适配工作量大，原生回调必须覆盖真实生命周期，而且一个看似简单的“加载 KV”需要同时满足 scheduler、connector 和 backend 的状态约束。

项目接受这一取舍。赛题最终需要的不只是一个更快的请求，而是能够解释为什么占用下降、为什么 I/O 被隐藏、为什么输出仍然正确的系统。当前路线把可解释和可验证放在策略自由度之前，为后续实验提供稳定口径。

---

# 4 AstraKV 系统框架设计

## 4.1 设计需求

从赛题描述出发，系统需要分析参数、KV 和专家的访存行为，利用按需加载和换出压缩高层驻留量，并通过预取减少 I/O 对推理的阻塞。如果直接把这三句话翻译成三个独立模块，容易得到“日志分析器、缓存搬运器、预取器”三个彼此脱节的程序。AstraKV 的设计从一开始就把它们视为同一对象生命周期的不同阶段。

一个对象被换出之前，系统要知道它是谁、当前在哪、是否仍被请求引用；一个对象被预取之前，系统要知道哪个请求将消费它、何时消费、目标层是否有空间；一个加载完成之后，系统要知道实际加载量，而不是只知道控制器曾经发出命令。由此得到六项系统需求。

**对象可识别。** KV 对象的身份必须覆盖模型和 token 兼容条件，并与 LMCache native key 对齐。

**状态可观测。** lookup、admission、load、compute、finish 和层级迁移要形成结构化事件。

**策略可解释。** 每个建议要保留复用、时延、大小、压力、deadline 和历史反馈等输入。

**动作可约束。** 版本、绑定、引用、时限、字节预算和并发不满足时，动作应被拒绝。

**失败可回退。** 任何外部 KV 路径失败时，原生 recompute 仍能完成请求。

**结果可复核。** 实验需要从 workload 一直追到 command、receipt、request accounting 和资源指标。

## 4.2 总体设计思路

AstraKV 采用“原生数据面、外置控制面、独立证据面”的总体结构。

原生数据面由 vLLM、LMCache connector、LocalCPUBackend 和 LocalDiskBackend 构成。它真正持有模型请求、GPU paged KV、CPU KV 与 SSD KV。AstraKV 不复制这份数据，也不建立第二套 GPU allocator。

外置控制面由 ProfileDB、策略模块、OnlinePolicyController、RuntimeControlHost、Execution Gate、Circuit Breaker 和 Action Service 构成。它根据观测结果生成计划，校验动作是否允许，再调用后端已有能力。

独立证据面记录 callback、binding、命令、回执、request accounting、resource samples 和 experiment manifest。证据面不参与注意力计算，却决定某次实验是否可以用于结论。

> **【待制图：图 4-1 AstraKV 总体系统架构图】**  
> 图分为三条横向泳道。最下方为数据面：vLLM Scheduler - LMCache Connector - GPU/CPU/SSD；中间为控制面：ProfileDB - Scorer/Planner - Online Controller - Gate/Breaker - Action Service；最上方为证据面：Callback - Binding - Command/Receipt - Accounting - Paired Validator。  
> 右侧单独放置 OS VM PoC 和 MoE Offline Tools，用虚线连接，表示它们尚未进入 KV-Core 主执行路径。

## 4.3 分层架构

系统进一步划分为八层。分层的目的不是追求形式上的完整，而是让每一层只回答一个问题。

### 4.3.1 工作负载与请求入口层

这一层负责把实验输入变成可复现请求。工作负载文件固定请求顺序、随机种子、sampling 参数、输出长度、context、prefix 关系和 cache state。benchmark client 在发送正式推理请求前，通过认证 context ingress 提交 request id、nonce、exact token IDs 和实验身份。

这里没有使用额外 warmup 请求代替 prefetch。cold 与 warm 状态由实验 manifest 和 pair-scoped LMCache store 显式定义；如果 warm store 不存在，脚本拒绝生成 warm 结论。vLLM 自带 prefix caching 在外部 KV 实验中关闭，避免两套缓存同时命中。

标准 workload 包括 repeated long prefix、random no reuse、constrained KV churn 和 queued concurrency。四类 workload 分别覆盖复用收益、无复用开销、容量替换压力和排队预取窗口。

### 4.3.2 对象身份与兼容层

这一层回答“这个对象能不能被当前请求使用”。`KVCompatibilityKey` 包含模型、模型 revision、tokenizer revision、chat template revision、dtype、RoPE 条件、KV layout、block/chunk、layer group 和 exact token prefix hash。

只比较文本不够。chat template 可能增加特殊 token，tokenizer revision 可能改变切分结果，同一文本在不同 dtype、RoPE 或模型 revision 下生成的 KV 也不兼容。`exact_token_prefix_hash` 因而只接受一维整数 token IDs，不接受原始字符串或 decode 后文本。

逻辑兼容之后，还要建立物理绑定。`RequestKVBinding` 把逻辑请求、runtime ReqMeta、physical object、native key 和 binding generation 联系起来。`PhysicalKVObject` 保存层级、大小和对象状态。对象删除后重新建立，即使 key 表面相同，也必须产生新的 generation。

> **【待制图：图 4-2 KV 对象身份组成图】**  
> 从左到右依次绘制 Request Context、Exact Token IDs、Compatibility Key、LMCache Native Key、PhysicalKVObject、Binding Generation。  
> 图下方注明：文本相似度、prompt hash 和 prefix label 均不能替代精确身份。

### 4.3.3 原生观测与事件层

初赛版本主要从 server log 和 benchmark 结果中提取 Cache Event。当前版本在保留日志路径的同时，引入版本锁定 callback，把 scheduler 与 connector 的关键阶段转成结构化事件。

七类回调覆盖：

1. `scheduler_exact_lookup`：scheduler 查询外部 KV，记录真实 lookup token 范围；
2. `scheduler_external_admission`：scheduler 决定本次请求准入多少 external tokens；
3. `connector_metadata`：connector 暴露 request、slot、token 和对象上下文；
4. `native_load_start`：原生外部 KV load 开始；
5. `native_load_completion`：记录实际加载 token 和 bytes；
6. `scheduler_compute_progress`：记录缺失部分的计算进展；
7. `request_finished`：请求完成、失败或取消，封闭生命周期。

事件层还解析 cache hit、cache miss、store、load、offload、prefetch、evict 和 resource sample。所有事件进入统一 trace schema，供 ProfileDB、在线 controller 和验收脚本消费。

### 4.3.4 Profile 与证据中间层

原始事件粒度太细，不能直接作为策略。ProfileDB 在 workload 和 chunk 两个层次聚合复用、命中、加载、预取、层级和质量数据。

`ChunkProfile` 记录 reuse count、cache hit/load/store、prefetch issue/hit/waste、bytes loaded、average load latency、tier counts 和最近访问。`WorkloadProfile` 记录一个 workload 的整体行为。`LayerSensitivityRecord` 和 `QualityGuardRecord` 保存离线质量分析，用于防止对敏感层或不可靠 profile 采取激进动作。

ProfileDB 既可以从离线 trace 构建，也可以由在线事件增量更新并写出 checkpoint。策略读取的是稳定中间层，而不是临时 grep 出来的日志字段。这使同一决策可以在离线模拟、shadow 和 active 中复用。

### 4.3.5 策略规划层

策略规划层分为对象价值、恢复方式和全局预算三个子问题。

`ChunkScorer` 评估对象的复用价值和驻留价值。输入包括复用倾向、deadline、加载时延、历史 prefetch hit/waste、对象大小和 memory pressure，输出 prefetch、keep、offload 或 drop 建议，以及可以写入报告的解释字段。

`LoadRecomputePlanner` 比较加载和重计算成本，输出 load、recompute、defer 或 drop。它可以结合 partial plan stats，估计完整加载、部分加载和重计算的 token/bytes。

`UnifiedObjectScheduler` 在 GPU budget 下合并 scorer 与 load/recompute 结果，生成最终对象计划。它仍然不移动 tensor，只产生标准化 decision record。

`OfflineEvictionSimulator` 用相同对象信息模拟 LRU、FIFO、Belady 和 AstraKV 策略，为安全门禁提供多个 workload 下的 proxy 结果。离线结果的作用是排除明显不稳定策略，不用于替代目标平台时延。

> **【待制图：图 4-3 策略形成过程图】**  
> 输入侧画 ProfileDB、deadline、tier snapshot、memory pressure；中间依次为 ChunkScorer、LoadRecomputePlanner、UnifiedObjectScheduler；输出侧为 keep/load/recompute/prefetch/offload/evict/drop/defer。  
> 对每个动作标出“advisory”或“可进入 execution gate”。

### 4.3.6 在线控制与安全层

`OnlinePolicyController` 把离线策略接到真实 runtime event。它根据对象当前位于 GPU、CPU、SSD 或 unknown，选择不同候选动作，并融合 ProfileDB、scheduler hint、sidecar prediction、runtime prefix reuse、prefetch window 和 profile guard。

控制器不能直接执行命令。`RuntimeExecutionGate` 校验 run id、backend 版本、capability、binding、generation、active refs、pin、pending I/O、deadline、单命令 bytes、时间窗口 bytes、命令数和并发数。任一硬条件失败，命令被拒绝并记录原因。

`CircuitBreaker` 统计失败、超时和压力。当错误累计超过策略阈值时，系统进入 cooldown，不再发送 active 动作。恢复需要显式健康信号，避免后端在不稳定状态下反复抖动。

`OfflineSafetyGate` 是 active 之前的另一道门。它要求策略至少覆盖多个 workload，并与 LRU/FIFO 等基线比较命中率、迁移量和 OOM proxy。在线 gate 解决“一条命令能不能执行”，离线 gate 解决“这套策略是否有资格进入真实执行”。

### 4.3.7 后端动作与原生接入层

动作层面向固定的 vLLM 0.23.0、LMCache 0.4.7 和 `lmcache-vllm-v1` connector。`BackendCapabilityPreflight` 先确认安装证据、运行版本、动作集合和对象粒度；版本或 patch identity 不符时，active 被阻止。

`LMCache047ActionEndpoint` 将标准动作转换为后端操作：

| 动作 | 实现语义 | 不允许的解释 |
| --- | --- | --- |
| drop | 从指定 LMCache location 删除对象 | 不是释放任意 GPU slot |
| offload | 确认 SSD 副本后移除 CPU 副本 | 不是 GPU 直接写 SSD |
| prefetch | LocalDiskBackend 读取并写入 LocalCPUBackend | 不是 HTTP warmup，不写 GPU |
| evict | 删除磁盘副本并处理临时 staging | 不能删除仍被引用或 generation 失效对象 |
| load target | 注册 request-owned external-prefix target | 真正 GPU load 由原生 connector 完成 |

`ProtectedRuntimeActionService` 使用 HMAC challenge/proof、owner-only Unix socket、命令完整性摘要、幂等 ledger 和 fsync 记录保护动作。命令重复到达时返回一致结果，进程恢复后也能区分已完成、失败和未知状态。

### 4.3.8 产物与验收层

这一层把系统运行转化为可复核材料。核心产物包括 backend capability、binding events、raw runtime events、runtime commands、command receipts、structured events 和 online profile checkpoint。KV-Core 流程进一步记录 callback smoke、native callbacks、native receipts、request accounting、context associations、prefetch tickets、policy decisions 和 UMA resource samples。

paired validator 检查 baseline 和 variant 是否使用同一 workload、模型条件、随机种子、请求顺序和 cache state，并核对 binding generation、command/receipt、terminal receipt 和 artifact hash。只有所有前置条件满足，实验才标为 `eligible=true`。

## 4.4 核心对象架构

### 4.4.1 RuntimeRequestContext

Benchmark client 与 EngineCore 不在同一进程，仅靠 HTTP request id 不能保证 scheduler 和控制器看到的是同一身份。`RuntimeRequestContextAuthority` 为 context 生成认证材料，receiver 校验 run、session、nonce 和 MAC，connector 在观察到真实 ReqMeta 后完成 association。

context receipt 分为“已记录”和“已关联”。前者只证明控制宿主收到输入，后者才证明真实 native request 与逻辑请求连接。这个区分能够防止实验脚本把 ingress 成功误写成 runtime identity 已闭合。

### 4.4.2 BackendBindingRegistry

Binding registry 保存 physical binding、active refs、pin、pending I/O 和 action reservation。一个动作从决策到执行可能跨越若干毫秒，在此期间对象可能被释放或重新分配。执行前重新核对 generation，可以阻止延迟命令作用于新对象。

Reservation 用于避免同一对象同时发生冲突动作。例如一个对象正在 prefetch 时，不应并发 evict；一个对象存在 active consumer 时，不应 drop。reservation 不是普通互斥锁，它还保留 run、command 和 terminal receipt 所需的身份信息。

### 4.4.3 PrefetchTicket

Prefetch ticket 是预取因果关系的中心对象。它包含 source/destination tier、native key、exact prefix、physical object、generation、预计与实际 bytes、deadline、TTL、状态和 consumer request。

只有目标请求在 ticket 有效期内消费了同一 native key、同一 prefix 和同一 generation 的 CPU-hot 对象，才记为 hit。预取完成但没有 consumer 记为 waste；完成晚于消费窗口记为 late；请求取消则记录 cancel。这样，报告中的 hit rate 不再由“发出过 warmup”推断，而由真实对象消费定义。

### 4.4.4 NativeKVLoadReceipt 与 RequestKVAccounting

`NativeKVLoadReceipt` 记录 lookup、admission 和实际 load 的 token/bytes。`RequestKVAccounting` 进一步把 loaded tokens、recomputed tokens、取消和失败原因汇总到请求级。

核心不变量为：

$$
N_{lookup}\ge N_{admitted}\ge N_{loaded}.
$$

如果系统只记录 lookup hit，却没有 admission 和 load completion，就不能证明 KV 已被 GPU 使用。request finished callback 负责封闭核算，避免中途事件被误当成最终结果。

## 4.5 RuntimeControlHost 架构

`RuntimeControlHost` 部署在 EngineCore 进程范围内。它同时承担四项职责：接收认证 request context、注册 backend bridge、运行 online policy worker、写出状态产物。

单 worker 拓扑下，scheduler connector 和 worker connector 对 LMCache engine 的可见性不同。当前实现根据实际 topology 确定 host owner，并采用进程内幂等 bridge 注册。首个 bridge 拥有 ingress，后续 bridge 保留自己的 native callback state，不重复占用 host。

Policy worker 使用队列消费事件。控制宿主默认 `off` 且 fail-closed，只有 mode、execution enabled、offline gate、backend preflight 和 live gate 同时允许时才向 action service 派发命令。

> **【待制图：图 4-4 RuntimeControlHost 进程与线程部署图】**  
> 展示 Benchmark Process、vLLM API Process、EngineCore Process、Worker/Connector、Policy Worker、Action UDS 和 LMCache Backend。  
> 标出 context HTTP loopback、native callback、policy queue 和 action Unix socket 四条通道。

## 4.6 数据流、控制流与证据流

三条流在系统中相互关联，但不能混为一谈。

**数据流** 从 token 输入进入 vLLM，经过 prefill/decode，GPU KV 由原生 connector 读写；外部副本在 CPU 和 SSD backend 之间移动。数据流决定真实推理结果。

**控制流** 从事件进入 ProfileDB，经 scorer、planner 和 controller 产生 action plan，再通过 gate、breaker 和 action service。控制流可以被拒绝、延迟或降级为 advisory。

**证据流** 从 callback、binding、command 和 receipt 汇入 artifact，最终进入 paired validator。证据流决定报告能否对数据流和控制流建立因果解释。

初赛版本主要具备数据流的 endpoint 路径和证据流的日志部分，控制流停留在 offline advisory。当前版本新增的主体工作，正是把三条流在对象身份上连接起来。

## 4.7 请求级端到端流程

一次候选 external-KV 请求经历以下阶段。

1. Benchmark client 读取固定 workload，生成 request context 和 exact token IDs。
2. Context authority 认证并发送 context，RuntimeControlHost 记录 receipt。
3. 正式 HTTP 请求进入 vLLM scheduler。
4. Scheduler/LMCache callback 生成 exact lookup 和 native key 证据。
5. Connector 观察真实 ReqMeta，完成 context association 和 physical binding。
6. Controller 读取 object state、ProfileDB、deadline 和 tier capability，生成 advisory decision。
7. Gate 校验 generation、refs、pin、pending I/O、budget 和 breaker。
8. 若动作为 prefetch/offload/drop/evict，Action Service 调用 LMCache backend；若为 load，注册 request-owned target，由原生 connector 在请求生命周期中完成。
9. Native callback 写出 load start/completion、compute progress 和 request finished。
10. Request accounting 汇总 lookup、admitted、loaded、recomputed 和 terminal reason。
11. Benchmark 保存输出 token、TTFT、TPOT、吞吐和质量结果。
12. Validator 检查 pair、hash、binding、command/receipt 和 accounting，给出 eligibility。

> **【待制图：图 4-5 请求级时序图】**  
> 参与者：Client、Context Host、Scheduler、Connector、Controller、Gate、Action Service、CPU/SSD Backend、Artifact Validator。  
> 需要同时画 demand load、prefetch hit 和 recompute fallback 三条分支。

## 4.8 运行模式

AstraKV 提供 off、shadow 和 active 三种模式。

**off 模式** 关闭策略执行，用于原生 baseline。必要的 benchmark 和环境产物仍可记录，但不产生 AstraKV active command。

**shadow 模式** 运行身份、观测、Profile 和决策链，只记录“如果 active 会做什么”，不修改 backend。E1 使用 shadow 验证生命周期是否完整。

**active 模式** 允许通过 gate 的命令进入 action service。active 并不是单个开关，它还依赖 execution enabled、offline gate、backend capability、binding、breaker 和预算。

这种逐级模式避免在身份链尚不可靠时直接修改真实缓存，也使策略可以在同一 workload 上先观察、再执行。

## 4.9 E0-E4 递进实验架构

系统把功能验收划分为 E0-E4，而不是一次打开全部能力。

| 阶段 | 配置 | 需要回答的问题 | 当前文档处理 |
| --- | --- | --- | --- |
| E0 | off | 原生路径的性能、正确性和资源基线是什么 | 【待实验】 |
| E1 | shadow | 七回调、context、binding、receipt 和 accounting 是否闭合 | 【待实验】 |
| E2 | active admission | external-prefix admission、native demand load 和 recompute 是否正确 | 【待实验】 |
| E3 | E2 + CPU prefetch | SSD->CPU prefetch 是否形成真实 lead time 和消费命中 | 【待实验】 |
| E4 | E3 + partial prefix | 部分前缀 load 与 suffix recompute 是否完整核算 | 【待实验】 |

E1 是生命周期门禁，不是性能 sweep。只有 E1 完整后，E2-E4 才有资格讨论容量和时延。E3 相对 E2，而不是相对 E0，隔离 prefetch 与 external admission 的作用；E4 相对 E3，隔离 partial-prefix 的作用。

## 4.10 OS 虚拟内存支线

OS VM 支线以 `MMapKVCache` 为核心。配置定义后备文件、block count、block size 和映射模式；运行时支持 block read/write、批量 prefetch、批量 evict hint 和 resident ratio。

`DgxSparkKVAdapter` 把 `KVChunkMeta` 映射为 mmap block 范围，记录 chunk write、read、prefetch、evict 和 residency。证据脚本生成 chunk records、动作事件、cold/warm latency 和驻留率。

这条支线回答“操作系统页缓存能否表达 KV 对象的按需驻留”，不回答“vLLM 已经使用这套映射”。将来若要接入真实 paged KV，需要重新处理 tensor layout、pinning、DMA、GPU page access 和 allocator ownership。

> **【待制图：图 4-6 OS VM PoC 结构图】**  
> 左侧为 KVChunkMeta 与 block range，中间为 sparse backing file + mmap，右侧为 first-touch、MADV_WILLNEED、MADV_DONTNEED、mincore。  
> 图上显式标注“独立机制 PoC，非 vLLM internal KV hook”。

## 4.11 参数与 MoE 扩展支线

参数侧包含 layer transfer 和 Qwen3 profile。layer transfer 测量层在 CPU/GPU 之间移动时的时间和显存变化；Qwen3 profile 比较 baseline、full-cache、partial-prefix+recompute，并计算 hidden-state CKA/cosine/L2/max-abs、teacher-forced loss/PPL 和逐层 KV 置零敏感性。

MoE 侧从 router log 或 `output_router_logits=True` 的 Hugging Face forward 中提取 route event，按 layer/expert 统计 activation 和 hotness。ExpertPredictor 根据历史和转移模型生成下一 token hint，ExpertLoaderPlanner 根据预算生成驻留建议。

两条支线复用了 AstraKV 的“对象 - 事件 - profile - planner - artifact”方法，但没有复用 KV-Core 的真实 action service。本文将它们作为系统可扩展性证据，不放入当前性能主张。

## 4.12 故障关闭与降级路径

系统预设以下失败是正常运行的一部分：external KV 不存在、CPU backend 没有空间、SSD I/O 超时、request cancel、binding stale、block allocation failure、patch mismatch、callback 缺失和 receipt 不完整。

对应处理为：

- identity 或 binding 失败：不创建 active command；
- prefetch 条件不足：记录 skip reason，继续 demand load/recompute；
- load 超时或不值得：交回原生 recompute；
- action service 失败：生成 terminal receipt，breaker 累计失败；
- breaker 打开：active 降级为 shadow/off；
- artifact 不一致：本轮 `eligible=false`，但保留原始结果用于诊断。

这种设计允许系统失败，却不允许失败被隐藏。对比赛报告而言，失败路径同样是系统能力的一部分，因为它决定了优化模块不会把 serving 结果变成不可解释状态。

---

# 5 关键技术与创新点

## 5.1 从性能数字转向可证明的运行时优化

AstraKV 最初也以 TTFT、P95、RSS 和磁盘 I/O 为主要结果。随着实验推进，团队发现仅凭端到端数字很难区分多个因素：请求是否被额外 warmup；vLLM 内置 prefix cache 是否命中；LMCache 是否只完成了 lookup 而没有 load；CPU staging 是否因为同步恢复而变热；输出变化是否来自 sampling。

当前版本的核心变化，是把“性能更低”拆成可以逐步证明的事实：目标请求已知、对象身份兼容、动作在消费前执行、真实 backend 状态改变、原生 connector 完成 load、请求实际消费、输出仍然正确。创新点不再集中在单个评分公式，而是集中在如何把这些事实组织成运行时控制系统。

## 5.2 精确 token、native key 与 binding generation 的联合身份

### 5.2.1 问题

文本相同不等于 token 相同，token 相同也不等于 KV 兼容；native key 相同在对象释放和重建后也可能对应新的物理生命周期。任何一个层次被忽略，都可能产生错误复用或 stale action。

### 5.2.2 设计

系统使用三层联合身份。

第一层是逻辑兼容：模型、revision、tokenizer、chat template、dtype、RoPE、layout 和 exact token prefix。第二层是缓存身份：LMCache 自己生成的 native key。第三层是物理生命周期：binding generation、engine/worker 和 runtime ReqMeta。

### 5.2.3 作用

这一结构把“前缀是否相同”“缓存对象是否相同”“当前物理实例是否仍然有效”分开判断。控制器不能用 prompt hash 越过 native key，也不能用旧 command 越过 generation。

### 5.2.4 边界

联合身份增加了 callback 和 association 的要求。任何层次缺失都会阻止 active，因此接入难度高于只按字符串查缓存，但它是安全复用的必要代价。

> **【待制图：图 5-1 三层身份与防错关系图】**  
> 展示逻辑兼容错误、native key 错误、ABA/stale generation 三种风险，以及对应防护字段。

## 5.3 ProfileDB 作为证据中间层

### 5.3.1 问题

策略若直接读取日志，很容易依赖某个版本的文本格式，也无法区分一次偶然命中和稳定复用。实验报告如果直接从多份日志拼数字，后续复核同样困难。

### 5.3.2 设计

ProfileDB 将 raw event 转换为 workload profile、chunk profile、quality guard 和 layer sensitivity。每个聚合字段都有事件来源，离线 profile 与在线 checkpoint 分开保存。

### 5.3.3 作用

同一 profile 可以驱动 scorer、load/recompute、object scheduler、offline simulator 和报告生成。策略输入、实验统计和解释文本来自同一中间层，减少口径漂移。

### 5.3.4 边界

ProfileDB 不自动保证数据正确。若 callback 与 request 没有关联，profile 只能降级为日志现象，不能用于 active。系统通过 quality/profile guard 和 experiment eligibility 限制这一点。

## 5.4 Load-vs-Recompute 与原生回退

### 5.4.1 问题

外部 KV 命中并不意味着必须加载。SSD 带宽不足、对象过大、deadline 太近时，重计算可能更快。反过来，长前缀的 prefill 很贵，即使加载需要 I/O，也可能值得。

### 5.4.2 设计

LoadRecomputePlanner 使用加载固定开销、对象 bytes、有效带宽、缺失 tokens 和单位 prefill 成本比较两条路径。决策还叠加 backend readiness、request status、memory pressure 和 partial plan。

### 5.4.3 作用

系统不把 external cache 变成必须命中的硬依赖。加载失败或不合算时，vLLM 原生 recompute 保证请求可完成；load receipt 又让实际加载量可以与重计算量核对。

### 5.4.4 边界

成本估计依赖目标 workload 和硬件。未经 bootstrap 或当前环境测量的带宽、prefill 单价只能作为保守先验，不能直接用于性能结论。

## 5.5 Request-owned 原生 KV 加载

### 5.5.1 问题

直接从控制器调用 retrieve 或写 slot mapping，会绕过 scheduler 的 block ownership。即使 tensor 内容看似正确，也可能与请求分配、取消和释放时机冲突。

### 5.5.2 设计

AstraKV 把 GPU KV writer 权限保留给原生 connector。控制器只能确定 external-prefix 上界并注册 load target；scheduler admission 决定为请求分配多少 external tokens；connector 在自己的生命周期中完成 load 并产生 native receipt。

### 5.5.3 作用

请求取消、slot 变化和 partial prefix 都由原生生命周期处理。AstraKV 不需要维护第二套 GPU block 状态，也不能伪造 load completion。

### 5.5.4 边界

generic load command 不直接完成 GPU load。若目标版本没有所需 scheduler/connector callback，系统只能 shadow 或 recompute。

## 5.6 可归因的 SSD 到 CPU 预取

### 5.6.1 问题

初赛 endpoint warmup 能缩短后续请求 TTFT，却提前执行了额外请求，无法说明内部 I/O 是否在目标请求等待期间被隐藏。只统计“prefetch command 成功”也无法知道对象最终是否被消费。

### 5.6.2 设计

当前预取满足四个条件：目标请求已经进入 ingress/queue；完整 token identity 和 native key 已知；SSD 中存在对象；动作发生在原生消费之前。PrefetchTicket 保存 deadline、TTL、generation、bytes 和 consumer。

### 5.6.3 作用

系统可以区分 hit、waste、late 和 cancel，并计算真实 lead time。E3 使用 E2 作为 baseline，只增加 CPU prefetch，使时延变化更接近预取本身的因果影响。

### 5.6.4 边界

预取只到 LocalCPUBackend。CPU-hot 不等于 GPU-ready，后续仍需原生 connector load。UMA 上 CPU prefetch 还可能增加总物理内存，因此必须与 resource sample 一起解释。

> **【待制图：图 5-2 warmup、demand load 与 native prefetch 对比时序图】**  
> 三行分别画 endpoint warmup、目标请求内同步 load、目标请求出现后的 SSD->CPU prefetch。  
> 对比额外请求、lead time、consumer 和 GPU load ownership。

## 5.7 连续对齐的 Partial Prefix 上界

### 5.7.1 问题

任意 token/layer 切片在理论上能提高灵活性，但真实 scheduler、slot mapping 和 LMCache chunk 都有自己的对齐要求。控制器若给出无法执行的切片，会造成“计划节省量”与“实际加载量”分离。

### 5.7.2 设计

当前 active 路径只允许从 token 0 开始、连续、block/chunk 对齐的 external-prefix upper bound。Lookup、admission、load 和 recompute 分别记录 token 数。

### 5.7.3 作用

限制后的 partial load 能直接映射到前缀复用语义，suffix 由原生 prefill 补齐。验收可以检查数量不变量，而不需要解释任意 tensor slice。

### 5.7.4 边界

离线 `PartialKVLoadPlanner` 可以表达更丰富的 layer/token span，但这些表达不会自动进入 active。论文应分别描述离线规划能力与真实执行能力。

## 5.8 证据驱动的执行开放

### 5.8.1 问题

在线策略的风险不只来自算法错误，还来自环境和生命周期不确定：版本不兼容、patch 未安装、对象已释放、后端压力过大、命令迟到、回执丢失。

### 5.8.2 设计

系统建立四道门。

1. Backend capability preflight 验证版本、patch 和动作能力。
2. Offline Safety Gate 验证策略在多个 workload 的 proxy 行为。
3. Runtime Execution Gate 验证单条命令的身份、状态和预算。
4. Circuit Breaker 在连续失败或压力异常时停止 active。

### 5.8.3 作用

策略建议与执行权分离。一个得分很高的 prefetch 仍可能因为 deadline、generation 或 CPU budget 被拒绝；拒绝原因进入 artifact，后续可以分析是策略不足还是环境不允许。

### 5.8.4 边界

门禁提高了系统安全性，也会降低早期实验的“成功率”。这不是性能优化本身，而是保证性能数字可以被信任的工程机制。

## 5.9 命令、回执与请求核算的双向闭合

### 5.9.1 问题

只有 command 没有 receipt，无法知道动作是否执行；只有 backend receipt 没有 request accounting，又无法知道动作是否被目标请求消费。

### 5.9.2 设计

动作服务为每条 command 生成 terminal receipt，原生 connector 为 load 生成 native receipt，request finished 时生成 accounting。Paired validator 检查 command/receipt 数量、binding generation、terminal state 和 token 数量关系。

### 5.9.3 作用

证据从控制器的意图闭合到 backend 的动作，再闭合到请求的消费。任何缺口都会使实验失去资格，而不是由报告脚本估算补齐。

### 5.9.4 边界

回执只能证明记录到的动作。若版本 patch 没有覆盖真实 callback，系统必须先修复生命周期，不允许手工追加 JSON 通过验收。

## 5.10 面向 UMA 的多口径内存结论

### 5.10.1 问题

在 UMA 平台上，把 KV 从“GPU tier”移到“CPU tier”可能没有释放物理页，只改变逻辑管理者。单看 `gpu_memory_utilization` 或某个进程的 GPU 指标容易得出错误结论。

### 5.10.2 设计

资源证据同时采集 cgroup memory、process RSS、vLLM available KV blocks、LMCache CPU occupancy、disk occupancy/I/O 和边界启动结果。capacity sweep 与普通 TTFT pair 分开运行。

### 5.10.3 作用

报告可以分别回答三类问题：调度器 KV budget 是否减少、同 SLO 下可承载规模是否提高、系统总物理内存是否下降。三种结论不再互相替代。

### 5.10.4 边界

如果平台无法提供可靠 case-level GPU framebuffer memory，系统不使用兼容字段假装测量成功，而是转向 RSS、cgroup 和容量边界证据。

## 5.11 版本锁定、低侵入的第三方适配

### 5.11.1 问题

vLLM 和 LMCache 的内部生命周期与 API 会变化。完全不修改第三方代码无法获得所需 callback，大规模 fork 又难以维护和复现。

### 5.11.2 设计

项目选择外置、版本锁定补丁：明确固定 vLLM 0.23.0、LMCache 0.4.7、connector 名称、patch id、source hash 和 required callbacks。补丁只增加 hook 与适配，不改变模型 kernel 和注意力算法。

### 5.11.3 作用

实验 manifest 能证明运行的是哪一份 adapter；版本升级时必须重新验证，而不是默认兼容。补丁目录与安装环境分开，便于审计和恢复。

### 5.11.4 边界

这条路线仍然存在版本耦合，不能描述为通用任意版本支持。当前论文应明确目标版本，未来适配通过 capability interface 扩展。

## 5.12 OS VM 机制与真实 Runtime 的边界协同

### 5.12.1 问题

赛题强调虚拟内存，但直接把真实 GPU KV 改造成页映射对象风险很高；完全停留在概念分析又无法证明机制可用。

### 5.12.2 设计

项目保留一条独立 VM PoC：使用稀疏文件、mmap、first-touch、madvise 和 mincore 验证逻辑空间、物理驻留、预取与回收。真实 runtime 则先通过 LMCache CPU/SSD 完成分层动作。

### 5.12.3 作用

两条路线分别回答“OS 机制是否可执行”和“真实 serving 对象如何安全管理”。它们共享 chunk metadata、event 和 evidence 口径，为后续结合留下接口。

### 5.12.4 边界

当前没有把 mmap 页直接交给 vLLM attention，也没有在 GPU fault path 上实现 userfaultfd。VM 结果只能作为机制证据，不能作为真实 KV-Core 内存收益。

## 5.13 从 KV 到参数和专家的可扩展对象方法

AstraKV 的对象方法可以概括为五个步骤：定义 identity、采集 event、构建 profile、生成 plan、记录 artifact。KV-Core 已把这五步推进到 runtime action；参数和 MoE 目前完成前三到四步。

这种扩展不是把所有对象强行塞进一个执行器。参数层、KV chunk 和 expert weight 的所有权、大小和访问时限不同，需要不同 backend adapter。系统统一的是证据和决策接口，而不是数据移动实现。

## 5.14 创新点总结

当前 AstraKV 的创新主要体现在系统组合和事实边界，而不是宣称一个单独算法已经获得确定收益。

1. 以 exact token、native key 和 generation 建立可执行的 KV 联合身份。
2. 以 ProfileDB 连接原始事件、离线策略、在线决策和报告证据。
3. 把 load 与 recompute 作为同一恢复问题，并始终保留原生回退。
4. 保持原生 connector 对 GPU KV 的唯一写入权，以 request-owned target 表达加载意图。
5. 用 PrefetchTicket 定义目标请求相关的 SSD->CPU 预取和真实 consumer。
6. 用连续、对齐的 partial-prefix 上界连接离线计划与原生 admission。
7. 通过 capability、offline gate、runtime gate 和 breaker 逐级开放 active。
8. 通过 command receipt、native receipt 和 request accounting 建立双向核算。
9. 在 UMA 上分离逻辑 KV capacity 与总物理内存结论。
10. 用独立 OS VM PoC 与真实 LMCache 路径分工验证机制和 runtime。

这些设计能否转化为容量和性能收益，仍需要第 8 章规划的 E0-E4 实验回答。当前章节只说明系统已经实现的机制及其产生原因。

---

# 6 核心模块实现

## 6.1 仓库实现结构

项目代码按对象、运行时、策略、实验和证据组织，而不是按一次性脚本堆放。

| 目录 | 主要内容 | 在系统中的位置 |
| --- | --- | --- |
| `astrakv/kv_cache/` | KV 元数据、block table、partial load | 对象与计划基础 |
| `astrakv/runtime/` | runtime core、context、binding、controller、gate、LMCache patch | 在线控制主线 |
| `astrakv/prefetch/` | chunk scorer | 对象价值评估 |
| `astrakv/scheduler/` | load/recompute 与 object scheduler | 统一策略规划 |
| `astrakv/vm/` | mmap KV、DGX adapter、layer offload、userfaultfd 原型 | OS VM 与参数 PoC |
| `astrakv/moe/` | expert predictor、expert loader planner | MoE 离线扩展 |
| `astrakv/benchmarks/` | runtime artifact、paired run 与 benchmark 数据结构 | 实验与证据 |
| `scripts/entrypoints/` | E0-E4、capacity 和 competition suite | 统一入口 |
| `scripts/reporting/` | acceptance、comparison、evidence report | 结果验收 |
| `third_party_patches/` | 版本锁定 vLLM/LMCache 补丁清单 | 原生 callback 适配 |
| `tests/` | 对象、策略、运行时、报告与边界测试 | 回归验证 |

## 6.2 KV 对象与 Partial Load

`metadata.py` 定义 `KVChunkMeta` 和 `MemoryTier`。`block_table.py` 保存 request、chunk、block 的映射关系，但不拥有实际 tensor。这样，元数据操作可以在没有 GPU 的环境中测试，也不会与 vLLM allocator 形成第二份状态。

`partial_load.py` 根据 layer/token span 生成 full、partial 和 skip 计划，并估算 loaded bytes 与 saved bytes。离线 planner 可用于研究层敏感度和不同前缀长度；进入 runtime 时，controller 通过 `_extract_partial_load_target` 把计划收紧为连续、从零开始和对齐的 target。

## 6.3 Runtime Core

`kv_runtime_core.py` 是 KV-Core 的数据契约中心。`RuntimeMode` 定义 off、shadow、active；`TierTopology` 区分 gpu-ssd 与 gpu-cpu-ssd；`KVCompatibilityKey`、`RequestKVBinding`、`PhysicalKVObject`、`TierCapabilitySnapshot` 和 `RequestKVIntent` 描述对象和请求；`PrefetchTicket` 与 `NativeKVLoadReceipt` 描述动作结果；`RequestKVAccounting` 封闭请求级核算。

该文件还提供一个轻量 load-vs-recompute 选择函数，供 runtime 层在不依赖完整离线 planner 时作保守判断。所有构造函数对空字符串、负 token 数、无效 tier 和不满足不变量的 receipt 进行校验，使错误尽量在对象创建阶段暴露。

## 6.4 Profile 与策略实现

`profile_db.py` 从 trace event 推断 workload id 和 chunk key，增量更新 workload/chunk profile，并可从 JSONL 重建数据库。质量 guard 与 layer sensitivity 使用独立 key，避免把性能统计与质量结论混在同一字段。

`scorer.py` 的 `ChunkScorer` 输出 `ChunkScore`，其中包含总分、各分项和 action。评分结果可以由 `explain_action` 转换为报告说明。需要再次说明，scorer 的 docstring 明确表示它不提交 prefetch，也不移动内存。

`decision.py` 的 `LoadRecomputePlanner` 输出 load、recompute、defer、drop 和 priority。`object_scheduler.py` 的 `UnifiedObjectScheduler` 将 candidate 与 GPU budget 合并，避免各 planner 分别占用同一预算。

`offline_eviction.py` 建立三层容量、对象访问和 prefetch hint 模型；`offline_safety.py` 聚合不同 workload 的策略表现，生成是否允许 runtime adapter 的 gate 结果。

## 6.5 Request Context 与 Binding

`request_context.py` 支持认证 loopback HTTP client、receiver、receipt 和 JSONL artifact。MAC payload 包含 session、run、request、nonce 和上下文摘要，receiver 拒绝重放、错误会话、非 loopback 目标和认证失败。

`backend_binding_registry.py` 为每个物理 binding 保存 generation、active ref、pin、pending I/O 和 action reservation。registry 的职责不是猜测对象在哪，而是把 callback 观察到的真实状态转换为可校验快照。

## 6.6 Backend Hook 与 Execution Gate

`backend_hook.py` 定义 `BackendExecutionSpec`、`BackendObjectBinding`、`BackendHookEvent`、`BackendActionCommand` 和 `BackendActionReceipt`。这些结构是 controller、action service 和 artifact 之间的 wire contract。

`runtime_execution_gate.py` 的 `ExecutionBudget` 定义单命令大小、窗口字节、命令率和并发上限。Gate 读取 backend capability、binding snapshot 和当前时间，返回 allow/deny 与具体 reason。Controller 不通过异常捕获绕开 deny，而是把 deny 当作正式决策结果。

`circuit_breaker.py` 使用 failure、timeout、pressure 与 cooldown 维护健康状态。其目标不是自动修复 LMCache，而是在异常持续时阻止控制面进一步扩大影响。

## 6.7 LMCache 0.4.7 动作适配

`lmcache047_runtime_patch.py` 同时包含 storage contract probe、action endpoint、request context consumer、manager proxy、connector lifecycle 和 hook 安装逻辑。

Action endpoint 在动作前检查 LocalCPUBackend、LocalDiskBackend 和 manager API 是否存在。prefetch 通过 disk `batched_get` 获得对象，再写入 CPU backend，并核对 CPU resident；offload 先确认 disk 副本，再删除 CPU 副本；evict 删除 disk 副本；drop 按 location 删除；load target 保存 runtime ReqMeta、token mask、slot mapping 与预计 bytes，等待原生 connector 消费。

对 partial prefix，适配层构造 load token mask，并从真实 slot mapping 和 KV cache 估算实际 bytes。它不接受任意非连续 token mask 作为 AstraKV active 结果。

`lmcache047_action_service.py` 把 endpoint 包装成受保护服务。服务校验 runtime probe challenge、HMAC proof、command digest 和 session，写入命令与回执 ledger，并通过 Unix domain socket 与 host 通信。

## 6.8 版本锁定补丁

`third_party_patches/vllm-0.23.0-lmcache-0.4.7/manifest.json` 声明 patch id、目标版本和七个 required callback。部署清单进一步保存目标 site-packages 文件路径和 SHA-256。

运行前 `verify_kv_core_connector_patch.py` 比较 deployment manifest 与 callback smoke。E2-E4 入口要求 manifest 与 smoke 文件存在；校验失败时脚本拒绝 active。诊断参数允许保留不合格运行，但不能把它转换为 eligible 结论。

## 6.9 RuntimeControlHost 与 Online Controller

`runtime_control_host.py` 负责 host config、online policy task、版本测量、context ingress、bridge registration 和 artifact lifecycle。Host 只在 EngineCore 进程范围拥有一份主要控制状态。

`online_controller.py` 根据当前 tier 分派候选动作。GPU resident 对象可 keep/drop/defer；CPU resident 对象可 keep/offload/drop/defer；SSD resident 对象可 load/prefetch/evict/drop/recompute/defer。Controller 还计算 prefetch window、prediction readiness、profile guard、load worthiness 和 live dispatch skip reason。

动作集合看起来较多，但真正 dispatch 的范围受 mode 和 capability 限制。keep、recompute、defer 主要表达计划；generic load 需要 native connector；prefetch、offload、drop、evict 才会在满足条件时进入 bridge。

## 6.10 Artifact 与 Reporting

Artifact 模块按照 run/pair 保存原始数据和派生产物。原始 callback、command、receipt 和 request result 不被报告脚本覆盖；派生 comparison 和 summary 保存输入 hash。

验收脚本不只计算均值，还检查：

- workload、环境和模型配置是否一致；
- baseline/variant 是否属于同一 pair；
- request id、binding 和 generation 是否闭合；
- command 是否存在唯一 terminal receipt；
- native load 与 request accounting 是否满足数量关系；
- cold/warm store 是否按 manifest 构造；
- 质量和输出是否满足实验要求。

## 6.11 OS VM、参数与 MoE 实现

`mmap_kv_cache.py` 在 Linux 上封装 `mmap`、`madvise` 和 `mincore`，支持块读写、预取、驱逐提示和统计。`dgx_spark_adapter.py` 维护 chunk 到 block range 的映射。`run_dgx_spark_vm_evidence.py` 生成独立证据包。

`layer_offload.py` 管理层对象在 CPU/GPU 间移动和显存测量。其 window 方法没有调用完整 Transformer forward，因此实验只能报告传输和驻留，不报告端到端推理收益。

`moe_events.py` 解析 router 日志和 JSONL；`expert_predictor.py` 建立 observation、transition model 和 prediction；`expert_loader.py` 根据 catalog、profile 和 budget 输出计划。三个模块都能写出 CSV/JSONL，便于在不接入 serving 的情况下评估预测覆盖和浪费。

## 6.12 测试结构

测试覆盖对象不变量、Profile 聚合、评分、load/recompute、offline simulator、gate、breaker、context 认证、binding generation、LMCache endpoint、action service、RuntimeControlHost、paired acceptance、VM 和 MoE 工具。

正式报告不应仅给出“测试数量”。测试章节需要区分纯 Python 单元测试、Linux VM 测试、LMCache adapter 测试、DGX integration 和 E0-E4 acceptance。当前稿件的测试数值留在第 8 章占位，待固定提交版本后重新运行。

---

# 7 系统运行流程与安全机制

## 7.1 启动前检查

系统启动前读取目标配置和 deployment manifest，确认模型、revision、dtype、vLLM、LMCache、connector、worker 数量和 topology。随后执行 patch verification 和 backend capability preflight。若版本或 callback smoke 不满足，runtime 保持 off/shadow。

每次实验创建新的 run id、输出目录、LMCache store 和认证 secret。baseline 与 variant 使用相同逻辑配置但分离存储目录，防止一侧写入污染另一侧。cold store 为空，warm store 必须由固定数据播种。

## 7.2 请求处理状态机

一个请求可以抽象为以下状态：

```text
CREATED
  -> CONTEXT_RECORDED
  -> NATIVE_ASSOCIATED
  -> LOOKED_UP
  -> ADMITTED
  -> LOADING / RECOMPUTING
  -> CONSUMED
  -> FINISHED / CANCELLED / FAILED
```

预取对象还具有：

```text
ISSUED -> IN_FLIGHT -> READY -> HIT
                       |       -> WASTE
                       -> LATE / CANCELLED / FAILED
```

状态不能通过报告脚本反推补齐。缺少 native association 的请求不能直接进入 ADMITTED；缺少 load completion 的对象不能进入 CONSUMED；没有 request finished 的核算不能视为完整样本。

> **【待制图：图 7-1 请求与预取双状态机】**  
> 上半部分画 request lifecycle，下半部分画 PrefetchTicket lifecycle，使用 consumer edge 连接 READY 与 CONSUMED。

## 7.3 并发与对象冲突

同一 physical object 可能被多个请求复用。Active refs 防止对象在消费期间被删除，pin 防止策略驱逐必须保留的对象，pending I/O 防止加载和换出交叉，reservation 防止同一 generation 并发执行冲突命令。

请求结束后，registry 更新 refs，并由策略决定对象继续 keep、offload 或 drop。对象重建时 generation 递增，旧 reservation 和迟到 receipt 不再匹配。

## 7.4 资源保护

ExecutionBudget 同时限制单对象和时间窗口。单对象限制避免一次命令搬运过大；窗口 bytes 限制避免短时间累计 I/O；命令率限制避免 controller 抖动；并发限制保护 LMCache staging。

CPU prefetch budget 在主配置中以显式比例表示。真实运行还要从 backend capability 和 resource sample 确认 LocalCPUBackend 可用容量，不能只根据配置声明。

## 7.5 正确性保护

正确性保护分为身份、数量和输出三部分。身份要求 compatibility、native key 和 generation 一致；数量要求 lookup、admission、load 和 recompute 闭合；输出要求 finish reason、token IDs 和必要的 loss/logprob/质量指标与 baseline 可比较。

Forced hit、partial hit、SSD miss、CPU miss、I/O timeout、request cancel、stale generation 和 block allocation failure 都应纳入用例。优化路径不能只测试成功命中。

## 7.6 降级与恢复

当 backend 初始化失败、staging 耗尽、disk load stall 或 breaker 打开时，系统停止 active dispatch。对尚未开始的动作记录 skip；对已经发出的动作等待 terminal receipt 或标记 unknown；请求数据面回退原生 recompute。

恢复需要新一轮 capability/health 证据。系统不会因为经过固定时间就假定后端恢复，也不会修改旧实验目录掩盖失败。

---

# 8 实验设计、验证方法与结果占位

## 8.1 实验目标

实验不以“跑出更低数字”为唯一目标，而是依次验证五个问题。

1. 原生 baseline 是否稳定、正确、可复现。
2. AstraKV 是否观察到完整请求与 KV 生命周期。
3. External admission、demand load 和 recompute 是否正确核算。
4. SSD->CPU prefetch 是否在目标请求消费前完成，并减少同步等待。
5. Partial prefix 是否在不破坏输出的前提下减少加载量或重计算量。

## 8.2 实验环境

| 项目 | 固定配置 |
| --- | --- |
| 硬件 | DGX Spark GB10，UMA |
| 模型 | Qwen3-8B，本地 revision，非量化 |
| dtype | bfloat16 |
| vLLM | 0.23.0 |
| LMCache | 0.4.7 |
| worker | 单 worker |
| max model length | 32768 |
| block size | 16 tokens |
| 默认 gpu memory utilization | 0.72 |

> **【待实验：表 8-1 完整软硬件环境】**  
> 待填字段：OS、kernel、CUDA、driver、Python、PyTorch、Transformers、磁盘型号与文件系统、可用内存、提交 hash、patch id、模型目录 hash。

## 8.3 工作负载

### 8.3.1 Repeated Long Prefix

固定一组长前缀 seed 与 revisit 请求，用于测量 exact reuse、native demand load 和 prefetch window。请求顺序、输出长度、采样参数和 prefix token 数固定。

### 8.3.2 Random No Reuse

构造 token 前缀互不复用的请求，测量 AstraKV 的观测、context 和策略开销，并检查 active 是否错误产生 prefetch。

### 8.3.3 Constrained KV Churn

在受限 KV budget 下交替访问多个对象，制造 keep/offload/drop/evict 冲突，用于容量和替换策略测试。

### 8.3.4 Queued Concurrency

通过并发和队列形成可测 lead time，验证目标请求已经出现但尚未消费 KV 时，SSD->CPU prefetch 能否与排队重叠。

> **【待实验：表 8-2 工作负载清单】**  
> 字段：workload id、请求数、context、shared prefix tokens、output tokens、batch/concurrency、seed、cache state、预期复用关系、输入 SHA-256。

## 8.4 E0：原生基线

E0 使用 off 模式，不启用 external admission、CPU prefetch 和 partial prefix。需要报告请求成功率、TTFT、TPOT、throughput、RSS/cgroup、KV block capacity 和输出正确性。

> **【待实验：表 8-3 E0 基线结果】**  
> **【待制图：图 8-1 E0 各 workload 的 TTFT p50/p95 与吞吐】**  
> **结论占位：** 只回答基线是否稳定、方差是否可接受、是否存在 OOM 或输出异常。

## 8.5 E1：Shadow 生命周期门禁

E1 不执行 active 动作。每个正式 native request 必须观察到七类 callback，并完成 context association、binding、native receipt 和 request accounting。

验收条件：

- required callbacks 与 observed callbacks 完全一致；
- deployment manifest compatible；
- control environment hash 一致；
- logical request 与 native ReqMeta 已关联；
- request accounting 非空；
- terminal receipt 数量与请求一致；
- 不通过手工 JSON 或日志估算补齐。

> **【待实验：表 8-4 E1 回调覆盖与身份闭合】**  
> 字段：run、request count、7 callback counts、associated count、binding count、receipt count、accounting count、errors、eligible。  
> **【待制图：图 8-2 E1 请求生命周期桑基图或时序覆盖图】**  
> **结论占位：** E1 只有在 `eligible=true` 后才能写“生命周期闭合”。

## 8.6 E2：External Admission 与 Demand Load

E2 在 gpu-ssd topology 下启用 request-scoped external-prefix admission，不启用 CPU prefetch。目标是验证 scheduler exact lookup、external admission、native demand load 和 suffix recompute。

关键不变量：

$$
N_{lookup}\ge N_{admitted}\ge N_{loaded},
$$

且请求前缀由 loaded 与 recomputed 或 terminal reason 完整解释。

> **【待实验：表 8-5 E2 请求级 token accounting】**  
> 字段：request id、lookup、admitted、loaded、recomputed、cancelled、load ms、prefill ms、native key、generation。  
> **【待实验：表 8-6 E0/E2 性能与容量对照】**  
> **【待制图：图 8-3 E2 loaded/recomputed token 堆叠图】**  
> **结论占位：** 回答 demand load 是否正确、miss 是否回退、容量是否变化；未达到 eligibility 时不比较收益。

## 8.7 E3：SSD 到 CPU 预取

E3 在 E2 基础上切换 gpu-cpu-ssd topology，并启用 CPU prefetch。所有请求、workload 和外部准入条件与 E2 保持一致，唯一新增变量为 prefetch。

每个 ticket 统计 issue time、ready time、consume time、lead time、bytes、hit/waste/late/cancel。性能主比较为 E3 相对 E2，而非 E3 相对 E0。

预设验收目标来自主配置：TTFT p95 改善至少 5%，吞吐回退不超过 2%，无复用 TTFT p95 回退不超过 2%，UMA 峰值不恶化超过 2%。这些数值是目标门槛，不是当前结果。

> **【待实验：表 8-7 E3 PrefetchTicket 汇总】**  
> 字段：issued、ready、hit、waste、late、cancel、hit rate、waste rate、median/p95 lead time、bytes。  
> **【待实验：表 8-8 E2/E3 Paired Performance】**  
> 字段：workload、cache state、N、TTFT p50/p95、bootstrap CI、TPOT、throughput、RSS/cgroup、CPU occupancy、disk read bytes。  
> **【待制图：图 8-4 Prefetch lead time 与 load latency 分布】**  
> **【待制图：图 8-5 E2/E3 TTFT p95 配对差值与置信区间】**  
> **结论占位：** 只有 ticket 被真实 consumer 消费且 pair eligible，才讨论 I/O 隐藏。

## 8.8 E4：Partial Prefix

E4 在 E3 基础上启用 continuous-prefix-from-zero、block-aligned partial upper bound。比较 E4 与 E3，隔离 partial prefix 的作用。

需要验证不同 external token cap 下的 loaded bytes、suffix recompute、TTFT、CPU/SSD I/O 和输出一致性。不能只报告 planner 估算节省量。

> **【待实验：表 8-9 E4 Partial Prefix Accounting】**  
> **【待制图：图 8-6 External Prefix 长度与 load/recompute 成本曲线】**  
> **【待制图：图 8-7 E3/E4 I/O 字节和 TTFT 对照】**  
> **结论占位：** 回答 partial prefix 是否在真实 receipt 中减少加载量，以及 suffix 是否由原生计算完成。

## 8.9 Capacity Sweep

容量实验独立于普通 TTFT pair。固定 workload 和 SLO，逐步降低 `gpu_memory_utilization` 或 KV block budget，观察成功率、最大 context/concurrency、TTFT p95、throughput 和 UMA peak。

主配置目标为同 SLO 下 KV capacity 至少改善 10%，可以表现为更低 block budget或更大 context/concurrency。若只有 vLLM block budget 变化而 UMA 总峰值不降，结论应写“KV capacity 改善”，不能写“物理内存降低”。

> **【待实验：表 8-10 Capacity Sweep】**  
> 字段：phase、gpu memory utilization、available KV blocks、max context、max concurrency、success、TTFT p95、throughput、RSS/cgroup、LMCache occupancy。  
> **【待制图：图 8-8 SLO 下最大可承载 context/concurrency】**  
> **【待制图：图 8-9 vLLM KV capacity 与 UMA peak 双轴图】**

## 8.10 正确性与质量

正确性测试覆盖 forced hit、partial hit、SSD miss、CPU miss、timeout、cancel、stale generation 和 allocation failure。请求输出至少比较 token IDs、finish reason 和长度；固定 teacher-forced 路径可比较 loss、PPL 和 hidden-state 指标。

> **【待实验：表 8-11 请求正确性与故障注入】**  
> **【待实验：表 8-12 Baseline/Variant 输出与质量指标】**  
> **结论占位：** 区分确定性 token 一致性、非确定 sampling 下的质量等价和故障回退正确性。

## 8.11 OS VM PoC 实验

VM 实验固定 backing file、block count、block size 和访问序列，依次执行 write、evict hint、cold read、prefetch、warm read，并用 mincore 记录 resident ratio。

> **【待实验：表 8-13 mmap VM 参数与结果】**  
> 字段：platform、kernel、filesystem、block count/size、evicted blocks、prefetched blocks、resident ratio、cold/warm latency、RSS、read bytes。  
> **【待制图：图 8-10 VM block 驻留热图】**  
> **【待制图：图 8-11 cold/warm read latency 分布】**  
> **结论占位：** 只描述 OS VM 机制，不外推为 vLLM KV-Core 收益。

## 8.12 MoE 离线分析实验

MoE 实验记录模型、prompt、token 数、layer、expert 数、top-k、route events、prediction coverage、waste 和 planner actions。

> **【待实验：表 8-14 MoE Route 与 Prediction】**  
> **【待制图：图 8-12 Layer-Expert 激活热图】**  
> **【待制图：图 8-13 Expert Prediction Coverage/Waste】**  
> **结论占位：** 只回答离线路由可观测性和预测质量，不写 online expert movement。

## 8.13 统计方法

每组性能实验需要报告样本量、重复次数、冷/热状态、p50/p95/p99、均值、标准差或四分位数。配对结果优先报告 per-request delta 和 bootstrap 95% confidence interval。若置信区间跨零，应写“未观察到稳定改善”，而不是只引用均值。

异常样本不能在不知道原因时删除。超时、取消和 OOM 单独统计；环境或证据不合格使整轮 ineligible，而不是从样本中剔除后继续计算。

## 8.14 实验资格总表

> **【待实验：表 8-15 最终实验资格矩阵】**

| Phase | Workload | Cache State | Correctness | Identity | Callback | Binding | Receipt | Accounting | Pair Hash | Eligible | Claim Scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 【待填】 | 【待填】 | 【待填】 | N/A | N/A | N/A | N/A | N/A | 【待填】 | 【待填】 | baseline |
| E1 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | lifecycle |
| E2 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | admission/load |
| E3 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | prefetch |
| E4 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | partial prefix |

---

# 9 工程组织、使用方法与复现路径

## 9.1 配置入口

目标配置位于 `configs/astrakv_kv_core_qwen3_8b.yaml`。配置文件固定目标版本、硬件、mode、topology、CPU prefetch budget、block size、partial load 约束、workload 和 acceptance threshold。

配置中的 acceptance 是实验门槛，不是默认结果。正式报告填数时应保存实际生效配置，而不是只引用仓库默认值。

## 9.2 Controlled Suite

`scripts/entrypoints/run_kv_core_controlled_suite.sh` 是 E0-E4 主入口。脚本校验 workload、phase、cache state、loopback host、offline profile 和 active deployment evidence；为每个 pair 创建独立 LMCache store；启动 server；运行 benchmark；收集 state artifact；调用 acceptance validator。

推荐复现顺序为：

```text
准备 canonical workload
  -> 固定模型和环境
  -> 运行 E1 小规模 lifecycle smoke
  -> 验证 patch manifest + callback smoke
  -> 运行 E2
  -> 运行 E3
  -> 运行 E4
  -> 独立运行 capacity sweep
  -> 汇总 eligible artifacts
```

不能在 E1 不合格时跳到 E2，也不能使用诊断选项把 ineligible 结果升级为性能结论。

## 9.3 产物归档

每轮实验至少保留：workload、环境清单、配置、server log、request results、callback smoke、native callbacks、native receipts、binding events、commands、command receipts、request accounting、resource samples、acceptance 和输入 hash。

报告图表应从归档 artifact 生成，不手工编辑原始 JSONL。失败运行同样保留，新实验使用新 run id 和新目录。

## 9.4 本地与 DGX 分工

本地环境适合运行纯 Python 单元测试、静态配置检查、报告工具和离线策略；Linux/DGX 环境负责 CUDA、vLLM、LMCache、Unix socket、mmap/mincore 和真实 benchmark。

两侧通过明确的同步脚本和提交 hash 对齐。远端 site-packages 补丁必须由仓库中的版本锁定 patch 安装，不在服务器上手工修改后直接跑结果。

---

# 10 当前边界、风险与后续工作

## 10.1 当前边界

当前代码已经覆盖 KV-Core 的对象、控制和证据结构，但最终实验数据尚未写入本文。OS VM 仍是独立 PoC，参数 offload 仍是传输骨架，MoE 仍是离线分析。报告在实验完成前不使用“已经降低物理内存”“已经隐藏 I/O”或“已经获得性能提升”等表述。

## 10.2 版本耦合风险

当前 callback 与 action endpoint 面向固定 vLLM/LMCache 版本。升级可能改变 scheduler connector、ReqMeta、token database、manager API 和 load lifecycle。后续需要把版本适配拆成独立 adapter，并为每个版本维护 capability test 和 deployment manifest。

## 10.3 UMA 指标风险

UMA 下缺少传统 case-level framebuffer 指标，RSS 又可能包含共享页和缓存页。后续实验需要固定采样进程、cgroup 范围和采样频率，并把 vLLM block capacity、LMCache occupancy 与系统总内存同时报告。

## 10.4 预取压力风险

CPU prefetch 可能抢占 demand load 的 staging，错误预取也会增加 SSD I/O。后续应在 E3 中加入 no-reuse、queued concurrency 和 I/O contention，对 prefetch budget、并发数和 TTL 做消融。

## 10.5 Profile 漂移

不同 workload、模型和上下文长度的复用与加载成本不同。离线 profile 若长期不更新，可能产生错误动作。后续可以增加时间衰减、profile version、在线校准和 drift detector，但任何在线更新仍要经过 gate。

## 10.6 从 KV 向参数与专家扩展

参数扩展需要真实 forward、层加载与计算重叠、权重格式和 pinned memory 管理；MoE 扩展需要 serving router callback、expert tensor catalog、真实 load receipt 和 miss stall。两者应复用当前 identity/evidence 方法，但分别实现 backend adapter，不能直接套用 KV command。

## 10.7 后续工作顺序

后续工作按依赖关系推进。

1. 固定提交和部署环境，完成全量本地回归。
2. 完成 E1 七回调、context association 和 request accounting 验收。
3. 完成 E2 demand load/recompute 正确性与 capacity 基线。
4. 完成 E3 PrefetchTicket、lead time 和 E2/E3 配对性能。
5. 完成 E4 partial prefix 与 suffix recompute 核算。
6. 在同一版本上复跑 OS VM、32K、质量与 MoE 离线实验。
7. 填充第 8 章所有图表，并按 eligibility 收紧摘要和结论。

---

# 11 总结

AstraKV 的起点是内存受限场景中的一个实际矛盾：长上下文和并发需要更大的 KV cache，而资源受限设备无法把所有运行时对象长期保留在最高层。初赛阶段先从真实 endpoint、分层后端、外部预热、压力边界和 OS VM PoC 入手，证明这些行为能够被测量，也发现仅靠 endpoint 和日志无法回答对象是否真正被原生运行时消费。

当前版本据此把系统重心转向对象身份和证据。Exact token IDs、compatibility key、native key 和 binding generation 定义“对象是谁”；ProfileDB、scorer、load/recompute 和 object scheduler定义“应该做什么”；RuntimeControlHost、gate、breaker 和 LMCache action service 定义“什么条件下允许执行”；native receipt、request accounting 和 paired validator 定义“如何证明动作完成并产生可用结果”。

这条路线没有绕开 vLLM 和 LMCache 的所有权。GPU paged KV 仍由原生 connector 写入，AstraKV 只管理外部对象、策略和受控后端动作；预取只从 SSD 到 CPU；partial load 限定为连续对齐前缀；失败时保留 recompute。OS VM、参数和 MoE 模块则作为不同成熟度的支线，分别验证页缓存机制、层敏感性和稀疏专家行为。

本文当前完成了问题、理论、路线、架构、实现和实验方法的完整描述。实验数值、图表和最终性能结论将在 E0-E4 与 capacity/quality/VM/MoE 结果完成后填入。届时结论只使用 `eligible=true` 的产物，并分别报告 KV capacity、逻辑层级变化和总物理内存，避免把配置变化或预热现象写成未被证明的系统收益。

---

# 参考文献占位

正式提交前应按学校或赛事要求统一为 GB/T 7714 或指定格式。下列条目仅标记正文已经涉及的方向，不在当前稿件中编造作者、年份、会议或页码。

1. 【待补】vLLM 与 PagedAttention 原始论文及项目文档。
2. 【待补】LMCache 论文、项目文档与 vLLM connector 文档。
3. 【待补】FlexInfer：面向内存受限 LLM 推理的分层/异步加载工作。
4. 【待补】NEO：CPU/GPU 协同推理相关工作。
5. 【待补】IMPRESS、Cake、Bidaw：KV 管理、缓存替换或预取相关工作。
6. 【待补】ShadowKV、RocketKV：选择性 KV 或 KV 压缩相关工作。
7. 【待补】Jenga、KTransformers、Klotski：MoE/专家驻留与异构推理相关工作。
8. 【待补】Linux `mmap`、`madvise`、`mincore` 官方手册或内核文档。
9. 【待补】DGX Spark GB10 与统一内存架构官方资料。
10. 【待补】Qwen3 模型技术报告与 tokenizer/chat template 文档。

---

# 附录 A 图表清单

| 编号 | 图表名称 | 类型 | 数据来源 | 状态 |
| --- | --- | --- | --- | --- |
| 图 1-1 | 内存受限 LLM 推理问题 | 概念图 | 理论与平台结构 | 【待制图】 |
| 图 1-2 | AstraKV 两阶段演进 | 路线图 | 初赛与当前架构 | 【待制图】 |
| 图 2-1 | Prefill/Decode/KV 生命周期 | 时序图 | runtime 设计 | 【待制图】 |
| 图 3-1 | 问题驱动演进 | 因果图 | 初赛局限与新机制 | 【待制图】 |
| 图 4-1 | 总体系统架构 | 架构图 | 源码模块 | 【待制图】 |
| 图 4-2 | KV 联合身份 | 对象关系图 | runtime core | 【待制图】 |
| 图 4-3 | 策略形成过程 | 数据流图 | profile/scorer/planner | 【待制图】 |
| 图 4-4 | RuntimeControlHost 部署 | 进程图 | runtime host | 【待制图】 |
| 图 4-5 | 请求级端到端流程 | 时序图 | callback/action | 【待制图】 |
| 图 4-6 | OS VM PoC | 机制图 | mmap adapter | 【待制图】 |
| 图 5-1 | 三层身份防错 | 关系图 | identity/binding | 【待制图】 |
| 图 5-2 | Warmup、Load、Prefetch 对比 | 时序图 | prefetch contract | 【待制图】 |
| 图 7-1 | 请求与预取状态机 | 状态图 | runtime contract | 【待制图】 |
| 图 8-1 至 8-13 | 性能、容量、正确性、VM、MoE | 实验图 | E0-E4 artifacts | 【待实验后制图】 |

---

# 附录 B 代码与能力索引

| 能力 | 主要文件 |
| --- | --- |
| KV 元数据与 partial plan | `astrakv/kv_cache/metadata.py`、`block_table.py`、`partial_load.py` |
| Runtime 对象契约 | `astrakv/runtime/kv_runtime_core.py` |
| Request context | `astrakv/runtime/request_context.py` |
| Physical binding | `astrakv/runtime/backend_binding_registry.py` |
| Backend capability/hook | `astrakv/runtime/backend_capabilities.py`、`backend_hook.py` |
| ProfileDB | `astrakv/runtime/profile_db.py` |
| Chunk score | `astrakv/prefetch/scorer.py` |
| Load/recompute | `astrakv/scheduler/decision.py` |
| Unified scheduler | `astrakv/scheduler/object_scheduler.py` |
| Offline simulator/safety | `astrakv/runtime/offline_eviction.py`、`offline_safety.py` |
| Online controller | `astrakv/runtime/online_controller.py` |
| Execution gate/breaker | `astrakv/runtime/runtime_execution_gate.py`、`circuit_breaker.py` |
| Runtime host | `astrakv/runtime/runtime_control_host.py` |
| LMCache action | `astrakv/runtime/lmcache047_runtime_patch.py` |
| Protected action service | `astrakv/runtime/lmcache047_action_service.py` |
| Artifact/paired run | `astrakv/runtime/artifact_contract.py`、`astrakv/benchmarks/runtime_artifacts.py`、`paired_run.py` |
| mmap KV | `astrakv/vm/mmap_kv_cache.py`、`dgx_spark_adapter.py` |
| 参数 PoC | `astrakv/vm/layer_offload.py`、`scripts/research/build_qwen3_kv_core_profile.py` |
| MoE | `astrakv/runtime/moe_events.py`、`astrakv/moe/expert_predictor.py`、`expert_loader.py` |
| E0-E4 入口 | `scripts/entrypoints/run_kv_core_controlled_suite.sh` |
| 目标配置 | `configs/astrakv_kv_core_qwen3_8b.yaml` |
| 版本补丁 | `third_party_patches/vllm-0.23.0-lmcache-0.4.7/manifest.json` |

---

# 附录 C 实验结果填充检查表

- [ ] 固定最终提交 hash、patch id 和环境 manifest。
- [ ] E1 required callbacks 达到 7/7，且每类均来自真实 native request。
- [ ] Request context 从 recorded 进入 associated，未使用 prompt hash 替代。
- [ ] 每个 active command 存在唯一 terminal receipt。
- [ ] 每个 load request 满足 lookup/admitted/loaded 数量关系。
- [ ] Suffix recompute 或 terminal reason 能解释所有未加载 token。
- [ ] Baseline/variant workload、seed、请求顺序、输出长度和 cache state 一致。
- [ ] Pair manifest 为 `eligible=true`。
- [ ] TTFT/TPOT/throughput 报告样本量与置信区间。
- [ ] 无复用 workload 报告控制开销和错误预取率。
- [ ] Prefetch 报告 hit、waste、late、cancel、lead time 和 bytes。
- [ ] Capacity 结论与 UMA 物理内存结论分开。
- [ ] 正确性覆盖 forced hit、partial hit、miss、timeout、cancel、stale generation。
- [ ] VM 结果明确标注为独立 PoC。
- [ ] MoE 结果明确标注为离线 route/prediction/planner。
- [ ] 所有 `【待实验】`、`【待制图】` 和 `【待补】` 在正式提交前处理。
