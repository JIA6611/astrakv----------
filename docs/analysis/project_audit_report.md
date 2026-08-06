# AstraKV-W 项目审计与竞赛准备度分析

Status: project audit and contest readiness analysis.

Date: 2026-06-04

面向赛事：2026 年全国大学生计算机系统能力大赛，操作系统设计赛 OS 功能挑战赛道。

赛题：内存受限环境的大语言模型推理优化。

## 一句话结论

如果今天提交比赛，AstraKV-W 大约处于“方向正确、工程框架和真实基线初步打通，但核心 runtime 优化尚未落地”的早期原型水平，距离决赛级作品还差一个真实可演示的 vLLM/LMCache 多级内存优化闭环、可信对照实验和稳定复现体系。

## 术语边界

本文严格使用以下分类：

| 类型 | 含义 | 当前项目中的例子 |
| --- | --- | --- |
| 真实功能 | 调用真实模型、真实 endpoint、真实 GPU 或真实系统探针得到的功能 | `scripts/run_real_benchmark.py` 访问 vLLM OpenAI endpoint；`results/local_smoke_test/` 中的真实 GPU smoke baseline |
| Synthetic 功能 | 用 Python 或配置模拟推理时延、KV 访问、prefetch、GPU KV 容量等行为 | `scripts/benchmark_runner.py`、`SyntheticBackend`、`SelectivePrefetchBenchmarkBackend`、`prefetch/selective_kv.py` |
| Skeleton 功能 | 数据结构、接口、状态记录或边界设计，暂不执行真实数据移动或 runtime 控制 | `runtime/`、`kv_cache/`、`offload/tier_placement.py`、`scheduler/hints.py`、`prefetch/async_engine.py` 默认 noop adapter |
| Mock 功能 | 用占位 adapter、配置意图、文档流程或空实现代替真实后端行为 | `AsyncPrefetchEngine._noop_adapter()`、LMCache launch wrapper 中的 backend intent、尚未验证的 LMCache CPU/Disk 配置 |
| 已验证模块 | 仓库中已有可读结果或可执行产物证明跑通过 | synthetic benchmark 结果、selective prefetch MVP 结果、vLLM endpoint smoke baseline |
| 未验证模块 | 有代码、配置或文档，但没有结果证明真实链路跑通过 | LMCache CPU/Disk 真实 offload、AstraKV-W runtime adapter、真实 prefetch control、真实 scheduler hint 闭环 |

## 任务 1：项目现状分析

### 总体结构

当前仓库包含以下主要目录：

| 目录 | 当前定位 | 审计判断 |
| --- | --- | --- |
| `runtime/` | Runtime object manager 和 adapter protocol | Skeleton |
| `kv_cache/` | KV chunk metadata 与 block table | Skeleton |
| `prefetch/` | Async prefetch skeleton 与 Selective KV Prefetch MVP | Skeleton + Synthetic |
| `offload/` | Tier placement intent manager | Skeleton |
| `scheduler/` | Scheduler hint dataclass | Skeleton |
| `benchmarks/` | Synthetic benchmark 配置与说明 | Synthetic |
| `scripts/` | benchmark runner、真实 endpoint benchmark、plot、metrics、launch wrappers | 真实工具 + Synthetic 工具 + wrapper |
| `configs/` | DGX Spark、vLLM、LMCache CPU/Disk、policy simulator 配置 | 配置意图，部分未验证 |
| `docs/` | 设计分析、边界、复现、竞赛方案、真实基线说明 | 文档较完整 |
| `results/` | 已生成 benchmark 与 smoke test 结果 | 部分已验证 |
| `third_party/` | vLLM、LMCache、FlashAttention、llama.cpp、SGLang、TensorRT-LLM 源码参考 | 已克隆/分析，当前未修改、未集成 |

### 已实现模块

已实现但性质不同：

| 模块 | 文件 | 已实现内容 | 类型 |
| --- | --- | --- | --- |
| KV 元数据 | `kv_cache/metadata.py` | `MemoryTier`、`KVChunkMeta`、token span、block ids、tier、cache key 等字段 | Skeleton |
| KV block table | `kv_cache/block_table.py` | chunk 到 request/layer/token/block ids 的记录、查询、释放 | Skeleton |
| Runtime object manager | `runtime/object_manager.py` | 串联 KV metadata、block table、placement、prefetch request | Skeleton |
| Runtime adapter protocol | `runtime/adapters.py` | `RuntimeRequest` 与 `RuntimeAdapter` protocol | Skeleton |
| Tier placement | `offload/tier_placement.py` | 记录 current tier、target tier、planned/resident/failed 状态 | Skeleton |
| Async prefetch engine | `prefetch/async_engine.py` | prefetch request lifecycle 与 adapter callback | Skeleton/Mock |
| Selective prefetch MVP | `prefetch/selective_kv.py` | CPU/GPU 两级 KV 模拟、async queue、LRU、hit/waste/eviction metrics | Synthetic |
| Synthetic benchmark runner | `scripts/benchmark_runner.py` | synthetic baseline 与 selective prefetch 对比、CSV/MD/PNG 输出 | Synthetic |
| Metrics collector | `scripts/metrics_collector.py` | 当前进程 RSS、`nvidia-smi`、scratch IO 测量 | 工具，真实探针但多用于 synthetic runner |
| Real endpoint benchmark | `scripts/run_real_benchmark.py` | streaming chat completion、TTFT、TPOT、latency、tokens/s、GPU memory sample | 真实功能 |
| DGX metrics collector | `scripts/dgx_metrics_collector.py` | 后台采样 GPU memory/util、进程 RSS、Linux diskstats | 真实工具，未见结果验证 |
| Plot tool | `scripts/plot_benchmarks.py` | 从 CSV 生成 benchmark PNG 图 | 工具 |
| Launch wrappers | `scripts/launch_vllm_server.*`、`scripts/launch_lmcache_vllm.*` | 启动 vLLM 或设置 LMCache config intent | wrapper，LMCache 部分未验证 |

### Skeleton 模块

Skeleton 的核心特征是“记录元数据，不移动真实 KV，不控制真实 runtime”。

| 模块 | Skeleton 证据 | 当前风险 |
| --- | --- | --- |
| `runtime/` | 只维护 `RuntimeObjectManager`、adapter protocol；没有 vLLM/SGLang/TensorRT-LLM adapter 实现 | 不能宣称 runtime 优化 |
| `kv_cache/` | `KVChunkMeta` 和 `KVBlockTable` 不持有 tensor，不接 vLLM block pool | 不能宣称真实 KV cache 管理 |
| `offload/` | `TierPlacementManager` 只记录 plan/status，不执行 CPU/SSD/GPU copy | 不能宣称 offload 已实现 |
| `scheduler/` | `SchedulerHint` 是被动 dataclass，不调度请求 | 不能宣称 scheduler-aware optimization |
| `prefetch/async_engine.py` | 默认 adapter 返回 `"noop prefetch adapter"` | 不能把 noop 视为真实预取 |

### Mock 模块

| 模块 | Mock/占位点 | 说明 |
| --- | --- | --- |
| Async prefetch 默认 adapter | `AsyncPrefetchEngine._noop_adapter()` | request 提交后直接 completed，没有 IO、DMA、LMCache、vLLM KV transfer |
| LMCache launch wrapper | `launch_lmcache_vllm.*` | 主要设置 `LMCACHE_CONFIG_FILE` 并调用 vLLM wrapper；文件中明确提示要按 installed LMCache integration flag 调整 |
| LMCache example configs | `configs/lmcache_cpu_example.yaml`、`configs/lmcache_disk_example.yaml` | 示例配置，未证明能被当前 LMCache 版本直接消费 |
| RuntimeAdapter protocol | `runtime/adapters.py` | 只有接口，没有真实 adapter class |

### Synthetic 模块

| 模块 | Synthetic 行为 | 可用价值 |
| --- | --- | --- |
| `SyntheticBackend` | 用固定公式模拟 prefill/decode time、cache discount、jitter | 验证 benchmark 输出链路 |
| `SyntheticKVCache` | Python LRU + prefetch set | 生成 cache hit/prefetch hit 指标 |
| `SelectiveKVPrefetchMVP` | 用 `asyncio.sleep` 模拟 CPU miss/prefetch latency | 验证 prefetch queue、residency、LRU、waste 统计 |
| `SelectivePrefetchBenchmarkBackend` | 同一 synthetic trace 跑 no-prefetch 与 selective prefetch | 观察策略模拟趋势，但不能代表真实模型速度 |

### 已验证模块

| 已验证项 | 证据 | 审计结论 |
| --- | --- | --- |
| Synthetic baseline benchmark | `results/baseline_synthetic_20260525_232503/benchmark_results.csv` | benchmark 输出链路可运行，但不是真实 runtime |
| Selective prefetch MVP | `results/selective_prefetch_mvp_20260525_235650/benchmark_results.csv` | prefetch 模拟器可运行，能输出 synthetic hit/waste/TPOT change |
| vLLM endpoint smoke baseline | `results/local_smoke_test/benchmark_results.csv`、`request_results.jsonl` | 真实 GPU endpoint 跑通，Qwen2.5-7B-Instruct 约 54 tok/s，GPU memory sample 约 22134 MB |
| 第三方源码分析 | `docs/codebase_analysis.md`、`docs/third_party_analysis.md`、`docs/clone_validation.md` | 分析扎实，但不是集成实现 |

### 未验证模块

| 未验证项 | 当前状态 | 需要的验证 |
| --- | --- | --- |
| vLLM + LMCache CPU offload | 有 config 和 wrapper，无真实结果 | 成功启动、endpoint benchmark、LMCache hit/offload metrics、显存/延迟对照 |
| vLLM + LMCache Disk offload | 有 config 和 wrapper，无真实结果 | 磁盘读写、cache load/store、延迟影响、容量提升 |
| AstraKV-W Runtime Adapter | 只有 protocol | 能从 vLLM/LMCache 获取 request/block/cache event 并落到 `KVChunkMeta` |
| 真实 Selective Prefetch | 只有 simulator | policy 调用 LMCache/vLLM public API 触发真实 prefetch/load |
| Scheduler hint 闭环 | 只有 hint object | hint 被真实 scheduler 或 adapter 消费并改变调度行为 |
| MoE expert activation | 无实现 | MoE 模型访存 trace、expert weight/cache/offload 策略 |

## 任务 2：真实运行链路分析

### 链路 A：当前真实 vLLM endpoint smoke benchmark

这是当前项目最重要的真实运行链路。

```text
用户/脚本
→ scripts/run_real_benchmark.py
→ urllib HTTP POST /v1/chat/completions
→ 已启动的 vLLM OpenAI-compatible endpoint
→ Qwen/Qwen2.5-7B-Instruct 模型真实推理
→ streaming SSE response
→ 客户端统计 TTFT/TPOT/latency/output tokens
→ nvidia-smi 请求前后采样 GPU memory
→ request_results.jsonl
→ benchmark_results.csv
→ benchmark_report.md
```

分类：

| 环节 | 类型 | 说明 |
| --- | --- | --- |
| `run_real_benchmark.py` HTTP 请求 | 真实 | 真实访问 endpoint |
| vLLM server | 真实，但在仓库外运行 | 仓库不 import/modify vLLM |
| 模型推理 | 真实 | 结果显示真实 GPU memory probe |
| TTFT/TPOT/latency | 真实客户端测量 | 样本少，仍可作为 smoke baseline |
| GPU memory | 真实但粗粒度 | 请求前后采样，不是连续峰值 |
| AstraKV-W runtime/offload/prefetch/scheduler | 未参与 | 不能宣称优化生效 |

### 链路 B：Synthetic baseline benchmark

```text
python scripts/benchmark_runner.py
→ load benchmarks/configs/baseline.yaml
→ build BenchmarkCase matrix
→ MetricsCollector.measure_scratch_io
→ SyntheticBackend.run_case
→ SyntheticKVCache.lookup
→ time.sleep 模拟 prefill/decode latency
→ MetricsCollector.snapshot
→ summarize_case
→ benchmark_results.csv
→ plot_benchmarks.py
→ benchmark_report.md + charts/*.png
```

分类：

| 环节 | 类型 | 说明 |
| --- | --- | --- |
| benchmark framework | 真实代码 | 可以实际运行 |
| latency、KV hit、throughput | Synthetic | 由公式和 sleep 产生 |
| GPU memory probe | 真实探针，环境依赖 | 结果中可为空 |
| vLLM/LMCache/model | 未参与 | 不是真实推理 |

### 链路 C：Selective KV Prefetch MVP synthetic comparison

```text
python scripts/benchmark_runner.py
→ load benchmarks/configs/selective_prefetch_mvp.yaml
→ SelectivePrefetchBenchmarkBackend.run_case_pair
→ 构造 synthetic decode trace
→ run synthetic_no_prefetch
→ run selective_prefetch_mvp
→ SelectiveKVPrefetchMVP.add_cpu_blocks
→ async prefetch queue + LRU GPU tier simulation
→ access block: GPU_HIT / PREFETCH_HIT / CPU_MISS
→ synthetic metrics: hit rate, waste rate, estimated GPU KV MB
→ CSV/MD/PNG
```

分类：

| 环节 | 类型 | 说明 |
| --- | --- | --- |
| prefetch queue、LRU、metrics | 真实 Python 逻辑 | 算法状态机可运行 |
| CPU/GPU tier movement | Synthetic | 没有 tensor、CUDA、DMA、LMCache |
| TPOT change | Synthetic | 模拟器里的变化，不是模型推理速度 |
| GPU KV memory reduction | Synthetic estimate | 来自 logical block size/capacity |

### 链路 D：规划中的 vLLM + LMCache CPU/Disk

```text
launch_lmcache_vllm.sh cpu/disk
→ 设置 LMCACHE_CONFIG_FILE
→ launch_vllm_server.sh
→ vLLM endpoint
→ 期望 LMCache connector 生效
→ run_real_benchmark.py
→ endpoint metrics + DGX metrics
```

当前判断：这是路线和 wrapper，不是已验证链路。没有看到对应的真实结果目录、LMCache hit/offload 记录、CPU/Disk cache event 或对照实验。因此它不能被写成已完成 offload。

## 任务 3：赛题映射分析

| 赛题要求 | 当前状态 | 证据 | 评审判断 |
| --- | --- | --- | --- |
| LLM 访存分析 | 部分完成 | `docs/codebase_analysis.md` 分析 vLLM/LMCache/SGLang/TensorRT-LLM/llama.cpp/FlashAttention | 文档分析较强，但缺少真实 trace 数据 |
| KV Cache 分析 | 部分完成 | `kv_cache/` metadata、block table；第三方 KV manager 分析；synthetic KV trace | 概念和模拟完成，真实 runtime KV event 未接入 |
| 参数加载分析 | 部分完成偏弱 | 文档提到 model parameter offload 参考；vLLM 真实 baseline 加载模型 | 没有参数加载 trace、page fault、weight offload 实验 |
| 专家激活 MoE | 未完成 | 当前模型为 Qwen2.5-7B-Instruct，非 MoE；无 expert trace | 赛题提到 MoE，但当前项目没有覆盖 |
| Offload | 部分完成偏弱 | `offload/tier_placement.py`、LMCache CPU/Disk configs/wrappers | 只有 intent 与计划，真实 offload 未验证 |
| 虚拟内存 | 未完成/文档参考 | llama.cpp mmap、TensorRT UVM 文档分析 | 没有 mmap/page fault/madvise/UVM 实现或实验 |
| 多级内存 | 部分完成 | `MemoryTier`、TierPlacement、Selective MVP 的 CPU/GPU tier、LMCache CPU/Disk configs | 数据模型具备，真实 GPU/CPU/SSD tier 闭环未完成 |
| Prefetch | 部分完成 | `SelectiveKVPrefetchMVP`、synthetic benchmark | 有模拟策略和指标，未接真实 vLLM/LMCache prefetch |
| Runtime 优化 | 未完成/刚起步 | 真实 vLLM baseline 只是测量；runtime adapter 未实现 | 还没有改变 runtime 行为 |
| Benchmark | 部分完成 | synthetic benchmark、真实 endpoint smoke baseline、plot、metrics | pipeline 有基础，但样本少、真实对照不足 |
| 可演示性 | 部分完成偏弱 | 可演示 synthetic chart 与 vLLM smoke baseline | 缺少“优化前后”的真实 demo |

## 任务 4：完成度评估

如果今天提交比赛，按评委视角估算：

| 维度 | 完成度 | 理由 |
| --- | ---: | --- |
| 工程完成度 | 35% | 目录清晰、文档较多、runner/metrics/plot/smoke benchmark 有实现；但核心系统链路未闭合 |
| Runtime 完成度 | 15% | 只有 adapter protocol 和 object manager skeleton；没有真实 vLLM adapter 或 scheduler/connector integration |
| Offload 完成度 | 10% | 只有 tier placement 和 LMCache wrapper/config intent；没有真实 CPU/SSD offload 成功结果 |
| Prefetch 完成度 | 30% | Synthetic MVP 完成度不错，有 hit/waste/LRU/queue；但未进入真实 runtime |
| Benchmark 完成度 | 45% | synthetic benchmark 和真实 endpoint smoke baseline 都有；但缺少 repeat、p50/p95、高并发、长上下文、真实 offload 对照 |
| Demo 完成度 | 25% | 可以演示 vLLM baseline 与 synthetic prefetch，但不能演示真实优化收益 |
| 答辩准备度 | 40% | 文档能讲清路线和边界；但评委追问“真实贡献在哪里”时证据不足 |

综合完成度：约 28%-35%。若按“参赛可提交”标准，项目可作为早期原型提交；若按“有竞争力作品”标准，目前仍明显不足。

## 任务 5：Gap Analysis

### P0：必须完成

| Gap | 为什么必须 | 建议动作 |
| --- | --- | --- |
| 真实 baseline vs variant 对照 | 没有对照就无法证明 runtime 优化 | 固定模型、GPU、prompt、context、batch，比较 vLLM only 与 vLLM+LMCache CPU/Disk |
| LMCache CPU/Disk 真实跑通 | 赛题核心是内存受限、多级内存、offload | 验证 LMCache integration flag，记录 hit/load/store/offload metrics |
| 真实 KV event/metric adapter | 当前 AstraKV-W 与真实 runtime 脱节 | 先做 read-only adapter：采 request id、token count、cache hit、load/store event |
| 复现命令修正 | `docs/reproduction.md` 中 `run_real_benchmark.py --config` 与脚本不一致 | 增加 `--config` 支持，或更新文档使用当前 CLI |
| 增加真实 benchmark 矩阵 | 当前真实结果只有 2 个请求 | context 512/1024/2048/4096/8192，batch 1/2/4，repeat 5+ |
| 连续 GPU/CPU/SSD 采样接入真实 benchmark | 请求前后 GPU memory 不够可信 | 将 `DgxMetricsCollector` 接入 `run_real_benchmark.py` 并输出 samples |
| 明确 demo 脚本 | 评委需要现场可演示 | 一键启动 baseline、运行 benchmark、生成 report、展示图表 |

### P1：强烈建议

| Gap | 价值 | 建议动作 |
| --- | --- | --- |
| Selective Prefetch 从 simulator 变成 adapter policy | 形成项目独有贡献 | 用真实 LMCache lookup/load API 或 vLLM connector metadata 驱动预取 |
| Trace schema | 支撑访存分析和答辩 | 定义并采集 cache_match、alloc、load、store、evict、prefetch events |
| 长上下文压力测试 | 贴合内存受限场景 | 测最大可承载 context、OOM rate、成功率、latency |
| CPU/Disk tier 容量收益 | 比单纯速度更贴合赛题 | 报告显存下降、可支持更长 context 或更多并发 |
| 报告图表规范化 | 提升可信度 | p50/p95、error bars、失败率、显存峰值曲线 |
| 单元测试 | 工程质量 | 给 metadata、block table、tier placement、prefetch MVP 加 pytest |

### P2：加分项

| Gap | 加分点 |
| --- | --- |
| UVM/mmap/page fault 实验 | 更贴近 OS/虚拟内存味道 |
| MoE expert activation trace | 覆盖赛题提到的专家激活 |
| Prefetch policy ablation | 比较 next-N、LRU-aware、deadline-aware、reuse-score 等策略 |
| Web/demo dashboard | 提升演示体验 |
| 与 FlexInfer 对标表 | 强化研究叙事 |
| 多模型测试 | Qwen 7B + 小 MoE + 更长上下文模型 |

## 任务 6：技术路线评估

当前路线：

```text
vLLM + LMCache + FlashAttention + Selective Prefetch + CPU Tier + SSD Tier + Runtime Adapter
```

### 是否符合赛题方向

符合。vLLM 提供真实 LLM serving runtime，LMCache 对应 KV cache offload/prefetch/storage，多级 CPU/SSD tier 对应内存受限，FlashAttention 提供 paged KV/kernel layout 参考，Runtime Adapter 避免直接改复杂第三方核心。

问题是：符合方向不等于已经实现。目前路线主要停在 reference analysis、skeleton、wrapper 和 synthetic simulator。

### 是否具有创新性

潜在创新性中等偏上，但当前已实现创新性偏弱。

潜在创新点：

- 面向内存受限 LLM inference 的 runtime-agnostic KV metadata 和 tier intent。
- 在 LMCache/vLLM 之上增加 selective prefetch policy。
- 同时报告容量、延迟、prefetch hit/waste、IO 指标，而不是只看吞吐。

当前不足：

- Selective prefetch 仍是 next-N synthetic trace。
- 没有真实 cache event 驱动。
- 没有证明比 LMCache 默认策略更好。

### 是否具有可行性

可行，但必须收敛。最现实路径不是改 FlashAttention 或 vLLM scheduler core，而是：

1. 先跑通 vLLM + LMCache CPU/Disk。
2. 从 LMCache/vLLM public metrics 或 connector event 读取真实 cache 状态。
3. 让 AstraKV-W policy 只做“决策和观测”，实际搬运交给 LMCache。
4. 用 benchmark 证明显存、成功率、长上下文能力或 TPOT/TTFT 的改善。

### 是否具有获奖潜力

如果停留在当前状态，获奖潜力较低。若在 P0/P1 上完成真实闭环，获奖潜力可以提升到中等偏上，尤其适合 OS 功能挑战赛中“系统可演示 + 多级内存 + 真实数据”的评审口味。

### 最应该投入时间的模块

优先级最高：

1. vLLM + LMCache CPU/Disk 真实验证。
2. `run_real_benchmark.py` 配置化、矩阵化、连续指标采样化。
3. Read-only LMCache/vLLM cache event adapter。
4. Selective prefetch policy 与真实 LMCache prefetch/load API 的最小闭环。
5. Demo runbook 和可复现实验包。

不建议近期投入：

- CUDA kernel 修改。
- FlashAttention 内核改造。
- 完整 scheduler 重写。
- MoE 优化，除非已有稳定核心闭环。

## 任务 7：V0.1 到 V1.0 Roadmap

### V0.1：真实基线闭环

| 项目 | 内容 |
| --- | --- |
| 目标 | 稳定复现 vLLM only 真实 baseline，修正 benchmark/config/docs 不一致 |
| 修改模块 | `scripts/run_real_benchmark.py`、`scripts/dgx_metrics_collector.py`、`configs/dgx_spark_vllm_qwen7b.yaml`、`docs/reproduction.md` |
| 测试方法 | 启动 vLLM，跑 context 512/1024/2048、batch 1、repeat 5 |
| 验收标准 | 生成 CSV/JSONL/MD/samples，成功率 100%，TTFT/TPOT p50/p95 可读 |
| 赛题贡献 | 建立真实测量基线和可复现链路 |

### V0.2：LMCache CPU/Disk Baseline

| 项目 | 内容 |
| --- | --- |
| 目标 | 跑通 vLLM + LMCache CPU 与 Disk backend |
| 修改模块 | `scripts/launch_lmcache_vllm.*`、`configs/dgx_spark_lmcache_cpu.yaml`、`configs/dgx_spark_lmcache_disk.yaml` |
| 测试方法 | vLLM only、LMCache CPU、LMCache Disk 三组同矩阵对照 |
| 验收标准 | endpoint 正常，LMCache 日志证明 backend 生效，GPU/CPU/SSD metrics 有数据 |
| 赛题贡献 | 进入真实多级内存和 offload 主题 |

### V0.3：真实 Cache Event Adapter

| 项目 | 内容 |
| --- | --- |
| 目标 | 将真实 cache hit/miss/load/store/offload event 映射到 AstraKV-W metadata |
| 修改模块 | 新增 `runtime/adapters_lmcache.py` 或 `runtime/adapters_vllm.py`，扩展 `kv_cache/metadata.py` |
| 测试方法 | replay endpoint requests，导出 trace JSONL |
| 验收标准 | 每个 request 能看到 token span、cache event、tier、bytes、latency |
| 赛题贡献 | 支撑 LLM 访存行为分析 |

### V0.4：Selective Prefetch Real Adapter

| 项目 | 内容 |
| --- | --- |
| 目标 | 让 prefetch policy 对真实 LMCache/vLLM cache events 做决策 |
| 修改模块 | `prefetch/selective_kv.py` 抽 policy，新增 adapter movement layer |
| 测试方法 | same trace 下对比 no-prefetch/default/astrakv-prefetch |
| 验收标准 | prefetch submitted/completed/hit/waste 来自真实 backend，不再只来自 simulator |
| 赛题贡献 | 形成项目核心 runtime 优化贡献 |

### V0.5：长上下文与内存压力实验

| 项目 | 内容 |
| --- | --- |
| 目标 | 证明内存受限场景下容量或稳定性改善 |
| 修改模块 | benchmark configs、run scripts、report generator |
| 测试方法 | context 512 到 8192/16384，batch 1/2/4，repeat 5+ |
| 验收标准 | 报告最大成功 context、OOM rate、GPU peak、latency p95 |
| 赛题贡献 | 直接回应 GPU 显存不足和边缘部署难题 |

### V0.6：Benchmark 可信度增强

| 项目 | 内容 |
| --- | --- |
| 目标 | 形成可审计实验体系 |
| 修改模块 | `scripts/run_real_benchmark.py`、plot/report 工具、`docs/reproduction.md` |
| 测试方法 | 固定 prompt set，记录环境版本、命令、config、git commit |
| 验收标准 | 一键复现 baseline/variant，报告含 p50/p95、error bars、失败率 |
| 赛题贡献 | 提升评委对 benchmark 的信任 |

### V0.7：Demo 集成

| 项目 | 内容 |
| --- | --- |
| 目标 | 现场演示“内存受限 -> offload/prefetch -> 指标变化” |
| 修改模块 | scripts、docs、可选 dashboard |
| 测试方法 | 单命令跑 smoke demo，展示 CSV/图表/日志 |
| 验收标准 | 10 分钟内完成演示，失败可降级到已归档结果 |
| 赛题贡献 | 提升可演示性 |

### V0.8：Policy Ablation

| 项目 | 内容 |
| --- | --- |
| 目标 | 证明 AstraKV-W policy 不是简单包装 |
| 修改模块 | prefetch policy、configs、report |
| 测试方法 | no-prefetch、LMCache default、next-N、reuse-aware、deadline-aware 对比 |
| 验收标准 | 至少一个真实场景下显存/成功率/latency 有可解释收益 |
| 赛题贡献 | 强化创新性 |

### V0.9：系统打磨

| 项目 | 内容 |
| --- | --- |
| 目标 | 工程质量、文档、测试和异常恢复 |
| 修改模块 | 全局测试、CI、docs、error handling |
| 测试方法 | unit tests + integration smoke + reproduction dry run |
| 验收标准 | 干净命令、无明显文档错配、失败信息可诊断 |
| 赛题贡献 | 工程完整度 |

### V1.0：参赛版本

| 项目 | 内容 |
| --- | --- |
| 目标 | 提交完整系统、报告、demo、复现实验 |
| 修改模块 | 只做稳定性修复和报告固化 |
| 测试方法 | 从 clean clone 复现 baseline/variant，现场演示 rehearsal |
| 验收标准 | 有真实优化闭环、可信 benchmark、清晰边界、可答辩 |
| 赛题贡献 | 达到有竞争力参赛作品标准 |

## 任务 8：测试体系分析

### 当前可以进行的测试

| 测试 | 命令 | 输出文件 | 验收标准 | 类型 |
| --- | --- | --- | --- | --- |
| 依赖 smoke test | `python -c "import yaml, psutil, matplotlib; print('AstraKV-W helper deps ok')"` | stdout | 成功打印，无 ImportError | 工具验证 |
| Synthetic baseline | `python scripts\benchmark_runner.py --config benchmarks\configs\baseline.yaml --output-dir results` | `results/baseline_synthetic_<timestamp>/benchmark_results.csv`、`benchmark_report.md`、`charts/*.png` | 生成结果目录，CSV 非空 | Synthetic |
| Selective prefetch MVP | `python scripts\benchmark_runner.py --config benchmarks\configs\selective_prefetch_mvp.yaml --output-dir results` | `results/selective_prefetch_mvp_<timestamp>/benchmark_results.csv`、charts | 出现 `synthetic_no_prefetch` 与 `selective_prefetch_mvp` 两组 backend | Synthetic |
| vLLM endpoint reachability | `curl http://127.0.0.1:8000/v1/models` | stdout JSON | 返回 model list，无连接错误 | 真实 endpoint |
| vLLM server launch | `scripts\launch_vllm_server.ps1` 或 `bash scripts/launch_vllm_server.sh` | server log | endpoint 可访问，模型加载成功 | 真实运行前置 |
| 真实 endpoint benchmark | `python scripts\run_real_benchmark.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --context-lengths 512 1024 --batch-sizes 1 --output-tokens 64 --repeat 1 --output-dir results/local_smoke_test` | `benchmark_results.csv`、`request_results.jsonl`、`benchmark_report.md` | `status=ok`，`success_count=request_count`，TTFT/TPOT 非空 | 真实 smoke |
| Plot existing CSV | `python scripts\plot_benchmarks.py --csv results\selective_prefetch_mvp_20260525_235650\benchmark_results.csv --output-dir results\selective_prefetch_mvp_20260525_235650\charts` | `charts/*.png` | PNG 生成成功 | 工具 |
| LMCache CPU launch wrapper | `bash scripts/launch_lmcache_vllm.sh cpu` 或 `scripts\launch_lmcache_vllm.ps1 -Backend cpu` | server log | 仅能证明 wrapper 执行；必须另查 LMCache 是否真正启用 | 未验证真实 offload |
| LMCache Disk launch wrapper | `bash scripts/launch_lmcache_vllm.sh disk` 或 `scripts\launch_lmcache_vllm.ps1 -Backend disk` | server log | 同上，需 LMCache 日志和结果验证 | 未验证真实 offload |

### 当前测试体系问题

- 没有 pytest/unit test。
- `docs/reproduction.md` 中示例使用 `run_real_benchmark.py --config ...`，但当前脚本 CLI 没有 `--config` 参数。
- `DgxMetricsCollector` 是可用类，但未接入 `run_real_benchmark.py` 主流程，真实 smoke result 没有 samples。
- LMCache CPU/Disk 测试只有 wrapper 和 config，没有真实结果。
- 没有自动比较 baseline/variant 的 report generator。

## 任务 9：评委视角审查

### 当前项目最大的优点

最大的优点是技术方向清楚，而且边界意识好。项目没有贸然修改 vLLM/FlashAttention/TensorRT-LLM 内核，而是先做第三方代码分析、adapter boundary、metadata model、benchmark pipeline 和真实 vLLM endpoint smoke baseline。这种路线对学生项目来说很稳，后续可扩展性也比较好。

另一个优点是已经明确区分 synthetic 与 real baseline。`docs/real_gpu_baseline.md` 和 `prefetch_design.md` 都说明不能把模拟结果当真实优化收益，这会提升答辩可信度。

### 当前项目最大的缺陷

最大缺陷是“核心贡献尚未进入真实 runtime”。赛题要的是内存受限环境下的 LLM inference runtime optimization，而当前 AstraKV-W 还没有控制真实 KV cache 搬运、offload、prefetch 或 scheduler 行为。

如果今天答辩，评委最可能追问：

- 你的代码到底改变了 vLLM 哪个运行时行为？
- LMCache CPU/Disk 真的启用了吗？证据在哪里？
- Prefetch hit rate 是真实模型运行产生的吗，还是 synthetic trace？
- 显存下降、吞吐提升、长上下文成功率提升在哪里？

当前项目对这些问题的证据还不足。

### 会被认为是“真正贡献”的工作

| 工作 | 为什么是真贡献 |
| --- | --- |
| 真实 vLLM + LMCache CPU/Disk 对照实验 | 直接对应多级内存和 offload |
| 真实 cache event adapter | 把项目从文档/模拟推进到 runtime 观测 |
| Selective prefetch 调用真实 backend | 形成 AstraKV-W 独有优化行为 |
| 长上下文/内存压力下成功率提升 | 比单个吞吐指标更符合赛题 |
| 完整 benchmark + p50/p95 + 复现包 | 评委可以验证，不只是口头叙述 |

### 会被认为只是工程包装的工作

| 工作 | 为什么可能被认为是包装 |
| --- | --- |
| 仅有第三方源码分析 | 有价值，但不是系统实现 |
| 仅有 adapter protocol | 没接真实 runtime 时只是接口 |
| 仅有 launch wrapper | 不能证明优化生效 |
| 仅有 synthetic prefetch speedup | 不能代表真实 LLM inference |
| 仅有 README/roadmap | 对比赛展示有帮助，但不能替代实验 |

## 任务 10：最终结论

### 当前等级判断

当前 AstraKV-W 更像一个“参赛方向预研 + benchmark/adapter 框架原型”，还不是完整参赛系统。

按竞赛提交口径：

- 可以提交为初版原型。
- 可以展示真实 vLLM baseline 跑通。
- 可以展示 synthetic selective prefetch 的设计思想。
- 不能宣称已经完成真实 runtime optimization。
- 不能宣称已经实现 CPU/SSD offload 优化收益。
- 不能宣称已经达到 FlexInfer 类系统效果。

### 决赛级差距

距离决赛级作品，至少还需要：

1. 一个真实可运行的 baseline/variant 对照系统。
2. 真实多级内存路径：GPU/CPU/SSD 至少两级 offload 跑通。
3. 一个 AstraKV-W 自己控制或影响的真实 policy：prefetch、eviction、placement 或 scheduler hint。
4. 可复现实验：同机器、同模型、同 workload、repeat、p50/p95、峰值显存、失败率。
5. 可演示闭环：启动、运行、采样、报告、图表一体化。

### 最终建议

接下来不要继续扩写设计文档，也不要急着碰 CUDA kernel。最应该做的是把真实链路打穿：

```text
vLLM only baseline
→ vLLM + LMCache CPU
→ vLLM + LMCache Disk
→ 真实 cache/offload metrics
→ AstraKV-W selective prefetch policy
→ baseline vs variant report
```

只要这条链路成立，项目就会从“准备充分的原型”变成“有系统贡献的参赛作品”。
