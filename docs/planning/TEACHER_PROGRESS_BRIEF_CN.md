# AstraKV-W 当前工作进展简要说明
## 1. 项目方向

我们目前围绕 OS 功能挑战赛赛题：

```text
Runtime Optimization of LLM Inference for the Memory Constraint System
```

推进一个名为 **AstraKV-W** 的方案。

整体思路是从操作系统内存管理角度优化内存受限场景下的 LLM 推理运行时，重点关注：

- KV Cache 访存行为；
- CPU / SSD memory tiering；
- on-demand loading；
- selective prefetch；
- profile-guided scheduling；
- memory pressure control；
- load-vs-recompute；
- 可选 MoE expert 访问行为分析；
- **OS 虚拟内存技术集成**：mmap+madvise KV-cache 分层管理、userfaultfd 按需缺页加载、layer-level 模型权重 offloading PoC。

当前方案基于 **vLLM + LMCache**，尽量采用非侵入式 runtime layer，不直接大规模修改 vLLM、LMCache 或 CUDA 内核。
项目已重组为标准 Python 包结构（`astrakv/`），提供统一 CLI 入口（`cli.py`）和项目工具链（`pyproject.toml`、`Makefile`）。

## 2. 目前已经完成的工作

### 2.1 本地非 GPU 工具链

目前本地已经完成了一套非 GPU 分析、调度和报告工具链，主要包括：

| 模块 | 当前进展 |
| --- | --- |
| Benchmark runner | 已完成真实 endpoint benchmark 脚本，用于后续连接 vLLM / LMCache 服务 |
| Metrics collector | 已完成 GPU / CPU / SSD 指标采集逻辑 |
| Cache event parser | 已完成 vLLM / LMCache 日志和 benchmark artifact 的 cache event 提取 |
| Unified trace store | 已完成统一 trace schema 和 trace store 构建 |
| ProfileDB | 已完成基于 trace 的 profile 数据库 |
| Chunk scorer | 已完成 KV chunk 的 prefetch / keep / offload / drop 评分 |
| Partial KV load planner | 已完成 partial KV load 的规划和字节节省估计 |
| Load-vs-recompute planner | 已完成 load / recompute / defer / drop 决策器 |
| Unified object scheduler | 已完成统一对象调度 hints 生成 |
| Memory pressure controller | 已完成内存压力分析和 pressure-aware hints 生成 |
| Quality evaluator | 已完成输出一致性、token divergence 和可选 PPL 评估 |
| Hidden-state / CKA evaluator | 已完成 hidden-state drift 和 CKA 评估工具 |
| MoE event parser | 已完成 MoE expert route event 提取工具 |
| MoE expert predictor | 已完成 next-token / history-window / profile-guided 专家预测 |
| Report generator | 已完成 competition report 生成工具 |
| Demo dashboard | 已完成 demo dashboard 生成工具 |
| MMap KV Cache | 已完成 mmap+madvise KV-cache 虚拟内存管理器（OS page fault 级 on-demand loading，MADV_WILLNEED 预取，MADV_DONTNEED 换出，mincore 驻留查询）|
| UFFD KV Loader | 已完成 userfaultfd 按需缺页加载器（Linux kernel >= 4.11，需 CAP_SYS_PTRACE）|
| Layer Offload PoC | 已完成 layer-level 模型权重 offloading 概念验证（对标 FlexInfer）|
| Edge Simulator | 已完成 cgroups v2 边缘设备内存模拟工具（16GB/24GB/32GB）|
| Ablation Runner | 已完成共享前缀消融实验自动化脚本（A/B/C/D 四组）|

### 2.2 项目工程化重组

近期完成了一次项目结构重组，使其符合成熟 Python 工程规范：

- **`astrakv/` 主包**：将所有散落的顶层模块（runtime, kv_cache, prefetch, offload, scheduler, moe, evaluation, vm, benchmarks, experiments）统一迁移到 `astrakv/` 包内；
- **统一 CLI**：创建 `cli.py` 提供 8 个子命令（benchmark, prefetch, analyze, report, vm, test, edge, ablation）；
- **项目配置**：添加 `pyproject.toml` 和 `Makefile`；
- **文档重组**：将 docs/ 按主题分类为 architecture/、analysis/、guides/、planning/。

### 2.3 在此之前补充的模块

最近主要补了两个非 GPU 缺口：

1. **Memory Pressure Controller**

   根据 benchmark、sample、trace 中的 GPU memory、CPU RSS、SSD traffic、OOM 和 error rate，判断当前内存压力等级，并生成 passive scheduler hints。

   输出包括：

   - `memory_pressure_decisions.csv`
   - `memory_pressure_hints.jsonl`
   - `memory_pressure_report.md`
   - `memory_pressure_manifest.json`

2. **Advanced Expert Predictor**

   在原有 MoE expert predictor 基础上，增加：

   - `next_token` predictor；
   - `history_window` predictor；
   - `profile_guided` predictor label；
   - expert transition score；
   - predictor ablation 字段。

   该模块目前用于 MoE expert route trace 的离线分析和 expert prefetch hints 生成。

## 3. 当前测试情况

本地单元测试已经通过（2026-06-10 更新）：

```text
python cli.py test        # 推荐方式
python -m pytest tests/ -v # 或直接调用 pytest

120 passed, 3 skipped (GPU-dependent tests)
```

目前测试覆盖了：

- cache event parser、trace schema、ProfileDB；
- chunk scorer、partial KV load、load-vs-recompute；
- unified object scheduler、memory pressure controller；
- quality evaluator、hidden-state / CKA evaluator；
- MoE expert loader、MoE expert predictor；
- benchmark / report / dashboard 工具；
- **MMap KV Cache 虚拟内存管理器**（18 个测试用例）；
- **Layer Offload PoC**（9 个测试用例，含 GPU skip）。

## 4. 当前还没有完成的部分

目前还缺少的是 **GPU 上的真实运行证据**。

**但虚拟内存相关模块已可在纯 CPU 环境运行并产出有效证据**：
- `python cli.py vm mmap` — 演示 OS page fault 级 on-demand loading、MADV_WILLNEED 预取、MADV_DONTNEED 换出；
- `python cli.py vm demo` — 运行通用 mmap 虚拟内存 demo（页面级访问、延迟、RSS 测量）；
- `python cli.py vm layer` — 层卸载 PoC（需 GPU + torch）。

也就是说，本地已经具备实验工具链和分析工具，但下面这些结论还必须通过真实 GPU 实验验证：

- vLLM-only baseline；
- vLLM + LMCache CPU backend；
- vLLM + LMCache disk backend；
- memory-constrained stress benchmark；
- endpoint-level selective prefetch；
- LMCache/vLLM cache hit / load / store / offload 事件；
- ProfileDB 基于真实 trace 的效果；
- memory pressure controller 基于真实采样数据的效果；
- scheduler hints 是否能进一步影响真实 runtime；
- 可选 MoE expert route / prediction 实验。

我们目前不会把 offline planner / passive hints 直接表述为“真实 runtime 已经生效”。这一点在后续实验和答辩中会单独说明。

## 5. 已整理的 GPU 验证计划

我们已经整理了一份 GPU 真实实验流程文档：

其中包括：

- 服务器配置建议；
- DGX Spark / 标准 GPU 服务器使用注意事项；
- smoke test 顺序；
- vLLM baseline；
- LMCache CPU baseline；
- LMCache disk baseline；
- cache event extraction；
- baseline comparison；
- memory-constrained stress；
- selective prefetch；
- trace store；
- ProfileDB；
- chunk scoring；
- memory pressure analysis；
- partial KV load planning；
- load-vs-recompute planning；
- unified object scheduler；
- quality / PPL / CKA；
- MoE 可选实验；
- final competition report；
- demo dashboard；
- 每一步验收标准。

## 6. 计划中的 GPU 验证主线

后续上 GPU 后，计划优先跑下面这条主线：

```text
1. vLLM smoke test
2. LMCache CPU smoke test
3. LMCache disk smoke test
4. vLLM-only 正式 baseline
5. vLLM + LMCache CPU 正式 baseline
6. vLLM + LMCache disk 正式 baseline
7. cache event extraction
8. baseline comparison
9. memory-constrained stress benchmark
10. endpoint-level selective prefetch
11. trace store + ProfileDB
12. memory pressure controller
13. chunk scoring / partial KV / load-vs-recompute / unified scheduler
14. quality evaluation
15. final competition report
```

优先关注指标：

- TTFT；
- TPOT；
- latency p50 / p95；
- throughput；
- GPU memory / unified memory pressure；
- CPU RSS；
- SSD read/write traffic；
- OOM rate；
- max successful context；
- max successful batch；
- KV cache event；
- prefetch submitted / completed / hit evidence。

