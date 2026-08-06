# 基础版本与增量贡献说明

## 基础版本和外部依赖

AstraKV-W 没有修改 vLLM、LMCache、SGLang、TensorRT-LLM、FlashAttention 或 CUDA kernel 的第三方核心源码。真实推理实验通过 OpenAI-compatible HTTP endpoint、启动脚本、配置文件、server log 和结构化结果文件与 vLLM / LMCache 协同。

项目参考和依赖的外部系统包括：

- vLLM：真实推理 endpoint、KV cache capacity 日志和 baseline 后端
- LMCache：CPU tier / Disk tier 分层缓存后端
- Qwen2.5-7B-Instruct：DGX Spark 实验模型
- FlexInfer、NEO、IMPRESS、Cake、Bidaw、KTransformers 等论文系统：相关工作和设计参照

## 本项目自研增量

本项目的主要增量集中在外围控制平面、证据链、策略分析和复现实验组织：

- KV chunk 元数据、block table、tier placement 和 object manager 等对象抽象
- cache event 抽取、trace schema、ProfileDB 和 memory pressure 分析
- ChunkScorer、PartialKVLoadPlanner、LoadRecomputePlanner 和 UnifiedObjectScheduler 等策略模块
- endpoint-level selective prefetch / warmup 实验和 benchmark-like prefetch 输出
- mmap KV cache、DGX Spark UMA adapter、VM evidence 和 OS 机制 PoC
- 一键 E2E、extended evidence、architecture demo、experiment figures 等复现脚本
- competition report、architecture demo report、figure report 和 artifact inventory 等报告链路

## 边界声明

策略链当前输出 advisory decision、CSV/JSONL 和报告证据，不声明已经替换 vLLM 内部 KV scheduler。OS VM/mmap 模块用于机制验证，不声明真实 vLLM KV cache 已完全 mmap 化。DGX Spark 上不使用 `gpu_memory_peak_mb` 作为 case-level GPU framebuffer memory 结论。
