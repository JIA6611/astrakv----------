# AstraKV-W GPU 真实实验完整流程

状态：GPU 验证执行手册  
日期：2026-06-08  
目标比赛：2026 全国大学生计算机系统能力大赛 OS 功能挑战赛  
赛题：Runtime Optimization of LLM Inference for the Memory Constraint System

## 0. 这份文档解决什么

这份文档用于指导你在真实 GPU 服务器上一次性完成 AstraKV-W 的比赛验收实验。

本地已经完成的内容包括：

- benchmark runner、metrics collector、cache event parser
- trace store、ProfileDB、chunk scorer
- partial KV load planner、load-vs-recompute planner
- unified object scheduler、memory pressure controller
- quality / PPL evaluator、hidden-state / CKA evaluator
- MoE expert event parser、MoE expert predictor
- competition report generator、demo dashboard generator
- **OS 虚拟内存集成**（mmap+madvise KV cache、userfaultfd 按需加载、layer offload PoC）
- **边缘设备内存模拟**（cgroups v2 16GB/24GB/32GB 配置）
- **共享前缀消融实验**（A/B/C/D 四组自动化脚本）
- **统一 CLI 入口**（`cli.py`：benchmark, prefetch, analyze, report, vm, test, edge, ablation）
- 项目已重组为 `astrakv/` 标准 Python 包结构

GPU 上要完成的是“真实证据”：

- vLLM 真实 baseline
- vLLM + LMCache CPU backend 真实 baseline
- vLLM + LMCache disk backend 真实 baseline
- GPU / CPU / SSD 连续采样
- LMCache/vLLM cache event 证据
- memory-constrained stress benchmark
- endpoint-level selective prefetch 对比
- ProfileDB / scheduler / memory pressure / report 全链路产物
- 可选 MoE 真实专家路由实验

## 1. 推荐服务器规格

### 1.1 最短参赛配置

适合：跑通 P0，证明 vLLM baseline、LMCache CPU/disk、stress、prefetch。

| 项目 | 建议 |
| --- | --- |
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | 1 x NVIDIA L40S 48GB，或 1 x A6000 48GB，或 1 x A100 40GB |
| CPU | 16 核以上 |
| 内存 | 128GB |
| 系统盘 | 200GB |
| NVMe 数据盘 | 1TB 推荐，最低 500GB |
| CUDA | 与 vLLM/PyTorch wheel 匹配 |
| Python | 3.10 或 3.11 |

说明：

- 48GB 级别 GPU 比较稳，可以跑 Qwen2.5-7B-Instruct 的 8192 context 实验。
- 40GB A100 也可用，但 batch=4、context=8192 可能出现 OOM，这反而可以作为 memory-constrained 证据。
- 24GB GPU 可以做 smoke 和部分 stress，但完整矩阵容易失败，不建议作为最终正式服务器。

### 1.2 推荐获奖配置

适合：完整 P0 + P1 + 一部分 P2，加上更稳定的长上下文和 MoE 证据。

| 项目 | 建议 |
| --- | --- |
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | 1 x A100 80GB / H100 80GB / H200 141GB |
| CPU | 32 核以上 |
| 内存 | 256GB |
| NVMe 数据盘 | 2TB |
| 网络 | 能稳定下载 Hugging Face 模型，或提前离线缓存 |

说明：

- 80GB GPU 是最推荐的正式评测配置。
- CPU 内存越大，LMCache CPU tier 和多轮实验越稳。
- NVMe 盘越大，LMCache disk backend 的缓存、日志、sample 和报告越不容易中途爆盘。

### 1.3 冲奖增强配置

适合：MoE、更多模型、更长 context、更复杂消融。

| 项目 | 建议 |
| --- | --- |
| GPU | 2 x A100 80GB / 2 x H100 80GB，或单卡 H200 141GB |
| CPU | 48 核以上 |
| 内存 | 512GB |
| NVMe 数据盘 | 4TB |

说明：

- MoE 模型权重和专家实验比 dense 7B 大得多。
- 如果只做 MoE route trace 而不做大模型完整 serving，可以先选择较小 MoE 模型。
- 如果要真实 expert weight loading / prefetch，需要更多工程和显存，不作为最短参赛路径。

## 2. 容量预估

### 2.1 显存估计

以 `Qwen/Qwen2.5-7B-Instruct` 为主模型：

| 项目 | 估计 |
| --- | --- |
| 模型权重 FP16/BF16 | 约 15GB 到 18GB |
| vLLM runtime / CUDA / graph / allocator overhead | 约 2GB 到 8GB |
| KV cache，8192 context，batch 越大越高 | 数 GB 到十几 GB |
| 完整 baseline 矩阵推荐显存 | 48GB 以上 |
| stress / 长上下文 / batch=4 推荐显存 | 80GB 更稳 |

比赛需要的是 memory-constrained 系统，不是所有 case 都必须成功。OOM、最大可成功 context、最大可成功 batch、成功率变化，都是 stress benchmark 的有效证据。

### 2.2 磁盘估计

| 内容 | 建议容量 |
| --- | ---: |
| Python 环境 / wheel / cache | 20GB 到 50GB |
| Qwen2.5-7B 模型缓存 | 20GB 到 40GB |
| vLLM / LMCache 日志 | 5GB 到 50GB |
| benchmark results / samples / charts | 5GB 到 30GB |
| LMCache disk store | 100GB 到 500GB |
| 预留空间 | 200GB 以上 |

最低建议：500GB NVMe。  
正式建议：1TB NVMe。  
冲奖建议：2TB 或以上。

### 2.3 一次完整实验大约耗时

| 阶段 | 估计耗时 |
| --- | ---: |
| 环境安装和模型下载 | 1 到 4 小时 |
| smoke 测试 | 20 到 40 分钟 |
| vLLM baseline 正式矩阵 | 1 到 3 小时 |
| LMCache CPU 正式矩阵 | 1 到 3 小时 |
| LMCache disk 正式矩阵 | 1 到 4 小时 |
| stress 三组实验 | 2 到 6 小时 |
| selective prefetch | 30 分钟到 2 小时 |
| 后处理报告 | 10 到 30 分钟 |
| 可选 MoE | 2 小时到 1 天以上 |

建议先跑 smoke，再跑正式矩阵。正式矩阵最好放在晚上连续跑。

## 3. 实验目录约定

在服务器上进入项目根目录：

```bash
cd /path/to/project3136859-384917
```

建议统一输出到：

```text
results/gpu/
```

推荐目录结构：

```text
results/gpu/
  logs/
  p0_1_vllm/
  p0_2_lmcache_cpu/
  p0_3_lmcache_disk/
  p0_5_cache_events_cpu/
  p0_5_cache_events_disk/
  p0_6_comparison/
  p0_7_stress_vllm/
  p0_7_stress_cpu/
  p0_7_stress_disk/
  p0_7_stress_analysis/
  p0_8_selective_prefetch/
  p1_1_trace_store/
  p1_2_profile_db/
  p1_3_chunk_scores/
  p1_5_partial_kv/
  p1_6_load_vs_recompute/
  p1_9_unified_scheduler/
  memory_pressure/
  quality/
  cka/
  competition_report/
  demo_dashboard/
  lmcache_disk_store/
```

创建目录：

```bash
mkdir -p results/gpu/logs
mkdir -p results/gpu/lmcache_disk_store
```

## 4. 环境准备

### 4.1 基础检查

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4.2 安装本项目 helper 依赖

```bash
pip install -r requirements.txt
python -c "import yaml, psutil, matplotlib, pandas, numpy; print('AstraKV-W helper deps ok')"

# 可选：以开发模式安装 astrakv 包
pip install -e .
```

### 4.3 检查 vLLM 和 LMCache

```bash
python -c "import vllm; print('vLLM', vllm.__version__)"
python -c "import lmcache; print('LMCache', getattr(lmcache, '__version__', 'unknown'))"
```

如果没有安装，需要根据服务器 CUDA/PyTorch 版本安装 vLLM 和 LMCache。官方文档参考：

- vLLM GPU 安装文档：<https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- LMCache Quickstart：<https://docs.lmcache.ai/getting_started/quickstart.html>
- LMCache vLLM integration：<https://docs.lmcache.ai/developer_guide/integration.html>

### 4.4 设置通用环境变量

```bash
export ASTRAKV_MODEL=Qwen/Qwen2.5-7B-Instruct
export ASTRAKV_HOST=127.0.0.1
export ASTRAKV_PORT=8000
export ASTRAKV_MAX_MODEL_LEN=8192
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
export LMCACHE_LOCAL_DISK=results/gpu/lmcache_disk_store
```

如果模型已提前下载到本地目录，可以把 `ASTRAKV_MODEL` 改成本地路径：

```bash
export ASTRAKV_MODEL=/data/models/Qwen2.5-7B-Instruct
```

Qwen2.5-7B-Instruct 模型页：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>

## 5. 总体执行顺序

必须按下面顺序执行：

```text
Step 0 环境检查
Step 1 smoke vLLM
Step 2 smoke LMCache CPU
Step 3 smoke LMCache disk
Step 4 正式 vLLM baseline
Step 5 正式 LMCache CPU baseline
Step 6 正式 LMCache disk baseline
Step 7 cache event extraction
Step 8 baseline comparison
Step 9 memory-constrained stress
Step 10 selective prefetch
Step 11 trace store
Step 12 ProfileDB
Step 13 chunk scoring
Step 14 memory pressure controller
Step 15 partial KV load planner
Step 16 load-vs-recompute planner
Step 17 unified object scheduler
Step 18 quality / PPL evidence
Step 19 optional CKA / hidden state
Step 20 optional MoE
Step 21 final competition report
Step 22 demo dashboard
```

不要一开始就跑完整矩阵。先 smoke，确认服务、模型、metrics、LMCache 都正常。

## 6. Step 0：记录机器信息

```bash
mkdir -p results/gpu/env

nvidia-smi | tee results/gpu/env/nvidia_smi.txt
python --version | tee results/gpu/env/python_version.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)" \
  | tee results/gpu/env/torch_cuda.txt
python -c "import vllm; print(vllm.__version__)" | tee results/gpu/env/vllm_version.txt
python -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))" | tee results/gpu/env/lmcache_version.txt
git rev-parse HEAD | tee results/gpu/env/git_commit.txt
df -h | tee results/gpu/env/disk.txt
free -h | tee results/gpu/env/memory.txt
```

验收标准：

- `torch.cuda.is_available()` 为 `True`。
- 能看到 GPU 型号和显存。
- vLLM 可以 import。
- LMCache 可以 import。
- 磁盘剩余空间建议大于 500GB。
- 本地测试全部通过：`python cli.py test`（预期 120 passed, 3 skipped）。

## 7. Step 1：vLLM smoke

### 7.1 启动 vLLM

终端 A：

```bash
mkdir -p results/gpu/logs
bash scripts/launch_vllm_server.sh 2>&1 | tee results/gpu/logs/p0_1_vllm_smoke_server.log
```

### 7.2 健康检查

终端 B：

```bash
curl http://127.0.0.1:8000/v1/models
```

### 7.3 跑 smoke benchmark

```bash
python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/gpu/p0_1_vllm_smoke \
  --context-lengths 512 \
  --batch-sizes 1 \
  --output-tokens 8 \
  --repeat 1
```

验收标准：

- `benchmark_results.csv` 存在。
- `request_results.jsonl` 存在。
- `samples/*_samples.csv` 存在。
- `benchmark_report.md` 中 success rate 不为 0。

停止服务：

```bash
pkill -f "vllm" || true
```

如果你的服务器不允许 `pkill`，手动停止终端 A。

## 8. Step 2：LMCache CPU smoke

终端 A：

```bash
bash scripts/launch_lmcache_vllm.sh cpu 2>&1 | tee results/gpu/logs/p0_2_lmcache_cpu_smoke_server.log
```

终端 B：

```bash
curl http://127.0.0.1:8000/v1/models

python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_cpu.yaml \
  --output-dir results/gpu/p0_2_lmcache_cpu_smoke \
  --context-lengths 512 \
  --batch-sizes 1 \
  --output-tokens 8 \
  --repeat 1
```

验收标准：

- server log 中能看到 LMCache 配置或 connector 相关信息。
- benchmark 正常完成。
- CPU RSS 指标存在。

停止服务：

```bash
pkill -f "vllm" || true
```

## 9. Step 3：LMCache disk smoke

终端 A：

```bash
export LMCACHE_LOCAL_DISK=results/gpu/lmcache_disk_store
bash scripts/launch_lmcache_vllm.sh disk 2>&1 | tee results/gpu/logs/p0_3_lmcache_disk_smoke_server.log
```

终端 B：

```bash
curl http://127.0.0.1:8000/v1/models

python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_disk.yaml \
  --output-dir results/gpu/p0_3_lmcache_disk_smoke \
  --context-lengths 512 \
  --batch-sizes 1 \
  --output-tokens 8 \
  --repeat 1
```

验收标准：

- server log 中能看到 LMCache disk/local disk 配置。
- `results/gpu/lmcache_disk_store` 目录存在。
- benchmark 正常完成。
- disk read/write 指标存在，哪怕 smoke 阶段可能比较小。

停止服务：

```bash
pkill -f "vllm" || true
```

## 10. Step 4：正式 vLLM baseline

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.78
bash scripts/launch_vllm_server.sh 2>&1 | tee results/gpu/logs/p0_1_vllm_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/gpu/p0_1_vllm
```

记录实际 run 目录：

```bash
find results/gpu/p0_1_vllm -maxdepth 2 -name benchmark_results.csv
```

下面用 `<VLLM_RUN>` 表示这个目录，例如：

```text
results/gpu/p0_1_vllm/20260608_230000
```

验收标准：

- `benchmark_results.csv`
- `benchmark_report.md`
- `benchmark_config.json`
- `request_results.jsonl`
- `samples/*_samples.csv`
- `charts/*.png`

停止服务。

## 11. Step 5：正式 LMCache CPU baseline

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
bash scripts/launch_lmcache_vllm.sh cpu 2>&1 | tee results/gpu/logs/p0_2_lmcache_cpu_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_cpu.yaml \
  --output-dir results/gpu/p0_2_lmcache_cpu
```

记录 `<CPU_RUN>`：

```bash
find results/gpu/p0_2_lmcache_cpu -maxdepth 2 -name benchmark_results.csv
```

验收标准：

- benchmark 正常生成。
- server log 证明 LMCache CPU backend 启动。
- CPU RSS 明显可采样。
- 可与 vLLM baseline 对齐 case。

停止服务。

## 12. Step 6：正式 LMCache disk baseline

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
export LMCACHE_LOCAL_DISK=results/gpu/lmcache_disk_store
bash scripts/launch_lmcache_vllm.sh disk 2>&1 | tee results/gpu/logs/p0_3_lmcache_disk_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_disk.yaml \
  --output-dir results/gpu/p0_3_lmcache_disk
```

记录 `<DISK_RUN>`：

```bash
find results/gpu/p0_3_lmcache_disk -maxdepth 2 -name benchmark_results.csv
```

验收标准：

- benchmark 正常生成。
- server log 证明 LMCache disk backend 启动。
- disk store 目录有数据或日志证明 disk tier 被配置。
- disk read/write 指标可采样。

停止服务。

## 13. Step 7：提取 cache events

CPU backend：

```bash
python scripts/extract_cache_events.py \
  --server-log results/gpu/logs/p0_2_lmcache_cpu_server.log \
  --request-results <CPU_RUN>/request_results.jsonl \
  --benchmark-results <CPU_RUN>/benchmark_results.csv \
  --output-dir results/gpu/p0_5_cache_events_cpu
```

Disk backend：

```bash
python scripts/extract_cache_events.py \
  --server-log results/gpu/logs/p0_3_lmcache_disk_server.log \
  --request-results <DISK_RUN>/request_results.jsonl \
  --benchmark-results <DISK_RUN>/benchmark_results.csv \
  --output-dir results/gpu/p0_5_cache_events_disk
```

验收标准：

- `cache_events.jsonl`
- `cache_event_summary.md`
- 至少能看到 benchmark/request/memory 相关事件。
- 如果 server log 中能解析出 hit/load/store/offload，要在报告中保留证据。

注意：

- 如果 `cache_events.jsonl` 主要来自 benchmark/sample，而没有真实 cache hit/load/store/offload，不能声称真实 KV hit rate。
- 这种情况下只能说“已完成采集链路，真实 cache event 需进一步打开 LMCache/vLLM 日志”。

## 14. Step 8：baseline 对比

```bash
python scripts/compare_real_runs.py \
  --run vllm=<VLLM_RUN> \
  --run lmcache_cpu=<CPU_RUN> \
  --run lmcache_disk=<DISK_RUN> \
  --output-dir results/gpu/p0_6_comparison
```

验收标准：

- `comparison_results.csv`
- `comparison_report.md`

重点看：

- GPU memory reduction
- CPU memory increase
- SSD traffic
- TTFT delta
- TPOT delta
- latency p95 delta
- throughput delta
- success rate

## 15. Step 9：memory-constrained stress benchmark

stress 需要分别跑三组。

### 15.1 vLLM stress

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
bash scripts/launch_vllm_server.sh 2>&1 | tee results/gpu/logs/p0_7_stress_vllm_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/stress_vllm_memory_constrained.yaml \
  --output-dir results/gpu/p0_7_stress_vllm
```

停止服务。

### 15.2 LMCache CPU stress

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
bash scripts/launch_lmcache_vllm.sh cpu 2>&1 | tee results/gpu/logs/p0_7_stress_cpu_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/stress_lmcache_cpu_memory_constrained.yaml \
  --output-dir results/gpu/p0_7_stress_cpu
```

停止服务。

### 15.3 LMCache disk stress

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
export LMCACHE_LOCAL_DISK=results/gpu/lmcache_disk_store
bash scripts/launch_lmcache_vllm.sh disk 2>&1 | tee results/gpu/logs/p0_7_stress_disk_server.log
```

终端 B：

```bash
python scripts/run_real_benchmark.py \
  --config configs/stress_lmcache_disk_memory_constrained.yaml \
  --output-dir results/gpu/p0_7_stress_disk
```

停止服务。

### 15.4 汇总 stress

记录 `<STRESS_VLLM_RUN>`、`<STRESS_CPU_RUN>`、`<STRESS_DISK_RUN>` 后执行：

```bash
python scripts/analyze_stress_results.py \
  --run vllm=<STRESS_VLLM_RUN> \
  --run lmcache_cpu=<STRESS_CPU_RUN> \
  --run lmcache_disk=<STRESS_DISK_RUN> \
  --output-dir results/gpu/p0_7_stress_analysis
```

验收标准：

- `stress_summary.csv`
- `stress_report.md`

重点看：

- max successful context
- max successful batch
- OOM rate
- success rate
- process RSS peak
- GPU utilization
- disk IO
- startup-level KV cache capacity
- latency p95

## 16. Step 10：real selective prefetch

建议使用 LMCache CPU 或 disk backend。

终端 A：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
bash scripts/launch_lmcache_vllm.sh cpu 2>&1 | tee results/gpu/logs/p0_8_lmcache_cpu_prefetch_server.log
```

终端 B 先跑 smoke：

```bash
python scripts/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/gpu/p0_8_selective_prefetch_smoke \
  --context-lengths 1024 \
  --repeat 1 \
  --output-tokens 32 \
  --warmup-output-tokens 1
```

确认成功后跑正式：

```bash
python scripts/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/gpu/p0_8_selective_prefetch
```

提取 prefetch 期间 cache events：

```bash
python scripts/extract_cache_events.py \
  --server-log results/gpu/logs/p0_8_lmcache_cpu_prefetch_server.log \
  --output-dir results/gpu/p0_8_cache_events
```

验收标准：

- `prefetch_results.csv`
- `prefetch_events.jsonl`
- `prefetch_report.md`
- `cache_events.jsonl`

注意：

- `prefetch_submitted` 表示真的向 endpoint 发出了 warmup/prefetch 请求。
- `prefetch_completed` 表示后端完成了这个请求。
- `prefetch_hit` 如果只是 TTFT/latency 改善，是 heuristic。
- 只有 cache log 证明 hit/load/store，才能强声称真实 KV cache hit。

## 17. Step 11：构建 unified trace store

```bash
python scripts/build_trace_store.py \
  --cache-events results/gpu/p0_5_cache_events_cpu/cache_events.jsonl \
  --cache-events results/gpu/p0_5_cache_events_disk/cache_events.jsonl \
  --prefetch-events results/gpu/p0_8_selective_prefetch/prefetch_events.jsonl \
  --samples <VLLM_RUN>/samples \
  --samples <CPU_RUN>/samples \
  --samples <DISK_RUN>/samples \
  --output-dir results/gpu/p1_1_trace_store
```

验收标准：

- `trace_events.jsonl`
- `trace_summary.md`

## 18. Step 12：构建 ProfileDB

```bash
python scripts/build_profile_db.py \
  --trace-events results/gpu/p1_1_trace_store/trace_events.jsonl \
  --workload-id gpu_qwen25_7b \
  --output-dir results/gpu/p1_2_profile_db
```

验收标准：

- `profile_db.json`
- `profile_db_report.md`

## 19. Step 13：Chunk Scoring

先使用默认 pressure：

```bash
python scripts/score_chunks.py \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --memory-pressure 0.50 \
  --output-dir results/gpu/p1_3_chunk_scores
```

验收标准：

- `chunk_scores.csv`
- `chunk_score_report.md`
- 可以看到 `prefetch` / `keep` / `offload` / `drop` 决策。

## 20. Step 14：Memory Pressure Controller

根据真实 benchmark 和 samples 计算压力：

```bash
python scripts/analyze_memory_pressure.py \
  --benchmark-results <VLLM_RUN>/benchmark_results.csv \
  --benchmark-results <CPU_RUN>/benchmark_results.csv \
  --benchmark-results <DISK_RUN>/benchmark_results.csv \
  --samples <VLLM_RUN>/samples \
  --samples <CPU_RUN>/samples \
  --samples <DISK_RUN>/samples \
  --gpu-capacity-mb 80000 \
  --cpu-capacity-mb 262144 \
  --output-dir results/gpu/memory_pressure
```

如果你的 GPU 不是 80GB，请修改：

```text
--gpu-capacity-mb
```

常见取值：

| GPU | `--gpu-capacity-mb` |
| --- | ---: |
| 24GB | 24576 |
| 40GB | 40960 |
| 48GB | 49152 |
| 80GB | 81920 |
| 141GB | 144384 |

验收标准：

- `memory_pressure_decisions.csv`
- `memory_pressure_hints.jsonl`
- `memory_pressure_report.md`
- `memory_pressure_manifest.json`

从 `memory_pressure_decisions.csv` 中取最大 `memory_pressure` 分数，记为 `<PRESSURE>`。

然后重跑 chunk scoring：

```bash
python scripts/score_chunks.py \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --memory-pressure <PRESSURE> \
  --output-dir results/gpu/p1_3_chunk_scores_pressure
```

## 21. Step 15：Partial KV Load Planner

```bash
python scripts/plan_partial_kv_load.py \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --output-dir results/gpu/p1_5_partial_kv
```

验收标准：

- `partial_kv_plan.csv`
- `partial_kv_summary.csv`
- `partial_kv_report.md`

注意：

- 这是 adapter-facing plan。
- 如果没有真实 backend 消费这个 plan，不能声称“物理 partial KV load 已发生”。
- 可以声称“partial KV load planning and byte-saving estimation”。

## 22. Step 16：Load-vs-Recompute Planner

```bash
python scripts/decide_load_vs_recompute.py \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --partial-plan results/gpu/p1_5_partial_kv/partial_kv_plan.csv \
  --memory-pressure <PRESSURE> \
  --output-dir results/gpu/p1_6_load_vs_recompute
```

验收标准：

- `load_recompute_decisions.csv`
- `load_recompute_hints.jsonl`
- `load_recompute_report.md`

注意：

- 这是被动决策和 hint。
- 如果没有 backend 执行 recompute，不能声称真实 recompute 已发生。

## 23. Step 17：Unified Object Scheduler

```bash
python scripts/run_unified_object_scheduler.py \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --chunk-scores results/gpu/p1_3_chunk_scores_pressure/chunk_scores.csv \
  --load-recompute-decisions results/gpu/p1_6_load_vs_recompute/load_recompute_decisions.csv \
  --memory-pressure <PRESSURE> \
  --gpu-budget-bytes 8589934592 \
  --output-dir results/gpu/p1_9_unified_scheduler
```

`--gpu-budget-bytes` 可以按实验目标调整：

| 目标 | 建议 |
| --- | ---: |
| 极端受限 | 2GB 到 4GB |
| 中等受限 | 8GB |
| 宽松 | 16GB 到 32GB |

验收标准：

- `object_schedule_decisions.csv`
- `object_scheduler_hints.jsonl`
- `object_scheduler_report.md`

注意：

- 这是 unified object scheduling hint。
- 如果没有 adapter 消费 hint，不能声称真实对象移动已发生。

## 24. Step 18：Quality / PPL Evaluation

如果 benchmark 输出中有 baseline 和 variant 的 `request_results.jsonl`，可以先做输出一致性：

```bash
python scripts/evaluate_quality.py \
  --baseline <VLLM_RUN>/request_results.jsonl \
  --variant <CPU_RUN>/request_results.jsonl \
  --output-dir results/gpu/quality_lmcache_cpu

python scripts/evaluate_quality.py \
  --baseline <VLLM_RUN>/request_results.jsonl \
  --variant <DISK_RUN>/request_results.jsonl \
  --output-dir results/gpu/quality_lmcache_disk
```

验收标准：

- `quality_results.csv`
- `quality_summary.md`
- output match rate
- token divergence
- char divergence

PPL 只有在输入记录包含 `ppl`、`loss` 或 `nll` 字段时才会计算。

## 25. Step 19：可选 CKA / hidden-state drift

只有当你额外导出 hidden states 时执行：

```bash
python scripts/evaluate_hidden_state_drift.py \
  --baseline results/gpu/hidden_states/baseline.jsonl \
  --variant results/gpu/hidden_states/variant.jsonl \
  --output-dir results/gpu/cka
```

验收标准：

- `hidden_state_drift_results.csv`
- `hidden_state_drift_report.md`
- CKA
- cosine similarity
- MSE
- L2 drift

如果没有 hidden state 导出，这一步跳过，不影响最短参赛路径。

## 26. Step 20：可选 MoE 实验

MoE 是加分项，不是最短参赛路径。

### 26.1 提取 expert route events

如果你有 MoE router log：

```bash
python scripts/extract_moe_expert_events.py \
  --router-log results/gpu/logs/moe_router.log \
  --output-dir results/gpu/moe_events
```

验收标准：

- `moe_expert_events.jsonl`
- `moe_expert_summary.csv`
- `moe_expert_report.md`

### 26.2 规划 expert loading

```bash
python scripts/plan_moe_expert_loading.py \
  --expert-summary results/gpu/moe_events/moe_expert_summary.csv \
  --output-dir results/gpu/moe_load_plan
```

### 26.3 预测 experts

默认 next-token：

```bash
python scripts/predict_moe_experts.py \
  --moe-events results/gpu/moe_events/moe_expert_events.jsonl \
  --predictor-name next_token \
  --output-dir results/gpu/moe_predict_next_token
```

history-window：

```bash
python scripts/predict_moe_experts.py \
  --moe-events results/gpu/moe_events/moe_expert_events.jsonl \
  --predictor-name history_window \
  --history-window 4 \
  --history-window-weight 0.25 \
  --transition-weight 0.35 \
  --output-dir results/gpu/moe_predict_history_window
```

profile-guided：

```bash
python scripts/predict_moe_experts.py \
  --moe-events results/gpu/moe_events/moe_expert_events.jsonl \
  --expert-load-plan results/gpu/moe_load_plan/moe_expert_load_plan.csv \
  --predictor-name profile_guided \
  --history-window 4 \
  --history-window-weight 0.25 \
  --transition-weight 0.35 \
  --load-plan-weight 0.25 \
  --gpu-resident-bonus 0.15 \
  --output-dir results/gpu/moe_predict_profile_guided
```

注意：

- 如果只有 route trace 和 prediction，可以声称 expert activation analysis 和 expert prediction。
- 不能声称 expert weights 已经真实 selective loading，除非 runtime log 证明 expert weights 被加载/迁移/命中。

## 27. Step 21：生成最终 competition report

根据实际存在的 artifact 填入路径。没有的可以先省略，或者保留让 report 标 missing。

```bash
python scripts/build_competition_report.py \
  --benchmark vllm=<VLLM_RUN>/benchmark_results.csv \
  --benchmark lmcache_cpu=<CPU_RUN>/benchmark_results.csv \
  --benchmark lmcache_disk=<DISK_RUN>/benchmark_results.csv \
  --comparison results/gpu/p0_6_comparison/comparison_results.csv \
  --stress results/gpu/p0_7_stress_analysis/stress_summary.csv \
  --trace-summary results/gpu/p1_1_trace_store/trace_summary.md \
  --profile-db results/gpu/p1_2_profile_db/profile_db.json \
  --chunk-scores results/gpu/p1_3_chunk_scores_pressure/chunk_scores.csv \
  --partial-load results/gpu/p1_5_partial_kv/partial_kv_summary.csv \
  --load-recompute results/gpu/p1_6_load_vs_recompute/load_recompute_decisions.csv \
  --object-schedule results/gpu/p1_9_unified_scheduler/object_schedule_decisions.csv \
  --quality results/gpu/quality_lmcache_cpu/quality_results.csv \
  --quality results/gpu/quality_lmcache_disk/quality_results.csv \
  --workload-manifest astrakv/benchmarks/prompts/competition_workload_manifest.json \
  --command "python cli.py test" \
  --output-dir results/gpu/competition_report
```

验收标准：

- `competition_report.md`
- `competition_report_manifest.json`
- `artifact_inventory.csv`

`artifact_inventory.csv` 会告诉你哪些证据缺失。

## 28. Step 22：生成 demo dashboard

```bash
python scripts/build_demo_dashboard.py \
  --competition-report results/gpu/competition_report/competition_report.md \
  --comparison results/gpu/p0_6_comparison/comparison_results.csv \
  --stress results/gpu/p0_7_stress_analysis/stress_summary.csv \
  --output-dir results/gpu/demo_dashboard
```

验收标准：

- `dashboard.html`
- `dashboard_data.json`
- `dashboard_manifest.json`

## 29. 最短参赛路径

如果时间有限，只跑这些：

```text
1. Step 0 环境信息
2. Step 1 vLLM smoke
3. Step 2 LMCache CPU smoke
4. Step 3 LMCache disk smoke
5. Step 4 正式 vLLM baseline
6. Step 5 正式 LMCache CPU baseline
7. Step 6 正式 LMCache disk baseline
8. Step 7 cache event extraction
9. Step 8 baseline comparison
10. Step 9 stress benchmark
11. Step 14 memory pressure controller
12. Step 21 final competition report
```

这条路径可以支撑的比赛叙事：

```text
vLLM-only baseline
-> LMCache CPU tier
-> LMCache disk tier
-> memory-constrained stress
-> GPU/CPU/SSD sampling
-> memory pressure analysis
-> performance-memory tradeoff report
```

## 30. 最优获奖路径

如果时间充足，跑这些：

```text
1. Step 0 到 Step 18 全部执行
2. Step 19 如果能导出 hidden states，就做 CKA
3. Step 20 如果有 MoE 模型和 route log，就做 MoE
4. Step 21 final competition report
5. Step 22 demo dashboard
```

这条路径可以支撑的比赛叙事：

```text
真实 vLLM/LMCache baseline
-> 真实 CPU/disk memory tiering
-> cache event trace
-> ProfileDB
-> pressure-aware chunk scoring
-> partial KV load planning
-> load-vs-recompute planning
-> unified object scheduler hints
-> endpoint selective prefetch
-> quality/CKA/MoE evidence
-> competition dashboard
```

## 31. 每一步必须保留的证据

| 证据 | 文件 |
| --- | --- |
| 服务器信息 | `results/gpu/env/*` |
| server logs | `results/gpu/logs/*.log` |
| benchmark metrics | `benchmark_results.csv` |
| request outputs | `request_results.jsonl` |
| continuous samples | `samples/*_samples.csv` |
| benchmark report | `benchmark_report.md` |
| cache events | `cache_events.jsonl` |
| cache summary | `cache_event_summary.md` |
| comparison | `comparison_report.md` |
| stress | `stress_report.md` |
| selective prefetch | `prefetch_report.md` |
| trace store | `trace_events.jsonl` |
| ProfileDB | `profile_db.json` |
| pressure controller | `memory_pressure_report.md` |
| scheduler hints | `*_hints.jsonl` |
| quality | `quality_summary.md` |
| final report | `competition_report.md` |
| dashboard | `dashboard.html` |

## 32. 哪些结论可以说，哪些不能说

### 可以说

如果对应 artifact 存在，可以说：

- 在真实 vLLM endpoint 上完成 baseline。
- 在真实 vLLM + LMCache CPU backend 上完成 baseline。
- 在真实 vLLM + LMCache disk backend 上完成 baseline。
- 采集了 TTFT、TPOT、latency、throughput、GPU memory、CPU memory、SSD traffic。
- 在 memory-constrained 设置下对比了成功率、OOM rate、max context、max batch。
- 构建了 cache event trace、ProfileDB、chunk scoring、memory pressure controller。
- 生成了 partial KV load / load-vs-recompute / unified scheduler 的 adapter-facing hints。
- endpoint-level selective prefetch 发出了真实 warmup 请求。

### 必须谨慎说

- KV hit rate：必须有 server log 或 cache event 支撑。
- prefetch hit rate：如果只有 latency/TTFT 改善，只能说 heuristic。
- partial KV load：如果没有 backend 消费 plan，只能说 planning/estimation。
- load-vs-recompute：如果没有真实 recompute runtime，只能说 scheduler decision。
- unified scheduler：如果没有 adapter 消费 hints，只能说 passive scheduling hints。
- expert hit rate：必须有真实 MoE route events。
- expert loading：必须有 runtime log 证明 expert weights 被加载/迁移/命中。
- PPL：必须有 `ppl/loss/nll` 字段或真实 eval pipeline。
- CKA：必须有 hidden states 导出。

## 33. 常见问题

### 33.1 vLLM 起不来

检查：

```bash
nvidia-smi
echo $ASTRAKV_MODEL
echo $ASTRAKV_MAX_MODEL_LEN
echo $ASTRAKV_GPU_MEMORY_UTILIZATION
```

降低：

```bash
export ASTRAKV_MAX_MODEL_LEN=4096
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.65
```

### 33.2 LMCache 没有启动

检查：

```bash
cat results/gpu/logs/p0_2_lmcache_cpu_server.log | grep -i lmcache
cat results/gpu/logs/p0_3_lmcache_disk_server.log | grep -i lmcache
```

确认：

- `LMCACHE_CONFIG_FILE` 正确。
- `--kv-transfer-config` 被传给 vLLM。
- 当前 vLLM/LMCache 版本支持 `LMCacheConnectorV1`。

### 33.3 disk traffic 为空

可能原因：

- Linux disk device 自动识别失败。
- smoke workload 太小。
- LMCache disk backend 没有真正产生磁盘读写。

处理：

- 在 config 的 `metrics.disk_device` 填具体设备名，例如 `nvme0n1`。
- 增大 context/repeat。
- 检查 `results/gpu/lmcache_disk_store` 是否有实际写入。

### 33.4 8192 context OOM

这是可接受的 memory-constrained 证据。记录：

- 哪个 backend OOM。
- 哪个 context/batch OOM。
- LMCache CPU/disk 是否提高 max context 或 success rate。

### 33.5 prefetch hit 不明显

先检查：

- workload 是否有 repeated prefix。
- LMCache 是否真的启用。
- server log 是否有 cache store/load/hit。
- repeat 是否太小。

可以增加：

```bash
--repeat 5
--context-lengths 2048 4096
```

## 34. 最终提交前检查清单

```text
[ ] results/gpu/env 有完整环境信息
[ ] vLLM baseline 完成
[ ] LMCache CPU baseline 完成
[ ] LMCache disk baseline 完成
[ ] 三组 benchmark 都有 samples
[ ] server logs 已保存
[ ] cache events 已提取
[ ] comparison report 已生成
[ ] stress report 已生成
[ ] memory pressure report 已生成
[ ] trace store 已生成
[ ] ProfileDB 已生成
[ ] chunk scores 已生成
[ ] partial KV plan 已生成
[ ] load-vs-recompute decisions 已生成
[ ] unified scheduler hints 已生成
[ ] selective prefetch report 已生成
[ ] quality report 已生成
[ ] competition report 已生成
[ ] demo dashboard 已生成
[ ] artifact_inventory.csv 中关键 P0 项不是 missing
[ ] 报告中没有把 passive hints 夸大成真实 runtime 生效
```

## 35. 推荐先跑的命令列表

第一次上 GPU，不要跑完整矩阵，先按这个最小序列：

```bash
pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))"

export ASTRAKV_MODEL=Qwen/Qwen2.5-7B-Instruct
export ASTRAKV_HOST=127.0.0.1
export ASTRAKV_PORT=8000
export ASTRAKV_MAX_MODEL_LEN=8192
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.72
export LMCACHE_LOCAL_DISK=results/gpu/lmcache_disk_store

mkdir -p results/gpu/logs results/gpu/lmcache_disk_store
```

然后依次跑：

```text
Step 1 vLLM smoke
Step 2 LMCache CPU smoke
Step 3 LMCache disk smoke
```

三步都通过，再开始正式实验。

## 36. 参考链接

- vLLM GPU installation: <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- LMCache quickstart: <https://docs.lmcache.ai/getting_started/quickstart.html>
- LMCache vLLM integration: <https://docs.lmcache.ai/developer_guide/integration.html>
- Qwen2.5-7B-Instruct model card: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
