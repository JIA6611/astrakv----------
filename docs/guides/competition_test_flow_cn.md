# 比赛提测流程说明

本文档用于组织 AstraKV-W 面向“内存受限环境的大语言模型推理优化”赛题的提测流程。目标是回答三个问题：

1. 为了覆盖赛题要求，需要跑哪些测试。
2. 每类测试产出什么证据。
3. AstraKV-W 如何和 baseline 设备 / baseline runtime 做公平对比。

## 0. 赛题要求到测试项的映射

| 赛题要求 | 本项目测试项 | 主要产物 |
| --- | --- | --- |
| 分析 LLM 推理访存行为，包括参数加载、KV cache、专家激活 | vLLM / LMCache benchmark、cache event 提取、memory sample、ProfileDB | `benchmark_results.csv`、`request_results.jsonl`、`samples/*.csv`、`cache_events.jsonl` |
| 使用虚拟内存相关技术，通过按需加载、数据换出减少物理内存占用 | LMCache CPU / Disk 分层基线、mmap VM evidence、memory pressure 分析 | LMCache server log、disk read/write 指标、`vm_evidence/`、memory pressure report |
| 通过预取掩盖 IO 延迟，提高推理性能 | AstraKV endpoint-level selective prefetch / warmup | `prefetch_results.csv`、`prefetch_events.jsonl`、`prefetch_report.md` |
| 体现嵌入式 / 边缘 / 内存受限场景 | DGX Spark 低 `gpu_memory_utilization` stress 测试 | stress benchmark、OOM / error rate、最大成功 context / batch |
| 强调整体系统观，不只优化单一指标 | baseline、LMCache、AstraKV、stress、报告统一对比 | `comparison_report.md`、`policy_ablation_report.md`、competition report |

## 1. 提测分几步

建议把提测分成 7 步。每一步都能独立证明一个能力，最后用对比报告串起来。

| 步骤 | 名称 | 是否必须 | 目的 |
| --- | --- | --- | --- |
| Step 1 | 环境与本地正确性测试 | 必须 | 证明代码、依赖、GPU 环境可用 |
| Step 2 | OS 虚拟内存 PoC 测试 | 必须 | 证明项目具备 VM / mmap 机制证据 |
| Step 3 | 真实 vLLM baseline | 必须 | 建立未启用 offload / AstraKV 的真实服务基线 |
| Step 4 | LMCache CPU / Disk baseline | 必须 | 建立应用层 KV 分层 / 换出基线 |
| Step 5 | AstraKV selective prefetch | 必须 | 用本项目架构对同一 backend 做预取优化对比 |
| Step 6 | 内存受限 stress 测试 | 建议必须 | 证明低内存 headroom 下的可用性、失败率和容量 |
| Step 7 | 结果对比与报告 | 必须 | 汇总 baseline vs AstraKV 的收益和代价 |

### 1.1 当前推荐一键入口

如果要复现当前最终证据链，优先使用扩展证据脚本。它会在主线 E2E 基础上补充 32K boundary、cache events、OS VM evidence、quality、policy chain、final report 和 archive：

```bash
cd /home/szl/Desktop/Inference-OS
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"

bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --skip-install \
  --continue-on-failure
```

默认写入：

```text
results/extended_evidence_<timestamp>/
  01_e2e/
  02_boundary_32k/
  03_cache_events/
  04_os_vm/
  05_quality/
  06_policy_chain/
  07_final_report/
  archive/
```

当前建议作为提交证据保留的结果目录：

- `results/extended_evidence_20260625_014917`：完整主线证据。
- `results/extended_g016_ctx32k_b16_out256`：`gpu_util=0.16` 的 32K 极限可运行证据。
- `results/extended_g015_ctx32k_b16_out256`：`gpu_util=0.15` 的 32K 启动失败下界。

仓库整理、可删除候选、结果目录保留策略见：

```text
docs/guides/repository_organization_cn.md
```

### 1.2 主线 E2E 入口

如果只是要按当前推荐流程完整提测，优先使用一键脚本：

```bash
cd /home/szl/Desktop/Inference-OS
bash scripts/entrypoints/run_competition_e2e.sh --skip-install
```

快速验证脚本和环境：

```bash
bash scripts/entrypoints/run_competition_e2e.sh --only smoke --skip-install
```

只跑更强的内存受限边界测试：

```bash
bash scripts/entrypoints/run_competition_e2e.sh --only extreme --gpu-util-extreme 0.40 --skip-install
```

一键脚本默认写入新的 timestamp 目录，例如 `results/competition_e2e_<timestamp>/`，不会覆盖已有结果。默认不需要 `sudo`，也不默认创建 cgroup；`--with-cgroup` 只作为显式增强入口，真实主机内存限制应由外部 cgroup runner 或管理员配置完成。

默认执行内容是 `official + extreme + report`。其中 official 用 `gpu_memory_utilization=0.60`，extreme 默认用更保守的 `0.45`。如果 0.45 全部成功，可以手动降到 0.40 或 0.35；如果 0.45 连服务都无法启动，可以升到 0.50。extreme 中出现 OOM、启动失败或部分 case 失败不是脚本失败，而是“边界证据”，需要进入 stress summary 和最终报告。

DGX Spark 上的 GPU memory 证据需要按平台能力解释：case-level `gpu_memory_peak_mb` 只作为兼容字段保留，只有 `nvidia-smi` 或 NVML 真能采样时才填写；如果这些接口返回 `[N/A]` 或 Not Supported，就保持空值，不把它作为结论指标。主要资源指标改用 `process_rss_peak_mb`、`gpu_util_peak_pct`、`disk_read_delta_mb`、`disk_write_delta_mb`、`sample_count`，再结合 vLLM server log 中的 `Model loading took ... GiB memory`、`Available KV cache memory ... GiB`、`GPU KV cache size ... tokens` 作为 startup-level KV capacity evidence。

### 1.3 32K 极限边界复现

当前不要继续用 49K context 压测 Qwen2.5-7B-Instruct，因为该模型配置的 `max_position_embeddings` 是 32768。49K 失败只能证明模型上下文上限，不是内存边界。

可运行上界：

```bash
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only boundary \
  --gpu-util-boundary 0.16 \
  --boundary-max-model-len 32768 \
  --boundary-context-lengths "24576 32768" \
  --boundary-batch-sizes "4 8 12 16" \
  --boundary-output-tokens 256 \
  --boundary-repeat 1 \
  --boundary-timeout 2400 \
  --output-root results/extended_g016_ctx32k_b16_out256 \
  --skip-install \
  --continue-on-failure
```

失败下界：

```bash
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only boundary \
  --gpu-util-boundary 0.15 \
  --boundary-max-model-len 32768 \
  --boundary-context-lengths "24576 32768" \
  --boundary-batch-sizes "4 8 12 16" \
  --boundary-output-tokens 256 \
  --boundary-repeat 1 \
  --boundary-timeout 2400 \
  --output-root results/extended_g015_ctx32k_b16_out256 \
  --skip-install \
  --continue-on-failure
```

解释口径：

- `0.15` 时 vLLM 可用 KV cache 约 `1.67 GiB`，低于 32K 启动至少需要的 `1.75 GiB`，所以启动失败。
- `0.16` 时 vLLM / LMCache CPU / LMCache Disk 均能完成 `32768 context / batch 16 / output 256`。
- 这两组共同证明当前实验确实压到了内存受限边界。

## 2. 对比原则

为了让 AstraKV-W 和 baseline 的比较可信，必须保证以下变量一致：

- 同一台 DGX Spark / 同一 GPU。
- 同一个模型：默认 `Qwen/Qwen2.5-7B-Instruct`。
- 同一套 context length、output tokens、repeat。
- 同一套 prompt 生成逻辑或同一 workload。
- 同一个 endpoint 地址和同一 vLLM / LMCache 版本。
- 只改变 backend 或策略：
  - `vllm`: plain vLLM baseline。
  - `lmcache_cpu`: vLLM + LMCache CPU tier。
  - `lmcache_disk`: vLLM + LMCache Disk tier。
  - `astrakv_prefetch`: vLLM + LMCache + AstraKV endpoint-level selective prefetch。

当前 AstraKV-W 的真实优化路径是 endpoint-level selective prefetch / warmup，不声称已经改写 vLLM 内部 KV block scheduler。报告中应明确区分：

- 真实 backend 对比：vLLM、LMCache、AstraKV warmup。
- OS VM 机制证据：mmap / VM PoC。
- 策略层证据：chunk scorer、ProfileDB、scheduler hint。

## 3. Step 1：环境与本地正确性测试

安装环境：

```bash
cd /home/szl/Desktop/Inference-OS
bash scripts/entrypoints/bootstrap_dgx_spark_env.sh --with-lmcache
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON=python
```

检查 GPU 和依赖：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))"
```

跑单元测试：

```bash
python cli.py test
```

一键 DGX Spark validation：

```bash
bash scripts/entrypoints/run_dgx_spark_validation.sh --skip-install
```

产物：

- `results/dgx_spark_validation/environment_report.txt`
- `results/dgx_spark_validation/pytest.log`
- `results/dgx_spark_validation/vm_evidence/`
- `results/dgx_spark_validation/mmap_smoke/`

判定标准：

- `torch.cuda.is_available()` 为 `True`。
- 单元测试通过。
- 生成环境报告和 VM evidence。

## 4. Step 2：OS 虚拟内存 PoC 测试

这一步不要求 vLLM 启动，用于证明“虚拟内存相关技术”的机制证据。

```bash
python cli.py vm mmap \
  --blocks 100 \
  --block-size-mb 1 \
  --output-dir results/step2_vm_mmap

python scripts/vm/run_dgx_spark_vm_evidence.py \
  --output-dir results/step2_dgx_vm_evidence \
  --chunks 8 \
  --total-blocks 64 \
  --block-size-mb 1
```

产物：

- mmap access trace
- RSS / latency / demand-load-like 指标
- VM evidence report

报告解释：

- 这一步证明 AstraKV-W 有 OS VM / mmap 机制 PoC。
- 它不等同于真实 vLLM KV cache 已经由 mmap 管理。
- 真实 runtime 证据需要结合 Step 3 到 Step 6。

## 5. Step 3：真实 vLLM Baseline

先准备模型。本地路径推荐：

```bash
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
```

终端 A 启动 plain vLLM：

```bash
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON=python
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60

bash scripts/launch/launch_vllm_server.sh 2>&1 | tee results/step3_vllm_server.log
```

终端 B 先跑 smoke：

```bash
curl http://127.0.0.1:8000/v1/models

python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/step3_vllm_smoke \
  --context-lengths 512 \
  --batch-sizes 1 \
  --output-tokens 32 \
  --repeat 1
```

smoke 成功后跑正式 baseline：

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/step3_vllm_baseline
```

产物：

- `benchmark_results.csv`
- `benchmark_report.md`
- `request_results.jsonl`
- `samples/<case>_samples.csv`
- `charts/`
- `results/step3_vllm_server.log`

关注指标：

- success rate
- TTFT / TPOT / latency p50 / p95
- throughput
- process RSS peak
- GPU utilization
- Disk read / write
- vLLM startup KV cache capacity evidence

## 6. Step 4：LMCache CPU / Disk Baseline

这一步用于建立“已有 offload / tiering 方案”的 baseline。AstraKV-W 之后必须和它对比，而不是只和 plain vLLM 对比。

### 6.1 LMCache CPU

```bash
pkill -f vllm || true
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON=python
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60

bash scripts/launch/launch_lmcache_vllm.sh cpu 2>&1 | tee results/step4_lmcache_cpu_server.log
```

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_cpu.yaml \
  --output-dir results/step4_lmcache_cpu
```

### 6.2 LMCache Disk

```bash
pkill -f vllm || true
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON=python
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
export LMCACHE_LOCAL_DISK=results/lmcache_disk_store
export LMCACHE_DISK_PATH="$LMCACHE_LOCAL_DISK"

bash scripts/launch/launch_lmcache_vllm.sh disk 2>&1 | tee results/step4_lmcache_disk_server.log
```

```bash
python scripts/benchmark/run_real_benchmark.py \
  --config configs/dgx_spark_lmcache_disk.yaml \
  --output-dir results/step4_lmcache_disk
```

提取 cache event：

```bash
python scripts/benchmark/extract_cache_events.py \
  --server-log results/step4_lmcache_cpu_server.log \
  --output-dir results/step4_cache_events_cpu

python scripts/benchmark/extract_cache_events.py \
  --server-log results/step4_lmcache_disk_server.log \
  --output-dir results/step4_cache_events_disk
```

关注指标：

- vLLM startup log 中可用 KV cache memory / KV cache tokens 是否变化。
- process RSS 是否上升。
- Disk read / write 是否有变化。
- GPU utilization 是否稳定。
- TTFT / TPOT / throughput 代价是多少。
- server log 是否证明 LMCache backend 生效。

## 7. Step 5：AstraKV Selective Prefetch 测试

AstraKV-W 当前真实对比路径是：

```text
vLLM + LMCache baseline
vs
vLLM + LMCache + AstraKV endpoint-level selective prefetch / warmup
```

推荐先用 LMCache CPU backend 跑，因为 CPU tier 方便观察 KV reuse 行为。

终端 A 启动 LMCache CPU：

```bash
pkill -f vllm || true
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_PYTHON=python
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60

bash scripts/launch/launch_lmcache_vllm.sh cpu 2>&1 | tee results/step5_astrakv_lmcache_cpu_server.log
```

终端 B 跑 AstraKV selective prefetch：

```bash
python scripts/benchmark/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/step5_astrakv_prefetch_smoke \
  --context-lengths 1024 \
  --repeat 1 \
  --output-tokens 32 \
  --warmup-output-tokens 1
```

正式运行：

```bash
python scripts/benchmark/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/step5_astrakv_prefetch_full
```

提取并附加 cache event：

```bash
python scripts/benchmark/extract_cache_events.py \
  --server-log results/step5_astrakv_lmcache_cpu_server.log \
  --output-dir results/step5_astrakv_cache_events

python scripts/benchmark/run_selective_prefetch_real.py \
  --config configs/astrakv_real_selective_prefetch.yaml \
  --output-dir results/step5_astrakv_prefetch_with_events \
  --cache-events results/step5_astrakv_cache_events/cache_events.jsonl
```

产物：

- `prefetch_results.csv`
- `prefetch_events.jsonl`
- `prefetch_report.md`
- `prefetch_config.json`
- `cache_events.jsonl`

如何解读：

- no-prefetch demand 是 AstraKV 内部控制组。
- prefetch request 是 AstraKV 发出的 endpoint warmup。
- prefetch-demand 是 warmup 后的真实请求。
- 如果 prefetch-demand 的 TTFT / latency 低于 no-prefetch，说明 warmup 有效果。
- 如果 cache event 里能看到 hit / load / store，则证据更强。

## 8. Step 6：内存受限 Stress 测试

这一步回应“内存受限系统”和“边缘 / 嵌入式场景”要求。

统一设置：

```bash
export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
```

分别跑：

```bash
# plain vLLM
bash scripts/launch/launch_vllm_server.sh 2>&1 | tee results/step6_stress_vllm_server.log
python scripts/benchmark/run_real_benchmark.py \
  --config configs/stress_vllm_memory_constrained.yaml \
  --output-dir results/step6_stress_vllm

# LMCache CPU
pkill -f vllm || true
bash scripts/launch/launch_lmcache_vllm.sh cpu 2>&1 | tee results/step6_stress_cpu_server.log
python scripts/benchmark/run_real_benchmark.py \
  --config configs/stress_lmcache_cpu_memory_constrained.yaml \
  --output-dir results/step6_stress_cpu

# LMCache Disk
pkill -f vllm || true
export LMCACHE_LOCAL_DISK=results/lmcache_disk_store
export LMCACHE_DISK_PATH="$LMCACHE_LOCAL_DISK"
bash scripts/launch/launch_lmcache_vllm.sh disk 2>&1 | tee results/step6_stress_disk_server.log
python scripts/benchmark/run_real_benchmark.py \
  --config configs/stress_lmcache_disk_memory_constrained.yaml \
  --output-dir results/step6_stress_disk
```

汇总：

```bash
python scripts/reporting/analyze_stress_results.py \
  --run vllm=results/step6_stress_vllm/<RUN_DIR> \
  --run lmcache_cpu=results/step6_stress_cpu/<RUN_DIR> \
  --run lmcache_disk=results/step6_stress_disk/<RUN_DIR> \
  --output-dir results/step6_stress_analysis
```

关注指标：

- 最大成功 context length。
- 最大成功 batch size。
- OOM / error rate。
- success rate。
- process RSS peak。
- Disk read / write。
- GPU utilization。
- startup-level KV cache capacity。
- latency p95。

## 9. Step 7：对比、消融和最终报告

### 9.1 vLLM vs LMCache 对比

```bash
python scripts/reporting/compare_real_runs.py \
  --run vllm=results/step3_vllm_baseline/<RUN_DIR> \
  --run lmcache_cpu=results/step4_lmcache_cpu/<RUN_DIR> \
  --run lmcache_disk=results/step4_lmcache_disk/<RUN_DIR> \
  --output-dir results/step7_baseline_comparison
```

这个对比回答：

- LMCache CPU / Disk 在相同 `gpu_memory_utilization` 下能否扩大可用 KV 容量边界，或改变长 context / 大 batch 的成功率。
- 这种分层换来了多少 process RSS、disk IO 和 latency 代价。
- 哪个 backend 更适合作为 AstraKV 预取的基础。

### 9.2 AstraKV 预取对比

`run_selective_prefetch_real.py` 自身会在同一组 case 中生成：

- `no_prefetch`
- `astrakv_prefetch`
- `astrakv_prefetch_demand`

查看：

```bash
cat results/step5_astrakv_prefetch_with_events/prefetch_report.md
```

重点看：

- no-prefetch latency
- prefetch demand latency
- latency improvement
- prefetch status
- cache event evidence

### 9.3 Policy Ablation 汇总

如果已有 benchmark、prefetch、chunk score 产物，可以生成策略消融表：

```bash
python scripts/policy/analyze_policy_ablation.py \
  --benchmark-run no_prefetch=results/step3_vllm_baseline/<RUN_DIR> \
  --benchmark-run lmcache_cpu_default=results/step4_lmcache_cpu/<RUN_DIR> \
  --benchmark-run lmcache_disk_default=results/step4_lmcache_disk/<RUN_DIR> \
  --prefetch-run astrakv_combined=results/step5_astrakv_prefetch_with_events \
  --output-dir results/step7_policy_ablation
```

### 9.4 最终报告

```bash
python scripts/reporting/build_competition_report.py \
  --benchmark vllm=results/step3_vllm_baseline/<RUN_DIR>/benchmark_results.csv \
  --benchmark lmcache_cpu=results/step4_lmcache_cpu/<RUN_DIR>/benchmark_results.csv \
  --benchmark lmcache_disk=results/step4_lmcache_disk/<RUN_DIR>/benchmark_results.csv \
  --comparison results/step7_baseline_comparison/comparison_results.csv \
  --command "python -m pytest tests/ -v" \
  --output-dir results/competition_report
```

## 10. AstraKV 和 baseline 的最终对比口径

最终报告建议按三层对比写。

### 第一层：plain vLLM baseline

目的：证明未使用 KV 分层、未使用 AstraKV 预取时的真实性能。

对比对象：

- `results/step3_vllm_baseline/<RUN_DIR>/benchmark_results.csv`

说明：

- 这是最干净的 runtime baseline。
- 主要指标是 TTFT、TPOT、throughput、process RSS、GPU utilization、disk IO，以及 startup-level KV cache capacity。

### 第二层：LMCache CPU / Disk baseline

目的：证明已有 KV offload / tiering 方案在同样模型和 workload 下的收益与代价。

对比对象：

- `results/step4_lmcache_cpu/<RUN_DIR>/benchmark_results.csv`
- `results/step4_lmcache_disk/<RUN_DIR>/benchmark_results.csv`

说明：

- 更高的成功率、更大的可跑 context / batch、或更好的 TTFT / TPOT 是收益。
- process RSS、disk IO、latency 上升是代价。
- 它们是 AstraKV-W 的重要 baseline，不能省略。

### 第三层：AstraKV-W selective prefetch

目的：证明本项目在 LMCache backend 上额外引入预取 / warmup 策略后，能减少后续 demand request 的等待。

对比对象：

- `results/step5_astrakv_prefetch_with_events/prefetch_results.csv`
- `results/step5_astrakv_prefetch_with_events/prefetch_report.md`

说明：

- 同一个 runner 内部已经包含 no-prefetch 与 prefetch-demand 对照。
- 如果 prefetch-demand TTFT / latency 下降，可以作为 AstraKV 的性能收益。
- 如果 cache event 支持 hit / store / load，可作为更强证据。
- 如果只靠 latency heuristic，报告里要明确这是 endpoint-level evidence，不要声称已实现内部 KV block 级调度。

## 11. 最小提测包

如果时间紧，至少提交以下结果：

1. `results/dgx_spark_validation/`
2. `results/step3_vllm_baseline/<RUN_DIR>/`
3. `results/step4_lmcache_cpu/<RUN_DIR>/`
4. `results/step4_lmcache_disk/<RUN_DIR>/`
5. `results/step5_astrakv_prefetch_with_events/`
6. `results/step7_baseline_comparison/`
7. `results/competition_report/`

对应说明：

- Step 3 是 plain baseline。
- Step 4 是 memory tiering baseline。
- Step 5 是 AstraKV-W 自身策略效果。
- Step 7 和 competition report 是最终可读证据。

## 12. 常见风险

| 风险 | 说明 | 处理 |
| --- | --- | --- |
| 模型路径不一致 | server 启动模型和 benchmark config model 不一致 | 统一使用本地 `ASTRAKV_MODEL`，必要时改 YAML 中 `backend.model` |
| LMCache 实际没启用 | 脚本启动了，但 connector 参数不被当前版本接受 | 看 server log，确认 `--kv-transfer-config` 和 LMCache init |
| 比较矩阵不一致 | baseline 和 variant 的 context / batch / output 不一致 | 使用同一 YAML 或同一 CLI override |
| AstraKV 过度宣称 | 当前真实路径是 endpoint warmup，不是内部 KV block 调度 | 报告里明确写 endpoint-level selective prefetch |
| 只有单次结果 | repeat 太少，p95 不稳定 | smoke 用 `repeat=1`，正式至少用 config 默认 repeat |
| 没有 cache event | 无法证明真实 cache hit | 保留 server log，运行 `extract_cache_events.py` |
