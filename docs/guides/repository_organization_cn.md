# 仓库整理与提测复现清单

本文档用于回答三个问题：

1. 当前哪些文件属于主线提测必须保留。
2. 哪些文件是历史产物、可选研究工具或可删除候选。
3. 后续如何按固定流程复现实验并整理结果。

注意：本文档只给出整理建议，不要求立即删除。删除前建议先确认对应文件没有被你当前报告、答辩材料或自定义脚本引用。

## 1. 推荐目录分层

### 1.1 文档

主线文档：

- `README.md`：项目入口、快速命令、整体说明。
- `docs/README.md`：文档索引。
- `docs/guides/competition_test_flow_cn.md`：比赛提测主流程。
- `docs/guides/repository_organization_cn.md`：仓库整理、文件取舍、复现清单。
- `results/competition_requirement_coverage_report.md`：赛题要求覆盖性分析。

辅助文档：

- `docs/guides/dgx_spark_setup.md`：DGX Spark 环境和手动运行说明。
- `docs/guides/dgx_spark_bootstrap.md`：DGX Spark 安装说明。
- `docs/guides/reproduction.md`：早期 P0 复现流程。
- `docs/architecture/*`：架构解释。
- `docs/planning/*`：规划和阶段总结。
- `docs/analysis/*`：早期分析、审计和第三方调研。

### 1.2 代码

主 Python 包：

- `astrakv/runtime/`：trace、ProfileDB、cache events、endpoint prefetch、VM backend。
- `astrakv/kv_cache/`：KV chunk 元数据、partial-load 规划。
- `astrakv/prefetch/`：预取引擎、selective KV 策略。
- `astrakv/scheduler/`：chunk action、object scheduling、load/recompute hint。
- `astrakv/vm/`：mmap KV cache、DGX Spark VM adapter、layer offload PoC。
- `astrakv/evaluation/`：quality / hidden-state 评估。
- `cli.py`：本地测试、VM demo、合成 benchmark 入口。

主线脚本：

- `scripts/entrypoints/run_competition_e2e.sh`
- `scripts/entrypoints/run_competition_extended_evidence.sh`
- `scripts/launch/launch_vllm_server.sh`
- `scripts/launch/launch_lmcache_vllm.sh`
- `scripts/benchmark/run_real_benchmark.py`
- `scripts/benchmark/run_selective_prefetch_real.py`
- `scripts/benchmark/dgx_metrics_collector.py`
- `scripts/benchmark/extract_cache_events.py`
- `scripts/reporting/analyze_stress_results.py`
- `scripts/reporting/compare_real_runs.py`
- `scripts/reporting/build_competition_report.py`
- `scripts/policy/build_trace_store.py`
- `scripts/policy/build_profile_db.py`
- `scripts/policy/score_chunks.py`
- `scripts/policy/decide_load_vs_recompute.py`
- `scripts/policy/run_unified_object_scheduler.py`
- `scripts/research/evaluate_quality.py`
- `scripts/vm/run_dgx_spark_vm_evidence.py`
- `scripts/vm/run_mmap_kv_cache.py`
- `scripts/entrypoints/run_dgx_spark_validation.sh`

### 1.3 配置

主线配置：

- `configs/dgx_spark_env.sh`
- `configs/dgx_spark_vllm_qwen7b.yaml`
- `configs/dgx_spark_lmcache_cpu.yaml`
- `configs/dgx_spark_lmcache_disk.yaml`
- `configs/astrakv_real_selective_prefetch.yaml`
- `configs/lmcache_cpu_constrained.yaml`
- `configs/lmcache_disk_constrained.yaml`
- `configs/stress_vllm_memory_constrained.yaml`
- `configs/stress_lmcache_cpu_memory_constrained.yaml`
- `configs/stress_lmcache_disk_memory_constrained.yaml`
- `configs/stress_vllm_extreme_memory_constrained.yaml`
- `configs/stress_lmcache_cpu_extreme_memory_constrained.yaml`
- `configs/stress_lmcache_disk_extreme_memory_constrained.yaml`

辅助配置：

- `configs/astrakv_selective_prefetch.yaml`：合成 / 策略模拟用，不代表真实 endpoint 性能。
- `configs/shared_prefix_workload.yaml`：共享前缀 workload 配置。
- `configs/policy_ablation_matrix.yaml`：消融矩阵配置。
- `configs/lmcache_cpu_example.yaml`
- `configs/lmcache_disk_example.yaml`

### 1.4 其他

必须保留或按需保留：

- `tests/`：单元测试。
- `requirements.txt`、`pyproject.toml`、`Makefile`：项目依赖和构建配置。
- `models/`：本地模型目录，体积大，通常不进 git，但运行时需要。
- `results/`：实验产物目录，默认不进 git，只保留关键报告和归档包。

可清理生成物：

- `__pycache__/`
- `.pytest_cache/`
- `.venv/`
- `*.egg-info/`
- `results/lmcache_disk_store/`
- 历史中间结果目录

## 2. 主线提测流程

### 2.1 最推荐完整流程

```bash
cd /home/szl/Desktop/Inference-OS
source .venv/bin/activate
source configs/dgx_spark_env.sh
export ASTRAKV_MODEL="$PWD/models/Qwen2.5-7B-Instruct"

bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --skip-install \
  --continue-on-failure
```

默认输出：

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

最重要产物：

- `07_final_report/competition_report.md`
- `07_final_report/artifact_inventory.csv`
- `07_final_report/competition_report_manifest.json`
- `archive/*.tar.gz`

### 2.2 快速 smoke

```bash
bash scripts/entrypoints/run_competition_e2e.sh \
  --only smoke \
  --skip-install
```

用途：

- 快速验证 Python 环境、基础 VM evidence、脚本可用性。
- 不作为最终性能结论。

### 2.3 主线 E2E

```bash
bash scripts/entrypoints/run_competition_e2e.sh \
  --skip-install \
  --continue-on-failure
```

用途：

- official stress
- extreme stress
- vLLM baseline
- LMCache CPU / Disk baseline
- AstraKV selective prefetch
- cache event
- policy ablation
- competition report

### 2.4 极限边界测试

当前最有说服力的边界测试建议使用 32K，不要超过模型原生 `max_position_embeddings=32768`。

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

- `0.15`：vLLM 因 KV cache capacity 不足无法启动 32K。
- `0.16`：vLLM / LMCache CPU / LMCache Disk 均能完成 32K + batch 16。
- 这两组可以共同证明内存受限边界。

### 2.5 给已有目录补报告

如果 boundary 已经跑完，只需要补 cache / report / archive：

```bash
bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only cache \
  --output-root results/extended_g016_ctx32k_b16_out256 \
  --skip-install

bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only report \
  --output-root results/extended_g016_ctx32k_b16_out256 \
  --skip-install

bash scripts/entrypoints/run_competition_extended_evidence.sh \
  --only archive \
  --output-root results/extended_g016_ctx32k_b16_out256 \
  --skip-install
```

## 3. 推荐保留的关键结果目录

建议保留：

- `results/extended_evidence_20260625_014917`
- `results/extended_g016_ctx32k_b16_out256`
- `results/extended_g015_ctx32k_b16_out256`
- `results/competition_requirement_coverage_report.md`

原因：

- `extended_evidence_20260625_014917` 是完整主线证据。
- `extended_g016_ctx32k_b16_out256` 是极限可运行证据。
- `extended_g015_ctx32k_b16_out256` 是启动失败边界证据。
- coverage report 对照了赛题要求。

可选保留：

- `results/competition_e2e_20260624_012006`
- `results/competition_e2e_20260624_105308`
- `results/extended_evidence_20260624_232940`

这些目录可以作为历史对照，但不是当前最强证据。

## 4. 可删除候选

### 4.1 高优先级可删：生成缓存

这些通常可以直接删除，删除后可重建：

- `.pytest_cache/`
- 所有 `__pycache__/`
- `astrakv_w.egg-info/`
- `.venv/`，如果你愿意重新安装环境。
- `results/lmcache_disk_store/`
- `results/*/lmcache_disk_store/`

建议命令：

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache astrakv_w.egg-info
```

如果 `.venv/` 很大且可重装：

```bash
rm -rf .venv
```

### 4.2 高优先级可归档：历史结果

这些是历史调参目录，不是当前最强结果。建议先打包或移动到外部磁盘，再决定删除：

- `results/competition_e2e_20260617_*`
- `results/competition_e2e_20260618_*`
- `results/competition_e2e_20260621_*`
- `results/extended_g020_ctx49k`
- `results/extended_g020_ctx32k_b16_out256`
- `results/extended_from_existing_smoke`
- `results/0-*`
- `results/step2_*`
- `results/step3_*`
- `results/step4_*`
- `results/step5_*`
- `results/step6_*`
- `results/step7_*`

保留前请确认其中没有你要写进论文或 PPT 的截图、表格、日志。

### 4.3 可选删除：非 Linux 入口

如果后续只在 DGX Spark Linux 上运行，可以考虑删除或归档：

- `scripts/archive/launch_vllm_server.ps1`
- `scripts/archive/launch_lmcache_vllm.ps1`

如果还需要 Windows / PowerShell 复现入口，则保留。

### 4.4 可选删除：非主线研究脚本

这些脚本不是当前比赛主线必须项，但可能对后续扩展有用。建议先归档，不建议马上删：

- `scripts/archive/run_ablation.sh`：早期 ablation runner，当前已由 E2E / extended 脚本替代。
- `scripts/archive/run_edge_sim_tests.sh`、`scripts/archive/setup_edge_sim.sh`：边缘模拟辅助，当前 DGX Spark 主线不用。
- `scripts/benchmark/benchmark_runner.py`：合成 benchmark，不能用于真实 vLLM 性能结论。
- `scripts/benchmark/metrics_collector.py`：合成 benchmark 本地指标 helper，真实 benchmark 使用 `dgx_metrics_collector.py`。
- `scripts/reporting/build_demo_dashboard.py`：展示用 dashboard，不是提测必需。
- `scripts/research/evaluate_hidden_state_drift.py`：需要 hidden-state 导出，当前主线不产生。
- `scripts/reporting/analyze_multi_model_evaluation.py`：多模型归档分析，当前主线只用 Qwen2.5-7B。
- `scripts/reporting/analyze_failure_recovery.py`：失败恢复分析，当前 final report 非必需。
- `scripts/reporting/analyze_memory_pressure.py`：额外 memory pressure hint，当前 final report 非必需。
- `scripts/reporting/plot_benchmarks.py`：图表辅助，`run_real_benchmark.py` 已生成基础图。
- `scripts/research/extract_moe_expert_events.py`
- `scripts/research/plan_moe_expert_loading.py`
- `scripts/research/predict_moe_experts.py`

MoE 相关脚本说明：当前主实验模型是 dense Qwen2.5-7B-Instruct，没有真实 MoE expert activation。因此这些脚本对当前提交不是必须，但如果后续换 MoE 模型可以复用。

### 4.5 可选删除：非主线配置

这些配置不是当前主提测入口必须项：

- `configs/astrakv_selective_prefetch.yaml`：策略模拟，不是真实 endpoint。
- `configs/lmcache_cpu_example.yaml`
- `configs/lmcache_disk_example.yaml`
- `configs/shared_prefix_workload.yaml`
- `configs/policy_ablation_matrix.yaml`

如果要保持示例完整性，可以保留；如果目标是压缩仓库，可以归档。

### 4.6 可选删除：历史 PDF / 规划文档

如果最终提交只保留 Markdown 和核心报告，可以考虑归档：

- `docs/guides/GPU_TESTING_WORKFLOW_CN.pdf`
- `docs/planning/TEACHER_PROGRESS_BRIEF_CN.pdf`
- `docs/planning/IMPLEMENTATION_PLAN.md`
- `docs/analysis/project_audit_report.md`
- `docs/analysis/codebase_analysis.md`
- `docs/analysis/third_party_analysis.md`

这些不是运行必需，但对答辩解释、开发过程追溯有帮助。

## 5. 不建议删除

不要删除：

- `scripts/entrypoints/run_competition_e2e.sh`
- `scripts/entrypoints/run_competition_extended_evidence.sh`
- `scripts/benchmark/run_real_benchmark.py`
- `scripts/benchmark/run_selective_prefetch_real.py`
- `scripts/reporting/build_competition_report.py`
- `scripts/benchmark/extract_cache_events.py`
- `scripts/benchmark/dgx_metrics_collector.py`
- `scripts/launch/launch_vllm_server.sh`
- `scripts/launch/launch_lmcache_vllm.sh`
- `configs/dgx_spark_*.yaml`
- `configs/stress_*memory_constrained.yaml`
- `configs/lmcache_*constrained.yaml`
- `configs/astrakv_real_selective_prefetch.yaml`
- `tests/`
- `astrakv/runtime/`
- `astrakv/prefetch/`
- `astrakv/kv_cache/`
- `astrakv/scheduler/`
- `astrakv/vm/`

## 6. 提交前检查清单

运行：

```bash
bash -n scripts/entrypoints/run_competition_e2e.sh
bash -n scripts/entrypoints/run_competition_extended_evidence.sh
python -m pytest tests/test_reporting_tools.py tests/test_policy_ablation.py tests/test_competition_report.py tests/test_quality_evaluation.py -q
```

确认：

- `results/extended_evidence_20260625_014917/07_final_report/competition_report.md` 存在。
- `results/extended_evidence_20260625_014917/07_final_report/artifact_inventory.csv` 没有 missing。
- `results/extended_g016_ctx32k_b16_out256/07_final_report/competition_report.md` 存在。
- `results/extended_g015_ctx32k_b16_out256/02_boundary_32k/vllm_server.log` 中有 KV cache 不足错误。

## 7. 最终报告表述边界

可以声明：

- DGX Spark 是 UMA 平台，适合作为边缘 / 嵌入式统一内存场景的代表性验证环境。
- 当前证据使用 startup-level KV capacity、process RSS、disk IO、cache events、stress boundary 来判断内存受限表现。
- `gpu_memory_utilization=0.15` 时 32K 启动失败，`0.16` 时可以完成 32K + batch 16。
- AstraKV-W 实现了 endpoint-level selective prefetch / warmup，并通过 cache events 和 prefetch results 记录收益。

不要声明：

- 不要说 case-level GPU memory 已下降。
- 不要说已经替换 vLLM 内部 KV scheduler。
- 不要说端到端 latency 全面优于 baseline。
- 不要把 49K context 失败当作内存不足证据；它是模型最大上下文限制。
