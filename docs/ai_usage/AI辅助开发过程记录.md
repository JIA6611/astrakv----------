# AI 辅助开发过程记录

本文件从 `~/.codex/sessions/2026/06` 中选取 5 条与 AstraKV-W 项目开发、测试和材料整理直接相关的 Codex 会话，作为“AI 辅助开发过程”的交互记录。摘录内容经过整理，保留问题、回答、产出和边界说明，原始完整记录以来源日志为准。

## 记录总览

| 序号 | 时间 | 主题 | 体现的 AI 辅助方式 | 主要产出 |
|---|---:|---|---|---|
| 1 | 2026-06-15 | DGX Spark 环境与提测流程梳理 | 架构适配、运行路径分析、文档重构 | 中文 README、比赛提测流程文档 |
| 2 | 2026-06-17 | 单元测试卡住问题定位与修复 | Debug、根因定位、测试工程化 | `test_layer_offload.py` 改为 mock 模型加载 |
| 3 | 2026-06-26 | 实验图表异常排查与正文图生成 | 数据审查、可视化脚本修改、结果验证 | 9 张正文图、figure manifest/report |
| 4 | 2026-06-27 | MoE route trace 离线实验补强 | 赛题缺口分析、研究脚本实现、真实模型 smoke | `run_moe_route_trace.py`、MoE route 结果链路 |
| 5 | 2026-06-28 | PPT 与代码实现一致性审查 | 材料审阅、代码证据对齐、表述风险收口 | PPT 对齐审阅报告、逐页修改建议 |

## 记录 1：DGX Spark 环境与提测流程梳理

> 来源日志: [rollout-2026-06-15T09-41-09-019ec8f0-91d2-7da2-918c-e41f4332834d.jsonl]

**Q**
```text
分析一下我在这个DGXSpark上面跑程序的话，程序运行脚本要做什么调整吗，结合上面的要求以及里面的所有文档
```

**A**
```text
结论：不需要重写运行脚本，仓库已经有 DGX Spark 路径；但正式在 DGX Spark 上跑时，需要调整启动环境、运行顺序和少量脚本/配置细节。核心点是：不要用 scripts/benchmark_runner.py 当正式结果，它是 synthetic；DGX Spark 正式数据应走 scripts/run_real_benchmark.py + vLLM/LMCache endpoint。

建议调整项包括：

1. 启动前 source configs/dgx_spark_env.sh。
2. vLLM baseline 使用 scripts/launch_vllm_server.sh 和 configs/dgx_spark_vllm_qwen7b.yaml。
3. 正式 benchmark 用 scripts/run_real_benchmark.py。
4. LMCache Disk 启动前明确设置 LMCACHE_DISK_PATH。
5. 按 plain vLLM、LMCache CPU、LMCache Disk、AstraKV selective prefetch、stress、report 的顺序组织提测。

最重要的判断：
- 当前脚本已经符合“DGX Spark 上跑真实 vLLM/LMCache 对比实验”的基本形态。
- AstraKV 当前真实路径是 endpoint-level selective prefetch / warmup，不应写成已经改写 vLLM 内部 KV block scheduler。
```

**产出**
```text
后续根据该会话整理了中文 README 和比赛提测流程文档：
- README.md
- docs/guides/competition_test_flow_cn.md

提测流程被整理成 7 步：环境与本地正确性、OS VM PoC、vLLM baseline、LMCache CPU/Disk baseline、AstraKV selective prefetch、内存受限 stress、对比/消融/最终报告。
```

## 记录 2：单元测试卡住问题定位与修复

> 来源日志: [rollout-2026-06-17T15-25-22-019ed478-6bb8-7882-a781-ea19ed1928c3.jsonl]

**Q**
```text
tests/test_layer_offload.py::LayerOffloadGPUTests::test_load_small_model_to_cpu，分析一下这个单元测试脚本为什么会卡住
```

**A**
```text
这个测试卡住的核心原因：它不是纯单元测试，而是在有 CUDA 时直接加载 Hugging Face 远程模型。

关键路径：
- tests/test_layer_offload.py 只判断 _HAS_TORCH and torch.cuda.is_available()，CUDA 可用时测试会运行。
- 测试使用模型名 Qwen/Qwen2.5-0.5B-Instruct。
- astrakv/vm/layer_offload.py 调用 AutoModelForCausalLM.from_pretrained(...)，这会同步访问 Hugging Face、下载权重、加载模型到 CPU。

所以你看到“卡住”的常见场景是：运行环境装了 accelerate 后，测试不会立刻失败，而是进入 Hugging Face 下载/加载权重阶段。pytest/IDE 里这个过程可能没有进度条，看起来像死锁。

建议：这个测试应该改成 integration/GPU smoke test，或者改为 mock AutoModelForCausalLM.from_pretrained / 使用本地 tiny 模型 / 要求本地缓存并设置 local_files_only=True。
```

**A**
```text
已修改 tests/test_layer_offload.py。

改动点：test_load_small_model_to_cpu 不再真实调用 Hugging Face 下载 Qwen/Qwen2.5-0.5B-Instruct，而是用假的 transformers.AutoModelForCausalLM 验证 load_model_to_cpu() 是否按预期传入：
- torch_dtype=torch.float16
- device_map="cpu"
- low_cpu_mem_usage=True
- 并调用了 model.eval()

验证结果：
.venv/bin/python -m pytest tests/test_layer_offload.py::LayerOffloadGPUTests::test_load_small_model_to_cpu -vv -s
1 passed

.venv/bin/python -m pytest tests/test_layer_offload.py -q
9 passed
```

**产出**
```text
将外部网络和模型下载依赖从单元测试中移除，使测试可以在本地稳定复现。
```

## 记录 3：实验图表异常排查与正文图生成

> 来源日志: [rollout-2026-06-26T15-07-46-019f02c1-8ce0-7741-bdb0-d073731c7b4f.jsonl]

**Q**
```text
/home/szl/Desktop/Inference-OS/results/experiment_figures_20260626_143733，你看一下这组图，我怎么感觉不对呢，为什么很多空白（你可能需要借鉴一下前面仓库的分析）
```

**A**
```text
看了，确实不太对，主要不是实验全空，而是作图脚本读字段读错了。

主要原因：
- 07_vm_mmap_evidence.png 空白：VM 结果的字段在 JSON 里是嵌套的，比如 latency.cold_read_ms、latency.warm_read_ms、stats.resident_ratio，原脚本只读顶层字段，所以全变成 0。
- 06_policy_ablation_actions.png 左半边空：policy CSV 里的字段叫 ttft_delta_pct_vs_baseline / latency_delta_pct_vs_baseline，原脚本没匹配到。
- 03_stress_boundary_latency_disk.png 看起来前面很多空：G016 32K batch16 的 P95 是 163s 到 235s，而普通 stress 是 5s 到 21s，线性坐标把前面压扁了。

处理：
- 修复 VM 嵌套字段解析。
- 修复 policy delta 字段映射。
- 将跨数量级 stress 图改为 log / symlog 尺度。
```

**Q**
```text
一共 9 张，全部是能直接落进报告的正文图。
先说统一画法
颜色固定一套：vLLM 用深蓝，LMCache CPU 用橙色，LMCache Disk 用绿色，Prefetch 可用紫色。
同一张图里尽量只放一种量纲；TTFT、P95、Latency、RSS、写盘量 不要硬塞到一个坐标轴。
...
分析一下这个里面的绘图建议添加内容，看看当前哪些你实现了，哪些没实现，要是实现了但是有出入的话按照给你的部分调整
```

**A**
```text
已按 9 张正文图建议重做生成逻辑，并重新生成到：
results/experiment_figures_20260626_151804/figures

现在目录里正好 9 张正文图：
- 10-1_dgx_experiment_flow.png
- 10-2_baseline_perf_resource.png
- 10-3_selective_prefetch_ttft_latency.png
- 10-4_stress_hierarchy.png
- 10-5_boundary_32k_five_panel.png
- 10-6_boundary_startup_threshold.png
- 10-7_cache_policy_flow.png
- 10-8_os_vm_mechanism_results.png
- 10-9_output_consistency.png

主要调整：
- 固定配色：vLLM 深蓝、LMCache CPU 橙色、LMCache Disk 绿色、Prefetch 紫色。
- 图 10-2 改成 2x2：TTFT、Latency、RSS、写盘量分开。
- 图 10-3 改成 TTFT / Latency 双面板分组柱状图。
- 图 10-5 改成 5 联小图：TTFT、TPOT、P95、RSS、写盘量。
- 图 10-6 改成 0.15/0.16 阈值边界图，并标注 1.67 GiB < 1.75 GiB。
- 图 10-7 补了策略链流程图。
- 图 10-8 改成左机制图 + 右侧多个结果块。
- 图 10-9 新增输出一致性图。

验证：
- .venv/bin/python -m py_compile scripts/plotting/build_experiment_figures.py 通过。
- bash -n scripts/entrypoints/run_experiment_figures.sh 通过。
- 重新生成成功，PNG 数量为 9。
```

**产出**
```text
修改 scripts/plotting/build_experiment_figures.py，形成第 10 章 9 张正文图的自动生成流程，并输出 figure_report.md 与 figure_manifest.json，便于报告和答辩材料引用。
```

## 记录 4：MoE route trace 离线实验补强

> 来源日志: [rollout-2026-06-17T15-25-22-019ed478-6bb8-7882-a781-ea19ed1928c3.jsonl]

**Q**
```text
/home/szl/Desktop/Inference-OS/results/moe_route_trace_qwen15_moe_smoke，在这里你看看
```

**A**
```text
看了，这组结果是有效的，而且我已经把后续离线 MoE 链路也接着跑完了。

现在这个目录里有两层证据：

1. 真实本地 MoE forward route trace
路径：results/moe_route_trace_qwen15_moe_smoke/

关键结果：
- 模型：models/Qwen1.5-MoE-A2.7B-Chat
- 设备：cuda
- dtype：bfloat16
- prompt 数：1
- input tokens：11
- route records：264
- 覆盖层数：24
- 覆盖专家数：60
- layer-expert pairs：638

这说明不是 synthetic 造数据，而是真的跑了 Qwen1.5-MoE 的 Hugging Face forward，并导出了 router top-k 专家选择。

2. AstraKV-W 离线专家对象链路
- expert_events/：规范化专家事件 1056 条，unique experts 60，unique layer-expert pairs 638。
- expert_loading/：planning decisions 638，keep_cpu 5，offload_ssd 633。
- expert_prediction/：evaluated predictions 240，expert prefetch hit rate 0.19375。

建议表述：
当前版本已完成真实本地 MoE 模型的 route trace smoke：在 Qwen1.5-MoE-A2.7B-Chat 上导出 24 层、60 个专家、264 条 token-level route records，并进一步规范化为 1056 条 expert route events，生成专家热度汇总、CPU/SSD 分层放置建议和被动 expert prefetch hint。

边界：
这仍不是 vLLM/LMCache serving 下的 MoE 端到端实验，不作为真实性能提升结论。
```

**Q**
```text
那报告应该怎么加，对应的架构图要不要更新，以及是否还需要绘图等等
```

**A**
```text
建议加，但保持克制：报告补一小节 + 一张表即可。架构图不需要大改，最多改两个节点文字；额外绘图不是必须。

建议插入位置：
第 10 章“系统测试情况”里，在“Cache Event 与策略链实验”之后、“OS 虚拟内存机制实验”之前，新增：
MoE 专家路由访存分析实验

核心写法：
- 说明这次是真实本地 Qwen1.5-MoE-A2.7B-Chat forward，不是 synthetic。
- 说明导出了 router top-k route trace。
- 说明后续进入 AstraKV-W 的专家事件规范化、专家放置计划、专家预取 hint 链路。
- 明确边界：不是 vLLM/LMCache serving 下的 MoE E2E，不宣称 expert weight runtime migration。
```

**产出**
```text
生成 results/moe_route_trace_qwen15_moe_smoke/report_insert_draft.md，包含建议插入报告的章节标题、正文、表格、边界说明、架构图是否更新的建议和证据路径。
```

## 记录 5：PPT 与代码实现一致性审查

> 来源日志: [rollout-2026-06-28T10-08-04-019f0bfb-e2a1-7b92-a32c-6877424c8f02.jsonl]

**Q**
```text
/home/szl/Desktop/Inference-OS/reports/AstraKV-W：项目开发与管理报告 (1).pptx，你看看能不能阅读这个PPT，对照一下咱们当前的实现是否都在里面提到了，有没有不一致的地方
```

**A**
```text
我读完了这份 46 页 PPT。结论是：主线基本都覆盖了，但有几处表述比当前代码成熟度更高，答辩时最好收口。

对得上的部分主要有这些：vLLM + LMCache 真实后端、Selective Prefetch、32K boundary、cache events、mmap 虚拟内存 PoC、ProfileDB / partial load / load-vs-recompute / unified object scheduler 这条策略链，以及报告/归档链路。

不一致或需要改口的地方主要有 4 个：

1. PPT 第 14 页写“129 项功能回归测试”，但当前直接测试日志是 131 passed，这个数字不一致。
2. PPT 把 ProfileDB / ChunkScorer / Partial Load / Load-Recompute / Unified Scheduler 写得像完整 runtime 闭环，但代码里这些大多还是 advisory / skeleton / analysis 层，没有真正消费这些 hint 的后端执行器。
3. runtime/endpoint_prefetch.py 现在做的是 endpoint-level warmup 请求，不是 vLLM/LMCache 内部 KV block 级预取。
4. 内存收益表述要谨慎。DGX Spark 上 GPU framebuffer memory 不可直接拿到，所以当前更稳的是说 RSS、disk I/O、KV capacity、cache events 和边界失败证据。

一句话：这份 PPT 适合拿来讲“证据链已经闭环”，但不适合讲“真实 runtime 已完全改造”。更稳的说法是“真实后端验证 + 策略原型 + VM PoC + 证据链闭环”。
```

**Q**
```text
这些数据是从哪里来的，有直接证据吗（我知道是在报告里面也有，但是有更直接的证据吗）然后其他的修改意见你可以整理成一版md，但是要先和Main.tex文档比较一下这些意见是否准确
```

**A**
```text
已把核对结果整理成一份独立 Markdown：
results/ppt_maintex_alignment_review_20260628.md

直接证据包括：
- results/cli_test_20260627_160923.log 明确是 131 passed
- results/experiment_figures_20260626_191251/figures/prefetch_summary.csv 支撑 73.8396% / 84.28%
- results/extended_g015_ctx32k_b16_out256/02_boundary_32k/vllm_server.log 直接支撑 0.15 启动失败边界
- results/extended_g016_ctx32k_b16_out256/02_boundary_32k/stress_analysis/stress_summary.csv 支撑 0.16 下 32K 边界成功

和 Main.tex 对照后，几条修改意见是成立的：
- 129 项需要改成 131 项
- ProfileDB / ChunkScorer / Unified Scheduler 不能说成完整 runtime 闭环
- endpoint-level warmup 不能说成块级预取
- DGX Spark 上不能写成“显存下降”
```

**A**
```text
按代码 + 结果 + Main.tex 再核对了一遍，并补了一份更完整的对照文档：
results/ppt_maintex_code_alignment_review_20260628.md

这次核对后，结论更明确：
- endpoint_prefetch.py 确实只是 endpoint-level warmup/prefetch
- ChunkScorer 和 UnifiedObjectScheduler 确实还是 advisory / analysis 层
- 131 passed 是当前直接测试日志，不是 129
- prefetch_summary.csv 里平均 TTFT 改善是 73.604656...%
- 4096 的 TTFT 降幅是 84.278819...%
- 0.15 启动失败边界和 0.16 成功边界都有直接日志/汇总证据

所以前面那份 PPT 判断是对的，但更准确地说，它是“主线对齐，但话说得比实现更满”。
```

**产出**
```text
形成 PPT 与 Main.tex、代码、结果三方对照审查材料：
- reports/ppt_maintex_alignment_review_20260628.md
- reports/ppt_maintex_code_alignment_review_20260628.md

该会话用于保证答辩材料不把 advisory 策略链、endpoint-level prefetch、DGX Spark 内存证据写成超过当前代码和实验支持范围的强结论。
```

## 总结

以上 5 条记录覆盖了 AI 辅助开发的主要环节：

1. 将 DGX Spark 运行环境和提测流程整理为可执行路径。
2. 定位并修复单元测试中由真实模型下载导致的卡住问题。
3. 修复实验图表脚本，并生成可进入报告正文的 9 张图。
4. 补强 MoE 专家路由访存分析证据链。
5. 对 PPT、Main.tex、代码和结果做一致性审查，收紧答辩表述。

这些记录体现的 AI 使用方式主要是代码阅读、Debug、实验结果解释、图表生成、报告材料审查和边界表述校正。AI 输出均需由仓库代码、日志、CSV/JSON、图表 manifest 和测试结果进行复核，不直接作为性能结论本身。
