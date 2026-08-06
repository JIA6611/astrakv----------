> 来源日志: [rollout-2026-05-20T13-18-17-019e43d2-03cb-7970-97b2-ecb3396b6c3a.jsonl]
## 记录 1

**Q：**

```text
<environment_context>
  <cwd>C:\Users\Lenovo\Desktop\调研</cwd>
  <shell>powershell</shell>
  <current_date>2026-05-20</current_date>
  <timezone>Asia/Shanghai</timezone>
</environment_context>
```
## 记录 2

**Q：**
```text
# ROLE
你是一位系统方向科研助理（Systems Research Assistant）。
擅长：
- Operating System
- LLM inference optimization
- Memory management
- AI Systems
- scientific literature review
- reproducible systems research
任务：
请根据我提供的调研文档 [内存受限LLM推理优化_论文地图与比赛路线图_2025_2026.md](内存受限LLM推理优化_论文地图与比赛路线图_2025_2026.md) ，
先自动下载论文 PDF，
再阅读论文原文，
最后生成系统化论文概况与比较分析。
目标：
帮助我研究：
Runtime Optimization of LLM Inference for the Memory Constraint System
注意：
不要只根据标题或摘要总结。
必须阅读论文 PDF 原文。
---
# INPUT
输入：
1. 调研文档
（包含论文名称、链接、阅读路线、研究主题）
工作原则：
- 优先下载论文 PDF
- 优先官方版本
- 优先 arXiv / conference page / author page
- 必须基于论文原文分析
- 禁止只读 abstract
---
# TASK 1 — 自动下载论文（必须）
根据调研文档：
逐篇执行：
1. 搜索论文
2. 找 PDF
3. 下载到本地文件夹
优先级：
1. official conference PDF
2. arXiv PDF
3. 作者主页 PDF
下载后：
验证：
- 文件是否可打开
- 是否完整
- 是否不是网页错误页
- 是否是最终论文版本
输出：
## Download Report
| Paper | Venue | Year | PDF Found? | Downloaded? | Source | Open Source Code? | GitHub |
例如：
| FlexInfer | MLSys | 2025 | Yes | Yes | MLSys PDF | Yes | xxx |
如果失败：
说明原因。
继续下一篇。
不要中断。
---
# TASK 2 — 阅读论文（重点）
下载完成后：
逐篇阅读 PDF。
阅读要求：
不要只看：
- abstract
- intro
- conclusion
重点理解：
1. Problem
解决什么问题？
2. Motivation
为什么以前方法不够？
3. System Design
系统架构是什么？
4. Key Mechanism
核心技术机制是什么？
例如：
- paging
- offloading
- prefetching
- async pipeline
- overlap compute & IO
- KV cache management
- expert loading
- virtual memory
5. Runtime Optimization
优化：
- latency
- throughput
- GPU memory
- CPU memory
- SSD IO
6. Memory Behavior
如何降低运行时内存？
7. Experiment
怎么验证？
8. Limitation
不足是什么？
9. Reproducibility
是否容易复现？
10. Competition Relevance
对比赛是否有帮助？
---
# TASK 3 — Paper Card（统一格式）
对每篇论文输出：
# Paper N
## Basic Info
- title
- venue
- year
- pdf source
- github
- open source?
## Problem
一句话：
它解决什么问题？
## Motivation
为什么以前方法不够？
## Core Idea
一句话概括：
它怎么解决？
## System Design
解释：
模块结构。
尽量画：
ASCII架构图。
例如：
Request
 ↓
KV Scheduler
 ↓
Prefetch Queue
 ↓
GPU Runtime
## Key Mechanisms
逐条解释：

- async prefetch
- memory paging
- offloading
- overlap IO
## Runtime Optimization
具体优化：
- memory ↓
- latency ↓
- throughput ↑
## Experiment
- baseline
- metric
- result
## Strength
优点
## Weakness
缺点
## Reproducibility
低 / 中 / 高
为什么？
## Competition Relevance
高 / 中 / 低
为什么？
## Innovation Inspiration（重点）
从它出发：
有哪些可改进点？
---
# TASK 4 — 横向比较
输出：
## Comparative Analysis
表格：
| Paper | Problem | Mechanism | Memory Strategy | Runtime Strategy | Open Source | Difficulty | Competition Value |
分析：
1. 哪些论文可组合
2. 哪些只是 incremental
3. 哪些是真创新
4. 哪些最值得复现
---
# TASK 5 — 阅读优先级
输出：
S级（必须读）
A级（推荐）
B级（选读）
理由：
- 比赛价值
- 可复现性
- 创新潜力
---
# TASK 6 — 强制自检（必须）
在输出前检查：
□ 是否真的下载了 PDF
□ 是否真的阅读原文
□ 是否不是只复述摘要
□ 是否检查了代码仓库
□ 是否给出创新启发
□ 是否做横向比较
如果没有做到：
自动补充。
禁止输出未经检查结果。
---
最终目标：
得到：
“下载好的论文 + 高质量论文概况 + 比赛导向分析 + 创新启发”
```

**A：**

```text
我会先读取这份调研文档，抽取论文清单、链接和优先级，然后逐篇定位 PDF 与代码仓库；接着把 PDF 下载到工作区中的专门文件夹，并用本地工具验证它们确实是可解析的论文文件。

文档读取出来有编码错位，但链接和表格主体仍然能识别；我会再用 UTF-8 方式确认一次，并按文档中的主表论文全部处理，包括 S/A/B 中出现的论文和代码链接。

论文清单已经明确：我会覆盖主表中的 13 篇，包括 FlexInfer、NEO、IMPRESS、Jenga、ShadowKV、Cake、Bidaw、ECHO、KTransformers、Fast On-device LLM Inference with NPUs、Klotski、CAKE、RocketKV。下一步我会定位官方 PDF URL，并把下载过程做成可验证的本地清单。

PDF 来源已基本定位：官方会议 PDF 优先可用，PMLR/USENIX/arXiv/OpenReview 的链接都能解析；其中 ECHO 的 OSDI 2026 页面目前只有开放页面文字，暂未出现 PDF 下载入口，我会继续检索 arXiv/作者页，不会因为这一篇卡住整体流程。

现在进入下载阶段：我会把 PDF 统一放到 `papers_memory_constrained_llm_2025_2026`，并保留一个本地校验表，方便后面把“下载来源”和“阅读证据”对齐起来。

下载命令已经跑完，但表格输出被 PowerShell 截断了，我会用 PDF 解析器逐个打开并抽取页数、标题页文本和错误信息，这比只看文件大小更可靠。

本地 PowerShell 网络被沙箱拦住了，下载全部失败且没有留下伪 PDF；按权限规则我会直接请求一次网络提升来完成下载，这一步是完成“自动下载论文”的必要动作。

下载完成了，但有几篇可能是 404/会议页占位被保存成 HTML 的风险；我正在用 PyMuPDF 检查每个文件签名、页数、首页文本，坏文件会单独重找来源。

校验结果很好：12 篇已经是可打开的完整 PDF；只有 `ECHO_OSDI2026` 没出现在可解析列表里，说明该目标文件没有成功形成 PDF，我会单独检查下载清单并重新找 ECHO 的可靠 PDF。

ECHO 目前确认是 USENIX OSDI 2026 页面已上线但系统 PDF 返回 404，且检索没有发现 arXiv/作者页 PDF；我会把它列为“PDF 未公开，无法原文精读”，避免假装读过。其余 12 篇我会继续做全文抽取和分节阅读。

全文文本已经抽取完，字符量从 5 万到 12 万不等，说明不是只读了摘要。接下来我会按系统论文阅读模板从每篇的 design、implementation、evaluation、limitations 等段落里抽取机制和证据。

我已经确认代码仓库状态：NEO、ShadowKV、KTransformers、mllm、CAKE、RocketKV 有明确公开仓库；FlexInfer、IMPRESS、Jenga、Cake、Bidaw、Klotski 暂未找到官方实现；ECHO 只有 USENIX 页面，无公开 PDF/代码。现在我会把阅读笔记压缩成统一 Paper Card，而不是把所有摘录堆给你。

我已经从每篇正文里抓到了关键实验结果和机制证据，比如 FlexInfer 的分阶段执行策略、NEO 的非对称 CPU-GPU pipeline、IMPRESS 的重要性 KV 与多层存储、Jenga 的两级 allocator、Cake 的 load/recompute 双向调度等。最后一步我会整理成中文报告，并明确哪些结论来自 PDF 原文、哪些只是页面/代码仓库检索。

我会把完整结果落成一份新的 Markdown 报告，里面包含下载表、逐篇 Paper Card、横向比较、阅读优先级和强制自检；对 ECHO 我会如实标注 PDF 未公开，避免把网页信息伪装成原文精读。

报告文件已经生成；我会最后快速确认文件存在、下载目录完整，并检查 manifest 与报告中的 ECHO 状态一致。

已完成下载、验证、原文阅读和系统化报告整理。

产物在这里：

- 论文 PDF 目录：[papers_memory_constrained_llm_2025_2026](</C:/Users/Lenovo/Desktop/调研/papers_memory_constrained_llm_2025_2026>)
- 精读分析报告：[内存受限LLM推理优化_论文精读报告_2026-05-20.md](</C:/Users/Lenovo/Desktop/调研/内存受限LLM推理优化_论文精读报告_2026-05-20.md>)
- 全文抽取文本目录：[text](</C:/Users/Lenovo/Desktop/调研/papers_memory_constrained_llm_2025_2026/text>)

本次共成功下载并验证了 12 篇可打开的 PDF：FlexInfer、NEO、IMPRESS、Jenga、ShadowKV、Cake、Bidaw、KTransformers、Fast On-device LLM Inference with NPUs、Klotski、CAKE、RocketKV。每篇都用 PyMuPDF 验证页数并抽取了全文文本，报告中的 Paper Card 基于原文的 design / implementation / evaluation / discussion 等部分整理，不只是摘要。

唯一例外是 `ECHO`：USENIX OSDI 2026 页面存在，但官方 PDF URL 当前返回 404，也未检索到 arXiv 或作者页 PDF，所以我在报告里如实标注为“PDF 未公开，未做原文精读”，没有把网页信息伪装成论文原文阅读。

报告中已经包含你要求的：

- `Download Report`
- 逐篇 `Paper Card`
- 系统设计 ASCII 图
- runtime / memory / experiment / limitation / reproducibility 分析
- 横向比较表
- S/A/B 阅读优先级
- 比赛导向组合路线
- 强制自检清单

核心结论也已经收束出来：最适合比赛落地的方向是 `selective KV prefetch + load-vs-recompute + multi-tier KV cache`，可以组合 IMPRESS、Cake、Bidaw、ShadowKV、CAKE/RocketKV 的思想，在 `vLLM + LMCache/FlexKV` 这类底座上做出比较稳、可展示、系统味道足的方案。
```