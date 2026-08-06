# QASPER 在线控制全套实验指南

本指南运行同一份不可变 QASPER 数据集的四个条件：

```text
random  x baseline
random  x online-policy-enabled
grouped x baseline
grouped x online-policy-enabled
```

每个条件都启动独立的冷 vLLM/LMCache 进程和独立的 LMCache LocalDisk 路径。除了是否开启在线策略以及 `random/grouped` 的发布顺序以外，模型、数据集、生成参数、版本元组和请求数保持一致。

## 1. 前置条件

必须在已锁定的 DGX Linux 环境运行：

```text
Python 3.12.3
vLLM 0.23.0
LMCache 0.4.7
LMCacheConnectorV1 / lmcache-vllm-v1 0.4.7
Qwen2.5-7B-Instruct 本地模型
```

先按 `docs/runtime_backend_compatibility.md` 创建环境，并确认：

```bash
.venv/bin/python -c "import vllm, lmcache; print(vllm.__version__, lmcache.__version__)"
nvidia-smi
.venv/bin/python -m unittest discover -s tests
```

数据集目录必须是受验证的只读包：

```text
datasets/task1_qasper/
  prompts/qasper_random_prompts.jsonl
  prompts/qasper_grouped_prompts.jsonl
  metadata/qasper_metadata.jsonl
  validation/*_prompt_validation.json
```

不要编辑上述文件。入口会先通过 adapter 校验 prompt hash、200 条计数和验证 manifest。

## 2. 一条命令运行完整套件

在仓库根目录执行：

```bash
bash scripts/entrypoints/run_task1_qasper_online_control_suite.sh \
  --task1-dir datasets/task1_qasper \
  --output-dir results/qasper-online-control-suite-20260721 \
  --model /opt/models/Qwen2.5-7B-Instruct
```

On DGX Spark, retain the default `--gpu-memory-utilization 0.72`. The host and
GPU share memory; raising it can let `earlyoom` terminate the vLLM EngineCore.
Use a higher value only after observing sufficient host-memory headroom.

首次排障可以保留发布顺序的前 3 条请求：

```bash
bash scripts/entrypoints/run_task1_qasper_online_control_suite.sh \
  --task1-dir datasets/task1_qasper \
  --output-dir results/qasper-online-control-smoke \
  --model /opt/models/Qwen2.5-7B-Instruct \
  --limit 3
```

`--limit` 仅截取已发布顺序的前缀，不改写请求、答案、ground truth 或 context；正式结果必须不传该参数或传 `--limit 0`。

入口自动完成：

1. 将已验证的目录包 materialize 为 `astra-runtime-workload-v1` JSONL；
2. 为每个条件生成新的 mode `0600` 的 runtime secret、session、状态目录和 LocalDisk 目录；
3. 启动带 `LMCacheConnectorV1`、LMCache 0.4.7 Hook 与 EngineCore-owned RuntimeControlHost 的 vLLM；
4. 通过 loopback authenticated request-context 将每个 QASPER `request_id` 发布给 Hook；
5. 用 `run_real_benchmark.py` 记录请求、TTFT、TPOT、延迟和连续资源样本；
6. enabled 条件等待至少一条 `drop / completed / removed > 0` 的真实回执；
7. 导出绑定、事件、命令、回执、trace 和 preflight，评估 QASPER EM/token-F1；
8. 调用严格 paired comparator，并始终写出 suite summary。

## 3. 结果布局

```text
results/qasper-online-control-suite-<timestamp>/
  random/
    canonical/
    baseline/                 # benchmark, request, quality, exported runtime artifacts
    baseline-state/           # EngineCore-owned raw preflight/bindings/events/ledger
    variant/
    variant-state/
    comparison/
  grouped/
    ...
  suite_summary.json
  suite_report.md
  suite_metadata.txt
```

`variant-state/runtime_command_receipts.jsonl` 中必须至少有一条：

```text
status = completed
action = drop
metadata.removed > 0
```

缺少该回执时入口会失败；不能用 HTTP 200、缓存文件数、日志文本或离线 metadata 替代。

## 4. 结果解释

`suite_summary.json` 只证明四个条件的必需证据是否齐全。`comparison/paired_run_manifest.json` 的 `eligible: true` 才允许将 baseline/variant 作为受控在线控制对照讨论。若 comparator 退出非零，入口仍保留完整原始数据和 validator 原因，但结果必须标为：

```text
non_paired_no_claims
```

即使 `eligible: true`，一次 random 与一次 grouped 也不构成性能收益结论。正式报告至少需要多个冷启动重复、随机化条件运行顺序、质量非劣门槛，以及同一 run 的请求/绑定/命令/回执完整覆盖。

## 5. 手工复核

```bash
.venv/bin/python scripts/reporting/summarize_task1_qasper_online_suite.py \
  --suite-dir results/qasper-online-control-suite-20260721

.venv/bin/python scripts/reporting/compare_real_runs.py \
  --run baseline=results/qasper-online-control-suite-20260721/random/baseline \
  --run variant=results/qasper-online-control-suite-20260721/random/variant \
  --output-dir results/qasper-online-control-suite-20260721/random/comparison-refresh
```

检查完毕后确认没有残留服务：

```bash
curl --max-time 2 -fsS http://127.0.0.1:18000/v1/models && exit 1 || true
```

不要提交或共享 `*-state/runtime.env`。其中包含运行时认证秘密，即使其文件权限为 `0600`。
