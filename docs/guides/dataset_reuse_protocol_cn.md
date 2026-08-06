# 数据集复用 Pilot 协议

本协议只衡量 tokenizer 与 chat template 得出的历史 prefix/block 复用机会，证据类别固定为 `modeled_dataset_metadata`。它不报告 vLLM 或 LMCache 的实际 hit、load、store、eviction，也不能据此宣称 endpoint 性能收益。

## 输入与不可变性

QASPER 原始请求只能使用 `datasets/task1_qasper` 中已发布的 `random` 或 `grouped` 顺序。不得添加、删除、重复、重排或改写 request、messages、question、answer、ground truth 或 context。每个原始 request 默认映射为一个 `single_request` workflow，不得自行拆分为 agent subtask。

先使用真实 tokenizer 生成原始 observation。`kv_bytes_per_token` 必须填入经模型配置核验后的正值，不能以猜测值替代：

```bash
python scripts/benchmark/observe_workflow_reuse.py \
  --task1-dir datasets/task1_qasper \
  --workload random \
  --output-dir results/observations/qasper-random \
  --kv-bytes-per-token <verified-positive-value>
```

对 `grouped` 使用独立输出目录重复该命令。生成的 manifest 必须保留输入 prompt SHA-256、输出 SHA-256、tokenizer/model 标识、block size 与 `kv_bytes_per_token`。

## 预注册选择

`configs/dataset_pilot_matrix.yaml` 固定记录 seed、10 条 smoke、50 条 pilot 和 `published_arrival_order_prefix`。选择始终沿发布到达顺序取前缀，不允许人工按复用指标挑选 request。每个 dataset 都必须给出 source ID 和 observation JSONL 的 SHA-256。

运行决策报告：

```bash
python scripts/reporting/build_reuse_opportunity_report.py \
  --config configs/dataset_pilot_matrix.yaml \
  --output-dir results/reuse_pilot
```

报告分别列出 request count、input-token 分布、reusable token ratio、unique prefix block、historical reuse-count 直方图、potential KV bytes、duplicated-prefix ratio 与输入哈希。

## 决策与边界

合法决策只有 `raw_workload_selected`、`raw_workload_selected_with_composed_stress`、`observation_incomplete`。三个语义独立数据集均未具备可审计 observation 时，必须输出 `observation_incomplete`。当前矩阵正处于该状态，因为后两个数据集尚未命名或提供。

当原始 workload 复用不足且另行定义 composed stress 时，composed artifact 必须使用独立 namespace，保存 source request ID、template version、seed 与输出哈希。它与 raw 的统计、QASPER accuracy 以及官方 random/grouped 结论严格分开，禁止合并 aggregate。
