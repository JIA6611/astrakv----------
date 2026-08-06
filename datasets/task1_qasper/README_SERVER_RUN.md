# 任务一：Qasper Workload 服务器运行说明

## 1. 这个文件夹是什么

这个文件夹是给服务器同学运行第一轮真实 endpoint 实验的输入包。

目标：

```text
用同一批 Qasper 公开 Benchmark 样本，对比 random workload 和 grouped workload。
不要修改 question。
不要修改 answer。
不要修改 ground_truth。
不要修改 context。
唯一变化是 request ordering。
```

当前主要要跑：

```text
vLLM vanilla + random
vLLM vanilla + grouped
```

如果服务器上已经有 LMCache 环境，再继续跑：

```text
LMCache CPU + random
LMCache CPU + grouped
LMCache Disk + random
LMCache Disk + grouped
```

## 2. 文件说明

### 必须输入文件

```text
prompts/qasper_random_prompts.jsonl
prompts/qasper_grouped_prompts.jsonl
```

这两个文件是服务器 endpoint 真正要读取的请求文件。

每一行是一条 request，关键字段包括：

```text
request_id
sample_id
dataset
task
workload_type
reuse_group
shared_context
prompt
messages
answer
ground_truth
metadata_ref
```

如果 runner 支持 OpenAI-compatible chat completions，优先使用：

```text
messages
```

如果 runner 只支持普通 prompt，则使用：

```text
prompt
```

### 辅助分析文件

```text
metadata/qasper_metadata.jsonl
metadata/qasper_statistics.json
metadata/qasper_manifest.json
metadata/qasper_sha256.json
```

这些用于后续分析 reuse_group、context block、prefetch/offload candidate、estimated reusable tokens。

### 原始 workload 顺序

```text
workloads/qasper_random_workload.jsonl
workloads/qasper_grouped_workload.jsonl
```

这些是 prompt 编译前的 workload 文件，用于验证 random/grouped 是否是同一批样本。

### 本地验证文件

```text
validation/random_prompt_manifest.json
validation/grouped_prompt_manifest.json
validation/random_prompt_statistics.json
validation/grouped_prompt_statistics.json
validation/random_prompt_validation.md
validation/grouped_prompt_validation.md
```

这几个文件证明 prompt 已经在本地通过 validation：

```text
prompt 不为空
request_id 唯一
sample_id 顺序保持
answer 不变
ground_truth 不变
question_hash 不变
answer_hash 不变
ground_truth_hash 不变
context_hash 不变
```

## 3. Prompt 基本统计

Random：

```text
Prompt count: 200
Shared-context prompt ratio: 0.4750
Average prompt tokens: 4541.31
Max prompt tokens: 18917
Total prompt tokens: 908263
Validation: PASSED
```

Grouped：

```text
Prompt count: 200
Shared-context prompt ratio: 0.4750
Average prompt tokens: 4541.31
Max prompt tokens: 18917
Total prompt tokens: 908263
Validation: PASSED
```

注意：

```text
最长 prompt 约 18,917 estimated tokens。
vLLM max_model_len 建议设置到 24K 或 32K。
如果显存不够，先跑 limit 3 或 limit 10。
```

## 4. 服务器启动 vLLM 示例

如果使用 Qwen2.5-7B-Instruct：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

如果显存不足，可以先降低并只跑短 smoke：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

如果出现：

```text
maximum context length exceeded
```

说明 `max_model_len` 不够，或者当前 prompt 太长。先用小规模 smoke 验证链路。

## 5. 运行前先检查 prompt 文件

在服务器上进入本文件夹后，先运行：

```bash
python - <<'PY'
import json
from pathlib import Path

for p in [
    "prompts/qasper_random_prompts.jsonl",
    "prompts/qasper_grouped_prompts.jsonl",
]:
    rows = [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    print(p, "count=", len(rows))
    print("first request_id:", rows[0]["request_id"])
    print("first sample_id:", rows[0]["sample_id"])
    print("has messages:", isinstance(rows[0].get("messages"), list))
    print("prompt chars:", len(rows[0]["prompt"]))
    print("max_tokens:", rows[0].get("max_tokens"))
    print("temperature:", rows[0].get("temperature"))
    print()
PY
```

期望看到：

```text
count= 200
has messages: True
max_tokens: 128
temperature: 0.0
```

## 6. 请求 endpoint 的格式

每条 JSONL 里已经有：

```text
messages
max_tokens
temperature
top_p
```

请求 `/v1/chat/completions` 时，请使用类似格式：

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": "<record.messages>",
  "max_tokens": 128,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": true
}
```

## 7. 推荐运行顺序

### Step 1：先跑 3 条 smoke

Random smoke：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_random_prompts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --limit 3 \
  --output-dir results/qasper_random_smoke
```

Grouped smoke：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_grouped_prompts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --limit 3 \
  --output-dir results/qasper_grouped_smoke
```

确认没有：

```text
HTTP 400
HTTP 500
max_model_len exceeded
CUDA OOM
```

### Step 2：跑完整 vLLM vanilla

Random：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_random_prompts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/vllm_qasper_random
```

Grouped：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_grouped_prompts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/vllm_qasper_grouped
```

### Step 3：如果有 LMCache CPU

假设 LMCache CPU endpoint 是：

```text
http://127.0.0.1:8001
```

运行：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_random_prompts.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/lmcache_cpu_qasper_random

python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_grouped_prompts.jsonl \
  --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/lmcache_cpu_qasper_grouped
```

### Step 4：如果有 LMCache Disk

假设 LMCache Disk endpoint 是：

```text
http://127.0.0.1:8002
```

运行：

```bash
python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_random_prompts.jsonl \
  --base-url http://127.0.0.1:8002 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/lmcache_disk_qasper_random

python run_prompt_workload_endpoint.py \
  --prompts prompts/qasper_grouped_prompts.jsonl \
  --base-url http://127.0.0.1:8002 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/lmcache_disk_qasper_grouped
```

## 8. 每个 request 需要保留的字段

输出 `request_results.jsonl` 时，请至少保留：

```text
request_id
sample_id
dataset
task
workload_type
reuse_group
shared_context
status
error
ttft_ms
tpot_ms
latency_ms
output_tokens
output_text
answer
ground_truth
prompt_hash
metadata_ref
```

如果能采集资源指标，也请保留：

```text
gpu_memory_mb_before
gpu_memory_mb_after
cpu_memory_mb_before
cpu_memory_mb_after
disk_read_mb
disk_write_mb
```

## 9. 跑完后请回传

请把每个实验目录完整打包回来，至少包含：

```text
request_results.jsonl
benchmark_results.csv
benchmark_report.md
server.log
```

如果有采样文件，也一起回传：

```text
samples/*.csv
nvidia-smi samples
disk io samples
lmcache logs
vllm logs
```

推荐回传目录结构：

```text
server_results/
  vllm_qasper_random/
    request_results.jsonl
    benchmark_results.csv
    benchmark_report.md
    server.log

  vllm_qasper_grouped/
    request_results.jsonl
    benchmark_results.csv
    benchmark_report.md
    server.log

  lmcache_cpu_qasper_random/
    request_results.jsonl
    benchmark_results.csv
    benchmark_report.md
    server.log

  lmcache_cpu_qasper_grouped/
    request_results.jsonl
    benchmark_results.csv
    benchmark_report.md
    server.log
```

## 10. 实验解释口径

本任务不是测模型是否更聪明，而是测系统行为：

```text
同一批公开 Benchmark 样本
同样 question
同样 answer
同样 ground_truth
同样 context
只改变 request ordering
比较 random vs grouped
观察 shared-context workload 是否能带来 cache hit / TTFT / memory 变化
```

请不要把本实验说成：

```text
我们修改了 Benchmark
我们重写了 vLLM scheduler
我们已经证明真实 KV cache hit 提升
```

当前服务器实验要验证的是：

```text
offline shared-context reuse opportunity 是否能在真实 endpoint 中转化为实际系统收益。
```
