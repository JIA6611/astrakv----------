# Text/KV Consistency 8K/16K 验证指南

这份指南用于验证两件事：

1. `8K + 16K` 的 controlled workload 是否按 `exact / sim90 / sim80` 三个相似度桶正确生成。
2. `cold / warm / hot` 条件下，是否采集到了足够完整的 KV 复用证据，尤其是逐请求的块级命中/加载统计。

## 1. 基本前提

- 当前默认 suite 已切换为 `8K/16K`，不再以 `32K/128K` 为默认目标。
- 主证据源是：
  - `runtime_events_raw.jsonl`
  - `runtime_structured_events.jsonl`
- `cache_events.jsonl` 只作为 server log fallback；如果最终只能依赖 fallback，报告会降级为不完整证据。

## 2. 运行套件

```bash
cd /home/zyx/astrakv2

RUN_ROOT="results/text_kv_consistency_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

export ASTRAKV_MODEL="/opt/models/Qwen3-8B"
export ASTRAKV_HOST="127.0.0.1"
export ASTRAKV_PORT="18000"
export ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT="17900"
export ASTRAKV_MAX_MODEL_LEN="16384"
export ASTRAKV_GPU_MEMORY_UTILIZATION="0.60"
export ASTRAKV_PREFIX_CACHING="true"

bash scripts/entrypoints/run_text_kv_consistency_suite.sh \
  --output-dir "$RUN_ROOT" \
  --model "$ASTRAKV_MODEL" \
  --host "$ASTRAKV_HOST" \
  --port "$ASTRAKV_PORT" \
  --context-port "$ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT" \
  --context-lengths "8192,16384" \
  --max-model-len "16384" \
  --gpu-memory-utilization "0.60" \
  --block-size-tokens "16" \
  --kv-bytes-per-token "1" \
  --timeout "1800" \
  2>&1 | tee "$RUN_ROOT/suite.log"
```

## 3. 先看顶层结论

```bash
python3 - <<'PY' "$RUN_ROOT/report/text_kv_consistency_report.json"
import json, sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
print("suite classification:", data["classification"]["label"])
for context_length, payload in sorted(data["contexts"].items(), key=lambda item: int(item[0])):
    checks = payload["consistency_checks"]
    print(
        context_length,
        payload["context_label"],
        payload["classification"]["label"],
        "block_evidence",
        f'{checks["warm_hot_rows_with_block_evidence"]}/{checks["warm_hot_rows"]}',
        "missing_block_evidence",
        checks["warm_hot_rows_missing_block_evidence"],
        "fallback_only_rows",
        checks["fallback_only_rows"],
    )
PY
```

重点看：

- `classification`
- `warm_hot_rows_with_block_evidence`
- `warm_hot_rows_missing_block_evidence`
- `fallback_only_rows`

## 4. 检查是否真的生成了 8K / 16K workload

```bash
python3 - <<'PY' "$RUN_ROOT/workload/ctx8k/text_kv_consistency_workload_manifest.json" "$RUN_ROOT/workload/ctx16k/text_kv_consistency_workload_manifest.json"
import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path, "r", encoding="utf-8"))
    print(
        path,
        "context_length=", data["context_length"],
        "target_block_size_tokens=", data["target_block_size_tokens"],
        "target_prefix_ratios=", data["target_prefix_ratios"],
    )
PY
```

## 5. 检查 warm/hot 是否真的带了块级字段

```bash
python3 - <<'PY' "$RUN_ROOT"
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
for context in ("ctx8k", "ctx16k"):
    for condition in ("warm", "hot"):
        path = root / condition / context / "run" / "runtime_events_raw.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        hits = [
            row for row in rows
            if row.get("record_type") == "event"
            and row.get("action") == "cache_hit"
        ]
        loads = [
            row for row in rows
            if row.get("record_type") == "event"
            and row.get("action") == "cache_load"
        ]
        hit_with_blocks = [
            row for row in hits
            if any(key in (row.get("metadata") or {}) for key in ("block_count_hit", "token_count_hit", "block_ids_hit"))
        ]
        load_with_blocks = [
            row for row in loads
            if any(key in (row.get("metadata") or {}) for key in ("block_count_load", "token_count_load", "block_ids_load"))
        ]
        print(
            context,
            condition,
            "cache_hit_events=", len(hits),
            "cache_hit_with_block_metadata=", len(hit_with_blocks),
            "cache_load_events=", len(loads),
            "cache_load_with_block_metadata=", len(load_with_blocks),
        )
PY
```

## 6. 如何解释报告里的逐请求字段

每个请求现在会重点给出：

- `expected_reusable_blocks`
- `expected_reusable_tokens`
- `observed_kv_hit_blocks`
- `observed_kv_load_blocks`
- `observed_kv_store_blocks`
- `observed_kv_reuse_blocks`
- `observed_kv_reuse_ratio`
- `kv_block_gap_vs_expected`
- `block_evidence_complete`
- `evidence_source`

解释原则：

- `observed_kv_reuse_blocks` 优先取 `cache_hit` 的块数；若没有 hit 块统计，再退回 `cache_load`。
- `kv_block_gap_vs_expected = observed_kv_reuse_blocks - expected_reusable_blocks`
- `block_evidence_complete=false` 代表这个请求虽然出现了 cache signal，但缺少足够的块级统计，不能作为强证据。

## 7. 一致性判定标准

- `consistent`
  - cold 请求按预期 miss；
  - warm/hot 都拿到了 cache signal；
  - warm/hot 都拿到了块级证据；
  - `exact >= sim90 >= sim80` 的 KV 复用块数或复用比例单调成立；
  - 至少有一部分 warm/hot 的 TTFT 相比 cold 改善。

- `partially_consistent`
  - 趋势基本对，但有请求缺少块级证据，或只能依赖 log fallback。

- `inconsistent`
  - 文本复用趋势和 KV 复用趋势没有形成可信对应关系。

## 8. 推荐回传的关键文件

如果需要继续排查，请优先回传：

- `$RUN_ROOT/report/text_kv_consistency_report.json`
- `$RUN_ROOT/report/text_kv_consistency_report.md`
- `$RUN_ROOT/warm/ctx8k/run/runtime_events_raw.jsonl`
- `$RUN_ROOT/warm/ctx16k/run/runtime_events_raw.jsonl`
- `$RUN_ROOT/suite.log`
