from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.text_kv_consistency import (
    _chat_token_ids,
    build_suite_report,
    build_workload_bundle,
    classify_consistency,
    write_workload_bundle,
)


class TextKvConsistencyTests(unittest.TestCase):
    def test_chat_token_ids_accepts_batch_encoding_like_objects(self) -> None:
        class BatchEncodingLike:
            input_ids = [11, 22, 33]

        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return BatchEncodingLike()

        self.assertEqual(
            _chat_token_ids(Tokenizer(), [{"role": "user", "content": "x"}]),
            [11, 22, 33],
        )

    def test_tokenizer_aligned_workload_backtracks_bucket_budget_to_shared_safe_length(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self._vocab: dict[str, int] = {}
                self._reverse: dict[int, str] = {}
                self._next_id = 100

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                pieces = [f" {piece}" for piece in str(text).split() if piece]
                token_ids: list[int] = []
                for piece in pieces:
                    token_id = self._vocab.get(piece)
                    if token_id is None:
                        token_id = self._next_id
                        self._next_id += 1
                        self._vocab[piece] = token_id
                        self._reverse[token_id] = piece
                    token_ids.append(token_id)
                return token_ids

            def decode(
                self,
                token_ids: list[int],
                skip_special_tokens: bool = True,
                clean_up_tokenization_spaces: bool = False,
            ) -> str:
                del skip_special_tokens, clean_up_tokenization_spaces
                return "".join(self._reverse[int(token_id)] for token_id in token_ids)

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                del tokenize, add_generation_prompt
                token_ids = [1]
                for message in messages:
                    token_ids.extend(self.encode(message["content"], add_special_tokens=False))
                    token_ids.append(2)
                exact_body_heavy = any(
                    "Bucket=exact." in str(message.get("content", ""))
                    and str(message.get("content", "")).count("alpha") >= 20
                    for message in messages
                )
                if exact_body_heavy:
                    token_ids.append(3)
                return {"input_ids": token_ids}

        bundle = build_workload_bundle(
            context_length=128,
            expected_output_tokens=8,
            block_size_tokens=4,
            tokenizer=Tokenizer(),
        )

        self.assertTrue(bundle.manifest["tokenizer_aligned"])
        self.assertEqual(bundle.manifest["prompt_token_budget"], 120)
        self.assertEqual(bundle.manifest["observed_prompt_tokens"], {"exact": 120, "sim90": 120, "sim80": 120})
        self.assertEqual(bundle.manifest["prompt_token_budget_by_bucket"]["exact"], 119)
        self.assertEqual(bundle.manifest["prompt_token_budget_by_bucket"]["sim90"], 120)
        self.assertEqual(bundle.manifest["prompt_token_budget_by_bucket"]["sim80"], 120)
        for row in bundle.warmup_rows + bundle.analysis_rows:
            self.assertEqual(row["metadata"]["observed_prompt_tokens"], 120)
            self.assertLessEqual(row["metadata"]["observed_prompt_tokens"] + row["expected_output_tokens"], 128)

    def test_workload_bundle_is_deterministic_and_block_aligned(self) -> None:
        first = build_workload_bundle(context_length=64, expected_output_tokens=8, block_size_tokens=4)
        second = build_workload_bundle(context_length=64, expected_output_tokens=8, block_size_tokens=4)

        self.assertEqual(first.analysis_rows, second.analysis_rows)
        self.assertEqual(first.warmup_rows, second.warmup_rows)
        self.assertEqual(first.replay_rows, second.replay_rows)
        self.assertEqual(first.manifest["expected_reusable_blocks"], {"exact": 16, "sim90": 14, "sim80": 13})
        self.assertEqual(first.manifest["target_mutation_start_block"], {"exact": 16, "sim90": 14, "sim80": 13})

        sim90_blocks = prompt_blocks(first.analysis_rows[1]["prompt"])
        sim80_blocks = prompt_blocks(first.analysis_rows[2]["prompt"])
        self.assertEqual(len(sim90_blocks), 16)
        self.assertTrue(sim90_blocks[13].startswith("n9s"))
        self.assertTrue(sim90_blocks[14].startswith("n9m"))
        self.assertTrue(sim80_blocks[12].startswith("n8s"))
        self.assertTrue(sim80_blocks[13].startswith("n8m"))

    def test_report_builder_groups_contexts_and_prefers_hook_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            contexts = ((320, "ctx8k"), (1280, "ctx16k"))

            for context_length, context_label in contexts:
                workload_dir = root / "workload" / context_label
                bundle = build_workload_bundle(
                    context_length=context_length,
                    expected_output_tokens=16,
                    block_size_tokens=16,
                )
                write_workload_bundle(workload_dir, bundle)

                text_observation_dir = root / "text_observation" / context_label
                text_observation_dir.mkdir(parents=True)
                observation_rows = []
                for row in bundle.analysis_rows:
                    target_ratio = float(row["metadata"]["target_prefix_ratio"])
                    token_count = int(row["metadata"]["block_count"]) * 16
                    observation_rows.append(
                        {
                            "evidence_class": "modeled_dataset_metadata",
                            "workflow_id": row["request_id"],
                            "parent_request_id": row["request_id"],
                            "subtask_index": 0,
                            "arrival_index": row["arrival_index"],
                            "dataset_id": "text-kv-consistency",
                            "workload_id": "text-kv-consistency-pairwise",
                            "adapter": "replay_jsonl",
                            "token_count": token_count,
                            "block_size_tokens": 16,
                            "block_hashes": [],
                            "historical_reused_tokens": int(round(token_count * target_ratio)),
                            "historical_reuse_count": 1,
                            "kv_bytes_per_token": 1,
                            "potential_kv_bytes": int(round(token_count * target_ratio)),
                        }
                    )
                write_jsonl(text_observation_dir / "workflow_reuse_observation_v1.jsonl", observation_rows)

                cold_ttft = {"exact-probe": 500.0, "sim90-probe": 540.0, "sim80-probe": 580.0}
                warm_ttft = {"exact-probe": 250.0, "sim90-probe": 300.0, "sim80-probe": 360.0}
                hot_ttft = {"exact-probe": 220.0, "sim90-probe": 280.0, "sim80-probe": 330.0}
                per_condition = {"cold": cold_ttft, "warm": warm_ttft, "hot": hot_ttft}
                hit_blocks_by_bucket = {"exact": 20, "sim90": 18, "sim80": 16}

                for condition, ttft_map in per_condition.items():
                    run_dir = root / condition / context_label / "run"
                    cache_dir = root / condition / context_label / "cache_events"
                    run_dir.mkdir(parents=True)
                    cache_dir.mkdir(parents=True)
                    request_rows = []
                    runtime_rows = []
                    for row in bundle.analysis_rows:
                        request_id = row["request_id"]
                        bucket = row["metadata"]["similarity_bucket"]
                        request_rows.append(
                            {
                                "request_id": request_id,
                                "status": "ok",
                                "error": "",
                                "ttft_ms": ttft_map[request_id],
                                "latency_ms": ttft_map[request_id] + 100.0,
                                "output_tokens_observed": 16,
                            }
                        )
                        if condition == "cold":
                            runtime_rows.append(
                                {
                                    "schema": "astrakv-backend-hook-v2",
                                    "record_type": "event",
                                    "run_id": f"{context_label}-{condition}",
                                    "event_id": f"{request_id}-miss",
                                    "request_id": request_id,
                                    "object_key": "prefix",
                                    "object_level": "prefix",
                                    "backend_object_id": "backend",
                                    "action": "cache_miss",
                                    "status": "completed",
                                    "timestamp_ns": 1,
                                    "binding_generation": 1,
                                    "metadata": {},
                                }
                            )
                        else:
                            block_count = hit_blocks_by_bucket[bucket]
                            runtime_rows.append(
                                {
                                    "schema": "astrakv-backend-hook-v2",
                                    "record_type": "event",
                                    "run_id": f"{context_label}-{condition}",
                                    "event_id": f"{request_id}-hit",
                                    "request_id": request_id,
                                    "object_key": "prefix",
                                    "object_level": "prefix",
                                    "backend_object_id": "backend",
                                    "action": "cache_hit",
                                    "status": "completed",
                                    "timestamp_ns": 1,
                                    "binding_generation": 1,
                                    "metadata": {
                                        "block_count_hit": block_count,
                                        "token_count_hit": block_count * 16,
                                    },
                                }
                            )
                    write_jsonl(run_dir / "request_results.jsonl", request_rows)
                    write_jsonl(run_dir / "runtime_events_raw.jsonl", runtime_rows)
                    write_jsonl(run_dir / "runtime_structured_events.jsonl", [])
                    write_jsonl(run_dir / "runtime_command_receipts.jsonl", [])
                    write_jsonl(cache_dir / "cache_events.jsonl", [])

            report = build_suite_report(root)
            self.assertEqual(report["classification"]["label"], "consistent")
            self.assertEqual(report["schema"], "astrakv-text-kv-consistency-report-v3")
            self.assertEqual(sorted(report["contexts"]), ["1280", "320"])
            self.assertEqual(report["contexts"]["320"]["classification"]["label"], "consistent")
            warm_exact = report["contexts"]["320"]["conditions"]["warm"]["requests"][0]
            self.assertEqual(warm_exact["evidence_source"], "runtime_events_raw")
            self.assertEqual(warm_exact["structured_cache_hit_blocks"], 20)
            self.assertEqual(warm_exact["expected_reusable_blocks"], 20)
            self.assertEqual(warm_exact["observed_kv_hit_blocks"], 20)
            self.assertEqual(warm_exact["observed_kv_reuse_blocks"], 20)
            self.assertEqual(warm_exact["kv_block_gap_vs_expected"], 0)
            self.assertEqual(warm_exact["observed_kv_reuse_ratio"], 1.0)
            self.assertTrue(warm_exact["block_evidence_complete"])

    def test_classification_downgrades_when_only_log_fallback_exists(self) -> None:
        result = classify_consistency(
            {
                "cold_expected_miss_count": 3,
                "cold_observed_miss_count": 3,
                "warm_hot_rows": 6,
                "warm_hot_rows_with_cache_signal": 6,
                "warm_hot_rows_with_block_evidence": 6,
                "warm_hot_rows_missing_block_evidence": 0,
                "warm_hot_monotonic_ok": True,
                "warm_hot_ttft_improved_count": 6,
                "fallback_only_rows": 6,
                "missing_artifacts": [],
            }
        )

        self.assertEqual(result["label"], "partially_consistent")

    def test_report_builder_downgrades_when_runtime_hits_lack_block_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            workload_dir = root / "workload" / "ctx8k"
            bundle = build_workload_bundle(
                context_length=320,
                expected_output_tokens=16,
                block_size_tokens=16,
            )
            write_workload_bundle(workload_dir, bundle)

            text_observation_dir = root / "text_observation" / "ctx8k"
            text_observation_dir.mkdir(parents=True)
            observation_rows = []
            for row in bundle.analysis_rows:
                token_count = int(row["metadata"]["block_count"]) * 16
                observation_rows.append(
                    {
                        "evidence_class": "modeled_dataset_metadata",
                        "workflow_id": row["request_id"],
                        "parent_request_id": row["request_id"],
                        "subtask_index": 0,
                        "arrival_index": row["arrival_index"],
                        "dataset_id": "text-kv-consistency",
                        "workload_id": "text-kv-consistency-pairwise",
                        "adapter": "replay_jsonl",
                        "token_count": token_count,
                        "block_size_tokens": 16,
                        "block_hashes": [],
                        "historical_reused_tokens": token_count,
                        "historical_reuse_count": 1,
                        "kv_bytes_per_token": 1,
                        "potential_kv_bytes": token_count,
                    }
                )
            write_jsonl(text_observation_dir / "workflow_reuse_observation_v1.jsonl", observation_rows)

            for condition, ttft_ms in (("cold", 500.0), ("warm", 250.0), ("hot", 220.0)):
                run_dir = root / condition / "ctx8k" / "run"
                cache_dir = root / condition / "ctx8k" / "cache_events"
                run_dir.mkdir(parents=True)
                cache_dir.mkdir(parents=True)
                request_rows = []
                runtime_rows = []
                for row in bundle.analysis_rows:
                    request_rows.append(
                        {
                            "request_id": row["request_id"],
                            "status": "ok",
                            "error": "",
                            "ttft_ms": ttft_ms,
                            "latency_ms": ttft_ms + 100.0,
                            "output_tokens_observed": 16,
                        }
                    )
                    runtime_rows.append(
                        {
                            "schema": "astrakv-backend-hook-v2",
                            "record_type": "event",
                            "run_id": f"ctx8k-{condition}",
                            "event_id": f"{row['request_id']}-{condition}",
                            "request_id": row["request_id"],
                            "object_key": "prefix",
                            "object_level": "prefix",
                            "backend_object_id": "backend",
                            "action": "cache_miss" if condition == "cold" else "cache_hit",
                            "status": "completed",
                            "timestamp_ns": 1,
                            "binding_generation": 1,
                            "metadata": {},
                        }
                    )
                write_jsonl(run_dir / "request_results.jsonl", request_rows)
                write_jsonl(run_dir / "runtime_events_raw.jsonl", runtime_rows)
                write_jsonl(run_dir / "runtime_structured_events.jsonl", [])
                write_jsonl(run_dir / "runtime_command_receipts.jsonl", [])
                write_jsonl(cache_dir / "cache_events.jsonl", [])

            report = build_suite_report(root)
            checks = report["contexts"]["320"]["consistency_checks"]
            self.assertEqual(report["classification"]["label"], "partially_consistent")
            self.assertEqual(checks["warm_hot_rows_with_block_evidence"], 0)
            self.assertEqual(checks["warm_hot_rows_missing_block_evidence"], 6)
            warm_exact = report["contexts"]["320"]["conditions"]["warm"]["requests"][0]
            self.assertEqual(warm_exact["observed_kv_reuse_blocks"], 0)
            self.assertEqual(warm_exact["kv_block_gap_vs_expected"], -20)
            self.assertFalse(warm_exact["block_evidence_complete"])


def prompt_blocks(prompt: str) -> list[str]:
    sections = prompt.split("\n\n")
    if len(sections) < 3:
        return []
    return [line for line in sections[1].splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
