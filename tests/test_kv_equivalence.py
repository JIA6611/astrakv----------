import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow
from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge, _exact_token_sequence_hash
from scripts.benchmark.materialize_kv_equivalence_workload import materialize, select_prompt_source


def source() -> RuntimeWorkloadRow:
    return RuntimeWorkloadRow(
        request_id="source", prompt="exact prompt", prefix_id="prefix", prefix_hash="prefix-hash",
        cache_key="cache-key", arrival_index=0, reuse_ratio=1.0, reuse_bucket="high",
        context_length=1024, expected_output_tokens=128,
        metadata={"exact_prefix": True, "messages": [{"role": "user", "content": "exact prompt"}]},
    )


class KVEquivalenceTests(unittest.TestCase):
    def test_prompt_source_synthesizes_exact_prefix_row(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            prompts_path = Path(raw_tmp) / "grouped_prompts.jsonl"
            prompts_path.write_text(json.dumps({
                "schema": "astra-workload-prompt-v1",
                "request_id": "qasper-grouped-000000",
                "sample_id": "sample-1",
                "workload_type": "grouped",
                "order": 0,
                "reuse_group": "group-1",
                "shared_context": True,
                "prompt": "shared context question",
                "metadata": {"estimated_prompt_tokens": 1024},
            }) + "\n", encoding="utf-8")
            source = select_prompt_source(prompts_path, "")
            self.assertTrue(source.metadata["exact_prefix"])
            self.assertEqual(
                source.metadata["messages"],
                [{"role": "user", "content": "shared context question"}],
            )
            self.assertEqual(source.cache_key, source.prefix_hash)
            self.assertTrue(source.cache_key.startswith("sha256:"))
            self.assertGreater(source.context_length, 0)
            rows = materialize(source, output_tokens=8)
            self.assertEqual(rows[-1].metadata["kv_core_equivalence_mode"], "force_recompute")
            self.assertEqual(rows[0].metadata["exact_prefix"], True)

    def test_workload_preserves_exact_input_and_only_recompute_has_test_mode(self):
        rows = materialize(source(), output_tokens=8)
        self.assertEqual([row.case for row in rows], [
            "kv_equivalence_seed", "kv_equivalence_loaded", "kv_equivalence_recompute",
        ])
        self.assertEqual({row.prompt for row in rows}, {"exact prompt"})
        self.assertEqual({row.cache_key for row in rows}, {"cache-key"})
        self.assertEqual({row.metadata["generation_seed"] for row in rows}, {0})
        self.assertEqual(rows[-1].metadata["kv_core_equivalence_mode"], "force_recompute")
        self.assertEqual(rows[1].sleep_before_s, 1.0)

    def test_force_recompute_is_cross_process_one_shot_and_exact_token_scoped(self):
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_KV_CORE_EQUIVALENCE_TEST": "true",
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False):
            ingress_bridge = VendorCallbackBridge(SimpleNamespace())
            tokens = (1, 2, 3, 4)
            ingress_bridge.ingress_request(SimpleNamespace(
                request_id="probe", metadata={
                    "exact_token_ids": tokens, "kv_core_equivalence_mode": "force_recompute",
                },
            ))
            exact_hash = _exact_token_sequence_hash(tokens)
            intent_dir = Path(raw_tmp) / "equivalence_recompute_intents"
            intent_path = intent_dir / f"{exact_hash}.json"
            self.assertTrue(intent_path.is_file())
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            self.assertEqual(intent["schema"], "astrakv-kv-equivalence-intent-v1")
            self.assertEqual(intent["exact_token_sequence_hash"], exact_hash)
            self.assertEqual(intent["logical_request_id"], "probe")
            self.assertGreater(intent["expires_at_ns"], intent["created_at_ns"])

            # A fresh bridge simulates the separate EngineCore process. It can
            # observe only the state-dir record, not ingress process memory.
            engine_core_bridge = VendorCallbackBridge(SimpleNamespace())
            wrong_hash = _exact_token_sequence_hash((1, 2, 3))
            self.assertFalse(engine_core_bridge._consume_equivalence_force_recompute(
                "native", len(tokens), wrong_hash,
            ))
            self.assertTrue(engine_core_bridge._consume_equivalence_force_recompute(
                "native", len(tokens), exact_hash,
            ))
            self.assertFalse(intent_path.exists())
            consumed_path = intent_dir / f"{exact_hash}.consumed.json"
            self.assertTrue(consumed_path.is_file())
            self.assertEqual(
                json.loads(consumed_path.read_text(encoding="utf-8")),
                intent,
            )
            self.assertFalse(engine_core_bridge._consume_equivalence_force_recompute(
                "native", len(tokens), exact_hash,
            ))
            records = [json.loads(line) for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["reason"], "equivalence_probe_force_recompute")
            self.assertTrue(records[-1]["test_only"])
            self.assertEqual(records[-1]["equivalence_intent_path"], str(consumed_path))

    def test_force_recompute_rejects_expired_cross_process_intent(self):
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_KV_CORE_EQUIVALENCE_TEST": "true",
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False):
            tokens = (5, 6, 7, 8)
            exact_hash = _exact_token_sequence_hash(tokens)
            ingress_bridge = VendorCallbackBridge(SimpleNamespace())
            ingress_bridge.ingress_request(SimpleNamespace(
                request_id="expired-probe", metadata={
                    "exact_token_ids": tokens, "kv_core_equivalence_mode": "force_recompute",
                },
            ))
            intent_path = (
                Path(raw_tmp) / "equivalence_recompute_intents" / f"{exact_hash}.json"
            )
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            intent["expires_at_ns"] = 0
            intent_path.write_text(json.dumps(intent), encoding="utf-8")

            engine_core_bridge = VendorCallbackBridge(SimpleNamespace())
            self.assertFalse(engine_core_bridge._consume_equivalence_force_recompute(
                "native", len(tokens), exact_hash,
            ))
            self.assertTrue(intent_path.is_file())
            self.assertFalse(intent_path.with_name(f"{exact_hash}.consumed.json").exists())


if __name__ == "__main__":
    unittest.main()
