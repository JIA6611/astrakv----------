import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.benchmarks.runtime_workload import RuntimeWorkloadRow
from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge, _exact_token_sequence_hash
from scripts.benchmark.materialize_kv_equivalence_workload import materialize


def source() -> RuntimeWorkloadRow:
    return RuntimeWorkloadRow(
        request_id="source", prompt="exact prompt", prefix_id="prefix", prefix_hash="prefix-hash",
        cache_key="cache-key", arrival_index=0, reuse_ratio=1.0, reuse_bucket="high",
        context_length=1024, expected_output_tokens=128,
        metadata={"exact_prefix": True, "messages": [{"role": "user", "content": "exact prompt"}]},
    )


class KVEquivalenceTests(unittest.TestCase):
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

    def test_force_recompute_is_one_shot_and_exact_token_scoped(self):
        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(os.environ, {
            "ASTRAKV_KV_CORE_EQUIVALENCE_TEST": "true",
            "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
        }, clear=False):
            bridge = VendorCallbackBridge(SimpleNamespace())
            tokens = (1, 2, 3, 4)
            bridge.ingress_request(SimpleNamespace(
                request_id="probe", metadata={
                    "exact_token_ids": tokens, "kv_core_equivalence_mode": "force_recompute",
                },
            ))
            exact_hash = _exact_token_sequence_hash(tokens)
            self.assertTrue(bridge._consume_equivalence_force_recompute("native", len(tokens), exact_hash))
            self.assertFalse(bridge._consume_equivalence_force_recompute("native", len(tokens), exact_hash))
            self.assertFalse(bridge._consume_equivalence_force_recompute("native", len(tokens), _exact_token_sequence_hash((1, 2, 3))))
            records = [json.loads(line) for line in (Path(raw_tmp) / "kv_core_policy_decisions.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["reason"], "equivalence_probe_force_recompute")
            self.assertTrue(records[-1]["test_only"])


if __name__ == "__main__":
    unittest.main()
