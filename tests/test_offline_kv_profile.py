from __future__ import annotations

import unittest

from astrakv.runtime.kv_runtime_core import KVCompatibilityKey, exact_token_prefix_hash
from astrakv.runtime.offline_kv_profile import OfflinePrefixProfile, build_manifest, validate_qwen3_8b_target


class OfflineKVProfileTests(unittest.TestCase):
    def key(self) -> KVCompatibilityKey:
        return KVCompatibilityKey("Qwen3-8B", "r", "t", "c", "bfloat16", "{}", "base", "paged", 16, 256, "all", exact_token_prefix_hash([1, 2]), "e", "w")

    def test_rejects_awq_profile_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "AWQ"):
            validate_qwen3_8b_target(model_id="Qwen3-8B", quantization="awq")

    def test_manifest_carries_exact_compatibility_and_online_scope(self) -> None:
        profile = OfflinePrefixProfile("sample", self.key(), "native", 2, 1.0, 2.7, {"0": 1.0}, {"0": 1.0}, {"0": 1.0}, {"0": 0.0}, {"0": 0.0})
        manifest = build_manifest(target={"model_id": "Qwen3-8B", "quantization": "unquantized"}, hardware={"memory_model": "uma"}, workload_sha256="x", profiles=[profile])
        self.assertEqual(manifest["profiles"][0]["compatibility_key"]["prefix_hash"], self.key().prefix_hash)
        self.assertIn("layer_kv_skipping", manifest["online_scope"]["must_not_influence"])


if __name__ == "__main__":
    unittest.main()
