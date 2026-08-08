from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from astrakv.runtime.kv_runtime_core import KVCompatibilityKey, exact_token_prefix_hash
from astrakv.runtime.offline_kv_profile import (
    OfflineKVProfileIndex,
    OfflinePrefixProfile,
    build_manifest,
    validate_qwen3_8b_target,
)


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

    def test_profile_index_requires_correct_exact_compatibility(self) -> None:
        profile = OfflinePrefixProfile(
            "sample", self.key(), "native", 2, 1.0, 2.7,
            {"0": 1.0}, {"0": 1.0}, {"0": 1.0}, {"0": 0.0}, {"0": 0.0},
            layer_sensitivity={"0": {"teacher_forced_loss_delta": 0.4}},
        )
        target = {
            "model_id": "Qwen3-8B", "model_revision": "r",
            "tokenizer_revision": "t", "chat_template_revision": "c",
            "dtype": "bfloat16", "block_size_tokens": 16,
            "chunk_size_tokens": 256, "quantization": "unquantized",
        }
        manifest = build_manifest(
            target=target, hardware={"memory_model": "uma"},
            workload_sha256="x", profiles=[profile],
        )
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "profile.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            index = OfflineKVProfileIndex.load(path)
        hint = index.hint_for(self.key())
        self.assertIsNotNone(hint)
        self.assertEqual(hint.sensitivity_rank, 0.5)
        mismatched = KVCompatibilityKey(
            "Qwen3-8B", "other", "t", "c", "torch.bfloat16", "{}",
            "base", "paged", 16, 256, "all", self.key().prefix_hash,
        )
        self.assertIsNone(index.hint_for(mismatched))

    def test_profile_index_rejects_failed_correctness(self) -> None:
        manifest = {
            "schema": "astrakv-kv-core-offline-profile-v2",
            "correctness": {"passed": False},
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "profile.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "correctness"):
                OfflineKVProfileIndex.load(path)


if __name__ == "__main__":
    unittest.main()
