from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.runtime.third_party_patch import PATCH_ID, PATCH_SCHEMA, REQUIRED_CALLBACKS, verify_connector_patch


class ThirdPartyPatchVerificationTests(unittest.TestCase):
    def test_accepts_exact_versions_hashes_marker_and_callback_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            source = root / "connector.py"
            source.write_text("patched connector", encoding="utf-8")
            manifest = root / "deployment.json"
            manifest.write_text(json.dumps({
                "schema": PATCH_SCHEMA, "patch_id": PATCH_ID,
                "callbacks": list(REQUIRED_CALLBACKS),
                "patch_marker": {"id": PATCH_ID, "path": "/opt/vllm/.astrakv-kv-core"},
                "source_files": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
            }), encoding="utf-8")
            result = verify_connector_patch(
                manifest, distribution_version=lambda package: "0.23.0" if package == "vllm" else "0.4.7",
                callback_smoke=lambda: True,
            )
            self.assertTrue(result.compatible)

    def test_fails_closed_when_the_callback_smoke_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            manifest = Path(raw_tmp) / "deployment.json"
            manifest.write_text("{}", encoding="utf-8")
            result = verify_connector_patch(manifest, distribution_version=lambda _package: "0")
            self.assertFalse(result.compatible)
            self.assertIn("callback_smoke_missing", result.reasons)


if __name__ == "__main__":
    unittest.main()
