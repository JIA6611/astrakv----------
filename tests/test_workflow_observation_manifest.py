from __future__ import annotations

import hashlib
import unittest

from scripts.benchmark.observe_workflow_reuse import tokenizer_manifest_metadata


class FakeTokenizer:
    name_or_path = "/opt/models/Qwen3-8B"
    init_kwargs = {"revision": "model-revision-1"}
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


class WorkflowObservationManifestTests(unittest.TestCase):
    def test_records_tokenizer_revision_and_chat_template_hash(self):
        metadata = tokenizer_manifest_metadata(FakeTokenizer())

        self.assertEqual(metadata["tokenizer_identifier"], "/opt/models/Qwen3-8B")
        self.assertEqual(metadata["tokenizer_revision"], "model-revision-1")
        self.assertEqual(
            metadata["chat_template_sha256"],
            hashlib.sha256(FakeTokenizer.chat_template.encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
