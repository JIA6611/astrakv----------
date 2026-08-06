from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from astrakv.benchmarks.task1_qasper_adapter import load_task1_qasper_directory


class Task1DirectoryAdapterTests(unittest.TestCase):
    def test_loads_immutable_grouped_directory_package(self) -> None:
        root = Path("datasets/task1_qasper")
        prompts = root / "prompts/qasper_grouped_prompts.jsonl"
        before = hashlib.sha256(prompts.read_bytes()).hexdigest()

        workload = load_task1_qasper_directory(root, "grouped")

        self.assertEqual(len(workload.rows), 200)
        self.assertEqual(workload.audit["input_kind"], "directory_package")
        self.assertEqual(workload.audit["prompt_sha256"], before)
        self.assertEqual(workload.rows[0].metadata["workload_type"], "grouped")
        self.assertEqual(hashlib.sha256(prompts.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
