import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from astrakv.benchmarks.task1_qasper_adapter import load_task1_qasper_workload, write_task1_qasper_artifacts


class Task1QasperAdapterTests(unittest.TestCase):
    def test_lossless_mapping_and_reuse_distance(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            archive = root / "task1.zip"
            write_archive(archive)
            workload = load_task1_qasper_workload(archive, "random")
            self.assertEqual(len(workload.rows), 200)
            self.assertEqual(workload.rows[0].prefix_id, "group-a")
            self.assertEqual(workload.rows[0].reuse_bucket, "medium")
            self.assertEqual(workload.rows[0].metadata["ground_truth"], "answer 0")
            artifacts = write_task1_qasper_artifacts(workload, root / "out")
            rows = [json.loads(line) for line in artifacts["workload"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["prompt"], "prompt 0")
            self.assertEqual(rows[0]["metadata"]["messages"][0]["content"], "prompt 0")
            distance = artifacts["reuse_distance"].read_text(encoding="utf-8")
            self.assertIn("reuse_distance", distance)

    def test_materialize_cli_accepts_validated_directory_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        task1_dir = root / "datasets" / "task1_qasper"
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "workload"
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark" / "materialize_task1_qasper_workload.py"),
                    "--task1-dir", str(task1_dir),
                    "--task1-workload", "random",
                    "--output-dir", str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            workload = output / "qasper_random_canonical_workload.jsonl"
            self.assertEqual(len(workload.read_text(encoding="utf-8").splitlines()), 200)


def write_archive(path: Path) -> None:
    prompts = []
    metadata = []
    for index in range(200):
        group = "group-a" if index < 2 else f"group-{index}"
        size = 2 if index < 2 else 1
        sample = f"sample-{index}"
        prompts.append({"request_id": f"req-{index}", "sample_id": sample, "prompt": f"prompt {index}", "order": index,
                        "reuse_group": group, "ground_truth": f"answer {index}", "answer": f"answer {index}",
                        "messages": [{"role": "user", "content": f"prompt {index}"}], "max_tokens": 128,
                        "dataset": "longbench", "task": "qasper", "audit": {}})
        metadata.append({"sample_id": sample, "reuse_group_size": size, "estimated_context_tokens": 10,
                         "estimated_kv_tokens": 10, "estimated_reusable_tokens": 5 if size > 1 else 0})
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("random", "grouped"):
            archive.writestr(f"prompts/qasper_{name}_prompts.jsonl", "\n".join(json.dumps(item) for item in prompts))
            archive.writestr(f"validation/{name}_prompt_validation.json", "{}")
        archive.writestr("metadata/qasper_metadata.jsonl", "\n".join(json.dumps(item) for item in metadata))
        archive.writestr("metadata/qasper_manifest.json", "{}")
        archive.writestr("metadata/qasper_sha256.json", "{}")


if __name__ == "__main__":
    unittest.main()
