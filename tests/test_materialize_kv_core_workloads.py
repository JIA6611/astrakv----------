"""Unit tests for the canonical KV-Core workload generator."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.runtime_workload import load_runtime_workload_jsonl  # noqa: E402
from scripts.benchmark.materialize_kv_core_canonical_workloads import (  # noqa: E402
    PromptSource,
    build_constrained_kv_churn,
    build_queued_concurrency,
    build_random_no_reuse,
    build_repeated_long_prefix,
)


def _source(dataset: str, group: str, prompt: str, *, shared: bool = True) -> PromptSource:
    return PromptSource(
        dataset=dataset,
        prompt=prompt,
        prompt_hash=f"sha256:{abs(hash(prompt))}",
        reuse_group=group,
        shared_context=shared,
        estimated_prompt_tokens=max(16, len(prompt) // 4),
    )


def _grouped(seed: int = 7) -> list[list[PromptSource]]:
    rng = random.Random(seed)
    groups: list[list[PromptSource]] = []
    for group_index in range(40):
        group = f"g{group_index:03d}"
        base = _source("qasper", group, f"Document {group_index} shared context. " * 20)
        extra = _source("qasper", group, f"Document {group_index} shared context. " * 20)
        rng.shuffle([base, extra])
        groups.append([base, extra])
    rng.shuffle(groups)
    return groups


def _unique_sources() -> list[PromptSource]:
    return [
        _source("qasper", f"u{index:03d}", f"Unique prompt {index} with no reuse. " * 8, shared=False)
        for index in range(60)
    ]


class MaterializeKVCoreWorkloadsTests(unittest.TestCase):
    def test_repeated_long_prefix_contract_and_reuse(self) -> None:
        rows = build_repeated_long_prefix(_grouped(), groups=4, revisits=5, output_tokens=128)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["request_id"] for row in rows}), 20)
        self.assertEqual(len({row["arrival_index"] for row in rows}), 20)
        groups = {row["prefix_id"] for row in rows}
        self.assertEqual(len(groups), 4)
        seen_groups: set[str] = set()
        for row in rows:
            self.assertGreater(row["context_length"], 0)
            if row["prefix_id"] not in seen_groups:
                seen_groups.add(row["prefix_id"])
                self.assertEqual(row["reuse_ratio"], 0.0)
                self.assertEqual(row["reuse_bucket"], "none")
                self.assertEqual(row["prefetch_lead_s"], 0.0)
            else:
                self.assertEqual(row["reuse_bucket"], "high")
                self.assertGreater(row["prefetch_lead_s"], 0.0)
                self.assertEqual(row["reuse_ratio"], 0.95)

    def test_random_no_reuse_is_unique_and_never_prefetched(self) -> None:
        rows = build_random_no_reuse(_unique_sources(), requests=40, output_tokens=128)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["prefix_id"] for row in rows}), 40)
        for row in rows:
            self.assertEqual(row["reuse_ratio"], 0.0)
            self.assertEqual(row["reuse_bucket"], "none")
            self.assertEqual(row["prefetch_lead_s"], 0.0)

    def test_constrained_kv_churn_interleaves_groups(self) -> None:
        rows = build_constrained_kv_churn(_grouped(), groups=8, revisits=3, output_tokens=16)
        self.assertEqual(len(rows), 24)
        first_wave = {row["prefix_id"] for row in rows[:8]}
        self.assertEqual(len(first_wave), 8)
        self.assertGreater(rows[8]["prefetch_lead_s"], 0.0)
        for row in rows[:8]:
            self.assertEqual(row["prefetch_lead_s"], 0.0)

    def test_queued_concurrency_round_robin(self) -> None:
        rows = build_queued_concurrency(_grouped(), groups=4, revisits=6, output_tokens=16)
        self.assertEqual(len(rows), 24)
        first_wave = {row["prefix_id"] for row in rows[:4]}
        self.assertEqual(len(first_wave), 4)
        self.assertTrue(all(row["prefetch_lead_s"] > 0.0 for row in rows[4:]))
        self.assertTrue(all(row["sleep_before_s"] == 0.0 for row in rows))

    def test_cli_generates_valid_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp) / "workload_prompts" / "qasper"
            prompts_dir.mkdir(parents=True)
            with (prompts_dir / "grouped_prompts.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(30):
                    payload = {
                        "schema": "astra-workload-prompt-v1",
                        "prompt": f"Paper {index} shared context. " * 20,
                        "reuse_group": f"ctx{index:03d}",
                        "shared_context": True,
                        "metadata": {"estimated_prompt_tokens": 128 + index},
                    }
                    handle.write(json.dumps(payload) + "\n")
            with (prompts_dir / "random_prompts.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(30):
                    payload = {
                        "schema": "astra-workload-prompt-v1",
                        "prompt": f"Random no-reuse prompt {index}. " * 8,
                        "metadata": {"estimated_prompt_tokens": 64 + index},
                    }
                    handle.write(json.dumps(payload) + "\n")
            output_dir = Path(tmp) / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "benchmark" / "materialize_kv_core_canonical_workloads.py"),
                    "--prompts-dir",
                    str(Path(tmp) / "workload_prompts"),
                    "--datasets",
                    "qasper",
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "0",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "repeated_long_prefix",
                "random_no_reuse",
                "constrained_kv_churn",
                "queued_concurrency",
            ):
                path = output_dir / f"{name}.jsonl"
                self.assertTrue(path.exists(), name)
                rows = load_runtime_workload_jsonl(path)
                self.assertGreater(len(rows), 0, name)
            manifest = json.loads((output_dir / "kv_core_workload_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "astrakv-kv-core-workload-manifest-v1")
            self.assertEqual(len(manifest["workloads"]), 4)


if __name__ == "__main__":
    unittest.main()
