import unittest
from pathlib import Path


class PolicyAbTtftSuiteEntrypointTests(unittest.TestCase):
    def test_entrypoint_wires_generation_paired_compare_and_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "entrypoints" / "run_policy_ab_ttft_suite.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        for required in (
            "generate_policy_ab_ttft_workload.py",
            "build_policy_ab_ttft_report.py",
            "compare_real_runs.py",
            "ASTRAKV_ENABLE_ONLINE_POLICY=true",
            "ASTRAKV_ENABLE_ONLINE_POLICY=false",
            "ASTRAKV_KV_CACHE_MEMORY_BYTES",
            "KV_CACHE_MEMORY_BYTES=\"2G\"",
            "--pair-id",
            "--pair-role",
            "--claim-scope online_control",
            "--output-tokens 1",
            "Qwen3-8B-AWQ",
        ):
            self.assertIn(required, content)

    def test_vllm_launcher_accepts_explicit_kv_cache_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "launch" / "launch_vllm_server.sh"
        content = script.read_text(encoding="utf-8")
        self.assertIn('KV_CACHE_MEMORY_BYTES="${ASTRAKV_KV_CACHE_MEMORY_BYTES:-}"', content)
        self.assertIn('CMD+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")', content)


if __name__ == "__main__":
    unittest.main()
