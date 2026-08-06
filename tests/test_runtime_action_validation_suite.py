import unittest
from pathlib import Path


class RuntimeActionValidationEntrypointTests(unittest.TestCase):
    def test_hot_load_smoke_requires_request_level_ttft_and_paired_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "entrypoints" / "run_runtime_action_hot_load_smoke.sh").read_text(encoding="utf-8")
        for required in (
            "hot-load-revisit must report a non-null TTFT",
            "hot-load-revisit must observe generated output tokens",
            "paired_run_manifest.json",
            "functional_and_performance_pass",
            "functional_pass_performance_inconclusive",
            "functional_fail",
        ):
            self.assertIn(required, script)

    def test_validation_suite_covers_all_runtime_action_scenarios(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "entrypoints" / "run_runtime_action_validation_suite.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        for required in (
            "hot_load",
            "cpu_offload",
            "ssd_prefetch",
            "cold_drop",
            "recompute_bias",
            "evict_cold_disk",
            "run_runtime_action_hot_load_smoke.sh",
            "ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH=true",
            "missing recompute no-dispatch evidence",
            "suite_summary.json",
            "scenario_evidence.json",
            "completed_receipt",
            "strategy_no_dispatch",
            "evidence_type",
            "--scenarios",
            "request identity and store/release lifecycle",
        ):
            self.assertIn(required, content)
        self.assertNotIn('if [[ "$scenario" == "cpu_offload" ]]; then', content)
        self.assertNotIn('if [[ "$scenario" == "cpu_offload" || "$scenario" == "ssd_prefetch" ]]; then', content)
        self.assertIn('"${benchmark_args[@]}"', content)
        self.assertNotIn('"${benchmark_args[@]}" \\', content)
        self.assertIn('ASTRAKV_RUNTIME_DISABLE_NATIVE_REQUEST_LOAD=true', content)
        self.assertIn('one two-tier topology', content)
        self.assertIn('configs/lmcache_runtime_action_validation.yaml', content)


if __name__ == "__main__":
    unittest.main()
