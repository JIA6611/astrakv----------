import unittest
from pathlib import Path


class TextKvConsistencySuiteEntrypointTests(unittest.TestCase):
    def test_entrypoint_wires_generation_observation_cache_extraction_and_reporting(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "entrypoints" / "run_text_kv_consistency_suite.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        for required in (
            "generate_text_kv_consistency_workload.py",
            "observe_workflow_reuse.py",
            "extract_cache_events.py",
            "compare_real_runs.py",
            "build_text_kv_consistency_report.py",
            "cold",
            "warm",
            "hot",
            "--context-lengths",
            "ctx8k",
            "ctx16k",
            "shared prefix ratio",
            "pairwise_reference_replay.jsonl",
            "analysis_workload.jsonl",
            "warmup_workload.jsonl",
        ):
            self.assertIn(required, content)
        self.assertIn('--unpaired', content)
        self.assertIn('MAX_MODEL_LEN="16384"', content)
        self.assertIn('CONTEXT_LENGTHS="8192,16384"', content)
        self.assertIn('ASTRAKV_ENABLE_ONLINE_POLICY=false', content)
        self.assertIn('ASTRAKV_PREFIX_CACHING=true', content)


if __name__ == "__main__":
    unittest.main()
