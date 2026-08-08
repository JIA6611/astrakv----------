from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DGXGPUSSDConfigTests(unittest.TestCase):
    def test_disk_configuration_disables_cpu_cache_layer(self):
        disk = yaml.safe_load((ROOT / "configs" / "lmcache_disk_example.yaml").read_text(encoding="utf-8"))
        run = yaml.safe_load((ROOT / "configs" / "dgx_spark_lmcache_disk.yaml").read_text(encoding="utf-8"))

        self.assertFalse(disk["local_cpu"])
        self.assertGreater(disk["max_local_cpu_size"], 0.0)
        self.assertEqual(run["lmcache"]["topology"], "gpu_ssd_only")
        self.assertFalse(run["lmcache"]["cpu_layer_enabled"])

    def test_launch_prefers_project_virtualenv_python(self):
        script = (ROOT / "scripts" / "launch" / "launch_vllm_server.sh").read_text(encoding="utf-8")

        self.assertIn('DEFAULT_PYTHON="$ROOT/.venv/bin/python"', script)
        self.assertIn('git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir', script)
        self.assertIn('PYTHON="${ASTRAKV_PYTHON:-$DEFAULT_PYTHON}"', script)
        self.assertIn('export PATH="$(dirname "$PYTHON"):${PATH}"', script)


    def test_launch_exposes_isolated_cache_controls(self):
        script = (ROOT / "scripts" / "launch" / "launch_vllm_server.sh").read_text(encoding="utf-8")

        self.assertIn("ASTRAKV_PREFIX_CACHING", script)
        self.assertIn("--no-enable-prefix-caching", script)
        self.assertIn("ASTRAKV_VLLM_DEV_MODE", script)
        self.assertIn("export VLLM_SERVER_DEV_MODE=1", script)

    def test_kv_core_suite_provisions_disk_staging_and_rejects_pressure(self):
        script = (
            ROOT / "scripts" / "entrypoints" / "run_kv_core_controlled_suite.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('local local_cpu="false" local_cpu_size="2.0"', script)
        self.assertIn("assert_lmcache_runtime_healthy", script)
        self.assertIn("No eviction candidates found in local cpu backend", script)


if __name__ == "__main__":
    unittest.main()
