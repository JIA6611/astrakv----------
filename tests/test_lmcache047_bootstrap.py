import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from astrakv.runtime.lmcache047_bootstrap import _is_vllm_engine_child, install_from_environment, installed_runtime_control_host
from astrakv.runtime.lmcache047_runtime_patch import patch_usage_context_cpu_info


class LMCache047BootstrapTests(unittest.TestCase):
    def test_engine_child_detection_excludes_resource_tracker(self):
        parent = b"python\\0-m\\0vllm.entrypoints.openai.api_server\\0"
        with patch.object(os, "getppid", return_value=42), patch("builtins.open", mock_open(read_data=parent)), patch.object(sys, "argv", ["-c"]):
            self.assertFalse(_is_vllm_engine_child())
        with patch.object(os, "getppid", return_value=42), patch("builtins.open", mock_open(read_data=parent)), patch.object(sys, "argv", ["-c", "--multiprocessing-fork"]):
            self.assertTrue(_is_vllm_engine_child())

    def test_usage_context_cpuinfo_failure_uses_telemetry_only_fallback(self):
        class UsageContext:
            def _get_cpu_info(self):
                raise ValueError("cpuinfo failed")

        patch_usage_context_cpu_info(UsageContext)
        count, cpu_type, family = UsageContext()._get_cpu_info()
        self.assertIsInstance(count, int)
        self.assertTrue(cpu_type)
        self.assertEqual(family, "")

    def test_disabled_bootstrap_does_not_install(self):
        with patch.dict(os.environ, {"ASTRAKV_ENABLE_LMCACHE047_HOOKS": "false"}, clear=False):
            self.assertFalse(install_from_environment(installer=lambda sink: object()))

    def test_enabled_bootstrap_installs_and_writes_event(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "events.jsonl"
            seen = []
            def installer(sink):
                sink({"action": "cache_hit"})
                seen.append(True)
                return object()
            with patch.dict(os.environ, {
                "ASTRAKV_ENABLE_LMCACHE047_HOOKS": "true",
                "ASTRAKV_LMCACHE047_EVENTS": str(output),
            }, clear=False):
                self.assertTrue(install_from_environment(installer=installer))
            self.assertEqual(seen, [True])
            self.assertIn("cache_hit", output.read_text(encoding="utf-8"))

    def test_control_host_mode_injects_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            captured = {}
            def installer(sink, **kwargs):
                captured.update(kwargs)
                return object()
            with patch("astrakv.runtime.lmcache047_bootstrap._INSTALLED", False), patch("astrakv.runtime.lmcache047_bootstrap._is_vllm_engine_child", return_value=True), patch.dict(os.environ, {
                "ASTRAKV_ENABLE_LMCACHE047_HOOKS": "true",
                "ASTRAKV_RUNTIME_CONTROL_RUN_ID": "run-a",
                "ASTRAKV_RUNTIME_CONTROL_STATE_DIR": raw_tmp,
                "ASTRAKV_RUNTIME_CONTROL_SECRET_HEX": "ab" * 32,
                "ASTRAKV_RUNTIME_CONTROL_ENGINE_ID": "engine",
                "ASTRAKV_RUNTIME_CONTROL_WORKER_ID": "worker",
                "ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE": "engine_child",
            }, clear=False):
                self.assertTrue(install_from_environment(installer=installer))
            self.assertIsNotNone(captured["binding_registry"])
            self.assertIsNotNone(captured["request_context_consumer"])
            self.assertTrue(callable(captured["runtime_request_identity_provider"]))
            self.assertTrue((Path(raw_tmp) / "backend_capabilities.json").exists())
            host = installed_runtime_control_host()
            self.assertIsNotNone(host)
            if host is not None:
                host.close()

    def test_engine_child_scope_leaves_api_process_unpatched(self):
        with patch("astrakv.runtime.lmcache047_bootstrap._INSTALLED", False), patch("astrakv.runtime.lmcache047_bootstrap._is_vllm_engine_child", return_value=False), patch.dict(os.environ, {
            "ASTRAKV_ENABLE_LMCACHE047_HOOKS": "true",
            "ASTRAKV_RUNTIME_CONTROL_RUN_ID": "run-a",
            "ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE": "engine_child",
        }, clear=False):
            self.assertFalse(install_from_environment(installer=lambda *_args, **_kwargs: self.fail("must not install")))


if __name__ == "__main__":
    unittest.main()
