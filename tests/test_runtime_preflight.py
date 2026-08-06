import json
import sys
import tempfile
import unittest
from pathlib import Path

from astrakv.vm.mmap_kv_cache import MMapKVCache, vm_platform_available
from scripts.benchmark.inspect_dgx_runtime import inspect_runtime


class RuntimePreflightTests(unittest.TestCase):
    def test_preflight_reports_unavailable_packages_without_claiming_hook(self) -> None:
        args = type("Args", (), {
            "model": "model", "launch_command": "", "lmcache_config": "", "disk_path": ".",
        })()
        report = inspect_runtime(args)
        self.assertIn("vllm", report["packages"])
        self.assertIn("structured_event_hook_status", report["connector"])
        self.assertNotEqual(report["connector"]["structured_event_hook_status"], "available")
        json.dumps(report)

    def test_non_linux_vm_reports_platform_boundary_after_safe_import(self) -> None:
        if sys.platform.startswith("linux"):
            self.assertTrue(vm_platform_available())
            return
        self.assertFalse(vm_platform_available())
        with tempfile.TemporaryDirectory() as raw_tmp:
            with self.assertRaisesRegex(RuntimeError, "unsupported_platform"):
                MMapKVCache(total_blocks=1, block_size_bytes=4096, backing_file=str(Path(raw_tmp) / "kv.bin"))


if __name__ == "__main__":
    unittest.main()
