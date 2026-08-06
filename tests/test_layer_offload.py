"""Tests for the layer offloading PoC module.

These tests validate the data structures and logic without requiring a GPU.
GPU-dependent tests are skipped when CUDA is not available.
"""

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from astrakv.vm.layer_offload import (
    LayerOffloadConfig,
    LayerOffloadManager,
    LayerOffloadResult,
)


class LayerOffloadResultTests(unittest.TestCase):
    """Unit tests for LayerOffloadResult dataclass."""

    def test_result_to_record(self) -> None:
        result = LayerOffloadResult(
            window_size=4,
            gpu_peak_mb=5120.5,
            latency_ms=1234.5,
            layer_load_avg_ms=12.3,
            metadata={"model": "test"},
        )
        record = result.to_record()
        self.assertEqual(record["window_size"], 4)
        self.assertEqual(record["gpu_peak_mb"], 5120.5)
        self.assertEqual(record["latency_ms"], 1234.5)
        self.assertEqual(record["layer_load_avg_ms"], 12.3)
        self.assertEqual(record["metadata"]["model"], "test")

    def test_default_result(self) -> None:
        result = LayerOffloadResult(window_size=1, gpu_peak_mb=0, latency_ms=0)
        record = result.to_record()
        self.assertEqual(record["window_size"], 1)
        self.assertEqual(record["gpu_peak_mb"], 0)


class LayerOffloadConfigTests(unittest.TestCase):
    """Unit tests for LayerOffloadConfig."""

    def test_default_config(self) -> None:
        cfg = LayerOffloadConfig(model_name="test-model")
        self.assertEqual(cfg.model_name, "test-model")
        self.assertEqual(cfg.gpu_layer_window, 4)
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.cpu_device, "cpu")
        self.assertEqual(cfg.dtype_str, "float16")

    @unittest.skipUnless(_HAS_TORCH, "torch not available")
    def test_default_config_dtype_resolution(self) -> None:
        cfg = LayerOffloadConfig(model_name="test-model")
        self.assertIs(cfg.dtype, torch.float16)

    @unittest.skipUnless(_HAS_TORCH, "torch not available")
    def test_custom_config(self) -> None:
        cfg = LayerOffloadConfig(
            model_name="custom-model",
            gpu_layer_window=8,
            device="cuda:1",
            dtype_str="float32",
        )
        self.assertEqual(cfg.gpu_layer_window, 8)
        self.assertEqual(cfg.device, "cuda:1")
        self.assertIs(cfg.dtype, torch.float32)


class LayerOffloadManagerInitTests(unittest.TestCase):
    """Tests for LayerOffloadManager initialization."""

    def test_init_with_config(self) -> None:
        cfg = LayerOffloadConfig(model_name="test-model")
        mgr = LayerOffloadManager(config=cfg)
        self.assertEqual(mgr.config.model_name, "test-model")

    def test_init_with_kwargs(self) -> None:
        mgr = LayerOffloadManager(model_name="test-model", gpu_layer_window=2)
        self.assertEqual(mgr.config.model_name, "test-model")
        self.assertEqual(mgr.config.gpu_layer_window, 2)


class LayerOffloadSaveResultsTests(unittest.TestCase):
    """Tests for the save_results utility."""

    def test_save_results_creates_files(self) -> None:
        results = {
            1: LayerOffloadResult(window_size=1, gpu_peak_mb=1000, latency_ms=500),
            2: LayerOffloadResult(window_size=2, gpu_peak_mb=2000, latency_ms=400),
        }
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "test_output"
            LayerOffloadManager.save_results(results, out_dir)

            csv_path = out_dir / "layer_offload_results.csv"
            report_path = out_dir / "layer_offload_report.md"

            self.assertTrue(csv_path.exists(), f"CSV not found: {csv_path}")
            self.assertTrue(report_path.exists(), f"Report not found: {report_path}")

            csv_content = csv_path.read_text()
            self.assertIn("window_size", csv_content)
            self.assertIn("gpu_peak_mb", csv_content)

            report_content = report_path.read_text()
            self.assertIn("Layer Offloading", report_content)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class LayerOffloadGPUTests(unittest.TestCase):
    """Tests for layer offloading load behavior."""

    def test_load_small_model_to_cpu(self) -> None:
        """Smoke test CPU loading without downloading a real Hugging Face model."""
        fake_model = MagicMock()
        fake_model.config.num_hidden_layers = 2
        fake_auto_model = SimpleNamespace(
            from_pretrained=MagicMock(return_value=fake_model)
        )

        mgr = LayerOffloadManager(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            gpu_layer_window=2,
        )
        with patch.dict(
            "sys.modules",
            {"transformers": SimpleNamespace(AutoModelForCausalLM=fake_auto_model)},
        ):
            model = mgr.load_model_to_cpu()

        fake_auto_model.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        fake_model.eval.assert_called_once_with()
        self.assertIsNotNone(model)
        self.assertGreater(model.config.num_hidden_layers, 0)


if __name__ == "__main__":
    unittest.main()
