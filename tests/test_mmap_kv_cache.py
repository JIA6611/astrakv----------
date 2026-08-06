"""Tests for the mmap-backed KV-cache virtual-memory manager."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from astrakv.vm.mmap_kv_cache import (
    MMapKVCache,
    MMapKVCacheConfig,
    MMapKVCacheStats,
    _madvise,
    _mincore,
    MADV_DONTNEED,
    MADV_WILLNEED,
)


if not sys.platform.startswith("linux"):
    raise unittest.SkipTest("madvise/mincore mmap PoC requires Linux")


class MMapKVCacheBasicTests(unittest.TestCase):
    """Basic sanity checks for MMapKVCache."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_init_and_close(self) -> None:
        cache = MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=8,
            block_size_bytes=4096,
        )
        self.assertEqual(cache.config.total_blocks, 8)
        self.assertEqual(cache.config.block_size_bytes, 4096)
        cache.close()

    def test_context_manager(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=4096,
        ) as cache:
            self.assertEqual(cache.config.total_blocks, 4)

    def test_write_and_read_block(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=1024,
        ) as cache:
            data = np.arange(512, dtype=np.float16)
            cache.write_block(0, data)
            result = cache.read_block(0)
            np.testing.assert_array_equal(result, data)

    def test_block_id_out_of_range(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=1024,
        ) as cache:
            with self.assertRaises(IndexError):
                cache.read_block(4)
            with self.assertRaises(IndexError):
                cache.read_block(-1)

    def test_write_wrong_size_raises(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=1024,
        ) as cache:
            bad_data = np.zeros(256, dtype=np.float16)  # 512 bytes, but block is 1024
            with self.assertRaises(ValueError):
                cache.write_block(0, bad_data)

    def test_backing_file_created(self) -> None:
        backing = self.tmp_path / "created.bin"
        self.assertFalse(backing.exists())
        with MMapKVCache(
            backing_file=str(backing),
            total_blocks=4,
            block_size_bytes=4096,
        ):
            pass
        self.assertTrue(backing.exists())
        self.assertEqual(os.path.getsize(backing), 4 * 4096)

    def test_prefetch_and_evict_return_bool(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=4096,
        ) as cache:
            self.assertIsInstance(cache.prefetch_block(0), bool)
            self.assertIsInstance(cache.evict_block(0), bool)

    def test_batch_prefetch_and_evict(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=16,
            block_size_bytes=4096,
        ) as cache:
            n_prefetch = cache.prefetch_batch([0, 1, 2])
            self.assertLessEqual(n_prefetch, 3)
            n_evict = cache.evict_batch(list(range(16)))
            self.assertLessEqual(n_evict, 16)

    def test_collect_stats_returns_valid_data(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=8,
            block_size_bytes=4096,
        ) as cache:
            data = np.zeros(2048, dtype=np.float16)
            cache.write_block(0, data)
            stats = cache.collect_stats()
            self.assertIsInstance(stats, MMapKVCacheStats)
            self.assertEqual(stats.total_blocks, 8)
            record = stats.to_record()
            self.assertIn("total_blocks", record)
            self.assertIn("resident_blocks", record)

    def test_get_resident_blocks_returns_dict(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=8,
            block_size_bytes=4096,
        ) as cache:
            resident = cache.get_resident_blocks()
            self.assertIsInstance(resident, dict)
            if resident:
                for v in resident.values():
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 1.0)

    def test_deterministic_read_after_write(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=1024,
        ) as cache:
            data1 = np.arange(512, dtype=np.float16)
            data2 = np.arange(512, 512 + 512, dtype=np.float16)
            cache.write_block(0, data1)
            cache.write_block(1, data2)
            np.testing.assert_array_equal(cache.read_block(0), data1)
            np.testing.assert_array_equal(cache.read_block(1), data2)


class MMapKVCacheStatsTests(unittest.TestCase):
    """Tests for MMapKVCacheStats dataclass."""

    def test_empty_stats(self) -> None:
        stats = MMapKVCacheStats()
        self.assertEqual(stats.total_blocks, 0)
        self.assertEqual(stats.avg_cold_read_us, 0.0)
        self.assertEqual(stats.avg_warm_read_us, 0.0)

    def test_stats_to_record(self) -> None:
        stats = MMapKVCacheStats(
            total_blocks=10,
            resident_blocks=3,
            prefetch_requests=5,
            cold_read_count=2,
            cold_read_total_us=500.0,
        )
        record = stats.to_record()
        self.assertEqual(record["total_blocks"], 10)
        self.assertEqual(record["resident_blocks"], 3)
        self.assertEqual(record["avg_cold_read_us"], 250.0)


class MMapKVCacheConfigTests(unittest.TestCase):
    """Tests for MMapKVCacheConfig."""

    def test_config_properties(self) -> None:
        cfg = MMapKVCacheConfig(
            total_blocks=100,
            block_size_bytes=1024 * 1024,
            backing_file="/tmp/test.bin",
        )
        self.assertEqual(cfg.total_size_bytes, 100 * 1024 * 1024)
        self.assertEqual(cfg.dtype, np.dtype("float16"))

    def test_custom_dtype(self) -> None:
        cfg = MMapKVCacheConfig(
            total_blocks=10,
            block_size_bytes=1024,
            backing_file="/tmp/test.bin",
            dtype_str="float32",
        )
        self.assertEqual(cfg.dtype, np.dtype("float32"))


class MMapKVCacheEdgeCases(unittest.TestCase):
    """Edge case and error path tests."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_closed_cache_raises_on_use(self) -> None:
        cache = MMapKVCache(
            backing_file=str(self.tmp_path / "test.bin"),
            total_blocks=4,
            block_size_bytes=4096,
        )
        cache.close()
        with self.assertRaises(RuntimeError):
            cache.read_block(0)

    def test_recreate_mismatched_backing_file(self) -> None:
        backing = self.tmp_path / "mismatch.bin"
        # Create a backing file with wrong size
        with open(backing, "wb") as f:
            f.truncate(100)
        # MMapKVCache should warn and recreate
        cache = MMapKVCache(
            backing_file=str(backing),
            total_blocks=4,
            block_size_bytes=4096,
        )
        self.assertEqual(os.path.getsize(backing), 4 * 4096)
        cache.close()

    def test_single_block_cache(self) -> None:
        with MMapKVCache(
            backing_file=str(self.tmp_path / "single.bin"),
            total_blocks=1,
            block_size_bytes=4096,
        ) as cache:
            data = np.zeros(2048, dtype=np.float16)
            cache.write_block(0, data)
            result = cache.read_block(0)
            np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    unittest.main()
