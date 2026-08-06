"""Tests for the DGX Spark mmap-backed KV chunk adapter."""

import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.vm.dgx_spark_adapter import DgxSparkKVAdapter, DgxSparkKVAdapterConfig


if not sys.platform.startswith("linux"):
    raise unittest.SkipTest("DGX mmap PoC requires Linux")


class DgxSparkKVAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_register_write_read_and_events(self) -> None:
        config = DgxSparkKVAdapterConfig(
            backing_file=str(self.tmp_path / "kv.bin"),
            total_blocks=4,
            block_size_bytes=4096,
            keep_backing_file=True,
        )
        chunk = KVChunkMeta(
            request_id="req",
            layer_id=0,
            start_token=0,
            end_token=128,
            chunk_id="chunk-a",
            tier=MemoryTier.SSD,
            size_bytes=4096,
        )
        with DgxSparkKVAdapter(config) as adapter:
            record = adapter.register_chunk(chunk)
            self.assertEqual(record.block_ids, (0,))
            self.assertEqual(record.chunk.device, "dgx-spark-mmap")

            data = np.arange(2048, dtype=np.float16)
            write_event = adapter.write_chunk("chunk-a", data)
            result, read_event = adapter.read_chunk("chunk-a")

            np.testing.assert_array_equal(result, data)
            self.assertEqual(write_event.action, "write_chunk")
            self.assertEqual(read_event.action, "read_chunk")
            self.assertGreaterEqual(adapter.resident_ratio("chunk-a"), 0.0)
            self.assertEqual(len(adapter.chunk_records()), 1)

    def test_prefetch_and_evict_chunk(self) -> None:
        config = DgxSparkKVAdapterConfig(
            backing_file=str(self.tmp_path / "kv.bin"),
            total_blocks=4,
            block_size_bytes=4096,
        )
        with DgxSparkKVAdapter(config) as adapter:
            adapter.register_chunk(
                KVChunkMeta(
                    request_id="req",
                    layer_id=0,
                    start_token=0,
                    end_token=128,
                    chunk_id="chunk-a",
                    tier=MemoryTier.SSD,
                    size_bytes=4096,
                )
            )
            prefetch = adapter.prefetch_chunk("chunk-a")
            evict = adapter.evict_chunk("chunk-a")

            self.assertEqual(prefetch.action, "prefetch_chunk")
            self.assertEqual(evict.action, "evict_chunk")
            self.assertEqual(len(adapter.events), 2)

    def test_out_of_blocks_raises(self) -> None:
        config = DgxSparkKVAdapterConfig(
            backing_file=str(self.tmp_path / "kv.bin"),
            total_blocks=1,
            block_size_bytes=4096,
        )
        with DgxSparkKVAdapter(config) as adapter:
            with self.assertRaises(IndexError):
                adapter.register_chunk(
                    KVChunkMeta(
                        request_id="req",
                        layer_id=0,
                        start_token=0,
                        end_token=128,
                        chunk_id="chunk-big",
                        tier=MemoryTier.SSD,
                        size_bytes=8192,
                    )
                )


if __name__ == "__main__":
    unittest.main()
