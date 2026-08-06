import unittest

from astrakv.kv_cache.block_table import KVBlockTable
from astrakv.kv_cache.metadata import KVChunkMeta, MemoryTier
from astrakv.offload.tier_placement import TierPlacementManager
from astrakv.runtime.object_manager import RuntimeObjectManager


class KVRuntimeTests(unittest.TestCase):
    def test_kv_metadata_validation_and_record(self) -> None:
        meta = KVChunkMeta(
            request_id="req-1",
            layer_id=2,
            start_token=4,
            end_token=12,
            block_ids=(7, 8),
            chunk_id="chunk-1",
            tier="cpu",
            size_bytes=1024,
        )

        self.assertEqual(meta.tier, MemoryTier.CPU)
        self.assertEqual(meta.token_count, 8)
        record = meta.to_record()
        self.assertEqual(record["chunk_id"], "chunk-1")
        self.assertEqual(record["tier"], "cpu")
        self.assertEqual(record["block_ids"], [7, 8])

        with self.assertRaises(ValueError):
            KVChunkMeta(request_id="bad", layer_id=0, start_token=-1, end_token=1)
        with self.assertRaises(ValueError):
            KVChunkMeta(request_id="bad", layer_id=0, start_token=2, end_token=1)
        with self.assertRaises(ValueError):
            KVChunkMeta(request_id="bad", layer_id=-1, start_token=0, end_token=1)

    def test_block_table_sorts_and_clears_request_chunks(self) -> None:
        table = KVBlockTable()
        later = KVChunkMeta(
            request_id="req-1",
            layer_id=1,
            start_token=16,
            end_token=32,
            chunk_id="later",
        )
        earlier = KVChunkMeta(
            request_id="req-1",
            layer_id=0,
            start_token=0,
            end_token=16,
            chunk_id="earlier",
        )

        table.add_chunk(later)
        table.add_chunk(earlier)

        listed = table.list_request_chunks("req-1")
        self.assertEqual([entry.chunk_id for entry in listed], ["earlier", "later"])

        removed = table.clear_request("req-1")
        self.assertEqual({entry.chunk_id for entry in removed}, {"earlier", "later"})
        self.assertEqual(table.list_request_chunks("req-1"), [])

    def test_tier_placement_transitions(self) -> None:
        manager = TierPlacementManager(default_tier=MemoryTier.GPU)
        meta = KVChunkMeta(
            request_id="req-1",
            layer_id=0,
            start_token=0,
            end_token=16,
            chunk_id="chunk-1",
            tier=MemoryTier.UNKNOWN,
        )

        registered = manager.register_chunk(meta)
        self.assertEqual(registered.current_tier, MemoryTier.GPU)
        self.assertEqual(registered.status, "resident")

        planned = manager.plan_move("chunk-1", MemoryTier.SSD, "memory_pressure")
        self.assertEqual(planned.current_tier, MemoryTier.GPU)
        self.assertEqual(planned.target_tier, MemoryTier.SSD)
        self.assertEqual(planned.status, "planned")

        self.assertEqual(manager.mark_in_flight("chunk-1").status, "in_flight")
        resident = manager.mark_resident("chunk-1")
        self.assertEqual(resident.current_tier, MemoryTier.SSD)
        self.assertEqual(resident.status, "resident")

        failed = manager.mark_failed("chunk-1", "adapter_error")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.reason, "adapter_error")

    def test_runtime_object_manager_snapshot_and_release(self) -> None:
        manager = RuntimeObjectManager()
        meta = KVChunkMeta(
            request_id="req-1",
            layer_id=0,
            start_token=0,
            end_token=16,
            chunk_id="chunk-1",
            tier=MemoryTier.CPU,
        )

        manager.register_chunk(meta)
        result = manager.prefetch_chunk_sync("chunk-1", target_tier=MemoryTier.GPU, priority=5)
        self.assertEqual(result.chunk_id, "chunk-1")

        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot.chunks), 1)
        self.assertEqual(snapshot.block_table[0]["chunk_id"], "chunk-1")
        self.assertEqual(snapshot.placements[0]["current_tier"], "cpu")
        self.assertEqual(snapshot.prefetch[0]["status"], "completed")

        removed = manager.release_request("req-1")
        self.assertEqual([item.chunk_id for item in removed], ["chunk-1"])
        self.assertIsNone(manager.get_chunk("chunk-1"))


if __name__ == "__main__":
    unittest.main()
