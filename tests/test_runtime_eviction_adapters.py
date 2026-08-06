import sys
import unittest
from astrakv.runtime.cache_events import CacheEvent
from astrakv.runtime.eviction import (
    MMapEvictionAdapter,
    ObjectLevel,
    OfflineEvictionDecision,
    VllmLmCacheArtifactAdapter,
)


def eviction_decision(object_key: str) -> OfflineEvictionDecision:
    return OfflineEvictionDecision(
        run_id="run-poc",
        decision_id="offline-1",
        request_id="req-1",
        object_key=object_key,
        object_level=ObjectLevel.PREFIX,
        predicted_action="offload",
        decision_index=1,
    )


class RuntimeEvictionAdapterTests(unittest.TestCase):
    def test_artifact_adapter_normalizes_log_evidence_but_refuses_execution(self) -> None:
        adapter = VllmLmCacheArtifactAdapter(
            run_id="run-a",
            request_objects={"req-1": {"prefix_id": "prefix-a", "arrival_index": 3}},
        )
        raw = CacheEvent(
            event_type="cache_offload",
            source="server.log",
            status="completed",
            request_id="req-1",
            tier="disk",
            line_number=12,
        )

        events = adapter.collect_runtime_events([raw])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].object_level, ObjectLevel.PREFIX)
        self.assertEqual(events[0].object_key, "prefix-a")
        self.assertEqual(events[0].provenance, "log_heuristic")
        self.assertFalse(events[0].is_ground_truth)
        self.assertEqual(adapter.apply_hint(eviction_decision("prefix-a")).status, "unsupported")

        unlinked = adapter.collect_runtime_events([CacheEvent(event_type="cache_offload", source="server.log")])
        self.assertEqual(unlinked, [])
        self.assertEqual(adapter.last_skipped_evidence[0]["reason"], "missing_stable_request_or_object_association")

    def test_mmap_adapter_emits_execution_receipt_and_failure(self) -> None:
        class FakeMMapAdapter:
            config = type("Config", (), {"target_tier": type("Tier", (), {"value": "ssd"})()})()

            def evict_chunk(self, chunk_id: str):
                if chunk_id == "missing":
                    raise KeyError(chunk_id)
                return type(
                    "Action",
                    (),
                    {"ok": True, "resident_ratio": 0.25, "latency_us": 42.0, "block_ids": (3,)},
                )()

            def prefetch_chunk(self, chunk_id: str):
                return self.evict_chunk(chunk_id)

            def read_chunk(self, chunk_id: str):
                return object(), self.evict_chunk(chunk_id)

        adapter = MMapEvictionAdapter(
            FakeMMapAdapter(),  # type: ignore[arg-type]
            run_id="run-poc",
            object_bindings={(ObjectLevel.PREFIX, "prefix-a"): "chunk-a"},
        )
        result = adapter.apply_hint(eviction_decision("prefix-a"))
        self.assertEqual(result.status, "executed")
        assert result.event is not None
        self.assertEqual(result.event.provenance, "vm_poc_execution")
        self.assertEqual(result.event.metadata["mmap_chunk_id"], "chunk-a")
        self.assertEqual(result.event.metadata["resident_ratio"], 0.25)

        prefetch = adapter.prefetch_object(eviction_decision("prefix-a"))
        read = adapter.read_object(eviction_decision("prefix-a"))
        self.assertEqual(prefetch.status, "executed")
        self.assertEqual(prefetch.event.actual_action, "prefetch")  # type: ignore[union-attr]
        self.assertEqual(read.status, "executed")
        self.assertEqual(read.event.actual_action, "read")  # type: ignore[union-attr]

        failed = MMapEvictionAdapter(FakeMMapAdapter(), run_id="run-poc").apply_hint(eviction_decision("missing"))  # type: ignore[arg-type]
        self.assertEqual(failed.status, "failed")
        self.assertIsNotNone(failed.event)

    @unittest.skipUnless(sys.platform.startswith("linux"), "madvise/mincore VM PoC requires Linux")
    def test_mmap_adapter_executes_real_linux_poc(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        from astrakv.kv_cache.metadata import KVChunkMeta
        from astrakv.vm.dgx_spark_adapter import DgxSparkKVAdapter, DgxSparkKVAdapterConfig

        with tempfile.TemporaryDirectory() as raw_tmp:
            config = DgxSparkKVAdapterConfig(
                backing_file=str(Path(raw_tmp) / "kv.bin"), total_blocks=4, block_size_bytes=4096
            )
            with DgxSparkKVAdapter(config) as kv_adapter:
                record = kv_adapter.register_chunk(KVChunkMeta(
                    request_id="req-1", layer_id=0, start_token=0, end_token=16,
                    chunk_id="mmap-chunk-a", size_bytes=4096,
                ))
                kv_adapter.write_chunk(record.chunk.chunk_id, np.zeros(2048, dtype=np.float16))
                result = MMapEvictionAdapter(
                    kv_adapter, run_id="run-poc",
                    object_bindings={(ObjectLevel.PREFIX, "prefix-a"): record.chunk.chunk_id},
                ).apply_hint(eviction_decision("prefix-a"))
                self.assertEqual(result.status, "executed")
                assert result.event is not None
                self.assertIn("resident_ratio", result.event.metadata)


if __name__ == "__main__":
    unittest.main()
