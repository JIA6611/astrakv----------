import asyncio
import unittest

from astrakv.kv_cache.metadata import MemoryTier
from astrakv.prefetch.async_engine import AsyncPrefetchEngine, PrefetchRequest, PrefetchResult, PrefetchStatus
from astrakv.prefetch.selective_kv import (
    KVAccessSource,
    KVBlockRef,
    SelectiveKVPrefetchConfig,
    SelectiveKVPrefetchMVP,
)
from astrakv.runtime.endpoint_prefetch import EndpointRequest, make_prefetch_request, normalize_base_url


class PrefetchTests(unittest.TestCase):
    def test_async_prefetch_engine_records_success_and_failure(self) -> None:
        async def ok_adapter(request: PrefetchRequest) -> PrefetchResult:
            return PrefetchResult(
                request_id=request.request_id,
                chunk_id=request.chunk_id,
                status=PrefetchStatus.COMPLETED,
                message="ok",
            )

        async def fail_adapter(request: PrefetchRequest) -> PrefetchResult:
            raise RuntimeError("boom")

        ok_engine = AsyncPrefetchEngine(adapter=ok_adapter)
        ok_request = PrefetchRequest(chunk_id="chunk-ok", request_id="pref-ok")
        ok_result = asyncio.run(ok_engine.submit(ok_request))
        self.assertEqual(ok_result.status, PrefetchStatus.COMPLETED)
        self.assertEqual(ok_engine.status("pref-ok"), PrefetchStatus.COMPLETED)
        self.assertEqual(ok_engine.to_records()[0]["message"], "ok")

        fail_engine = AsyncPrefetchEngine(adapter=fail_adapter)
        fail_request = PrefetchRequest(chunk_id="chunk-fail", request_id="pref-fail")
        fail_result = asyncio.run(fail_engine.submit(fail_request))
        self.assertEqual(fail_result.status, PrefetchStatus.FAILED)
        self.assertIn("boom", fail_result.message)
        self.assertEqual(fail_engine.status("pref-fail"), PrefetchStatus.FAILED)

    def test_selective_kv_prefetch_hit_and_metrics(self) -> None:
        async def run() -> dict[str, float | int]:
            engine = SelectiveKVPrefetchMVP(
                SelectiveKVPrefetchConfig(
                    gpu_capacity_blocks=2,
                    block_size_bytes=1024,
                    prefetch_window=1,
                    prefetch_latency_ms=0,
                    cpu_miss_latency_ms=0,
                    gpu_hit_latency_ms=0,
                    prefetch_hit_latency_ms=0,
                )
            )
            engine.add_cpu_blocks(
                [
                    KVBlockRef(block_id="a", size_bytes=1024),
                    KVBlockRef(block_id="b", size_bytes=1024),
                ]
            )
            await engine.start()
            trace = ["a", "b"]
            submitted = await engine.submit_predictions(trace, 0)
            await asyncio.sleep(0.01)
            access = await engine.access("b")
            await engine.close()
            self.assertEqual(submitted, 1)
            self.assertEqual(access.source, KVAccessSource.PREFETCH_HIT)
            return engine.to_metrics_record()

        metrics = asyncio.run(run())
        self.assertEqual(metrics["prefetch_submitted"], 1)
        self.assertEqual(metrics["prefetch_completed"], 1)
        self.assertEqual(metrics["prefetch_hits"], 1)
        self.assertEqual(metrics["demand_lookups"], 1)
        self.assertEqual(metrics["prefetch_hit_rate"], 1.0)

    def test_endpoint_prefetch_request_metadata_is_adapter_compatible(self) -> None:
        endpoint_request = EndpointRequest(
            request_id="endpoint-prefetch",
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
        request = make_prefetch_request(
            chunk_id="chunk-1",
            endpoint_request=endpoint_request,
            target_tier=MemoryTier.GPU,
            priority=3,
            metadata={"case": "ctx128"},
        )

        self.assertEqual(request.chunk_id, "chunk-1")
        self.assertEqual(request.target_tier, MemoryTier.GPU)
        self.assertEqual(request.priority, 3)
        self.assertIs(request.metadata["endpoint_request"], endpoint_request)
        self.assertEqual(request.metadata["case"], "ctx128")
        self.assertEqual(normalize_base_url("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
