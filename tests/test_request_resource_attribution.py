import csv
import tempfile
import threading
import unittest
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

from astrakv.runtime.trace_schema import load_memory_samples, trace_from_cache_record, trace_from_prefetch_record
from scripts.benchmark.dgx_metrics_collector import DgxMetricsCollector, DgxSample
from scripts.benchmark.run_real_benchmark import run_batch, run_one_request


class RequestResourceAttributionTests(unittest.TestCase):
    def make_collector(self, output_csv: Path) -> DgxMetricsCollector:
        return DgxMetricsCollector(
            output_csv=output_csv,
            run_id="run-7",
            case="case-a",
        )

    def test_csv_sample_includes_identity_and_exclusive_request_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output_csv = Path(raw_tmp) / "samples.csv"
            collector = self.make_collector(output_csv)
            with patch.object(collector, "_sample_resource_metrics", return_value={
                "cpu_rss_mb": 12.0,
                "gpu_used_mb": None,
                "gpu_util_pct": None,
                "disk_read_mb": None,
                "disk_write_mb": None,
            }):
                with collector.request_scope("req-1"):
                    collector.samples.append(collector._sample())
            collector._write_csv()

            with output_csv.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["run_id"], "run-7")
            self.assertEqual(row["case"], "case-a")
            self.assertEqual(row["request_id"], "req-1")
            self.assertEqual(row["active_request_ids"], "req-1")
            self.assertEqual(row["attribution_mode"], "exclusive_request")
            self.assertEqual(row["sample_path"], str(output_csv))
            self.assertNotEqual(row["request_started_s"], "")
            self.assertEqual(row["request_ended_s"], "")

    def test_request_scope_snapshot_is_sorted_and_cleanup_is_deterministic(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with collector.request_scope("req-b"):
            with collector.request_scope("req-a"):
                self.assertEqual(collector.active_request_ids(), ("req-a", "req-b"))
            self.assertEqual(collector.active_request_ids(), ("req-b",))
        self.assertEqual(collector.active_request_ids(), ())

    def test_reused_request_id_gets_a_new_window_and_reclaims_old_state(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with patch("scripts.benchmark.dgx_metrics_collector.time.time", side_effect=[10.0, 11.0, 20.0, 21.0]), patch.object(
            collector,
            "_sample_resource_metrics",
            return_value={"cpu_rss_mb": 1.0, "gpu_used_mb": None, "gpu_util_pct": None, "disk_read_mb": None, "disk_write_mb": None},
        ):
            with collector.request_scope("reused"):
                first = collector._sample()
            self.assertEqual(collector._request_windows, {})
            with collector.request_scope("reused"):
                second = collector._sample()
            self.assertEqual(collector._request_windows, {})

        self.assertEqual(first.request_started_s, 10.0)
        self.assertEqual(second.request_started_s, 20.0)

    def test_exit_cannot_change_the_window_of_an_inflight_sample_snapshot(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        test_case = self

        class InterleavingWindows(dict):
            def _wait_for_exit_race(self) -> None:
                lookup_started.set()
                test_case.assertTrue(release_lookup.wait(timeout=2.0))

            def __contains__(self, key):
                self._wait_for_exit_race()
                return super().__contains__(key)

            def get(self, key, default=None):
                self._wait_for_exit_race()
                return super().get(key, default)

        scope = collector.request_scope("req-1")
        scope.__enter__()
        collector._request_windows = InterleavingWindows(collector._request_windows)
        sampled = []
        sampling_thread = threading.Thread(target=lambda: sampled.append(collector._sample()))
        sampling_thread.start()
        self.assertTrue(lookup_started.wait(timeout=2.0))
        exiting_thread = threading.Thread(target=lambda: scope.__exit__(None, None, None))
        exiting_thread.start()
        release_lookup.set()
        sampling_thread.join(timeout=2.0)
        exiting_thread.join(timeout=2.0)

        self.assertFalse(sampling_thread.is_alive())
        self.assertFalse(exiting_thread.is_alive())
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0].request_id, "req-1")
        self.assertIsNone(sampled[0].request_ended_s)
        self.assertEqual(collector._request_windows, {})

    def test_sample_keeps_the_original_positional_constructor_contract(self) -> None:
        sample = DgxSample(1.0, 2.0, None, None, None, None)

        self.assertEqual(sample.timestamp_s, 1.0)
        self.assertEqual(sample.cpu_rss_mb, 2.0)
        self.assertEqual(sample.run_id, "")
        self.assertEqual(sample.attribution_mode, "case_boundary")

    def test_overlapping_shared_scopes_remove_their_own_boundary(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        first = collector.shared_batch_scope(("first-a", "first-b"))
        second = collector.shared_batch_scope(("second-a", "second-b"))
        first.__enter__()
        second.__enter__()
        try:
            first.__exit__(None, None, None)
            with collector.request_scope("second-a"):
                sample = collector._sample()
            self.assertEqual(sample.shared_request_ids, ("second-a", "second-b"))
            self.assertEqual(sample.attribution_mode, "shared_batch")
        finally:
            second.__exit__(None, None, None)
        self.assertEqual(collector._sample().attribution_mode, "case_boundary")

    def test_sampling_with_two_shared_scopes_emits_an_ambiguous_union(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with collector.shared_batch_scope(("first-a", "first-b")):
            with collector.shared_batch_scope(("second-a", "second-b")):
                with collector.request_scope("first-a"), collector.request_scope("second-a"):
                    sample = collector._sample()

        self.assertEqual(sample.attribution_mode, "shared_batch_ambiguous")
        self.assertEqual(sample.request_id, "")
        self.assertEqual(sample.shared_request_ids, ("first-a", "first-b", "second-a", "second-b"))
        self.assertEqual(len(sample.shared_boundary_ids), 2)

    def test_shared_boundary_with_independent_active_request_is_ambiguous(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with collector.shared_batch_scope(("batch-a", "batch-b")):
            with collector.request_scope("batch-a"), collector.request_scope("unrelated"):
                sample = collector._sample()

        self.assertEqual(sample.attribution_mode, "shared_batch_ambiguous")
        self.assertEqual(sample.request_id, "")
        self.assertEqual(sample.shared_request_ids, ("batch-a", "batch-b"))
        self.assertEqual(sample.active_request_ids, ("batch-a", "unrelated"))

    def test_shared_boundary_without_active_requests_is_idle_before_and_after_batch(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with collector.shared_batch_scope(("batch-a", "batch-b")):
            before_start = collector._sample()
            with collector.request_scope("batch-a"), collector.request_scope("batch-b"):
                active = collector._sample()
            after_finish = collector._sample()

        self.assertEqual(before_start.attribution_mode, "case_boundary")
        self.assertEqual(after_finish.attribution_mode, "case_boundary")
        self.assertEqual(before_start.request_id, "")
        self.assertEqual(after_finish.request_id, "")
        self.assertEqual(active.attribution_mode, "shared_batch")

    def test_shared_scope_exception_cleanup_does_not_leak_boundary(self) -> None:
        collector = self.make_collector(Path("unused.csv"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with collector.shared_batch_scope(("req-a", "req-b")):
                raise RuntimeError("boom")
        self.assertEqual(collector._sample().attribution_mode, "case_boundary")

    def test_single_request_result_has_time_window_and_exclusive_sample_association(self) -> None:
        collector = self.make_collector(Path("artifacts") / "samples.csv")
        stream = iter([
            {"choices": [{"delta": {"content": "ok"}}]},
            {"usage": {"completion_tokens": 1}, "choices": []},
        ])
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", return_value=stream):
            result = run_one_request(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", request_id="req-1", batch_size=1, context_length=4,
                output_tokens=2, timeout=1, temperature=0, top_p=1, system_prompt="system",
                prompt_seed="seed", prompt_token_scale=1, request_metadata={"run_id": "run-7"},
                metrics_collector=collector,
            )

        self.assertEqual(result.run_id, "run-7")
        self.assertEqual(result.request_id, "req-1")
        self.assertEqual(result.case, "case-a")
        self.assertEqual(result.sample_path, str(collector.output_csv))
        self.assertEqual(result.attribution_mode, "exclusive_request")
        self.assertLessEqual(result.request_started_s, result.request_ended_s)
        self.assertEqual(collector.active_request_ids(), ())

    def test_concurrent_batch_uses_shared_boundary_not_exclusive_request_samples(self) -> None:
        collector = self.make_collector(Path("artifacts") / "samples.csv")
        observed_samples = []

        def sample_request(**kwargs):
            with kwargs["metrics_collector"].request_scope(kwargs["request_id"]):
                observed_samples.append(kwargs["metrics_collector"]._sample())
            return kwargs["request_id"]

        with patch("scripts.benchmark.run_real_benchmark.run_one_request", side_effect=sample_request) as request:
            results = run_batch(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", repeat_index=0, batch_size=2, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, run_id="run-7", metrics_collector=collector,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(len(set(results)), 2)
        self.assertTrue(all(result.startswith("run-7:case-a:") for result in results))
        self.assertTrue(all(call.kwargs["attribution_mode"] == "shared_batch" for call in request.call_args_list))
        self.assertTrue(all(sample.attribution_mode == "shared_batch" for sample in observed_samples))
        self.assertTrue(all(len(sample.shared_request_ids) == 2 for sample in observed_samples))
        self.assertTrue(all(request_id.startswith("run-7:case-a:") for sample in observed_samples for request_id in sample.shared_request_ids))
        self.assertEqual(collector.active_request_ids(), ())

    def test_same_run_and_case_batch_invocations_namespace_generated_request_ids(self) -> None:
        def return_request_id(**kwargs):
            return kwargs["request_id"]

        with patch("scripts.benchmark.run_real_benchmark.run_one_request", side_effect=return_request_id):
            first = run_batch(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", repeat_index=0, batch_size=2, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, run_id="run-7", batch_nonce="11111111-1111-4111-8111-111111111111",
            )
            second = run_batch(
                base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                case="case-a", repeat_index=0, batch_size=2, context_length=4, output_tokens=2,
                timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                prompt_token_scale=1, run_id="run-7", batch_nonce="22222222-2222-4222-8222-222222222222",
            )

        self.assertTrue(set(first).isdisjoint(second))
        self.assertTrue(all(request_id.startswith("run-7:case-a:") for request_id in first + second))

    def test_batch_nonce_override_must_be_unique_and_uuid_like_before_dispatch(self) -> None:
        kwargs = {
            "base_url": "http://endpoint", "api_key": "empty", "model": "model", "backend": "backend",
            "case": "case-nonce", "repeat_index": 0, "batch_size": 1, "context_length": 4,
            "output_tokens": 2, "timeout": 1, "temperature": 0, "top_p": 1,
            "system_prompt": "system", "prompt_seed": "seed", "prompt_token_scale": 1, "run_id": "run-nonce",
        }
        with patch("scripts.benchmark.run_real_benchmark.run_one_request") as request:
            with self.assertRaisesRegex(ValueError, "UUID"):
                run_batch(**kwargs, batch_nonce="not-a-nonce")
            nonce = str(uuid4())
            run_batch(**kwargs, batch_nonce=nonce)
            with self.assertRaisesRegex(ValueError, "already used"):
                run_batch(**kwargs, batch_nonce=nonce)

        self.assertEqual(request.call_count, 1)

    def test_real_batch_request_lifecycle_marks_samples_shared(self) -> None:
        collector = self.make_collector(Path("artifacts") / "samples.csv")
        both_requests_started = threading.Event()
        release_requests = threading.Event()
        lock = threading.Lock()
        started_count = 0

        def blocked_stream(_url, payload, _api_key, _timeout):
            nonlocal started_count
            with lock:
                started_count += 1
                if started_count == 2:
                    both_requests_started.set()
            self.assertTrue(release_requests.wait(timeout=2.0))
            yield {"choices": [{"delta": {"content": payload["user"]}}]}
            yield {"usage": {"completion_tokens": 1}, "choices": []}

        batch_result = []
        with patch("scripts.benchmark.run_real_benchmark.stream_chat_completion", side_effect=blocked_stream):
            thread = threading.Thread(
                target=lambda: batch_result.extend(run_batch(
                    base_url="http://endpoint", api_key="empty", model="model", backend="backend",
                    case="case-a", repeat_index=0, batch_size=2, context_length=4, output_tokens=2,
                    timeout=1, temperature=0, top_p=1, system_prompt="system", prompt_seed="seed",
                    prompt_token_scale=1, run_id="run-7", metrics_collector=collector,
                )),
            )
            thread.start()
            self.assertTrue(both_requests_started.wait(timeout=2.0))
            sample = collector._sample()
            release_requests.set()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(sample.attribution_mode, "shared_batch")
        self.assertEqual(len(sample.shared_request_ids), 2)
        self.assertTrue(all(request_id.startswith("run-7:case-a:") for request_id in sample.shared_request_ids))
        self.assertEqual(sample.active_request_ids, sample.shared_request_ids)
        self.assertTrue(all(result.attribution_mode == "shared_batch" for result in batch_result))
        self.assertEqual(collector.active_request_ids(), ())

    def test_collector_csv_identity_maps_to_trace_without_assigning_shared_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output_csv = Path(raw_tmp) / "samples.csv"
            collector = self.make_collector(output_csv)
            with patch.object(collector, "_sample_resource_metrics", return_value={
                "cpu_rss_mb": 12.0,
                "gpu_used_mb": None,
                "gpu_util_pct": None,
                "disk_read_mb": None,
                "disk_write_mb": None,
            }):
                with collector.request_scope("exclusive"):
                    collector.samples.append(collector._sample())
                with collector.shared_batch_scope(("shared-a", "shared-b")):
                    with collector.request_scope("shared-a"):
                        collector.samples.append(collector._sample())
            collector._write_csv()

            exclusive, shared = load_memory_samples(output_csv)
        self.assertEqual(exclusive.request_id, "exclusive")
        self.assertEqual(exclusive.case, "case-a")
        self.assertEqual(exclusive.run_id, "run-7")
        self.assertEqual(exclusive.attribution_mode, "exclusive_request")
        self.assertEqual(exclusive.metadata["run_id"], "run-7")
        self.assertEqual(exclusive.metadata["attribution_mode"], "exclusive_request")
        self.assertEqual(shared.request_id, "")
        self.assertEqual(shared.case, "case-a")
        self.assertEqual(shared.run_id, "run-7")
        self.assertEqual(shared.attribution_mode, "shared_batch")
        self.assertEqual(shared.metadata["attribution_mode"], "shared_batch")
        self.assertEqual(shared.metadata["shared_request_ids"], "shared-a,shared-b")

    def test_cache_and_prefetch_trace_events_propagate_run_id_from_record_or_metadata(self) -> None:
        cache = trace_from_cache_record({
            "event_type": "cache_hit", "source": "cache.log", "request_id": "req-1",
            "run_id": "run-record", "metadata": {"run_id": "run-metadata"},
        })
        prefetch = trace_from_prefetch_record({
            "event_type": "prefetch_completed", "request_id": "req-2",
            "metadata": {"run_id": "run-metadata"},
        })

        self.assertEqual(cache.run_id, "run-record")
        self.assertEqual(prefetch.run_id, "run-metadata")


if __name__ == "__main__":
    unittest.main()
