import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_bridge import OnlineBackendBridge
from astrakv.runtime.backend_hook import BackendExecutionSpec, BackendObjectBinding, BackendHookEvent, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.offline_safety import OfflineSafetyGate
from astrakv.runtime.online_controller import OnlinePolicyController
from astrakv.runtime.prediction_sidecar import (
    PredictorCandidateRecord,
    PredictionSidecarIndex,
    SidecarPrediction,
)
from scripts.benchmark.materialize_grouped_exact_next_workload import main as materialize_grouped_main
from scripts.reporting.build_predictor_candidate_report import main as build_candidate_main
from scripts.reporting.build_sidecar_prediction import main as build_sidecar_main


def gate():
    base = {
        "schema": "astrakv-offline-eviction-v1",
        "simulation_status": "valid",
        "trace_sha256": "t",
        "profile_db_sha256": "p",
        "workload_sha256": "w",
        "profile_source": "separate_profiling_run",
        "capacities": {"gpu_bytes": 1, "cpu_bytes": 1, "ssd_bytes": 1},
        "policies": [
            {"policy": "astrakv", "request_count": 1, "total_hits": 1, "migration_bytes": 1, "oom_unavoided": 0},
            {"policy": "lru", "request_count": 1, "total_hits": 0, "migration_bytes": 2},
            {"policy": "fifo", "request_count": 1, "total_hits": 0, "migration_bytes": 3},
        ],
    }
    return OfflineSafetyGate.evaluate([{**base, "workload_id": item} for item in ("a", "b", "c")])


class PredictionSidecarTests(unittest.TestCase):
    def test_candidate_report_rejects_benchmark_aware_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark-aware"):
            PredictorCandidateRecord.from_record(
                {
                    "schema": "astrakv-predictor-candidate-report-v1",
                    "request_id": "req-1",
                    "candidate_object_id": "prefix-a",
                    "object_level": "prefix",
                    "predicted_class": "exact-next",
                    "lead_distance_requests": 1,
                    "estimated_reusable_tokens": 128,
                    "estimated_kv_bytes": 256,
                    "confidence": 0.95,
                    "reason": "exact_next_locality",
                    "reuse_group": "do-not-expose",
                }
            )

    def test_sidecar_index_filters_expired_predictions(self) -> None:
        index = PredictionSidecarIndex(
            [
                SidecarPrediction(
                    run_id="run-a",
                    request_id="req-1",
                    candidate_object_id="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    score=0.9,
                    recommended_lead_time_ms=250.0,
                    confidence=0.95,
                    reason="exact_next_locality",
                    evidence_source="offline",
                    predicted_class="exact-next",
                    expires_at_ns=1,
                )
            ],
            run_id="run-a",
        )
        self.assertIsNone(
            index.advisory_for(
                request_id="req-1",
                candidate_object_id="prefix-a",
                object_level=ObjectLevel.PREFIX,
                now_ns=2,
            )
        )


class OnlineControllerPredictionTests(unittest.TestCase):
    def test_exact_next_sidecar_prefers_prefetch_for_cold_ssd_object(self) -> None:
        registry = BackendBindingRegistry(
            run_id="run",
            engine_instance_id="engine",
            worker_id="worker",
        )
        context = RequestContext("run", "req-1", "prefix-a", ObjectLevel.PREFIX)
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        completed = registry.complete_operation(
            "key-a",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        )
        released = registry.observe("key-a", HookAction.RELEASE, "completed", context)
        assert completed.binding is not None
        execution_spec = BackendExecutionSpec(
            spec_id="spec-1",
            binding_id=completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            backend_object_id=completed.binding.backend_object_id,
            object_key="prefix-a",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "drop": {"status": "ready"},
                "evict": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-1",
            "prefix-a",
            ObjectLevel.PREFIX,
            completed.binding.backend_object_id,
            completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
            binding_registry=registry,
        )
        prediction = PredictionSidecarIndex(
            [
                SidecarPrediction(
                    run_id="run",
                    request_id="req-1",
                    candidate_object_id="prefix-a",
                    object_level=ObjectLevel.PREFIX,
                    score=0.99,
                    recommended_lead_time_ms=250.0,
                    confidence=0.98,
                    reason="exact_next_locality",
                    evidence_source="offline",
                    predicted_class="exact-next",
                    expires_at_ns=9_999_999_999_999_999_999,
                )
            ],
            run_id="run",
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            prediction_source=prediction,
        )
        self.assertTrue(
            controller.ingest(
                BackendHookEvent(
                    "run",
                    "event-store",
                    "req-1",
                    "prefix-a",
                    ObjectLevel.PREFIX,
                    completed.binding.backend_object_id,
                    HookAction.CACHE_STORE,
                    "completed",
                    1,
                    tier_after="ssd",
                    binding_generation=completed.binding.binding_generation,
                    metadata={"binding_id": completed.binding.binding_id},
                )
            )
        )
        self.assertTrue(
            controller.ingest(
                BackendHookEvent(
                    "run",
                    "event-release",
                    "req-1",
                    "prefix-a",
                    ObjectLevel.PREFIX,
                    released.binding.backend_object_id if released.binding is not None else completed.binding.backend_object_id,
                    HookAction.RELEASE,
                    "completed",
                    2,
                    tier_before="ssd",
                    tier_after="ssd",
                    binding_generation=completed.binding.binding_generation,
                    metadata={"binding_id": completed.binding.binding_id},
                )
            )
        )

        proposed = controller.propose_for("prefix-a", ObjectLevel.PREFIX)
        self.assertEqual(proposed.predicted_action, "prefetch")
        self.assertTrue(proposed.metadata["prediction_present"])
        self.assertEqual(proposed.metadata["prediction_reason"], "exact_next_locality")

    def test_sidecar_does_not_override_same_request_load_target_guard(self) -> None:
        registry = BackendBindingRegistry(
            run_id="run",
            engine_instance_id="engine",
            worker_id="worker",
        )
        context = RequestContext(
            "run",
            "req-2",
            "prefix-b",
            ObjectLevel.PREFIX,
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
        )
        submitted = registry.observe("key-b", HookAction.CACHE_STORE, "submitted", context)
        completed = registry.complete_operation(
            "key-b",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
        )
        released = registry.observe(
            "key-b",
            HookAction.RELEASE,
            "completed",
            context,
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
        )
        assert completed.binding is not None
        execution_spec = BackendExecutionSpec(
            spec_id="spec-2",
            binding_id=completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            backend_object_id=completed.binding.backend_object_id,
            object_key="prefix-b",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "load": {
                    "status": "ready",
                    "load_target_id": "target-current",
                    "runtime_reqmeta_id": "reqmeta-current",
                    "metadata": {"runtime_reqmeta_id": "reqmeta-current"},
                },
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-2",
            "prefix-b",
            ObjectLevel.PREFIX,
            completed.binding.backend_object_id,
            completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            metadata={"runtime_reqmeta_id": "reqmeta-current"},
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
            binding_registry=registry,
        )
        prediction = PredictionSidecarIndex(
            [
                SidecarPrediction(
                    run_id="run",
                    request_id="req-2",
                    candidate_object_id="prefix-b",
                    object_level=ObjectLevel.PREFIX,
                    score=0.95,
                    recommended_lead_time_ms=250.0,
                    confidence=0.95,
                    reason="exact_next_locality",
                    evidence_source="offline",
                    predicted_class="exact-next",
                    expires_at_ns=9_999_999_999_999_999_999,
                )
            ],
            run_id="run",
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            prediction_source=prediction,
        )
        self.assertTrue(
            controller.ingest(
                BackendHookEvent(
                    "run",
                    "event-store",
                    "req-2",
                    "prefix-b",
                    ObjectLevel.PREFIX,
                    completed.binding.backend_object_id,
                    HookAction.CACHE_STORE,
                    "completed",
                    1,
                    tier_after="ssd",
                    metadata={"runtime_reqmeta_id": "reqmeta-current"},
                    binding_generation=completed.binding.binding_generation,
                )
            )
        )
        self.assertTrue(
            controller.ingest(
                BackendHookEvent(
                    "run",
                    "event-release",
                    "req-2",
                    "prefix-b",
                    ObjectLevel.PREFIX,
                    released.binding.backend_object_id if released.binding is not None else completed.binding.backend_object_id,
                    HookAction.RELEASE,
                    "completed",
                    2,
                    tier_before="ssd",
                    tier_after="ssd",
                    metadata={"runtime_reqmeta_id": "reqmeta-current"},
                    binding_generation=completed.binding.binding_generation,
                )
            )
        )

        proposed = controller.propose_for("prefix-b", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "prefetch")
        self.assertTrue(proposed.metadata["same_request_load_target"])

    def test_sidecar_respects_prefetch_waste_tolerance(self) -> None:
        registry = BackendBindingRegistry(
            run_id="run",
            engine_instance_id="engine",
            worker_id="worker",
        )
        context = RequestContext("run", "req-3", "prefix-c", ObjectLevel.PREFIX)
        submitted = registry.observe("key-c", HookAction.CACHE_STORE, "submitted", context)
        completed = registry.complete_operation(
            "key-c",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        )
        registry.observe("key-c", HookAction.RELEASE, "completed", context)
        assert completed.binding is not None
        execution_spec = BackendExecutionSpec(
            spec_id="spec-3",
            binding_id=completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            backend_object_id=completed.binding.backend_object_id,
            object_key="prefix-c",
            object_level=ObjectLevel.PREFIX,
            runtime_owner="owner",
            owner_channel="channel",
            key_identity="key",
            lifecycle="released",
            actions={
                "prefetch": {"status": "ready"},
                "evict": {"status": "ready"},
                "drop": {"status": "ready"},
            },
        )
        binding = BackendObjectBinding(
            "run",
            "req-3",
            "prefix-c",
            ObjectLevel.PREFIX,
            completed.binding.backend_object_id,
            completed.binding.binding_id,
            binding_generation=completed.binding.binding_generation,
            execution_spec=execution_spec,
        )
        bridge = OnlineBackendBridge(
            run_id="run",
            bindings=[binding],
            hook_client=object(),
            hook_url="http://127.0.0.1:7900/actions",
            gate=gate(),
            binding_registry=registry,
        )
        prediction = PredictionSidecarIndex(
            [
                SidecarPrediction(
                    run_id="run",
                    request_id="req-3",
                    candidate_object_id="prefix-c",
                    object_level=ObjectLevel.PREFIX,
                    score=0.99,
                    recommended_lead_time_ms=250.0,
                    confidence=0.99,
                    reason="exact_next_locality",
                    evidence_source="offline",
                    predicted_class="exact-next",
                    expires_at_ns=9_999_999_999_999_999_999,
                )
            ],
            run_id="run",
        )
        controller = OnlinePolicyController(
            run_id="run",
            workload_id="w",
            bridge=bridge,
            prediction_source=prediction,
        )
        for event in (
            BackendHookEvent("run", "event-store", "req-3", "prefix-c", ObjectLevel.PREFIX, completed.binding.backend_object_id, HookAction.CACHE_STORE, "submitted", 1, tier_after="ssd", binding_generation=completed.binding.binding_generation, metadata={"binding_id": completed.binding.binding_id}),
            BackendHookEvent("run", "event-prefetch", "req-3", "prefix-c", ObjectLevel.PREFIX, completed.binding.backend_object_id, HookAction.PREFETCH, "completed", 2, tier_before="ssd", tier_after="cpu", binding_generation=completed.binding.binding_generation, metadata={"binding_id": completed.binding.binding_id}),
            BackendHookEvent("run", "event-offload", "req-3", "prefix-c", ObjectLevel.PREFIX, completed.binding.backend_object_id, HookAction.OFFLOAD, "completed", 3, tier_before="cpu", tier_after="ssd", binding_generation=completed.binding.binding_generation, metadata={"binding_id": completed.binding.binding_id}),
        ):
            self.assertTrue(controller.ingest(event))

        proposed = controller.propose_for("prefix-c", ObjectLevel.PREFIX)
        self.assertNotEqual(proposed.predicted_action, "prefetch")
        self.assertEqual(proposed.metadata["prefetch_waste_count"], 1)


class PredictorScriptTests(unittest.TestCase):
    def test_materializer_uses_reuse_group_as_prefix_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            grouped = root / "grouped_prompts.jsonl"
            grouped.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "request_id": "req-1",
                                "prompt": "alpha prompt",
                                "order": 0,
                                "reuse_group": "sha256:group-a",
                                "shared_context": True,
                                "context_hash": "ctx-a",
                            }
                        ),
                        json.dumps(
                            {
                                "request_id": "req-2",
                                "prompt": "beta prompt",
                                "order": 1,
                                "reuse_group": "sha256:group-a",
                                "shared_context": True,
                                "context_hash": "ctx-a",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "out"
            with patch("sys.argv", ["materialize", "--grouped-prompts-jsonl", str(grouped), "--output-dir", str(output), "--dataset", "qasper"]):
                self.assertEqual(materialize_grouped_main(), 0)
            workload_path = output / "qasper_grouped_exact_next_canonical_workload.jsonl"
            rows = [json.loads(line) for line in workload_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["prefix_id"], "sha256:group-a")
            self.assertEqual(rows[1]["prefix_id"], "sha256:group-a")
            self.assertEqual(rows[0]["cache_key"], "sha256:group-a")
            self.assertEqual(rows[1]["cache_key"], "sha256:group-a")

    def test_candidate_and_sidecar_scripts_exclude_benchmark_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            analysis = root / "unified_reuse_analysis.jsonl"
            analysis.write_text(
                json.dumps(
                    {
                        "source_kind": "grouped_prompt",
                        "source_name": "qasper",
                        "request_id": "req-1",
                        "reuse_group": "sha256:group-a",
                        "runtime_object_key": "runtime-prefix-a",
                        "reuse_class": "exact-next",
                        "group_size": 2,
                        "next_same_group_distance": 1,
                        "estimated_reusable_tokens": 256,
                        "estimated_kv_bytes": 512,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_dir = root / "candidates"
            with patch(
                "sys.argv",
                [
                    "build_candidate",
                    "--analysis-jsonl",
                    str(analysis),
                    "--output-dir",
                    str(candidate_dir),
                    "--source-name",
                    "qasper",
                    "--predicted-class",
                    "exact-next",
                ],
            ):
                self.assertEqual(build_candidate_main(), 0)
            candidate_rows = [
                json.loads(line)
                for line in (candidate_dir / "predictor_candidate_report.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn("reuse_group", candidate_rows[0])
            sidecar_dir = root / "sidecar"
            with patch(
                "sys.argv",
                [
                    "build_sidecar",
                    "--candidate-report",
                    str(candidate_dir / "predictor_candidate_report.jsonl"),
                    "--output-dir",
                    str(sidecar_dir),
                    "--run-id",
                    "run-a",
                ],
            ):
                self.assertEqual(build_sidecar_main(), 0)
            sidecar_rows = [
                json.loads(line)
                for line in (sidecar_dir / "sidecar_prediction.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(sidecar_rows[0]["candidate_object_id"], "runtime-prefix-a")
            self.assertNotIn("reuse_group", sidecar_rows[0])
            self.assertNotIn("task", sidecar_rows[0])

    def test_grouped_exact_next_entrypoint_declares_sidecar_runtime_env(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = (root / "scripts" / "entrypoints" / "run_grouped_exact_next_predictor_suite.sh").read_text(encoding="utf-8")
        for required in (
            "qasper",
            "multifieldqa_en",
            "ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH",
            "materialize_grouped_exact_next_workload.py",
            "build_sidecar_prediction.py",
            "build_observation_feasibility_report.py",
        ):
            self.assertIn(required, content)

    def test_sync_to_dgx_defaults_match_dgx_plan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = (root / "scripts" / "dev" / "sync_to_dgx.ps1").read_text(encoding="utf-8")
        self.assertIn('[int]$Port = 10000', content)
        self.assertIn('[string]$RemoteRoot = "/home/zyx/astrakv-W"', content)


if __name__ == "__main__":
    unittest.main()
