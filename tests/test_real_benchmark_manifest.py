import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrakv.benchmarks.experiment_manifest import file_sha256, normalized_lmcache_config_sha256
from scripts.benchmark.run_real_benchmark import (
    finalize_experiment_manifest,
    parse_args,
    write_effective_config,
)
from scripts.reporting.compare_real_runs import RunInput, comparison_claim


class RealBenchmarkManifestTests(unittest.TestCase):
    def test_warmup_claim_scope_is_accepted_by_runner(self) -> None:
        with patch(
            "scripts.benchmark.run_real_benchmark.sys.argv",
            ["run_real_benchmark.py", "--claim-scope", "online_control_warmup"],
        ):
            args = parse_args()
        self.assertEqual(args.claim_scope, "online_control_warmup")

    def test_controlled_suite_exports_static_server_controls_to_benchmark(self) -> None:
        suite = Path(__file__).resolve().parents[1] / "scripts" / "entrypoints" / "run_kv_core_controlled_suite.sh"
        text = suite.read_text(encoding="utf-8")
        benchmark_start = text.index('scripts/benchmark/run_real_benchmark.py')
        static_controls = text[max(0, benchmark_start - 1000):benchmark_start]
        self.assertIn('ASTRAKV_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"', static_controls)
        self.assertIn('ASTRAKV_MAX_MODEL_LEN="32768"', static_controls)
        self.assertIn('ASTRAKV_PREFIX_CACHING=false', static_controls)
        self.assertIn('ASTRAKV_KV_TRANSFER_CONFIG=', static_controls)
        self.assertIn('LMCACHE_CONFIG_FILE=', static_controls)

    def test_deployment_requires_exact_reverse_patch_for_existing_adapter(self) -> None:
        deployment = Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "prepare_kv_core_v2_deployment.sh"
        text = deployment.read_text(encoding="utf-8")
        self.assertIn('patch --batch --dry-run --reverse -p1 < "$PATCH_FILE"', text)
        self.assertIn("installed adapter is not an exact v3 patch result", text)

    def test_lmcache_control_hash_ignores_only_pair_scoped_disk_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = root / "baseline.yaml"
            variant = root / "variant.yaml"
            baseline.write_text(
                "local_cpu: false\nmax_local_cpu_size: 2.0\nlocal_disk: /tmp/pair/baseline\nmax_local_disk_size: 80.0\n",
                encoding="utf-8",
            )
            variant.write_text(
                "local_cpu: false\nmax_local_cpu_size: 2.0\nlocal_disk: /tmp/pair/variant\nmax_local_disk_size: 80.0\n",
                encoding="utf-8",
            )
            self.assertEqual(
                normalized_lmcache_config_sha256(baseline),
                normalized_lmcache_config_sha256(variant),
            )
            variant.write_text(variant.read_text(encoding="utf-8").replace("80.0", "79.0"), encoding="utf-8")
            self.assertNotEqual(
                normalized_lmcache_config_sha256(baseline),
                normalized_lmcache_config_sha256(variant),
            )

    def test_exported_command_and_config_redact_runtime_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            output = root / "run"
            output.mkdir()
            workload = root / "workload.jsonl"
            workload.write_text(json.dumps({"case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            write_csv(output / "benchmark_results.csv", ["case", "request_count", "success_count"], [{"case": "case-a", "request_count": 1, "success_count": 1}])
            (output / "request_results.jsonl").write_text(json.dumps({"schema": "astrakv-benchmark-request-v2", "run_id": "run-1", "request_id": "req-1", "case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            args = manifest_args(workload)
            args.request_context_url = "http://127.0.0.1:17991/request-context"
            args.enable_samples = False
            args.run_name = "run-1"
            args.timeout = 60.0
            args.temperature = 0.0
            args.top_p = 1.0
            args.prompt_token_scale = 1.0
            args.metrics_interval = 0.5
            args.disk_device = ""
            args.process_name_filters = ()
            secret = "a" * 64
            api_key = "api-secret"

            with patch("scripts.benchmark.run_real_benchmark.sys.argv", [
                "run_real_benchmark.py", "--request-context-secret-hex", secret, f"--api-key={api_key}",
            ]):
                finalize_experiment_manifest(output, args, {})
            write_effective_config(
                output / "benchmark_config.json", args,
                {"runtime": {"request_context_secret_hex": secret}, "backend": {"api_key": api_key}},
            )

            manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
            config = json.loads((output / "benchmark_config.json").read_text(encoding="utf-8"))
            self.assertNotIn(secret, manifest["command"])
            self.assertNotIn(api_key, manifest["command"])
            self.assertNotIn(secret, json.dumps(config))
            self.assertNotIn(api_key, json.dumps(config))
            self.assertIn("[REDACTED]", manifest["command"])
            self.assertEqual(config["raw_config"]["runtime"]["request_context_secret_hex"], "[REDACTED]")
            self.assertEqual(config["raw_config"]["backend"]["api_key"], "[REDACTED]")

    def test_final_manifest_exports_runtime_state_without_manual_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            output, state = root / "run", root / "state"
            output.mkdir()
            state.mkdir()
            workload = root / "workload.jsonl"
            workload.write_text(json.dumps({"case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            write_csv(output / "benchmark_results.csv", ["case", "request_count", "success_count"], [{"case": "case-a", "request_count": 1, "success_count": 1}])
            (output / "request_results.jsonl").write_text(json.dumps({"schema": "astrakv-benchmark-request-v2", "run_id": "run-1", "request_id": "req-1", "case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            write_jsonl(state / "bindings.jsonl", [{"schema": "astrakv-backend-hook-v2", "record_type": "binding", "run_id": "run-1", "binding_id": "binding-1", "binding_generation": 1, "request_id": "req-1", "object_key": "prefix-1", "object_level": "prefix", "backend_object_id": "backend-1"}])
            write_jsonl(state / "events.jsonl", [{"schema": "astrakv-backend-hook-v2", "record_type": "event", "run_id": "run-1", "request_id": "req-1", "object_key": "prefix-1", "object_level": "prefix", "backend_object_id": "backend-1", "metadata": {"binding_id": "binding-1", "binding_generation": 1}}])
            write_jsonl(state / "commands.jsonl", [])
            write_jsonl(state / "receipts.jsonl", [])
            (state / "request_context_associations.jsonl").write_text(
                json.dumps({
                    "schema": "astrakv-request-context-v1",
                    "record_type": "request_context_receipt",
                    "run_id": "run-1",
                    "request_id": "req-1",
                    "request_nonce": "11111111-1111-4111-8111-111111111111",
                    "runtime_request_id": "reqmeta-1",
                    "runtime_event_id": "runtime-context:reqmeta-1",
                    "status": "associated",
                }) + "\n",
                encoding="utf-8",
            )
            (state / "preflight.json").write_text(json.dumps({"schema": "preflight", "run_id": "run-1", "compatible": True}), encoding="utf-8")
            args = manifest_args(workload, runtime_state_dir=str(state))

            finalize_experiment_manifest(output, args, {})

            payload = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["artifact_paths"]).issuperset(
                    {
                        "backend_capabilities",
                        "backend_binding_events",
                        "runtime_events_raw",
                        "astrakv_runtime_commands",
                        "runtime_command_receipts",
                        "request_context_associations",
                        "runtime_structured_events",
                        "online_profile_checkpoint",
                        "trace",
                    }
                ),
                True,
            )

    def test_final_manifest_records_actual_runner_artifacts_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "run"
            output.mkdir(parents=True)
            workload = Path(raw_tmp) / "workload.jsonl"
            workload.write_text(json.dumps({"case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            write_csv(output / "benchmark_results.csv", ["case", "request_count", "success_count"], [{"case": "case-a", "request_count": 1, "success_count": 1}])
            (output / "request_results.jsonl").write_text(json.dumps({"schema": "astrakv-benchmark-request-v2", "run_id": "run-1", "request_id": "req-1", "case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                run_id="run-1", workload_id="workload-1", workload_jsonl=str(workload), model="model", model_revision="r1",
                tokenizer_revision="t1", dtype="bf16", quantization="unquantized", random_seed="7", cache_state="cold",
                connector_version="lmcache-0.4.7", context_lengths=[128], batch_sizes=[1], output_tokens=16, repeat=1,
                temperature=0.0, top_p=1.0, backend="vllm", base_url="http://127.0.0.1:8000", pair_id="pair-1",
                pair_role="baseline", claim_scope="benchmark", online_artifact=(), config="",
            )

            finalize_experiment_manifest(output, args, {})

            payload = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "astrakv-experiment-manifest-v2")
            self.assertEqual(payload["pair_id"], "pair-1")
            self.assertEqual(payload["pair_role"], "baseline")
            self.assertEqual(payload["claim_scope"], "benchmark")
            for role in ("workload", "matrix", "environment", "benchmark", "requests", "quality"):
                artifact = output / payload["artifact_paths"][role]
                self.assertTrue(artifact.is_file())
                self.assertEqual(payload["artifact_hashes"][role], file_sha256(artifact))
            self.assertEqual(payload["workload_sha256"], payload["artifact_hashes"]["workload"])
            self.assertEqual(payload["matrix_sha256"], payload["artifact_hashes"]["matrix"])
            self.assertEqual(payload["environment_sha256"], payload["artifact_hashes"]["environment"])

    def test_finalized_runner_fixtures_are_accepted_by_default_paired_compare(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            workload = root / "workload.jsonl"
            workload.write_text(json.dumps({"case": "case-a", "sample_id": "sample-a"}) + "\n", encoding="utf-8")
            baseline = make_finalized_run(root / "baseline", workload, "baseline")
            variant = make_finalized_run(root / "variant", workload, "variant")

            claim, validation = comparison_claim([
                RunInput("baseline", baseline / "benchmark_results.csv"),
                RunInput("variant", variant / "benchmark_results.csv"),
            ], unpaired=False)

            self.assertEqual(claim, "paired_claim_eligible")
            self.assertIsNotNone(validation)
            self.assertTrue(validation.eligible)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def manifest_args(workload: Path, **extra: object) -> SimpleNamespace:
    values = dict(
        run_id="run-1", workload_id="workload-1", workload_jsonl=str(workload), model="model", model_revision="r1",
        tokenizer_revision="t1", dtype="bf16", quantization="unquantized", random_seed="7", cache_state="cold",
        connector_version="lmcache-0.4.7", context_lengths=[128], batch_sizes=[1], output_tokens=16, repeat=1,
        temperature=0.0, top_p=1.0, backend="vllm", base_url="http://127.0.0.1:8000", pair_id="pair-1",
        pair_role="baseline", claim_scope="online_control", online_artifact=(), config="", runtime_state_dir="",
    )
    values.update(extra)
    return SimpleNamespace(**values)


def make_finalized_run(output: Path, workload: Path, role: str) -> Path:
    output.mkdir(parents=True)
    write_csv(output / "benchmark_results.csv", ["case", "request_count", "success_count"], [{"case": "case-a", "request_count": 1, "success_count": 1}])
    (output / "request_results.jsonl").write_text(json.dumps({"schema": "astrakv-benchmark-request-v2", "run_id": "run-1", "request_id": "req-1", "case": "case-a", "sample_id": "sample-a", "output_text": "ok", "status": "ok"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        run_id="run-1", workload_id="workload-1", workload_jsonl=str(workload), model="model", model_revision="r1",
        tokenizer_revision="t1", dtype="bf16", quantization="unquantized", random_seed="7", cache_state="cold",
        connector_version="lmcache-0.4.7", context_lengths=[128], batch_sizes=[1], output_tokens=16, repeat=1,
        temperature=0.0, top_p=1.0, backend="vllm", base_url="http://127.0.0.1:8000", pair_id="pair-1",
        pair_role=role, claim_scope="benchmark", online_artifact=(), config="",
    )
    finalize_experiment_manifest(output, args, {})
    return output


if __name__ == "__main__":
    unittest.main()
