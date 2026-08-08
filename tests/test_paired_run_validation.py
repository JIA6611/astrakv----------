import csv
import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.experiment_manifest import ExperimentManifest, file_sha256, write_experiment_manifest
from astrakv.runtime.backend_capabilities import build_installation_evidence
from astrakv.benchmarks.paired_run import PairedRunInput, validate_paired_runs, write_paired_run_manifest


class PairedRunValidationTests(unittest.TestCase):
    def test_experiment_manifest_redacts_sensitive_command_options(self) -> None:
        record = ExperimentManifest(
            run_id="run-1",
            command="runner --request-context-secret-hex exposed-secret --api-key=exposed-api-key",
        ).to_record()

        self.assertNotIn("exposed-secret", record["command"])
        self.assertNotIn("exposed-api-key", record["command"])
        self.assertIn("[REDACTED]", record["command"])

    def test_experiment_manifest_writer_hashes_declared_relative_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            benchmark = root / "benchmark_results.csv"
            benchmark.write_text("case\ncase-a\n", encoding="utf-8")
            write_experiment_manifest(
                root / "experiment_manifest.json",
                ExperimentManifest(run_id="run-1", artifact_paths={"benchmark": "benchmark_results.csv"}),
            )

            payload = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_hashes"]["benchmark"], file_sha256(benchmark))

    def test_matching_paired_runs_with_online_evidence_are_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertTrue(result.eligible)
            self.assertEqual(result.errors, ())
            self.assertEqual(result.record["pair_id"], "pair-1")
            self.assertEqual(result.record["coverage"]["cases"], ["case-a", "case-b"])
            self.assertIn("runtime_command_receipts", result.record["artifact_hashes"]["variant"])

    def test_rejects_mismatched_reproducibility_hashes_and_case_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline")
            variant = make_run(root / "variant", role="variant", workload="different", cases=("case-a",))

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("workload_sha256_mismatch", result.errors)
            self.assertIn("case_coverage_mismatch", result.errors)

    def test_control_environment_allows_runtime_mode_difference_but_rejects_changed_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline")
            variant = make_run(root / "variant", role="variant")
            attach_control_environment(baseline, {"model": "m@r1", "kv_budget": 1024})
            attach_control_environment(variant, {"model": "m@r1", "kv_budget": 1024})
            # Full environment remains a per-run audit artifact and can differ.
            (variant / "environment.json").write_text(json.dumps({"runtime_mode": "shadow"}), encoding="utf-8")
            manifest_path = variant / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["environment_sha256"] = file_sha256(variant / "environment.json")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            refresh_artifact_hashes(variant)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertTrue(result.eligible)
            attach_control_environment(variant, {"model": "m@r1", "kv_budget": 768})
            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))
            self.assertFalse(result.eligible)
            self.assertIn("control_environment_sha256_mismatch", result.errors)

    def test_rejects_missing_artifacts_and_unproven_quality_samples(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline")
            variant = make_run(root / "variant", role="variant")
            (variant / "quality_results.csv").unlink()
            request_rows = read_jsonl(baseline / "request_results.jsonl")
            request_rows[0]["sample_id"] = "unmatched-sample"
            write_jsonl(baseline / "request_results.jsonl", request_rows)
            refresh_artifact_hashes(baseline)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("missing_artifact:variant:quality", result.errors)
            self.assertIn("sample_provenance_mismatch", result.errors)

    def test_online_control_claim_requires_consistent_command_receipt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)
            receipts = read_jsonl(variant / "runtime_command_receipts.jsonl")
            receipts[0]["command_id"] = "wrong-command"
            write_jsonl(variant / "runtime_command_receipts.jsonl", receipts)
            refresh_artifact_hashes(variant)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("receipt_command_mismatch:variant", result.errors)
            output = root / "paired_run_manifest.json"
            write_paired_run_manifest(output, result)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["eligible"])
            self.assertIn("receipt_command_mismatch:variant", payload["errors"])

    def test_rejects_mixed_claim_scope_and_forged_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant")
            payload = json.loads((variant / "experiment_manifest.json").read_text(encoding="utf-8"))
            payload["workload_sha256"] = "0" * 64
            (variant / "experiment_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("claim_scope_mismatch", result.errors)
            self.assertIn("source_hash_mismatch:variant:workload", result.errors)

    def test_rejects_malformed_duplicate_and_incomplete_online_chain_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)
            with (variant / "runtime_events_raw.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("not json\n")
            commands = read_jsonl(variant / "astrakv_runtime_commands.jsonl")
            commands.append(dict(commands[0]))
            write_jsonl(variant / "astrakv_runtime_commands.jsonl", commands)
            refresh_artifact_hashes(variant)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("malformed_jsonl:variant:runtime_events_raw", result.errors)
            self.assertIn("duplicate_command_id:variant", result.errors)

    def test_rejects_runtime_event_projection_with_observation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)
            with (variant / "runtime_events_raw.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema": "astrakv-backend-hook-v2", "record_type": "observation", "run_id": "run-1"}) + "\n")
            refresh_artifact_hashes(variant)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("invalid_event_row:variant", result.errors)

    def test_rejects_non_successful_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)
            receipts = read_jsonl(variant / "runtime_command_receipts.jsonl")
            receipts[0]["status"] = "failed"
            write_jsonl(variant / "runtime_command_receipts.jsonl", receipts)
            refresh_artifact_hashes(variant)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertFalse(result.eligible)
            self.assertIn("terminal_receipt_count_mismatch:variant:command-1", result.errors)

    def test_accepts_reused_binding_generation_across_multiple_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            baseline = make_run(root / "baseline", role="baseline", online_control=True)
            variant = make_run(root / "variant", role="variant", online_control=True)
            for run_root in (baseline, variant):
                bindings = read_jsonl(run_root / "backend_binding_events.jsonl")
                bindings[0]["request_id"] = "req-case-b"
                write_jsonl(run_root / "backend_binding_events.jsonl", bindings)
                refresh_artifact_hashes(run_root)

            result = validate_paired_runs(PairedRunInput("baseline", baseline), PairedRunInput("variant", variant))

            self.assertTrue(result.eligible)
            self.assertEqual(result.errors, ())


def make_run(root: Path, *, role: str, workload: str = "workload", cases: tuple[str, ...] = ("case-a", "case-b"), online_control: bool = False) -> Path:
    root.mkdir(parents=True)
    workload_path = root / "workload.jsonl"
    write_jsonl(workload_path, [{"case": case, "sample_id": case} for case in cases])
    matrix = root / "matrix.json"
    matrix.write_text(json.dumps({"cases": list(cases)}), encoding="utf-8")
    environment = root / "environment.json"
    environment.write_text(json.dumps({"image": "sha256:image", "model": "m@r1"}), encoding="utf-8")
    write_rows(root / "benchmark_results.csv", ["case", "request_count", "success_count"], [
        {"case": case, "request_count": 1, "success_count": 1} for case in cases
    ])
    write_jsonl(root / "request_results.jsonl", [{"case": case, "sample_id": case, "request_id": f"req-{case}"} for case in cases])
    write_rows(root / "quality_results.csv", ["sample_id", "exact_match"], [{"sample_id": case, "exact_match": 1} for case in cases])
    artifact_paths = {
        "benchmark": "benchmark_results.csv", "requests": "request_results.jsonl", "quality": "quality_results.csv",
    }
    if online_control:
        binding = {"schema": "astrakv-backend-hook-v2", "record_type": "binding", "binding_id": "binding-1", "binding_generation": 1, "run_id": "run-1", "request_id": "req-case-a", "object_key": "object-a", "object_level": "prefix", "backend_object_id": "backend-a", "verified": True}
        write_jsonl(root / "backend_binding_events.jsonl", [binding])
        write_jsonl(root / "runtime_events_raw.jsonl", [{"schema": "astrakv-backend-hook-v2", "record_type": "event", "event_id": "event-1", "binding_id": "binding-1", "binding_generation": 1, "request_id": "req-case-a", "run_id": "run-1", "object_key": "object-a", "object_level": "prefix", "backend_object_id": "backend-a", "action": "cache_store", "status": "completed", "timestamp_ns": 1}])
        write_jsonl(root / "astrakv_runtime_commands.jsonl", [{"schema": "astrakv-backend-hook-v2", "record_type": "command", "command_id": "command-1", "decision_id": "decision-1", "binding_id": "binding-1", "binding_generation": 1, "request_id": "req-case-a", "run_id": "run-1", "object_key": "object-a", "object_level": "prefix", "backend_object_id": "backend-a", "action": "drop", "issued_at_ns": 2}])
        write_jsonl(root / "runtime_command_receipts.jsonl", [{"schema": "astrakv-backend-hook-v2", "record_type": "receipt", "receipt_id": "receipt-1", "command_id": "command-1", "binding_id": "binding-1", "binding_generation": 1, "run_id": "run-1", "backend_object_id": "backend-a", "action": "drop", "status": "completed", "timestamp_ns": 3}])
        write_jsonl(root / "runtime_structured_events.jsonl", [])
        write_jsonl(root / "trace_events.jsonl", [{"schema": "astrakv-trace-event-v1", "request_id": f"req-{case}", "run_id": "run-1"} for case in cases])
        (root / "online_profile_checkpoint.json").write_text(json.dumps({"schema": "astrakv-online-profile-v1", "run_id": "run-1", "event_count": 0, "last_event_id": None, "objects": {}, "events": [], "event_fingerprints": {}}), encoding="utf-8")
        evidence = build_installation_evidence(source="test", method="test", session_id="session", vllm_version="0.23.0", lmcache_version="0.4.7", connector_name="lmcache-vllm-v1", connector_version="0.4.7", endpoint_identity="http://127.0.0.1:7900/actions")
        (root / "backend_capabilities.json").write_text(json.dumps({"schema": "astrakv-backend-capabilities-v1", "compatible": True, "execution_eligible": True, "run_id": "run-1", "endpoint_identity": "http://127.0.0.1:7900/actions", "backend_versions": {"vllm": "0.23.0", "lmcache": "0.4.7"}, "connector": {"name": "lmcache-vllm-v1", "version": "0.4.7"}, "allowed_actions": ["drop"], "object_levels": ["prefix"], "capability_flags": {"version": True, "installation_evidence": True}, "installation_evidence": evidence.to_record()}), encoding="utf-8")
        artifact_paths.update({
            "backend_capabilities": "backend_capabilities.json",
            "backend_binding_events": "backend_binding_events.jsonl",
            "runtime_events_raw": "runtime_events_raw.jsonl",
            "astrakv_runtime_commands": "astrakv_runtime_commands.jsonl",
            "runtime_command_receipts": "runtime_command_receipts.jsonl",
            "runtime_structured_events": "runtime_structured_events.jsonl",
            "online_profile_checkpoint": "online_profile_checkpoint.json",
            "trace": "trace_events.jsonl",
        })
    manifest = ExperimentManifest(
        run_id="run-1", workload_id=workload, workload_path=str(workload_path), workload_sha256=file_sha256(workload_path),
        model="m", model_revision="r1", tokenizer_revision="t1", dtype="bf16", quantization="unquantized", random_seed="7", cache_state="cold",
        pair_id="pair-1", pair_role=role, matrix_sha256=file_sha256(matrix), environment_sha256=file_sha256(environment),
        artifact_paths={**artifact_paths, "workload": "workload.jsonl", "matrix": "matrix.json", "environment": "environment.json"}, claim_scope="online_control" if online_control else "benchmark",
    )
    write_experiment_manifest(root / "experiment_manifest.json", manifest)
    refresh_artifact_hashes(root)
    return root


def refresh_artifact_hashes(root: Path) -> None:
    path = root / "experiment_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_hashes"] = {role: file_sha256(root / relpath) for role, relpath in payload["artifact_paths"].items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def attach_control_environment(root: Path, payload: dict[str, object]) -> None:
    control = root / "control_environment.json"
    control.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path = root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["control_environment_sha256"] = file_sha256(control)
    manifest["artifact_paths"]["control_environment"] = control.name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    refresh_artifact_hashes(root)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
