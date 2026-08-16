import json
import tempfile
import unittest
from pathlib import Path

from astrakv.benchmarks.runtime_artifacts import export_online_control_artifacts, refresh_online_control_manifest


class RuntimeArtifactExportTests(unittest.TestCase):
    def test_exporter_writes_validator_ready_projection_and_trace(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            state = root / "state"
            output = root / "output"
            state.mkdir()
            output.mkdir()
            binding = {
                "schema": "astrakv-backend-hook-v2", "record_type": "binding", "run_id": "run-a",
                "binding_id": "binding-a", "binding_generation": 1, "request_id": "request-a",
                "object_key": "prefix-a", "object_level": "prefix", "backend_object_id": "backend-a", "verified": True,
            }
            write_jsonl(state / "bindings.jsonl", [binding, {**binding, "action": "release", "status": "completed"}])
            write_jsonl(state / "events.jsonl", [
                {"schema": "astrakv-backend-hook-v2", "record_type": "event", "run_id": "run-a", "request_id": "request-a", "object_key": "prefix-a", "object_level": "prefix", "backend_object_id": "backend-a", "action": "cache_store", "status": "completed", "metadata": {"binding_id": "binding-a", "binding_generation": 1}},
                {"schema": "astrakv-backend-hook-v2", "record_type": "observation", "run_id": "run-a"},
            ])
            write_jsonl(state / "commands.jsonl", [])
            write_jsonl(state / "receipts.jsonl", [])
            (state / "preflight.json").write_text(json.dumps({"schema": "preflight", "run_id": "run-a", "compatible": True}), encoding="utf-8")
            write_jsonl(output / "request_results.jsonl", [{"run_id": "run-a", "request_id": "request-a", "case": "case-a", "sample_id": "sample-a"}])

            exported = export_online_control_artifacts(state, output)

            self.assertEqual(
                set(exported),
                {
                    "backend_capabilities",
                    "backend_binding_events",
                    "runtime_events_raw",
                    "astrakv_runtime_commands",
                    "runtime_command_receipts",
                    "runtime_structured_events",
                    "native_policy_installation",
                    "native_cache_policy_evictions",
                    "kv_core_native_callbacks",
                    "kv_core_native_receipts",
                    "online_profile_checkpoint",
                    "trace",
                },
            )
            self.assertEqual(len(read_jsonl(exported["backend_binding_events"])), 1)
            events = read_jsonl(exported["runtime_events_raw"])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["record_type"], "event")
            self.assertEqual(read_jsonl(exported["trace"]), [{"schema": "astrakv-trace-event-v1", "run_id": "run-a", "request_id": "request-a"}])
            self.assertTrue(exported["online_profile_checkpoint"].is_file())

    def test_refresh_updates_manifest_hashes_after_action_artifacts_change(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            state, output = root / "state", root / "output"
            state.mkdir()
            output.mkdir()
            write_jsonl(state / "bindings.jsonl", [])
            write_jsonl(state / "events.jsonl", [])
            write_jsonl(state / "commands.jsonl", [{"record_type": "command", "command_id": "command-a"}])
            write_jsonl(state / "receipts.jsonl", [{"record_type": "receipt", "command_id": "command-a"}])
            (state / "preflight.json").write_text("{}", encoding="utf-8")
            write_jsonl(output / "request_results.jsonl", [{"run_id": "run-a", "request_id": "request-a"}])
            (output / "experiment_manifest.json").write_text(json.dumps({"artifact_paths": {}, "artifact_hashes": {}}), encoding="utf-8")

            refreshed = refresh_online_control_manifest(state, output)

            self.assertEqual(
                set(refreshed["artifact_paths"]).issuperset(
                    {
                        "backend_capabilities",
                        "backend_binding_events",
                        "runtime_events_raw",
                        "astrakv_runtime_commands",
                        "runtime_command_receipts",
                        "runtime_structured_events",
                        "native_policy_installation",
                        "native_cache_policy_evictions",
                        "kv_core_native_callbacks",
                        "kv_core_native_receipts",
                        "online_profile_checkpoint",
                        "trace",
                    }
                ),
                True,
            )
            self.assertEqual(
                refreshed["artifact_hashes"]["astrakv_runtime_commands"],
                sha256(output / "astrakv_runtime_commands.jsonl"),
            )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
