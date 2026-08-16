from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.validate_e12_mainline_smoke import validate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class ValidateE12MainlineSmokeTests(unittest.TestCase):
    def test_runner_builds_complete_pairs_on_a_same_server_ssd_seed(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/entrypoints/run_e12_mainline_audit_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ASTRAKV_ABLATION_WARMUP_PASSES=1", script)
        self.assertIn("ASTRAKV_ABLATION_WARMUP_SAME_SERVER=true", script)
        self.assertIn("ASTRAKV_ABLATION_MEASURE_PHASES=far,near", script)
        self.assertIn("ASTRAKV_ABLATION_PREDICTION_PHASES=far", script)
        self.assertIn("--interleave-pattern fire-consume", script)
        self.assertIn("ASTRAKV_PREFIX_CACHING=false", script)

    def _valid_bundle(self, root: Path) -> tuple[Path, Path]:
        run = root / "run"
        state = root / "state"
        run.mkdir()
        state.mkdir()
        write_jsonl(state / "kv_core_policy_decisions.jsonl", [
            {
                "request_id": "req-1",
                "prefetch_id": "a-1",
                "physical_object_id": "physical-1",
                "binding_generation": 2,
                "action": "prefetch_ssd_to_cpu",
                "status": "submitted",
                "timestamp_ns": 10,
            },
            {
                "request_id": "req-1",
                "physical_object_id": "physical-1",
                "binding_generation": 2,
                "action": "admit_external_prefix",
                "reason": "native_load_cheaper",
                "candidate_external_tokens": 1024,
                "timestamp_ns": 20,
            },
        ])
        write_jsonl(state / "kv_core_prefetch_tickets.jsonl", [{
            "prefetch_id": "a-1",
            "physical_object_id": "physical-1",
            "binding_generation": 2,
            "target_request_id": "req-1",
            "consumer_request_id": "req-1",
            "status": "consumed",
            "completed_bytes": 4096,
        }])
        write_jsonl(run / "runtime_events_raw.jsonl", [{
            "schema": "astrakv-backend-hook-v2",
            "record_type": "event",
            "run_id": "run-1",
            "request_id": "req-1",
            "event_id": "release-1",
            "action": "release",
            "status": "completed",
            "object_key": "prefix-1",
            "backend_object_id": "backend-1",
            "binding_id": "binding-1",
            "binding_generation": 2,
            "timestamp_ns": 30,
            "metadata": {"bridge_eligible": True},
        }])
        write_jsonl(run / "astrakv_runtime_commands.jsonl", [{
            "record_type": "command",
            "run_id": "run-1",
            "request_id": "req-1",
            "command_id": "b-command-1",
            "action": "prefetch",
            "object_key": "prefix-1",
            "backend_object_id": "backend-1",
            "binding_id": "binding-1",
            "binding_generation": 2,
            "issued_at_ns": 40,
            "metadata": {
                "prefetch_kind": "next_use",
                "dispatch_origin": "release_completed",
            },
        }])
        write_jsonl(run / "runtime_command_receipts.jsonl", [{
            "record_type": "receipt",
            "run_id": "run-1",
            "command_id": "b-command-1",
            "receipt_id": "b-receipt-1",
            "action": "prefetch",
            "status": "completed",
            "bytes": 4096,
            "backend_object_id": "backend-1",
            "binding_id": "binding-1",
            "binding_generation": 2,
            "timestamp_ns": 50,
        }])
        write_jsonl(run / "request_results.jsonl", [{
            "run_id": "run-1", "request_id": "req-1", "status": "ok",
        }])
        return run, state

    def test_accepts_one_strict_ordered_control_chain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            result = validate(run, state)

        self.assertTrue(result["eligible"])
        self.assertEqual(result["validation_status"], "PASS")
        self.assertEqual(result["complete_chain_count"], 1)
        chain = result["complete_chains"][0]
        self.assertEqual(chain["request_id"], "req-1")
        self.assertEqual(chain["lookup"]["action"], "admit_external_prefix")
        self.assertEqual(chain["prefetch_b"]["moved_bytes"], 4096)

    def test_rejects_bundle_without_prefetch_b_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            (run / "runtime_command_receipts.jsonl").write_text("", encoding="utf-8")
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["validation_status"], "INCOMPLETE")
        self.assertIn("no_complete_control_chain", result["errors"])

    def test_prefetched_flag_cannot_substitute_for_positive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            receipts = [json.loads(line) for line in (
                run / "runtime_command_receipts.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            receipts[0]["bytes"] = 0
            receipts[0]["metadata"] = {"prefetched": 1}
            write_jsonl(run / "runtime_command_receipts.jsonl", receipts)
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["complete_chain_count"], 0)

    def test_rejects_prefetch_a_ticket_for_a_different_target_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            tickets = [json.loads(line) for line in (
                state / "kv_core_prefetch_tickets.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            tickets[0]["target_request_id"] = "req-other"
            write_jsonl(state / "kv_core_prefetch_tickets.jsonl", tickets)
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["complete_chain_count"], 0)

    def test_rejects_chain_with_prefetch_b_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            commands = [json.loads(line) for line in (
                run / "astrakv_runtime_commands.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            commands[0]["issued_at_ns"] = 25
            write_jsonl(run / "astrakv_runtime_commands.jsonl", commands)
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["complete_chain_count"], 0)

    def test_rejects_receipt_with_conflicting_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            receipts = [json.loads(line) for line in (
                run / "runtime_command_receipts.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            receipts[0]["run_id"] = "run-2"
            write_jsonl(run / "runtime_command_receipts.jsonl", receipts)
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["validation_status"], "INVALID")
        self.assertIn("run_id_not_unique", result["errors"])

    def test_rejects_failed_anchor_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            write_jsonl(run / "request_results.jsonl", [{
                "run_id": "run-1", "request_id": "req-1", "status": "failed",
            }])
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["validation_status"], "INCOMPLETE")
        self.assertEqual(result["complete_chain_count"], 0)

    def test_drop_command_does_not_substitute_for_release_event(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            (run / "runtime_events_raw.jsonl").write_text("", encoding="utf-8")
            write_jsonl(run / "astrakv_runtime_commands.jsonl", [{
                "run_id": "run-1",
                "request_id": "req-1",
                "command_id": "drop-1",
                "action": "drop",
                "issued_at_ns": 30,
            }])
            result = validate(run, state)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["stage_request_counts"]["release"], 0)

    def test_native_eviction_is_reported_but_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run, state = self._valid_bundle(Path(raw))
            write_jsonl(run / "native_policy_installation.jsonl", [{
                "run_id": "run-1", "status": "installed",
            }])
            write_jsonl(run / "native_cache_policy_evictions.jsonl", [
                {"run_id": "run-1", "status": "selected"},
                {"run_id": "run-1", "status": "completed"},
            ])
            result = validate(run, state)

        self.assertTrue(result["eligible"])
        self.assertTrue(result["native_capacity_branch"]["installation_observed"])
        self.assertEqual(result["native_capacity_branch"]["completed_count"], 1)
        self.assertFalse(result["native_capacity_branch"]["in_pass_criteria"])


if __name__ == "__main__":
    unittest.main()
