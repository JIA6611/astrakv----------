"""Unit tests for E5/E5C acceptance validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.validate_kv_core_acceptance import PHASES, validate_external_reaping


def _write_reaps(run: Path, rows: list[dict[str, object]]) -> None:
    (run / "kv_core_external_reaps.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


class E5AcceptanceTests(unittest.TestCase):
    def test_phases_include_e5_and_e5c(self) -> None:
        self.assertIn("E5", PHASES)
        self.assertIn("E5C", PHASES)

    def test_active_phase_accepts_reap_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            _write_reaps(run, [{
                "status": "invalidated",
                "freed_bytes": 6144,
                "demoted_keys": 2,
                "invalidated_keys": 2,
            }])
            errors: list[str] = []
            record = validate_external_reaping(run, errors, expect_reaps=True)
            self.assertEqual(errors, [])
            self.assertEqual(record["reap_count"], 1)
            self.assertEqual(record["freed_bytes"], 6144)

    def test_active_phase_fails_closed_without_reaps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            _write_reaps(run, [])
            errors: list[str] = []
            validate_external_reaping(run, errors, expect_reaps=True)
            self.assertIn("e5_external_reap_evidence_missing", errors)

    def test_active_phase_rejects_zero_freed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            _write_reaps(run, [{"status": "demoted", "freed_bytes": 0}])
            errors: list[str] = []
            validate_external_reaping(run, errors, expect_reaps=True)
            self.assertIn("e5_reap_freed_bytes_missing", errors)

    def test_control_phase_fails_closed_on_reap_leak(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            _write_reaps(run, [{"status": "invalidated", "freed_bytes": 1024}])
            errors: list[str] = []
            validate_external_reaping(run, errors, expect_reaps=False)
            self.assertIn("e5c_reap_control_leak", errors)

    def test_control_phase_accepts_clean_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            _write_reaps(run, [])
            errors: list[str] = []
            record = validate_external_reaping(run, errors, expect_reaps=False)
            self.assertEqual(errors, [])
            self.assertEqual(record["reap_count"], 0)


if __name__ == "__main__":
    unittest.main()
