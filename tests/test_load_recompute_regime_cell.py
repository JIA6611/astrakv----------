"""Unit tests for variant-only regime cell semantics."""

from __future__ import annotations

import unittest

from scripts.reporting.validate_load_recompute_regime_cell import validate_arm_semantics


def _row(*, lookup: int, allocated: int, loaded: int, recomputed: int = 0) -> dict:
    return {
        "lookup_hit_tokens": lookup,
        "allocated_external_tokens": allocated,
        "actual_loaded_tokens": loaded,
        "recomputed_tokens": recomputed,
        "recompute_confirmed": True,
    }


class RegimeCellSemanticsTests(unittest.TestCase):
    def test_full_requires_complete_loaded_hit(self) -> None:
        errors: list[str] = []
        validate_arm_semantics(
            "full",
            [_row(lookup=4096, allocated=4096, loaded=4096)],
            [],
            [{"prefetch_id": "p", "source_tier": "ssd", "target_tier": "cpu", "status": "consumed", "completed_bytes": 1, "target_request_id": "r", "consumer_request_id": "r", "physical_object_id": "o", "binding_generation": 1, "prefix_hash": "h", "native_key": "k", "compatibility_identity": "c"}],
            errors,
        )
        self.assertEqual(errors, [])

    def test_partial_rejects_degenerate_full_load(self) -> None:
        errors: list[str] = []
        validate_arm_semantics(
            "partial",
            [_row(lookup=2048, allocated=2048, loaded=2048)],
            [],
            [],
            errors,
        )
        self.assertIn("e4_partial_prefix_evidence_missing", errors)

    def test_partial_accepts_strict_native_prefix_split(self) -> None:
        errors: list[str] = []
        validate_arm_semantics(
            "partial",
            [_row(lookup=4096, allocated=2048, loaded=2048)],
            [],
            [{"prefetch_id": "p", "source_tier": "ssd", "target_tier": "cpu", "status": "consumed", "completed_bytes": 1, "target_request_id": "r", "consumer_request_id": "r", "physical_object_id": "o", "binding_generation": 1, "prefix_hash": "h", "native_key": "k", "compatibility_identity": "c"}],
            errors,
        )
        self.assertEqual(errors, [])

    def test_recompute_only_requires_zero_load_and_forced_decisions(self) -> None:
        errors: list[str] = []
        validate_arm_semantics(
            "recompute_only",
            [_row(lookup=4096, allocated=0, loaded=0, recomputed=4600)],
            [{"reason": "equivalence_probe_force_recompute", "test_only": True}],
            [],
            errors,
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
