from __future__ import annotations

import unittest

from scripts.reporting.build_e11_slo_demo import build_demo_result, render_terminal


def _attribution(rows: list[dict], *, divergence: int | None = 4) -> dict:
    return {
        "cells": {
            "rep-1/qasper__scan_pollution_past_observed": {
                "first_divergence_ordinal": divergence,
                "requests": rows,
            }
        },
        "evidence_gaps": [],
    }


class E11SloDemoTest(unittest.TestCase):
    def test_builds_positive_p95_and_slo_result(self) -> None:
        rows = [
            {"phase": "post_divergence", "lru_ttft_ms": 1700, "astrakv_ttft_ms": 1500},
            {"phase": "post_divergence", "lru_ttft_ms": 1600, "astrakv_ttft_ms": 1400},
            {"phase": "post_divergence", "lru_ttft_ms": 1800, "astrakv_ttft_ms": 1550},
            {"phase": "pre_divergence", "lru_ttft_ms": 1000, "astrakv_ttft_ms": 1300},
        ]

        result = build_demo_result(_attribution(rows), slo_ms=1600)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["lru_ttft_p95_ms"], 1800.0)
        self.assertEqual(result["metrics"]["astrakv_ttft_p95_ms"], 1550.0)
        self.assertAlmostEqual(result["metrics"]["improvement_percent"], 13.89)
        self.assertFalse(result["slo"]["lru_pass"])
        self.assertTrue(result["slo"]["astrakv_pass"])
        terminal = render_terminal(result)
        self.assertIn("LMCache LRU             1800.00 ms", terminal)
        self.assertIn("AstraKV-W evict-B       1550.00 ms", terminal)
        self.assertIn("P95 reduction             13.89 %", terminal)
        self.assertNotIn("SLO", terminal)

    def test_refuses_to_claim_result_without_enough_paired_requests(self) -> None:
        rows = [
            {"phase": "post_divergence", "lru_ttft_ms": 1800, "astrakv_ttft_ms": 1200},
            {"phase": "pre_divergence", "lru_ttft_ms": 1800, "astrakv_ttft_ms": 1200},
        ]

        result = build_demo_result(_attribution(rows), slo_ms=1600)

        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["eligible"])
        self.assertNotIn("metrics", result)

    def test_reports_real_no_improvement_instead_of_forcing_positive_result(self) -> None:
        rows = [
            {"phase": "post_divergence", "lru_ttft_ms": 1200, "astrakv_ttft_ms": 1300},
            {"phase": "post_divergence", "lru_ttft_ms": 1250, "astrakv_ttft_ms": 1350},
            {"phase": "post_divergence", "lru_ttft_ms": 1300, "astrakv_ttft_ms": 1400},
        ]

        result = build_demo_result(_attribution(rows), slo_ms=1600)

        self.assertEqual(result["status"], "no_improvement")
        self.assertLess(result["metrics"]["improvement_percent"], 0)


if __name__ == "__main__":
    unittest.main()
