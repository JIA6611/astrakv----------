from __future__ import annotations

import unittest

from scripts.reporting.evaluate_e11_regime_matrix import (
    classify_winner,
    regime_for_cell,
)


class EvaluateE11RegimeMatrixTest(unittest.TestCase):
    def test_extracts_regime_from_repeat_dataset_cell(self) -> None:
        self.assertEqual(
            regime_for_cell("rep-2/qasper__scan_pollution_past_observed"),
            "scan_pollution_past_observed",
        )

    def test_astrakv_wins_only_with_directional_ci_and_quality(self) -> None:
        summary = {
            "paired_count": 12,
            "mean_delta_ci_percent": [-8.0, -3.0],
        }
        self.assertEqual(
            classify_winner(
                summary,
                correctness_valid=True,
                astrakv_quality_not_worse=True,
                lru_quality_not_worse=False,
                astrakv_quality_improved=True,
            ),
            "astrakv_wins",
        )
        self.assertEqual(
            classify_winner(
                summary,
                correctness_valid=True,
                astrakv_quality_not_worse=False,
                lru_quality_not_worse=False,
            ),
            "inconclusive",
        )

    def test_lru_wins_and_tie_are_symmetric(self) -> None:
        self.assertEqual(
            classify_winner(
                {"paired_count": 12, "mean_delta_ci_percent": [2.5, 6.0]},
                correctness_valid=True,
                astrakv_quality_not_worse=False,
                lru_quality_not_worse=True,
                lru_quality_improved=True,
            ),
            "lru_wins",
        )
        self.assertEqual(
            classify_winner(
                {"paired_count": 12, "mean_delta_ci_percent": [-1.0, 1.5]},
                correctness_valid=True,
                astrakv_quality_not_worse=True,
                lru_quality_not_worse=True,
            ),
            "tie",
        )

    def test_smoke_cannot_emit_a_formal_winner(self) -> None:
        self.assertEqual(
            classify_winner(
                {"paired_count": 5, "mean_delta_ci_percent": [-8.0, -3.0]},
                correctness_valid=True,
                astrakv_quality_not_worse=True,
                lru_quality_not_worse=False,
                formal_design_ready=False,
            ),
            "inconclusive",
        )


if __name__ == "__main__":
    unittest.main()
