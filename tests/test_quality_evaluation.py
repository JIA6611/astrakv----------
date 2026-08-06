import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from astrakv.evaluation.quality import (
    compare_output_records,
    normalize_text,
    summarize_quality,
    token_divergence_rate,
)
from scripts.research.evaluate_quality import compare_record_sets, load_jsonl, write_records_jsonl, write_results_csv


class QualityEvaluationTests(unittest.TestCase):
    def test_normalized_match_and_divergence(self) -> None:
        baseline = {"sample_id": "s1", "output": "Hello   World"}
        variant = {"sample_id": "s1", "output": "hello world"}

        comparison = compare_output_records(baseline, variant)

        self.assertFalse(comparison.exact_match)
        self.assertTrue(comparison.normalized_match)
        self.assertEqual(comparison.token_edit_distance, 0)
        self.assertEqual(comparison.token_divergence_rate, 0.0)
        self.assertEqual(normalize_text(" A\nB "), "a b")

    def test_output_text_field_is_supported(self) -> None:
        comparison = compare_output_records(
            {"sample_id": "s1", "output_text": "Generated answer"},
            {"sample_id": "s1", "output_text": "Generated answer"},
        )

        self.assertEqual(comparison.baseline_text, "Generated answer")
        self.assertEqual(comparison.variant_text, "Generated answer")
        self.assertTrue(comparison.exact_match)

    def test_token_divergence_detects_changed_tokens(self) -> None:
        divergence = token_divergence_rate("the quick brown fox", "the quick red fox")

        self.assertEqual(divergence, 0.25)

    def test_ppl_is_computed_from_loss_or_nll(self) -> None:
        loss_comparison = compare_output_records(
            {"sample_id": "loss", "output": "a", "loss": 1.0},
            {"sample_id": "loss", "output": "a", "loss": 1.2},
        )
        nll_comparison = compare_output_records(
            {"sample_id": "nll", "output": "a", "negative_log_likelihood": 4.0, "token_count": 2},
            {"sample_id": "nll", "output": "a", "negative_log_likelihood": 6.0, "token_count": 3},
        )

        self.assertAlmostEqual(loss_comparison.baseline_ppl or 0.0, math.exp(1.0))
        self.assertAlmostEqual(loss_comparison.ppl_delta or 0.0, math.exp(1.2) - math.exp(1.0))
        self.assertAlmostEqual(nll_comparison.baseline_ppl or 0.0, math.exp(2.0))
        self.assertAlmostEqual(nll_comparison.variant_ppl or 0.0, math.exp(2.0))

    def test_summary_rates(self) -> None:
        comparisons = [
            compare_output_records({"sample_id": "a", "output": "same"}, {"sample_id": "a", "output": "same"}),
            compare_output_records({"sample_id": "b", "output": "one two"}, {"sample_id": "b", "output": "one three"}),
        ]

        summary = summarize_quality(comparisons)

        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(summary.ok_count, 2)
        self.assertEqual(summary.exact_match_rate, 0.5)
        self.assertEqual(summary.normalized_match_rate, 0.5)
        self.assertGreater(summary.mean_token_divergence_rate, 0.0)

    def test_compare_record_sets_aligns_by_sample_id(self) -> None:
        baseline = [
            {"sample_id": "a", "output": "alpha"},
            {"sample_id": "b", "output": "beta"},
        ]
        variant = [
            {"sample_id": "b", "output": "beta changed"},
            {"sample_id": "a", "output": "alpha"},
        ]

        comparisons = compare_record_sets(baseline, variant)
        by_id = {item.sample_id: item for item in comparisons}

        self.assertTrue(by_id["a"].exact_match)
        self.assertFalse(by_id["b"].exact_match)

    def test_jsonl_and_csv_outputs_are_writable(self) -> None:
        comparisons = [
            compare_output_records(
                {"sample_id": "a", "output": "alpha", "ppl": 2.0},
                {"sample_id": "a", "output": "alpha", "ppl": 2.1},
            )
        ]
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            records_path = tmp / "quality_records.jsonl"
            csv_path = tmp / "quality_results.csv"
            write_records_jsonl(records_path, comparisons)
            write_results_csv(csv_path, comparisons)

            loaded = load_jsonl(records_path)
            self.assertEqual(loaded[0]["sample_id"], "a")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["exact_match"], "1")
            self.assertIn("metadata", rows[0])

    def test_missing_variant_creates_empty_output_comparison(self) -> None:
        comparisons = compare_record_sets([{"sample_id": "a", "output": "alpha"}], [])

        self.assertEqual(comparisons[0].sample_id, "a")
        self.assertFalse(comparisons[0].exact_match)
        self.assertEqual(comparisons[0].variant_text, "")


if __name__ == "__main__":
    unittest.main()
