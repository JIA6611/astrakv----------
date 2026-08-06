import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.evaluation.hidden_state import (
    HiddenStateRecord,
    compare_record_sets,
    hidden_state_from_record,
    linear_cka,
    load_hidden_state_jsonl,
    summarize_comparisons,
)
from scripts.research.evaluate_hidden_state_drift import (
    write_manifest,
    write_records_jsonl,
    write_report,
    write_results_csv,
)


class HiddenStateDriftTests(unittest.TestCase):
    def test_linear_cka_identical_states_is_one(self) -> None:
        matrix = ((1.0, 2.0), (3.0, 4.0), (5.0, 7.0))

        value = linear_cka(matrix, matrix)

        self.assertIsNotNone(value)
        self.assertAlmostEqual(value or 0.0, 1.0)

    def test_record_parser_accepts_vector_and_matrix(self) -> None:
        vector = hidden_state_from_record(
            {"sample_id": "v", "layer_id": 0, "token_index": 1, "hidden_state": [1, 2, 3]},
            source="hidden.jsonl",
            line_number=1,
        )
        matrix = hidden_state_from_record(
            {"sample_id": "m", "layer": 2, "hidden_states": [[1, 2], [3, 4]]},
            source="hidden.jsonl",
            line_number=2,
        )

        self.assertEqual(vector.shape, (1, 3))
        self.assertEqual(vector.layer_id, 0)
        self.assertEqual(vector.token_index, 1)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.layer_id, 2)

    def test_compare_reports_drift_and_shape_mismatch(self) -> None:
        baseline = [
            HiddenStateRecord(sample_id="a", layer_id=1, token_index=None, values=((1.0, 2.0), (3.0, 4.0))),
            HiddenStateRecord(sample_id="bad", layer_id=1, token_index=None, values=((1.0, 2.0),)),
        ]
        variant = [
            HiddenStateRecord(sample_id="a", layer_id=1, token_index=None, values=((1.0, 2.0), (3.0, 5.0))),
            HiddenStateRecord(sample_id="bad", layer_id=1, token_index=None, values=((1.0, 2.0), (3.0, 4.0))),
        ]

        comparisons = compare_record_sets(baseline, variant)
        by_id = {item.sample_id: item for item in comparisons}

        self.assertEqual(by_id["a"].status, "ok")
        self.assertAlmostEqual(by_id["a"].mse or 0.0, 0.25)
        self.assertAlmostEqual(by_id["a"].l2_drift or 0.0, 1.0)
        self.assertEqual(by_id["bad"].status, "shape_mismatch")

    def test_missing_variant_is_reported(self) -> None:
        comparisons = compare_record_sets(
            [HiddenStateRecord(sample_id="only", layer_id=0, token_index=None, values=((1.0, 2.0),))],
            [],
        )

        self.assertEqual(comparisons[0].status, "missing_variant")
        self.assertEqual(comparisons[0].baseline_shape, (1, 2))
        self.assertEqual(comparisons[0].variant_shape, (0, 0))

    def test_jsonl_outputs_report_and_manifest_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            baseline_path = tmp / "baseline_hidden.jsonl"
            variant_path = tmp / "variant_hidden.jsonl"
            baseline_path.write_text(
                json.dumps({"sample_id": "s1", "layer_id": 1, "hidden_state": [[1, 2], [3, 4]]}) + "\n",
                encoding="utf-8",
            )
            variant_path.write_text(
                json.dumps({"sample_id": "s1", "layer_id": 1, "hidden_state": [[1, 2], [3, 4.5]]}) + "\n",
                encoding="utf-8",
            )

            baseline = load_hidden_state_jsonl(baseline_path)
            variant = load_hidden_state_jsonl(variant_path)
            comparisons = compare_record_sets(baseline, variant)
            records_path = tmp / "hidden_state_drift_records.jsonl"
            csv_path = tmp / "hidden_state_drift_results.csv"
            report_path = tmp / "hidden_state_drift_report.md"
            manifest_path = tmp / "hidden_state_drift_manifest.json"
            args = Namespace(baseline_jsonl=str(baseline_path), variant_jsonl=str(variant_path))

            write_records_jsonl(records_path, comparisons)
            write_results_csv(csv_path, comparisons)
            write_report(report_path, args, comparisons, records_path, csv_path)
            write_manifest(manifest_path, args, comparisons, records_path, csv_path, report_path)

            loaded_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded_records[0]["sample_id"], "s1")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "ok")
            self.assertIn("# Hidden-State Drift Evaluation Report", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["ok_count"], 1)
            self.assertFalse(math.isnan(float(rows[0]["cosine_similarity"])))

    def test_summary_aggregates_ok_items_only(self) -> None:
        comparisons = compare_record_sets(
            [HiddenStateRecord(sample_id="a", layer_id=1, token_index=None, values=((1.0, 2.0), (3.0, 4.0)))],
            [HiddenStateRecord(sample_id="a", layer_id=1, token_index=None, values=((1.0, 2.0), (3.0, 4.0)))],
        )

        summary = summarize_comparisons(comparisons)

        self.assertEqual(summary.comparison_count, 1)
        self.assertEqual(summary.ok_count, 1)
        self.assertAlmostEqual(summary.mean_cosine_similarity or 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
