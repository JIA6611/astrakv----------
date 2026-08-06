import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "benchmark" / "evaluate_qasper_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_qasper_quality", MODULE)
QUALITY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(QUALITY)


class QasperQualityTests(unittest.TestCase):
    def test_exact_match_and_token_f1(self) -> None:
        exact = QUALITY.score({"status": "ok", "output_text": "The Answer!", "ground_truth": "the answer", "sample_id": "a"})
        self.assertTrue(exact["exact_match"])
        self.assertEqual(exact["token_f1"], 1.0)
        partial = QUALITY.score({"status": "ok", "output_text": "blue green", "ground_truth": "blue red", "sample_id": "b"})
        self.assertAlmostEqual(partial["token_f1"], 0.5)
        failed = QUALITY.score({"status": "error", "output_text": "answer", "ground_truth": "answer"})
        self.assertFalse(failed["exact_match"])
        self.assertEqual(failed["token_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
