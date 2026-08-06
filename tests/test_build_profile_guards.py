import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from astrakv.runtime.profile_db import ProfileDB
from scripts.policy.build_profile_guards import main


class BuildProfileGuardsTests(unittest.TestCase):
    def test_build_profile_guards_merges_quality_and_hidden_state_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            quality = tmp / "quality_results.csv"
            hidden = tmp / "hidden_state_drift_results.csv"
            output = tmp / "out"

            quality.write_text(
                "sample_id,exact_match,normalized_match,ppl_delta\n"
                "chunk-a,1,1,0.05\n"
                "chunk-b,0,0,0.40\n",
                encoding="utf-8",
            )
            hidden.write_text(
                "sample_id,layer_id,token_index,status,baseline_shape,variant_shape,element_count,cka,cosine_similarity,mse,l2_drift,max_abs_diff,reason,metadata\n"
                "a,4,,ok,1x2,1x2,2,0.99,1.0,0.0,0.01,0.01,,{}\n"
                "b,7,,ok,1x2,1x2,2,0.80,0.9,0.1,0.30,0.20,,{}\n",
                encoding="utf-8",
            )

            argv = [
                "build_profile_guards.py",
                "--workload-id", "w",
                "--quality-csv", str(quality),
                "--hidden-state-drift-csv", str(hidden),
                "--output-dir", str(output),
            ]
            with mock.patch("sys.argv", argv):
                rc = main()

            self.assertEqual(rc, 0)
            db = ProfileDB.load(output / "profile_db_with_guards.json")

            layer4 = db.get_layer_sensitivity(4, workload_id="w")
            layer7 = db.get_layer_sensitivity(7, workload_id="w")
            guard_a = db.get_quality_guard(workload_id="w", chunk_id="chunk-a")
            guard_b = db.get_quality_guard(workload_id="w", chunk_id="chunk-b")

            assert layer4 is not None and layer7 is not None and guard_a is not None and guard_b is not None
            self.assertTrue(layer4.partial_load_allowed)
            self.assertFalse(layer7.partial_load_allowed)
            self.assertTrue(guard_a.recompute_allowed)
            self.assertFalse(guard_b.partial_load_allowed)
            self.assertFalse(guard_b.recompute_allowed)

            summary = json.loads((output / "profile_guard_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["layer_sensitivity_records"], 2)
            self.assertEqual(summary["counts"]["quality_guard_records"], 2)


if __name__ == "__main__":
    unittest.main()
