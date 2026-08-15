from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.analyze_profile_transfer_experiment import analyze


class AnalyzeProfileTransferExperimentTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        files = {
            "subset/transfer_subset_manifest.json": {
                "limit": 2,
                "anti_leakage": {
                    "passed": True,
                    "train_test_original_request_id_overlap": [],
                    "train_test_prompt_hash_overlap": [],
                    "shared_prefix_count": 2,
                },
            },
            "hints/prefix_prefetch_hint_report.json": {
                "hint_count": 2,
                "workload_jsonl": "/tmp/run/train-run/qasper/measured.jsonl",
            },
            "profile_acceptance.json": {
                "functional_acceptance": {"passed": True},
                "prefetch_b": {
                    "completed": 2,
                    "consumed_ticket_count": 2,
                    "completed_bytes": 4096,
                },
                "native_load": {
                    "baseline": {"p50_ms": 10},
                    "variant": {"p50_ms": 5},
                    "paired_median_delta_percent": -50,
                    "variant_wins": 2,
                    "paired_count": 2,
                },
            },
            "profile-test/qasper/predicted_far_ttft_summary.json": {
                "baseline_p50_ms": 100,
                "variant_p50_ms": 80,
            },
        }
        for relative, payload in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        profile = root / "profile/profile_db.json"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text("{}", encoding="utf-8")
        for role in ("baseline", "variant"):
            state = root / f"profile-test/qasper/{role}-state"
            state.mkdir(parents=True, exist_ok=True)
            (state / "runtime.env").write_text("ASTRAKV_ONLINE_PREFETCH_MODE=prefix_only\n")
        auth = root / "profile-test/qasper/variant-state/predictive_prefetch_authorizations.jsonl"
        auth.write_text(
            "".join(
                json.dumps({"request_id": f"r{i}", "prefetch_origin": "profile_b"}) + "\n"
                for i in range(2)
            ),
            encoding="utf-8",
        )

    def test_accepts_disjoint_train_only_profile_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            self._fixture(root)
            result = analyze(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["execution"]["variant_origins"], ["profile_b"])

    def test_rejects_test_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            self._fixture(root)
            (root / "profile-test/qasper/variant-state/runtime.env").write_text(
                "ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH=/tmp/test-sidecar.jsonl\n",
                encoding="utf-8",
            )
            result = analyze(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["test_sidecar_absent"])


if __name__ == "__main__":
    unittest.main()
