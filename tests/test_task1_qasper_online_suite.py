import csv
import json
import tempfile
import unittest
from pathlib import Path


class Task1QasperOnlineSuiteEntrypointTests(unittest.TestCase):
    def test_entrypoint_runs_all_conditions_with_runtime_context_and_quality(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "entrypoints" / "run_task1_qasper_online_control_suite.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        for required in (
            "random", "grouped", "baseline", "variant",
            "--request-context-url", "--runtime-state-dir",
            "evaluate_qasper_quality.py", "compare_real_runs.py",
            "summarize_task1_qasper_online_suite.py", "git-common-dir",
        ):
            self.assertIn(required, content)

    def test_lmcache_launcher_uses_explicit_json_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts" / "launch" / "launch_lmcache_vllm.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ -z "${ASTRAKV_KV_TRANSFER_CONFIG:-}" ]]', launcher)
        self.assertIn('ASTRAKV_KV_TRANSFER_CONFIG=\'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}\'', launcher)
        self.assertIn("exec bash scripts/launch/launch_vllm_server.sh", launcher)

    def test_suite_defaults_to_conservative_dgx_memory_utilization(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "entrypoints" / "run_task1_qasper_online_control_suite.sh").read_text(encoding="utf-8")
        self.assertIn('GPU_MEMORY_UTILIZATION="0.72"', script)
        self.assertIn('default 0.72', script)

    def test_suite_declares_unquantized_model_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "entrypoints" / "run_task1_qasper_online_control_suite.sh").read_text(encoding="utf-8")
        self.assertIn('--quantization "unquantized"', script)

    def test_repeat_summary_requires_zero_margin_quality_noninferiority(self) -> None:
        from scripts.reporting.summarize_task1_qasper_online_repeats import aggregate_repeat_summaries

        good = repeat_summary(0.2, 0.4, 3)
        result = aggregate_repeat_summaries([good, good])
        self.assertTrue(result["workloads"]["random"]["quality_noninferior_zero_margin"])
        degraded = repeat_summary(0.2, 0.19, 3)
        self.assertFalse(aggregate_repeat_summaries([good, degraded])["workloads"]["random"]["quality_noninferior_zero_margin"])

    def test_summary_requires_receipt_for_enabled_run(self) -> None:
        from scripts.reporting.summarize_task1_qasper_online_suite import summarize_suite

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for workload in ("random", "grouped"):
                for role in ("baseline", "variant"):
                    run = root / workload / role
                    run.mkdir(parents=True)
                    write_request_results(run / "request_results.jsonl")
                    write_quality(run / "qasper_quality_summary.csv")
                    (run / "experiment_manifest.json").write_text(json.dumps({"run_id": f"{workload}-{role}"}), encoding="utf-8")
                    state = root / workload / f"{role}-state"
                    state.mkdir()
                    (state / "backend_capabilities.json").write_text("{}", encoding="utf-8")
                    (state / "backend_binding_events.jsonl").write_text("{}\n", encoding="utf-8")
                    (state / "runtime_events_raw.jsonl").write_text("{}\n", encoding="utf-8")
                    (state / "runtime_command_receipts.jsonl").write_text("", encoding="utf-8")
                write_comparison(root / workload / "comparison")
            with self.assertRaisesRegex(ValueError, "missing completed receipt"):
                summarize_suite(root)

    def test_summary_requires_eligible_paired_comparison_outputs(self) -> None:
        from scripts.reporting.summarize_task1_qasper_online_suite import summarize_suite

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for workload in ("random", "grouped"):
                for role in ("baseline", "variant"):
                    run = root / workload / role
                    run.mkdir(parents=True)
                    write_request_results(run / "request_results.jsonl")
                    write_quality(run / "qasper_quality_summary.csv")
                    (run / "experiment_manifest.json").write_text(json.dumps({"run_id": f"{workload}-{role}"}), encoding="utf-8")
                    state = root / workload / f"{role}-state"
                    state.mkdir()
                    (state / "backend_capabilities.json").write_text("{}", encoding="utf-8")
                    (state / "backend_binding_events.jsonl").write_text("{}\n", encoding="utf-8")
                    (state / "runtime_events_raw.jsonl").write_text("{}\n", encoding="utf-8")
                    receipts = '' if role == 'baseline' else json.dumps({"status": "completed", "metadata": {"removed": 1}}) + "\n"
                    (state / "runtime_command_receipts.jsonl").write_text(receipts, encoding="utf-8")
                comparison = root / workload / "comparison"
                comparison.mkdir(parents=True)
                (comparison / "compare_exit_status.txt").write_text("0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing paired_run_manifest.json"):
                summarize_suite(root)

    def test_summary_includes_paired_comparison_status(self) -> None:
        from scripts.reporting.summarize_task1_qasper_online_suite import summarize_suite

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for workload in ("random", "grouped"):
                for role in ("baseline", "variant"):
                    run = root / workload / role
                    run.mkdir(parents=True)
                    write_request_results(run / "request_results.jsonl")
                    write_quality(run / "qasper_quality_summary.csv")
                    (run / "experiment_manifest.json").write_text(json.dumps({"run_id": f"{workload}-{role}"}), encoding="utf-8")
                    state = root / workload / f"{role}-state"
                    state.mkdir()
                    (state / "backend_capabilities.json").write_text("{}", encoding="utf-8")
                    (state / "backend_binding_events.jsonl").write_text("{}\n", encoding="utf-8")
                    (state / "runtime_events_raw.jsonl").write_text("{}\n", encoding="utf-8")
                    receipts = '' if role == 'baseline' else json.dumps({"status": "completed", "metadata": {"removed": 1}}) + "\n"
                    (state / "runtime_command_receipts.jsonl").write_text(receipts, encoding="utf-8")
                write_comparison(root / workload / "comparison")
            summary = summarize_suite(root)
            self.assertTrue(summary["workloads"]["random"]["comparison"]["paired_claim_eligible"])
            self.assertEqual(summary["workloads"]["grouped"]["comparison"]["claim_scope"], "online_control")


def write_request_results(path: Path) -> None:
    path.write_text(json.dumps({"request_id": "request-1", "sample_id": "sample-1", "status": "ok"}) + "\n", encoding="utf-8")


def write_quality(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "exact_match", "value": "0.0"})


def write_comparison(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "compare_exit_status.txt").write_text("0\n", encoding="utf-8")
    (path / "paired_run_manifest.json").write_text(json.dumps({"eligible": True, "claim_scope": "online_control"}), encoding="utf-8")
    (path / "comparison_results.csv").write_text("case\nsample-1\n", encoding="utf-8")
    (path / "comparison_report.md").write_text("# Comparison\n", encoding="utf-8")


def repeat_summary(baseline: float, variant: float, drops: int) -> dict[str, object]:
    role = lambda quality: {"quality": {"exact_match": str(quality), "token_f1": str(quality)}, "completed_drop_count": drops}
    return {"workloads": {workload: {"baseline": role(baseline), "variant": role(variant)} for workload in ("random", "grouped")}}


if __name__ == "__main__":
    unittest.main()
