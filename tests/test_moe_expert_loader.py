import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from astrakv.moe.expert_loader import (
    ExpertCatalogEntry,
    ExpertLoadAction,
    ExpertLoadPlannerConfig,
    ExpertProfile,
    SelectiveExpertLoaderPlanner,
    load_expert_catalog,
    load_expert_profiles_from_summary,
    summarize_decisions,
    write_decisions_csv,
    write_hints_jsonl,
)
from scripts.research.plan_moe_expert_loading import merge_profiles, write_manifest, write_report


class MoEExpertLoaderTests(unittest.TestCase):
    def test_hot_expert_gets_gpu_and_warm_expert_gets_cpu(self) -> None:
        profiles = [
            ExpertProfile(layer_id=1, expert_id="hot", activation_count=100, token_count=80, avg_score=0.9, hotness_share=0.70),
            ExpertProfile(layer_id=1, expert_id="warm", activation_count=35, token_count=30, avg_score=0.6, hotness_share=0.25),
            ExpertProfile(layer_id=1, expert_id="cold", activation_count=2, token_count=2, avg_score=0.1, hotness_share=0.01),
        ]
        catalog = {
            "layer1:experthot": ExpertCatalogEntry(layer_id=1, expert_id="hot", size_bytes=100, current_tier="cpu"),
            "layer1:expertwarm": ExpertCatalogEntry(layer_id=1, expert_id="warm", size_bytes=100, current_tier="ssd"),
            "layer1:expertcold": ExpertCatalogEntry(layer_id=1, expert_id="cold", size_bytes=100, current_tier="ssd"),
        }
        planner = SelectiveExpertLoaderPlanner(
            ExpertLoadPlannerConfig(gpu_budget_bytes=100, cpu_budget_bytes=100, default_expert_bytes=100, drop_threshold=0.03)
        )

        decisions = planner.plan(profiles, catalog=catalog)
        by_expert = {decision.expert_id: decision for decision in decisions}

        self.assertEqual(by_expert["hot"].action, ExpertLoadAction.LOAD_GPU)
        self.assertEqual(by_expert["hot"].target_tier, "gpu")
        self.assertEqual(by_expert["hot"].gpu_bytes_after, 100)
        self.assertEqual(by_expert["warm"].action, ExpertLoadAction.KEEP_CPU)
        self.assertEqual(by_expert["warm"].target_tier, "cpu")
        self.assertEqual(by_expert["cold"].action, ExpertLoadAction.DROP)

    def test_budget_exhaustion_offloads_cold_experts_to_ssd(self) -> None:
        profiles = [
            ExpertProfile(layer_id=0, expert_id="0", activation_count=100, avg_score=0.9, hotness_share=0.6),
            ExpertProfile(layer_id=0, expert_id="1", activation_count=80, avg_score=0.8, hotness_share=0.4),
        ]
        planner = SelectiveExpertLoaderPlanner(
            ExpertLoadPlannerConfig(
                gpu_budget_bytes=100,
                cpu_budget_bytes=0,
                default_expert_bytes=100,
                warm_threshold=0.05,
                drop_threshold=0.0,
                ssd_enabled=True,
            )
        )

        decisions = planner.plan(profiles)
        by_expert = {decision.expert_id: decision for decision in decisions}

        self.assertEqual(by_expert["0"].target_tier, "gpu")
        self.assertEqual(by_expert["1"].action, ExpertLoadAction.OFFLOAD_SSD)
        self.assertEqual(by_expert["1"].target_tier, "ssd")

    def test_summary_and_catalog_loaders_preserve_zero_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            summary_path = tmp / "moe_expert_summary.csv"
            catalog_path = tmp / "catalog.csv"
            write_rows(
                summary_path,
                ["layer_id", "expert_id", "activation_count", "token_count", "avg_score", "bytes", "latency_ms", "hotness_share"],
                [
                    {
                        "layer_id": 0,
                        "expert_id": 0,
                        "activation_count": 10,
                        "token_count": 8,
                        "avg_score": 0.75,
                        "bytes": 64,
                        "latency_ms": 1.5,
                        "hotness_share": 1.0,
                    }
                ],
            )
            write_rows(
                catalog_path,
                ["layer_id", "expert_id", "size_bytes", "current_tier", "weight_path"],
                [{"layer_id": 0, "expert_id": 0, "size_bytes": 128, "current_tier": "cpu", "weight_path": "layer0_expert0.safetensors"}],
            )

            profiles = load_expert_profiles_from_summary(summary_path)
            catalog = load_expert_catalog(catalog_path)

            self.assertEqual(profiles[0].layer_id, 0)
            self.assertEqual(profiles[0].expert_id, "0")
            self.assertIn("layer0:expert0", catalog)
            self.assertEqual(catalog["layer0:expert0"].size_bytes, 128)

    def test_merge_profiles_recomputes_hotness(self) -> None:
        merged = merge_profiles(
            [
                ExpertProfile(layer_id=2, expert_id="4", activation_count=10, avg_score=0.8, hotness_share=0.5),
                ExpertProfile(layer_id=2, expert_id="4", activation_count=30, avg_score=0.4, hotness_share=0.5),
                ExpertProfile(layer_id=2, expert_id="5", activation_count=10, avg_score=0.1, hotness_share=0.5),
            ]
        )

        by_expert = {profile.expert_id: profile for profile in merged}
        self.assertEqual(by_expert["4"].activation_count, 40)
        self.assertAlmostEqual(by_expert["4"].avg_score, 0.5)
        self.assertAlmostEqual(by_expert["4"].hotness_share, 0.8)
        self.assertAlmostEqual(by_expert["5"].hotness_share, 0.2)

    def test_outputs_are_writable(self) -> None:
        decisions = SelectiveExpertLoaderPlanner(
            ExpertLoadPlannerConfig(gpu_budget_bytes=100, cpu_budget_bytes=100, default_expert_bytes=100)
        ).plan([ExpertProfile(layer_id=1, expert_id="hot", activation_count=20, token_count=10, avg_score=0.9, hotness_share=1.0)])

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            plan_path = tmp / "moe_expert_load_plan.csv"
            hints_path = tmp / "moe_expert_load_hints.jsonl"
            report_path = tmp / "moe_expert_load_report.md"
            manifest_path = tmp / "moe_expert_load_manifest.json"
            args = Namespace(
                expert_summary="summary.csv",
                moe_events="",
                expert_catalog="catalog.csv",
                gpu_budget_bytes=100,
                cpu_budget_bytes=100,
                default_expert_bytes=100,
                hot_threshold=0.45,
                warm_threshold=0.18,
                drop_threshold=0.02,
                disable_ssd=False,
            )

            write_decisions_csv(plan_path, decisions)
            write_hints_jsonl(hints_path, decisions)
            write_report(report_path, args, [], decisions, plan_path, hints_path)
            write_manifest(manifest_path, args, [], decisions, plan_path, hints_path, report_path)

            with plan_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["expert_id"], "hot")
            hints = [json.loads(line) for line in hints_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(hints[0]["metadata"]["object_type"], "moe_expert")
            self.assertIn("# MoE Expert Selective Loading Plan", report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["total_experts"], 1)
            self.assertEqual(summarize_decisions(decisions)["planned_gpu_bytes"], 100)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
