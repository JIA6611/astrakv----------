from __future__ import annotations

import unittest

from scripts.benchmark.materialize_e11_native_cpu_mvp_source import select_schedule


def row(group: str, order: int, size: int, tokens: int = 1000) -> dict:
    return {
        "reuse_group": group,
        "request_id": f"{group}-{order}",
        "order": order,
        "prompt": f"prompt {group} {order}",
        "metadata": {"context_token_estimate": tokens},
        "shared_context": size > 1,
    }


class E11NativeCPUMVPSourceTest(unittest.TestCase):
    def test_minimal_schedule_has_hot_cold_pressure_then_reverse_revisits(self):
        rows = []
        order = 0
        for group, size, tokens in (
            ("hot-a", 3, 3600),
            ("hot-b", 3, 3500),
            ("cold-long", 1, 2400),
            ("cold-short", 1, 900),
            ("cold-mid", 1, 1200),
        ):
            for _ in range(size):
                rows.append(row(group, order, size, tokens))
                order += 1

        selected = select_schedule(
            rows, hot_groups=2, cold_groups=2, regime="scan_pollution_past_observed",
        )

        self.assertEqual(
            [item["reuse_group"] for item in selected],
            [
                "hot-a", "hot-a", "hot-a", "hot-b", "hot-b", "hot-b",
                "cold-short", "cold-mid", "hot-b",
            ],
        )
        self.assertEqual([item["order"] for item in selected], list(range(9)))
        self.assertEqual(
            [item["metadata"]["e11_temperature"] for item in selected],
            ["hot", "hot", "hot", "hot", "hot", "hot", "cold", "cold", "hot"],
        )
        self.assertEqual(
            [item["metadata"]["e11_phase"] for item in selected],
            ["first", "revisit", "revisit", "first", "revisit", "revisit", "first", "first", "revisit"],
        )

    def test_recency_aligned_keeps_hot_visits_consecutive(self):
        rows = []
        order = 0
        for group, size in (("hot-a", 3), ("hot-b", 3), ("cold-a", 1), ("cold-b", 1)):
            for _ in range(size):
                rows.append(row(group, order, size))
                order += 1

        selected = select_schedule(rows, regime="recency_aligned")

        self.assertEqual(
            [item["reuse_group"] for item in selected],
            ["cold-a", "cold-b", "hot-a", "hot-a", "hot-a", "hot-b", "hot-b", "hot-b"],
        )
        self.assertTrue(all(
            item["metadata"]["e11_profile_quality"] == "past_observed_only"
            for item in selected
        ))

    def test_stale_profile_regime_marks_explicit_hint_overrides(self):
        rows = []
        order = 0
        for group, size in (("hot-a", 3), ("hot-b", 3), ("cold-a", 1), ("cold-b", 1)):
            for _ in range(size):
                rows.append(row(group, order, size))
                order += 1

        selected = select_schedule(rows, regime="profile_shift_or_stale")

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0]["_e11_policy_reuse_ratio"], 0.0)
        self.assertTrue(all("_e11_policy_reuse_ratio" in item for item in selected))
        self.assertTrue(all(
            item["metadata"]["e11_profile_quality"] == "past_observed_phase_shift"
            for item in selected
        ))

    def test_requires_two_hot_groups(self):
        rows = [row("hot-a", index, 3) for index in range(3)] + [row("cold", 3, 1)]
        with self.assertRaises(ValueError):
            select_schedule(rows, hot_groups=2, cold_groups=1)


if __name__ == "__main__":
    unittest.main()
