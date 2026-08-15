from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting.compare_prefetch_sidecar_blocks import compare


class ComparePrefetchSidecarBlocksTests(unittest.TestCase):
    def test_averages_repeated_prompts_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            blocks = []
            values = {
                "forward": ([100.0, 200.0, 300.0], [110.0, 180.0, 240.0]),
                "reverse": ([140.0, 220.0, 320.0], [70.0, 150.0, 210.0]),
            }
            for name, (baseline_values, variant_values) in values.items():
                baseline_path = root / f"{name}-baseline.jsonl"
                variant_path = root / f"{name}-variant.jsonl"
                for path, ttfts in (
                    (baseline_path, baseline_values), (variant_path, variant_values),
                ):
                    path.write_text(
                        "".join(
                            json.dumps({
                                "request_id": f"r{index}",
                                "prefetch_pair_id": f"p{index}",
                                "prefetch_phase": "far",
                                "ttft_ms": value,
                            }) + "\n"
                            for index, value in enumerate(ttfts)
                        ),
                        encoding="utf-8",
                    )
                blocks.append((name, baseline_path, variant_path))
            result = compare(blocks)
            averaged = result["prompt_averaged_across_blocks"]
            self.assertEqual(averaged["paired_count"], 3)
            self.assertEqual(averaged["baseline_p50_ms"], 210.0)
            self.assertEqual(averaged["variant_p50_ms"], 165.0)
            self.assertEqual(
                result["interpretation"]["block_p50_directions"],
                ["improve", "improve"],
            )


if __name__ == "__main__":
    unittest.main()
