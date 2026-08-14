"""Unit tests for the final Prefetch-A/B acceptance report builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.reporting.build_prefetch_final_report import build_report


def _e3() -> dict:
    return {
        "acceptance": {
            "a_consumed_ticket_found": True,
            "b_completed_receipt_found": True,
            "b_failures_have_diagnostics": True,
        },
        "roles": [],
    }


def _validation() -> dict:
    return {
        "cells": [
            {
                "dataset": "qasper", "cell": "A0B0",
                "benchmark": {"ttft_p50_ms": 100.0, "ttft_p95_ms": 120.0},
                "prefetch_a": {"tickets_consumed": 0, "tickets_wasted": 0},
                "prefetch_b": {"completed_with_bytes": 0, "receipt_count": 0},
            },
            {
                "dataset": "qasper", "cell": "A1B1",
                "benchmark": {"ttft_p50_ms": 80.0, "ttft_p95_ms": 95.0},
                "prefetch_a": {"tickets_consumed": 8, "tickets_wasted": 2},
                "prefetch_b": {"completed_with_bytes": 5, "receipt_count": 6},
            },
        ],
        "both_cell_conflict_totals": {
            "invalidate_removed_chunk_count": 0,
            "b_noop_when_a_resident": 0,
            "dual_accounting_cells": 0,
        },
    }


def _adaptation() -> dict:
    return {
        "windows": [
            {"window": 0, "arrival_range": [0, 9], "request_count": 10,
             "ttft_p50_ms": 100.0, "ttft_p95_ms": 120.0,
             "prefetch_a": {"decision_count": 1},
             "prefetch_b": {"completed_with_bytes": 1}},
            {"window": 4, "arrival_range": [40, 49], "request_count": 10,
             "ttft_p50_ms": 80.0, "ttft_p95_ms": 90.0,
             "prefetch_a": {"decision_count": 6},
             "prefetch_b": {"completed_with_bytes": 5}},
        ],
    }


class BuildPrefetchFinalReportTests(unittest.TestCase):
    def test_report_contains_all_sections_and_derived_rates(self) -> None:
        report = build_report(
            e3=_e3(),
            ablation=_validation(),
            transfer=_validation(),
            adaptation=_adaptation(),
        )
        self.assertIn("# AstraKV 双预取（A/B）验收报告", report)
        self.assertIn("## 1. 功能层验收（E3）", report)
        self.assertIn("## 2. 四格对比表", report)
        self.assertIn("## 3. B 泛化（Profile-B", report)
        self.assertIn("## 4. 在线学习自适应", report)
        self.assertIn("## 5. A/B 叠加（both 格）", report)
        self.assertIn("80.0", report)
        self.assertIn("80.0%", report)  # A hit rate 8/(8+2)

    def test_missing_inputs_produce_placeholder_sections(self) -> None:
        report = build_report(e3=None, ablation=None, transfer=None, adaptation=None)
        self.assertIn("未提供 E3 acceptance JSON", report)
        self.assertIn("未提供标准 2×2 validation JSON", report)
        self.assertIn("未提供 transfer validation JSON", report)
        self.assertIn("未提供 adaptation JSON", report)


if __name__ == "__main__":
    unittest.main()
