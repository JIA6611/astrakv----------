"""Build the final target-1 text/KV consistency report from suite artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.benchmarks.text_kv_consistency import build_suite_report, write_suite_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "report"
    report = build_suite_report(suite_dir)
    written = write_suite_report(output_dir, report)
    print(f"JSON report written to {written['json']}")
    print(f"Markdown report written to {written['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
