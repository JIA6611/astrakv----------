"""Write a lossless canonical workload view of task-one QASPER ZIP data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.benchmarks.task1_qasper_adapter import (
    load_task1_qasper_directory,
    load_task1_qasper_workload,
    write_task1_qasper_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task1-zip")
    source.add_argument("--task1-dir")
    parser.add_argument("--task1-workload", required=True, choices=("random", "grouped"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    workload = (
        load_task1_qasper_directory(args.task1_dir, args.task1_workload)
        if args.task1_dir
        else load_task1_qasper_workload(args.task1_zip, args.task1_workload)
    )
    artifacts = write_task1_qasper_artifacts(workload, args.output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
