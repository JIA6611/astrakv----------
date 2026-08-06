"""Evaluate reproducible offline eviction results before runtime execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astrakv.runtime.offline_safety import OfflineSafetyGate, write_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="offline_eviction_manifest.json; provide once per workload")
    parser.add_argument("--output", default="results/offline_safety_gate.json")
    args = parser.parse_args()
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifest]
    gate = OfflineSafetyGate.evaluate(manifests)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_gate(output, gate)
    print(f"Offline safety gate {gate.result.status}: {output}")
    return 0 if gate.result.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
