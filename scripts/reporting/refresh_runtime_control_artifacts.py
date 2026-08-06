"""Refresh online-control artifacts after a runtime action without rerunning a benchmark."""

from __future__ import annotations

import argparse
import json

from astrakv.benchmarks.runtime_artifacts import refresh_online_control_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = refresh_online_control_manifest(args.runtime_state_dir, args.output_dir)
    print(json.dumps({"run_id": manifest.get("run_id"), "artifact_paths": manifest.get("artifact_paths")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
