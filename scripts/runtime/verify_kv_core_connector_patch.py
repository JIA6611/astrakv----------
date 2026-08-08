#!/usr/bin/env python3
"""Fail closed unless the deployed vLLM/LMCache KV-Core patch is proven."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrakv.runtime.third_party_patch import PATCH_ID, REQUIRED_CALLBACKS, verify_connector_patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--callback-smoke", required=True, help="JSON record emitted by the patched live connector.")
    parser.add_argument("--output", help="Optional verification JSON output path.")
    args = parser.parse_args()
    smoke_path = Path(args.callback_smoke)

    def callback_smoke() -> bool:
        try:
            payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("patch_id") == PATCH_ID and tuple(payload.get("callbacks") or ()) == REQUIRED_CALLBACKS and payload.get("passed") is True

    result = verify_connector_patch(args.deployment_manifest, callback_smoke=callback_smoke)
    encoded = json.dumps(result.record, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
