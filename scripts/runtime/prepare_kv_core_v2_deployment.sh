#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ASTRAKV_PYTHON:-python3}"
DEPLOYMENT_DIR="${1:-$ROOT/deployments/kv-core-v3-$(date -u +%Y%m%dT%H%M%SZ)}"
PATCH_FILE="$ROOT/third_party_patches/vllm-0.23.0-lmcache-0.4.7/lmcache-vllm_v1_adapter.patch"
PATCH_ID="astrakv-kv-core-vllm-0.23.0-lmcache-0.4.7-v3"

readarray -t paths < <("$PYTHON" - <<'PY'
from importlib import metadata, util
for package, expected in (("vllm", "0.23.0"), ("lmcache", "0.4.7")):
    actual = metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package} version mismatch: expected {expected}, found {actual}")
adapter = util.find_spec("lmcache.integration.vllm.vllm_v1_adapter")
connector = util.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector")
if adapter is None or connector is None or not adapter.origin or not connector.origin:
    raise SystemExit("unable to locate version-locked connector sources")
print(adapter.origin)
print(connector.origin)
PY
)

ADAPTER="${paths[0]}"
VLLM_CONNECTOR="${paths[1]}"
SITE_ROOT="$(cd "$(dirname "$ADAPTER")/../../.." && pwd)"
mkdir -p "$DEPLOYMENT_DIR"
cp -a "$ADAPTER" "$DEPLOYMENT_DIR/vllm_v1_adapter.before-v2.py"

if grep -q 'astrakv_allocated_external_tokens' "$ADAPTER"; then
  # An existing AstraKV marker is not proof that it is this exact contract.
  # The reverse dry-run proves the installed source is precisely the result
  # of applying this self-contained v3 patch to clean LMCache 0.4.7.
  if ! (cd "$SITE_ROOT" && patch --batch --dry-run --reverse -p1 < "$PATCH_FILE"); then
    echo "installed adapter is not an exact v3 patch result; refusing to certify it" >&2
    echo "Preserve the current source as evidence, then reinstall clean lmcache==0.4.7" >&2
    echo "or restore a verified clean adapter before rerunning this script." >&2
    exit 2
  fi
else
  (cd "$SITE_ROOT" && patch --dry-run -p1 < "$PATCH_FILE")
  (cd "$SITE_ROOT" && patch --forward --backup --suffix=.astrakv-v1.bak -p1 < "$PATCH_FILE")
fi
for marker in 'from astrakv.runtime.vendor_callback_bridge import VendorCallbackBridge' 'import time' 'VendorCallbackBridge.from_environment(self)' 'connector_metadata(' 'native_load_start(' 'load_shortfall_unsafe' 'scheduler_compute_progress' 'request_finished(' 'astrakv_allocated_external_tokens'; do
  grep -Fq "$marker" "$ADAPTER" || {
    echo "partial or legacy KV-Core patch detected; missing marker: $marker" >&2
    echo "Refusing to layer v2 onto modified vendor source. Restore the managed" >&2
    echo "${ADAPTER}.astrakv-v1.bak after verifying it is clean LMCache 0.4.7," >&2
    echo "or reinstall lmcache==0.4.7, then rerun this deployment script." >&2
    exit 2
  }
done

"$PYTHON" -m py_compile "$ADAPTER" "$VLLM_CONNECTOR"
printf '%s\n' "$PATCH_ID" > "$DEPLOYMENT_DIR/PATCH_MARKER"

ADAPTER_SHA="$(sha256sum "$ADAPTER" | awk '{print $1}')"
CONNECTOR_SHA="$(sha256sum "$VLLM_CONNECTOR" | awk '{print $1}')"
PATCH_SHA="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
export ASTRAKV_DEPLOYMENT_DIR="$DEPLOYMENT_DIR"
export ASTRAKV_ADAPTER_PATH="$ADAPTER"
export ASTRAKV_ADAPTER_SHA="$ADAPTER_SHA"
export ASTRAKV_VLLM_CONNECTOR_PATH="$VLLM_CONNECTOR"
export ASTRAKV_VLLM_CONNECTOR_SHA="$CONNECTOR_SHA"
export ASTRAKV_PATCH_FILE="$PATCH_FILE"
export ASTRAKV_PATCH_SHA="$PATCH_SHA"
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

deployment = Path(os.environ["ASTRAKV_DEPLOYMENT_DIR"]).resolve()
patch_id = "astrakv-kv-core-vllm-0.23.0-lmcache-0.4.7-v3"
payload = {
    "schema": "astrakv-kv-core-connector-patch-v2",
    "patch_id": patch_id,
    "vllm_version": "0.23.0",
    "lmcache_version": "0.4.7",
    "callbacks": [
        "scheduler_exact_lookup",
        "scheduler_external_admission",
        "connector_metadata",
        "native_load_start",
        "native_load_completion",
        "scheduler_compute_progress",
        "request_finished",
    ],
    "patch_marker": {"id": patch_id, "path": str(deployment / "PATCH_MARKER")},
    "patch_file": {
        "path": os.environ["ASTRAKV_PATCH_FILE"],
        "sha256": os.environ["ASTRAKV_PATCH_SHA"],
    },
    "source_files": [
        {"path": os.environ["ASTRAKV_ADAPTER_PATH"], "sha256": os.environ["ASTRAKV_ADAPTER_SHA"]},
        {"path": os.environ["ASTRAKV_VLLM_CONNECTOR_PATH"], "sha256": os.environ["ASTRAKV_VLLM_CONNECTOR_SHA"]},
    ],
}
(deployment / "deployment.manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
)
print(deployment / "deployment.manifest.json")
PY

echo "KV-Core v2 deployment prepared at: $DEPLOYMENT_DIR"
