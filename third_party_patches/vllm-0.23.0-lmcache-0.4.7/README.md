# AstraKV-W connector patch contract

This directory is an auditable deployment contract, not a replacement for
vLLM or LMCache source.  The active runtime is unavailable until a deployment
manifest generated on the DGX records the exact installed source files and
hashes.

The patch must use the installed `vllm==0.23.0` and `lmcache==0.4.7` source
trees.  It must add only these callback points, all attached to the native
request lifecycle:

1. exact-token scheduler lookup;
2. scheduler external-token/block admission;
3. request metadata construction;
4. request-owned native load start, used only to attribute CPU-hot prefetch;
5. post-`start_load_kv` completion;
6. scheduler compute progress;
7. native request finish.

The patch must never call `engine.retrieve`, construct slot mappings, or write
paged GPU KV outside native `start_load_kv`.  SSD-to-CPU prefetch must use a
real `LocalCPUBackend`, have a `PrefetchTicket`, and must not construct a GPU
load receipt.

Before `ASTRAKV_KV_CORE_MODE=active` is used, copy `manifest.json` to an
immutable deployment location and replace the marker and `source_files` with
absolute paths plus SHA-256 values from the installed sources.  The marker
must be written by the applied vendor patch and the service must emit a
callback smoke record that contains all seven lifecycle callbacks.  Run
`scripts/runtime/verify_kv_core_connector_patch.py` with both artifacts; a
failure means shadow/off only.

Deployment order is intentionally two-stage: run
`scripts/runtime/prepare_kv_core_v2_deployment.sh`, then exercise a repeated
exact-prefix request in `E1` shadow mode to produce all seven callback events.
Use that shadow smoke artifact with `verify_kv_core_connector_patch.py` before
starting E2-E4 active mode. A no-reuse request is not a valid full callback
smoke because it correctly has no native load start/completion.
