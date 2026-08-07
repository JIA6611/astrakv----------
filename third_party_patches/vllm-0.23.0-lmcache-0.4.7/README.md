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
4. post-`start_load_kv` completion.

The patch must never call `engine.retrieve`, construct slot mappings, or write
paged GPU KV outside native `start_load_kv`.  SSD-to-CPU prefetch must use a
real `LocalCPUBackend`, have a `PrefetchTicket`, and must not construct a GPU
load receipt.

Before `ASTRAKV_KV_CORE_MODE=active` is used, copy `manifest.json` to an
immutable deployment location and replace the marker and `source_files` with
absolute paths plus SHA-256 values from the installed sources.  The marker
must be written by the applied vendor patch and the service must emit a
callback smoke record that contains all four lifecycle callbacks.  Run
`scripts/runtime/verify_kv_core_connector_patch.py` with both artifacts; a
failure means shadow/off only.
