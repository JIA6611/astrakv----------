"""Opt-in startup bootstrap for the version-locked LMCache runtime patch."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from astrakv.runtime.runtime_control_host import RuntimeControlHost, RuntimeControlHostConfig
from astrakv.runtime.kv_core_connector import KVCoreConnectorCallbacks
from astrakv.runtime.kv_runtime_core import RuntimeMode, TierCapabilitySnapshot, TierTopology
from astrakv.runtime.third_party_patch import PATCH_ID

_LOCK = threading.Lock()
_INSTALLED = False
_HOST: RuntimeControlHost | None = None
_KV_CORE_CALLBACKS: KVCoreConnectorCallbacks | None = None


def _is_vllm_engine_child() -> bool:
    """Identify the vLLM spawn child before it sets its EngineCore title."""
    # Both the EngineCore spawn child and multiprocessing.resource_tracker have
    # the API server as a parent.  Only the former has the spawn-fork marker.
    if "--multiprocessing-fork" not in sys.argv:
        return False
    try:
        parent_cmdline = open(f"/proc/{os.getppid()}/cmdline", "rb").read().decode("utf-8", "replace")
    except OSError:
        return any(argument == "VLLM::EngineCore" for argument in sys.argv)
    return "vllm.entrypoints.openai.api_server" in parent_cmdline


def installed_runtime_control_host() -> RuntimeControlHost | None:
    """Return the process-owned host for operational teardown and inspection."""
    return _HOST


def installed_kv_core_callbacks() -> KVCoreConnectorCallbacks | None:
    """Return callbacks for the vendor-patched connector, never for legacy hooks."""
    return _KV_CORE_CALLBACKS


def _environment_capability() -> TierCapabilitySnapshot:
    topology = TierTopology(os.environ.get("ASTRAKV_KV_CORE_TOPOLOGY", "gpu_ssd"))
    cpu_enabled = os.environ.get("ASTRAKV_KV_CORE_LOCAL_CPU", "false") == "true"
    return TierCapabilitySnapshot(
        topology=topology,
        local_cpu_enabled=cpu_enabled,
        local_disk_enabled=os.environ.get("ASTRAKV_KV_CORE_LOCAL_DISK", "true") == "true",
        cpu_capacity_bytes=int(os.environ.get("ASTRAKV_KV_CORE_CPU_CAPACITY_BYTES", "0")),
        cpu_used_bytes=int(os.environ.get("ASTRAKV_KV_CORE_CPU_USED_BYTES", "0")),
        ssd_capacity_bytes=int(os.environ.get("ASTRAKV_KV_CORE_SSD_CAPACITY_BYTES", "0")),
        ssd_used_bytes=int(os.environ.get("ASTRAKV_KV_CORE_SSD_USED_BYTES", "0")),
        available_kv_blocks=int(os.environ.get("ASTRAKV_KV_CORE_AVAILABLE_KV_BLOCKS", "0")),
        external_token_cap=int(os.environ.get("ASTRAKV_KV_CORE_EXTERNAL_TOKEN_CAP", "0")),
        uma_available_bytes=int(os.environ.get("ASTRAKV_KV_CORE_UMA_AVAILABLE_BYTES", "0")),
        memory_pressure=float(os.environ.get("ASTRAKV_KV_CORE_MEMORY_PRESSURE", "0")),
        queue_depth=int(os.environ.get("ASTRAKV_KV_CORE_QUEUE_DEPTH", "0")),
    )


def _active_patch_verified() -> bool:
    path = os.environ.get("ASTRAKV_KV_CORE_PATCH_VERIFICATION", "")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("compatible") is True and payload.get("patch_id") == PATCH_ID


def install_from_environment(
    *,
    installer: Callable[..., Any] | None = None,
    vendor_engine_child: bool = False,
    start_runtime_host: bool = True,
) -> bool:
    global _HOST, _INSTALLED, _KV_CORE_CALLBACKS
    vendor_patch = os.environ.get("ASTRAKV_KV_CORE_VENDOR_PATCH", "false") == "true"
    if os.environ.get("ASTRAKV_ENABLE_LMCACHE047_HOOKS", "false") != "true" and not vendor_patch:
        return False
    if _INSTALLED:
        return True
    output = Path(os.environ.get("ASTRAKV_LMCACHE047_EVENTS", "results/lmcache047_events.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)

    def sink(event: dict[str, Any]) -> None:
        try:
            with _LOCK, output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Observation must never break a serving request.
            return

    mode = RuntimeMode(os.environ.get("ASTRAKV_KV_CORE_MODE", "off"))
    if mode is RuntimeMode.ACTIVE and not _active_patch_verified():
        raise RuntimeError("KV-Core active mode requires a verified vLLM/LMCache connector patch")
    if installer is None:
        from astrakv.runtime.lmcache047_runtime_patch import install_lmcache047_hooks
        installer = install_lmcache047_hooks
    run_id = os.environ.get("ASTRAKV_RUNTIME_CONTROL_RUN_ID", "")
    if mode is RuntimeMode.ACTIVE and not run_id:
        raise RuntimeError("KV-Core active mode requires a request-context runtime control host")
    if run_id and os.environ.get("ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE", "") == "engine_child":
        if not vendor_engine_child and not _is_vllm_engine_child():
            return False
    if vendor_patch and run_id and not start_runtime_host:
        # A vLLM worker owns the LMCache storage engine but not the scheduler
        # control-plane port.  It still needs process-local callback state for
        # native load receipts, while the EngineCore scheduler remains the
        # sole RuntimeControlHost owner.
        _KV_CORE_CALLBACKS = KVCoreConnectorCallbacks(
            mode=mode, capability=_environment_capability(),
        )
        _INSTALLED = True
        print("AstraKV KV-Core worker callbacks installed", flush=True)
        return True
    if run_id:
        try:
            secret = bytes.fromhex(os.environ["ASTRAKV_RUNTIME_CONTROL_SECRET_HEX"])
            host = RuntimeControlHost(RuntimeControlHostConfig(
                run_id=run_id,
                state_dir=Path(os.environ["ASTRAKV_RUNTIME_CONTROL_STATE_DIR"]),
                secret=secret,
                engine_instance_id=os.environ["ASTRAKV_RUNTIME_CONTROL_ENGINE_ID"],
                worker_id=os.environ["ASTRAKV_RUNTIME_CONTROL_WORKER_ID"],
                context_port=int(os.environ.get("ASTRAKV_RUNTIME_CONTROL_CONTEXT_PORT", "0")),
                session_id=os.environ.get("ASTRAKV_RUNTIME_CONTROL_SESSION_ID", ""),
                online_policy_enabled=os.environ.get("ASTRAKV_ENABLE_ONLINE_POLICY", "false") == "true",
                online_policy_dispatch_on_release=os.environ.get("ASTRAKV_ENABLE_ONLINE_POLICY_RELEASE_DISPATCH", "true") == "true",
                offline_gate_record=(
                    json.loads(open(os.environ["ASTRAKV_ONLINE_OFFLINE_GATE_PATH"], encoding="utf-8").read())
                    if os.environ.get("ASTRAKV_ONLINE_OFFLINE_GATE_PATH") else None
                ),
                prediction_sidecar_path=(
                    Path(os.environ["ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH"])
                    if os.environ.get("ASTRAKV_ONLINE_PREDICTION_SIDECAR_PATH") else None
                ),
                profile_db_path=(
                    Path(os.environ["ASTRAKV_ONLINE_PROFILE_DB_PATH"])
                    if os.environ.get("ASTRAKV_ONLINE_PROFILE_DB_PATH") else None
                ),
                scheduler_hints_path=(
                    Path(os.environ["ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH"])
                    if os.environ.get("ASTRAKV_ONLINE_SCHEDULER_HINTS_PATH") else None
                ),
                online_prefetch_dispatch_enabled=(
                    os.environ.get("ASTRAKV_ENABLE_ONLINE_PREFETCH_DISPATCH", "true") == "true"
                ),
                online_prefetch_mode=os.environ.get("ASTRAKV_ONLINE_PREFETCH_MODE", "disabled"),
                prefetch_dispatch_independent_of_mode=(
                    os.environ.get("ASTRAKV_PREFETCH_DISPATCH_INDEPENDENT_OF_MODE", "false") == "true"
                ),
                online_evict_dispatch_enabled=(
                    os.environ.get("ASTRAKV_ONLINE_EVICT_DISPATCH_ENABLED", "true") == "true"
                ),
                evict_pressure_gate_enabled=(
                    os.environ.get("ASTRAKV_EVICT_PRESSURE_GATE_ENABLED", "true") == "true"
                ),
                evict_pressure_trigger=float(os.environ.get("ASTRAKV_EVICT_PRESSURE_TRIGGER", "0.8")),
                evict_cpu_capacity_bytes=int(os.environ.get("ASTRAKV_EVICT_CPU_CAPACITY_BYTES", "0")),
                evict_ssd_capacity_bytes=int(os.environ.get("ASTRAKV_EVICT_SSD_CAPACITY_BYTES", "0")),
                evict_cold_score_threshold=float(
                    os.environ.get("ASTRAKV_EVICT_COLD_SCORE_THRESHOLD", "0.35")
                ),
                global_evict_scan_enabled=(
                    os.environ.get("ASTRAKV_EVICT_GLOBAL_SCAN_ENABLED", "true") == "true"
                ),
                global_evict_scan_min_interval_s=float(
                    os.environ.get("ASTRAKV_EVICT_GLOBAL_SCAN_MIN_INTERVAL_S", "5.0")
                ),
                global_evict_scan_max_victims=int(
                    os.environ.get("ASTRAKV_EVICT_GLOBAL_SCAN_MAX_VICTIMS", "4")
                ),
                evict_periodic_scan_enabled=(
                    os.environ.get("ASTRAKV_EVICT_PERIODIC_SCAN_ENABLED", "false") == "true"
                ),
                evict_periodic_scan_interval_s=float(
                    os.environ.get("ASTRAKV_EVICT_PERIODIC_SCAN_INTERVAL_S", "1.0")
                ),
                kv_core_mode=mode,
            ))
        except (KeyError, ValueError) as exc:
            raise RuntimeError("invalid ASTRAKV_RUNTIME_CONTROL_* configuration") from exc
        try:
            host.start()
            if vendor_patch:
                # The vendor bridge consumes these callbacks (A: arrival
                # promotion, admission).  The legacy hook layer is ALSO
                # installed so bindings/events reach the online policy worker,
                # which is what drives B (predictive prefetch).
                _KV_CORE_CALLBACKS = KVCoreConnectorCallbacks(mode=mode, capability=_environment_capability())
                host.install_hooks(installer)
            else:
                host.install_hooks(installer)
        except Exception:
            host.close()
            raise
        _HOST = host
    elif mode is not RuntimeMode.ACTIVE:
        installer(sink)
    _INSTALLED = True
    print(
        "AstraKV KV-Core vendor callbacks installed"
        if vendor_patch
        else "AstraKV LMCache 0.4.7 legacy hooks installed",
        flush=True,
    )
    return True
