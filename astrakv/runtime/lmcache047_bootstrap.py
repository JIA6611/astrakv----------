"""Opt-in startup bootstrap for the version-locked LMCache runtime patch."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from astrakv.runtime.runtime_control_host import RuntimeControlHost, RuntimeControlHostConfig

_LOCK = threading.Lock()
_INSTALLED = False
_HOST: RuntimeControlHost | None = None


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


def install_from_environment(
    *,
    installer: Callable[..., Any] | None = None,
) -> bool:
    global _HOST, _INSTALLED
    if os.environ.get("ASTRAKV_ENABLE_LMCACHE047_HOOKS", "false") != "true":
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

    if installer is None:
        from astrakv.runtime.lmcache047_runtime_patch import install_lmcache047_hooks
        installer = install_lmcache047_hooks
    run_id = os.environ.get("ASTRAKV_RUNTIME_CONTROL_RUN_ID", "")
    if run_id and os.environ.get("ASTRAKV_RUNTIME_CONTROL_PROCESS_SCOPE", "") == "engine_child":
        if not _is_vllm_engine_child():
            return False
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
            ))
        except (KeyError, ValueError) as exc:
            raise RuntimeError("invalid ASTRAKV_RUNTIME_CONTROL_* configuration") from exc
        try:
            host.start()
            host.install_hooks(installer)
        except Exception:
            host.close()
            raise
        _HOST = host
    else:
        installer(sink)
    _INSTALLED = True
    print("AstraKV LMCache 0.4.7 runtime hooks installed", flush=True)
    return True
