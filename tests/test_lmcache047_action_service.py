import os
import json
import stat
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_capabilities import RuntimeProbeProof
from astrakv.runtime.backend_hook import BackendActionCommand, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.lmcache047_action_service import (
    ProtectedRuntimeActionService,
    UnixDomainSocketActionServer,
    command_integrity_digest,
)
from astrakv.runtime.lmcache047_runtime_patch import LMCache047ActionEndpoint


class Manager:
    def __init__(self):
        self.removed = []
        self.prefetch_reads = []
        self.load_requests = []
        self.storage_backends = {"LocalCPUBackend": object(), "LocalDiskBackend": object()}
        self.cpu_present = True
        self.disk_present = True
        self.lmcache_engine = self

    def remove(self, key, locations=None):
        self.removed.append((key, locations))
        return 1

    def batched_contains(self, keys, search_range=None, pin=False):
        location = None if not search_range else search_range[0]
        if location == "LocalCPUBackend" and self.cpu_present:
            return 1, {"LocalCPUBackend": list(keys)}
        if location == "LocalDiskBackend" and self.disk_present:
            return 1, {"LocalDiskBackend": list(keys)}
        return 0, {}

    def batched_get(self, keys, location=None):
        self.prefetch_reads.append((list(keys), location))
        if location == "LocalDiskBackend":
            self.cpu_present = True
        class MemoryObj:
            def get_physical_size(self):
                return 2048
        return [MemoryObj() for _ in keys]

    def batched_remove(self, keys, locations=None):
        if locations == ["LocalCPUBackend"]:
            self.cpu_present = False
        if locations == ["LocalDiskBackend"]:
            self.disk_present = False
        self.removed.append((list(keys), locations))
        return len(keys)

    def retrieve(
        self, tokens, token_mask, *, kvcaches, slot_mapping, vllm_cached_tokens, request_configs, req_id,
    ):
        self.load_requests.append({
            "tokens": list(tokens),
            "slot_mapping": list(slot_mapping),
            "vllm_cached_tokens": vllm_cached_tokens,
            "request_configs": dict(request_configs),
            "req_id": req_id,
        })
        self.cpu_present = True
        return [False, False, True, True]


def released_endpoint(*, enabled=True):
    registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
    context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
    submitted = registry.observe("physical-key", HookAction.CACHE_STORE, "submitted", context)
    binding = registry.complete_operation(
        "physical-key", HookAction.CACHE_STORE, "completed", context,
        submitted.event.metadata["operation_lease"],
    ).binding
    registry.observe("physical-key", HookAction.RELEASE, "completed", context)
    manager = Manager()
    endpoint = LMCache047ActionEndpoint(registry, action_registration_enabled=enabled)
    endpoint.register_binding(binding, "physical-key", manager)
    return registry, binding, manager, endpoint


def command(binding, lease, *, command_id="command-1", deadline_ns=None):
    metadata = {"reservation_lease": lease}
    if deadline_ns is not None:
        metadata["deadline_ns"] = deadline_ns
    value = BackendActionCommand(
        run_id="run", command_id=command_id, decision_id="decision", request_id="request",
        object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
        backend_object_id=binding.backend_object_id, action=HookAction.DROP, issued_at_ns=1,
        metadata=metadata, binding_generation=binding.binding_generation,
    )
    return replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})


class ProtectedRuntimeActionServiceTests(unittest.TestCase):
    def service(self, endpoint, directory):
        return ProtectedRuntimeActionService(
            action_endpoint=endpoint, state_dir=directory, secret=b"x" * 32,
            source="lmcache047", method="action-service", session_id="session", now_ns=lambda: 100,
        )

    def proof(self, service, value):
        challenge = service.new_challenge_for(value)
        return challenge, service.issue_runtime_proof(challenge)

    def test_persists_terminal_receipt_and_replays_it_after_restart(self):
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = self.service(endpoint, directory)
            value = command(binding, lease, deadline_ns=101)
            challenge, proof = self.proof(first, value)
            receipt = first.submit(value, challenge, proof)
            restarted = self.service(endpoint, directory)
            replay = restarted.submit(value, challenge, proof)
            self.assertEqual(receipt.status, "completed")
            self.assertEqual(replay, receipt)
            self.assertEqual(len(manager.removed), 1)
            self.assertTrue((Path(directory) / "commands.jsonl").read_text(encoding="utf-8").strip())
            self.assertTrue((Path(directory) / "receipts.jsonl").read_text(encoding="utf-8").strip())

    def test_restart_closes_interrupted_command_with_a_terminal_receipt(self):
        _, binding, manager, endpoint = released_endpoint()
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            value = command(binding, "unconsumed", deadline_ns=101)
            record = {"schema": "astrakv-runtime-action-ledger-v1", "record_type": "command",
                      "command_sha256": command_integrity_digest(value), "command": value.to_record()}
            (Path(directory) / "commands.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            restarted = self.service(endpoint, directory)
            challenge, proof = self.proof(restarted, value)
            receipt = restarted.submit(value, challenge, proof)
            self.assertEqual(receipt.status, "interrupted_before_terminal_receipt")
            self.assertIn("interrupted_before_terminal_receipt", (Path(directory) / "receipts.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(manager.removed, [])

    def test_rejects_expired_command_without_consuming_lease(self):
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            value = command(binding, lease, deadline_ns=99)
            challenge, proof = self.proof(service, value)
            self.assertEqual(service.submit(value, challenge, proof).status, "expired")
        self.assertEqual(manager.removed, [])
        self.assertEqual(registry.reservation_state(lease), "reserved")

    def test_rejects_invalid_proof_without_invoking_action(self):
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            value = command(binding, lease, deadline_ns=101)
            challenge, proof = self.proof(service, value)
            invalid = RuntimeProbeProof(proof.nonce, proof.source, proof.method, proof.session_id, "bad")
            self.assertEqual(service.submit(value, challenge, invalid).status, "invalid_proof")
        self.assertEqual(manager.removed, [])

    def test_reports_lease_mismatch_as_terminal_receipt(self):
        registry, binding, manager, endpoint = released_endpoint()
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            value = command(binding, "wrong-lease", deadline_ns=101)
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "reservation_not_available")
        self.assertEqual(manager.removed, [])

    def test_default_endpoint_is_explicitly_observational_only(self):
        registry, binding, manager, endpoint = released_endpoint(enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            value = command(binding, "unused", deadline_ns=101)
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "observational_only")
        self.assertEqual(manager.removed, [])

    def test_accepts_offload_and_records_tier_and_bytes_in_terminal_receipt(self):
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            metadata = {"reservation_lease": lease, "deadline_ns": 101}
            value = BackendActionCommand(
                run_id="run", command_id="command-offload", decision_id="decision", request_id="request",
                object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
                backend_object_id=binding.backend_object_id, action=HookAction.OFFLOAD, issued_at_ns=1,
                target_tier="ssd", metadata=metadata, binding_generation=binding.binding_generation,
            )
            value = replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.tier_before, "cpu")
        self.assertEqual(receipt.tier_after, "ssd")
        self.assertEqual(receipt.bytes, 2048)
        self.assertEqual(receipt.metadata["offloaded"], 1)
        self.assertEqual(receipt.metadata["source_location"], "LocalCPUBackend")
        self.assertEqual(receipt.metadata["target_location"], "LocalDiskBackend")
        self.assertEqual(manager.removed, [(["physical-key"], ["LocalCPUBackend"])])

    def test_accepts_prefetch_and_records_tier_and_bytes_in_terminal_receipt(self):
        registry, binding, manager, endpoint = released_endpoint()
        manager.cpu_present = False
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            metadata = {"reservation_lease": lease, "deadline_ns": 101}
            value = BackendActionCommand(
                run_id="run", command_id="command-prefetch", decision_id="decision", request_id="request",
                object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
                backend_object_id=binding.backend_object_id, action=HookAction.PREFETCH, issued_at_ns=1,
                target_tier="cpu", metadata=metadata, binding_generation=binding.binding_generation,
            )
            value = replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.tier_before, "ssd")
        self.assertEqual(receipt.tier_after, "cpu")
        self.assertEqual(receipt.bytes, 2048)
        self.assertEqual(receipt.metadata["prefetched"], 1)
        self.assertEqual(receipt.metadata["source_location"], "LocalDiskBackend")
        self.assertEqual(receipt.metadata["target_location"], "LocalCPUBackend")
        self.assertEqual(manager.prefetch_reads, [(["physical-key"], "LocalDiskBackend")])
        self.assertEqual(manager.removed, [])

    def test_accepts_load_and_records_runtime_target_metadata_in_terminal_receipt(self):
        class FakeTensor:
            def element_size(self):
                return 256
            def numel(self):
                return 16

        registry, binding, manager, endpoint = released_endpoint()
        endpoint.register_load_target(
            target_id="load-target-1",
            runtime_reqmeta_id="reqmeta-1",
            token_ids=[101, 102, 103, 104],
            slot_mapping=[0, 1, 2, 3],
            vllm_cached_tokens=2,
            lmcache_cached_tokens=4,
            request_configs={"case": "load"},
            kvcaches=[FakeTensor()],
        )
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            metadata = {"reservation_lease": lease, "deadline_ns": 101, "load_target_id": "load-target-1"}
            value = BackendActionCommand(
                run_id="run", command_id="command-load", decision_id="decision", request_id="request",
                object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
                backend_object_id=binding.backend_object_id, action=HookAction.CACHE_LOAD, issued_at_ns=1,
                target_tier="gpu", metadata=metadata, binding_generation=binding.binding_generation,
            )
            value = replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.tier_before, "ssd")
        self.assertEqual(receipt.tier_after, "gpu")
        self.assertEqual(receipt.bytes, 2048)
        self.assertEqual(receipt.metadata["loaded"], 2)
        self.assertEqual(receipt.metadata["load_target_id"], "load-target-1")
        self.assertEqual(receipt.metadata["runtime_reqmeta_id"], "reqmeta-1")
        self.assertEqual(receipt.metadata["source_location"], "LocalDiskBackend")
        self.assertEqual(receipt.metadata["target_location"], "vllm_paged_kv")
        self.assertEqual(manager.load_requests, [{
            "tokens": [101, 102, 103, 104],
            "slot_mapping": [0, 1, 2, 3],
            "vllm_cached_tokens": 2,
            "request_configs": {"case": "load"},
            "req_id": "reqmeta-1",
        }])

    def test_accepts_native_request_load_without_direct_retrieve(self):
        class FakeTensor:
            def element_size(self):
                return 256
            def numel(self):
                return 16

        registry, binding, manager, endpoint = released_endpoint()
        endpoint.register_load_target(
            target_id="load-target-native",
            runtime_reqmeta_id="reqmeta-native",
            token_ids=[201, 202, 203, 204],
            slot_mapping=[0, 1, 2, 3],
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            request_configs={"case": "hot_load"},
            kvcaches=[FakeTensor()],
            native_request_load=True,
        )
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            metadata = {"reservation_lease": lease, "deadline_ns": 101, "load_target_id": "load-target-native"}
            value = BackendActionCommand(
                run_id="run", command_id="command-load-native", decision_id="decision", request_id="request",
                object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
                backend_object_id=binding.backend_object_id, action=HookAction.CACHE_LOAD, issued_at_ns=1,
                target_tier="gpu", metadata=metadata, binding_generation=binding.binding_generation,
            )
            value = replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.tier_before, "ssd")
        self.assertEqual(receipt.tier_after, "gpu")
        self.assertEqual(receipt.bytes, 4096)
        self.assertEqual(receipt.metadata["loaded"], 4)
        self.assertTrue(receipt.metadata["native_request_load"])
        self.assertEqual(receipt.metadata["load_target_id"], "load-target-native")
        self.assertEqual(receipt.metadata["runtime_reqmeta_id"], "reqmeta-native")
        self.assertEqual(manager.load_requests, [])

    def test_accepts_evict_and_records_distinct_disk_only_terminal_receipt(self):
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id, binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id, request_id="request", object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(endpoint, directory)
            metadata = {"reservation_lease": lease, "deadline_ns": 101}
            value = BackendActionCommand(
                run_id="run", command_id="command-evict", decision_id="decision", request_id="request",
                object_key="prefix", object_level=ObjectLevel.PREFIX, binding_id=binding.binding_id,
                backend_object_id=binding.backend_object_id, action=HookAction.EVICT, issued_at_ns=1,
                target_tier="ssd", metadata=metadata, binding_generation=binding.binding_generation,
            )
            value = replace(value, metadata={**metadata, "command_sha256": command_integrity_digest(value)})
            challenge, proof = self.proof(service, value)
            receipt = service.submit(value, challenge, proof)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.tier_before, "ssd")
        self.assertEqual(receipt.tier_after, "cpu")
        self.assertEqual(receipt.bytes, 2048)
        self.assertEqual(receipt.metadata["evicted"], 1)
        self.assertEqual(receipt.metadata["source_location"], "LocalDiskBackend")
        self.assertEqual(receipt.metadata["target_location"], "none")

    @unittest.skipUnless(os.name == "posix", "Unix-domain socket permissions require POSIX")
    def test_unix_socket_is_owner_only(self):
        _, _, _, endpoint = released_endpoint()
        with tempfile.TemporaryDirectory() as directory:
            socket_path = str(Path(directory) / "runtime.sock")
            server = UnixDomainSocketActionServer(self.service(endpoint, directory), socket_path)
            server.start()
            try:
                self.assertEqual(stat.S_IMODE(os.stat(socket_path).st_mode), 0o600)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
