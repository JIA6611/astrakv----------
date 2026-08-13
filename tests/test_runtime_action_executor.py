import unittest

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_hook import BackendActionCommand, HookAction
from astrakv.runtime.eviction import ObjectLevel
from astrakv.runtime.lmcache047_runtime_patch import LMCache047ActionEndpoint
from astrakv.runtime.runtime_action_executor import RuntimeActionExecutor


class _Manager:
    def __init__(self) -> None:
        self.removed = []
        self.prefetch_reads = []
        self.load_requests = []
        self.storage_backends = {
            "LocalCPUBackend": type("HotCPU", (), {"use_hot": True})(),
            "LocalDiskBackend": object(),
        }
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
                return 4096
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
    manager = _Manager()
    endpoint = LMCache047ActionEndpoint(registry, action_registration_enabled=enabled)
    endpoint.register_binding(binding, "physical-key", manager)
    return registry, binding, manager, endpoint


class RuntimeActionExecutorTests(unittest.TestCase):
    def test_drop_executor_preserves_existing_owner_only_drop_path(self) -> None:
        registry, binding, manager, endpoint = released_endpoint()
        lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        command = BackendActionCommand(
            run_id="run",
            command_id="command-1",
            decision_id="decision-1",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.DROP,
            issued_at_ns=1,
            metadata={"reservation_lease": lease},
            binding_generation=binding.binding_generation,
        )

        result = RuntimeActionExecutor(endpoint).execute(command)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(manager.removed, [("physical-key", None)])

    def test_offload_executor_executes_real_owner_only_cpu_to_disk_transition(self) -> None:
        registry, binding, manager, endpoint = released_endpoint(enabled=True)
        lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        command = BackendActionCommand(
            run_id="run",
            command_id="command-1",
            decision_id="decision-1",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.OFFLOAD,
            issued_at_ns=1,
            target_tier="ssd",
            metadata={"reservation_lease": lease},
            binding_generation=binding.binding_generation,
        )

        result = RuntimeActionExecutor(endpoint).execute(command)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tier_before"], "cpu")
        self.assertEqual(result["tier_after"], "ssd")
        self.assertEqual(result["bytes"], 4096)
        self.assertEqual(result["offloaded"], 1)
        self.assertEqual(manager.removed, [(["physical-key"], ["LocalCPUBackend"])])

    def test_load_prefetch_and_evict_executors_execute_real_owner_only_paths(self) -> None:
        class FakeTensor:
            def element_size(self):
                return 256
            def numel(self):
                return 16

        registry, binding, manager, endpoint = released_endpoint(enabled=True)
        manager.cpu_present = False
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
        prefetch_lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        prefetch = BackendActionCommand(
            run_id="run",
            command_id="command-1",
            decision_id="decision-1",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.PREFETCH,
            issued_at_ns=1,
            target_tier="cpu",
            metadata={"reservation_lease": prefetch_lease},
            binding_generation=binding.binding_generation,
        )
        load = BackendActionCommand(
            run_id="run",
            command_id="command-2",
            decision_id="decision-2",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.CACHE_LOAD,
            issued_at_ns=1,
            target_tier="gpu",
            metadata={"load_target_id": "load-target-1"},
            binding_generation=binding.binding_generation,
        )
        evict = BackendActionCommand(
            run_id="run",
            command_id="command-3",
            decision_id="decision-3",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.EVICT,
            issued_at_ns=1,
            target_tier="ssd",
            metadata={},
            binding_generation=binding.binding_generation,
        )

        prefetch_result = RuntimeActionExecutor(endpoint).execute(prefetch)
        load_lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        load = BackendActionCommand(
            run_id=load.run_id,
            command_id=load.command_id,
            decision_id=load.decision_id,
            request_id=load.request_id,
            object_key=load.object_key,
            object_level=load.object_level,
            binding_id=load.binding_id,
            backend_object_id=load.backend_object_id,
            action=load.action,
            issued_at_ns=load.issued_at_ns,
            target_tier=load.target_tier,
            metadata={**load.metadata, "reservation_lease": load_lease},
            binding_generation=load.binding_generation,
        )
        load_result = RuntimeActionExecutor(endpoint).execute(load)
        evict_lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        evict = BackendActionCommand(
            run_id=evict.run_id,
            command_id=evict.command_id,
            decision_id=evict.decision_id,
            request_id=evict.request_id,
            object_key=evict.object_key,
            object_level=evict.object_level,
            binding_id=evict.binding_id,
            backend_object_id=evict.backend_object_id,
            action=evict.action,
            issued_at_ns=evict.issued_at_ns,
            target_tier=evict.target_tier,
            metadata={"reservation_lease": evict_lease},
            binding_generation=evict.binding_generation,
        )
        evict_result = RuntimeActionExecutor(endpoint).execute(evict)

        self.assertEqual(prefetch_result["status"], "completed")
        self.assertEqual(prefetch_result["tier_before"], "ssd")
        self.assertEqual(prefetch_result["tier_after"], "cpu")
        self.assertEqual(prefetch_result["bytes"], 4096)
        self.assertEqual(prefetch_result["prefetched"], 1)
        self.assertEqual(load_result["status"], "completed")
        self.assertEqual(load_result["tier_before"], "ssd")
        self.assertEqual(load_result["tier_after"], "gpu")
        self.assertEqual(load_result["bytes"], 2048)
        self.assertEqual(load_result["loaded"], 2)
        self.assertEqual(load_result["runtime_reqmeta_id"], "reqmeta-1")
        self.assertEqual(evict_result["status"], "completed")
        self.assertEqual(evict_result["tier_before"], "ssd")
        self.assertEqual(evict_result["tier_after"], "cpu")
        self.assertEqual(evict_result["bytes"], 4096)
        self.assertEqual(evict_result["evicted"], 1)
        self.assertEqual(manager.prefetch_reads, [(["physical-key"], "LocalDiskBackend"), (["physical-key"], "LocalDiskBackend")])
        self.assertEqual(manager.load_requests, [{
            "tokens": [101, 102, 103, 104],
            "slot_mapping": [0, 1, 2, 3],
            "vllm_cached_tokens": 2,
            "request_configs": {"case": "load"},
            "req_id": "reqmeta-1",
        }])

    def test_endpoint_execute_action_uses_unified_runtime_executor(self) -> None:
        endpoint = LMCache047ActionEndpoint()
        command = BackendActionCommand(
            run_id="run",
            command_id="command-1",
            decision_id="decision-1",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id="binding-1",
            backend_object_id="object-1",
            action=HookAction.PREFETCH,
            issued_at_ns=1,
        )

        result = endpoint.execute_action(command)

        self.assertEqual(result["status"], "observational_only")


if __name__ == "__main__":
    unittest.main()
