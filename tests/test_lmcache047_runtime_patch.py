import unittest
import tempfile
import threading
import types
from unittest.mock import patch

from astrakv.runtime.backend_binding_registry import BackendBindingRegistry, RequestContext
from astrakv.runtime.backend_hook import BackendActionCommand, HookAction
from astrakv.runtime.lmcache047_runtime_patch import (
    LMCache047ActionEndpoint,
    LMCache047RequestContextConsumer,
    _ConnectorLifecycle,
    _owner_action_batched_get,
    patch_local_disk_remove_race,
    _prefetch_ready,
    install_lmcache047_hooks,
    probe_lmcache047_storage_contract,
)
from astrakv.runtime.request_context import (
    RuntimeRequestContext,
    RuntimeRequestContextAuthority,
    RuntimeRequestContextReceiver,
    RuntimeRequestIdentity,
)
from astrakv.runtime.eviction import ObjectLevel


class LMCache047RuntimePatchTests(unittest.TestCase):
    def test_local_disk_remove_race_tolerates_missing_file(self):
        class Meta:
            def __init__(self, path: str, size: int) -> None:
                self.path = path
                self.size = size

        class FakeBackend:
            def __init__(self) -> None:
                self.disk_lock = threading.RLock()
                self.dict = {}
                self.usage = 0
                self.stats_monitor = types.SimpleNamespace(update_local_storage_usage=lambda usage: None)
                self.cache_policy = types.SimpleNamespace(update_on_force_evict=lambda key: None)
                self.batched_msg_sender = None

            def update_local_storage_usage(self, usage: int) -> None:
                self.usage = usage

            def remove(self, key, force=True):
                raise AssertionError("original remove should be replaced by the race patch")

        backend = FakeBackend()
        missing_path = tempfile.mktemp(suffix=".pt")
        backend.dict["key"] = Meta(missing_path, 10)
        backend.usage = 10
        manager = types.SimpleNamespace(storage_backends={"LocalDiskBackend": backend})

        self.assertEqual(patch_local_disk_remove_race(manager), 1)
        # The backing file does not exist: remove must not raise, must return
        # True, and must release the lock.
        self.assertTrue(backend.remove("key"))
        self.assertEqual(backend.usage, 0)
        self.assertFalse(backend.disk_lock._is_owned())
        # Second remove on the same key returns False without crashing.
        self.assertFalse(backend.remove("key"))

    def test_owner_action_read_bypasses_only_its_own_reservation_guard(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)

        class Storage:
            storage_backends = {"LocalCPUBackend": object(), "LocalDiskBackend": object()}

            def batched_put(self, keys, objects, **kwargs):
                return None

            def get(self, key, location=None):
                return None

            def batched_get(self, keys, location=None):
                return [object() for _ in keys]

            def remove(self, key, locations=None):
                return 0

        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()

        class Factory:
            def get_or_create_lmcache_engine(self):
                return engine

        install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_provider=lambda: None,
        )
        Factory().get_or_create_lmcache_engine()

        submitted = registry.observe("physical-key", HookAction.CACHE_STORE, "submitted", context)
        registry.complete_operation(
            "physical-key",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        )
        registry.observe("physical-key", HookAction.RELEASE, "completed", context)
        lease = registry.reserve_action(
            binding_id=submitted.binding.binding_id,
            binding_generation=submitted.binding.binding_generation,
            backend_object_id=submitted.binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        self.assertIsNotNone(lease)

        self.assertEqual(storage.batched_get(["physical-key"], location="LocalDiskBackend"), [None])
        self.assertIsNotNone(
            _owner_action_batched_get(storage, "physical-key", location="LocalDiskBackend")[0]
        )

    def test_native_request_load_can_be_disabled_for_prefetch_validation(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        endpoint = LMCache047ActionEndpoint(binding_registry=registry)
        with patch.dict("os.environ", {"ASTRAKV_RUNTIME_DISABLE_NATIVE_REQUEST_LOAD": "true"}):
            lifecycle = _ConnectorLifecycle(records.append, registry, None, None, endpoint)
            self.assertIsNone(
                lifecycle.note_load_target(
                    "reqmeta-prefetch",
                    token_ids=[101, 102],
                    slot_mapping=[0, 1],
                    lmcache_cached_tokens=2,
                )
            )
        self.assertEqual(records, [])

    def test_unpin_waits_for_local_disk_callback_before_releasing_store_lease(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        endpoint = LMCache047ActionEndpoint(binding_registry=registry, _require_verified_local_disk_completion=True)
        lifecycle = _ConnectorLifecycle(records.append, registry, None, None, endpoint)
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
        manager = type("Manager", (), {
            "storage_backends": {"LocalDiskBackend": object()},
            "lmcache_engine": type("Engine", (), {"retrieve": lambda self, *args, **kwargs: []})(),
        })()
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        token = lifecycle._active_reqmeta_id.set("reqmeta-a")
        try:
            lifecycle.track_submission("key-a", submitted, context, manager)
        finally:
            lifecycle._active_reqmeta_id.reset(token)

        lifecycle.release_after_unpin("reqmeta-a", source="lookup_unpin")
        self.assertEqual(registry.snapshot(submitted.binding.binding_id)["pending_io"], 1)
        self.assertEqual(records, [])

        lease = submitted.event.metadata["operation_lease"]
        completed = registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, lease)
        endpoint.mark_store_completed(completed.binding, "key-a", manager)
        lifecycle.complete_store("key-a", lease)

        self.assertEqual(registry.snapshot(submitted.binding.binding_id)["pending_io"], 0)
        events = [record for record in records if record.get("record_type") == "event"]
        self.assertEqual([(record["action"], record["status"]) for record in events], [("release", "completed")])
        bindings = [record for record in records if record.get("record_type") == "binding"]
        self.assertEqual(bindings[-1]["execution_spec"]["actions"]["load"]["status"], "ready")
        self.assertTrue(endpoint.action_registration_enabled)

    def test_factory_capture_emits_real_key_events_and_executes_drop(self):
        events = []

        class Manager:
            def __init__(self):
                self.removed = []
            def batched_put(self, keys, objects, **kwargs):
                return None
            def get(self, key, location=None):
                return object()
            def remove(self, key, locations=None):
                self.removed.append((key, locations))
                return 1

        manager = Manager()
        engine = type("Engine", (), {"storage_manager": manager})()
        class Factory:
            def get_or_create_lmcache_engine(self):
                return engine

        endpoint = install_lmcache047_hooks(
            events.append,
            factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
        )
        Factory().get_or_create_lmcache_engine()
        manager.batched_put(["key-a"], [object()])
        manager.get("key-a")
        receipt = endpoint.drop("key-a", locations=["LocalDiskBackend"])

        self.assertEqual([event["action"] for event in events], ["cache_store_submitted", "cache_hit"])
        self.assertEqual(receipt["status"], "observational_only")
        self.assertEqual(manager.removed, [])

    def test_unknown_version_refuses_installation(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported runtime versions"):
            install_lmcache047_hooks(
                lambda event: None,
                factory_cls=object,
                versions={"vllm": "0.23.1", "lmcache": "0.4.7"},
            )

    def test_post_init_binds_storage_manager_created_after_factory(self):
        events = []
        class ManagerStorage:
            def batched_put(self, keys, objects, **kwargs): pass
            def get(self, key, location=None): return None
            def batched_get(self, keys, location=None): return [object(), None]
            def remove(self, key, locations=None): return 0
        storage = ManagerStorage()
        engine = type("Engine", (), {"storage_manager": None})()
        class Factory:
            def get_or_create_lmcache_engine(self): return engine
        class Manager:
            _lmcache_engine = engine
            def post_init(self):
                engine.storage_manager = storage

        install_lmcache047_hooks(events.append, factory_cls=Factory, manager_cls=Manager, versions={"vllm": "0.23.0", "lmcache": "0.4.7"})
        Factory().get_or_create_lmcache_engine()
        Manager().post_init()
        storage.get("key-after-post-init")
        self.assertEqual(events[-1]["action"], "cache_miss")
        storage.batched_get(["key-a", "key-b"])
        self.assertEqual([item["action"] for item in events[-2:]], ["cache_hit", "cache_miss"])

    def test_registry_context_emits_normalized_binding_and_cache_store_events(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def batched_get(self, keys, location=None): return [None for _ in keys]
            def remove(self, key, locations=None): return 0
        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()
        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_provider=lambda: context,
        )
        Factory().get_or_create_lmcache_engine()
        storage.batched_put(["key-a"], [object()])
        storage.get("key-a")
        storage.batched_get(["key-a"])
        storage.remove("key-a")

        event_records = [record for record in records if record.get("record_type") == "event"]
        self.assertEqual([record["action"] for record in event_records], ["cache_store", "cache_miss", "cache_miss", "drop"])
        self.assertEqual(event_records[0]["status"], "submitted")
        self.assertEqual(event_records[0]["request_id"], "request")
        binding_records = [record for record in records if record.get("record_type") == "binding"]
        self.assertTrue(binding_records)
        self.assertTrue(all(record["schema"] == "astrakv-backend-hook-v2" for record in binding_records + event_records))
        self.assertEqual(binding_records[0]["event_id"], event_records[0]["event_id"])

    def test_store_observes_physical_size_bytes_for_execution_gate(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)

        class FakeMemory:
            def get_physical_size(self):
                return 37748736

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def batched_get(self, keys, location=None): return [None for _ in keys]
            def remove(self, key, locations=None): return 0
        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()

        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_provider=lambda: context,
        )
        Factory().get_or_create_lmcache_engine()
        storage.batched_put(["key-a"], [FakeMemory()])

        binding_records = [record for record in records if record.get("record_type") == "binding"]
        self.assertTrue(binding_records)
        metadata = dict(binding_records[0].get("metadata") or {})
        self.assertEqual(int(metadata.get("size_bytes") or 0), 37748736)
        snapshot = registry.snapshot(binding_records[0]["binding_id"])
        self.assertEqual(int(snapshot.get("size_bytes") or 0), 37748736)

    def test_registry_context_emits_block_statistics_when_storage_key_carries_hints(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
        key = {"block_ids": [7, 8], "block_size_tokens": 16}

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return object()
            def batched_get(self, keys, location=None): return [object() for _ in keys]
            def remove(self, key, locations=None): return 0

        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()

        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_provider=lambda: context,
        )
        Factory().get_or_create_lmcache_engine()
        storage.batched_put([key], [object()])
        storage.get(key)

        event_records = [record for record in records if record.get("record_type") == "event"]
        store_event = next(record for record in event_records if record["action"] == "cache_store")
        hit_event = next(record for record in event_records if record["action"] == "cache_hit")
        self.assertEqual(store_event["metadata"]["block_count_store"], 2)
        self.assertEqual(store_event["metadata"]["token_count_store"], 32)
        self.assertEqual(hit_event["metadata"]["block_ids_hit"], [7, 8])
        self.assertEqual(hit_event["metadata"]["block_count_hit"], 2)
        self.assertEqual(hit_event["metadata"]["token_count_hit"], 32)

    def test_action_endpoint_refuses_raw_and_pending_objects_then_executes_released_binding(self):
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)

        class Manager:
            def __init__(self): self.removed = []
            def remove(self, key, locations=None):
                self.removed.append((key, locations))
                return 1
        manager = Manager()
        endpoint = LMCache047ActionEndpoint(binding_registry=registry, action_registration_enabled=True)
        endpoint.register_binding(submitted.binding, "key-a", manager)
        params = {
            "binding_id": submitted.binding.binding_id,
            "binding_generation": submitted.binding.binding_generation,
            "request_id": "request",
            "object_key": "prefix",
            "object_level": ObjectLevel.PREFIX,
        }
        self.assertEqual(endpoint.drop(submitted.binding.backend_object_id, **params)["status"], "not_eligible")
        self.assertEqual(endpoint.drop("key-a", **params)["status"], "unbound")

        registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, submitted.event.metadata["operation_lease"])
        self.assertEqual(endpoint.drop(submitted.binding.backend_object_id, **params)["status"], "not_eligible")
        registry.observe("key-a", HookAction.RELEASE, "completed", context)
        lease = registry.reserve_action(
            binding_id=submitted.binding.binding_id, binding_generation=submitted.binding.binding_generation,
            backend_object_id=submitted.binding.backend_object_id, request_id="request", object_key="prefix", object_level=ObjectLevel.PREFIX,
        )
        params.update({"reservation_lease": lease, "command_id": "command-1"})
        self.assertEqual(endpoint.drop(submitted.binding.backend_object_id, **params)["status"], "completed")
        self.assertEqual(endpoint.drop(submitted.binding.backend_object_id, **params)["status"], "reservation_not_available")
        self.assertEqual(manager.removed, [("key-a", None)])

    def test_production_storage_contract_is_observational_without_verified_terminal_callbacks(self):
        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def batched_get(self, keys, location=None): return [None for _ in keys]
            def remove(self, key, locations=None): return 0

        probe = probe_lmcache047_storage_contract(Storage)
        self.assertTrue(probe["compatible"])
        self.assertFalse(probe["action_registration_enabled"])
        self.assertEqual(probe["blocked_reason"], "no_verified_terminal_store_and_release_callbacks")

        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()
        class Factory:
            def get_or_create_lmcache_engine(self): return engine
        endpoint = install_lmcache047_hooks(
            records.append, factory_cls=Factory, versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_provider=lambda: RequestContext("run", "request", "prefix", ObjectLevel.PREFIX),
        )
        Factory().get_or_create_lmcache_engine()
        storage.batched_put(["key-a"], [object()])
        binding = next(record for record in records if record.get("record_type") == "binding")
        self.assertFalse(endpoint.action_registration_enabled)
        self.assertEqual(endpoint.drop(binding["backend_object_id"], binding_id=binding["binding_id"], binding_generation=1, request_id="request", object_key="prefix", object_level="prefix")["status"], "observational_only")

    def test_local_disk_completion_callback_completes_exact_store_lease_before_action_registration(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)

        class LocalDiskBackend:
            __module__ = "lmcache.v1.storage_backend.local_disk_backend"

            def __init__(self):
                self.callbacks = []

            def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None, on_complete_callback=None):
                self.callbacks.append(on_complete_callback)

        disk = LocalDiskBackend()

        class Storage:
            def __init__(self):
                self.storage_backends = {
                    "LocalCPUBackend": type("HotCPU", (), {"use_hot": True})(),
                    "LocalDiskBackend": disk,
                }
                self.lmcache_engine = type("Engine", (), {"retrieve": lambda *_args, **_kwargs: [True]})()
            def batched_put(self, keys, objects, transfer_spec=None, location=None):
                self.storage_backends["LocalDiskBackend"].batched_submit_put_task(
                    keys, objects, transfer_spec=transfer_spec,
                )
            def get(self, key, location=None): return None
            def batched_contains(self, keys, search_range=None, pin=False):
                location = None if not search_range else search_range[0]
                if location in {"LocalCPUBackend", "LocalDiskBackend"}:
                    return len(keys), {location: list(keys)}
                return 0, {}
            def batched_get(self, keys, location=None):
                class MemoryObj:
                    def get_physical_size(self):
                        return 1024
                return [MemoryObj() for _ in keys]
            def batched_remove(self, keys, locations=None): return len(keys)
            def remove(self, key, locations=None): return 0

        storage = Storage()
        engine = type("Engine", (), {"storage_manager": storage})()
        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        endpoint = install_lmcache047_hooks(
            records.append, factory_cls=Factory,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry, request_context_provider=lambda: context,
        )
        Factory().get_or_create_lmcache_engine()
        storage.batched_put(["key-a"], [object()])

        submitted = next(record for record in records if record.get("record_type") == "event")
        binding = next(record for record in records if record.get("record_type") == "binding")
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(registry.snapshot(binding["binding_id"])["pending_io"], 1)
        self.assertEqual(len(disk.callbacks), 1)
        self.assertIsNotNone(disk.callbacks[0])

        disk.callbacks[0]("key-a")

        completed = [record for record in records if record.get("record_type") == "event"][-1]
        self.assertEqual((completed["action"], completed["status"]), ("cache_store", "completed"))
        self.assertEqual(completed["tier_after"], "cpu")
        self.assertEqual(registry.snapshot(binding["binding_id"])["pending_io"], 0)
        self.assertFalse(endpoint.action_registration_enabled)
        released = registry.observe("key-a", HookAction.RELEASE, "completed", context)
        execution_spec = endpoint.register_binding(released.binding, "key-a", storage)
        self.assertTrue(endpoint.action_registration_enabled)
        assert execution_spec is not None
        self.assertEqual(execution_spec.binding_generation, released.binding.binding_generation)
        self.assertEqual(execution_spec.actions["drop"]["status"], "ready")
        self.assertEqual(execution_spec.actions["offload"]["status"], "ready")
        self.assertIsNone(execution_spec.actions["offload"]["blocked_reason"])
        self.assertEqual(execution_spec.actions["load"]["status"], "ready")
        self.assertIsNone(execution_spec.actions["load"]["blocked_reason"])
        self.assertEqual(execution_spec.actions["prefetch"]["status"], "ready")
        self.assertIsNone(execution_spec.actions["prefetch"]["blocked_reason"])
        self.assertEqual(execution_spec.actions["evict"]["status"], "ready")
        self.assertIsNone(execution_spec.actions["evict"]["blocked_reason"])

    def test_release_binding_record_is_enriched_with_execution_spec_after_verified_registration(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        endpoint = LMCache047ActionEndpoint(binding_registry=registry, _require_verified_local_disk_completion=True)
        lifecycle = _ConnectorLifecycle(records.append, registry, None, None, endpoint)
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("key-a", HookAction.CACHE_STORE, "submitted", context)
        token = lifecycle._active_reqmeta_id.set("reqmeta-a")
        try:
            lifecycle.track_submission("key-a", submitted, context)
        finally:
            lifecycle._active_reqmeta_id.reset(token)

        lease = submitted.event.metadata["operation_lease"]
        completed = registry.complete_operation("key-a", HookAction.CACHE_STORE, "completed", context, lease)
        endpoint.mark_store_completed(completed.binding, "key-a", object())
        lifecycle.release_after_unpin("reqmeta-a", source="lookup_unpin")
        lifecycle.complete_store("key-a", lease)

        binding_records = [record for record in records if record.get("record_type") == "binding"]
        self.assertEqual(binding_records[-1]["status"], "completed")
        self.assertIn("execution_spec", binding_records[-1])
        self.assertEqual(binding_records[-1]["execution_spec"]["binding_generation"], 1)
        self.assertEqual(binding_records[-1]["execution_spec"]["actions"]["drop"]["status"], "ready")
        self.assertEqual(binding_records[-1]["execution_spec"]["actions"]["load"]["status"], "blocked")

    def test_connector_lifecycle_correlates_reqmeta_and_releases_only_after_real_storage_submission(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        authority = RuntimeRequestContextAuthority.install(run_id="run", session_id="session", secret=b"a" * 32, ttl_s=60)
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        consumer = LMCache047RequestContextConsumer(receiver)
        handoff = RuntimeRequestContext(
            "run", "logical-request", "case", "nonce", 1.0,
            metadata={"scenario": "metadata-propagation", "reuse_ratio": 0.15},
        )
        receipt = receiver.receive(handoff.to_record(), authority.context_headers(handoff, receiver.endpoint_identity))
        self.assertEqual(receipt.status, "recorded")

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def remove(self, key, locations=None): return 0

        class Engine:
            def __init__(self):
                self.storage_manager = Storage()
                self.unpinned = []
            def store(self, *args, **kwargs):
                self.storage_manager.batched_put(["physical-key-7"], [object()])
            def lookup_unpin(self, req_id): self.unpinned.append(req_id)

        engine = Engine()
        request = type("ReqMeta", (), {"req_id": "reqmeta-7"})()
        metadata = type("Metadata", (), {"requests": [request]})()
        parent = type("Parent", (), {"_get_connector_metadata": lambda self: metadata})()
        class Connector:
            def __init__(self): self.lmcache_engine = engine; self._parent = parent
            def save_kv_layer(self, *args, **kwargs): self.lmcache_engine.store(req_id="reqmeta-7")
            def wait_for_save(self): self.lmcache_engine.lookup_unpin("reqmeta-7")
            def request_finished(self, request, block_ids): return False, None
            def get_finished(self, finished_req_ids): return None, None

        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        endpoint = install_lmcache047_hooks(
            records.append, factory_cls=Factory, connector_cls=Connector,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"}, binding_registry=registry,
            request_context_consumer=consumer,
            runtime_request_identity_provider=lambda req_id: RuntimeRequestIdentity("run", "logical-request", "nonce") if req_id == "reqmeta-7" else None,
        )
        Factory().get_or_create_lmcache_engine()
        connector = Connector()
        connector.save_kv_layer("layer", None, None)
        connector.get_finished({"reqmeta-7"})
        connector.wait_for_save()

        events = [record for record in records if record.get("record_type") == "event"]
        self.assertEqual([event["action"] for event in events], ["cache_store", "release"])
        self.assertEqual([event["status"] for event in events], ["submitted", "completed"])
        self.assertEqual(events[0]["request_id"], "logical-request")
        self.assertEqual(events[0]["metadata"]["runtime_reqmeta_id"], "reqmeta-7")
        self.assertEqual(events[0]["metadata"]["scenario"], "metadata-propagation")
        self.assertEqual(events[0]["metadata"]["reuse_ratio"], 0.15)
        self.assertNotIn("cache_store", [event["action"] for event in events[1:]])
        self.assertFalse(endpoint.action_registration_enabled)
        self.assertEqual(endpoint.drop(events[0]["backend_object_id"])["status"], "observational_only")

    def test_connector_store_exception_emits_failed_terminal_event_but_never_completed_or_release(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        authority = RuntimeRequestContextAuthority.install(run_id="run", session_id="session", secret=b"b" * 32, ttl_s=60)
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        consumer = LMCache047RequestContextConsumer(receiver)
        handoff = RuntimeRequestContext("run", "logical-request", "case", "nonce", 1.0)
        receiver.receive(handoff.to_record(), authority.context_headers(handoff, receiver.endpoint_identity))

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def remove(self, key, locations=None): return 0
        class Engine:
            def __init__(self): self.storage_manager = Storage()
            def store(self, *args, **kwargs):
                self.storage_manager.batched_put(["physical-key-8"], [object()])
                raise RuntimeError("store failed")
        engine = Engine()
        request = type("ReqMeta", (), {"req_id": "reqmeta-8"})()
        metadata = type("Metadata", (), {"requests": [request]})()
        parent = type("Parent", (), {"_get_connector_metadata": lambda self: metadata})()
        class Connector:
            def __init__(self): self.lmcache_engine = engine; self._parent = parent
            def save_kv_layer(self, *args, **kwargs): self.lmcache_engine.store(req_id="reqmeta-8")
            def wait_for_save(self): raise AssertionError("must not be called")
            def request_finished(self, request, block_ids): return False, None
            def get_finished(self, finished_req_ids): return None, None
        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        install_lmcache047_hooks(
            records.append, factory_cls=Factory, connector_cls=Connector,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"}, binding_registry=registry,
            request_context_consumer=consumer,
            runtime_request_identity_provider=lambda req_id: RuntimeRequestIdentity("run", "logical-request", "nonce") if req_id == "reqmeta-8" else None,
        )
        Factory().get_or_create_lmcache_engine()
        with self.assertRaisesRegex(RuntimeError, "store failed"):
            Connector().save_kv_layer("layer", None, None)

        events = [record for record in records if record.get("record_type") == "event"]
        self.assertEqual([(event["action"], event["status"]) for event in events], [("cache_store", "submitted"), ("cache_store", "failed")])

    def test_connector_request_finished_is_teardown_safe_when_engine_has_already_gone_away(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        authority = RuntimeRequestContextAuthority.install(run_id="run", session_id="session", secret=b"e" * 32, ttl_s=60)
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        consumer = LMCache047RequestContextConsumer(receiver)

        class Connector:
            def __init__(self):
                self.lmcache_engine = object()
                self._parent = None
            def save_kv_layer(self, *args, **kwargs):
                return None
            def wait_for_save(self, *args, **kwargs):
                return None
            def start_load_kv(self, request):
                return request
            def wait_for_layer_load(self, request):
                return request
            def request_finished(self, request, block_ids):
                self.lmcache_engine = None
                raise AssertionError("assert self.lmcache_engine is not None")
            def get_finished(self, finished_req_ids):
                return None, None
            def get_num_new_matched_tokens(self, request):
                return 0
            def update_state_after_alloc(self, request, num_computed_tokens=0):
                return request

        class Factory:
            def get_or_create_lmcache_engine(self):
                return object()

        install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            connector_cls=Connector,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_consumer=consumer,
            runtime_request_identity_provider=lambda req_id: RuntimeRequestIdentity("run", "logical-request", "nonce") if req_id == "reqmeta-12" else None,
        )

        connector = Connector()
        request = type("ReqMeta", (), {"req_id": "reqmeta-12"})()
        result = connector.request_finished(request, [])

        self.assertEqual(result, (False, None))

    def test_connector_load_lifecycle_registers_dynamic_load_target_and_emits_ready_signal(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        authority = RuntimeRequestContextAuthority.install(run_id="run", session_id="session", secret=b"c" * 32, ttl_s=60)
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        consumer = LMCache047RequestContextConsumer(receiver)
        handoff = RuntimeRequestContext(
            "run",
            "logical-request",
            "case",
            "nonce",
            1.0,
            metadata={"object_key": "shared-prefix", "prefix_id": "shared-prefix", "scenario": "hot_load"},
        )
        receiver.receive(handoff.to_record(), authority.context_headers(handoff, receiver.endpoint_identity))

        class FakeTensor:
            def element_size(self):
                return 256
            def numel(self):
                return 16

        class Storage:
            def batched_put(self, keys, objects, **kwargs): return None
            def get(self, key, location=None): return None
            def remove(self, key, locations=None): return 0

        class Engine:
            def __init__(self):
                self.storage_manager = Storage()
            def retrieve(self, *args, **kwargs):
                return [True, True, True, True]

        engine = Engine()
        manager = type("Manager", (), {
            "storage_backends": {"LocalDiskBackend": object()},
            "lmcache_engine": engine,
        })()

        request = type("ReqMeta", (), {"req_id": "reqmeta-9"})()
        metadata = type("Metadata", (), {"requests": [request]})()
        parent = type("Parent", (), {"_get_connector_metadata": lambda self: metadata})()

        class LoadRequest:
            req_id = "reqmeta-9"
            token_ids = [101, 102, 103, 104]
            slot_mapping = [0, 1, 2, 3]
            request_configs = {"case": "hot_load"}
            kvcaches = [FakeTensor()]

        class Connector:
            def __init__(self):
                self.lmcache_engine = engine
                self._parent = parent
            def start_load_kv(self, request):
                return request
            def wait_for_layer_load(self, request):
                return request
            def save_kv_layer(self, *args, **kwargs): return None
            def wait_for_save(self): return None
            def request_finished(self, request, block_ids): return False, None
            def get_finished(self, finished_req_ids): return None, None
            def get_num_new_matched_tokens(self, request): return len(request.token_ids)
            def update_state_after_alloc(self, request, num_computed_tokens=0): return request

        class Factory:
            def get_or_create_lmcache_engine(self): return engine

        endpoint = install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            connector_cls=Connector,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_consumer=consumer,
            runtime_request_identity_provider=lambda req_id: RuntimeRequestIdentity("run", "logical-request", "nonce") if req_id == "reqmeta-9" else None,
        )
        endpoint.action_registration_enabled = True

        released_context = RequestContext("run", "stored-request", "shared-prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("physical-key-9", HookAction.CACHE_STORE, "submitted", released_context)
        registry.complete_operation(
            "physical-key-9",
            HookAction.CACHE_STORE,
            "completed",
            released_context,
            submitted.event.metadata["operation_lease"],
        )
        released = registry.observe("physical-key-9", HookAction.RELEASE, "completed", released_context)
        assert released.binding is not None
        endpoint.register_binding(released.binding, "physical-key-9", manager)

        Factory().get_or_create_lmcache_engine()
        connector = Connector()
        request_obj = LoadRequest()
        connector.start_load_kv(request_obj)
        connector.get_num_new_matched_tokens(request_obj)
        connector.update_state_after_alloc(request_obj, num_computed_tokens=0)
        connector.wait_for_layer_load(request_obj)

        binding_records = [record for record in records if record.get("record_type") == "binding" and record.get("request_id") == "stored-request"]
        self.assertTrue(binding_records)
        latest_binding = binding_records[-1]
        self.assertEqual(latest_binding["execution_spec"]["actions"]["load"]["load_target_id"], "load-target:reqmeta-9")
        self.assertEqual(latest_binding["execution_spec"]["actions"]["load"]["runtime_reqmeta_id"], "reqmeta-9")
        self.assertTrue(latest_binding["execution_spec"]["actions"]["load"]["native_request_load"])

        ready_events = [
            record for record in records
            if record.get("record_type") == "event"
            and record.get("action") == "cache_load"
            and record.get("status") == "available"
        ]
        self.assertEqual(len(ready_events), 1)
        self.assertEqual(ready_events[0]["request_id"], "stored-request")
        self.assertEqual(ready_events[0]["metadata"]["load_target_id"], "load-target:reqmeta-9")
        self.assertEqual(ready_events[0]["metadata"]["runtime_reqmeta_id"], "reqmeta-9")
        self.assertTrue(ready_events[0]["metadata"]["native_request_load"])
        self.assertEqual(ready_events[0]["metadata"]["block_count_load"], 4)
        self.assertEqual(ready_events[0]["metadata"]["token_count_load"], 4)
        self.assertEqual(ready_events[0]["metadata"]["block_size_tokens"], 1)

    def test_connector_retrieve_context_allows_reserved_key_reads_for_active_request(self):
        records = []
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        authority = RuntimeRequestContextAuthority.install(run_id="run", session_id="session", secret=b"d" * 32, ttl_s=60)
        receiver = RuntimeRequestContextReceiver("http://127.0.0.1:9988/request-context", authority)
        consumer = LMCache047RequestContextConsumer(receiver)
        handoff = RuntimeRequestContext(
            "run",
            "logical-request",
            "case",
            "nonce",
            1.0,
            metadata={"object_key": "shared-prefix", "prefix_id": "shared-prefix", "scenario": "hot_load"},
        )
        receiver.receive(handoff.to_record(), authority.context_headers(handoff, receiver.endpoint_identity))

        class Storage:
            def __init__(self):
                self.batched_get_calls = 0
            def batched_put(self, keys, objects, **kwargs):
                return None
            def get(self, key, location=None):
                return None
            def batched_get(self, keys, location=None):
                self.batched_get_calls += 1
                return [object() for _ in keys]
            def remove(self, key, locations=None):
                return 0

        storage = Storage()

        class Engine:
            def __init__(self):
                self.storage_manager = storage
            def retrieve(self, *args, **kwargs):
                return self.storage_manager.batched_get(["physical-key-11"], location="LocalDiskBackend")

        engine = Engine()
        request = type("ReqMeta", (), {"req_id": "reqmeta-11"})()
        metadata = type("Metadata", (), {"requests": [request]})()
        parent = type("Parent", (), {"_get_connector_metadata": lambda self: metadata})()

        class Connector:
            def __init__(self):
                self.lmcache_engine = engine
                self._parent = parent
            def start_load_kv(self, request):
                return request
            def wait_for_layer_load(self, request):
                return request
            def save_kv_layer(self, *args, **kwargs):
                return None
            def wait_for_save(self):
                return None
            def request_finished(self, request, block_ids):
                return False, None
            def get_finished(self, finished_req_ids):
                return None, None
            def get_num_new_matched_tokens(self, request):
                return len(request.token_ids)
            def update_state_after_alloc(self, request, num_computed_tokens=0):
                return request

        class Factory:
            def get_or_create_lmcache_engine(self):
                return engine

        endpoint = install_lmcache047_hooks(
            records.append,
            factory_cls=Factory,
            connector_cls=Connector,
            versions={"vllm": "0.23.0", "lmcache": "0.4.7"},
            binding_registry=registry,
            request_context_consumer=consumer,
            runtime_request_identity_provider=lambda req_id: RuntimeRequestIdentity("run", "logical-request", "nonce") if req_id == "reqmeta-11" else None,
        )
        endpoint.action_registration_enabled = True

        released_context = RequestContext("run", "stored-request", "shared-prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("physical-key-11", HookAction.CACHE_STORE, "submitted", released_context)
        registry.complete_operation(
            "physical-key-11",
            HookAction.CACHE_STORE,
            "completed",
            released_context,
            submitted.event.metadata["operation_lease"],
        )
        registry.observe("physical-key-11", HookAction.RELEASE, "completed", released_context)
        binding = registry.current_binding(
            binding_id=submitted.binding.binding_id,
            binding_generation=submitted.binding.binding_generation,
            request_id="stored-request",
            object_key="shared-prefix",
            object_level=ObjectLevel.PREFIX,
        )
        assert binding is not None
        endpoint.register_binding(binding, "physical-key-11", type("Manager", (), {"storage_backends": {"LocalDiskBackend": object()}, "lmcache_engine": engine})())
        registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="stored-request",
            object_key="shared-prefix",
            object_level=ObjectLevel.PREFIX,
        )

        class LoadRequest:
            req_id = "reqmeta-11"
            token_ids = [101]
            slot_mapping = [0]
            request_configs = {"case": "hot_load"}
            kvcaches = []

        Factory().get_or_create_lmcache_engine()
        connector = Connector()
        connector.start_load_kv(LoadRequest())
        result = connector.lmcache_engine.retrieve(req_id="reqmeta-11")

        self.assertEqual(storage.batched_get_calls, 1)
        self.assertEqual(len(result), 1)
        deferred_events = [
            record for record in records
            if record.get("record_type") == "event"
            and record.get("action") == "cache_hit"
            and record.get("status") == "deferred"
        ]
        self.assertEqual(deferred_events, [])

    def test_dynamic_load_target_rebuilds_blocked_spec_from_verified_completed_store(self):
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        endpoint = LMCache047ActionEndpoint(binding_registry=registry, _require_verified_local_disk_completion=True)
        context = RequestContext("run", "stored-request", "shared-prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("physical-key-10", HookAction.CACHE_STORE, "submitted", context)
        completed = registry.complete_operation(
            "physical-key-10",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        )
        manager = type("Manager", (), {
            "storage_backends": {"LocalDiskBackend": object()},
            "lmcache_engine": type("Engine", (), {"retrieve": lambda self, *args, **kwargs: []})(),
        })()
        endpoint.mark_store_completed(completed.binding, "physical-key-10", manager)
        released = registry.observe("physical-key-10", HookAction.RELEASE, "completed", context)
        assert released.binding is not None

        endpoint.register_binding(released.binding, "physical-key-10", None)
        _, updated = endpoint.register_dynamic_load_target(
            object_key="shared-prefix",
            object_level=ObjectLevel.PREFIX,
            target_id="load-target-10",
            runtime_reqmeta_id="reqmeta-10",
            token_ids=[101, 102, 103, 104],
            slot_mapping=[0, 1, 2, 3],
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            request_configs={"scenario": "hot_load"},
            kvcaches=[],
            target_tier="gpu",
        )

        assert updated is not None
        self.assertEqual(updated.execution_spec.actions["load"]["status"], "ready")
        self.assertEqual(updated.execution_spec.actions["load"]["load_target_id"], "load-target-10")

    def test_later_binding_registration_inherits_existing_future_load_target_for_same_object(self):
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        endpoint = LMCache047ActionEndpoint(binding_registry=registry, _require_verified_local_disk_completion=True)
        manager = type("Manager", (), {
            "storage_backends": {"LocalDiskBackend": object()},
            "lmcache_engine": type("Engine", (), {"retrieve": lambda self, *args, **kwargs: []})(),
        })()
        context = RequestContext("run", "stored-request", "shared-prefix", ObjectLevel.PREFIX)

        first_submitted = registry.observe("physical-key-a", HookAction.CACHE_STORE, "submitted", context)
        first_completed = registry.complete_operation(
            "physical-key-a",
            HookAction.CACHE_STORE,
            "completed",
            context,
            first_submitted.event.metadata["operation_lease"],
        )
        endpoint.mark_store_completed(first_completed.binding, "physical-key-a", manager)
        first_released = registry.observe("physical-key-a", HookAction.RELEASE, "completed", context)
        assert first_released.binding is not None
        endpoint.register_binding(first_released.binding, "physical-key-a", manager)
        _, updated = endpoint.register_dynamic_load_target(
            object_key="shared-prefix",
            object_level=ObjectLevel.PREFIX,
            target_id="load-target-future",
            runtime_reqmeta_id="reqmeta-future",
            token_ids=[101, 102, 103, 104],
            slot_mapping=[0, 1, 2, 3],
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4,
            request_configs={"scenario": "hot_load"},
            kvcaches=[],
            target_tier="gpu",
        )
        assert updated is not None

        second_submitted = registry.observe("physical-key-b", HookAction.CACHE_STORE, "submitted", context)
        second_completed = registry.complete_operation(
            "physical-key-b",
            HookAction.CACHE_STORE,
            "completed",
            context,
            second_submitted.event.metadata["operation_lease"],
        )
        endpoint.mark_store_completed(second_completed.binding, "physical-key-b", manager)
        second_released = registry.observe("physical-key-b", HookAction.RELEASE, "completed", context)
        assert second_released.binding is not None

        inherited_spec = endpoint.register_binding(second_released.binding, "physical-key-b", manager)
        assert inherited_spec is not None
        self.assertEqual(inherited_spec.actions["load"]["load_target_id"], "load-target-future")
        self.assertEqual(inherited_spec.actions["load"]["runtime_reqmeta_id"], "reqmeta-future")

    def test_prefetch_ready_requires_hot_cpu_backend(self) -> None:
        manager = type("Manager", (), {
            "storage_backends": {
                "LocalCPUBackend": object(),
                "LocalDiskBackend": object(),
            },
            "batched_contains": lambda *args, **kwargs: (0, {}),
            "batched_get": lambda *args, **kwargs: [],
        })()
        supported, blocked_reason = _prefetch_ready(manager)
        self.assertFalse(supported)
        self.assertEqual(blocked_reason, "prefetch_target_backend_not_hot:LocalCPUBackend")
        manager.storage_backends["LocalCPUBackend"] = type("HotCPU", (), {"use_hot": True})()
        self.assertEqual(_prefetch_ready(manager), (True, ""))

    def _released_prefetch_binding(self, manager):
        registry = BackendBindingRegistry(run_id="run", engine_instance_id="engine", worker_id="worker")
        context = RequestContext("run", "request", "prefix", ObjectLevel.PREFIX)
        submitted = registry.observe("physical-key", HookAction.CACHE_STORE, "submitted", context)
        binding = registry.complete_operation(
            "physical-key",
            HookAction.CACHE_STORE,
            "completed",
            context,
            submitted.event.metadata["operation_lease"],
        ).binding
        registry.observe("physical-key", HookAction.RELEASE, "completed", context)
        endpoint = LMCache047ActionEndpoint(registry, action_registration_enabled=True)
        endpoint.register_binding(binding, "physical-key", manager)
        return registry, binding, endpoint

    def _prefetch_command(self, registry, binding, *, command_id="command-prefetch"):
        lease = registry.reserve_action(
            binding_id=binding.binding_id,
            binding_generation=binding.binding_generation,
            backend_object_id=binding.backend_object_id,
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
        )
        return BackendActionCommand(
            run_id="run",
            command_id=command_id,
            decision_id="decision",
            request_id="request",
            object_key="prefix",
            object_level=ObjectLevel.PREFIX,
            binding_id=binding.binding_id,
            backend_object_id=binding.backend_object_id,
            action=HookAction.PREFETCH,
            issued_at_ns=1,
            target_tier="cpu",
            metadata={"reservation_lease": lease},
            binding_generation=binding.binding_generation,
        )

    def test_prefetch_executor_success_records_completed_receipt(self) -> None:
        class HotCPU:
            use_hot = True

        class SuccessManager:
            def __init__(self):
                self.storage_backends = {
                    "LocalCPUBackend": HotCPU(),
                    "LocalDiskBackend": object(),
                }
                self.cpu_present = False

            def batched_contains(self, keys, search_range=None, pin=False):
                location = None if not search_range else search_range[0]
                if location == "LocalCPUBackend":
                    return (1, {"LocalCPUBackend": list(keys)}) if self.cpu_present else (0, {})
                if location == "LocalDiskBackend":
                    return len(keys), {location: list(keys)}
                return 0, {}

            def batched_get(self, keys, location=None):
                class MemoryObj:
                    def get_physical_size(self):
                        return 2048

                self.cpu_present = True
                return [MemoryObj() for _ in keys]

            def remove(self, key, locations=None):
                return 0

        registry, binding, endpoint = self._released_prefetch_binding(SuccessManager())
        receipt = endpoint.execute_action(self._prefetch_command(registry, binding))
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["tier_before"], "ssd")
        self.assertEqual(receipt["tier_after"], "cpu")
        self.assertEqual(receipt["bytes"], 2048)
        self.assertEqual(receipt["prefetched"], 1)

    def test_prefetch_executor_missing_memory_objects_records_failure_reason(self) -> None:
        class HotCPU:
            use_hot = True

        class MissingMemoryManager:
            storage_backends = {
                "LocalCPUBackend": HotCPU(),
                "LocalDiskBackend": object(),
            }

            def batched_contains(self, keys, search_range=None, pin=False):
                location = None if not search_range else search_range[0]
                if location == "LocalCPUBackend":
                    return 0, {}
                if location == "LocalDiskBackend":
                    return len(keys), {location: list(keys)}
                return 0, {}

            def batched_get(self, keys, location=None):
                return [None for _ in keys]

            def remove(self, key, locations=None):
                return 0

        registry, binding, endpoint = self._released_prefetch_binding(MissingMemoryManager())
        receipt = endpoint.execute_action(self._prefetch_command(registry, binding))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["failure_reason"], "missing_memory_objects")
        self.assertEqual(receipt["missing_memory_obj_count"], 1)
        self.assertEqual(receipt["prefetched"], 0)
        self.assertIn("cpu_used_bytes", receipt)
        self.assertIn("cpu_capacity_bytes", receipt)
        self.assertIn("cpu_prefetch_budget_bytes", receipt)
        self.assertIn("memory_pressure", receipt)

    def test_prefetch_executor_backend_exception_records_diagnostics(self) -> None:
        class HotCPU:
            use_hot = True

        class FailingManager:
            storage_backends = {
                "LocalCPUBackend": HotCPU(),
                "LocalDiskBackend": object(),
            }

            def batched_contains(self, keys, search_range=None, pin=False):
                location = None if not search_range else search_range[0]
                if location == "LocalCPUBackend":
                    return 0, {}
                if location == "LocalDiskBackend":
                    return len(keys), {location: list(keys)}
                return 0, {}

            def batched_get(self, keys, location=None):
                raise RuntimeError("staging allocation failed")

            def remove(self, key, locations=None):
                return 0

        registry, binding, endpoint = self._released_prefetch_binding(FailingManager())
        receipt = endpoint.execute_action(self._prefetch_command(registry, binding))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["failure_reason"], "prefetch_backend_exception")
        self.assertEqual(receipt["error_type"], "RuntimeError")
        self.assertIn("staging allocation failed", receipt["error"])
        self.assertIn("cpu_used_bytes", receipt)
        self.assertIn("memory_pressure", receipt)


if __name__ == "__main__":
    unittest.main()
