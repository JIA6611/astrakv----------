import unittest
from dataclasses import replace

from astrakv.runtime.backend_capabilities import (
    SUPPORTED_LMCACHE_VERSION,
    SUPPORTED_VLLM_VERSION,
    build_installation_evidence,
    normalize_loopback_endpoint,
    preflight_backend_capabilities,
)
from astrakv.runtime.eviction import ObjectLevel


class BackendCapabilityPreflightTests(unittest.TestCase):
    def evidence(
        self,
        hook_url: str = "http://127.0.0.1:7900/actions",
        vllm_version: str | None = SUPPORTED_VLLM_VERSION,
        connector_name: str | None = "lmcache-vllm-v1",
    ):
        return build_installation_evidence(
            source="lmcache047_runtime_patch", method="signature_probe", session_id="session-a",
            vllm_version=vllm_version, lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name=connector_name, connector_version=SUPPORTED_LMCACHE_VERSION,
            endpoint_identity=normalize_loopback_endpoint(hook_url),
        )

    def test_exact_version_locked_runtime_advertises_only_supported_contract(self) -> None:
        result = preflight_backend_capabilities(
            vllm_version=SUPPORTED_VLLM_VERSION,
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1",
            connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop", "offload", "load", "prefetch", "evict"),
            available_object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True,
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=self.evidence(),
        )

        self.assertTrue(result.compatible)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(result.allowed_actions, ("drop", "offload", "load", "prefetch", "evict"))
        self.assertEqual(result.action_status["drop"]["status"], "allowed")
        self.assertEqual(result.action_status["offload"]["status"], "allowed")
        self.assertIsNone(result.action_status["offload"]["reason"])
        self.assertEqual(result.action_status["load"]["status"], "allowed")
        self.assertIsNone(result.action_status["load"]["reason"])
        self.assertEqual(result.action_status["prefetch"]["status"], "allowed")
        self.assertIsNone(result.action_status["prefetch"]["reason"])
        self.assertEqual(result.action_status["evict"]["status"], "allowed")
        self.assertIsNone(result.action_status["evict"]["reason"])
        self.assertEqual(result.object_levels, (ObjectLevel.PREFIX,))
        self.assertTrue(result.capability_flags["binding_generation"])
        self.assertTrue(result.execution_eligible)
        record = result.to_record()
        self.assertEqual(record["backend_versions"], {
            "vllm": "0.23.0",
            "lmcache": "0.4.7",
        })
        self.assertEqual(record["connector"]["version"], "0.4.7")
        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["endpoint_identity"], "http://127.0.0.1:7900/actions")
        self.assertTrue(record["capability_flags"]["installation_evidence"])
        self.assertEqual(record["action_status"]["load"]["status"], "allowed")
        self.assertEqual(record["action_status"]["evict"]["status"], "allowed")
        self.assertEqual(record["installation_evidence"]["schema"], "astrakv-installation-evidence-v1")
        self.assertIsNone(record["blocked_reason"])

    def test_mismatched_or_unknown_versions_are_explicitly_blocked(self) -> None:
        mismatch = preflight_backend_capabilities(
            vllm_version="0.22.0",
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1",
            connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop",),
            available_object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True,
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=self.evidence(vllm_version="0.22.0"),
        )
        unknown = preflight_backend_capabilities(
            vllm_version=None,
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1",
            connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop",),
            available_object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True,
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=self.evidence(vllm_version=None),
        )

        self.assertFalse(mismatch.compatible)
        self.assertEqual(mismatch.blocked_reason, "unsupported_vllm_version:0.22.0")
        self.assertFalse(unknown.compatible)
        self.assertEqual(unknown.blocked_reason, "unknown_vllm_version")

    def test_unobserved_connector_action_or_object_level_is_deny_by_default(self) -> None:
        result = preflight_backend_capabilities(
            vllm_version=SUPPORTED_VLLM_VERSION,
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name=None,
            connector_version=SUPPORTED_LMCACHE_VERSION,
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=self.evidence(connector_name=None),
        )

        self.assertFalse(result.compatible)
        self.assertFalse(result.execution_eligible)
        self.assertEqual(result.allowed_actions, ())
        self.assertEqual(result.object_levels, ())
        self.assertEqual(result.blocked_reason, "unknown_connector_name")
        self.assertFalse(result.capability_flags["connector_identity"])
        self.assertFalse(result.capability_flags["drop_action"])
        self.assertFalse(result.capability_flags["prefix_object_level"])

    def test_binding_generation_requires_an_explicit_probe(self) -> None:
        result = preflight_backend_capabilities(
            vllm_version=SUPPORTED_VLLM_VERSION,
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1",
            connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop",),
            available_object_levels=(ObjectLevel.PREFIX,),
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=self.evidence(),
        )

        self.assertFalse(result.execution_eligible)
        self.assertEqual(result.blocked_reason, "binding_generation_not_observed")
        self.assertFalse(result.capability_flags["binding_generation"])

    def test_preflight_requires_bound_run_endpoint_and_installation_evidence(self) -> None:
        result = preflight_backend_capabilities(
            vllm_version=SUPPORTED_VLLM_VERSION,
            lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1",
            connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop",),
            available_object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True,
            run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions",
            installation_evidence=None,
        )

        self.assertFalse(result.execution_eligible)
        self.assertEqual(result.blocked_reason, "missing_installation_evidence")
        self.assertFalse(result.capability_flags["installation_evidence"])

    def test_execution_validation_rejects_tampered_cached_or_evidence_fields(self) -> None:
        preflight = preflight_backend_capabilities(
            vllm_version=SUPPORTED_VLLM_VERSION, lmcache_version=SUPPORTED_LMCACHE_VERSION,
            connector_name="lmcache-vllm-v1", connector_version=SUPPORTED_LMCACHE_VERSION,
            available_actions=("drop",), available_object_levels=(ObjectLevel.PREFIX,),
            binding_generation_observed=True, run_id="run-1",
            hook_url="http://127.0.0.1:7900/actions", installation_evidence=self.evidence(),
        )
        forged_evidence = replace(preflight.installation_evidence, probe_digest="0" * 64)

        self.assertTrue(preflight.validate_for_execution("run-1", "http://127.0.0.1:7900/actions"))
        self.assertFalse(replace(preflight, compatible=False).validate_for_execution(
            "run-1", "http://127.0.0.1:7900/actions",
        ))
        self.assertFalse(replace(preflight, blocked_reason="forged").validate_for_execution(
            "run-1", "http://127.0.0.1:7900/actions",
        ))
        self.assertFalse(replace(preflight, vllm_version="0.0.0").validate_for_execution(
            "run-1", "http://127.0.0.1:7900/actions",
        ))
        self.assertFalse(replace(preflight, capability_flags={}).validate_for_execution(
            "run-1", "http://127.0.0.1:7900/actions",
        ))
        self.assertFalse(replace(preflight, installation_evidence=forged_evidence).validate_for_execution(
            "run-1", "http://127.0.0.1:7900/actions",
        ))

    def test_loopback_endpoint_rejects_port_zero(self) -> None:
        self.assertIsNone(normalize_loopback_endpoint("http://127.0.0.1:0/actions"))

if __name__ == "__main__":
    unittest.main()
