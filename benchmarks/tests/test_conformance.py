"""Adapter conformance tests (mock today; reuse for every future runtime)."""

from __future__ import annotations

import unittest

from benchmarks.adapters.base import CapabilityNotSupported
from benchmarks.adapters.mock import MockRuntime, MockRuntimeConfig
from benchmarks.core.capabilities import CAPABILITY_NAMES, CapabilityStatus
from benchmarks.tests.conformance import (
    make_conformance_request,
    run_conformance_suite,
)


class AdapterConformanceMixin:
    """Mixin for future runtime adapters: override ``make_adapter``."""

    def make_adapter(self):
        raise NotImplementedError

    def test_conformance_suite(self) -> None:
        adapter = self.make_adapter()
        adapter.initialize()
        try:
            adapter.load_model()
            checks = run_conformance_suite(adapter)
            failures = [c for c in checks if not c.passed]
            self.assertEqual(
                failures,
                [],
                "\n".join(f"{c.name}: {c.detail}" for c in failures),
            )
        finally:
            adapter.unload_model()
            adapter.shutdown()

    def test_shutdown_is_clean(self) -> None:
        adapter = self.make_adapter()
        adapter.initialize()
        adapter.load_model()
        adapter.unload_model()
        adapter.shutdown()  # must not raise


class TestMockAdapterConformance(AdapterConformanceMixin, unittest.TestCase):
    def make_adapter(self):
        return MockRuntime(MockRuntimeConfig(time_scale=0.0))


class TestMockCapabilities(unittest.TestCase):
    def test_capability_model(self) -> None:
        runtime = MockRuntime()
        caps = runtime.get_capabilities()
        for name in CAPABILITY_NAMES:
            self.assertIsInstance(caps.status(name), CapabilityStatus)
        self.assertTrue(caps.is_supported("supports_streaming"))
        self.assertTrue(caps.is_supported("supports_fixed_decode_length"))
        self.assertTrue(caps.is_supported("supports_prefix_cache"))
        self.assertTrue(caps.is_supported("supports_cache_reset"))

    def test_prefix_cache_disabled(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(prefix_cache_enabled=False))
        caps = runtime.get_capabilities()
        self.assertEqual(
            caps.status("supports_prefix_cache"), CapabilityStatus.UNSUPPORTED
        )

    def test_static_batching_gate(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(time_scale=0.0))
        runtime.initialize()
        runtime.load_model()
        request = make_conformance_request(8, 2)
        with self.assertRaises(CapabilityNotSupported):
            runtime.generate_batch([request])
        runtime.shutdown()

    def test_memory_info_not_fabricated(self) -> None:
        runtime = MockRuntime()
        self.assertIsNone(runtime.get_memory_info())


if __name__ == "__main__":
    unittest.main()
