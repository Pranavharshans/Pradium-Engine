"""ExLlamaV3 adapter conformance and EXL3-integrity tests.

These tests need a real CUDA GPU, the pinned ExLlamaV3 wheel and the frozen
EXL3 checkpoint, so they are skipped unless ``PRADIUM_EXLLAMAV3_MODEL`` points
at the checkpoint directory (and ``torch``/``exllamav3`` import successfully).

Run on the benchmark VM with:

    PRADIUM_EXLLAMAV3_MODEL=/workspace/models/minicpm5-2b-exl3 \
        python -m pytest benchmarks/tests/test_adapter_exllamav3.py -q
"""

from __future__ import annotations

import os
import unittest

from benchmarks.adapters.base import RuntimeAdapterError
from benchmarks.core.request import BenchmarkRequest, new_request_id
from benchmarks.tests.conformance import (
    make_conformance_request,
    run_conformance_suite,
)

MODEL_DIR = os.environ.get("PRADIUM_EXLLAMAV3_MODEL", "")
MAX_BATCH_SIZE = int(os.environ.get("PRADIUM_EXLLAMAV3_BATCH", "16"))
#: KV cache capacity used by the campaign configuration.
MAX_NUM_TOKENS = int(os.environ.get("PRADIUM_EXLLAMAV3_KV_TOKENS", "32768"))


def _runtime_available() -> tuple[bool, str]:
    if not MODEL_DIR:
        return False, "PRADIUM_EXLLAMAV3_MODEL is not set"
    if not os.path.isdir(MODEL_DIR):
        return False, f"model directory does not exist: {MODEL_DIR}"
    try:
        import torch  # noqa: F401,PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, f"torch unavailable: {exc!r}"
    try:
        import exllamav3  # noqa: F401,PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, f"exllamav3 unavailable: {exc!r}"
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False, "no CUDA device"
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA probe failed: {exc!r}"
    return True, ""


_AVAILABLE, _REASON = _runtime_available()


@unittest.skipUnless(_AVAILABLE, f"exllamav3 runtime unavailable: {_REASON}")
class TestExLlamaV3Adapter(unittest.TestCase):
    """Behavioural conformance plus EXL3 integrity checks."""

    adapter = None
    caps = None

    @classmethod
    def setUpClass(cls) -> None:
        from benchmarks.adapters.exllamav3 import ExLlamaV3RuntimeConfig, build_exllamav3_runtime

        config = ExLlamaV3RuntimeConfig(
            model_dir=MODEL_DIR,
            max_num_tokens=MAX_NUM_TOKENS,
            max_batch_size=MAX_BATCH_SIZE,
            model="ewin-reg/MiniCPM5-2B-EXL3-Quantized",
        )
        cls.adapter = build_exllamav3_runtime(config=config)
        cls.adapter.initialize()
        cls.adapter.load_model()
        cls.caps = cls.adapter.get_capabilities()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.adapter is not None:
            cls.adapter.unload_model()
            cls.adapter.shutdown()

    # --- generic conformance (shared with every runtime) -------------------
    def test_conformance_suite(self) -> None:
        checks = run_conformance_suite(self.adapter)
        failures = [c for c in checks if not c.passed]
        self.assertEqual(failures, [], "\n".join(f"{c.name}: {c.detail}" for c in failures))

    def test_fixed_length_is_exact(self) -> None:
        request = make_conformance_request(24, 17)
        result = self.adapter.generate(request)
        self.assertEqual(len(result.token_ids), 17)
        for token_id in result.token_ids:
            self.assertIsInstance(token_id, int)

    def test_greedy_is_deterministic(self) -> None:
        """Official comparisons need temperature-0 determinism across runs."""
        first = self.adapter.generate(
            BenchmarkRequest(
                request_id=new_request_id("det"),
                input_ids=list(range(200, 232)),
                prompt_token_count=32,
                max_new_tokens=24,
                greedy=True,
                temperature=0.0,
            )
        )
        second = self.adapter.generate(
            BenchmarkRequest(
                request_id=new_request_id("det"),
                input_ids=list(range(200, 232)),
                prompt_token_count=32,
                max_new_tokens=24,
                greedy=True,
                temperature=0.0,
            )
        )
        self.assertEqual(first.token_ids, second.token_ids)

    # --- EXL3 integrity (campaign section 20) ------------------------------
    def test_model_runs_as_exl3_not_dequantized(self) -> None:
        audit = self.adapter.get_runtime_info().get("exl3_audit", {})
        self.assertGreater(audit.get("linear_modules", 0), 0, "no Linear modules discovered")
        self.assertTrue(
            audit.get("exl3_only"),
            f"non-EXL3 linear modules present: {audit.get('quant_type_counts')} "
            f"{audit.get('non_exl3_linears')}",
        )

    def test_storage_bitrate_matches_declared_4bpw(self) -> None:
        audit = self.adapter.get_runtime_info().get("exl3_audit", {})
        info = audit.get("storage_info", {})
        self.assertNotIn("error", info, f"storage info unavailable: {info}")
        self.assertAlmostEqual(float(info["bpw_layer"]), 4.0, delta=0.6)

    def test_every_linear_is_exl3(self) -> None:
        """The 42 blocks contribute 7 linears each; none may be fp16."""
        counts = self.adapter.get_runtime_info()["exl3_audit"]["quant_type_counts"]
        self.assertEqual(set(counts), {"exl3"}, f"unexpected quant types: {counts}")
        self.assertGreaterEqual(counts["exl3"], 42 * 7)

    def test_load_fails_hard_when_exl3_cannot_be_verified(self) -> None:
        """A run that cannot prove EXL3 must refuse to load, not report numbers."""
        from benchmarks.adapters.exllamav3 import ExLlamaV3RuntimeConfig, build_exllamav3_runtime

        adapter = build_exllamav3_runtime(
            config=ExLlamaV3RuntimeConfig(model_dir="/nonexistent", require_exl3=True)
        )
        adapter.initialize()
        try:
            with self.assertRaises(RuntimeAdapterError):
                adapter.load_model()
        finally:
            adapter.shutdown()

    # --- KV cache capacity -------------------------------------------------
    def test_kv_cache_capacity_is_the_configured_32k(self) -> None:
        info = self.adapter.get_runtime_info()
        cache = info.get("kv_cache", {})
        self.assertEqual(cache.get("max_num_tokens"), MAX_NUM_TOKENS)
        stats = info.get("cache_stats_at_load") or self.adapter.get_internal_metrics().get(
            "cache_stats", {}
        )
        self.assertEqual(stats.get("page_size"), 256)
        self.assertGreaterEqual(stats.get("max_tokens", 0), MAX_NUM_TOKENS)

    def test_long_prompt_fills_more_than_a_quarter_of_the_cache(self) -> None:
        """A ~9k-token prompt must actually occupy >8k cached tokens (32K pool)."""
        request = BenchmarkRequest(
            request_id=new_request_id("longctx"),
            input_ids=list(range(1000, 1000 + 9000)),
            prompt_token_count=9000,
            max_new_tokens=4,
            greedy=True,
        )
        result = self.adapter.generate(request)
        self.assertEqual(len(result.token_ids), 4)
        stats = self.adapter.get_internal_metrics().get("cache_stats", {})
        # Completed pages become reusable cached pages; the partial tail page is
        # never hashed, so 9000 tokens occupy 8960 cached tokens.
        self.assertGreater(stats.get("cached_tokens", 0), 8192)
        self.assertLessEqual(stats.get("cached_tokens", 0), MAX_NUM_TOKENS)
        self.assertEqual(stats.get("max_tokens"), MAX_NUM_TOKENS)
        self.adapter.reset_cache()

    # --- capabilities ------------------------------------------------------
    def test_capabilities_are_honest(self) -> None:
        self.assertTrue(self.caps.is_supported("supports_streaming"))
        self.assertTrue(self.caps.is_supported("supports_input_ids"))
        self.assertTrue(self.caps.is_supported("supports_fixed_decode_length"))
        self.assertTrue(self.caps.is_supported("supports_continuous_batching"))
        self.assertTrue(self.caps.is_supported("supports_prefix_cache"))
        self.assertFalse(self.caps.is_supported("supports_cuda_graphs"))

    # --- cache control -----------------------------------------------------
    def test_reset_cache_clears_cached_tokens(self) -> None:
        """A cold-start boundary must be observable, not assumed."""
        request = make_conformance_request(4000, 4)
        self.adapter.generate(request)
        warm = self.adapter.get_internal_metrics().get("cache_stats", {})
        self.assertGreater(warm.get("cached_tokens", 0), 0, "no prefix was cached to reset")
        self.adapter.reset_cache()
        cold = self.adapter.get_internal_metrics().get("cache_stats", {})
        self.assertEqual(cold.get("cached_tokens"), 0)

    def test_reset_cache_makes_the_next_prefill_recompute(self) -> None:
        """After a reset the same prompt must be prefilled again (no free reuse).

        ``alloc_cached_pages`` counts pages served from the prompt cache; a new
        page table zeroes the counter, so a first post-reset job that reports
        zero proves the prompt really was recomputed.
        """
        request = make_conformance_request(6000, 4)

        # 1. cold: alloc_cached_pages counters start at zero on a fresh page table
        self.adapter.reset_cache()
        self.adapter.generate(request)
        cold = self.adapter.get_internal_metrics().get("cache_stats", {})
        self.assertEqual(cold.get("alloc_cached_pages"), 0, "first job reused pages it cannot have")
        self.assertGreater(cold.get("cached_tokens", 0), 0, "nothing was cached to reuse")

        # 2. warm: the identical prompt must now be served from the prompt cache
        self.adapter.generate(request)
        warm = self.adapter.get_internal_metrics().get("cache_stats", {})
        self.assertGreater(warm.get("alloc_cached_pages", 0), 0, "warm run reused nothing")

        # 3. reset: the identical prompt must be recomputed from scratch again
        self.adapter.reset_cache()
        self.adapter.generate(request)
        after = self.adapter.get_internal_metrics().get("cache_stats", {})
        self.assertEqual(after.get("alloc_cached_pages"), 0)

    # --- streaming ---------------------------------------------------------
    def test_streaming_token_timing_is_real(self) -> None:
        request = make_conformance_request(64, 16)
        events = list(self.adapter.generate_stream(request))
        self.assertTrue(events[-1].done)
        tokens = [e for e in events if not e.done]
        self.assertEqual(len(tokens), 16)
        self.assertIsNotNone(events[-1].prefill_ms)
        self.assertGreaterEqual(events[-1].prefill_ms, 0.0)

    # --- batching ----------------------------------------------------------
    def test_load_model_is_reentrant(self) -> None:
        """The startup suite re-loads an already-loaded adapter to time a cold start.

        That must replace the model rather than map a second copy and leave a
        second scheduler thread driving the engine.
        """
        request = make_conformance_request(64, 8)
        before = self.adapter.generate(request)
        import torch

        vram_before = torch.cuda.memory_allocated(0)
        self.adapter.load_model()
        after_load_vram = torch.cuda.memory_allocated(0)
        self.assertLess(
            after_load_vram, vram_before * 1.5,
            "re-loading the adapter mapped a second copy of the weights",
        )
        after = self.adapter.generate(make_conformance_request(64, 8))
        self.assertEqual(len(after.token_ids), 8)
        self.assertEqual(len(before.token_ids), 8)

    def test_static_batch_returns_one_result_per_request(self) -> None:
        requests = [
            BenchmarkRequest(
                request_id=new_request_id("b"),
                input_ids=list(range(300, 332)),
                prompt_token_count=32,
                max_new_tokens=8,
                batch_size=4,
            )
            for _ in range(4)
        ]
        results = self.adapter.generate_batch(requests)
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(len(result.token_ids), 8)

    def test_missing_model_dir_is_reported(self) -> None:
        from benchmarks.adapters.exllamav3 import ExLlamaV3RuntimeConfig, build_exllamav3_runtime

        adapter = build_exllamav3_runtime(
            config=ExLlamaV3RuntimeConfig(model_dir="/nonexistent/model/dir")
        )
        adapter.initialize()
        try:
            with self.assertRaises(RuntimeAdapterError):
                adapter.load_model()
        finally:
            adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
