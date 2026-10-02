"""Suite runner tests (fast mock configurations)."""

from __future__ import annotations

import tempfile
import unittest

from benchmarks.adapters.mock import MockRuntime, MockRuntimeConfig
from benchmarks.core.config import load_config
from benchmarks.core.results import EXEC_OOM, EXEC_SUCCESS, EXEC_UNSUPPORTED
from benchmarks.core.session import Session
from benchmarks.runners.base import RunnerContext
from benchmarks.runners.batching import run_batching
from benchmarks.runners.concurrency import run_concurrency
from benchmarks.runners.context import run_context
from benchmarks.runners.matrix import run_matrix
from benchmarks.runners.mixed import run_mixed
from benchmarks.runners.prefix_cache import run_prefix_cache
from benchmarks.runners.prompts import PromptProvider
from benchmarks.runners.scheduler import run_scheduler
from benchmarks.runners.soak import run_soak
from benchmarks.runners.startup import run_startup
from benchmarks.telemetry.sampler import TelemetrySampler

CONFIG = load_config()


def make_ctx(tmp: str, runtime: MockRuntime, suite: str = "matrix") -> RunnerContext:
    session = Session(
        f"20261002-000000_mock_validation_{suite}", tmp, "mock", "validation", "quick"
    )
    session.write_environment({"benchmark_version": "PRADIUM-RUNTIME-BENCH-v1"})
    telemetry = TelemetrySampler(interval_ms=5, enable_gpu=False)
    telemetry.start()
    runtime.initialize()
    runtime.load_model()
    return RunnerContext(
        adapter=runtime,
        session=session,
        telemetry=telemetry,
        config=CONFIG,
        suite=suite,
        benchmark_mode="validation",
        run_policy="quick",
        performance_valid=False,
        environment={},
    )


FAST = MockRuntimeConfig(time_scale=0.01, decode_tok_s=2000.0, prefill_tok_s=200_000.0)


class TestMatrixRunner(unittest.TestCase):
    def test_all_profiles_with_exact_lengths(self) -> None:
        from benchmarks.core.workloads import PROFILES, PROFILE_ORDER

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "matrix")
            provider = PromptProvider.from_spec("simple")
            records = run_matrix(
                ctx, provider, profiles=PROFILE_ORDER, warmups=1, measured_runs=1
            )
            measured = [r for r in records if not r.warmup]
            self.assertEqual(len(measured), 9)
            for record in measured:
                self.assertEqual(record.execution_status, EXEC_SUCCESS)
                profile = PROFILES[record.profile]
                self.assertEqual(record.prompt_tokens, profile.prompt_tokens)
                self.assertEqual(record.requested_output_tokens, profile.output_tokens)
                self.assertEqual(record.actual_output_tokens, profile.output_tokens)
            warmups = [r for r in records if r.warmup]
            self.assertEqual(len(warmups), 9)


class TestContextRunner(unittest.TestCase):
    def test_oom_recorded_and_suite_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MockRuntime(
                MockRuntimeConfig(
                    time_scale=0.0,
                    oom_above_input_tokens=4096,
                    prefill_tok_s=200_000.0,
                )
            )
            ctx = make_ctx(tmp, runtime, "context")
            provider = PromptProvider.from_spec("simple")
            records = run_context(
                ctx,
                provider,
                input_tokens=(128, 1024, 8192),
                warmups=0,
                measured_runs=1,
            )
            statuses = {r.profile: r.execution_status for r in records}
            self.assertEqual(statuses["CTX-128"], EXEC_SUCCESS)
            self.assertEqual(statuses["CTX-1024"], EXEC_SUCCESS)
            self.assertEqual(statuses["CTX-8192"], EXEC_OOM)
            self.assertEqual(len(records), 3)  # nothing silently dropped


class TestConcurrencyRunner(unittest.TestCase):
    def test_levels_and_group_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "concurrency")
            provider = PromptProvider.from_spec("simple")
            records = run_concurrency(
                ctx,
                provider,
                profiles=("SS",),
                levels=(1, 2, 4),
                warmups=0,
                measured_runs=1,
            )
            by_conc = {}
            for record in records:
                by_conc.setdefault(record.concurrency, 0)
                by_conc[record.concurrency] += 1
            self.assertEqual(by_conc, {1: 1, 2: 2, 4: 4})
            for record in records:
                self.assertIsNotNone(record.aggregate_output_tok_s)


class TestPrefixCacheRunner(unittest.TestCase):
    def test_reuse_measured_per_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "prefix-cache")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_prefix_cache(
                ctx,
                provider,
                ratios=(0.5, 1.0),
                scenarios=("same_session", "cross_session"),
                requests_per_ratio=2,
                output_tokens=4,
                measured_runs=1,
            )
            measured = [r for r in records if not r.warmup]
            self.assertEqual(len(measured), 8)
            same = extras["prefix_cache"]["1.00:same_session"]
            cross = extras["prefix_cache"]["1.00:cross_session"]
            self.assertGreater(same["reused_tokens"][0], 0)
            # after a session boundary the mock cache is cold: no reuse
            self.assertEqual(cross["reused_tokens"][0], 0)
            self.assertEqual(same["shared_prefix_tokens"], 1008)


class TestSchedulerRunner(unittest.TestCase):
    def test_interference_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MockRuntime(
                MockRuntimeConfig(time_scale=0.05, decode_tok_s=500.0, prefill_tok_s=20_000.0)
            )
            ctx = make_ctx(tmp, runtime, "scheduler")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_scheduler(
                ctx,
                provider,
                background_profile="ML",
                intruder_profile="LS",
                background_requests=3,
                intruder_delay_ms=20,
                output_scale=0.05,
            )
            self.assertEqual(len(records), 4)
            block = extras["scheduler_interference"]
            self.assertEqual(block["status"], "OK")
            self.assertIsNotNone(block["intruder_ttft_ms"])
            self.assertIsNotNone(block["max_output_token_stall_ms"])
            self.assertEqual(len(block["per_request"]), 3)
            self.assertGreater(
                block["itl_before_samples"] + block["itl_during_samples"] + block["itl_after_samples"],
                0,
            )


class TestBatchingRunner(unittest.TestCase):
    def test_unsupported_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "batching")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_batching(
                ctx, provider, profile="SS", batch_sizes=(2,), warmups=0, measured_runs=1
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.execution_status == EXEC_UNSUPPORTED for r in records))
            self.assertFalse(extras["batching"]["2"]["supported"])

    def test_supported_batch_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MockRuntime(
                MockRuntimeConfig(time_scale=0.0, static_batching=True)
            )
            ctx = make_ctx(tmp, runtime, "batching")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_batching(
                ctx, provider, profile="SS", batch_sizes=(2,), warmups=0, measured_runs=1
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.execution_status == EXEC_SUCCESS for r in records))
            self.assertTrue(extras["batching"]["2"]["supported"])
            for record in records:
                self.assertIn("batch_formation_ms", record.internal_metrics)


class TestStartupRunner(unittest.TestCase):
    def test_milestones_and_cold_warm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "startup")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_startup(
                ctx, provider, request_profile="SS", warm_request_count=2
            )
            milestones = extras["startup"]
            self.assertEqual(len(records), 3)  # 1 cold + 2 warm
            self.assertIsNotNone(milestones["runtime_init_ms"])
            self.assertIsNotNone(milestones["model_load_ms"])
            self.assertIsNotNone(milestones["process_start_to_ready_ms"])
            self.assertIsNotNone(milestones["cold_first_request_ttft_ms"])
            self.assertIsNotNone(milestones["warm_request_ttft_ms_median"])
            cold = [r for r in records if r.metadata.get("phase") == "cold"]
            warm = [r for r in records if r.metadata.get("phase") == "warm"]
            self.assertEqual(len(cold), 1)
            self.assertEqual(len(warm), 2)


class TestMixedAndSoak(unittest.TestCase):
    def test_mixed_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "mixed")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_mixed(
                ctx,
                provider,
                workloads=("interactive_burst", "mixed_generation"),
                warmups=0,
                measured_runs=1,
            )
            self.assertEqual(len(records), 6 + 3)
            self.assertIn("interactive_burst", extras["mixed"])
            for record in records:
                self.assertIsNotNone(record.request_id)

    def test_soak_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, MockRuntime(FAST), "soak")
            provider = PromptProvider.from_spec("simple")
            records, extras = run_soak(
                ctx,
                provider,
                duration_s=0.2,
                profiles=("SS",),
                concurrency=1,
            )
            self.assertGreaterEqual(len(records), 1)
            block = extras["soak"]
            self.assertGreaterEqual(block["iterations"], 1)
            self.assertIn("analysis", block)


if __name__ == "__main__":
    unittest.main()
