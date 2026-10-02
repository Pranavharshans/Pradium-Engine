"""Regression tests for cache fairness, deadlines and measurement boundaries."""
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.adapters.base import StreamToken
from benchmarks.adapters.mock import MockRuntime, MockRuntimeConfig
from benchmarks.runners.base import execute_request, run_group
from benchmarks.runners.batching import run_batching
from benchmarks.runners.prefix_cache import run_prefix_cache
from benchmarks.runners.prompts import PromptProvider
from benchmarks.tests.test_runners import make_ctx, make_request


class TestMeasurementRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_ctx(self.tmp.name, MockRuntime(MockRuntimeConfig(time_scale=0)))
        self.provider = PromptProvider.from_spec('simple')

    def tearDown(self):
        self.ctx.telemetry.stop()
        self.tmp.cleanup()

    def test_text_fallback_round_trips(self):
        request = self.provider.profile_request('SS')
        self.assertEqual(request.prompt_text, self.provider.tokenizer.decode(request.input_ids))

    def test_prefix_prime_has_only_intended_shared_prefix(self):
        seen = []
        def capture(ctx, request, **kwargs):
            seen.append(request)
            return SimpleNamespace(prefix_reused_tokens=None)
        with patch('benchmarks.runners.prefix_cache.execute_request', capture):
            run_prefix_cache(self.ctx, self.provider, ratios=[0.0, .5, 1.0],
                             scenarios=['cross_request'], requests_per_ratio=2,
                             measured_runs=1)
        for prime, measured in zip(seen[::2], seen[1::2]):
            common = 0
            for a, b in zip(prime.input_ids, measured.input_ids):
                if a != b:
                    break
                common += 1
            self.assertEqual(common, measured.prefix_token_count)
            self.assertNotEqual(prime.input_ids, measured.input_ids)

    def test_cache_cleared_between_ordinary_groups(self):
        with patch.object(self.ctx.adapter, 'reset_cache') as clear:
            run_group(self.ctx, [make_request()])
            run_group(self.ctx, [make_request()])
        self.assertEqual(clear.call_count, 2)

    def test_trace_io_excluded_from_concurrent_wall_clock(self):
        original = self.ctx.session.write_trace
        def slow(trace):
            time.sleep(.05)
            return original(trace)
        with patch.object(self.ctx.session, 'write_trace', slow):
            records, aggregate = run_group(self.ctx, [make_request(), make_request()])
        expected = (max(r.last_token_ns for r in records) -
                    min(r.request_submitted_ns for r in records)) / 1e6
        self.assertEqual(aggregate['wall_clock_ms'], expected)

    def test_blocked_stream_times_out_and_quarantines_runtime(self):
        release = threading.Event()
        exited = threading.Event()
        def blocked(request):
            try:
                release.wait()
                yield StreamToken(None, 0, done=True)
            finally:
                exited.set()
        self.ctx.timeout_s = .01
        try:
            with patch.object(self.ctx.adapter, 'generate_stream', side_effect=blocked) as generate:
                started = time.monotonic()
                result = execute_request(self.ctx, make_request())
                self.assertLess(time.monotonic() - started, .5)
                self.assertEqual(result.execution_status, 'TIMEOUT')
                again = execute_request(self.ctx, make_request())
                self.assertEqual(again.execution_status, 'TIMEOUT')
                self.assertEqual(generate.call_count, 1)
        finally:
            release.set()
            self.assertTrue(exited.wait(1))

    def test_done_only_after_deadline_is_timeout(self):
        def delayed(request):
            time.sleep(.03)
            yield StreamToken(None, 0, done=True)
        self.ctx.timeout_s = .005
        with patch.object(self.ctx.adapter, 'generate_stream', delayed):
            result = execute_request(self.ctx, make_request())
        self.assertEqual(result.execution_status, 'TIMEOUT')
        time.sleep(.04)  # Let the adapter thread leave before temporary cleanup.

    def test_overproduction_is_performance_invalid(self):
        self.ctx.performance_valid = True
        with patch.object(self.ctx.adapter, 'generate_stream', return_value=iter([
            StreamToken(1, 0), StreamToken(2, 1), StreamToken(None, 2, done=True)])):
            result = execute_request(self.ctx, make_request(out=1))
        self.assertEqual(result.correctness_status, 'FAIL')
        self.assertFalse(result.performance_valid)

    def test_incomplete_stream_is_failed(self):
        with patch.object(self.ctx.adapter, 'generate_stream', return_value=iter([
            StreamToken(1, 0), StreamToken(2, 1)])):
            result = execute_request(self.ctx, make_request(out=2))
        self.assertEqual(result.execution_status, 'FAILED')
        self.assertFalse(result.performance_valid)

    def test_missing_batch_results_preserve_all_failures(self):
        self.ctx.suite = 'batching'
        with patch.object(self.ctx.adapter, 'generate_batch', return_value=[]):
            results, _ = run_batching(self.ctx, self.provider, profile='SS',
                                     batch_sizes=[2], warmups=0, measured_runs=1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.execution_status == 'FAILED' for r in results))
        self.assertEqual(self.ctx.session.raw_count, 2)
        self.assertEqual(len(self.ctx.session.failures_path.read_text().splitlines()), 2)
