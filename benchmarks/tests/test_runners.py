"""Collection engine tests: raw records, failures, traces, concurrency, mock."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from benchmarks.adapters.mock import MockFailureSpec, MockRuntime, MockRuntimeConfig
from benchmarks.core.config import load_config
from benchmarks.core.request import BenchmarkRequest, new_request_id
from benchmarks.core.results import (
    CORRECT_FAIL,
    CORRECT_PASS,
    CORRECT_SKIPPED,
    CORRECT_WARNING,
    EXEC_FAILED,
    EXEC_OOM,
    EXEC_SUCCESS,
    EXEC_TIMEOUT,
)
from benchmarks.core.session import Session, SessionError
from benchmarks.runners.base import RunnerContext, execute_request, run_group
from benchmarks.telemetry.sampler import TelemetrySampler

CONFIG = load_config()


def make_ctx(tmp: str, runtime: MockRuntime) -> RunnerContext:
    session = Session("20261002-000000_mock_validation_unit", tmp, "mock", "validation", "quick")
    session.write_environment({"benchmark_version": "PRADIUM-RUNTIME-BENCH-v2"})
    telemetry = TelemetrySampler(interval_ms=5, enable_gpu=False)
    telemetry.start()
    runtime.initialize()
    runtime.load_model()
    return RunnerContext(
        adapter=runtime,
        session=session,
        telemetry=telemetry,
        config=CONFIG,
        suite="matrix",
        benchmark_mode="validation",
        run_policy="quick",
        performance_valid=False,
        environment={},
    )


def make_request(prompt: int = 64, out: int = 8, **kwargs) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=new_request_id(),
        input_ids=list(range(prompt)),
        prompt_token_count=prompt,
        max_new_tokens=out,
        **kwargs,
    )


class TestCollection(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_raw_record_and_trace_persisted(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(time_scale=0.05))
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(64, 8, profile="SS"))
        self.assertEqual(record.execution_status, EXEC_SUCCESS)
        self.assertEqual(record.actual_output_tokens, 8)
        self.assertEqual(record.correctness_status, CORRECT_PASS)
        self.assertIsNotNone(record.ttft_ms)
        self.assertEqual(record.metric_provenance["prefill_ms"], "REPORTED_RUNTIME")
        self.assertEqual(record.metric_provenance["queue_ms"], "REPORTED_RUNTIME")
        self.assertEqual(record.metric_provenance["scheduler_ms"], "UNAVAILABLE")
        self.assertFalse(record.performance_valid)

        lines = (Path(ctx.session.raw_path)).read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["benchmark_version"], "PRADIUM-RUNTIME-BENCH-v2")
        self.assertEqual(payload["execution_status"], "SUCCESS")

        trace_files = list((Path(ctx.session.directory) / "traces").glob("*.json"))
        self.assertEqual(len(trace_files), 1)
        trace = json.loads(trace_files[0].read_text())
        self.assertEqual(len(trace["tokens"]), 8)
        self.assertIn("relative_timestamp_ns", trace["tokens"][0])

    def test_deterministic_mock_output(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(time_scale=0.0))
        ctx = make_ctx(self.tmp, runtime)
        request = make_request(16, 4)
        first = execute_request(ctx, request).generated_token_ids
        second = execute_request(ctx, request).generated_token_ids
        self.assertEqual(first, second)

    def test_oom_recorded_not_fatal(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(time_scale=0.0, oom_above_input_tokens=100)
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(512, 4))
        self.assertEqual(record.execution_status, EXEC_OOM)
        self.assertEqual(record.correctness_status, CORRECT_SKIPPED)
        self.assertIn("OOM", record.error)
        failures = [
            json.loads(line)
            for line in Path(ctx.session.failures_path).read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(failures[0]["status"], "OOM")
        self.assertEqual(failures[0]["input_tokens"], 512)
        # suite continues afterwards
        ok = execute_request(ctx, make_request(32, 4))
        self.assertEqual(ok.execution_status, EXEC_SUCCESS)

    def test_exception_recorded(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(
                time_scale=0.0, failures=[MockFailureSpec("exception", {"profile": "SS"})]
            )
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(32, 4, profile="SS"))
        self.assertEqual(record.execution_status, EXEC_FAILED)
        self.assertIn("simulated runtime exception", record.error)

    def test_timeout_recorded(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(
                time_scale=0.0,
                failures=[MockFailureSpec("timeout", {"profile": "SS"}, after_tokens=2)],
            )
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(32, 8, profile="SS"))
        self.assertEqual(record.execution_status, EXEC_TIMEOUT)

    def test_early_eos_recorded_as_warning(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(
                time_scale=0.0,
                failures=[MockFailureSpec("early_eos", {"profile": "MM"}, after_tokens=3)],
            )
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(64, 16, profile="MM"))
        self.assertEqual(record.execution_status, EXEC_SUCCESS)
        self.assertTrue(record.early_termination)
        self.assertEqual(record.actual_output_tokens, 3)
        self.assertEqual(record.correctness_status, CORRECT_WARNING)

    def test_empty_output_is_correctness_fail(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(
                time_scale=0.0,
                failures=[MockFailureSpec("early_eos", {"profile": "SS"}, after_tokens=0)],
            )
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(ctx, make_request(32, 8, profile="SS"))
        self.assertEqual(record.execution_status, EXEC_SUCCESS)
        self.assertEqual(record.correctness_status, CORRECT_FAIL)
        self.assertIn("empty output", record.error)


class TestConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_c4_overlap_and_aggregate(self) -> None:
        # 8 tokens at 100 tok/s -> ~70 ms per request, enough to overlap.
        runtime = MockRuntime(
            MockRuntimeConfig(time_scale=1.0, decode_tok_s=100.0, ttft_overhead_ms=1.0)
        )
        ctx = make_ctx(self.tmp, runtime)
        requests = [make_request(32, 8, profile="SS") for _ in range(4)]
        results, aggregates = run_group(ctx, requests, concurrency=4)

        self.assertEqual([r.execution_status for r in results], [EXEC_SUCCESS] * 4)

        # All four requests genuinely overlap in time.
        spans = sorted(
            (r.request_submitted_ns, r.last_token_ns) for r in results
        )
        max_start = max(s for s, _ in spans)
        min_end = min(e for _, e in spans)
        self.assertLess(max_start, min_end, "C4 requests did not overlap")

        # Individual timing stays intact.
        for record in results:
            self.assertIsNotNone(record.ttft_ms)
            self.assertEqual(record.actual_output_tokens, 8)
            self.assertIsNotNone(record.decode_tok_s)

        # Aggregate throughput uses the global wall-clock window.
        total_output = sum(r.actual_output_tokens for r in results)
        wall_s = aggregates["wall_clock_ms"] / 1000.0
        self.assertGreater(wall_s, 0)
        self.assertAlmostEqual(
            aggregates["aggregate_output_tok_s"], total_output / wall_s, places=6
        )
        self.assertEqual(aggregates["group_requests"], 4)
        # concurrent wall clock is much smaller than the sum of per-request times
        per_request_sum_s = sum(r.e2e_ms for r in results) / 1000.0
        self.assertLess(wall_s, per_request_sum_s)

    def test_group_aggregates_persisted_in_raw_jsonl(self) -> None:
        import json
        from pathlib import Path

        runtime = MockRuntime(MockRuntimeConfig(time_scale=0.05, decode_tok_s=200.0))
        ctx = make_ctx(self.tmp, runtime)
        requests = [make_request(16, 6) for _ in range(2)]
        run_group(ctx, requests, concurrency=2)
        lines = Path(ctx.session.raw_path).read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            payload = json.loads(line)
            self.assertIsNotNone(payload["aggregate_output_tok_s"])
            self.assertIsNotNone(payload["aggregate_decode_tok_s"])
            self.assertIsNotNone(payload["wall_clock_ms"])

    def test_barrier_thread_mechanics(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(time_scale=0.5, decode_tok_s=200.0))
        ctx = make_ctx(self.tmp, runtime)
        requests = [make_request(16, 6) for _ in range(3)]
        lock = threading.Lock()
        gate = threading.Event()
        records = []

        def worker(req):
            record = execute_request(ctx, req, gate=gate, concurrency=3)
            with lock:
                records.append(record)

        threads = [threading.Thread(target=worker, args=(r,)) for r in requests]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join()
        self.assertEqual(len(records), 3)
        submits = [r.request_submitted_ns for r in records]
        self.assertLess(max(submits) - min(submits), 50 * 1_000_000)


class TestMockPrefixCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_prefix_cache_speeds_up_prefill(self) -> None:
        runtime = MockRuntime(MockRuntimeConfig(time_scale=1.0, prefill_tok_s=2000.0))
        ctx = make_ctx(self.tmp, runtime)
        first = execute_request(
            ctx,
            make_request(
                512, 2, prefix_id="p1", prefix_token_count=448, profile="MM"
            ),
        )
        second = execute_request(
            ctx,
            make_request(
                512, 2, prefix_id="p1", prefix_token_count=448, profile="MM"
            ),
        )
        self.assertGreater(first.prefill_ms, 0)
        self.assertGreater(second.prefill_ms, 0)
        self.assertLess(second.prefill_ms, first.prefill_ms)
        self.assertEqual(second.prefix_reused_tokens, 448)
        self.assertEqual(first.prefix_reused_tokens, 0)
        self.assertIsNotNone(second.cache_hit_rate)

    def test_prefix_cache_unavailable_when_disabled(self) -> None:
        runtime = MockRuntime(
            MockRuntimeConfig(time_scale=0.0, prefix_cache_enabled=False)
        )
        ctx = make_ctx(self.tmp, runtime)
        record = execute_request(
            ctx,
            make_request(512, 2, prefix_id="p1", prefix_token_count=448, profile="MM"),
        )
        self.assertEqual(record.prefix_reused_tokens, 0)
        self.assertIsNone(record.cache_hit_rate)


class TestSessionSafety(unittest.TestCase):
    def test_no_silent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Session("fixed-id", tmp, "mock", "validation", "quick")
            session.close()
            with self.assertRaises(SessionError):
                Session("fixed-id", tmp, "mock", "validation", "quick")


if __name__ == "__main__":
    unittest.main()
