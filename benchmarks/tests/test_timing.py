"""Timing-metric tests using deterministic fake timestamps (no real sleeps)."""

from __future__ import annotations

import unittest

from benchmarks.core.timing import (
    TokenEvent,
    compute_request_timing,
    degradation_ratio,
    group_aggregate_throughput,
    percentile,
    prefill_metrics,
    scaling_efficiency,
)

MS = 1_000_000  # ns per ms


def _events(timestamps_ms: list[float], start_id: int = 10) -> list[TokenEvent]:
    return [
        TokenEvent(i, start_id + i, int(t * MS)) for i, t in enumerate(timestamps_ms)
    ]


class TestRequestTiming(unittest.TestCase):
    def test_reference_scenario(self) -> None:
        # submit = 0 ms; first token 100 ms; then 120/140/160/180 ms.
        events = _events([100, 120, 140, 160, 180])
        m = compute_request_timing(0, events)
        self.assertEqual(m.ttft_ms, 100.0)
        self.assertEqual(m.e2e_ms, 180.0)
        self.assertEqual(m.decode_ms, 80.0)
        self.assertAlmostEqual(m.decode_tok_s, 4 / 0.08)
        self.assertAlmostEqual(m.tpot_ms, 20.0)
        self.assertEqual(m.itl_mean_ms, 20.0)
        self.assertEqual(m.itl_median_ms, 20.0)
        self.assertEqual(m.itl_p95_ms, 20.0)
        self.assertEqual(m.itl_p99_ms, 20.0)
        self.assertEqual(m.itl_max_ms, 20.0)
        self.assertEqual(m.itl_stdev_ms, 0.0)
        self.assertEqual(len(m.itl_samples_ms), 4)

    def test_ttft_offsets_submit(self) -> None:
        events = _events([500, 520])
        m = compute_request_timing(int(200 * MS), events)
        self.assertEqual(m.ttft_ms, 300.0)
        self.assertEqual(m.e2e_ms, 320.0)

    def test_single_token(self) -> None:
        events = _events([100])
        m = compute_request_timing(0, events)
        self.assertEqual(m.ttft_ms, 100.0)
        self.assertIsNone(m.decode_ms)
        self.assertIsNone(m.decode_tok_s)
        self.assertIsNone(m.tpot_ms)
        self.assertEqual(m.itl_mean_ms, None)

    def test_no_tokens(self) -> None:
        m = compute_request_timing(0, [])
        self.assertIsNone(m.ttft_ms)
        self.assertIsNone(m.e2e_ms)
        self.assertEqual(m.token_count, 0)

    def test_uneven_itl(self) -> None:
        events = _events([100, 110, 140, 340])
        m = compute_request_timing(0, events)
        self.assertEqual(m.itl_samples_ms, [10.0, 30.0, 200.0])
        self.assertEqual(m.itl_max_ms, 200.0)
        self.assertEqual(m.itl_min_ms, 10.0)
        self.assertAlmostEqual(m.itl_mean_ms, 80.0)
        self.assertAlmostEqual(m.tpot_ms, 240.0 / 3)
        self.assertAlmostEqual(m.decode_tok_s, 3 / 0.24)

    def test_output_tok_s_per_request(self) -> None:
        events = _events([100, 120, 140, 160, 180])
        m = compute_request_timing(0, events)
        self.assertAlmostEqual(m.output_tok_s_per_request, 5 / 0.18)


class TestPercentiles(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(percentile([], 50), None)
        self.assertEqual(percentile([5], 99), 5)
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 0), 1)
        self.assertEqual(percentile([1, 2, 3, 4], 100), 4)

    def test_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            percentile([1], 101)


class TestDerived(unittest.TestCase):
    def test_prefill_unavailable(self) -> None:
        self.assertEqual(prefill_metrics(None, 1024), (None, None, "UNAVAILABLE"))
        self.assertEqual(prefill_metrics(0.0, 1024), (None, None, "UNAVAILABLE"))

    def test_prefill_reported(self) -> None:
        ms, tok_s, prov = prefill_metrics(2048.0, 1024)
        self.assertEqual(prov, "REPORTED_RUNTIME")
        self.assertEqual(ms, 2048.0)
        self.assertAlmostEqual(tok_s, 500.0)

    def test_group_aggregate_throughput(self) -> None:
        agg = group_aggregate_throughput(1000, 500, 2 * 1_000_000_000, 4)
        self.assertAlmostEqual(agg["aggregate_output_tok_s"], 500.0)
        self.assertAlmostEqual(agg["aggregate_prompt_tok_s"], 250.0)
        self.assertAlmostEqual(agg["aggregate_total_tok_s"], 750.0)
        self.assertAlmostEqual(agg["requests_per_second"], 2.0)
        self.assertAlmostEqual(agg["completed_requests_per_minute"], 120.0)

    def test_scaling_efficiency(self) -> None:
        self.assertAlmostEqual(scaling_efficiency(400.0, 4, 100.0), 1.0)
        self.assertAlmostEqual(scaling_efficiency(200.0, 4, 100.0), 0.5)
        self.assertIsNone(scaling_efficiency(None, 4, 100.0))
        self.assertIsNone(scaling_efficiency(400.0, 4, 0.0))

    def test_degradation(self) -> None:
        self.assertAlmostEqual(degradation_ratio(150.0, 100.0), 0.5)
        self.assertAlmostEqual(degradation_ratio(50.0, 100.0), -0.5)
        self.assertIsNone(degradation_ratio(None, 100.0))
        self.assertIsNone(degradation_ratio(50.0, 0.0))


if __name__ == "__main__":
    unittest.main()
