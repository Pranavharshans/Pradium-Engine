"""Telemetry collection tests, including failure handling."""

from __future__ import annotations

import time
import unittest

from benchmarks.telemetry.cpu import CpuMonitor
from benchmarks.telemetry.gpu import GpuMonitor
from benchmarks.telemetry.memory import MemoryMonitor
from benchmarks.telemetry.sampler import TelemetrySampler


class TestGpuTelemetry(unittest.TestCase):
    def test_probe_never_raises(self) -> None:
        from benchmarks.telemetry.gpu import probe_gpus

        gpus = probe_gpus()
        self.assertIsInstance(gpus, list)

    def test_disabled_monitor(self) -> None:
        monitor = GpuMonitor(enabled=False)
        self.assertFalse(monitor.available)
        self.assertIsNone(monitor.sample(0))
        self.assertEqual(monitor.info()["backend"], "disabled")


class TestCpuTelemetry(unittest.TestCase):
    def test_cpu_monitor(self) -> None:
        monitor = CpuMonitor()
        sample = monitor.sample(time.perf_counter_ns())
        self.assertIsNotNone(sample.process_cpu_time_s)
        self.assertGreaterEqual(sample.process_cpu_time_s, 0.0)
        self.assertGreaterEqual(sample.thread_count or 1, 1)

    def test_cpu_pct_between_samples(self) -> None:
        monitor = CpuMonitor()
        t1 = time.perf_counter_ns()
        monitor.sample(t1)
        # burn a little cpu
        x = 0
        for i in range(200_000):
            x += i * i
        sample = monitor.sample(time.perf_counter_ns())
        self.assertIsNotNone(sample.process_cpu_pct)
        self.assertGreaterEqual(sample.process_cpu_pct, 0.0)


class TestMemoryTelemetry(unittest.TestCase):
    def test_memory_snapshot(self) -> None:
        monitor = MemoryMonitor()
        snapshot = monitor.snapshot_mb()
        self.assertIsNotNone(snapshot["process_rss_peak_mb"])
        self.assertGreater(snapshot["process_rss_peak_mb"], 0)


class TestSampler(unittest.TestCase):
    def test_sampling_and_window_stats(self) -> None:
        sampler = TelemetrySampler(interval_ms=5, enable_gpu=False)
        sampler.start()
        try:
            time.sleep(0.05)
        finally:
            sampler.stop()
        stats = sampler.window_stats(0, time.perf_counter_ns() + 10**12)
        self.assertGreaterEqual(stats.sample_count, 2)
        self.assertIsNotNone(stats.get("process_rss_peak_mb", "max"))
        self.assertIsNone(stats.get("util_gpu_pct", "avg"))  # no gpu: unavailable
        info = sampler.info()
        self.assertEqual(info["gpu"]["backend"], "disabled")
        self.assertIsNotNone(info["overhead_per_sample_ms"])

    def test_window_excludes_outside_samples(self) -> None:
        sampler = TelemetrySampler(interval_ms=5, enable_gpu=False)
        before = sampler.sample_once()["timestamp_ns"]
        time.sleep(0.02)
        after = sampler.sample_once()["timestamp_ns"]
        stats = sampler.window_stats(before, before)  # only the first sample
        self.assertGreaterEqual(stats.sample_count, 1)
        self.assertGreaterEqual(
            sampler.window_stats(0, after + 10**12).sample_count,
            stats.sample_count,
        )

    def test_telemetry_failure_does_not_crash(self) -> None:
        sampler = TelemetrySampler(interval_ms=5, enable_gpu=False)

        def boom(ts):  # simulate a failing monitor
            raise RuntimeError("telemetry backend died")

        sampler.gpu.sample = boom  # type: ignore[method-assign]
        sampler.gpu.available = True
        record = sampler.sample_once()
        self.assertIsNone(record["gpu"])
        self.assertTrue(any("gpu" in e for e in sampler.errors()))
        # other telemetry still collected
        self.assertIsNotNone(record["cpu"])
        self.assertIsNotNone(record["memory"])


if __name__ == "__main__":
    unittest.main()
