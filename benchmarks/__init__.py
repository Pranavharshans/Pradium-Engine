"""Pradium Runtime Benchmark Suite.

Framework-neutral inference-runtime benchmarking. The benchmark core owns
workloads, prompts, token accounting, telemetry, statistics, result format and
reporting; runtime-specific behavior is implemented only through adapters.
"""

from .core.version import BENCHMARK_VERSION

__all__ = ["BENCHMARK_VERSION"]
