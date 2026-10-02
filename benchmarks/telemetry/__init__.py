"""Telemetry collection: GPU, CPU, host memory, and background sampling.

All telemetry degrades gracefully: when a backend is unavailable (no NVML, no
psutil, no NVIDIA GPU), metrics are recorded as ``None`` with provenance
``UNAVAILABLE`` — never fabricated and never zero-filled.
"""

from .sampler import TelemetrySampler, WindowStats

__all__ = ["TelemetrySampler", "WindowStats"]
