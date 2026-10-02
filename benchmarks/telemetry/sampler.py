"""Background telemetry sampler.

One low-overhead background thread polls GPU/CPU/memory at a configurable
interval (default 100 ms; deep telemetry uses a smaller interval). Window
statistics are computed for each request from the samples covering it.

Overhead characterization: :meth:`TelemetrySampler.overhead_per_sample_ms`
measures the cost of one polling cycle so instrumentation overhead is itself
measurable rather than assumed.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .cpu import CpuMonitor
from .gpu import GpuMonitor
from .memory import MemoryMonitor

from ..core.timing import now_ns

DEFAULT_INTERVAL_MS = 100
DEEP_INTERVAL_MS = 25

_GPU_FIELDS = (
    "util_gpu_pct",
    "util_mem_pct",
    "power_w",
    "temperature_c",
    "vram_used_mb",
    "clock_mhz",
    "mem_clock_mhz",
    "pcie_rx_mb_s",
    "pcie_tx_mb_s",
)
_CPU_FIELDS = ("process_cpu_pct", "system_cpu_pct")
_MEM_FIELDS = ("process_rss_mb", "process_rss_peak_mb")


@dataclass
class WindowStats:
    """Aggregated telemetry over one window (e.g. one request)."""

    sample_count: int = 0
    gpu_samples: int = 0
    values: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def get(self, metric: str, stat: str = "avg") -> float | None:
        return self.values.get(metric, {}).get(stat)

    def to_dict(self) -> dict[str, Any]:
        return {"sample_count": self.sample_count, "gpu_samples": self.gpu_samples,
                "values": self.values}


def _aggregate(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


class TelemetrySampler:
    """Collects GPU/CPU/memory telemetry in the background."""

    def __init__(
        self,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        deep: bool = False,
        gpu_index: int = 0,
        enable_gpu: bool = True,
    ) -> None:
        self.interval_s = max(interval_ms, 1) / 1000.0
        self.deep = deep
        self.gpu = GpuMonitor(gpu_index, enabled=enable_gpu)
        self.cpu = CpuMonitor()
        self.memory = MemoryMonitor()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._overhead_ms: float | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="bench-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "TelemetrySampler":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            self.sample_once()
            elapsed = time.perf_counter() - started
            if self._overhead_ms is None:
                self._overhead_ms = elapsed * 1000.0
            else:
                self._overhead_ms = 0.9 * self._overhead_ms + 0.1 * elapsed * 1000.0
            self._stop.wait(max(0.0, self.interval_s - elapsed))

    def sample_once(self) -> dict[str, Any]:
        """Take one synchronous sample of every monitor."""
        ts = now_ns()
        record: dict[str, Any] = {"timestamp_ns": ts}
        try:
            gpu = self.gpu.sample(ts)
            record["gpu"] = gpu.to_dict() if gpu else None
        except Exception as exc:  # telemetry must never crash a run
            record["gpu"] = None
            self._errors.append(f"gpu: {exc}")
        try:
            cpu = self.cpu.sample(ts)
            record["cpu"] = cpu.to_dict()
        except Exception as exc:
            record["cpu"] = None
            self._errors.append(f"cpu: {exc}")
        try:
            mem = self.memory.sample(ts)
            record["memory"] = mem.to_dict()
        except Exception as exc:
            record["memory"] = None
            self._errors.append(f"memory: {exc}")
        with self._lock:
            self._samples.append(record)
        return record

    # --- queries ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Point-in-time snapshot (used for phase boundaries like VRAM before/after)."""
        return self.sample_once()

    def vram_used_mb(self) -> float | None:
        gpu = self.snapshot().get("gpu")
        return gpu.get("vram_used_mb") if gpu else None

    def window_stats(self, start_ns: int, end_ns: int) -> WindowStats:
        """Aggregate every sample whose timestamp lies within [start_ns, end_ns]."""
        with self._lock:
            samples = [
                s for s in self._samples if start_ns <= s["timestamp_ns"] <= end_ns
            ]
        stats = WindowStats(sample_count=len(samples))
        values: dict[str, dict[str, float | None]] = {}

        gpu_vals: dict[str, list[float]] = {f: [] for f in _GPU_FIELDS}
        cpu_vals: dict[str, list[float]] = {f: [] for f in _CPU_FIELDS}
        mem_vals: dict[str, list[float]] = {f: [] for f in _MEM_FIELDS}

        for sample in samples:
            gpu = sample.get("gpu")
            if gpu:
                stats.gpu_samples += 1
                for f in _GPU_FIELDS:
                    if gpu.get(f) is not None:
                        gpu_vals[f].append(float(gpu[f]))
            cpu = sample.get("cpu")
            if cpu:
                for f in _CPU_FIELDS:
                    if cpu.get(f) is not None:
                        cpu_vals[f].append(float(cpu[f]))
            mem = sample.get("memory")
            if mem:
                for f in _MEM_FIELDS:
                    if mem.get(f) is not None:
                        mem_vals[f].append(float(mem[f]))

        for f in _GPU_FIELDS:
            values[f] = _aggregate(gpu_vals[f])
        for f in _CPU_FIELDS:
            values[f] = _aggregate(cpu_vals[f])
        for f in _MEM_FIELDS:
            values[f] = _aggregate(mem_vals[f])
        stats.values = values
        return stats

    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    def overhead_per_sample_ms(self) -> float | None:
        return self._overhead_ms

    def info(self) -> dict[str, Any]:
        return {
            "interval_ms": int(self.interval_s * 1000),
            "deep": self.deep,
            "gpu": self.gpu.info(),
            "cpu": self.cpu.info(),
            "memory": self.memory.info(),
            "samples_collected": len(self._samples),
            "errors": self.errors()[:10],
            "overhead_per_sample_ms": self._overhead_ms,
        }
