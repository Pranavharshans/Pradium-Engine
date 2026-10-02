"""CPU telemetry.

Uses psutil when available; otherwise falls back to stdlib ``resource`` for
process CPU time and ``os.getloadavg`` for a system-load estimate (marked as
estimated provenance in results). Thread counts come from the process when
available, else from the threading module as an approximation.
"""

from __future__ import annotations

import os
import resource
import threading
import time
from dataclasses import dataclass
from typing import Any

NS_PER_S = 1_000_000_000.0


@dataclass(frozen=True)
class CpuSample:
    """One CPU telemetry sample."""

    timestamp_ns: int
    process_cpu_pct: float | None = None
    system_cpu_pct: float | None = None
    process_cpu_time_s: float | None = None
    thread_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "process_cpu_pct": self.process_cpu_pct,
            "system_cpu_pct": self.system_cpu_pct,
            "process_cpu_time_s": self.process_cpu_time_s,
            "thread_count": self.thread_count,
        }


def _process_cpu_time_s() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _thread_count() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().num_threads())
    except Exception:
        return threading.active_count()


def _system_cpu_pct() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        try:
            load1, _, _ = os.getloadavg()
            return min(100.0 * load1 / (os.cpu_count() or 1), 100.0)
        except (OSError, AttributeError):
            return None


class CpuMonitor:
    """Tracks process and system CPU usage between samples."""

    def __init__(self) -> None:
        self.backend = "psutil"
        try:
            import psutil  # type: ignore  # noqa: F401
        except Exception:
            self.backend = "resource"
        self._last_time_ns: int | None = None
        self._last_cpu_s: float | None = None

    def sample(self, timestamp_ns: int) -> CpuSample:
        cpu_time = _process_cpu_time_s()
        process_pct: float | None = None
        if self._last_time_ns is not None and self._last_cpu_s is not None:
            wall_s = (timestamp_ns - self._last_time_ns) / NS_PER_S
            if wall_s > 0:
                process_pct = max(
                    0.0, 100.0 * (cpu_time - self._last_cpu_s) / wall_s
                )
        self._last_time_ns = timestamp_ns
        self._last_cpu_s = cpu_time
        return CpuSample(
            timestamp_ns=timestamp_ns,
            process_cpu_pct=process_pct,
            system_cpu_pct=_system_cpu_pct(),
            process_cpu_time_s=cpu_time,
            thread_count=_thread_count(),
        )

    def info(self) -> dict[str, Any]:
        return {"backend": self.backend}
