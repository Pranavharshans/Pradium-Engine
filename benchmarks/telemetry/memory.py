"""Host memory telemetry (process RSS + system RAM).

Uses psutil when available, ``/proc/self/statm`` on Linux, and falls back to
``resource.getrusage`` peak RSS on other platforms. When only peak RSS is
obtainable it is recorded as peak, never as a current value.
"""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass
from typing import Any

MB = 1024 * 1024


@dataclass(frozen=True)
class MemorySample:
    """One host-memory telemetry sample (MiB)."""

    timestamp_ns: int
    process_rss_mb: float | None = None
    process_rss_peak_mb: float | None = None
    system_total_mb: float | None = None
    system_available_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "process_rss_mb": self.process_rss_mb,
            "process_rss_peak_mb": self.process_rss_peak_mb,
            "system_total_mb": self.system_total_mb,
            "system_available_mb": self.system_available_mb,
        }


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / MB, 2)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="utf-8") as fh:
                pages = int(fh.read().split()[1])
            return round(pages * os.sysconf("SC_PAGE_SIZE") / MB, 2)
        except (OSError, ValueError, IndexError):
            pass
    return None


def _process_rss_peak_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = float(usage.ru_maxrss)
    # macOS reports bytes, Linux reports kilobytes.
    if sys.platform == "darwin":
        return round(peak / MB, 2)
    return round(peak / 1024.0, 2)


def _system_ram() -> tuple[float | None, float | None]:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return round(mem.total / MB, 2), round(mem.available / MB, 2)
    except Exception:
        pass
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            avail = None
        return round(total / MB, 2), (round(avail / MB, 2) if avail else None)
    except (ValueError, OSError, AttributeError):
        return None, None


class MemoryMonitor:
    """Samples process and system memory."""

    def __init__(self) -> None:
        self.backend = "psutil"
        try:
            import psutil  # type: ignore  # noqa: F401
        except Exception:
            self.backend = "stdlib"

    def sample(self, timestamp_ns: int) -> MemorySample:
        total, available = _system_ram()
        return MemorySample(
            timestamp_ns=timestamp_ns,
            process_rss_mb=_process_rss_mb(),
            process_rss_peak_mb=_process_rss_peak_mb(),
            system_total_mb=total,
            system_available_mb=available,
        )

    def snapshot_mb(self) -> dict[str, float | None]:
        sample = self.sample(0)
        return {
            "process_rss_mb": sample.process_rss_mb,
            "process_rss_peak_mb": sample.process_rss_peak_mb,
            "system_total_mb": sample.system_total_mb,
            "system_available_mb": sample.system_available_mb,
        }

    def info(self) -> dict[str, Any]:
        return {"backend": self.backend}
