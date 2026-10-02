"""Statistical methodology (frozen for PRADIUM-RUNTIME-BENCH-v1).

For every numeric performance metric the summary records:

``count, mean, median, min, max, standard deviation, p50, p90, p95, p99,
coefficient of variation``

The headline central statistic is **MEDIAN** — never the fastest run.
Warmup runs are excluded from these statistics.
"""

from __future__ import annotations

import math
import statistics as _statistics
from typing import Any, Sequence

from ..core.timing import percentile


def describe(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    """Full descriptive statistics for a numeric sample (None values dropped)."""
    data = [float(v) for v in values if v is not None]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "cv": None,
        }
    mean = _statistics.fmean(data)
    stdev = _statistics.stdev(data) if len(data) >= 2 else 0.0
    cv = (stdev / mean) if mean else None
    return {
        "count": len(data),
        "mean": mean,
        "median": _statistics.median(data),
        "min": min(data),
        "max": max(data),
        "stdev": stdev,
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "cv": cv,
    }


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def finite(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            f = float(value)
            if math.isfinite(f):
                out.append(f)
    return out
