"""Timing and metric definitions (frozen for PRADIUM-RUNTIME-BENCH-v1).

All latency measurement uses a monotonic high-resolution clock
(``time.perf_counter_ns()``). Wall-clock timestamps are recorded separately for
metadata only and are never used in latency arithmetic.

Formulas
--------

::

    TTFT        = first_token_timestamp - request_submit_timestamp        [ms]

    decode_window   = last_token_timestamp - first_token_timestamp        [ms]
    decode_tok_s    = (N - 1) / decode_window_s        (N > 1, N tokens)
    decode_ms       = decode_window                                          [ms]
    TPOT        = decode_window_ms / (N - 1)            (N > 1)          [ms/token]
    E2E         = last_token_timestamp - request_submit_timestamp        [ms]

    ITL_k       = token_timestamp_k - token_timestamp_{k-1}  (k = 2..N)     [ms]

Prefill latency is NOT inferred from TTFT (TTFT includes queueing, scheduling,
preprocessing, prefill, sampling and first-token delivery). Prefill values are
stored only when the runtime reports them or an explicit estimate is produced;
every metric carries provenance.
"""

from __future__ import annotations

import math
import statistics as _statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

NS_PER_MS = 1_000_000.0
NS_PER_S = 1_000_000_000.0

# --- Metric provenance (frozen vocabulary) --------------------------------

PROVENANCE_MEASURED_EXTERNAL = "MEASURED_EXTERNAL"
PROVENANCE_REPORTED_RUNTIME = "REPORTED_RUNTIME"
PROVENANCE_DERIVED = "DERIVED"
PROVENANCE_ESTIMATED = "ESTIMATED"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

PROVENANCE_VALUES: tuple[str, ...] = (
    PROVENANCE_MEASURED_EXTERNAL,
    PROVENANCE_REPORTED_RUNTIME,
    PROVENANCE_DERIVED,
    PROVENANCE_ESTIMATED,
    PROVENANCE_UNAVAILABLE,
)


def now_ns() -> int:
    """Monotonic, high-resolution timestamp in nanoseconds."""
    return time.perf_counter_ns()


def ns_to_ms(delta_ns: float) -> float:
    return float(delta_ns) / NS_PER_MS


def ms_to_s(delta_ms: float) -> float:
    return float(delta_ms) / 1000.0


# --- Percentiles -----------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in [0, 100]); None for empty input."""
    if not values:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q out of range: {q}")
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    rank = (q / 100.0) * (len(data) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return data[low]
    frac = rank - low
    return data[low] * (1.0 - frac) + data[high] * frac


# --- Token events ----------------------------------------------------------


@dataclass(frozen=True)
class TokenEvent:
    """One emitted output token with its monotonic arrival timestamp."""

    token_index: int
    token_id: int
    timestamp_ns: int

    def to_dict(self, base_ns: int | None = None) -> dict[str, Any]:
        rel = self.timestamp_ns - base_ns if base_ns is not None else self.timestamp_ns
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "relative_timestamp_ns": rel,
        }


@dataclass
class TimingMetrics:
    """Derived timing for one request. Units: ms, tok/s, ms/token."""

    submit_ns: int
    first_token_ns: int | None
    last_token_ns: int | None
    token_count: int

    ttft_ms: float | None = None
    e2e_ms: float | None = None

    decode_ms: float | None = None
    decode_tok_s: float | None = None
    tpot_ms: float | None = None

    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_max_ms: float | None = None
    itl_min_ms: float | None = None
    itl_stdev_ms: float | None = None

    output_tok_s_per_request: float | None = None

    itl_samples_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submit_ns": self.submit_ns,
            "first_token_ns": self.first_token_ns,
            "last_token_ns": self.last_token_ns,
            "token_count": self.token_count,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "decode_ms": self.decode_ms,
            "decode_tok_s": self.decode_tok_s,
            "tpot_ms": self.tpot_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "itl_median_ms": self.itl_median_ms,
            "itl_p95_ms": self.itl_p95_ms,
            "itl_p99_ms": self.itl_p99_ms,
            "itl_max_ms": self.itl_max_ms,
            "itl_min_ms": self.itl_min_ms,
            "itl_stdev_ms": self.itl_stdev_ms,
            "output_tok_s_per_request": self.output_tok_s_per_request,
        }


def compute_request_timing(submit_ns: int, events: Sequence[TokenEvent]) -> TimingMetrics:
    """Compute all timing metrics from raw token timestamps.

    ``events`` must be ordered by emission. With N >= 1 tokens:

    * TTFT = first_token_ns - submit_ns
    * decode window = last_token_ns - first_token_ns (0 when N == 1)
    * decode_tok_s = (N - 1) / decode_window_s when N > 1
    * TPOT = decode_window_ms / (N - 1) when N > 1
    * E2E = last_token_ns - submit_ns
    * ITL = differences between consecutive token timestamps
    """
    metrics = TimingMetrics(
        submit_ns=submit_ns,
        first_token_ns=events[0].timestamp_ns if events else None,
        last_token_ns=events[-1].timestamp_ns if events else None,
        token_count=len(events),
    )
    if not events:
        return metrics

    metrics.ttft_ms = ns_to_ms(events[0].timestamp_ns - submit_ns)
    metrics.e2e_ms = ns_to_ms(events[-1].timestamp_ns - submit_ns)

    if len(events) >= 2:
        itl_ms = [
            ns_to_ms(events[i].timestamp_ns - events[i - 1].timestamp_ns)
            for i in range(1, len(events))
        ]
        metrics.itl_samples_ms = itl_ms
        decode_ms = ns_to_ms(events[-1].timestamp_ns - events[0].timestamp_ns)
        metrics.decode_ms = decode_ms
        n_decoded = len(events) - 1
        if decode_ms > 0:
            metrics.decode_tok_s = (n_decoded / ms_to_s(decode_ms))
            metrics.tpot_ms = decode_ms / n_decoded
        else:
            metrics.decode_tok_s = None
            metrics.tpot_ms = None
        metrics.itl_mean_ms = _statistics.fmean(itl_ms)
        metrics.itl_median_ms = _statistics.median(itl_ms)
        metrics.itl_p95_ms = percentile(itl_ms, 95)
        metrics.itl_p99_ms = percentile(itl_ms, 99)
        metrics.itl_max_ms = max(itl_ms)
        metrics.itl_min_ms = min(itl_ms)
        metrics.itl_stdev_ms = (
            _statistics.stdev(itl_ms) if len(itl_ms) >= 2 else 0.0
        )

    # Per-request output throughput over the whole request (incl. prefill).
    e2e_s = ms_to_s(metrics.e2e_ms) if metrics.e2e_ms else 0.0
    if e2e_s > 0:
        metrics.output_tok_s_per_request = len(events) / e2e_s
    return metrics


def prefill_metrics(
    prefill_ms: float | None, prompt_tokens: int
) -> tuple[float | None, float | None, str]:
    """Return ``(prefill_ms, prefill_tok_s, provenance)``.

    Never derived from TTFT. Without a runtime-reported value the provenance is
    ``UNAVAILABLE`` and both values are ``None``.
    """
    if prefill_ms is None or prefill_ms <= 0:
        return None, None, PROVENANCE_UNAVAILABLE
    tok_s = prompt_tokens / ms_to_s(prefill_ms)
    return float(prefill_ms), float(tok_s), PROVENANCE_REPORTED_RUNTIME


def group_aggregate_throughput(
    total_output_tokens: int,
    total_prompt_tokens: int,
    wall_clock_ns: int,
    request_count: int,
) -> dict[str, float | None]:
    """Aggregate throughput from the global wall-clock interval.

    ``aggregate_output_tok_s = total output tokens / wall-clock seconds``.
    The wall-clock interval spans the concurrent workload, never the sum of
    per-request durations.
    """
    if wall_clock_ns <= 0:
        return {
            "aggregate_output_tok_s": None,
            "aggregate_prompt_tok_s": None,
            "aggregate_total_tok_s": None,
            "requests_per_second": None,
            "completed_requests_per_minute": None,
            "wall_clock_ms": 0.0,
        }
    seconds = wall_clock_ns / NS_PER_S
    return {
        "aggregate_output_tok_s": total_output_tokens / seconds,
        "aggregate_prompt_tok_s": total_prompt_tokens / seconds,
        "aggregate_total_tok_s": (total_output_tokens + total_prompt_tokens) / seconds,
        "requests_per_second": request_count / seconds,
        "completed_requests_per_minute": request_count / (seconds / 60.0),
        "wall_clock_ms": ns_to_ms(wall_clock_ns),
    }


def scaling_efficiency(
    aggregate_decode_tok_s: float | None,
    concurrency: int,
    decode_tok_s_c1: float | None,
) -> float | None:
    """``aggregate_decode(Cn) / (n * decode(C1))``."""
    if not aggregate_decode_tok_s or not decode_tok_s_c1 or concurrency < 1:
        return None
    return aggregate_decode_tok_s / (concurrency * decode_tok_s_c1)


def degradation_ratio(current: float | None, baseline: float | None) -> float | None:
    """Relative change vs baseline: ``current / baseline - 1`` (None if undefined)."""
    if current is None or baseline in (None, 0):
        return None
    return current / baseline - 1.0
