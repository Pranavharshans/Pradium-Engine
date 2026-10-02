"""Soak / stability suite.

Long-running repeated execution of representative workloads while monitoring
VRAM, RAM, CPU, GPU utilization, power, TTFT, TPOT and throughput. Detects
memory growth, latency degradation, throughput degradation, failures and
unbounded cache growth.

Soak is never part of quick mode. Duration is configurable (30/60 minutes
suggested); validation runs may use a short duration.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.request import BenchmarkRequest
from ..core.results import EXEC_SUCCESS, RawResult
from ..core.timing import now_ns
from .base import RunnerContext, run_group
from .prompts import PromptProvider


def run_soak(
    ctx: RunnerContext,
    provider: PromptProvider,
    duration_s: float = 1800.0,
    profiles: tuple[str, ...] | list[str] = ("MM", "SL", "LL"),
    concurrency: int = 1,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Repeatedly run representative workloads until ``duration_s`` elapses."""
    records: list[RawResult] = []
    iterations: list[dict[str, Any]] = []
    started_ns = now_ns()
    deadline = time.monotonic() + duration_s
    iteration = 0

    while time.monotonic() < deadline:
        profile_name = profiles[iteration % len(profiles)]
        requests: list[BenchmarkRequest] = [
            provider.profile_request(
                profile_name,
                run_index=iteration,
                warmup=iteration == 0,
                concurrency_group=f"C{concurrency}" if concurrency > 1 else None,
                metadata={"soak_iteration": iteration, "slot": slot},
            )
            for slot in range(concurrency)
        ]
        group, aggregates = run_group(ctx, requests, concurrency=concurrency)
        records.extend(group)
        successful = [r for r in group if r.execution_status == EXEC_SUCCESS]
        memory = ctx.telemetry.snapshot()
        iterations.append(
            {
                "iteration": iteration,
                "profile": profile_name,
                "ttft_ms": successful[0].ttft_ms if successful else None,
                "tpot_ms": successful[0].tpot_ms if successful else None,
                "decode_tok_s": successful[0].decode_tok_s if successful else None,
                "aggregate_output_tok_s": aggregates.get("aggregate_output_tok_s"),
                "vram_used_mb": (memory.get("gpu") or {}).get("vram_used_mb"),
                "ram_used_mb": (memory.get("memory") or {}).get("process_rss_mb"),
                "failed_requests": len(group) - len(successful),
            }
        )
        iteration += 1

    elapsed_s = (now_ns() - started_ns) / 1e9
    extras = {
        "soak": {
            "duration_requested_s": duration_s,
            "duration_actual_s": elapsed_s,
            "iterations": iteration,
            "profiles": list(profiles),
            "concurrency": concurrency,
            "series": iterations,
            "analysis": _analyze_series(iterations),
        }
    }
    return records, extras


def _quartile_mean(values: list[float], first: bool) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 4:
        return None
    quarter = max(1, len(vals) // 4)
    selected = vals[:quarter] if first else vals[-quarter:]
    return sum(selected) / len(selected)


def _analyze_series(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect degradation: memory growth, latency growth, throughput drop."""
    def series(key: str) -> list[float]:
        return [float(it[key]) for it in iterations if it.get(key) is not None]

    analysis: dict[str, Any] = {
        "failed_requests_total": sum(it["failed_requests"] for it in iterations),
    }
    for key, name in (
        ("ram_used_mb", "ram_growth_mb"),
        ("vram_used_mb", "vram_growth_mb"),
        ("ttft_ms", "ttft_growth_ratio"),
        ("decode_tok_s", "decode_throughput_change_ratio"),
    ):
        values = series(key)
        first = _quartile_mean(values, True)
        last = _quartile_mean(values, False)
        if first is None or last is None or first == 0:
            analysis[name] = None
            continue
        if key.endswith("_mb"):
            analysis[name] = last - first
        else:
            analysis[name] = last / first - 1.0
    return analysis
