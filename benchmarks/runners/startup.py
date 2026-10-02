"""Startup benchmark runner.

Measures process start -> runtime init -> model load -> ready -> first request
-> first token, cold vs warm. Runtime-reported milestones (CUDA init, graph
capture, ...) are recorded when the adapter exposes them; otherwise the
corresponding fields stay unavailable. Idle CPU/RAM/VRAM are captured before
initialization.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.timing import now_ns
from .base import RunnerContext, execute_request
from .prompts import PromptProvider


def run_startup(
    ctx: RunnerContext,
    provider: PromptProvider,
    request_profile: str = "MM",
    warm_request_count: int = 5,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Measure startup milestones and cold/warm first-request latency."""
    records: list[RawResult] = []
    milestones: dict[str, Any] = {}

    # --- idle resources before anything is initialized ---
    idle = ctx.telemetry.snapshot()
    milestones["idle_cpu_pct"] = (idle.get("cpu") or {}).get("process_cpu_pct")
    milestones["idle_ram_mb"] = (idle.get("memory") or {}).get("process_rss_mb")
    milestones["idle_vram_mb"] = (idle.get("gpu") or {}).get("vram_used_mb")

    t_process_start = now_ns()

    # --- runtime initialization ---
    t0 = time.perf_counter()
    ctx.adapter.initialize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    milestones["runtime_init_ms"] = init_ms

    # --- model load ---
    t0 = time.perf_counter()
    ctx.adapter.load_model()
    load_ms = (time.perf_counter() - t0) * 1000.0
    milestones["model_load_ms"] = load_ms

    # --- runtime-reported milestones (CUDA init, weight mapping, graph capture) ---
    reported = ctx.adapter.get_startup_milestones() or {}
    for key in (
        "cuda_init_ms",
        "weight_mapping_ms",
        "cuda_graph_capture_ms",
        "server_ready_ms",
    ):
        milestones[key] = reported.get(key)

    ready_ns = now_ns()
    milestones["process_start_to_ready_ms"] = (ready_ns - t_process_start) / 1e6

    # --- cold first request ---
    cold_request = provider.profile_request(
        request_profile,
        run_index=0,
        warmup=False,
        metadata={"phase": "cold"},
    )
    cold = execute_request(ctx, cold_request, concurrency=1)
    records.append(cold)
    milestones["cold_first_request_ttft_ms"] = cold.ttft_ms
    milestones["process_start_to_first_token_ms"] = (
        (cold.first_token_ns - t_process_start) / 1e6 if cold.first_token_ns else None
    )

    # --- warm requests ---
    warm_ttfts: list[float] = []
    for i in range(warm_request_count):
        request = provider.profile_request(
            request_profile,
            run_index=i + 1,
            warmup=False,
            metadata={"phase": "warm"},
        )
        record = execute_request(ctx, request, concurrency=1)
        records.append(record)
        if record.ttft_ms is not None:
            warm_ttfts.append(record.ttft_ms)
    milestones["warm_request_ttft_ms_median"] = (
        sorted(warm_ttfts)[len(warm_ttfts) // 2] if warm_ttfts else None
    )
    milestones["warm_request_count"] = warm_request_count
    milestones["process_start_ns"] = t_process_start
    return records, milestones
