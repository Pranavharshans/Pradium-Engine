"""Concurrency runner: the full matrix at C1/C2/C4/C8.

Cn means n simultaneous active requests released from a shared start gate
(never sequential submission). Every request keeps its own timing record;
aggregate throughput comes from the group's global wall-clock window.
"""

from __future__ import annotations

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.workloads import CONCURRENCY_LEVELS, PROFILE_ORDER
from .base import RunnerContext, run_group
from .prompts import PromptProvider


def run_concurrency(
    ctx: RunnerContext,
    provider: PromptProvider,
    profiles: tuple[str, ...] | list[str] = PROFILE_ORDER,
    levels: tuple[int, ...] | list[int] = CONCURRENCY_LEVELS,
    warmups: int = 3,
    measured_runs: int = 10,
) -> list[RawResult]:
    """Run every profile at every concurrency level.

    The full matrix (SS..LL) x (C1, C2, C4, C8) is available; quick mode
    selects a subset of profiles and levels.
    """
    records: list[RawResult] = []
    for name in profiles:
        for level in levels:
            records.extend(
                _run_case(ctx, provider, name, int(level), warmups, measured_runs)
            )
    return records


def _run_case(
    ctx: RunnerContext,
    provider: PromptProvider,
    profile: str,
    concurrency: int,
    warmups: int,
    measured_runs: int,
) -> list[RawResult]:
    def build(run_index: int, warmup: bool) -> list[BenchmarkRequest]:
        return [
            provider.profile_request(
                profile,
                run_index=run_index,
                warmup=warmup,
                concurrency_group=f"C{concurrency}",
                metadata={"slot": slot, "concurrency": concurrency},
            )
            for slot in range(concurrency)
        ]

    records: list[RawResult] = []
    for i in range(warmups):
        group, _ = run_group(ctx, build(i, True), concurrency=concurrency)
        records.extend(group)
    for i in range(measured_runs):
        group, _ = run_group(ctx, build(i, False), concurrency=concurrency)
        records.extend(group)
    return records
