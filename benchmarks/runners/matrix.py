"""3x3 matrix runner (the core workload suite).

Runs SS..LL with the frozen input/output token counts, at C1 by default (the
concurrency suite covers C1/C2/C4/C8 across the full matrix).
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.workloads import PROFILE_ORDER
from .base import RunnerContext, run_group
from .prompts import PromptProvider


def run_matrix(
    ctx: RunnerContext,
    provider: PromptProvider,
    profiles: tuple[str, ...] | list[str] = PROFILE_ORDER,
    warmups: int = 3,
    measured_runs: int = 10,
    concurrency: int = 1,
) -> list[RawResult]:
    """Run the requested matrix profiles with warmups + measured runs.

    Warmup runs are recorded in ``raw.jsonl`` (flagged ``warmup``) but excluded
    from headline statistics by the analysis stage.
    """
    records: list[RawResult] = []
    for name in profiles:
        records.extend(
            _run_profile(ctx, provider, name, warmups, measured_runs, concurrency)
        )
    return records


def _run_profile(
    ctx: RunnerContext,
    provider: PromptProvider,
    profile: str,
    warmups: int,
    measured_runs: int,
    concurrency: int,
) -> list[RawResult]:
    def build(run_index: int, warmup: bool) -> list[BenchmarkRequest]:
        return [
            provider.profile_request(
                profile,
                run_index=run_index,
                warmup=warmup,
                concurrency_group=f"C{concurrency}" if concurrency > 1 else None,
                metadata={"slot": slot} if concurrency > 1 else None,
            )
            for slot in range(concurrency)
        ]

    return _warmup_and_measure(ctx, build, warmups, measured_runs, concurrency)


def _warmup_and_measure(
    ctx: RunnerContext,
    build: Callable[[int, bool], list[BenchmarkRequest]],
    warmups: int,
    measured_runs: int,
    concurrency: int,
) -> list[RawResult]:
    records: list[RawResult] = []
    for i in range(warmups):
        group, _ = run_group(ctx, build(i, True), concurrency=concurrency)
        records.extend(group)
    for i in range(measured_runs):
        group, _ = run_group(ctx, build(i, False), concurrency=concurrency)
        records.extend(group)
    return records


def matrix_summary_payload(records: list[RawResult]) -> dict[str, Any]:
    """Compact per-profile overview used by report generation."""
    out: dict[str, Any] = {}
    for record in records:
        if record.warmup:
            continue
        out.setdefault(record.profile or "?", []).append(
            {
                "run_index": record.run_index,
                "execution_status": record.execution_status,
                "ttft_ms": record.ttft_ms,
                "decode_tok_s": record.decode_tok_s,
                "actual_output_tokens": record.actual_output_tokens,
            }
        )
    return out
