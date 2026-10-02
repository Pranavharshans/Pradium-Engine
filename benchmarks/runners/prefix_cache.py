"""Prefix / KV-reuse suite.

Tests approximate reusable-prefix ratios (0 / 25 / 50 / 75 / 90 / ~100%) where
the ratio is defined on token IDs. At ~100% each request is a large identical
prefix plus a small unique suffix — never exact duplicate full prompts.

Scenarios (capability-gated later by real runtimes):

* ``same_session``   — prime and measure within one session (cache intact)
* ``cross_request``  — prime with one request, measure a different request
* ``cross_session``  — a session boundary (``reset_cache``) between prime and
  measure: reuse only survives if the runtime truly caches across sessions
"""

from __future__ import annotations

from typing import Any

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.workloads import (
    PREFIX_REQUESTS_PER_RATIO,
    PREFIX_REUSE_RATIOS,
    PREFIX_TOTAL_INPUT_TOKENS,
    PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
)
from .base import RunnerContext, execute_request
from .prompts import PromptProvider

SCENARIOS = ("same_session", "cross_request", "cross_session")


def run_prefix_cache(
    ctx: RunnerContext,
    provider: PromptProvider,
    ratios: tuple[float, ...] | list[float] = PREFIX_REUSE_RATIOS,
    scenarios: tuple[str, ...] | list[str] = SCENARIOS,
    total_input_tokens: int = PREFIX_TOTAL_INPUT_TOKENS,
    requests_per_ratio: int = PREFIX_REQUESTS_PER_RATIO,
    output_tokens: int = 256,
    unique_suffix_tokens_at_full_reuse: int = PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
    measured_runs: int = 3,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Run the prefix-reuse matrix; returns records plus per-case metadata."""
    records: list[RawResult] = []
    extras: dict[str, Any] = {"prefix_cache": {}}

    for ratio in ratios:
        group = provider.prefix_group(
            ratio,
            total_input_tokens,
            requests_per_ratio,
            unique_suffix_tokens_at_full_reuse,
        )
        for scenario in scenarios:
            for run_index in range(measured_runs):
                case = _run_case(
                    ctx,
                    provider,
                    ratio,
                    scenario,
                    group,
                    output_tokens,
                    run_index,
                )
                records.extend(case["records"])
                key = f"{ratio:.2f}:{scenario}"
                entry = extras["prefix_cache"].setdefault(
                    key,
                    {
                        "ratio": ratio,
                        "scenario": scenario,
                        "total_input_tokens": group.total_input_tokens,
                        "shared_prefix_tokens": group.shared_prefix_tokens,
                        "unique_suffix_tokens": group.unique_suffix_tokens,
                        "expected_reuse_ratio": group.expected_reuse_ratio(),
                        "requests": 0,
                        "reused_tokens": [],
                    },
                )
                entry["requests"] += len(case["records"])
                entry["reused_tokens"].extend(case["reused_tokens"])
    return records, extras


def _run_case(
    ctx: RunnerContext,
    provider: PromptProvider,
    ratio: float,
    scenario: str,
    group,
    output_tokens: int,
    run_index: int,
) -> dict[str, Any]:
    records: list[RawResult] = []
    reused_tokens: list[int] = []

    caps = ctx.adapter.get_capabilities()
    if not caps.is_usable("supports_cache_reset"):
        ctx.performance_valid = False
    else:
        ctx.adapter.clear_prefix_cache()

    for index, prompt in enumerate(group.requests):
        if caps.is_usable("supports_cache_reset"):
            ctx.adapter.clear_prefix_cache()
        # Prime a different suffix, preserving only the intended shared prefix.
        prime_prompt = group.requests[(index + 1) % len(group.requests)]
        if prime_prompt.input_ids == prompt.input_ids:
            raise ValueError("prefix test requires distinct prime and measure prompts")
        # --- prime the shared prefix (recorded as warmup, excluded from stats) ---
        prime_request = provider.request(
            prime_prompt,
            output_tokens,
            run_index=run_index,
            warmup=True,
            profile=f"PREFIX-{ratio:.2f}",
            prefix_id=group.prefix_id,
            prefix_token_count=group.shared_prefix_tokens,
            metadata={
                "scenario": scenario,
                "reuse_ratio": ratio,
                "role": "prime",
                "expected_reuse_ratio": group.expected_reuse_ratio(),
            },
        )
        execute_request(ctx, prime_request, concurrency=1)

        # --- session boundary for cross-session testing ---
        if scenario == "cross_session":
            ctx.adapter.reset_cache()

        # --- measured request sharing the prefix ---
        request = provider.request(
            prompt,
            output_tokens,
            run_index=run_index,
            warmup=False,
            profile=f"PREFIX-{ratio:.2f}",
            prefix_id=group.prefix_id,
            prefix_token_count=group.shared_prefix_tokens,
            metadata={
                "scenario": scenario,
                "reuse_ratio": ratio,
                "role": "measure",
                "expected_reuse_ratio": group.expected_reuse_ratio(),
            },
        )
        record = execute_request(ctx, request, concurrency=1)
        records.append(record)
        if record.prefix_reused_tokens is not None:
            reused_tokens.append(record.prefix_reused_tokens)

    return {"records": records, "reused_tokens": reused_tokens}
