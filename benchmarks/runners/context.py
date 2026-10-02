"""Extended context-scaling runner.

Input lengths: 128 .. 32768 with a fixed 256-token output. Cases that cannot
run on the current hardware/model/runtime are recorded as OOM / UNSUPPORTED /
FAILED and the suite continues — cases are never silently dropped.
"""

from __future__ import annotations

from ..core.results import RawResult
from ..core.workloads import CONTEXT_INPUT_TOKENS, CONTEXT_OUTPUT_TOKENS
from .base import RunnerContext, run_group
from .matrix import _warmup_and_measure
from .prompts import PromptProvider


def run_context(
    ctx: RunnerContext,
    provider: PromptProvider,
    input_tokens: tuple[int, ...] | list[int] = CONTEXT_INPUT_TOKENS,
    output_tokens: int = CONTEXT_OUTPUT_TOKENS,
    warmups: int = 3,
    measured_runs: int = 10,
) -> list[RawResult]:
    """Run every context length; failures are recorded, execution continues."""
    records: list[RawResult] = []
    for length in input_tokens:
        try:
            prompt = provider.context_prompt(length)
        except Exception as exc:  # materialization failure is itself recorded
            records.append(
                _failed_case(ctx, length, output_tokens, error=str(exc))
            )
            continue

        def build(run_index: int, warmup: bool, length=length, prompt=prompt):
            return [
                provider.request(
                    prompt,
                    output_tokens,
                    run_index=run_index,
                    warmup=warmup,
                    profile=f"CTX-{length}",
                    input_class="CTX",
                    output_class="M",
                    metadata={"context_input_tokens": length},
                )
            ]

        group = _warmup_and_measure(ctx, build, warmups, measured_runs, 1)
        records.extend(group)
    return records


def _failed_case(
    ctx: RunnerContext, length: int, output_tokens: int, error: str
) -> RawResult:
    from datetime import datetime, timezone

    from ..core.results import EXEC_FAILED, CORRECT_SKIPPED

    defaults = ctx.record_defaults()
    record = RawResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        profile=f"CTX-{length}",
        input_class="CTX",
        output_class="M",
        prompt_tokens=length,
        requested_output_tokens=output_tokens,
        actual_output_tokens=0,
        execution_status=EXEC_FAILED,
        correctness_status=CORRECT_SKIPPED,
        error=error,
        **defaults,
    )
    ctx.session.append_raw(record)
    return record
