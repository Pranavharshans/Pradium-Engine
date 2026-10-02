"""Static batching infrastructure.

Static batch size (Bn) is deliberately NOT treated as equivalent to
concurrency (Cn): batches are formed and dispatched as one unit through the
runtime's batch API, and batch formation time is measured separately.

Without streaming batch execution, per-request TTFT/ITL/decode timing is
unavailable and is recorded as such — never fabricated.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..adapters.base import CapabilityNotSupported
from ..core.results import (
    CORRECT_SKIPPED,
    EXEC_UNSUPPORTED,
    RawResult,
    FailureRecord,
)
from ..core.timing import (
    PROVENANCE_DERIVED,
    PROVENANCE_MEASURED_EXTERNAL,
    PROVENANCE_UNAVAILABLE,
    now_ns,
    ns_to_ms,
    prefill_metrics,
)
from ..core.workloads import BATCH_SIZES
from .base import RunnerContext, isolate_prefix_cache
from .prompts import PromptProvider


def run_batching(
    ctx: RunnerContext,
    provider: PromptProvider,
    profile: str = "MM",
    batch_sizes: tuple[int, ...] | list[int] = BATCH_SIZES,
    warmups: int = 1,
    measured_runs: int = 3,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Run static batch sizes B1..B8 where the runtime supports them."""
    records: list[RawResult] = []
    extras: dict[str, Any] = {"batching": {}}
    capabilities = ctx.adapter.get_capabilities()

    for batch_size in batch_sizes:
        case_records: list[RawResult] = []
        for run_index in range(warmups + measured_runs):
            warmup = run_index < warmups
            case_records.extend(
                _run_batch(ctx, provider, profile, batch_size, run_index, warmup)
            )
        for record in case_records:
            if record.execution_status != "SUCCESS":
                record.performance_valid = False
                ctx.session.append_failure(FailureRecord(
                    benchmark_version=record.benchmark_version,
                    session_id=record.session_id, timestamp=record.timestamp,
                    suite=ctx.suite, runtime=record.runtime, model=record.model,
                    profile=record.profile, concurrency=1,
                    input_tokens=record.prompt_tokens,
                    requested_output_tokens=record.requested_output_tokens,
                    status=record.execution_status, error=record.error or "unknown error",
                    request_id=record.request_id, run_index=record.run_index,
                ))
            elif record.correctness_status != "PASS":
                record.performance_valid = False
            ctx.session.append_raw(record)
        records.extend(case_records)
        successful = [
            r for r in case_records if r.execution_status == "SUCCESS"
        ]
        extras["batching"][str(batch_size)] = {
            "supported": capabilities.is_usable("supports_static_batching"),
            "requests": len(case_records),
            "successful_requests": len(successful),
        }
    return records, extras


def _run_batch(
    ctx: RunnerContext,
    provider: PromptProvider,
    profile: str,
    batch_size: int,
    run_index: int,
    warmup: bool,
) -> list[RawResult]:
    from ..core.workloads import PROFILES

    outputs = PROFILES[profile].output_tokens
    requests = [
        provider.profile_request(
            profile,
            run_index=run_index,
            warmup=warmup,
            batch_size=batch_size,
            concurrency_group=f"B{batch_size}",
            metadata={"slot": slot, "batch_size": batch_size},
        )
        for slot in range(batch_size)
    ]
    isolate_prefix_cache(ctx)
    defaults = ctx.record_defaults()

    # --- batch formation + dispatch ---
    formed_at = time.perf_counter()
    results: list[Any] = []
    error: str | None = None
    status = "SUCCESS"
    window_start = now_ns()
    try:
        # formation time: assembling the batch before dispatch
        batch = list(requests)
        formation_ms = (time.perf_counter() - formed_at) * 1000.0
        results = ctx.adapter.generate_batch(batch)
        if len(results) != len(requests):
            raise ValueError(f"batch returned {len(results)} results for {len(requests)} requests")
    except CapabilityNotSupported as exc:
        status, error = EXEC_UNSUPPORTED, str(exc)
        formation_ms = (time.perf_counter() - formed_at) * 1000.0
    except Exception as exc:  # noqa: BLE001
        status, error = "FAILED", f"{type(exc).__name__}: {exc}"
        formation_ms = (time.perf_counter() - formed_at) * 1000.0
    window_end = now_ns()

    if status == EXEC_UNSUPPORTED:
        return [
            RawResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                profile=profile,
                request_id=request.request_id,
                run_index=run_index,
                warmup=warmup,
                prompt_hash=request.metadata.get("prompt_hash"),
                input_ids_hash=request.input_ids_hash,
                prompt_tokens=request.prompt_token_count,
                requested_output_tokens=outputs,
                actual_output_tokens=0,
                batch_size=batch_size,
                concurrency=1,
                execution_status=EXEC_UNSUPPORTED,
                correctness_status=CORRECT_SKIPPED,
                error=error,
                metadata=dict(request.metadata),
                internal_metrics={"batch_formation_ms": formation_ms},
                metric_provenance={
                    "batch_formation_ms": PROVENANCE_MEASURED_EXTERNAL,
                    "ttft_ms": PROVENANCE_UNAVAILABLE,
                },
                **defaults,
            )
            for request in requests
        ]

    if status != "SUCCESS":
        return [
            RawResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                profile=profile,
                request_id=request.request_id,
                run_index=run_index,
                warmup=warmup,
                prompt_tokens=request.prompt_token_count,
                requested_output_tokens=outputs,
                actual_output_tokens=0,
                batch_size=batch_size,
                execution_status=status,
                correctness_status=CORRECT_SKIPPED,
                error=error,
                metadata=dict(request.metadata),
                internal_metrics={"batch_formation_ms": formation_ms},
                metric_provenance={
                    "batch_formation_ms": PROVENANCE_MEASURED_EXTERNAL,
                },
                **defaults,
            )
            for request in requests
        ]

    batch_wall_ms = ns_to_ms(window_end - window_start)
    total_output = sum(len(r.token_ids) for r in results)
    total_prompt = sum(r.prompt_token_count for r in requests)
    aggregates = {
        "wall_clock_ms": batch_wall_ms,
        "aggregate_output_tok_s": (
            total_output / (batch_wall_ms / 1000.0) if batch_wall_ms > 0 else None
        ),
        "aggregate_prompt_tok_s": (
            total_prompt / (batch_wall_ms / 1000.0) if batch_wall_ms > 0 else None
        ),
        "aggregate_total_tok_s": (
            (total_output + total_prompt) / (batch_wall_ms / 1000.0)
            if batch_wall_ms > 0
            else None
        ),
    }

    records: list[RawResult] = []
    for request, result in zip(requests, results):
        prefill_ms, prefill_tok_s, prefill_prov = prefill_metrics(
            result.prefill_ms, request.prompt_token_count
        )
        record = RawResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            profile=profile,
            request_id=request.request_id,
            run_index=run_index,
            warmup=warmup,
            prompt_hash=request.metadata.get("prompt_hash"),
            input_ids_hash=request.input_ids_hash,
            prompt_tokens=request.prompt_token_count,
            requested_output_tokens=request.max_new_tokens,
            actual_output_tokens=len(result.token_ids),
            batch_size=batch_size,
            concurrency=1,
            prefill_ms=prefill_ms,
            prefill_tok_s=prefill_tok_s,
            # static batch dispatch: no per-request token timestamps available
            ttft_ms=None,
            decode_ms=None,
            decode_tok_s=None,
            tpot_ms=None,
            e2e_ms=batch_wall_ms,
            generated_token_ids=list(result.token_ids),
            finish_reason=result.finish_reason,
            metadata=dict(request.metadata),
            internal_metrics={
                "batch_formation_ms": formation_ms,
                **result.internal_metrics,
            },
            wall_clock_ms=batch_wall_ms,
            aggregate_output_tok_s=aggregates["aggregate_output_tok_s"],
            aggregate_prompt_tok_s=aggregates["aggregate_prompt_tok_s"],
            aggregate_total_tok_s=aggregates["aggregate_total_tok_s"],
            metric_provenance={
                "batch_formation_ms": PROVENANCE_MEASURED_EXTERNAL,
                "prefill_ms": prefill_prov,
                "prefill_tok_s": prefill_prov,
                "ttft_ms": PROVENANCE_UNAVAILABLE,
                "decode_ms": PROVENANCE_UNAVAILABLE,
                "itl": PROVENANCE_UNAVAILABLE,
                "e2e_ms": PROVENANCE_MEASURED_EXTERNAL,
                "aggregate_output_tok_s": PROVENANCE_DERIVED,
            },
            execution_status="SUCCESS",
            correctness_status=(
                "PASS"
                if len(result.token_ids) == request.max_new_tokens
                else "WARNING"
            ),
            fairness_notes=[
                "static batch dispatch: per-request token timing unavailable"
            ],
            **defaults,
        )
        records.append(record)
    return records
