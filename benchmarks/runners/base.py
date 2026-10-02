"""Shared runner machinery: request execution, timing capture, raw collection.

Collection is strictly separated from aggregation/reporting: everything a
request produces is appended to ``raw.jsonl`` (and ``traces/``) here, before
any analysis runs. A crash in reporting can never destroy measurement data.

Timing uses a monotonic clock (``time.perf_counter_ns``); wall-clock timestamps
appear only as record metadata.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..adapters.base import (
    RuntimeAdapter,
    RuntimeOutOfMemoryError,
    RuntimeTimeoutError,
    StreamToken,
)
from ..core.config import BenchmarkConfig
from ..core.request import BenchmarkRequest
from ..core.results import (
    CORRECT_FAIL,
    CORRECT_PASS,
    CORRECT_SKIPPED,
    CORRECT_WARNING,
    EXEC_FAILED,
    EXEC_OOM,
    EXEC_SUCCESS,
    EXEC_TIMEOUT,
    FailureRecord,
    RawResult,
    TokenTrace,
)
from ..core.session import Session
from ..core.timing import (
    PROVENANCE_DERIVED,
    PROVENANCE_MEASURED_EXTERNAL,
    PROVENANCE_REPORTED_RUNTIME,
    PROVENANCE_UNAVAILABLE,
    TokenEvent,
    compute_request_timing,
    group_aggregate_throughput,
    now_ns,
    prefill_metrics,
)
from ..core.version import BENCHMARK_VERSION
from ..telemetry.sampler import TelemetrySampler, WindowStats


@dataclass
class RunnerContext:
    """Everything a suite runner needs; no framework-specific state."""

    adapter: RuntimeAdapter
    session: Session
    telemetry: TelemetrySampler
    config: BenchmarkConfig
    suite: str
    benchmark_mode: str  # quick | validation | official
    run_policy: str  # quick | standard | official
    performance_valid: bool
    environment: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    model_revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    bpw: float | None = None
    timeout_s: float | None = None
    vram_loaded_mb: float | None = None

    def record_defaults(self) -> dict[str, Any]:
        """Common raw-record fields for this context."""
        runtime_info = self.adapter.get_runtime_info()
        gpu_block = self.environment.get("gpu", {})
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "session_id": self.session.session_id,
            "suite": self.suite,
            "runtime": self.adapter.name,
            "runtime_version": runtime_info.get("version"),
            "runtime_commit": runtime_info.get("commit"),
            "runtime_config": runtime_info.get("config", {}),
            "model": self.model or runtime_info.get("model"),
            "model_revision": self.model_revision or runtime_info.get("model_revision"),
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "quantization": self.quantization,
            "bpw": self.bpw,
            "gpu": gpu_block.get("name"),
            "gpu_uuid": gpu_block.get("uuid"),
            "driver": gpu_block.get("driver"),
            "cuda": gpu_block.get("cuda"),
            "benchmark_mode": self.benchmark_mode,
            "run_policy": self.run_policy,
            "performance_valid": self.performance_valid,
        }


def _window_value(stats: WindowStats, metric: str, stat: str) -> float | None:
    return stats.get(metric, stat)


def execute_request(
    ctx: RunnerContext,
    request: BenchmarkRequest,
    gate: threading.Event | None = None,
    concurrency: int = 1,
    batch_size: int = 1,
) -> RawResult:
    """Execute one request end to end and persist its raw record.

    Captures: submission timestamp (after any start barrier), per-token arrival
    timestamps, runtime-reported prefill/queue/scheduler metrics, correctness
    data, telemetry over the request window, and the full token timing trace.
    """
    if gate is not None:
        gate.wait()
    submit_ns = now_ns()
    events: list[TokenEvent] = []
    final: StreamToken | None = None
    error: str | None = None
    status = EXEC_SUCCESS
    vram_before = ctx.telemetry.vram_used_mb()
    ram_before = ctx.telemetry.snapshot().get("memory", {}).get("process_rss_mb")

    try:
        for token in ctx.adapter.generate_stream(request):
            if token.done:
                final = token
                break
            events.append(TokenEvent(token.index, token.token_id or 0, now_ns()))
            if (
                ctx.timeout_s is not None
                and (now_ns() - submit_ns) > ctx.timeout_s * 1e9
            ):
                raise RuntimeTimeoutError(
                    f"request exceeded timeout of {ctx.timeout_s}s"
                )
    except RuntimeOutOfMemoryError as exc:
        status, error = EXEC_OOM, str(exc)
    except RuntimeTimeoutError as exc:
        status, error = EXEC_TIMEOUT, str(exc)
    except Exception as exc:  # noqa: BLE001 - failures are recorded, not fatal
        status, error = EXEC_FAILED, f"{type(exc).__name__}: {exc}"

    end_ns = events[-1].timestamp_ns if events else now_ns()
    timing = compute_request_timing(submit_ns, events)
    window = ctx.telemetry.window_stats(submit_ns, end_ns)

    # --- correctness (independent of execution status) ---
    actual_tokens = len(events)
    early_termination = (
        status == EXEC_SUCCESS and actual_tokens < request.max_new_tokens
    )
    if status != EXEC_SUCCESS:
        correctness = CORRECT_SKIPPED
    elif actual_tokens == 0:
        correctness = CORRECT_FAIL
        error = (error + "; " if error else "") + "empty output"
    elif early_termination:
        correctness = CORRECT_WARNING
        reason = (
            "unexpected early termination (fixed decode length requested)"
            if request.fixed_decode_length and request.ignore_eos
            else "early termination"
        )
        error = (error + "; " if error else "") + reason
    else:
        correctness = CORRECT_PASS

    # --- provenance ---
    prefill_ms, prefill_tok_s, prefill_prov = prefill_metrics(
        final.prefill_ms if final else None, request.prompt_token_count
    )
    provenance: dict[str, str] = {
        "ttft_ms": PROVENANCE_MEASURED_EXTERNAL,
        "e2e_ms": PROVENANCE_MEASURED_EXTERNAL,
        "decode_ms": PROVENANCE_MEASURED_EXTERNAL,
        "decode_tok_s": PROVENANCE_DERIVED,
        "tpot_ms": PROVENANCE_DERIVED,
        "itl": PROVENANCE_DERIVED,
        "prefill_ms": prefill_prov,
        "prefill_tok_s": prefill_prov,
        "queue_ms": (
            PROVENANCE_REPORTED_RUNTIME
            if final and final.queue_ms is not None
            else PROVENANCE_UNAVAILABLE
        ),
        "scheduler_ms": (
            PROVENANCE_REPORTED_RUNTIME
            if final and final.scheduler_ms is not None
            else PROVENANCE_UNAVAILABLE
        ),
    }

    internal = dict(final.internal_metrics) if final else {}
    cache_hit = internal.get("cache_hit_rate")
    provenance["cache_hit_rate"] = (
        PROVENANCE_REPORTED_RUNTIME if cache_hit is not None else PROVENANCE_UNAVAILABLE
    )
    provenance["kv_memory_mb"] = PROVENANCE_UNAVAILABLE

    # --- fairness notes ---
    fairness_notes: list[str] = []
    if not ctx.adapter.get_capabilities().is_usable("supports_input_ids"):
        fairness_notes.append("runtime retokenized prompt text (no input_ids support)")
    if (
        request.fixed_decode_length
        and not ctx.adapter.get_capabilities().is_usable("supports_fixed_decode_length")
    ):
        fairness_notes.append("runtime cannot guarantee fixed decode length")

    defaults = ctx.record_defaults()
    record = RawResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        profile=request.profile,
        input_class=request.input_class,
        output_class=request.output_class,
        run_index=request.run_index,
        warmup=request.warmup,
        prompt_hash=request.metadata.get("prompt_hash"),
        input_ids_hash=request.input_ids_hash,
        prompt_tokens=request.prompt_token_count,
        requested_output_tokens=request.max_new_tokens,
        actual_output_tokens=actual_tokens,
        batch_size=batch_size,
        concurrency=concurrency,
        prefix_tokens=request.prefix_token_count,
        prefix_reused_tokens=internal.get("prefix_reused_tokens"),
        prefix_recomputed_tokens=internal.get("prefix_recomputed_tokens"),
        prefix_id=request.prefix_id,
        expected_reuse_ratio=request.metadata.get("expected_reuse_ratio"),
        request_submitted_ns=submit_ns,
        first_token_ns=timing.first_token_ns,
        last_token_ns=timing.last_token_ns,
        ttft_ms=timing.ttft_ms,
        prefill_ms=prefill_ms,
        prefill_tok_s=prefill_tok_s,
        decode_ms=timing.decode_ms,
        decode_tok_s=timing.decode_tok_s,
        tpot_ms=timing.tpot_ms,
        itl_mean_ms=timing.itl_mean_ms,
        itl_median_ms=timing.itl_median_ms,
        itl_p95_ms=timing.itl_p95_ms,
        itl_p99_ms=timing.itl_p99_ms,
        itl_max_ms=timing.itl_max_ms,
        itl_min_ms=timing.itl_min_ms,
        itl_stdev_ms=timing.itl_stdev_ms,
        e2e_ms=timing.e2e_ms,
        queue_ms=final.queue_ms if final else None,
        scheduler_ms=final.scheduler_ms if final else None,
        output_tok_s_per_request=timing.output_tok_s_per_request,
        gpu_util_avg=_window_value(window, "util_gpu_pct", "avg"),
        gpu_util_peak=_window_value(window, "util_gpu_pct", "max"),
        gpu_mem_util_avg=_window_value(window, "util_mem_pct", "avg"),
        gpu_mem_util_peak=_window_value(window, "util_mem_pct", "max"),
        gpu_power_avg=_window_value(window, "power_w", "avg"),
        gpu_power_median=_window_value(window, "power_w", "median"),
        gpu_power_peak=_window_value(window, "power_w", "max"),
        gpu_temperature_avg=_window_value(window, "temperature_c", "avg"),
        gpu_temperature_peak=_window_value(window, "temperature_c", "max"),
        gpu_clock_avg=_window_value(window, "clock_mhz", "avg"),
        gpu_mem_clock_avg=_window_value(window, "mem_clock_mhz", "avg"),
        pcie_rx_avg=_window_value(window, "pcie_rx_mb_s", "avg"),
        pcie_tx_avg=_window_value(window, "pcie_tx_mb_s", "avg"),
        vram_before_mb=vram_before,
        vram_loaded_mb=ctx.vram_loaded_mb,
        vram_peak_mb=_window_value(window, "vram_used_mb", "max"),
        vram_after_mb=ctx.telemetry.vram_used_mb(),
        ram_before_mb=ram_before,
        ram_peak_mb=_window_value(window, "process_rss_peak_mb", "max"),
        ram_after_mb=ctx.telemetry.snapshot().get("memory", {}).get("process_rss_mb"),
        cpu_avg=_window_value(window, "process_cpu_pct", "avg"),
        cpu_peak=_window_value(window, "process_cpu_pct", "max"),
        system_cpu_avg=_window_value(window, "system_cpu_pct", "avg"),
        process_cpu_time_s=_window_value(window, "process_cpu_time_s", "max"),
        threads_peak=int(_window_value(window, "thread_count", "max") or 0) or None,
        generated_token_ids=[e.token_id for e in events],
        generated_text=None,
        finish_reason=(final.finish_reason if final else "error"),
        early_termination=early_termination,
        cache_hit_rate=cache_hit,
        cache_lookup_overhead_ms=internal.get("cache_lookup_overhead_ms"),
        cache_insertion_overhead_ms=internal.get("cache_insertion_overhead_ms"),
        internal_metrics=internal,
        metric_provenance=provenance,
        execution_status=status,
        correctness_status=correctness,
        error=error,
        fairness_notes=fairness_notes,
        **defaults,
    )

    # --- persist immediately (raw data is sacred) ---
    ctx.session.append_raw(record)
    if events:
        trace = TokenTrace(
            request_id=request.request_id,
            submit_ns=submit_ns,
            tokens=[
                {
                    "token_index": e.token_index,
                    "token_id": e.token_id,
                    "relative_timestamp_ns": e.timestamp_ns - submit_ns,
                }
                for e in events
            ],
        )
        ctx.session.write_trace(trace)
    if status != EXEC_SUCCESS:
        ctx.session.append_failure(
            FailureRecord(
                benchmark_version=BENCHMARK_VERSION,
                session_id=ctx.session.session_id,
                timestamp=record.timestamp,
                suite=ctx.suite,
                runtime=ctx.adapter.name,
                model=record.model,
                profile=request.profile,
                concurrency=concurrency,
                input_tokens=request.prompt_token_count,
                requested_output_tokens=request.max_new_tokens,
                status=status,
                error=error or "unknown error",
                request_id=request.request_id,
                run_index=request.run_index,
            )
        )
    return record


def run_group(
    ctx: RunnerContext,
    requests: list[BenchmarkRequest],
    concurrency: int | None = None,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Execute a group of requests with synchronized start.

    Requests are released from a shared gate so Cn means genuinely overlapping
    execution (not sequential submission). Actual submission timestamps are
    recorded per request. Aggregate throughput is computed from the group's
    global wall-clock interval, never from summed per-request rates.
    """
    concurrency = concurrency or len(requests)
    results: list[RawResult] = []
    if len(requests) == 1:
        record = execute_request(ctx, requests[0], gate=None, concurrency=concurrency)
        results.append(record)
        wall_ns = 0
        if record.request_submitted_ns is not None and record.last_token_ns is not None:
            wall_ns = record.last_token_ns - record.request_submitted_ns
        aggregates = _group_aggregates(ctx, results, wall_ns)
        _fill_aggregates(results, aggregates)
        return results, aggregates

    gate = threading.Event()
    window_start_holder: list[int] = []
    results_lock = threading.Lock()

    def worker(request: BenchmarkRequest) -> None:
        record = execute_request(
            ctx, request, gate=gate, concurrency=concurrency, batch_size=request.batch_size
        )
        with results_lock:
            results.append(record)

    threads = [
        threading.Thread(target=worker, args=(req,), name=f"bench-req-{req.request_id}")
        for req in requests
    ]
    for thread in threads:
        thread.start()
    window_start_holder.append(now_ns())
    gate.set()
    for thread in threads:
        thread.join()
    wall_ns = now_ns() - window_start_holder[0]

    aggregates = _group_aggregates(ctx, results, wall_ns)
    _fill_aggregates(results, aggregates)
    return results, aggregates


def _fill_aggregates(results: list[RawResult], aggregates: dict[str, Any]) -> None:
    for record in results:
        record.aggregate_output_tok_s = aggregates["aggregate_output_tok_s"]
        record.aggregate_prompt_tok_s = aggregates["aggregate_prompt_tok_s"]
        record.aggregate_total_tok_s = aggregates["aggregate_total_tok_s"]
        record.aggregate_decode_tok_s = aggregates["aggregate_decode_tok_s"]
        record.requests_per_second = aggregates["requests_per_second"]
        record.completed_requests_per_minute = aggregates[
            "completed_requests_per_minute"
        ]
        record.wall_clock_ms = aggregates["wall_clock_ms"]


def _group_aggregates(
    ctx: RunnerContext, results: list[RawResult], wall_ns: int
) -> dict[str, Any]:
    """Aggregate throughput from actual global wall-clock execution."""
    successful = [r for r in results if r.execution_status == EXEC_SUCCESS]
    total_output = sum(r.actual_output_tokens for r in successful)
    total_prompt = sum(r.prompt_tokens for r in successful)
    decode_intervals = sum(max(0, r.actual_output_tokens - 1) for r in successful)
    aggregates: dict[str, Any] = group_aggregate_throughput(
        total_output, total_prompt, wall_ns, len(successful)
    )
    if wall_ns > 0:
        aggregates["aggregate_decode_tok_s"] = decode_intervals / (wall_ns / 1e9)
    else:
        aggregates["aggregate_decode_tok_s"] = None
    aggregates["group_requests"] = len(results)
    aggregates["group_successful_requests"] = len(successful)
    aggregates["group_output_tokens"] = total_output
    return aggregates


def warmup_and_measure(
    ctx: RunnerContext,
    build_request,
    warmups: int,
    measured_runs: int,
    concurrency: int = 1,
) -> list[RawResult]:
    """Run warmups (recorded but excluded from statistics) then measured runs.

    ``build_request(run_index, warmup)`` must return a list of
    :class:`BenchmarkRequest` (one per concurrent slot).
    """
    records: list[RawResult] = []
    for i in range(warmups):
        requests = build_request(i, True)
        group, _ = run_group(ctx, requests, concurrency=concurrency)
        records.extend(group)
    for i in range(measured_runs):
        requests = build_request(i, False)
        group, _ = run_group(ctx, requests, concurrency=concurrency)
        records.extend(group)
    return records
