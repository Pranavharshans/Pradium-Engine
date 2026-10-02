"""Scheduler-interference suite.

Scenario: requests A/B/C are decoding; request D (a large prefill) arrives.
The suite measures what happens to A/B/C's streams: ITL before / during /
after D's prefill window, maximum output-token stall, p95 ITL, D's TTFT,
aggregate throughput.

Per-token timing traces are the source of truth; phase statistics are derived
from them and kept separate from the raw per-request records.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.timing import percentile
from ..core.workloads import (
    SCHEDULER_BACKGROUND_PROFILE,
    SCHEDULER_BACKGROUND_REQUESTS,
    SCHEDULER_INTRUDER_PROFILE,
)
from .base import RunnerContext, execute_request
from .prompts import PromptProvider


def load_trace(session, request_id: str) -> dict[str, Any] | None:
    """Load the persisted token timing trace for a request."""
    path = Path(session.directory) / "traces" / f"{request_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_scheduler(
    ctx: RunnerContext,
    provider: PromptProvider,
    background_profile: str = SCHEDULER_BACKGROUND_PROFILE,
    intruder_profile: str = SCHEDULER_INTRUDER_PROFILE,
    background_requests: int = SCHEDULER_BACKGROUND_REQUESTS,
    intruder_delay_ms: int = 250,
    output_scale: float = 1.0,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Run the large-prefill interference scenario once and analyze phases."""
    from ..core.workloads import PROFILES

    bg_output = max(4, int(PROFILES[background_profile].output_tokens * output_scale))
    intruder_output = PROFILES[intruder_profile].output_tokens

    background: list[BenchmarkRequest] = [
        provider.profile_request(
            background_profile,
            run_index=0,
            warmup=False,
            concurrency_group="scheduler-background",
            metadata={
                "role": "background",
                "slot": i,
                "background_output_tokens": bg_output,
            },
        )
        for i in range(background_requests)
    ]
    # Override output length when scaled for fast validation.
    if bg_output != PROFILES[background_profile].output_tokens:
        for request in background:
            request.max_new_tokens = bg_output

    gate = threading.Event()
    records: list[RawResult] = []
    lock = threading.Lock()

    def worker(request: BenchmarkRequest) -> None:
        record = execute_request(
            ctx, request, gate=gate, concurrency=background_requests
        )
        with lock:
            records.append(record)

    threads = [
        threading.Thread(target=worker, args=(r,), name=f"bench-bg-{r.request_id}")
        for r in background
    ]
    for thread in threads:
        thread.start()
    gate.set()

    # --- the large prefill arrives while A/B/C decode ---
    time.sleep(max(0.0, intruder_delay_ms) / 1000.0)
    intruder = provider.profile_request(
        intruder_profile,
        run_index=0,
        warmup=False,
        concurrency_group="scheduler-intruder",
        metadata={"role": "intruder"},
    )
    intruder_record = execute_request(ctx, intruder, concurrency=1)
    with lock:
        records.append(intruder_record)

    for thread in threads:
        thread.join()

    background_records = [
        r for r in records if r.request_id != intruder_record.request_id
    ]
    extras = _analyze_interference(ctx, background_records, intruder_record)
    return records, extras


def _analyze_interference(
    ctx: RunnerContext,
    background_records: list[RawResult],
    intruder_record: RawResult,
) -> dict[str, Any]:
    """Derive ITL phase statistics from persisted token timing traces."""
    t_arrival = intruder_record.request_submitted_ns
    t_first_token = intruder_record.first_token_ns
    if t_arrival is None or t_first_token is None:
        return {
            "scheduler_interference": {
                "status": "UNAVAILABLE",
                "detail": "intruder produced no timing",
            }
        }

    phase_itls: dict[str, list[float]] = {"before": [], "during": [], "after": []}
    max_stall_ms = 0.0
    per_request: list[dict[str, Any]] = []

    for record in background_records:
        trace = load_trace(ctx.session, record.request_id) if record.request_id else None
        if not trace or not trace.get("tokens"):
            continue
        submit = trace["submit_ns"]
        timestamps = [submit + t["relative_timestamp_ns"] for t in trace["tokens"]]
        request_phases = {"before": [], "during": [], "after": []}
        for i in range(1, len(timestamps)):
            itl_ms = (timestamps[i] - timestamps[i - 1]) / 1e6
            anchor = timestamps[i - 1]
            if anchor < t_arrival:
                phase = "before"
            elif anchor < t_first_token:
                phase = "during"
            else:
                phase = "after"
            request_phases[phase].append(itl_ms)
            phase_itls[phase].append(itl_ms)
            max_stall_ms = max(max_stall_ms, itl_ms)
        all_itls = [
            (timestamps[i] - timestamps[i - 1]) / 1e6
            for i in range(1, len(timestamps))
        ]
        per_request.append(
            {
                "request_id": record.request_id,
                "ttft_ms": record.ttft_ms,
                "decode_tok_s": record.decode_tok_s,
                "itl_mean_ms": record.itl_mean_ms,
                "itl_max_ms": record.itl_max_ms,
                "phase_itl_mean_ms": {
                    phase: (sum(vals) / len(vals) if vals else None)
                    for phase, vals in request_phases.items()
                },
                "phase_token_counts": {
                    phase: len(vals) + 1 if vals else 0
                    for phase, vals in request_phases.items()
                },
                "p95_itl_ms": percentile(all_itls, 95),
            }
        )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "scheduler_interference": {
            "status": "OK",
            "background_requests": len(background_records),
            "intruder_profile": intruder_record.profile,
            "intruder_ttft_ms": intruder_record.ttft_ms,
            "intruder_prefill_ms": intruder_record.prefill_ms,
            "intruder_input_tokens": intruder_record.prompt_tokens,
            "itl_before_ms": mean(phase_itls["before"]),
            "itl_during_ms": mean(phase_itls["during"]),
            "itl_after_ms": mean(phase_itls["after"]),
            "itl_before_samples": len(phase_itls["before"]),
            "itl_during_samples": len(phase_itls["during"]),
            "itl_after_samples": len(phase_itls["after"]),
            "max_output_token_stall_ms": max_stall_ms,
            "p95_itl_ms": percentile(phase_itls["before"] + phase_itls["during"] + phase_itls["after"], 95),
            "background_aggregate_decode_tok_s": sum(
                (r.decode_tok_s or 0.0) for r in background_records
            ),
            "per_request": per_request,
        }
    }
