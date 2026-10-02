"""Raw benchmark result records.

Raw data is sacred: every measured (and warmup) request produces one
:class:`RawResult` appended to ``raw.jsonl`` immediately after execution. Only
afterwards are summaries, reports and plots generated.

Statuses are deliberately separated:

* ``execution_status``  — did the request execute (SUCCESS / OOM / FAILED / ...)
* ``performance_valid`` — may these numbers be used for performance comparison
* ``correctness_status`` — PASS / WARNING / FAIL about output content
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .timing import PROVENANCE_UNAVAILABLE

# --- Execution statuses ----------------------------------------------------

EXEC_SUCCESS = "SUCCESS"
EXEC_OOM = "OOM"
EXEC_FAILED = "FAILED"
EXEC_TIMEOUT = "TIMEOUT"
EXEC_UNSUPPORTED = "UNSUPPORTED"

EXECUTION_STATUSES: tuple[str, ...] = (
    EXEC_SUCCESS,
    EXEC_OOM,
    EXEC_FAILED,
    EXEC_TIMEOUT,
    EXEC_UNSUPPORTED,
)

# --- Correctness statuses --------------------------------------------------

CORRECT_PASS = "PASS"
CORRECT_WARNING = "WARNING"
CORRECT_FAIL = "FAIL"
CORRECT_SKIPPED = "SKIPPED"

CORRECTNESS_STATUSES: tuple[str, ...] = (
    CORRECT_PASS,
    CORRECT_WARNING,
    CORRECT_FAIL,
    CORRECT_SKIPPED,
)

# --- Finish reasons --------------------------------------------------------

FINISH_LENGTH = "length"
FINISH_EOS = "eos"
FINISH_ABORTED = "aborted"
FINISH_ERROR = "error"
FINISH_UNKNOWN = "unknown"


@dataclass
class TokenTrace:
    """Full per-token timing trace, saved under ``traces/`` (never discarded)."""

    request_id: str
    submit_ns: int
    tokens: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "submit_ns": self.submit_ns,
            "tokens": self.tokens,
        }


@dataclass
class RawResult:
    """One raw benchmark record (the unit stored in ``raw.jsonl``)."""

    # Identity / provenance of the benchmark itself
    benchmark_version: str
    session_id: str
    timestamp: str
    suite: str

    # Runtime
    runtime: str
    runtime_version: str | None = None
    runtime_commit: str | None = None
    runtime_config: dict[str, Any] = field(default_factory=dict)

    # Model / tokenizer
    model: str | None = None
    model_revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    bpw: float | None = None

    # Hardware
    gpu: str | None = None
    gpu_uuid: str | None = None
    driver: str | None = None
    cuda: str | None = None

    # Workload placement
    profile: str | None = None
    input_class: str | None = None
    output_class: str | None = None
    request_id: str | None = None
    run_index: int = 0
    warmup: bool = False
    benchmark_mode: str = "validation"
    run_policy: str = "standard"
    performance_valid: bool = False

    # Prompt accounting
    prompt_hash: str | None = None
    input_ids_hash: str | None = None
    prompt_tokens: int = 0
    requested_output_tokens: int = 0
    actual_output_tokens: int = 0

    # Placement
    batch_size: int = 1
    concurrency: int = 1

    # Prefix reuse
    prefix_tokens: int = 0
    prefix_reused_tokens: int | None = None
    prefix_recomputed_tokens: int | None = None
    prefix_id: str | None = None
    expected_reuse_ratio: float | None = None

    # Timing (monotonic clock; ns absolute are session-relative monotonic values)
    request_submitted_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None

    ttft_ms: float | None = None
    prefill_ms: float | None = None
    prefill_tok_s: float | None = None

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

    e2e_ms: float | None = None

    queue_ms: float | None = None
    scheduler_ms: float | None = None

    # Per-request throughput
    output_tok_s_per_request: float | None = None

    # GPU telemetry (window over the request)
    gpu_util_avg: float | None = None
    gpu_util_peak: float | None = None
    gpu_mem_util_avg: float | None = None
    gpu_mem_util_peak: float | None = None
    gpu_power_avg: float | None = None
    gpu_power_median: float | None = None
    gpu_power_peak: float | None = None
    gpu_temperature_avg: float | None = None
    gpu_temperature_peak: float | None = None
    gpu_clock_avg: float | None = None
    gpu_mem_clock_avg: float | None = None
    pcie_rx_avg: float | None = None
    pcie_tx_avg: float | None = None

    # Memory telemetry
    vram_before_mb: float | None = None
    vram_loaded_mb: float | None = None
    vram_peak_mb: float | None = None
    vram_after_mb: float | None = None
    vram_model_mb: float | None = None
    vram_kv_cache_mb: float | None = None
    vram_cuda_graph_mb: float | None = None
    vram_workspace_mb: float | None = None
    vram_runtime_overhead_mb: float | None = None
    kv_memory_mb: float | None = None

    ram_before_mb: float | None = None
    ram_peak_mb: float | None = None
    ram_after_mb: float | None = None

    # CPU telemetry
    cpu_avg: float | None = None
    cpu_peak: float | None = None
    system_cpu_avg: float | None = None
    process_cpu_time_s: float | None = None
    threads_peak: int | None = None

    # Group aggregates (filled for concurrent groups from the global window)
    wall_clock_ms: float | None = None
    aggregate_output_tok_s: float | None = None
    aggregate_prompt_tok_s: float | None = None
    aggregate_total_tok_s: float | None = None
    aggregate_decode_tok_s: float | None = None
    requests_per_second: float | None = None
    completed_requests_per_minute: float | None = None

    # Correctness data
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text: str | None = None
    finish_reason: str = FINISH_UNKNOWN
    early_termination: bool = False

    # Request metadata preserved (phase, role, scenario, slot, ...)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal runtime metrics (cache, queue, scheduler) where exposed
    cache_hit_rate: float | None = None
    cache_lookup_overhead_ms: float | None = None
    cache_insertion_overhead_ms: float | None = None
    internal_metrics: dict[str, Any] = field(default_factory=dict)

    # Metric provenance: metric name -> provenance category
    metric_provenance: dict[str, str] = field(
        default_factory=lambda: {
            "prefill_ms": PROVENANCE_UNAVAILABLE,
            "queue_ms": PROVENANCE_UNAVAILABLE,
            "scheduler_ms": PROVENANCE_UNAVAILABLE,
            "kv_memory_mb": PROVENANCE_UNAVAILABLE,
            "cache_hit_rate": PROVENANCE_UNAVAILABLE,
        }
    )

    # Statuses
    execution_status: str = EXEC_SUCCESS
    correctness_status: str = CORRECT_SKIPPED
    error: str | None = None
    fairness_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FailureRecord:
    """A failure persisted to ``failures.jsonl`` (never silently dropped)."""

    benchmark_version: str
    session_id: str
    timestamp: str
    suite: str
    runtime: str
    model: str | None
    profile: str | None
    concurrency: int
    input_tokens: int
    requested_output_tokens: int
    status: str
    error: str
    request_id: str | None = None
    run_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
