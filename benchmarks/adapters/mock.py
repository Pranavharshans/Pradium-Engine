"""Mock runtime adapter.

Simulates an inference runtime well enough to validate the entire benchmark
pipeline — initialization, model load, streaming, TTFT, inter-token delays,
concurrency contention, prefill interference, deterministic outputs, failures
and optional prefix-cache reuse — WITHOUT any real model.

**The mock runtime is only for infrastructure testing.** Mock performance is
never real benchmark data: every mock result carries
``runtime = mock``, ``benchmark_mode = validation``, ``performance_valid = false``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..core.capabilities import CapabilityStatus, RuntimeCapabilities
from ..core.request import BenchmarkRequest
from .base import (
    CapabilityNotSupported,
    GenerationResult,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeOutOfMemoryError,
    RuntimeTimeoutError,
    StreamToken,
)

MOCK_RUNTIME_NAME = "mock"

#: Failure kinds that can be injected for infrastructure testing.
FAILURE_KINDS = ("oom", "exception", "early_eos", "partial", "timeout")


@dataclass
class MockFailureSpec:
    """An injected failure for requests matching ``match``."""

    kind: str
    match: dict[str, Any] = field(default_factory=dict)
    after_tokens: int = 1

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"unknown mock failure kind: {self.kind!r}")

    def matches(self, request: BenchmarkRequest) -> bool:
        for key, expected in self.match.items():
            actual = getattr(request, key, None)
            if actual is None:
                actual = request.metadata.get(key)
            if actual != expected:
                return False
        return True


@dataclass
class MockRuntimeConfig:
    """Tuning knobs for the simulated runtime (validation only)."""

    time_scale: float = 1.0
    prefill_tok_s: float = 20_000.0
    decode_tok_s: float = 80.0
    ttft_overhead_ms: float = 4.0
    queue_ms: float = 1.0
    contention_factor: float = 0.12
    prefill_interference: float = 1.5
    prefix_cache_enabled: bool = True
    prefix_cache_skip_fraction: float = 0.9
    fixed_decode_length: bool = True
    streaming: bool = True
    static_batching: bool = False
    cross_request_prefix_cache: bool = True
    oom_above_input_tokens: int | None = None
    load_model_fails_oom: bool = False
    model: str = "mock-model"
    model_revision: str = "mock-rev-1"
    failures: list[MockFailureSpec] = field(default_factory=list)


class MockRuntime(RuntimeAdapter):
    """Deterministic simulated runtime for benchmark validation."""

    name = MOCK_RUNTIME_NAME

    def __init__(self, config: MockRuntimeConfig | None = None) -> None:
        self.config = config or MockRuntimeConfig()
        self._initialized = False
        self._model_loaded = False
        self._lock = threading.Lock()
        self._active_requests = 0
        self._prefill_active = 0
        self._prefix_cache: dict[str, int] = {}
        self._milestones: dict[str, float | None] = {}
        self._request_counter = 0

    # --- helpers ----------------------------------------------------------

    def _sleep_ms(self, ms: float) -> None:
        seconds = (ms / 1000.0) * self.config.time_scale
        if seconds > 0:
            time.sleep(seconds)

    def _check_ready(self) -> None:
        if not self._model_loaded:
            raise RuntimeAdapterError("mock runtime: model not loaded")

    @staticmethod
    def _token_id(request_id: str, index: int) -> int:
        seed = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)
        return 1000 + ((seed + index) % 50_000)

    # --- lifecycle --------------------------------------------------------

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        started = time.perf_counter()
        self._sleep_ms(20.0)
        self._initialized = True
        self._milestones["runtime_init_ms"] = (time.perf_counter() - started) * 1000.0

    def load_model(self) -> None:
        if not self._initialized:
            raise RuntimeAdapterError("mock runtime: initialize() first")
        started = time.perf_counter()
        if self.config.load_model_fails_oom:
            self._milestones["model_load_ms"] = None
            raise RuntimeOutOfMemoryError("mock runtime: simulated model-load OOM")
        self._sleep_ms(50.0)
        self._model_loaded = True
        self._milestones["model_load_ms"] = (time.perf_counter() - started) * 1000.0

    def unload_model(self) -> None:
        self._model_loaded = False
        self._prefix_cache.clear()

    def shutdown(self) -> None:
        self._model_loaded = False
        self._initialized = False
        self._prefix_cache.clear()

    # --- generation -------------------------------------------------------

    def _find_failure(
        self, request: BenchmarkRequest
    ) -> MockFailureSpec | None:
        for spec in self.config.failures:
            if spec.matches(request):
                return spec
        return None

    def _enter_prefill(self) -> None:
        with self._lock:
            self._prefill_active += 1

    def _exit_prefill(self) -> None:
        with self._lock:
            self._prefill_active -= 1

    def generate_stream(self, request: BenchmarkRequest) -> Iterator[StreamToken]:
        self._check_ready()
        with self._lock:
            self._active_requests += 1
            self._request_counter += 1
        try:
            yield from self._generate_stream_inner(request)
        finally:
            with self._lock:
                self._active_requests -= 1

    def _generate_stream_inner(
        self, request: BenchmarkRequest
    ) -> Iterator[StreamToken]:
        failure = self._find_failure(request)

        # --- resource failures before any token ---
        if (
            self.config.oom_above_input_tokens is not None
            and request.prompt_token_count > self.config.oom_above_input_tokens
        ):
            raise RuntimeOutOfMemoryError(
                f"mock runtime: simulated OOM at {request.prompt_token_count} tokens"
            )
        if failure and failure.kind == "oom":
            raise RuntimeOutOfMemoryError("mock runtime: simulated OOM")
        if failure and failure.kind == "exception":
            raise RuntimeError("mock runtime: simulated runtime exception")

        # --- queueing ---
        self._sleep_ms(self.config.queue_ms)

        # --- prefill (with prefix-cache simulation) ---
        reused = 0
        cached = self._prefix_cache.get(request.prefix_id or "")
        if (
            self.config.prefix_cache_enabled
            and request.prefix_id
            and request.prefix_token_count > 0
            and cached
        ):
            reused = min(request.prefix_token_count, cached)

        effective_prompt = request.prompt_token_count - reused
        prefill_ms = (
            effective_prompt / self.config.prefill_tok_s * 1000.0
            + self.config.ttft_overhead_ms
        )
        self._enter_prefill()
        try:
            self._sleep_ms(prefill_ms)
        finally:
            self._exit_prefill()

        # Remember the prefix for later requests (cross-request reuse).
        if self.config.prefix_cache_enabled and request.prefix_id:
            self._prefix_cache[request.prefix_id] = request.prompt_token_count

        cache_hit_rate = (
            reused / request.prompt_token_count
            if request.prompt_token_count and self.config.prefix_cache_enabled
            else None
        )
        internal_metrics: dict[str, Any] = {}
        if self.config.prefix_cache_enabled:
            internal_metrics["prefix_reused_tokens"] = reused
            internal_metrics["prefix_recomputed_tokens"] = effective_prompt
            internal_metrics["cache_hit_rate"] = cache_hit_rate
            internal_metrics["cache_lookup_overhead_ms"] = 0.05 * self.config.time_scale
            internal_metrics["cache_insertion_overhead_ms"] = 0.08 * self.config.time_scale
        if request.prefix_token_count and not self.config.prefix_cache_enabled:
            internal_metrics["prefix_reused_tokens"] = 0
            internal_metrics["prefix_recomputed_tokens"] = request.prompt_token_count

        # --- decode ---
        target_tokens = request.max_new_tokens
        stop_after: int | None = None
        finish_reason = "length"
        if failure and failure.kind == "early_eos":
            stop_after = min(failure.after_tokens, target_tokens)
            finish_reason = "eos"
        elif failure and failure.kind == "partial":
            stop_after = min(failure.after_tokens, target_tokens)
            finish_reason = "aborted"
        elif failure and failure.kind == "timeout":
            stop_after = max(1, min(failure.after_tokens, target_tokens))

        emitted = 0
        while emitted < target_tokens:
            if stop_after is not None and emitted >= stop_after:
                if failure and failure.kind == "timeout":
                    raise RuntimeTimeoutError("mock runtime: simulated timeout")
                break
            with self._lock:
                contention = 1.0 + self.config.contention_factor * max(
                    0, self._active_requests - 1
                )
                if self._prefill_active > 0 and self._active_requests > 1:
                    contention *= self.config.prefill_interference
            itl_ms = (1000.0 / self.config.decode_tok_s) * contention
            self._sleep_ms(itl_ms)
            yield StreamToken(
                token_id=self._token_id(request.request_id, emitted),
                index=emitted,
                text=None,
            )
            emitted += 1

        if request.prefix_id and self.config.prefix_cache_enabled:
            self._prefix_cache[request.prefix_id] = request.prompt_token_count

        # Runtime-reported timings reflect the simulated (scaled) durations.
        scale = self.config.time_scale
        yield StreamToken(
            token_id=None,
            index=emitted,
            done=True,
            finish_reason=finish_reason,
            prefill_ms=prefill_ms * scale,
            queue_ms=self.config.queue_ms * scale,
            scheduler_ms=None,
            internal_metrics=internal_metrics,
        )

    def generate(self, request: BenchmarkRequest) -> GenerationResult:
        token_ids: list[int] = []
        final: StreamToken | None = None
        for token in self.generate_stream(request):
            if token.done:
                final = token
            elif token.token_id is not None:
                token_ids.append(token.token_id)
        assert final is not None
        return GenerationResult(
            token_ids=token_ids,
            text=None,
            finish_reason=final.finish_reason or "unknown",
            prefill_ms=final.prefill_ms,
            queue_ms=final.queue_ms,
            scheduler_ms=final.scheduler_ms,
            internal_metrics=final.internal_metrics,
        )

    def generate_batch(
        self, requests: list[BenchmarkRequest]
    ) -> list[GenerationResult]:
        if not self.config.static_batching:
            raise CapabilityNotSupported("mock runtime: static batching disabled")
        # Simulate batch formation: all requests prefill together, then decode.
        formation_ms = 0.5 * len(requests)
        self._sleep_ms(formation_ms)
        return [self.generate(r) for r in requests]

    # --- cache control ----------------------------------------------------

    def reset_cache(self) -> None:
        self._prefix_cache.clear()

    def clear_prefix_cache(self, prefix_id: str | None = None) -> None:
        if prefix_id is None:
            self._prefix_cache.clear()
        else:
            self._prefix_cache.pop(prefix_id, None)

    # --- introspection ----------------------------------------------------

    def get_capabilities(self) -> RuntimeCapabilities:
        prefix = (
            CapabilityStatus.SUPPORTED
            if self.config.prefix_cache_enabled
            else CapabilityStatus.UNSUPPORTED
        )
        return RuntimeCapabilities(
            supports_streaming=(
                CapabilityStatus.SUPPORTED
                if self.config.streaming
                else CapabilityStatus.UNSUPPORTED
            ),
            supports_input_ids=CapabilityStatus.SUPPORTED,
            supports_fixed_decode_length=(
                CapabilityStatus.SUPPORTED
                if self.config.fixed_decode_length
                else CapabilityStatus.UNSUPPORTED
            ),
            supports_prefix_cache=prefix,
            supports_cross_request_prefix_cache=(
                prefix
                if self.config.cross_request_prefix_cache
                else CapabilityStatus.UNSUPPORTED
            ),
            supports_continuous_batching=CapabilityStatus.SUPPORTED,
            supports_static_batching=(
                CapabilityStatus.SUPPORTED
                if self.config.static_batching
                else CapabilityStatus.UNSUPPORTED
            ),
            supports_chunked_prefill=CapabilityStatus.PARTIAL,
            supports_cuda_graphs=CapabilityStatus.UNSUPPORTED,
            supports_kv_quantization=CapabilityStatus.UNSUPPORTED,
            supports_cache_reset=CapabilityStatus.SUPPORTED,
            supports_internal_queue_metrics=CapabilityStatus.SUPPORTED,
            supports_internal_scheduler_metrics=CapabilityStatus.SUPPORTED,
            supports_internal_kv_metrics=CapabilityStatus.SUPPORTED,
            notes={
                "supports_chunked_prefill": "mock simulation only",
            },
        )

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": "1.0.0",
            "validation_only": True,
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "config": {
                "time_scale": self.config.time_scale,
                "prefill_tok_s": self.config.prefill_tok_s,
                "decode_tok_s": self.config.decode_tok_s,
                "prefix_cache_enabled": self.config.prefix_cache_enabled,
            },
        }

    def get_memory_info(self) -> dict[str, Any] | None:
        # The mock simulates no real memory; decomposition is not fabricated.
        return None

    def get_startup_milestones(self) -> dict[str, float | None]:
        return dict(self._milestones)

    def get_internal_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_requests": self._active_requests,
                "prefill_active": self._prefill_active,
                "prefix_cache_entries": len(self._prefix_cache),
            }
