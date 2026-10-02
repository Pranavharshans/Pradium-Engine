"""Generic runtime adapter interface.

Every inference runtime (ExLlamaV3, SGLang, TensorFold, vLLM, Pradium, ...)
integrates by implementing :class:`RuntimeAdapter`. The benchmark core talks to
this interface only and consumes pre-tokenized ``input_ids`` wherever possible,
so one runtime's tokenization path cannot change the workload.

Adapters declare capabilities (:mod:`benchmarks.core.capabilities`) instead of
the benchmark assuming features exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..core.capabilities import RuntimeCapabilities
from ..core.request import BenchmarkRequest


class RuntimeAdapterError(RuntimeError):
    """Base class for runtime adapter failures."""


class RuntimeOutOfMemoryError(RuntimeAdapterError):
    """Raised when the runtime cannot allocate memory for a request or model."""


class RuntimeTimeoutError(RuntimeAdapterError):
    """Raised when the runtime exceeds its request deadline."""


class CapabilityNotSupported(RuntimeAdapterError):
    """Raised when an operation is requested that the capability model forbids."""


@dataclass
class StreamToken:
    """One streamed output token (or the final control event).

    The final event of a stream carries ``done=True`` plus runtime-reported
    timings (``prefill_ms``, ``queue_ms``, ``scheduler_ms``) and internal
    metrics (cache hit rate, KV stats, ...) when the runtime exposes them.
    """

    token_id: int | None
    index: int
    text: str | None = None
    done: bool = False
    finish_reason: str | None = None
    prefill_ms: float | None = None
    queue_ms: float | None = None
    scheduler_ms: float | None = None
    internal_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Non-streaming generation output."""

    token_ids: list[int]
    text: str | None = None
    finish_reason: str = "length"
    prefill_ms: float | None = None
    queue_ms: float | None = None
    scheduler_ms: float | None = None
    internal_metrics: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(ABC):
    """Abstract inference-runtime adapter.

    Lifecycle::

        adapter = create_adapter("mock")
        adapter.initialize(...)
        adapter.load_model()
        for token in adapter.generate_stream(request):
            ...
        adapter.unload_model()
        adapter.shutdown()
    """

    #: Human-readable runtime name (recorded in every raw result).
    name: str = "abstract"

    # --- lifecycle --------------------------------------------------------

    @abstractmethod
    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the runtime (process, CUDA context, ...)."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Raises :class:`RuntimeOutOfMemoryError` on OOM."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model weights."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down the runtime completely."""

    # --- generation -------------------------------------------------------

    @abstractmethod
    def generate(self, request: BenchmarkRequest) -> GenerationResult:
        """Non-streaming generation. Must respect ``max_new_tokens``."""

    @abstractmethod
    def generate_stream(self, request: BenchmarkRequest) -> Iterator[StreamToken]:
        """Streaming generation; token order must match emission order.

        The last event must have ``done=True`` and a ``finish_reason``.
        """

    def generate_batch(
        self, requests: list[BenchmarkRequest]
    ) -> list[GenerationResult]:
        """Static batch generation. Only for runtimes declaring
        ``supports_static_batching``; concurrency (Cn) is not a batch (Bn)."""
        raise CapabilityNotSupported(
            f"{self.name} does not support static batch generation"
        )

    # --- cache control ----------------------------------------------------

    def reset_cache(self) -> None:
        """Reset internal caches where supported."""

    def clear_prefix_cache(self, prefix_id: str | None = None) -> None:
        """Clear prefix/KV cache (optionally one prefix) where supported."""

    # --- introspection ----------------------------------------------------

    @abstractmethod
    def get_capabilities(self) -> RuntimeCapabilities:
        """Declare capabilities; the benchmark never assumes features exist."""

    @abstractmethod
    def get_runtime_info(self) -> dict[str, Any]:
        """Return runtime metadata: name, version, commit, config, model, ..."""

    @abstractmethod
    def get_memory_info(self) -> dict[str, Any] | None:
        """Return runtime-reported memory decomposition if exposed.

        May return ``None``; the benchmark never fabricates decomposition.
        """

    # --- optional hooks ---------------------------------------------------

    def get_startup_milestones(self) -> dict[str, float | None]:
        """Optional: runtime-reported startup milestones (ms), e.g. CUDA graph
        capture time. Missing milestones are recorded as unavailable."""
        return {}

    def get_internal_metrics(self) -> dict[str, Any]:
        """Optional: queue/scheduler/KV metrics for the current state."""
        return {}
