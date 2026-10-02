"""Runtime adapter package.

The benchmark core never contains framework-specific inference logic; every
runtime plugs into :class:`benchmarks.adapters.base.RuntimeAdapter`.
"""

from .base import (
    CapabilityNotSupported,
    GenerationResult,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeOutOfMemoryError,
    RuntimeTimeoutError,
    StreamToken,
)
from .registry import available_adapters, create_adapter, register_adapter

__all__ = [
    "CapabilityNotSupported",
    "GenerationResult",
    "RuntimeAdapter",
    "RuntimeAdapterError",
    "RuntimeOutOfMemoryError",
    "RuntimeTimeoutError",
    "StreamToken",
    "available_adapters",
    "create_adapter",
    "register_adapter",
]
