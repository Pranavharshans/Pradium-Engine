"""Runtime capability model.

Adapters declare capabilities instead of the benchmark core assuming features
exist. Every capability has one of four statuses so partial or unknown support
is never conflated with full support.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum


class CapabilityStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


#: Canonical capability names, frozen for PRADIUM-RUNTIME-BENCH-v1.
CAPABILITY_NAMES: tuple[str, ...] = (
    "supports_streaming",
    "supports_input_ids",
    "supports_fixed_decode_length",
    "supports_prefix_cache",
    "supports_cross_request_prefix_cache",
    "supports_continuous_batching",
    "supports_static_batching",
    "supports_chunked_prefill",
    "supports_cuda_graphs",
    "supports_kv_quantization",
    "supports_cache_reset",
    "supports_internal_queue_metrics",
    "supports_internal_scheduler_metrics",
    "supports_internal_kv_metrics",
)


class UnknownCapabilityError(ValueError):
    """Raised when a capability outside the canonical set is declared."""


@dataclass
class RuntimeCapabilities:
    """Declared runtime capabilities (status per capability name)."""

    supports_streaming: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_input_ids: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_fixed_decode_length: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_prefix_cache: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_cross_request_prefix_cache: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_continuous_batching: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_static_batching: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_chunked_prefill: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_cuda_graphs: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_kv_quantization: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_cache_reset: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_internal_queue_metrics: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_internal_scheduler_metrics: CapabilityStatus = CapabilityStatus.UNKNOWN
    supports_internal_kv_metrics: CapabilityStatus = CapabilityStatus.UNKNOWN

    #: Optional human-readable notes per capability (e.g. why PARTIAL).
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = set(CAPABILITY_NAMES)
        for name in self.notes:
            if name not in known:
                raise UnknownCapabilityError(f"unknown capability in notes: {name!r}")
        for f in fields(self):
            if f.name == "notes":
                continue
            value = getattr(self, f.name)
            if not isinstance(value, CapabilityStatus):
                setattr(self, f.name, CapabilityStatus(value))

    def status(self, name: str) -> CapabilityStatus:
        if name not in CAPABILITY_NAMES:
            raise UnknownCapabilityError(f"unknown capability: {name!r}")
        return getattr(self, name)

    def is_supported(self, name: str) -> bool:
        return self.status(name) is CapabilityStatus.SUPPORTED

    def is_usable(self, name: str) -> bool:
        """True when support is full or partial."""
        return self.status(name) in (
            CapabilityStatus.SUPPORTED,
            CapabilityStatus.PARTIAL,
        )

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {name: self.status(name).value for name in CAPABILITY_NAMES}
        if self.notes:
            out["notes"] = dict(self.notes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RuntimeCapabilities":
        unknown = set(data) - set(CAPABILITY_NAMES) - {"notes"}
        if unknown:
            raise UnknownCapabilityError(f"unknown capabilities: {sorted(unknown)}")
        kwargs = {
            name: CapabilityStatus(data[name]) for name in CAPABILITY_NAMES if name in data
        }
        notes = dict(data.get("notes") or {})  # type: ignore[arg-type]
        return cls(notes=notes, **kwargs)
