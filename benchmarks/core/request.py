"""Generic benchmark request object.

Runners never pass framework-specific arguments to a runtime; everything a
runtime needs is expressed in :class:`BenchmarkRequest`. Runtimes consume
pre-tokenized ``input_ids`` wherever possible so one runtime's tokenization
path cannot change the workload.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any


class RequestValidationError(ValueError):
    """Raised when a benchmark request violates the workload contract."""


def new_request_id(prefix: str = "req") -> str:
    """Return a readable unique request id."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def hash_token_ids(input_ids: list[int]) -> str:
    """Stable SHA-256 hash of a token id sequence."""
    payload = json.dumps(list(input_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class BenchmarkRequest:
    """A single generic inference request.

    Attributes are unit-explicit: token counts are counts, timing fields in the
    derived results are milliseconds. ``temperature`` is a plain float;
    ``greedy=True`` selects deterministic decoding (v1 official mode).
    """

    request_id: str
    input_ids: list[int]
    prompt_token_count: int
    max_new_tokens: int

    profile: str | None = None
    input_class: str | None = None
    output_class: str | None = None

    # Deterministic generation for official performance mode.
    temperature: float = 0.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0

    # Output-length control.
    fixed_decode_length: bool = True
    ignore_eos: bool = True

    # Concurrency / batching placement.
    concurrency_group: str | None = None
    batch_size: int = 1

    # Prefix / KV reuse.
    prefix_id: str | None = None
    prefix_token_count: int = 0

    # Optional prompt text for runtimes that cannot consume input_ids.
    # Any use of it is recorded as a fairness deviation in the raw result.
    prompt_text: str | None = None

    # Bookkeeping.
    warmup: bool = False
    run_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise RequestValidationError("request_id must be non-empty")
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.input_ids, list):
            raise RequestValidationError("input_ids must be a list of ints")
        if not all(isinstance(t, int) for t in self.input_ids):
            raise RequestValidationError("input_ids must contain only integers")
        if self.prompt_token_count != len(self.input_ids):
            raise RequestValidationError(
                f"prompt_token_count={self.prompt_token_count} does not match "
                f"len(input_ids)={len(self.input_ids)}"
            )
        if self.prompt_token_count <= 0:
            raise RequestValidationError("prompt_token_count must be positive")
        if self.max_new_tokens <= 0:
            raise RequestValidationError("max_new_tokens must be positive")
        if self.prefix_token_count < 0 or self.prefix_token_count > self.prompt_token_count:
            raise RequestValidationError(
                "prefix_token_count must be within [0, prompt_token_count]"
            )
        if self.prefix_token_count and not self.prefix_id:
            raise RequestValidationError("prefix_id is required when prefix_token_count > 0")
        if self.batch_size < 1:
            raise RequestValidationError("batch_size must be >= 1")
        if not self.greedy and self.temperature <= 0.0:
            raise RequestValidationError(
                "non-greedy requests require temperature > 0"
            )

    @property
    def input_ids_hash(self) -> str:
        return hash_token_ids(self.input_ids)

    def to_dict(self, include_input_ids: bool = False) -> dict[str, Any]:
        """Serialize. Input ids are excluded by default (hashed instead)."""
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "profile": self.profile,
            "input_class": self.input_class,
            "output_class": self.output_class,
            "prompt_token_count": self.prompt_token_count,
            "input_ids_hash": self.input_ids_hash,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "greedy": self.greedy,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "fixed_decode_length": self.fixed_decode_length,
            "ignore_eos": self.ignore_eos,
            "concurrency_group": self.concurrency_group,
            "batch_size": self.batch_size,
            "prefix_id": self.prefix_id,
            "prefix_token_count": self.prefix_token_count,
            "warmup": self.warmup,
            "run_index": self.run_index,
            "metadata": dict(self.metadata),
            "uses_prompt_text": self.prompt_text is not None,
        }
        if include_input_ids:
            out["input_ids"] = list(self.input_ids)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRequest":
        payload = dict(data)
        if "input_ids" not in payload:
            raise RequestValidationError("from_dict requires input_ids")
        # Derived fields are recomputed on construction.
        payload.pop("input_ids_hash", None)
        payload.pop("uses_prompt_text", None)
        return cls(**payload)
