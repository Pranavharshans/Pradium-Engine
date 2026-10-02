"""Canonical benchmark workload definitions.

Frozen as part of PRADIUM-RUNTIME-BENCH-v1. Changing any constant in this
module requires a benchmark version bump.

Core 3x3 input x output matrix
------------------------------

Input sizes (prompt tokens):  S = 128, M = 1024, L = 4096
Output sizes (generated tokens): S = 64, M = 256, L = 1024

L is deliberately capped at 4096 so the core matrix stays practical on
constrained GPUs and larger models. Extreme contexts belong to the separate
context-scaling suite (:data:`CONTEXT_INPUT_TOKENS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Frozen token-length definitions ---------------------------------------

INPUT_TOKENS: dict[str, int] = {"S": 128, "M": 1024, "L": 4096}
OUTPUT_TOKENS: dict[str, int] = {"S": 64, "M": 256, "L": 1024}

INPUT_CLASS_BY_SIZE: dict[int, str] = {128: "S", 1024: "M", 4096: "L"}
OUTPUT_CLASS_BY_SIZE: dict[int, str] = {64: "S", 256: "M", 1024: "L"}

#: Canonical profile order used everywhere (row-major through the matrix).
PROFILE_ORDER: tuple[str, ...] = ("SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL")

PROFILE_PURPOSE: dict[str, str] = {
    "SS": "tiny interactive request; framework fixed overhead; shortest user-facing request",
    "SM": "normal short-prompt generation; interactive decode",
    "SL": "decode-heavy workload; sustained generation speed",
    "MS": "prefill-heavy short answer; coding/agent lookup-like workload",
    "MM": "canonical normal workload; general-purpose benchmark",
    "ML": "substantial prefill + substantial decode",
    "LS": "strongly prefill-dominated; RAG / long agent context",
    "LM": "long-context interactive workload",
    "LL": "heavy end-to-end workload; prefill + sustained decode",
}


@dataclass(frozen=True)
class Profile:
    """One canonical workload of the 3x3 matrix."""

    name: str
    prompt_tokens: int
    output_tokens: int
    input_class: str
    output_class: str
    purpose: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "input_class": self.input_class,
            "output_class": self.output_class,
            "purpose": self.purpose,
        }


def build_profiles() -> dict[str, Profile]:
    """Build the nine canonical profiles from the frozen S/M/L definitions."""
    profiles: dict[str, Profile] = {}
    for in_class in ("S", "M", "L"):
        for out_class in ("S", "M", "L"):
            name = f"{in_class}{out_class}"
            profiles[name] = Profile(
                name=name,
                prompt_tokens=INPUT_TOKENS[in_class],
                output_tokens=OUTPUT_TOKENS[out_class],
                input_class=in_class,
                output_class=out_class,
                purpose=PROFILE_PURPOSE[name],
            )
    return profiles


PROFILES: dict[str, Profile] = build_profiles()


def matrix_layout() -> dict[str, object]:
    """Return a machine-readable rendering of the 3x3 matrix layout."""
    return {
        "inputs": dict(INPUT_TOKENS),
        "outputs": dict(OUTPUT_TOKENS),
        "rows": [
            {
                "input_class": in_class,
                "input_tokens": INPUT_TOKENS[in_class],
                "columns": [
                    {
                        "output_class": out_class,
                        "output_tokens": OUTPUT_TOKENS[out_class],
                        "profile": f"{in_class}{out_class}",
                    }
                    for out_class in ("S", "M", "L")
                ],
            }
            for in_class in ("S", "M", "L")
        ],
    }


# --- Extended context-scaling suite ----------------------------------------

CONTEXT_INPUT_TOKENS: tuple[int, ...] = (
    128,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
CONTEXT_OUTPUT_TOKENS: int = 256

#: Statuses recorded when a context length cannot be executed.
CONTEXT_FAILURE_STATUSES: tuple[str, ...] = ("OOM", "UNSUPPORTED", "FAILED")

# --- Concurrency and batching ---------------------------------------------

CONCURRENCY_LEVELS: tuple[int, ...] = (1, 2, 4, 8)
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8)

# --- Prefix / KV reuse suite ----------------------------------------------

#: Approximate reusable-prefix ratios tested by the prefix-cache suite.
PREFIX_REUSE_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

#: Default total input size for prefix-cache requests.
PREFIX_TOTAL_INPUT_TOKENS: int = 1024
#: Number of requests per reuse ratio group.
PREFIX_REQUESTS_PER_RATIO: int = 4
#: For ratio 1.0 the suffix is a small unique tail on a large shared prefix.
PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE: int = 16

# --- Scheduler / mixed workloads ------------------------------------------

#: Number of concurrent decoding requests disturbed by the large prefill.
SCHEDULER_BACKGROUND_REQUESTS: int = 3
#: Background requests decode long enough for the interference to be observed.
SCHEDULER_BACKGROUND_PROFILE: str = "ML"
#: The intruding large-prefill request.
SCHEDULER_INTRUDER_PROFILE: str = "LS"

#: Mixed workload definitions: named request mixes used by the mixed suite.
MIXED_WORKLOADS: dict[str, dict[str, object]] = {
    "interactive_burst": {
        "description": "several SS/SM requests arriving closely together",
        "requests": ["SS", "SS", "SM", "SS", "SM", "SS"],
        "concurrency": 6,
    },
    "agent_like": {
        "description": "repeated medium/long prefixes with small changing suffixes",
        "requests": ["MS", "MS", "MS", "MS"],
        "concurrency": 4,
        "shared_prefix_tokens": 896,
    },
    "mixed_generation": {
        "description": "short, medium and long outputs simultaneously",
        "requests": ["MS", "MM", "ML"],
        "concurrency": 3,
    },
    "mixed_prompt_sizes": {
        "description": "128, 1024 and 4096 token inputs simultaneously",
        "requests": ["SM", "MM", "LM"],
        "concurrency": 3,
    },
    "long_job_interference": {
        "description": "LL running while smaller interactive requests arrive",
        "requests": ["LL", "SS", "SS", "SM"],
        "concurrency": 4,
    },
}


@dataclass(frozen=True)
class ContextCase:
    """One extended-context case; output length is fixed at 256 tokens."""

    input_tokens: int
    output_tokens: int = CONTEXT_OUTPUT_TOKENS
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"CTX-{self.input_tokens}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def context_cases() -> list[ContextCase]:
    return [ContextCase(n) for n in CONTEXT_INPUT_TOKENS]


def prefix_ratio_spec(
    ratio: float,
    total_input_tokens: int = PREFIX_TOTAL_INPUT_TOKENS,
    unique_suffix_tokens: int | None = None,
) -> tuple[int, int]:
    """Return ``(shared_prefix_tokens, unique_suffix_tokens)`` for a reuse ratio.

    The ratio is defined on token IDs: ``shared_prefix_tokens / total``. For
    the ~100% case a large identical prefix plus a small unique suffix is used
    instead of fully duplicated prompts.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"prefix ratio out of range: {ratio}")
    if ratio >= 1.0:
        suffix = unique_suffix_tokens or PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE
        if suffix >= total_input_tokens:
            raise ValueError("unique suffix must be smaller than the total input")
        return total_input_tokens - suffix, suffix
    shared = int(round(total_input_tokens * ratio))
    return shared, total_input_tokens - shared
