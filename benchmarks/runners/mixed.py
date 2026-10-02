"""Mixed workload suite (scheduler stress beyond the standard matrix).

Workloads (frozen in ``core.workloads.MIXED_WORKLOADS``):

* ``interactive_burst``    — several SS/SM requests arriving closely together
* ``agent_like``           — repeated medium/long prefixes, small changing suffixes
* ``mixed_generation``     — short/medium/long outputs simultaneously
* ``mixed_prompt_sizes``   — 128/1024/4096 token inputs simultaneously
* ``long_job_interference``— LL running while interactive requests arrive

Per-request timing is preserved individually in every case.
"""

from __future__ import annotations

from typing import Any

from ..core.request import BenchmarkRequest
from ..core.results import RawResult
from ..core.workloads import MIXED_WORKLOADS, PROFILES
from .base import RunnerContext, run_group
from .prompts import PromptProvider


def run_mixed(
    ctx: RunnerContext,
    provider: PromptProvider,
    workloads: tuple[str, ...] | list[str] | None = None,
    warmups: int = 1,
    measured_runs: int = 3,
) -> tuple[list[RawResult], dict[str, Any]]:
    """Run the named mixed workloads."""
    names = tuple(workloads or tuple(MIXED_WORKLOADS))
    records: list[RawResult] = []
    extras: dict[str, Any] = {"mixed": {}}

    for name in names:
        if name not in MIXED_WORKLOADS:
            raise KeyError(f"unknown mixed workload: {name}")
        spec = MIXED_WORKLOADS[name]
        case_records: list[RawResult] = []
        for run_index in range(warmups + measured_runs):
            warmup = run_index < warmups
            requests = _build_requests(provider, name, spec, run_index, warmup)
            group, aggregates = run_group(
                ctx, requests, concurrency=int(spec["concurrency"])
            )
            case_records.extend(group)
        records.extend(case_records)
        extras["mixed"][name] = {
            "description": spec["description"],
            "requests": [str(p) for p in spec["requests"]],
            "concurrency": spec["concurrency"],
            "records": len(case_records),
        }
    return records, extras


def _build_requests(
    provider: PromptProvider,
    name: str,
    spec: dict[str, Any],
    run_index: int,
    warmup: bool,
) -> list[BenchmarkRequest]:
    requests: list[BenchmarkRequest] = []
    shared_prefix = int(spec.get("shared_prefix_tokens", 0) or 0)
    for slot, profile_name in enumerate(spec["requests"]):
        profile = PROFILES[profile_name]
        prefix_id = f"mixed-{name}" if shared_prefix else None
        requests.append(
            provider.profile_request(
                profile_name,
                run_index=run_index,
                warmup=warmup,
                concurrency_group=f"mixed:{name}",
                prefix_id=prefix_id,
                prefix_token_count=min(shared_prefix, profile.prompt_tokens),
                metadata={
                    "slot": slot,
                    "mixed_workload": name,
                    "role": "interactive" if profile.output_tokens <= 256 else "long_job",
                },
            )
        )
    return requests
